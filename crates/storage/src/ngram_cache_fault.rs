// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-25: Batched NVMe fault-in for `NgramRowCache::resolve`: every miss of a
//! batch is read with positional reads, serially for small batches, else on scoped
//! threads.
//!
//! Owner: storage, n-gram row cache.
//! Invariants:
//! - A job writes only its own slot region and 4-byte scale region. `resolve` gives
//!   each miss a distinct slot, since it pins every slot it hands out.
//!
//! The file handles are shared (`read_at` on `&File`), each worker has its own aligned
//! bounce, and the only other shared state is the atomic job index and the error.

use std::fs::File;
use std::sync::Mutex;
use std::sync::atomic::{AtomicUsize, Ordering};

use anyhow::{Context, Result};

use super::ngram_cache::{AlignedBlock, BLOCK};

const PARALLEL_MIN: usize = 4;

/// 2026-09-25: Most worker threads one `fault_all` call spawns.
const MAX_WORKERS: usize = 16;

/// 2026-09-25: One miss, resolved to file offsets and destination addresses, so the
/// workers need no access to the cache.
pub(super) struct FaultJob {
    pub(super) row_id: u64,
    pub(super) slot: u32,
    /// 2026-09-25: 4 KiB-aligned file offset of the first block holding the row.
    pub(super) block_off: u64,
    pub(super) within: usize,
    /// 2026-09-25: 1, or 2 when the row crosses a block boundary.
    pub(super) nblocks: usize,
    pub(super) file_idx: usize,
    /// 2026-09-25: Address of the job's arena slot; `[dst, dst + row_stride)` is
    /// disjoint from every other job's.
    pub(super) dst: usize,
    /// 2026-09-25: Per-row scale tables: `(scale_block_off, scale_within, scale_dst)`.
    pub(super) scale: Option<(u64, usize, usize)>,
}

// 2026-09-25: SAFETY: `dst` and `scale.2` are addresses in the pinned arenas; each
// job's regions are disjoint, and the arenas outlive the scoped threads.
unsafe impl Send for FaultJob {}
unsafe impl Sync for FaultJob {}

fn run_one(
    job: &FaultJob,
    file: &File,
    scale_file: Option<&File>,
    row_stride: usize,
    bounce: &mut AlignedBlock,
) -> Result<()> {
    // 2026-09-25: The blocks covering a row near the end of a file can run past EOF,
    // so only the bytes through the row's end, `within + row_stride`, are required.
    crate::tier::pio::read_at_least_at(
        file,
        bounce.blocks(job.nblocks),
        job.block_off,
        job.within + row_stride,
    )
    .with_context(|| format!("NgramRowCache: read row {}", job.row_id))?;
    // 2026-09-25: SAFETY: this job's disjoint region inside the live pinned arena.
    let dst = unsafe { std::slice::from_raw_parts_mut(job.dst as *mut u8, row_stride) };
    dst.copy_from_slice(&bounce.blocks(job.nblocks)[job.within..job.within + row_stride]);

    if let Some((sblock, swithin, sdst)) = job.scale {
        let sf = scale_file.expect("scale job without scale file");
        // 2026-09-25: A scale near the end of its file sits in a block that runs past
        // EOF too.
        crate::tier::pio::read_at_least_at(sf, bounce.blocks(1), sblock, swithin + 4)
            .with_context(|| format!("NgramRowCache: read scale {}", job.row_id))?;
        // 2026-09-25: SAFETY: this job's disjoint 4-byte region in the scale arena.
        let sdst = unsafe { std::slice::from_raw_parts_mut(sdst as *mut u8, 4) };
        sdst.copy_from_slice(&bounce.blocks(1)[swithin..swithin + 4]);
    }
    Ok(())
}

/// 2026-09-25: `METRALE_PLE_SERIAL_FAULT=1` faults every batch serially. Read once per
/// process.
fn serial_forced() -> bool {
    static V: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *V.get_or_init(|| std::env::var("METRALE_PLE_SERIAL_FAULT").as_deref() == Ok("1"))
}

/// 2026-09-25: Fault every job in: serially below [`PARALLEL_MIN`] jobs or with
/// `METRALE_PLE_SERIAL_FAULT=1`, else on up to [`MAX_WORKERS`] scoped threads. If any
/// job fails, one failed job's error is returned; a worker stops after its own error
/// or once another worker has recorded one.
pub(super) fn fault_all(
    jobs: &[FaultJob],
    files: &[&File],
    scale_file: Option<&File>,
    row_stride: usize,
    bounce: &mut AlignedBlock,
) -> Result<()> {
    // 2026-09-25: Per job, since two jobs in one batch can read different files.
    let pick = |job: &FaultJob| -> Result<&File> {
        files.get(job.file_idx).copied().ok_or_else(|| {
            anyhow::anyhow!(
                "NgramRowCache: row {} names backing file {} of {}",
                job.row_id,
                job.file_idx,
                files.len()
            )
        })
    };
    if jobs.len() < PARALLEL_MIN || serial_forced() {
        for job in jobs {
            run_one(job, pick(job)?, scale_file, row_stride, bounce)?;
        }
        return Ok(());
    }
    let next = AtomicUsize::new(0);
    let first_err: Mutex<Option<anyhow::Error>> = Mutex::new(None);
    let workers = jobs.len().min(MAX_WORKERS);
    std::thread::scope(|s| {
        for _ in 0..workers {
            s.spawn(|| {
                let mut bounce = AlignedBlock::new();
                loop {
                    let i = next.fetch_add(1, Ordering::Relaxed);
                    let Some(job) = jobs.get(i) else { break };
                    let f = match pick(job) {
                        Ok(f) => f,
                        Err(e) => {
                            *first_err.lock().unwrap() = Some(e);
                            break;
                        }
                    };
                    if let Err(e) = run_one(job, f, scale_file, row_stride, &mut bounce) {
                        *first_err.lock().unwrap() = Some(e);
                        break;
                    }
                    if first_err.lock().unwrap().is_some() {
                        break;
                    }
                }
            });
        }
    });
    match first_err.into_inner().unwrap() {
        Some(e) => Err(e),
        None => Ok(()),
    }
}

/// 2026-09-25: Blocks covering a row that starts `within` bytes into a block: 2 when it
/// crosses the block's end, else 1. Two suffice because `open_at` refuses
/// `row_stride > BLOCK`.
pub(super) fn nblocks_for(within: usize, row_stride: usize) -> usize {
    if within + row_stride > BLOCK { 2 } else { 1 }
}
