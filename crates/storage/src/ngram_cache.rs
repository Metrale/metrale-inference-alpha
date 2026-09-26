// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-25: NVMe-backed row cache for n-gram embedding tables. Rows fault from the
//! checkpoint file into a pinned arena, and the gather kernel reads the arena by slot.
//!
//! Owner: storage, n-gram row cache.
//! Invariants:
//! - `map[id] == s` exactly when `slot_row[s] == id`, on every path out of `resolve`.
//! - A slot pinned since the last `end_batch` is never chosen as an eviction victim.
//!
//! The n-gram ids are computed on the host, so `resolve` maps each id to its arena slot
//! and the caller hands the gather kernel slot indices in place of ids;
//! `table_dev_va` is the table it gathers from. The arena's device address equals its
//! host address (`ExpertArena::new` refuses a host where it does not), so a miss is
//! read from NVMe straight into its slot, with no host-to-device copy. Eviction is
//! CLOCK (second chance) over the unpinned slots.
//!
//! On unix the files are opened `O_DIRECT`, which needs 4 KiB-aligned transfers: a
//! miss reads the one or two blocks covering its row into an aligned bounce buffer
//! and copies the row out. The cache holds single rows, not blocks.

use std::collections::HashMap;
use std::fs::File;
use std::path::Path;

use anyhow::{Context, Result, bail};

use crate::expert_arena::ExpertArena;

/// 2026-09-25: O_DIRECT transfer size, and the stride `ExpertArena::new` requires a
/// multiple of.
pub(crate) const BLOCK: usize = 4096;

/// 2026-09-25: One table's backing files and its resident row cache.
pub struct NgramRowCache {
    /// 2026-09-25: Pinned, GPU-addressable `[slots, row_stride]` rows.
    arena: ExpertArena,
    /// 2026-09-25: Backing file (shard 0's, for a segmented table). Unsegmented, row
    /// `i` is at `base_offset + i * row_stride`, so a tensor is read in place from a
    /// safetensors shard. The offset need not be block-aligned: a row that crosses a
    /// block boundary is read as two blocks (`ngram_cache_fault::nblocks_for`).
    file: File,
    /// 2026-09-25: The other files of a segmented table whose shards span several
    /// files, in first-seen order. `Segments::shard_file` value `n > 0` is
    /// `extra_files[n - 1]`.
    extra_files: Vec<File>,
    base_offset: u64,
    /// 2026-09-25: Set by `open_segmented`, for a table stored as shard tensors that
    /// are not consecutive in the file; each shard has its own base. `None`:
    /// `base_offset` locates every row.
    segments: Option<Segments>,
    /// 2026-09-25: FP8 scales by slot; `None` for a BF16 table.
    scales: Option<ScaleCache>,
    row_stride: usize,
    slots: usize,
    rows_total: u64,
    map: HashMap<u64, u32>,
    /// 2026-09-25: Slot to resident row id; `u64::MAX` marks an empty slot.
    slot_row: Vec<u64>,
    refbit: Vec<bool>,
    /// 2026-09-25: Slots pinned for the batch in flight; `victim` skips them.
    pinned: Vec<bool>,
    hand: usize,
    bounce: AlignedBlock,
    pub hits: u64,
    pub misses: u64,
    pub evictions: u64,
}

/// 2026-09-25: A table split into equal-sized shards at arbitrary file offsets.
struct Segments {
    /// 2026-09-25: Byte offset of each shard's first row, indexed by shard.
    bases: Vec<u64>,
    /// 2026-09-25: Backing file of each shard: 0 is `file`, `n > 0` is
    /// `extra_files[n - 1]`.
    shard_file: Vec<usize>,
    /// 2026-09-25: Rows in every shard; `row_byte` finds a row's shard by division.
    rows_per: u64,
}

/// 2026-09-25: f32 scales of an FP8 table, in a device-visible `[slots]` array
/// indexed by slot, parallel to the row arena.
struct ScaleCache {
    arena: ExpertArena,
    /// 2026-09-25: Per-row scale file. `None` after `set_constant_scale`, which fills
    /// the arena once; nothing is faulted for it.
    file: Option<File>,
}

/// 2026-09-25: A 4 KiB-aligned host buffer for O_DIRECT reads.
pub(crate) struct AlignedBlock {
    buf: Vec<u8>,
    off: usize,
}

impl AlignedBlock {
    /// 2026-09-25: Room for two aligned blocks: a row that is not block-aligned can
    /// cross one boundary, and `open_at` refuses `row_stride > BLOCK`.
    pub(crate) fn new() -> Self {
        // 2026-09-25: Three blocks hold an aligned two-block window at any address.
        let buf = vec![0u8; BLOCK * 3];
        let addr = buf.as_ptr() as usize;
        let off = (BLOCK - (addr % BLOCK)) % BLOCK;
        Self { buf, off }
    }
    pub(crate) fn blocks(&mut self, n: usize) -> &mut [u8] {
        &mut self.buf[self.off..self.off + n * BLOCK]
    }
}

impl NgramRowCache {
    /// 2026-09-25: Open `path` as the backing store for a table of `rows_total` rows
    /// of `row_stride` bytes, caching `slots` of them in pinned GPU-addressable
    /// memory. `scale_path` supplies the per-row f32 scales of an FP8 table.
    pub fn open(
        path: &Path,
        scale_path: Option<&Path>,
        rows_total: u64,
        row_stride: usize,
        slots: usize,
    ) -> Result<Self> {
        Self::open_at(path, 0, scale_path, rows_total, row_stride, slots)
    }

    /// 2026-09-25: As [`Self::open`], with the table starting at `base_offset` in the
    /// file. Zero geometry or `row_stride > BLOCK` is an error.
    #[allow(clippy::too_many_arguments)]
    pub fn open_at(
        path: &Path,
        base_offset: u64,
        scale_path: Option<&Path>,
        rows_total: u64,
        row_stride: usize,
        slots: usize,
    ) -> Result<Self> {
        if row_stride == 0 || slots == 0 {
            bail!("NgramRowCache: zero geometry (row_stride={row_stride}, slots={slots})");
        }
        if row_stride > BLOCK {
            bail!(
                "NgramRowCache: row_stride {row_stride} exceeds the {BLOCK}-byte \
                 O_DIRECT block; a row would span more than the two blocks the \
                 seam-handling fetch reads"
            );
        }
        // 2026-09-25: `slots * row_stride` bytes, rounded up to whole blocks for
        // `ExpertArena`.
        let bytes = slots * row_stride;
        let blocks = bytes.div_ceil(BLOCK);
        let arena =
            ExpertArena::new(1, blocks as u32, BLOCK).context("NgramRowCache: pinned arena")?;
        let file = open_direct(path)?;
        let scales = match scale_path {
            Some(sp) => {
                let sbytes = slots * 4;
                let sblocks = sbytes.div_ceil(BLOCK);
                Some(ScaleCache {
                    arena: ExpertArena::new(1, sblocks as u32, BLOCK)
                        .context("NgramRowCache: scale arena")?,
                    file: Some(open_direct(sp)?),
                })
            }
            None => None,
        };
        Ok(Self {
            arena,
            file,
            base_offset,
            segments: None,
            extra_files: Vec::new(),
            scales,
            row_stride,
            slots,
            rows_total,
            map: HashMap::with_capacity(slots * 2),
            slot_row: vec![u64::MAX; slots],
            refbit: vec![false; slots],
            pinned: vec![false; slots],
            hand: 0,
            bounce: AlignedBlock::new(),
            hits: 0,
            misses: 0,
            evictions: 0,
        })
    }

    /// 2026-09-25: As [`Self::open_at`], for a table split into shards at arbitrary
    /// offsets, possibly in several files. `shards[i]` is shard `i`'s file and the
    /// offset of its first row; every shard holds `rows_per_shard` rows.
    #[allow(clippy::too_many_arguments)]
    pub fn open_segmented(
        shards: &[(std::path::PathBuf, u64)],
        rows_per_shard: u64,
        scale_path: Option<&Path>,
        row_stride: usize,
        slots: usize,
    ) -> Result<Self> {
        if shards.is_empty() || rows_per_shard == 0 {
            bail!(
                "NgramRowCache: segmented table needs shards and rows \
                 (shards={}, rows_per_shard={rows_per_shard})",
                shards.len()
            );
        }
        // 2026-09-25: One `File` per distinct path, in first-seen order. Shard 0's
        // file is the cache's own `file`; the rest are `extra_files`.
        let mut order: Vec<&Path> = Vec::new();
        let mut shard_file = Vec::with_capacity(shards.len());
        for (p, _) in shards {
            let idx = order
                .iter()
                .position(|q| *q == p.as_path())
                .unwrap_or_else(|| {
                    order.push(p.as_path());
                    order.len() - 1
                });
            shard_file.push(idx);
        }
        let bases: Vec<u64> = shards.iter().map(|(_, o)| *o).collect();
        let rows_total = bases.len() as u64 * rows_per_shard;
        let mut c = Self::open_at(order[0], 0, scale_path, rows_total, row_stride, slots)?;
        for p in &order[1..] {
            c.extra_files.push(open_direct(p)?);
        }
        c.segments = Some(Segments {
            bases,
            shard_file,
            rows_per: rows_per_shard,
        });
        Ok(c)
    }

    /// 2026-09-25: Device VA of the row arena, the table the gather kernels index by
    /// slot.
    pub fn table_dev_va(&self) -> Result<u64> {
        self.arena.slot_dev_va(0, 0)
    }

    /// 2026-09-25: Give every slot the same scale, for an FP8 table with one scale
    /// for all rows. The scale arena is filled here and never faulted.
    ///
    /// # Errors
    /// If the scale arena cannot be allocated.
    pub fn set_constant_scale(&mut self, scale: f32) -> Result<()> {
        let sbytes = self.slots * 4;
        let sblocks = sbytes.div_ceil(BLOCK);
        let arena = ExpertArena::new(1, sblocks as u32, BLOCK)
            .context("NgramRowCache: constant scale arena")?;
        let p = arena.slot_host_ptr(0, 0)?.cast::<f32>();
        for i in 0..self.slots {
            // 2026-09-25: SAFETY: the arena is at least `slots * 4` bytes and was just
            // allocated here, so nothing else holds a reference into it.
            unsafe { p.add(i).write(scale) };
        }
        self.scales = Some(ScaleCache { arena, file: None });
        Ok(())
    }

    /// 2026-09-25: Device VA of the `[slots]` f32 scale array; `None` for a BF16
    /// table.
    pub fn scale_dev_va(&self) -> Result<Option<u64>> {
        match &self.scales {
            Some(s) => Ok(Some(s.arena.slot_dev_va(0, 0)?)),
            None => Ok(None),
        }
    }

    pub fn stats(&self) -> (u64, u64, u64) {
        (self.hits, self.misses, self.evictions)
    }

    /// 2026-09-25: Resolve `row_ids` to slot indices, faulting misses in from NVMe.
    ///
    /// Every returned slot stays pinned until [`Self::end_batch`], so a later resolve
    /// in the same batch cannot evict a row the gather has yet to read. An id at or
    /// past the table's row count, every slot pinned, or a failed read is an error.
    pub fn resolve(&mut self, row_ids: &[u64], out_slots: &mut Vec<u32>) -> Result<()> {
        out_slots.clear();
        out_slots.reserve(row_ids.len());
        // 2026-09-25: First, with no I/O: pin hits and give every miss a victim slot. A
        // repeated missing id hits the map on its second occurrence, so each
        // distinct row faults once.
        let mut jobs: Vec<crate::ngram_cache_fault::FaultJob> = Vec::new();
        for &id in row_ids {
            if id >= self.rows_total {
                bail!(
                    "NgramRowCache: row id {id} >= table rows {} (hash/table mismatch)",
                    self.rows_total
                );
            }
            let slot = match self.map.get(&id) {
                Some(&s) => {
                    self.hits += 1;
                    self.refbit[s as usize] = true;
                    self.pinned[s as usize] = true;
                    s
                }
                None => {
                    self.misses += 1;
                    let s = self.victim()?;
                    self.map.insert(id, s);
                    self.slot_row[s as usize] = id;
                    self.refbit[s as usize] = true;
                    self.pinned[s as usize] = true;
                    jobs.push(self.fault_job(id, s)?);
                    s
                }
            };
            out_slots.push(slot);
        }
        if !jobs.is_empty() {
            let files: Vec<&File> = std::iter::once(&self.file)
                .chain(self.extra_files.iter())
                .collect();
            let r = crate::ngram_cache_fault::fault_all(
                &jobs,
                &files,
                self.scales.as_ref().and_then(|sc| sc.file.as_ref()),
                self.row_stride,
                &mut self.bounce,
            );
            if let Err(e) = r {
                // 2026-09-25: Unmap and unpin every slot this batch faulted: any of
                // them may hold a partial row.
                for j in &jobs {
                    self.map.remove(&j.row_id);
                    self.slot_row[j.slot as usize] = u64::MAX;
                    self.pinned[j.slot as usize] = false;
                    self.refbit[j.slot as usize] = false;
                }
                return Err(e);
            }
        }
        Ok(())
    }

    /// 2026-09-25: Resolve one miss to file offsets and destination addresses for
    /// [`crate::ngram_cache_fault::fault_all`].
    fn fault_job(&self, id: u64, slot: u32) -> Result<crate::ngram_cache_fault::FaultJob> {
        let (file_idx, byte) = self.row_byte(id);
        let block_off = byte - (byte % BLOCK as u64);
        let within = (byte - block_off) as usize;
        let nblocks = crate::ngram_cache_fault::nblocks_for(within, self.row_stride);
        // 2026-09-25: SAFETY: address arithmetic only; the fault worker writes the
        // disjoint `[dst, dst+row_stride)` region while the arena is live.
        let dst = unsafe {
            self.arena
                .slot_host_ptr(0, 0)?
                .add(slot as usize * self.row_stride)
        } as usize;
        let scale = match &self.scales {
            // 2026-09-25: A constant scale has no file and is never faulted: every
            // slot already holds it.
            Some(sc) if sc.file.is_some() => {
                let sbyte = id * 4;
                let sblock = sbyte - (sbyte % BLOCK as u64);
                let swithin = (sbyte - sblock) as usize;
                // 2026-09-25: SAFETY: as above, for a disjoint 4-byte region.
                let sdst = unsafe { sc.arena.slot_host_ptr(0, 0)?.add(slot as usize * 4) };
                Some((sblock, swithin, sdst as usize))
            }
            Some(_) | None => None,
        };
        Ok(crate::ngram_cache_fault::FaultJob {
            row_id: id,
            slot,
            block_off,
            within,
            nblocks,
            dst,
            scale,
            file_idx,
        })
    }

    /// 2026-09-25: Release the batch's pins; call after the gather kernels are issued.
    pub fn end_batch(&mut self) {
        for p in &mut self.pinned {
            *p = false;
        }
    }

    /// 2026-09-25: CLOCK second-chance victim among the unpinned slots; an error when
    /// every slot is pinned.
    fn victim(&mut self) -> Result<u32> {
        for _ in 0..(self.slots * 2) {
            let s = self.hand;
            self.hand = (self.hand + 1) % self.slots;
            if self.pinned[s] {
                continue;
            }
            if self.refbit[s] {
                self.refbit[s] = false;
                continue;
            }
            if self.slot_row[s] != u64::MAX {
                let old = self.slot_row[s];
                self.map.remove(&old);
                self.evictions += 1;
            }
            return Ok(s as u32);
        }
        bail!(
            "NgramRowCache: every one of {} slots is pinned by the batch in flight — \
             raise the cache size or lower max-prefill-tokens",
            self.slots
        )
    }

    /// 2026-09-25: Backing-file index and byte offset of row `id`. The index is 0
    /// unless a segmented table's shard lives in another file.
    fn row_byte(&self, id: u64) -> (usize, u64) {
        match &self.segments {
            None => (0, self.base_offset + id * self.row_stride as u64),
            Some(seg) => {
                let shard = (id / seg.rows_per) as usize;
                let local = id % seg.rows_per;
                (
                    seg.shard_file[shard],
                    seg.bases[shard] + local * self.row_stride as u64,
                )
            }
        }
    }
}

#[cfg(unix)]
fn open_direct(path: &Path) -> Result<File> {
    use std::os::unix::fs::OpenOptionsExt;
    std::fs::OpenOptions::new()
        .read(true)
        .custom_flags(libc::O_DIRECT)
        .open(path)
        .with_context(|| format!("NgramRowCache: open {} (O_DIRECT)", path.display()))
}

#[cfg(not(unix))]
fn open_direct(path: &Path) -> Result<File> {
    File::open(path).with_context(|| format!("NgramRowCache: open {}", path.display()))
}

#[cfg(test)]
#[path = "ngram_cache/tests.rs"]
mod tests;
