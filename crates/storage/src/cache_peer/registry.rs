// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-25: The cache peer's shared paging arenas, one per (kind,
//! blob_bytes): an anonymous mapping that every paging connection of that
//! shape registers, one residency over its slots backed by an O_DIRECT swap
//! file, and the split of the swap-disk budget between arenas.
//!
//! Owner: metrale-storage peers.
//! Invariants:
//! - At most one arena exists per (kind, blob_bytes). Later requests for the
//!   same key get it back, whatever `arena_bytes` they ask for.
//! - An arena, its swap file and its ledger reservation are never removed
//!   from the registry.

use crate::tier::{DirectSwapFile, Residency};
use anyhow::{Context, Result, bail};

use super::server_impl::RdmaConfig;
use crate::snapshot_swap::MmapSlotArena;

/// 2026-09-25: An anonymous private mapping, unmapped on drop.
pub(super) struct Mmap {
    pub(super) addr: *mut libc::c_void,
    pub(super) len: usize,
}

impl Mmap {
    pub(super) fn anon(len: usize) -> Result<Self> {
        // 2026-09-25: SAFETY: an anonymous private mapping of `len` bytes.
        let addr = unsafe {
            libc::mmap(
                std::ptr::null_mut(),
                len,
                libc::PROT_READ | libc::PROT_WRITE,
                libc::MAP_PRIVATE | libc::MAP_ANONYMOUS,
                -1,
                0,
            )
        };
        if addr == libc::MAP_FAILED {
            bail!(
                "mmap anon {len} failed: {}",
                std::io::Error::last_os_error()
            );
        }
        Ok(Self { addr, len })
    }
}

impl Drop for Mmap {
    fn drop(&mut self) {
        // 2026-09-25: SAFETY: addr/len come from a successful mmap; drop runs once.
        unsafe { libc::munmap(self.addr, self.len) };
    }
}

/// 2026-09-25: One (kind, blob_bytes) arena: the mapping its paging
/// connections register, and the residency that owns its slots.
pub(super) struct SharedPaging {
    pub(super) arena: Mmap,
    pub(super) residency: std::sync::Mutex<Residency<MmapSlotArena, DirectSwapFile>>,
    _reservation: crate::blade_cap::Reservation,
}
// 2026-09-25: SAFETY: `arena.addr` is a fixed mapping, and the residency that
// reads and writes its slots is behind a Mutex.
unsafe impl Send for SharedPaging {}
unsafe impl Sync for SharedPaging {}

/// 2026-09-25: The paging arenas by (kind, blob_bytes), and the swap-disk
/// budget left to give out.
#[derive(Default)]
struct PagingRegistry {
    arenas: std::collections::HashMap<(u8, usize), std::sync::Arc<SharedPaging>>,
    /// 2026-09-25: Swap-disk bytes not yet given to an arena.
    remaining_cap: u64,
    cap_init: bool,
    legacy_cleaned: bool,
}

/// 2026-09-25: A process-wide static: it splits one swap-disk budget
/// (`swap_cap_bytes`) between arenas that independent connections create, so
/// they cannot each claim the whole cap.
static SHARED_PAGING: std::sync::LazyLock<std::sync::Mutex<PagingRegistry>> =
    std::sync::LazyLock::new(|| std::sync::Mutex::new(PagingRegistry::default()));

/// 2026-09-25: An arena's disk-slot cap (0 = unbounded) and the new shared
/// remainder:
/// - a per-kind cap of 0: unbounded, remainder unchanged;
/// - a non-zero per-kind cap: `cap / blob_bytes` slots, at least 1, remainder
///   unchanged;
/// - no per-kind cap and a shared cap of 0: unbounded, remainder unchanged;
/// - otherwise: all whole records of the remainder, at least 1, which are
///   taken out of it.
fn carve_disk_slots(
    per_kind_cap: Option<u64>,
    shared_cap: u64,
    shared_remaining: u64,
    blob_bytes: u64,
) -> (usize, u64) {
    let bb = blob_bytes.max(1);
    match per_kind_cap {
        Some(0) => (0, shared_remaining),
        Some(cap) => (((cap / bb) as usize).max(1), shared_remaining),
        None if shared_cap == 0 => (0, shared_remaining),
        None => {
            let recs = (shared_remaining / bb) as usize;
            (
                recs.max(1),
                shared_remaining.saturating_sub(recs as u64 * bb),
            )
        }
    }
}

/// 2026-09-25: The arena for (kind, blob_bytes), created on first request:
/// the ledger is charged `arena_bytes` once per arena, before the mapping,
/// and the disk cap comes from `carve_disk_slots`. After creation the
/// residency is used under its own Mutex, not the registry lock.
pub(super) fn get_or_init_shared_paging(
    rdma: &RdmaConfig,
    kind: u8,
    arena_bytes: usize,
    blob_bytes: usize,
    ledger: &std::sync::Arc<crate::blade_cap::CommitLedger>,
) -> Result<std::sync::Arc<SharedPaging>> {
    let mut reg = SHARED_PAGING.lock().expect("paging registry poisoned");
    let key = (kind, blob_bytes);
    if let Some(sh) = reg.arenas.get(&key) {
        return Ok(sh.clone());
    }
    let swap_dir = rdma
        .swap_dir
        .as_ref()
        .context("paging client but peer has no --swap-dir configured")?;
    std::fs::create_dir_all(swap_dir).ok();
    // 2026-09-25: On the first arena: set the disk budget, and delete a
    // `metrale-snap-shared.swap`, a file name no code here writes.
    if !reg.cap_init {
        reg.remaining_cap = rdma.swap_cap_bytes;
        reg.cap_init = true;
    }
    if !reg.legacy_cleaned {
        let _ = std::fs::remove_file(swap_dir.join("metrale-snap-shared.swap"));
        reg.legacy_cleaned = true;
    }
    let reservation = ledger
        .try_reserve(arena_bytes as u64)
        .context("paging blade cap")?;
    let arena = Mmap::anon(arena_bytes)?;
    let num_slots = arena_bytes / blob_bytes;
    let (max_disk_slots, new_remaining) = carve_disk_slots(
        rdma.per_kind_swap_cap_bytes.get(&kind).copied(),
        rdma.swap_cap_bytes,
        reg.remaining_cap,
        blob_bytes as u64,
    );
    reg.remaining_cap = new_remaining;
    let swap_path = swap_dir.join(format!("metrale-snap-{kind}-{blob_bytes}.swap"));
    let swap = DirectSwapFile::create(&swap_path, blob_bytes)?;
    // 2026-09-25: SAFETY: the Mmap moves into the same `SharedPaging` as the
    // residency that holds this view, and the registry never drops it.
    let slot_arena = unsafe { MmapSlotArena::new(arena.addr as *mut u8, blob_bytes, num_slots) };
    let residency = Residency::new_capped(slot_arena, swap, max_disk_slots)?;
    tracing::info!(
        "cache-peer paging arena kind={kind} shape={blob_bytes}B: {num_slots} slots RAM \
         ({:.1} GiB) + NVMe swap {} (disk cap {} records; budget {:.0} GiB left)",
        arena_bytes as f64 / (1024.0 * 1024.0 * 1024.0),
        swap_path.display(),
        if max_disk_slots == 0 {
            "unbounded".to_string()
        } else {
            max_disk_slots.to_string()
        },
        reg.remaining_cap as f64 / (1024.0 * 1024.0 * 1024.0),
    );
    let sh = std::sync::Arc::new(SharedPaging {
        arena,
        residency: std::sync::Mutex::new(residency),
        _reservation: reservation,
    });
    reg.arenas.insert(key, sh.clone());
    Ok(sh)
}

#[cfg(test)]
#[path = "registry_tests.rs"]
mod tests;
