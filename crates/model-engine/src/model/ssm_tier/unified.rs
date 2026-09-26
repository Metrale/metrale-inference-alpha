// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-25: `METRALE_SSM_TIER_UNIFIED` and [`UnifiedSnapshotStore`], a spill store over
//! `metrale_storage::tier::Residency`.
//!
//! Owner: model-engine (SSM snapshot tier).
//! Invariants:
//! - `UnifiedSnapshotStore::put` never returns `Ok(false)` for a blob of `blob_bytes`.
//! - A store built by [`UnifiedSnapshotStore::new`] never drops a record to make room.

use std::sync::atomic::{AtomicBool, Ordering};

use anyhow::{Result, anyhow};
use parking_lot::Mutex;

use super::{BlobStoreStats, SnapshotBlobStore, SnapshotTransport};

fn unified_flag_truthy(v: Option<&str>) -> bool {
    matches!(
        v.map(str::trim),
        Some("1") | Some("true") | Some("on") | Some("yes")
    )
}

/// 2026-09-25: Whether `METRALE_SSM_TIER_UNIFIED` is `1`, `true`, `on` or `yes`
/// (off when unset). When it is, the selectors build [`UnifiedSnapshotStore`]s
/// in place of `MemBlobStore` (FIFO eviction) and `RdmaSnapshotStore` (refuses a
/// put when full).
pub(crate) fn ssm_tier_unified() -> bool {
    unified_flag_truthy(std::env::var("METRALE_SSM_TIER_UNIFIED").ok().as_deref())
}

/// 2026-09-25: Adapts a [`SnapshotTransport`] to `metrale_storage::tier::SlotArena`.
/// Slot `i` lives at offset `i × slot_bytes`, the same offsets
/// [`super::RdmaSnapshotStore`] uses.
pub(super) struct TransportSlotArena {
    pub(super) transport: Box<dyn SnapshotTransport>,
    pub(super) slot_bytes: usize,
    pub(super) num_slots: usize,
}

impl metrale_storage::tier::SlotArena for TransportSlotArena {
    fn slot_bytes(&self) -> usize {
        self.slot_bytes
    }
    fn num_slots(&self) -> usize {
        self.num_slots
    }
    fn read_slot(&self, slot: usize, out: &mut [u8]) -> Result<()> {
        if slot >= self.num_slots || out.len() != self.slot_bytes {
            anyhow::bail!("TransportSlotArena::read_slot({slot}) out of range / size mismatch");
        }
        self.transport
            .read_blob((slot * self.slot_bytes) as u64, out)
    }
    fn write_slot(&mut self, slot: usize, bytes: &[u8]) -> Result<()> {
        if slot >= self.num_slots || bytes.len() != self.slot_bytes {
            anyhow::bail!("TransportSlotArena::write_slot({slot}) out of range / size mismatch");
        }
        self.transport
            .write_blob((slot * self.slot_bytes) as u64, bytes)
    }
}

/// 2026-09-25: A [`SnapshotBlobStore`] over a mutex-guarded
/// `metrale_storage::tier::Residency`. `put` never returns `Ok(false)` for a
/// right-sized blob: a full hot arena spills its least recently used blob into
/// the swap tier. Whether the swap tier may drop records is fixed at construction:
///
/// * [`UnifiedSnapshotStore::new`]: uncapped; `Residency` never drops a record to
///   make room.
/// * [`UnifiedSnapshotStore::new_capped`]: at most `max_disk_slots` swap records;
///   the coldest one is dropped, and a later `get` for it misses.
///
/// The mutex is held across the arena and swap I/O of each call, including a
/// victim spill. `RdmaSnapshotStore` instead writes outside its lock.
pub(crate) struct UnifiedSnapshotStore {
    inner: Mutex<
        metrale_storage::tier::Residency<
            Box<dyn metrale_storage::tier::SlotArena>,
            Box<dyn metrale_storage::tier::SwapStore>,
        >,
    >,
    blob_bytes: usize,
    /// 2026-09-25: Copy of the residency's cap (0 = uncapped), read without the lock.
    max_disk_slots: usize,
    /// 2026-09-25: Set on the first put after the disk cap dropped a record, so
    /// the cap warning is logged once.
    cap_engaged: AtomicBool,
    /// 2026-09-25: Set on the first put after `Residency` wrote a swap record, so
    /// the info log is written once. Until then a 0-byte swap file is expected.
    disk_engaged: AtomicBool,
    pub stats: BlobStoreStats,
}

impl UnifiedSnapshotStore {
    /// 2026-09-25: Uncapped swap tier (`max_disk_slots = 0`): `Residency` never
    /// drops a record to make room. `build_decode_tier_store` uses this constructor, and
    /// `uncapped_new_never_drops` tests it.
    pub(super) fn new(
        arena: Box<dyn metrale_storage::tier::SlotArena>,
        swap: Box<dyn metrale_storage::tier::SwapStore>,
        blob_bytes: usize,
    ) -> Result<Self> {
        Self::new_capped(arena, swap, blob_bytes, 0)
    }

    /// 2026-09-25: At most `max_disk_slots` swap records (0 = uncapped, as
    /// [`UnifiedSnapshotStore::new`]); when full, `Residency` drops the coldest
    /// record. `build_tier_store` uses it: a dropped blob makes
    /// `try_fault_in_ssm_snapshot` return `None`, and the prefix is recomputed.
    /// The decode rollback ring never uses a store: `save_decode` and
    /// `restore_decode` copy device to device.
    pub(super) fn new_capped(
        arena: Box<dyn metrale_storage::tier::SlotArena>,
        swap: Box<dyn metrale_storage::tier::SwapStore>,
        blob_bytes: usize,
        max_disk_slots: usize,
    ) -> Result<Self> {
        let residency = metrale_storage::tier::Residency::new_capped(arena, swap, max_disk_slots)?;
        Ok(Self {
            inner: Mutex::new(residency),
            blob_bytes,
            max_disk_slots,
            cap_engaged: AtomicBool::new(false),
            disk_engaged: AtomicBool::new(false),
            stats: BlobStoreStats::default(),
        })
    }

    /// 2026-09-25: Records in the swap tier now.
    pub(crate) fn disk_records(&self) -> usize {
        self.inner.lock().disk_count()
    }

    /// 2026-09-25: Records dropped so far by the disk cap (always 0 when uncapped).
    pub(crate) fn disk_evictions(&self) -> u64 {
        self.inner.lock().stats().disk_evictions
    }
}

impl SnapshotBlobStore for UnifiedSnapshotStore {
    fn put(&self, key: u64, bytes: &[u8]) -> Result<bool> {
        // 2026-09-25: A wrong-sized blob is refused before it reaches a slot, as in
        // `RdmaSnapshotStore::put`.
        if bytes.len() != self.blob_bytes {
            self.stats.put_rejects.fetch_add(1, Ordering::Relaxed);
            return Ok(false);
        }
        let (disk_evictions, disk_records, spills, hot_slots) = {
            let mut r = self.inner.lock();
            r.put_blob(key, bytes)?;
            let n = r.arena().num_slots();
            (
                r.stats().disk_evictions,
                r.disk_count(),
                r.stats().spills_to_disk,
                n,
            )
        };
        self.stats.puts.fetch_add(1, Ordering::Relaxed);
        // 2026-09-25: The first swap-file write. `Residency` writes a record only
        // when the hot arena has no free slot, so the first `hot_slots` distinct
        // keys stay in RAM and the swap file stays at 0 bytes until then.
        if spills > 0 && !self.disk_engaged.swap(true, Ordering::Relaxed) {
            tracing::info!(
                "SSM tier disk tier ENGAGED: hot arena full ({hot_slots} slots); first record \
                 written — the swap file is no longer 0 bytes ({disk_records} records on disk)"
            );
        }
        // 2026-09-25: From the first cap drop on, warm turns for dropped prefixes
        // recompute instead of faulting in. Logged once.
        if self.max_disk_slots > 0
            && disk_evictions > 0
            && !self.cap_engaged.swap(true, Ordering::Relaxed)
        {
            tracing::warn!(
                "SSM tier disk cap engaged: dropping coldest snapshots (cap {} records = \
                 {:.2} GiB, {disk_records} on disk); warm turns for dropped prefixes \
                 recompute instead of faulting in",
                self.max_disk_slots,
                gib(self.max_disk_slots, self.blob_bytes),
            );
        }
        Ok(true)
    }

    fn get(&self, key: u64, out: &mut [u8]) -> Result<bool> {
        if out.len() != self.blob_bytes {
            self.stats.get_misses.fetch_add(1, Ordering::Relaxed);
            return Ok(false);
        }
        let hit = self.inner.lock().get_blob(key, out)?;
        if hit {
            self.stats.get_hits.fetch_add(1, Ordering::Relaxed);
        } else {
            self.stats.get_misses.fetch_add(1, Ordering::Relaxed);
        }
        Ok(hit)
    }

    fn remove(&self, key: u64) {
        self.inner.lock().remove(key);
    }

    fn len(&self) -> usize {
        self.inner.lock().total_keys()
    }

    fn bytes_resident(&self) -> usize {
        // 2026-09-25: Hot-arena blobs only; swap records are not counted.
        self.inner.lock().resident_count() * self.blob_bytes
    }
}

/// 2026-09-25: What [`build_unified_swap`] built. The O_DIRECT arm falls back to
/// host RAM with only an info log, so [`log_unified_tier`] needs this to say
/// whether a disk budget bounds disk or RAM.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum SwapBacking {
    ODirect,
    HostRam,
}

/// 2026-09-25: The unified stores' swap tier. `METRALE_SSM_TIER_SWAP_DIR` selects an
/// O_DIRECT swap file, which needs a 4 KiB-multiple blob. Otherwise, or when the
/// file cannot be created, host-RAM records.
pub(super) fn build_unified_swap(
    blob_bytes: usize,
    tag: &str,
) -> (Box<dyn metrale_storage::tier::SwapStore>, SwapBacking) {
    if let Some(dir) = std::env::var("METRALE_SSM_TIER_SWAP_DIR")
        .ok()
        .filter(|s| !s.is_empty())
    {
        if blob_bytes > 0 && blob_bytes.is_multiple_of(4096) {
            let make = || -> Result<metrale_storage::tier::DirectSwapFile> {
                std::fs::create_dir_all(&dir)?;
                let path = std::path::Path::new(&dir)
                    .join(format!("metrale-ssm-{tag}.{}.swap", std::process::id()));
                metrale_storage::tier::DirectSwapFile::create(&path, blob_bytes)
            };
            match make() {
                Ok(f) => {
                    tracing::info!("unified SSM tier ({tag}): O_DIRECT swap file in {dir}");
                    return (Box::new(f), SwapBacking::ODirect);
                }
                Err(e) => tracing::info!(
                    "unified SSM tier ({tag}): swap dir {dir} unusable ({e:#}); \
                     using host-RAM swap"
                ),
            }
        } else {
            tracing::info!(
                "unified SSM tier ({tag}): blob_bytes {blob_bytes} is not a 4 KiB multiple \
                 (O_DIRECT stride); using host-RAM swap"
            );
        }
    }
    (
        Box::new(metrale_storage::tier::MemSwapStore::new(blob_bytes)),
        SwapBacking::HostRam,
    )
}

/// 2026-09-25: Hot-arena slot count for the host-RAM unified stores:
/// `METRALE_SSM_TIER_SLOTS`, 64 when unset or unparsable, at least 1.
pub(super) fn unified_hot_slots() -> usize {
    std::env::var("METRALE_SSM_TIER_SLOTS")
        .ok()
        .and_then(|s| s.parse::<usize>().ok())
        .unwrap_or(64)
        .max(1)
}

fn gib(records: usize, blob_bytes: usize) -> f64 {
    (records as f64) * (blob_bytes as f64) / (1024.0 * 1024.0 * 1024.0)
}

pub(super) const DISK_GB_VAR: &str = "METRALE_SSM_TIER_DISK_GB";

/// 2026-09-25: Whether `METRALE_SSM_TIER_DISK_GB` is set and non-empty;
/// `build_tier_store` warns when it is set without the unified arms.
pub(super) fn disk_gb_requested() -> bool {
    std::env::var_os(DISK_GB_VAR).is_some_and(|v| !v.is_empty())
}

/// 2026-09-25: Swap-record cap for the unified spill tier from [`DISK_GB_VAR`]
/// (GiB, fractional accepted). Unset or `0` means uncapped (`0`).
pub(super) fn ssm_tier_disk_slots(blob_bytes: usize) -> Result<usize> {
    disk_slots_from(std::env::var(DISK_GB_VAR).ok().as_deref(), blob_bytes)
}

/// 2026-09-25: Env-free core of [`ssm_tier_disk_slots`].
///
/// A strict parse, unlike the lenient `unified_hot_slots`: a bad slot count only
/// mis-sizes an arena, but a misread budget here would mean uncapped.
///
/// `Residency` caps records, not bytes, and checks that the swap record size
/// equals the arena slot size, so the budget is `records × blob_bytes`.
pub(super) fn disk_slots_from(raw: Option<&str>, blob_bytes: usize) -> Result<usize> {
    let Some(raw) = raw.map(str::trim).filter(|s| !s.is_empty()) else {
        return Ok(0);
    };
    let gb: f64 = raw
        .parse()
        .map_err(|e| anyhow!("{DISK_GB_VAR}={raw:?} is not a number: {e}"))?;
    if !gb.is_finite() || gb < 0.0 {
        return Err(anyhow!(
            "{DISK_GB_VAR}={raw:?} must be a finite value >= 0 (0 = unbounded)"
        ));
    }
    if gb == 0.0 {
        return Ok(0);
    }
    if blob_bytes == 0 {
        return Err(anyhow!(
            "{DISK_GB_VAR}={raw:?} with blob_bytes 0 — no snapshot geometry to size against"
        ));
    }
    let cap_bytes = (gb * (1u64 << 30) as f64) as u64;
    let records = (cap_bytes / blob_bytes as u64) as usize;
    // 2026-09-25: Refuse fewer than 2 records instead of flooring to 1 (as the
    // peer's `carve_disk_slots` does): `records - 1` below would reach 0, which
    // `Residency` reads as uncapped.
    if records < 2 {
        return Err(anyhow!(
            "{DISK_GB_VAR}={raw:?} ({cap_bytes} B) cannot hold two SSM snapshots of \
             {blob_bytes} B — raise the budget or unset it for an unbounded tier"
        ));
    }
    // 2026-09-25: One record of headroom: `Residency::locate`'s `OnDisk` arm
    // takes the faulting key out of `disk_lru` while its record is still live,
    // so `make_disk_room` counts one fewer than exist and the swap file can
    // reach `max_disk_slots + 1` records.
    Ok(records - 1)
}

/// 2026-09-25: The unified tier's construction log. `arm` names the backing
/// ("over RDMA peer …", "in host RAM"). With a cap it states the steady and
/// worst-case sizes, and whether the cap bounds disk or RAM.
pub(super) fn log_unified_tier(
    arm: &str,
    hot_slots: usize,
    blob_bytes: usize,
    max_disk_slots: usize,
    backing: SwapBacking,
) {
    // 2026-09-25: The hot arena takes the first `hot_slots` distinct keys, so
    // the swap file stays at 0 bytes until then; `UnifiedSnapshotStore::put`
    // logs the first swap write.
    let first_disk = hot_slots + 1;
    if max_disk_slots == 0 {
        tracing::info!(
            "SSM spill tier = UNIFIED residency {arm} ({hot_slots} hot slots × \
             {blob_bytes} B, LRU spill, never rejects); disk writes begin at spill \
             #{first_disk} of distinct keys (until then the swap file is 0 bytes BY DESIGN)"
        );
        return;
    }
    let steady = gib(max_disk_slots, blob_bytes);
    let worst = gib(max_disk_slots + 1, blob_bytes);
    match backing {
        SwapBacking::ODirect => tracing::info!(
            "SSM spill tier = UNIFIED residency {arm} ({hot_slots} hot slots × {blob_bytes} B); \
             disk cap {DISK_GB_VAR} → {max_disk_slots} records × {blob_bytes} B = {steady:.2} GiB \
             steady, {worst:.2} GiB worst case (one extra record is live during a fault-in); \
             bounding: O_DIRECT swap file; disk writes begin at spill #{first_disk} of \
             distinct keys (until then the swap file is 0 bytes BY DESIGN)"
        ),
        // 2026-09-25: Host-RAM swap: the budget bounds RAM, not disk.
        SwapBacking::HostRam => tracing::warn!(
            "SSM spill tier = UNIFIED residency {arm} ({hot_slots} hot slots × {blob_bytes} B); \
             disk cap {DISK_GB_VAR} → {max_disk_slots} records × {blob_bytes} B = {steady:.2} GiB \
             steady, {worst:.2} GiB worst case; bounding: host-RAM swap (O_DIRECT unavailable — \
             this budget caps RAM, not disk; set METRALE_SSM_TIER_SWAP_DIR)"
        ),
    }
}

#[cfg(test)]
#[path = "unified_tests.rs"]
mod tests;
