// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-25: The read half of the SSM spill tier: faulting a spilled snapshot back
//! into a Marconi slot, reclaiming a slot from the prefix cache, and retiring index
//! entries whose spill the store refused.
//!
//! Owner: model-engine SSM snapshot pool.
//! Invariants:
//! - When the tier is set and the prefix cache yields a victim, `reclaim_from_cache`
//!   frees the victim's slot whatever the spill outcome.
//! - Once `fault_in_slot` has enqueued its scatter, it calls `synchronize(stream)`
//!   before returning, even when an enqueue failed.

use anyhow::Result;
use metrale_gpu_runtime::gpu::GpuBackend;
use metrale_telemetry::prefix_cache::{PrefixCache, TierEvict};

use metrale_cache::kv_cache::PagedKvCache;

use super::ssm_snapshot::SsmSnapshotPool;
use super::ssm_spill_gate::spill_min_tokens;

impl SsmSnapshotPool {
    /// 2026-09-25: Fault in the blob for `key` into Marconi slot `snap_slot`: read it
    /// into the staging buffer shared with `spill_slot` and scatter it H2D into the
    /// slot's per-layer h and conv state. Returns `false` when the region is disabled
    /// or the store has no blob for `key`. The trailing synchronise means a later
    /// `restore` from this slot reads the scattered bytes.
    pub(super) fn fault_in_slot(
        &self,
        snap_slot: usize,
        key: u64,
        store: &dyn super::ssm_tier::SnapshotBlobStore,
        gpu: &dyn GpuBackend,
        stream: u64,
    ) -> Result<bool> {
        if !self.is_enabled() {
            return Ok(false);
        }
        let timing = std::env::var_os("METRALE_SSM_TIER_TIMING").is_some();
        let t0 = std::time::Instant::now();
        let bytes = self.spill_blob_bytes();
        let mut guard = self.spill_staging.acquire(gpu, bytes)?;
        let kind = guard.kind();
        let blob = guard.as_mut_slice();
        let hit = store.get(key, blob)?;
        let get_us = t0.elapsed().as_micros();
        if !hit {
            return Ok(false);
        }
        let scatter = self.scatter_async(snap_slot, blob, gpu, stream);
        // 2026-09-25: Synchronise before checking the scatter result: queued chunks
        // still read the shared staging buffer.
        gpu.synchronize(stream)?;
        scatter?;
        if timing {
            tracing::info!(
                "SSM fault-in: {} B  store.get(RDMA read)={}us  scatter+sync={}us  total={}us  \
                 staging={}",
                bytes,
                get_us,
                t0.elapsed().as_micros() - get_us,
                t0.elapsed().as_micros(),
                kind,
            );
        }
        Ok(true)
    }

    /// 2026-09-25: Free a Marconi slot by evicting a snapshot from the prefix cache's
    /// snapshot index; KV blocks are not evicted. With a `tier`, the victim comes from
    /// `evict_snapshot_to_tier(spill_min_tokens())`: a `Spill` victim's bytes go to
    /// the tier and its index entry stays findable, while a victim shallower than the
    /// gate ([`super::ssm_spill_gate`]) is dropped. Without a tier the LRU snapshot is
    /// dropped. Returns whether a slot was freed.
    pub(super) fn reclaim_from_cache(
        &self,
        prefix_cache: &dyn metrale_telemetry::prefix_cache::PrefixCache,
        _kv_cache: &mut PagedKvCache,
        tier: Option<&dyn super::ssm_tier::SnapshotBlobStore>,
        gpu: &dyn GpuBackend,
    ) -> bool {
        if let Some(store) = tier {
            // 2026-09-25: `spill_slot` runs on the default stream and synchronises it
            // first, so a `save` queued there into the victim slot has landed.
            let Some(evict) = prefix_cache.evict_snapshot_to_tier(spill_min_tokens()) else {
                return false;
            };
            match evict {
                TierEvict::Spill { slot, key, .. } => {
                    let stream = gpu.default_stream();
                    match self.spill_slot(slot, key, store, gpu, stream) {
                        Ok(true) => {}
                        Ok(false) => {
                            retire_refused_spill(prefix_cache, store, key);
                        }
                        Err(e) => {
                            // 2026-09-25: The key is kept: an error does not show the
                            // bytes are absent, and a later fault-in miss retires it.
                            tracing::warn!(
                                "SSM spill failed ({e:#}); freeing slot, key {key} retained — \
                                 entry will miss on fault-in and be retired there"
                            );
                        }
                    }
                }
                TierEvict::Drop { .. } => log_spill_gate_skip(&evict),
            }
            self.free(evict.slot());
            return true;
        }
        if let Some(snap) = prefix_cache.evict_snapshot_lru() {
            self.free(snap);
            true
        } else {
            false
        }
    }
}

/// 2026-09-25: Retire the index entry after `spill_slot` returned `Ok(false)`.
/// `evict_to_tier` already marked the entry tiered, so without this it would stay
/// findable with no bytes behind it. The index entry is forgotten first, and the
/// blob removed only when a still-tiered entry was removed
/// (`forget_snapshot_tier_key`).
pub(super) fn retire_refused_spill(
    prefix_cache: &dyn metrale_telemetry::prefix_cache::PrefixCache,
    store: &dyn super::ssm_tier::SnapshotBlobStore,
    key: u64,
) {
    if prefix_cache.forget_snapshot_tier_key(key) {
        store.remove(key);
        tracing::warn!(
            "SSM spill tier refused a blob for key {key}; retired the entry rather than \
             leaving it findable-but-empty — a dead tier key costs one live-snapshot spill \
             per warm turn to rediscover"
        );
    }
}

/// 2026-09-25: Log a spill skipped by the cost gate (`TierEvict::Drop`).
pub(super) fn log_spill_gate_skip(evict: &TierEvict) {
    if let TierEvict::Drop { depth, .. } = *evict {
        tracing::info!(
            "SSM spill SKIPPED (cost gate): victim depth {depth} < \
             METRALE_SSM_SPILL_MIN_TOKENS={} — dropped instead; a ~45ms spill cannot repay \
             {depth} tokens of prefill",
            spill_min_tokens(),
        );
    }
}

#[cfg(test)]
#[path = "ssm_snapshot_spill_tests.rs"]
mod tier_tests;

#[cfg(test)]
#[path = "ssm_snapshot_reap_tests.rs"]
mod reap_tests;
