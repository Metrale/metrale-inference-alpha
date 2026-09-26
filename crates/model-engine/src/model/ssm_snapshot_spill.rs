// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-25: The write half of the SSM spill tier: spilling a Marconi snapshot to a
//! blob store, acquiring a slot for a fault-in, and one fault-in cycle per key.
//!
//! Owner: model-engine SSM snapshot pool.
//! Invariants:
//! - Once `spill_slot` has enqueued its gather, it calls `synchronize(stream)` before
//!   returning, even when an enqueue failed.
//! - A slot popped by `try_pop_free_slot` has no side-table entries.

#![allow(unused_imports, dead_code)]

use std::sync::atomic::{AtomicBool, Ordering};

use anyhow::Result;
use metrale_cache::kv_cache::PagedKvCache;
use metrale_gpu_runtime::gpu::GpuBackend;
use metrale_telemetry::prefix_cache::TierEvict;

use super::ssm_snapshot::SsmSnapshotPool;
use super::ssm_snapshot_faultin::{log_spill_gate_skip, retire_refused_spill};
use super::ssm_spill_gate::spill_min_tokens;

/// 2026-09-25: Latch so `log_spill_shape_once` logs the spill path once per process.
static SPILL_SHAPE_LOGGED: AtomicBool = AtomicBool::new(false);

impl SsmSnapshotPool {
    /// 2026-09-25: Tag Marconi slot `snap_slot` with `session_hash`; 0 leaves it untagged.
    pub(super) fn tag_session(&self, snap_slot: usize, session_hash: u64) {
        if session_hash != 0 {
            self.session_tags.lock().insert(snap_slot, session_hash);
        }
    }

    /// 2026-09-25: Pop a free Marconi slot without evicting and clear its side tables;
    /// `None` when the region is disabled or has no free slot. The caller must `free`
    /// a slot it does not keep.
    pub(super) fn try_pop_free_slot(&self) -> Option<usize> {
        if !self.is_enabled() {
            return None;
        }
        let snap_slot = self.free_slots.lock().pop()?;
        self.clear_slot_bookkeeping(snap_slot);
        Some(snap_slot)
    }

    /// 2026-09-25: Acquire a Marconi slot for a fault-in. Pop a free slot; otherwise
    /// take a victim from `evict_snapshot_to_tier`, which picks only resident entries
    /// (`skip_tiered` in `radix_tree/snapshot_tier.rs`). A `Spill` victim is spilled
    /// and its index entry stays findable; a `Drop` victim's entry was removed by the
    /// cost gate. Either way its slot is freed and popped. `None` when there is
    /// neither a free slot nor a resident victim.
    pub(super) fn acquire_or_spill_slot(
        &self,
        prefix_cache: &dyn metrale_telemetry::prefix_cache::PrefixCache,
        store: &dyn super::ssm_tier::SnapshotBlobStore,
        gpu: &dyn GpuBackend,
    ) -> Option<usize> {
        if let Some(s) = self.try_pop_free_slot() {
            return Some(s);
        }
        let evict = prefix_cache.evict_snapshot_to_tier(spill_min_tokens())?;
        if let TierEvict::Spill { slot, key, .. } = evict {
            let stream = gpu.default_stream();
            match self.spill_slot(slot, key, store, gpu, stream) {
                Ok(true) => {}
                Ok(false) => retire_refused_spill(prefix_cache, store, key),
                Err(e) => tracing::warn!(
                    "SSM spill during fault-in acquire failed ({e:#}); freeing slot anyway, \
                     key {key} RETAINED (an error is not evidence of absence)"
                ),
            }
        } else {
            log_spill_gate_skip(&evict);
        }
        self.free(evict.slot());
        self.try_pop_free_slot()
    }

    /// 2026-09-25: One fault-in cycle for a tiered anchor: acquire a slot, read the
    /// blob into it, and on success promote the index entry to that slot and tag its
    /// session. `None` means nothing was restored. The caller,
    /// `TransformerModel::try_fault_in_ssm_snapshot` (`trait_impl/ssm_fault_in.rs`),
    /// applies the resident-hit, store-present and `METRALE_SSM_FAULT_MIN_TOKENS`
    /// gates. As a pool method the cycle runs in CPU-only tests
    /// (`ssm_snapshot_reap_tests.rs`: `MockGpuBackend`, `RadixTree`, `MemBlobStore`).
    pub(in crate::model) fn fault_in_for_key(
        &self,
        prefix_cache: &dyn metrale_telemetry::prefix_cache::PrefixCache,
        store: &dyn super::ssm_tier::SnapshotBlobStore,
        gpu: &dyn GpuBackend,
        key: u64,
        session_hash: u64,
        depth: usize,
        stream: u64,
    ) -> Option<usize> {
        let slot = self.acquire_or_spill_slot(prefix_cache, store, gpu)?;
        match self.fault_in_slot(slot, key, store, gpu, stream) {
            Ok(true) => {
                // 2026-09-25: `false` means no index entry owns this key any more,
                // so the index will not find the restored slot again.
                if !prefix_cache.promote_snapshot(key, slot) {
                    tracing::warn!(
                        "SSM tier fault-in restored key {key} but no index entry accepted the \
                         promotion — this prefix will recompute next turn"
                    );
                }
                // 2026-09-25: The acquired slot is untagged; tag it with the session
                // whose lookup matched this key.
                self.tag_session(slot, session_hash);
                tracing::info!(
                    "SSM tier fault-in: restored spilled snapshot at token {depth} into slot {slot}"
                );
                Some(slot)
            }
            // 2026-09-25: Miss: the store has no bytes for this key. Retire the index
            // entry so the prefix recomputes once, instead of repeating a victim
            // spill and a miss on every warm turn.
            Ok(false) => {
                self.free(slot);
                // 2026-09-25: Forget the index entry first; `forget_snapshot_tier_key`
                // removes only a still-tiered entry. Remove the blob only when it did.
                if prefix_cache.forget_snapshot_tier_key(key) {
                    // 2026-09-25: `get` can miss while keeping the record, e.g. on a
                    // length mismatch in `RdmaSnapshotStore::get`, so remove it here.
                    store.remove(key);
                    tracing::info!(
                        "SSM tier reap: no blob for key {key} (depth {depth} tok) — retired the \
                         index entry; this prefix now recomputes once instead of re-spilling a \
                         live snapshot every turn. Sustained reaps mean METRALE_SSM_TIER_DISK_GB \
                         is undersized for the working set."
                    );
                }
                None
            }
            // 2026-09-25: An error does not show the blob is gone, so the key is
            // kept; the miss arm retires it once a `get` reports absence.
            Err(e) => {
                self.free(slot);
                tracing::warn!(
                    "SSM tier fault-in failed ({e:#}); key {key} RETAINED (an error is not \
                     evidence of absence) — recomputing this turn, will retry next turn"
                );
                None
            }
        }
    }

    /// 2026-09-25: Bytes in one slot's spill blob: every SSM layer's h and conv state,
    /// laid out `[h_0 conv_0 h_1 conv_1 … h_{L-1} conv_{L-1}]`.
    pub(super) fn spill_blob_bytes(&self) -> usize {
        self.num_ssm_layers * (self.h_bytes + self.conv_bytes)
    }

    /// 2026-09-25: Release the shared spill and fault-in staging buffer. The pool
    /// holds no `gpu` handle, so `TransformerModel::drop` calls this.
    pub(crate) fn free_staging(&self, gpu: &dyn GpuBackend) {
        self.spill_staging.free(gpu);
    }

    /// 2026-09-25: Spill Marconi slot `snap_slot`: gather its per-layer h and conv
    /// state into the staging blob with async D2H copies on `stream`, then `put` it
    /// under `key` (the index entry's prefix hash). Returns the store's verdict, or
    /// `false` when the region is disabled. `stream` is synchronised before the
    /// gather, so a `save` queued on it into this slot has landed, and after it, so
    /// every chunk has landed before `put` reads the blob.
    pub(super) fn spill_slot(
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
        gpu.synchronize(stream)?;
        let bytes = self.spill_blob_bytes();
        let mut guard = self.spill_staging.acquire(gpu, bytes)?;
        let kind = guard.kind();
        self.log_spill_shape_once(bytes, kind);
        // 2026-09-25: Not zeroed: the gather writes every byte.
        let blob = guard.as_mut_slice();
        let gather = self.gather_async(snap_slot, blob, gpu, stream);
        // 2026-09-25: One synchronise for all chunks, run even when an enqueue failed:
        // chunks already queued are still writing into the shared staging buffer.
        gpu.synchronize(stream)?;
        gather?;
        let t_put = std::time::Instant::now();
        let r = store.put(key, blob)?;
        if timing {
            tracing::info!(
                "SSM spill: {} B  gather+sync={}us  store.put={}us  total={}us  staging={}",
                bytes,
                t_put.duration_since(t0).as_micros(),
                t_put.elapsed().as_micros(),
                t0.elapsed().as_micros(),
                kind,
            );
        }
        Ok(r)
    }

    /// 2026-09-25: Enqueue the per-layer D2H chunks of `snap_slot` into `blob`; the
    /// caller synchronises.
    fn gather_async(
        &self,
        snap_slot: usize,
        blob: &mut [u8],
        gpu: &dyn GpuBackend,
        stream: u64,
    ) -> Result<()> {
        let per_layer = self.h_bytes + self.conv_bytes;
        for i in 0..self.num_ssm_layers {
            let off = i * per_layer;
            gpu.copy_d2h_async(
                self.h_snapshots[i].offset(snap_slot * self.h_bytes),
                &mut blob[off..off + self.h_bytes],
                stream,
            )?;
            gpu.copy_d2h_async(
                self.conv_snapshots[i].offset(snap_slot * self.conv_bytes),
                &mut blob[off + self.h_bytes..off + per_layer],
                stream,
            )?;
        }
        Ok(())
    }

    /// 2026-09-25: Enqueue the per-layer H2D chunks of `blob` into `snap_slot`; the
    /// caller synchronises. `blob` is the staging buffer, which the caller holds
    /// until after that synchronise, as `copy_h2d_async_retained` requires. The CUDA
    /// `copy_h2d_async` would synchronise after every chunk from a page-locked source
    /// (`cuda_backend/gpu_impl.rs`).
    pub(super) fn scatter_async(
        &self,
        snap_slot: usize,
        blob: &[u8],
        gpu: &dyn GpuBackend,
        stream: u64,
    ) -> Result<()> {
        let per_layer = self.h_bytes + self.conv_bytes;
        for i in 0..self.num_ssm_layers {
            let off = i * per_layer;
            gpu.copy_h2d_async_retained(
                &blob[off..off + self.h_bytes],
                self.h_snapshots[i].offset(snap_slot * self.h_bytes),
                stream,
            )?;
            gpu.copy_h2d_async_retained(
                &blob[off + self.h_bytes..off + per_layer],
                self.conv_snapshots[i].offset(snap_slot * self.conv_bytes),
                stream,
            )?;
        }
        Ok(())
    }

    fn log_spill_shape_once(&self, bytes: usize, kind: &str) {
        if SPILL_SHAPE_LOGGED.swap(true, Ordering::Relaxed) {
            return;
        }
        tracing::info!(
            "SSM spill path: {} async D2H chunks + 1 stream sync into a reusable {bytes} B \
             {kind} staging buffer (was {} blocking copies + a fresh heap blob, measured \
             ~400ms/spill)",
            2 * self.num_ssm_layers,
            2 * self.num_ssm_layers,
        );
    }
}
