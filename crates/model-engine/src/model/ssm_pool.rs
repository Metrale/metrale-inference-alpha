// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-25: The SSM state pool: fixed per-slot device regions for every SSM layer's h and
//! conv state, plus the MTP verify intermediates and checkpoints.
//!
//! Owner: model-engine.
//! Invariants:
//! - `release` frees exactly `owned_allocations`; the per-layer pointers are views into
//!   them and are only cleared.
//! - `free_slots` starts as `0..max_slots`; slot `max_slots` is the reserved dummy.

#![allow(unused_imports, dead_code)]

use parking_lot::Mutex;
use std::collections::HashMap;
use std::sync::Arc;

use anyhow::{Result, bail};
use metrale_cache::kv_cache::PagedKvCache;
use metrale_config::{LayerType, ModelConfig};
use metrale_gpu_runtime::buffers::BufferArena;
use metrale_gpu_runtime::gpu::{DevicePtr, GpuBackend, GraphHandle, KernelHandle};

use super::block_mgmt::{
    apply_evicted_blocks, ensure_blocks_through_decode, ensure_blocks_through_prefill,
    extract_layer_refs, reuse_prefix_match_disk_ids,
};
use super::ssm_snapshot::SsmSnapshotPool;
use super::types::{PinnedMetaStaging, TransformerModel};
use crate::traits::{ChunkedPrefillPageMetadata, Model, SequenceState};
use metrale_model_layers::layer::{
    AttnMetadataDev, ForwardContext, GdnPrefillBuffers, LayerState, SsmLayerState, TransformerLayer,
};
use metrale_model_layers::layers::ops;
use metrale_model_layers::speculative::DraftProposer;
use metrale_model_layers::weight_map::{DenseWeight, MtpWeights, QuantizedWeight};

pub(crate) use super::ssm_pool_slots::SlotGuard;

/// 2026-09-25: Pre-allocated device pool for SSM layer states.
///
/// Each slot has fixed device addresses for its h and conv state in every SSM
/// layer, so a captured CUDA graph's embedded addresses stay valid across
/// replays.
pub(crate) struct SsmStatePool {
    /// 2026-09-25: Base pointers returned by `GpuBackend::alloc`, owned by this
    /// pool. The per-family vectors below may be interior layer views into one
    /// contiguous allocation, so they are never freed directly.
    owned_allocations: Vec<DevicePtr>,
    pub(super) h_state_pools: Vec<DevicePtr>,
    pub(super) conv_state_pools: Vec<DevicePtr>,
    /// 2026-09-25: Per-layer H intermediate pools (MTP on, snapshot mode only):
    /// `h_inter_offsets.last()` blobs of `h_stored_bytes` each, slot `s` starting
    /// at blob `h_inter_offsets[s]`.
    pub(super) h_intermediate_pools: Vec<DevicePtr>,
    pub(super) conv_intermediate_pools: Vec<DevicePtr>,
    /// 2026-09-25: Per-layer checkpoint pools (MTP on): one blob per MTP slot,
    /// plus the MTP dummy.
    pub(super) h_checkpoint_pools: Vec<DevicePtr>,
    pub(super) conv_checkpoint_pools: Vec<DevicePtr>,
    /// 2026-09-25: FP32-width h blob bytes per layer (`config.ssm_h_state_bytes()`);
    /// the element count is `h_bytes / 4`.
    pub(super) h_bytes: usize,
    /// 2026-09-25: Storage width of one h blob inside this pool, what the h
    /// pools allocate, offset and copy by. Equals `h_bytes`, or half of it
    /// under the f16-sized pool (`ssm_reserve::ssm_h_stored_bytes`).
    pub(super) h_stored_bytes: usize,
    /// 2026-09-25: Under the f16-sized pool only: one FP32 h-state staging blob
    /// per slot (dummy included), `[max_slots + 1] × h_bytes` in one
    /// allocation. `None` under an FP32-sized pool.
    ///
    /// Shared across layers; see
    /// [`metrale_model_layers::ssm_reserve::ssm_h_prefill_stage_bytes`] for why
    /// one blob per slot suffices.
    pub(super) h_prefill_stage_pool: Option<DevicePtr>,
    pub(super) conv_bytes: usize,
    /// 2026-09-25: Number of claimable slots, excluding the reserved dummy slot
    /// at index `max_slots`.
    pub(super) max_slots: usize,
    /// 2026-09-25: Number of claimable slots covered by the MTP intermediate and
    /// checkpoint pools (`ssm_reserve::mtp_state_slots(max_slots)`; 0 without
    /// MTP). Those pools allocate `mtp_slots + 1` slots, their own dummy at
    /// index `mtp_slots`. Up to 32 slots this equals `max_slots`. Above that
    /// the scheduler dispatches speculation only when every active slot is
    /// below the same cap, and [`Self::mtp_slot`] clamps a stray access onto
    /// the MTP dummy.
    pub(super) mtp_slots: usize,
    pub(super) num_ssm_layers: usize,
    pub(super) has_mtp: bool,
    /// 2026-09-25: Per-slot count of the conv intermediates, the same for every
    /// slot (the K ceiling). Uniform H counts are `num_intermediates - 1`.
    /// Conv does not tier; see `ssm_reserve::verify_slot_h_intermediates` for
    /// why.
    pub(super) num_intermediates: usize,
    /// 2026-09-25: Per-slot H-intermediate counts, one entry per MTP slot plus
    /// the MTP dummy at index `mtp_slots` (always full width: pad rows may
    /// write any token index). Values from
    /// `ssm_reserve::verify_slot_h_intermediates`; all 0 in replay mode; empty
    /// without MTP.
    pub(super) h_inter_counts: Vec<usize>,
    /// 2026-09-25: Prefix sums over `h_inter_counts` (len + 1): slot `s`'s H
    /// intermediates start at blob `h_inter_offsets[s]` of each layer's pool;
    /// the last entry is the per-layer pool size in blobs. Fixed at
    /// allocation, so per-slot addresses stay stable for CUDA graphs.
    pub(super) h_inter_offsets: Vec<usize>,
    /// 2026-09-25: Verify-rollback mode (`--ssm-rollback-mode`, experimental).
    /// Under `Replay` the per-token intermediate pools are not allocated (the
    /// per-slot checkpoints and the input rings are) and speculative verify
    /// refuses through [`Self::require_verify_rollback_supported`].
    pub(super) rollback_mode: metrale_model_layers::ssm_reserve::SsmRollbackMode,
    /// 2026-09-25: Replay-mode verify-window input ring, one region per SSM
    /// layer: `(mtp_slots + 1) × (K-1)` rows of qkvz and gates
    /// (`ssm_reserve::ssm_replay_ring_bytes` / `ssm_replay_row_bytes`). Empty in
    /// snapshot mode. Nothing fills it: verify refuses in replay mode.
    pub(super) replay_input_rings: Vec<DevicePtr>,
    pub(super) free_slots: Mutex<Vec<usize>>,
}

/// 2026-09-25: Prefix-sum layout for per-slot H intermediates: returns
/// `(offsets, total)` where `offsets[s]` is slot `s`'s first blob index and
/// `total` the pool size in blobs. Pure, so the offset arithmetic is tested on
/// its own.
fn h_inter_layout(counts: &[usize]) -> (Vec<usize>, usize) {
    let mut offsets = Vec::with_capacity(counts.len() + 1);
    let mut acc = 0usize;
    for &c in counts {
        offsets.push(acc);
        acc += c;
    }
    offsets.push(acc);
    (offsets, acc)
}

/// 2026-09-25: Allocate one zeroed region of `bytes` per SSM layer, preferring
/// one contiguous block so `pools[l] == pools[0] + l * bytes`. Returns `(per-layer
/// pointers, allocations to free)`.
///
/// The uniform layer stride lets `model::ssm_batched_copy` issue each
/// verify-state copy set as one pitched 2-D copy per family. If the single
/// block cannot be allocated, it falls back to one allocation per layer; the
/// strided-run check then declines and the copies run one per layer. With no
/// layers or zero bytes it returns NULL pointers and allocates nothing.
fn alloc_layer_pools(
    gpu: &dyn GpuBackend,
    num_ssm_layers: usize,
    bytes: usize,
) -> Result<(Vec<DevicePtr>, Vec<DevicePtr>)> {
    if num_ssm_layers == 0 || bytes == 0 {
        return Ok((vec![DevicePtr::NULL; num_ssm_layers], Vec::new()));
    }
    if let Some(total) = bytes.checked_mul(num_ssm_layers)
        && let Ok(base) = gpu.alloc(total)
    {
        gpu.memset(base, 0, total)?;
        let layers = (0..num_ssm_layers)
            .map(|l| base.offset(l * bytes))
            .collect();
        return Ok((layers, vec![base]));
    }
    tracing::warn!(
        "SSM pool: {num_ssm_layers} × {bytes} B did not fit one contiguous block — \
         falling back to per-layer allocations (verify-state rollback keeps the \
         per-layer copy loop)"
    );
    let mut pools = Vec::with_capacity(num_ssm_layers);
    for _ in 0..num_ssm_layers {
        let p = gpu.alloc(bytes)?;
        gpu.memset(p, 0, bytes)?;
        pools.push(p);
    }
    Ok((pools.clone(), pools))
}

impl SsmStatePool {
    /// 2026-09-25: `h_f16_pool` is `qwen3_ssm::ssm_h_f16_pool_enabled()` at the
    /// one production call site (`impl_a1.rs`); it is a parameter so pool
    /// geometry is testable without the process-global flag.
    pub(super) fn new(
        config: &ModelConfig,
        max_slots: usize,
        has_mtp: bool,
        num_intermediates: usize,
        num_drafts: usize,
        h_f16_pool: bool,
        rollback_mode: metrale_model_layers::ssm_reserve::SsmRollbackMode,
        gpu: &dyn GpuBackend,
    ) -> Result<Self> {
        use metrale_circuit::state::Holding;
        use metrale_model_layers::ssm_reserve::{PoolPlan, PoolShape, PoolState, pool_counts};
        // 2026-09-30: Every size below is the pool plan's, the one the server's preflight
        // reserves (`ssm_reserve::PoolPlan`): unit counts from `pool_counts`, bytes from the
        // circuit's state declarations. Slot `max_slots` is a reserved dummy (`dummy_slot()`)
        // that batched-decode padding rows use, so padding never writes a claimed slot's
        // state. Every slot gets the full H width when the model captures DFlash layers
        // (DFlash's K=γ verify uses every slot at full width) or when `num_intermediates` is
        // not the MTP `num_drafts + 1`.
        let counts = pool_counts(&PoolShape {
            max_slots,
            spec: has_mtp,
            num_intermediates,
            num_drafts,
            uniform_h: !config.dflash_capture_layers.is_empty()
                || num_intermediates != num_drafts + 1,
            rollback: rollback_mode,
        });
        // 2026-10-02: A prompt-lookup copy tier (`ssm_reserve::copy_tier`) widens the verify
        // rows: `pool_counts` sized every slot's conv intermediates (and the copy slots' h) for
        // it, so the conv stride below follows it too. Uniform-h is judged on the MTP K above,
        // before the widening, so the tier never turns the ladder off.
        let num_intermediates = if has_mtp {
            metrale_model_layers::ssm_reserve::tier_rows(
                num_intermediates,
                metrale_model_layers::ssm_reserve::copy_tier(),
            )
        } else {
            num_intermediates
        };
        let plan = PoolPlan::new(config, &counts, h_f16_pool)?;
        let h_bytes = plan.h_f32_unit;
        let h_stored_bytes = plan.h_stored_unit;
        let conv_bytes = plan.conv_unit;
        let num_ssm_layers = plan.layers;
        let total_slots = counts.slots;

        let mut owned_allocations = Vec::new();
        let mut h_intermediate_pools = Vec::new();
        let mut conv_intermediate_pools = Vec::new();
        let mut h_checkpoint_pools = Vec::new();
        let mut conv_checkpoint_pools = Vec::new();

        let (h_state_pools, allocations) = alloc_layer_pools(
            gpu,
            num_ssm_layers,
            plan.layer_bytes(PoolState::H, Holding::Live),
        )?;
        owned_allocations.extend(allocations);
        let (conv_state_pools, allocations) = alloc_layer_pools(
            gpu,
            num_ssm_layers,
            plan.layer_bytes(PoolState::Conv, Holding::Live),
        )?;
        owned_allocations.extend(allocations);

        // 2026-09-25: The FP32 prefill staging arena exists only when the h
        // slots are narrowed (`h_stored_bytes < h_bytes`); otherwise
        // `ssm_h_prefill_stage_bytes` returns 0 and nothing is allocated.
        let h_prefill_stage_pool = {
            let bytes = metrale_model_layers::ssm_reserve::ssm_h_prefill_stage_bytes(
                total_slots,
                h_bytes,
                h_stored_bytes < h_bytes,
            );
            if bytes == 0 {
                None
            } else {
                let p = gpu.alloc(bytes)?;
                gpu.memset(p, 0, bytes)?;
                owned_allocations.push(p);
                tracing::info!(
                    "SSM f16-sized h pool: FP32 prefill staging arena {} MB ({total_slots} slots × {h_bytes} B)",
                    bytes / (1024 * 1024)
                );
                Some(p)
            }
        };

        // 2026-09-25: MTP verify pools cover only the slots speculative dispatch can reach
        // (`ssm_reserve::mtp_state_slots`, which the scheduler's `spec_slot_cap` uses), plus
        // their own dummy; the h intermediates per slot are `pool_counts`' (K-1, tiered by
        // the MTP ladder, the dummy at full width; 0 under replay, which keeps none).
        let replay = rollback_mode == metrale_model_layers::ssm_reserve::SsmRollbackMode::Replay;
        let h_inter_counts: Vec<usize> = counts
            .verify
            .as_ref()
            .map(|v| v.h_steps.clone())
            .unwrap_or_default();
        let mtp_slots = h_inter_counts.len().saturating_sub(1);
        let (h_inter_offsets, h_inter_total) = h_inter_layout(&h_inter_counts);
        let mut replay_input_rings = Vec::new();
        if has_mtp {
            let ni = num_intermediates;
            let mtp_total = mtp_slots + 1;
            if !replay {
                let (layers, allocations) = alloc_layer_pools(
                    gpu,
                    num_ssm_layers,
                    plan.layer_bytes(PoolState::H, Holding::Steps),
                )?;
                h_intermediate_pools = layers;
                owned_allocations.extend(allocations);
                let (layers, allocations) = alloc_layer_pools(
                    gpu,
                    num_ssm_layers,
                    plan.layer_bytes(PoolState::Conv, Holding::Steps),
                )?;
                conv_intermediate_pools = layers;
                owned_allocations.extend(allocations);
            } else {
                // 2026-09-25: Replay: verify-window input rows instead of state
                // snapshots, (mtp_total slots incl. dummy) × (K-1) rows of
                // qkvz and gates per layer, sized by the function preflight
                // also reserves through.
                let row = metrale_model_layers::ssm_reserve::ssm_replay_row_bytes(
                    config.ssm_qkvz_size(),
                    config.linear_num_value_heads,
                );
                let ring =
                    metrale_model_layers::ssm_reserve::ssm_replay_ring_bytes(1, row, ni, mtp_total);
                let (layers, allocations) = alloc_layer_pools(gpu, num_ssm_layers, ring)?;
                replay_input_rings = layers;
                owned_allocations.extend(allocations);
            }

            // 2026-09-25: One checkpoint per MTP slot (dummy included) per
            // layer, in both rollback modes.
            let (layers, allocations) = alloc_layer_pools(
                gpu,
                num_ssm_layers,
                plan.layer_bytes(PoolState::H, Holding::Checkpoint),
            )?;
            h_checkpoint_pools = layers;
            owned_allocations.extend(allocations);
            let (layers, allocations) = alloc_layer_pools(
                gpu,
                num_ssm_layers,
                plan.layer_bytes(PoolState::Conv, Holding::Checkpoint),
            )?;
            conv_checkpoint_pools = layers;
            owned_allocations.extend(allocations);

            let mtp_mb = num_ssm_layers
                * (h_inter_total * h_stored_bytes
                    + mtp_total * (ni * conv_bytes + h_stored_bytes + conv_bytes))
                / (1024 * 1024);
            // 2026-09-25: Log baseline: full-width uniform sizing, per slot
            // (ni-1) H + ni conv + a checkpoint.
            let full_h = mtp_total * ni.saturating_sub(1);
            if mtp_slots < max_slots || h_inter_total < full_h {
                let saved_mb = (num_ssm_layers
                    * (max_slots - mtp_slots)
                    * (ni.saturating_sub(1) * h_stored_bytes
                        + ni * conv_bytes
                        + h_stored_bytes
                        + conv_bytes)
                    + num_ssm_layers * (full_h - h_inter_total) * h_stored_bytes)
                    / (1024 * 1024);
                tracing::info!(
                    "SSM MTP pools (conv {ni}/slot, h tiered {:?}..{:?} + checkpoints): \
                     {mtp_mb} MB, covering {mtp_slots}/{max_slots} slots (spec dispatch \
                     width; saves {saved_mb} MB vs full-width uniform; kill switch \
                     METRALE_MTP_POOL_FULL_WIDTH)",
                    h_inter_counts.iter().min(),
                    h_inter_counts.iter().max(),
                );
            } else {
                tracing::info!("SSM MTP pools ({ni} intermediates + checkpoints): {mtp_mb} MB");
            }
        }

        // 2026-09-25: free_slots holds claimable indices only; the dummy at
        // index `max_slots` is not among them.
        let free_slots: Vec<usize> = (0..max_slots).rev().collect();

        let total_mb = num_ssm_layers * max_slots * (h_stored_bytes + conv_bytes) / (1024 * 1024);
        tracing::info!(
            "SSM state pool: {max_slots} slots × {num_ssm_layers} layers = {total_mb} MB",
        );

        Ok(Self {
            owned_allocations,
            h_state_pools,
            conv_state_pools,
            h_intermediate_pools,
            conv_intermediate_pools,
            h_checkpoint_pools,
            conv_checkpoint_pools,
            h_bytes,
            h_stored_bytes,
            h_prefill_stage_pool,
            conv_bytes,
            max_slots,
            mtp_slots,
            num_ssm_layers,
            has_mtp,
            num_intermediates,
            h_inter_counts,
            h_inter_offsets,
            rollback_mode,
            replay_input_rings,
            free_slots: Mutex::new(free_slots),
        })
    }
}

/// 2026-09-25: Free every allocation the pool owns and clear the per-layer
/// views. Every free is attempted; the first error is returned.
impl metrale_core::scope::ModelResource<dyn GpuBackend> for SsmStatePool {
    fn label(&self) -> &'static str {
        "ssm state pool"
    }

    fn release(&mut self, gpu: &dyn GpuBackend) -> anyhow::Result<()> {
        let mut first_error = None;
        for ptr in self.owned_allocations.drain(..) {
            if let Err(e) = gpu.free(ptr)
                && first_error.is_none()
            {
                first_error = Some(e);
            }
        }
        self.h_state_pools.clear();
        self.conv_state_pools.clear();
        self.h_intermediate_pools.clear();
        self.conv_intermediate_pools.clear();
        self.h_checkpoint_pools.clear();
        self.conv_checkpoint_pools.clear();
        self.h_prefill_stage_pool = None;
        self.replay_input_rings.clear();
        match first_error {
            Some(e) => Err(e),
            None => Ok(()),
        }
    }
}

#[cfg(test)]
#[path = "ssm_pool_h_inter_layout_tests.rs"]
mod h_inter_layout_tests;

#[cfg(test)]
#[path = "ssm_pool_h_stored_geometry_tests.rs"]
mod h_stored_geometry_tests;

#[cfg(test)]
#[path = "ssm_pool_slot_guard_tests.rs"]
mod slot_guard_tests;

#[cfg(test)]
#[path = "ssm_pool_state_plan_tests.rs"]
mod state_plan_tests;
