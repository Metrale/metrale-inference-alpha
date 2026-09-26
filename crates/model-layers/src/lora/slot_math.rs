// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-25: GPU-free LoRA slot math: victim selection for the promotion
//! cache, the per-row `seq_slot` buffer, scale-table values, the routed-prefill
//! slot and pair, and the per-slot byte layout of the adapter pool.
//!
//! Owner: model-layers (lora).
//! Invariants:
//! - `pool_slot_bytes`, `module_slot_offsets` and `pack_slot` (`loading.rs`)
//!   walk one layout: layers ascending, then `LoraModule::ALL` filtered by
//!   `applies_to_layer`, A (`max_rank * in` elements) then B
//!   (`out * max_rank`), BF16.

use metrale_config::ModelConfig;

use super::*;
use crate::layers::ops::lora_delta::LoraPair;

pub(crate) const BF16_BYTES: usize = 2;

/// 2026-09-25: Pick the slot a promotion overwrites, from `cache`, the
/// `(slot_index, view)` pairs the caller passes (model-engine passes
/// `cache_slot_views()`, slots `[pinned, max_loras)`):
///   1. the first unfilled slot;
///   2. else the `ref_count == 0` slot with the smallest `last_used`;
///   3. else `Err(PoolFull)`.
pub fn select_victim_slot(cache: &[(usize, SlotView)]) -> Result<usize, VictimError> {
    if let Some((idx, _)) = cache.iter().find(|(_, v)| !v.filled) {
        return Ok(*idx);
    }
    cache
        .iter()
        .filter(|(_, v)| v.ref_count == 0)
        .min_by_key(|(_, v)| v.last_used)
        .map(|(idx, _)| *idx)
        .ok_or(VictimError::PoolFull)
}

/// 2026-09-25: The host `seq_slot[padded_n]` buffer the bgmv reads, one entry
/// per row:
///   row i < n: `adapter_slots[i]` if `>= 0`, else `active` (a negative slot
///     means the active adapter);
///   row i >= n (padding): `-1`, which `lora_bgmv.cu` treats as no delta.
pub fn build_seq_slot_host(adapter_slots: &[i32], padded_n: usize, active: i32) -> Vec<i32> {
    let n = adapter_slots.len();
    (0..padded_n)
        .map(|i| {
            if i < n {
                let s = adapter_slots[i];
                if s >= 0 { s } else { active }
            } else {
                -1
            }
        })
        .collect()
}

/// 2026-09-25: Values of the `[max_loras]` f32 scale table: entry `k` is
/// adapter `k`'s `scaling()` (alpha/r, or alpha/√r under rsLoRA), 0.0 for
/// `k >= adapters.len()`.
pub(crate) fn scale_table_values(adapters: &[LoraAdapterInput<'_>], max_loras: usize) -> Vec<f32> {
    let mut v = vec![0.0f32; max_loras];
    for (k, a) in adapters.iter().enumerate() {
        v[k] = a.peft.scaling();
    }
    v
}

/// 2026-09-25: The predicate behind [`LoraWeights::routed_prefill_slot`].
/// Resolves `adapter_slot` (`>= 0` is that slot, negative is `active`) and
/// returns it only when it is not `active` and is `< num_slots`; otherwise
/// `None`.
pub fn routed_prefill_slot_of(adapter_slot: i32, active: usize, num_slots: usize) -> Option<usize> {
    let resolved = if adapter_slot >= 0 {
        adapter_slot as usize
    } else {
        active
    };
    (resolved != active && resolved < num_slots).then_some(resolved)
}

/// 2026-09-25: The [`LoraPair`] for (global layer, module) in a slot's
/// global-layer-indexed `layers`. `None` when the index is out of range, the
/// layer has no pairs, or the adapter has no pair for that module.
pub fn select_routed_pair(
    layers: &[Option<LoraLayerWeights>],
    global_layer_idx: usize,
    module: LoraModule,
) -> Option<&LoraPair> {
    layers
        .get(global_layer_idx)
        .and_then(|o| o.as_ref())
        .and_then(|l| l.module_pair(module))
}

/// 2026-09-25: Bytes of one pool slot: over every layer and every module that
/// `applies_to_layer`, `(max_rank * in + out * max_rank) * 2`.
pub(crate) fn pool_slot_bytes(cfg: &ModelConfig, max_rank: usize) -> usize {
    (0..cfg.num_hidden_layers)
        .map(|layer| {
            LoraModule::ALL
                .iter()
                .filter(|m| m.applies_to_layer(cfg, layer))
                .map(|m| {
                    let (out, inp) = m.dims(cfg);
                    (max_rank * inp + out * max_rank) * BF16_BYTES
                })
                .sum::<usize>()
        })
        .sum()
}

/// 2026-09-25: Byte offset of slot `slot` in the pool:
/// `slot * pool_slot_bytes`.
// 2026-09-25: Its callers are `rdma_stage` and tests; `rdma_stage` compiles only
// on unix, with `cuda` or under test (`mod.rs`).
#[cfg_attr(not(all(feature = "cuda", unix)), allow(dead_code))]
pub(crate) fn slot_base_offset(slot: usize, cfg: &ModelConfig, max_rank: usize) -> usize {
    slot * pool_slot_bytes(cfg, max_rank)
}

/// 2026-09-25: The (a_off, b_off) of (layer, module) within a slot: the
/// offsets `pack_slot` reaches for that pair. `None` when `applies_to_layer`
/// is false for it or the layer is out of range. Called by `rdma_stage` and
/// tests.
#[cfg_attr(not(all(feature = "cuda", unix)), allow(dead_code))]
pub(crate) fn module_slot_offsets(
    cfg: &ModelConfig,
    max_rank: usize,
    target_layer: usize,
    target_module: LoraModule,
) -> Option<(usize, usize)> {
    // 2026-09-25: The same walk as `pack_slot` and `pool_slot_bytes`; RDMA
    // landing writes at these offsets.
    let mut off = 0usize;
    for layer_idx in 0..cfg.num_hidden_layers {
        for module in LoraModule::ALL {
            if !module.applies_to_layer(cfg, layer_idx) {
                continue;
            }
            let (out_dim, in_dim) = module.dims(cfg);
            let a_off = off;
            let b_off = off + max_rank * in_dim * BF16_BYTES;
            off = b_off + out_dim * max_rank * BF16_BYTES;
            if layer_idx == target_layer && module == target_module {
                return Some((a_off, b_off));
            }
        }
    }
    None
}

#[cfg(test)]
#[path = "slot_math_tests.rs"]
mod tests;
