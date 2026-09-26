// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-25: GPU-free MoE LoRA routing: the per-request fold decision
//! ([`resolve_moe_lora_route`]), the refusal of a decode batch the
//! single-active fold cannot serve, and the per-row adapter maps the MoE
//! LoRA kernels read (`< 0` means no fold on that row).
//!
//! Owner: model-layers (lora).
//! Invariants: none beyond the types.

use crate::layer::MoeLoraRoute;

/// 2026-09-25: An error when `route` is `Refuse`. The batched decode entries
/// (model-engine `decode_batch_compute_main`, `mixed_forward`) call it before
/// any graph lookup: [`build_moe_row_adapter_decode`] writes a `Refuse` row as
/// `-1`, so the batch would otherwise run that request without its adapter.
pub fn ensure_decode_route_servable(route: MoeLoraRoute, path: &str) -> anyhow::Result<()> {
    anyhow::ensure!(
        !matches!(route, MoeLoraRoute::Refuse),
        "MoE LoRA {path}: a sequence routes to a non-active adapter under single-active \
         phase-1; refusing rather than mis-folding the active adapter. One adapter per batch."
    );
    Ok(())
}

/// 2026-09-25: The MoE LoRA fold decision for one request:
/// - `!has_moe_lora` ⇒ `Fold`;
/// - `adapter_slot < 0` (base request) ⇒ `Skip`;
/// - `adapter_slot == active` ⇒ `Fold`;
/// - any other slot ⇒ `Refuse`, since only the active adapter is installed.
///
/// Model-engine's `moe_lora_route` passes `has_moe_lora` = "a LoRA pool is
/// loaded" and `active = -1` when none is.
pub fn resolve_moe_lora_route(adapter_slot: i32, active: i32, has_moe_lora: bool) -> MoeLoraRoute {
    if !has_moe_lora {
        return MoeLoraRoute::Fold;
    }
    if adapter_slot < 0 {
        return MoeLoraRoute::Skip;
    }
    if adapter_slot == active {
        MoeLoraRoute::Fold
    } else {
        MoeLoraRoute::Refuse
    }
}

/// 2026-09-25: The `[total_tokens]` per-row adapter map of a packed prefill
/// batch. `cu_seqlens_host` is the `[batch + 1]` prefix sum of per-stream token
/// counts; stream `b`'s `adapter_slots[b]` is written, unchanged, over rows
/// `cu_seqlens_host[b]..cu_seqlens_host[b + 1]`. Only tests call it today.
///
/// `None`, before any row is written, when `cu_seqlens_host` has fewer than two
/// entries, `adapter_slots` is not `batch` long, the first entry is not 0, or
/// any boundary is negative or decreasing.
pub fn build_moe_row_adapter_host(
    cu_seqlens_host: &[i32],
    adapter_slots: &[i32],
) -> Option<Vec<i32>> {
    if cu_seqlens_host.len() < 2 {
        return None;
    }
    let batch = cu_seqlens_host.len() - 1;
    if adapter_slots.len() != batch {
        return None;
    }
    if cu_seqlens_host[0] != 0 {
        return None;
    }
    // 2026-09-25: Check every boundary before sizing the map from the last
    // one, so `[0, 4, 2]` cannot write past it.
    for b in 0..batch {
        let start = cu_seqlens_host[b];
        let end = cu_seqlens_host[b + 1];
        if start < 0 || end < start {
            return None;
        }
    }
    let total = cu_seqlens_host[batch];
    let mut map = vec![-1i32; total as usize];
    for b in 0..batch {
        let start = cu_seqlens_host[b];
        let end = cu_seqlens_host[b + 1];
        let slot = adapter_slots[b];
        for row in start..end {
            map[row as usize] = slot;
        }
    }
    Some(map)
}

/// 2026-09-25: The `[padded_n]` per-row adapter map of a decode batch, one
/// token per sequence, which `moe_lora_gather_bgmv.cu` reads as
/// `row_adapter[row / top_k]` and skips when negative. Row `i` is `active`
/// when `resolve_moe_lora_route` gives `Fold`, else `-1`; so `Skip`, `Refuse`
/// and padding rows (`i >= adapter_slots.len()`) are all `-1`, which is not the
/// attention `seq_slot` convention (there `-1` means the active adapter).
/// Model-engine uploads it in `upload_moe_row_adapter`.
pub fn build_moe_row_adapter_decode(
    adapter_slots: &[i32],
    padded_n: usize,
    active: i32,
    has_moe_lora: bool,
) -> Vec<i32> {
    (0..padded_n)
        .map(|i| match adapter_slots.get(i).copied() {
            Some(slot) => match resolve_moe_lora_route(slot, active, has_moe_lora) {
                MoeLoraRoute::Fold => active,
                MoeLoraRoute::Skip | MoeLoraRoute::Refuse => -1,
            },
            None => -1,
        })
        .collect()
}

#[cfg(test)]
#[path = "moe_row_adapter_tests.rs"]
mod tests;
