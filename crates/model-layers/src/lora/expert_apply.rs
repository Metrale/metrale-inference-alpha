// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-25: Host-driven MoE LoRA folds built on `apply_lora_delta`: the
//! router delta onto the routing logits (`apply_router_lora`), and per-expert
//! deltas onto the expert-sorted grouped-GEMM output
//! (`apply_expert_lora_sorted`), whose row blocks come from a host copy of
//! `expert_offsets`. Their only caller in the tree is model-engine's
//! `tests/moe_lora_delta_parity.rs`; the MoE layer folds through
//! `layers/moe/lora.rs` and `ops::moe_lora_grouped`.
//!
//! Owner: model-layers (lora).
//! Invariants:
//! - Each `apply_lora_delta` call gets at most `max(max_rows, 1)` rows.

use anyhow::Result;
use metrale_gpu_runtime::gpu::{DevicePtr, GpuBackend};

use super::{ExpertLoraLayer, ExpertProj};
use crate::layers::ops::lora_delta::{LoraKernels, LoraPair, apply_lora_delta};

const BF16_BYTES: u64 = 2;

/// 2026-09-25: One expert's contiguous block of sorted rows. The planner,
/// `expert_delta_workitems`, emits none with `rows == 0`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ExpertWork {
    pub expert: u16,
    pub row_off: u32,
    pub rows: u32,
}

/// 2026-09-25: Work items for the adapted experts, from the `expert_offsets`
/// prefix sum (`[num_experts + 1]`; expert `e`'s sorted rows are
/// `expert_offsets[e]..expert_offsets[e + 1]`). An expert index past the
/// table, or with `end <= start`, is skipped.
pub fn expert_delta_workitems(expert_offsets: &[u32], adapted: &[u16]) -> Vec<ExpertWork> {
    let n_experts = expert_offsets.len().saturating_sub(1);
    let mut work = Vec::with_capacity(adapted.len());
    for &e in adapted {
        let e_us = e as usize;
        if e_us >= n_experts {
            continue;
        }
        let start = expert_offsets[e_us];
        let end = expert_offsets[e_us + 1];
        if end <= start {
            continue;
        }
        work.push(ExpertWork {
            expert: e,
            row_off: start,
            rows: end - start,
        });
    }
    work
}

/// 2026-09-25: Fold one projection's delta over `rows` contiguous rows, one
/// `apply_lora_delta` call per block of at most `max_rows` rows, so scratch of
/// `max_rows * max_rank` (`lora_xa`) and `max_rows * n_out` (`lora_delta`)
/// BF16 suffices.
#[allow(clippy::too_many_arguments)]
fn fold_chunked(
    gpu: &dyn GpuBackend,
    kernels: &LoraKernels,
    pair: &LoraPair,
    x: DevicePtr,
    base_out: DevicePtr,
    rows: u32,
    max_rows: u32,
    lora_xa: DevicePtr,
    lora_delta: DevicePtr,
    stream: u64,
) -> Result<()> {
    let step = max_rows.max(1);
    let mut done = 0u32;
    while done < rows {
        let m = (rows - done).min(step);
        let x_row = x.offset((done as u64 * pair.k_in as u64 * BF16_BYTES) as usize);
        let out_row = base_out.offset((done as u64 * pair.n_out as u64 * BF16_BYTES) as usize);
        apply_lora_delta(
            gpu, kernels, pair, x_row, out_row, m, lora_xa, lora_delta, stream,
        )?;
        done += m;
    }
    Ok(())
}

/// 2026-09-25: Fold the router (`mlp.gate`) delta onto `gate_logits`
/// (`[n, num_experts]`, in place) from `router_in` (`[n, hidden]`), in blocks
/// of at most `max_rows` rows.
#[allow(clippy::too_many_arguments)]
pub fn apply_router_lora(
    gpu: &dyn GpuBackend,
    kernels: &LoraKernels,
    pair: &LoraPair,
    router_in: DevicePtr,
    gate_logits: DevicePtr,
    n: u32,
    max_rows: u32,
    lora_xa: DevicePtr,
    lora_delta: DevicePtr,
    stream: u64,
) -> Result<()> {
    fold_chunked(
        gpu,
        kernels,
        pair,
        router_in,
        gate_logits,
        n,
        max_rows,
        lora_xa,
        lora_delta,
        stream,
    )
}

/// 2026-09-25: Fold `proj`'s per-expert deltas onto the expert-sorted
/// grouped-GEMM output. `x` is the sorted input (`[total_expanded, k_in]`),
/// `base_out` the sorted output (`[total_expanded, n_out]`, in place), and
/// `expert_offsets_host` a host copy of `expert_offsets`. Only experts with a
/// pair for `proj` and at least one row are folded, each over its own row
/// block.
#[allow(clippy::too_many_arguments)]
pub fn apply_expert_lora_sorted(
    gpu: &dyn GpuBackend,
    kernels: &LoraKernels,
    layer: &ExpertLoraLayer,
    proj: ExpertProj,
    expert_offsets_host: &[u32],
    x: DevicePtr,
    base_out: DevicePtr,
    max_rows: u32,
    lora_xa: DevicePtr,
    lora_delta: DevicePtr,
    stream: u64,
) -> Result<()> {
    let work = expert_delta_workitems(expert_offsets_host, &layer.adapted_experts());
    for w in work {
        let Some(pair) = layer.pair(w.expert, proj) else {
            continue;
        };
        let x_row = x.offset((w.row_off as u64 * pair.k_in as u64 * BF16_BYTES) as usize);
        let out_row = base_out.offset((w.row_off as u64 * pair.n_out as u64 * BF16_BYTES) as usize);
        fold_chunked(
            gpu, kernels, pair, x_row, out_row, w.rows, max_rows, lora_xa, lora_delta, stream,
        )?;
    }
    Ok(())
}

#[cfg(test)]
#[path = "expert_apply_tests.rs"]
mod tests;
