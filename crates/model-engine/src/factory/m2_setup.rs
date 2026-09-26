// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-25: The load-time MoE transpose pass for `minimax_m2`, `step3p7` and `deepseek_v4`:
//! hybrid, unified, full, gate+up-only or none, chosen from the layout levers and free memory.
//!
//! Owner: model-engine factory.
//! Invariants: none beyond the types.

use anyhow::Result;
use metrale_config::ModelConfig;
use metrale_gpu_runtime::gpu::GpuBackend;

use metrale_model_layers::layer::TransformerLayer;

/// 2026-09-25: Run the MoE transpose pass for `minimax_m2`, `step3p7` and `deepseek_v4`, and
/// return `Ok(())` without changes for every other model type.
pub(super) fn maybe_run_minimax_m2_moe_transpose(
    config: &ModelConfig,
    gpu: &dyn GpuBackend,
    layers: &mut [Box<dyn TransformerLayer>],
) -> Result<()> {
    // 2026-09-25: `deepseek_v4` is included because the prefill GEMMs for E8M0 (native MXFP4)
    // routed experts exist only on the transposed tables; the non-transposed fallback panics on
    // E8M0 weights (`experts_scale_kind.expect(Nvfp4)` in `moe/forward_prefill_routed.rs`).
    if config.model_type != "minimax_m2"
        && config.model_type != "step3p7"
        && config.model_type != "deepseek_v4"
    {
        return Ok(());
    }
    let unified_layout = std::env::var("METRALE_UNIFIED_MOE_LAYOUT")
        .map(|v| v == "1" || v.eq_ignore_ascii_case("true"))
        .unwrap_or(false);
    let hybrid_layout = std::env::var("METRALE_HYBRID_MOE_LAYOUT")
        .map(|v| v == "1" || v.eq_ignore_ascii_case("true"))
        .unwrap_or(false);
    let local_experts: usize = (0..config.num_experts)
        .filter(|e| config.is_local_expert(*e))
        .count();
    // 2026-09-25: One projection of one expert, sized as NVFP4: packed `n*k/2` bytes plus scales
    // `n*k/16` bytes, i.e. `n*k * 9/16` with `n*k = inter * hidden`.
    let per_expert_one: usize = config.moe_intermediate_size * config.hidden_size * 9 / 16;
    let cost_full: usize = local_experts * 3 * per_expert_one * config.num_hidden_layers;
    let cost_gate_up: usize = local_experts * 2 * per_expert_one * config.num_hidden_layers;
    let safety: usize = std::env::var("METRALE_MOE_TRANSPOSE_SAFETY_MB")
        .ok()
        .and_then(|v| v.parse::<usize>().ok())
        .map(|mb| mb * 1024 * 1024)
        .unwrap_or(2 * 1024 * 1024 * 1024);
    let free = gpu.free_memory()?;
    let gb = |b: usize| b as f64 / (1024.0 * 1024.0 * 1024.0);
    // 2026-09-25: Hybrid keeps the originals and adds transposed copies, so it needs
    // `2 * cost_full`. When it does not fit, the next matching branch runs, which is the unified
    // layout only if `METRALE_UNIFIED_MOE_LAYOUT` is also set.
    let hybrid_fits = hybrid_layout && free >= 2 * cost_full + safety;
    if hybrid_layout && !hybrid_fits {
        tracing::warn!(
            "MoE transpose pass (hybrid layout): METRALE_HYBRID_MOE_LAYOUT=1 \
             requested but doesn't fit (need {:.1} GB, free {:.1} GB) — \
             falling back to unified-layout (decode regression).",
            gb(2 * cost_full + safety),
            gb(free),
        );
    }
    if hybrid_fits {
        // 2026-09-25: Decode keeps reading the originals (`MoeLayer::use_t_layout_for_decode` is
        // false under the hybrid layout); prefill reads the transposed copies.
        tracing::info!(
            "MoE transpose pass (hybrid layout): METRALE_HYBRID_MOE_LAYOUT=1, \
             dual-layout (cost {:.1} GB), free pre-pass {:.1} GB → RUNNING",
            gb(2 * cost_full),
            gb(free),
        );
        for layer in layers.iter_mut() {
            layer.transpose_moe_for_prefill_hybrid(gpu, config)?;
        }
        tracing::info!(
            "MoE transpose pass (hybrid layout): done, {:.1} GB free",
            gb(gpu.free_memory()?)
        );
    } else if unified_layout {
        // 2026-09-25: Phased transpose that frees the originals between phases; no free-memory
        // check is made for it.
        tracing::info!(
            "MoE transpose pass (unified layout): METRALE_UNIFIED_MOE_LAYOUT=1, \
             phased transpose with frees, free pre-pass {:.1} GB → RUNNING",
            gb(free),
        );
        for layer in layers.iter_mut() {
            layer.transpose_moe_for_prefill_unified(gpu, config)?;
        }
        tracing::info!(
            "MoE transpose pass (unified layout): done, {:.1} GB free",
            gb(gpu.free_memory()?)
        );
    } else if free >= cost_full + safety {
        tracing::info!(
            "MoE transpose pass: cost {:.1} GB (full gate+up+down), \
             free {:.1} GB → RUNNING",
            gb(cost_full),
            gb(free),
        );
        for layer in layers.iter_mut() {
            layer.transpose_moe_for_prefill(gpu, config)?;
        }
        tracing::info!(
            "MoE transpose pass: done, {:.1} GB free",
            gb(gpu.free_memory()?)
        );
    } else if free >= cost_gate_up + safety {
        tracing::info!(
            "MoE transpose pass: full cost {:.1} GB > free {:.1} GB; \
             falling back to gate+up only (cost {:.1} GB) → RUNNING",
            gb(cost_full),
            gb(free),
            gb(cost_gate_up),
        );
        for layer in layers.iter_mut() {
            layer.transpose_moe_gate_up_for_prefill(gpu, config)?;
        }
        tracing::info!(
            "MoE transpose pass: gate+up done, {:.1} GB free \
             (down: per-prefill scratch transpose)",
            gb(gpu.free_memory()?),
        );

        // 2026-09-25: down_proj is not transposed persistently. One scratch, sized for one
        // layer's local experts, is shared by every MoE layer; each layer's prefill fills it from
        // its untransposed `down_ptrs` (`MoeLayer::transpose_down_into_scratch`). Decode keeps
        // reading `down_ptrs`.
        let n_per_expert_packed: usize = config.hidden_size * config.moe_intermediate_size / 2;
        let n_per_expert_scale: usize = config.hidden_size * config.moe_intermediate_size / 16;
        let local_experts_count: usize = (0..config.num_experts)
            .filter(|e| config.is_local_expert(*e))
            .count();
        let scratch_packed_bytes = local_experts_count * n_per_expert_packed;
        let scratch_scale_bytes = local_experts_count * n_per_expert_scale;
        let scratch_packed = gpu.alloc(scratch_packed_bytes)?;
        let scratch_scale = gpu.alloc(scratch_scale_bytes)?;

        let mut packed_ptrs_host = Vec::<u8>::with_capacity(config.num_experts * 8);
        let mut scale_ptrs_host = Vec::<u8>::with_capacity(config.num_experts * 8);
        let mut local_idx = 0usize;
        for e in 0..config.num_experts {
            let (p_ptr, s_ptr) = if config.is_local_expert(e) {
                let p = scratch_packed.0 + (local_idx * n_per_expert_packed) as u64;
                let s = scratch_scale.0 + (local_idx * n_per_expert_scale) as u64;
                local_idx += 1;
                (p, s)
            } else {
                (0u64, 0u64)
            };
            packed_ptrs_host.extend_from_slice(&p_ptr.to_le_bytes());
            scale_ptrs_host.extend_from_slice(&s_ptr.to_le_bytes());
        }
        let packed_ptrs_t = gpu.alloc(config.num_experts * 8)?;
        gpu.copy_h2d(&packed_ptrs_host, packed_ptrs_t)?;
        let scale_ptrs_t = gpu.alloc(config.num_experts * 8)?;
        gpu.copy_h2d(&scale_ptrs_host, scale_ptrs_t)?;
        for layer in layers.iter_mut() {
            layer.set_moe_down_transpose_scratch(
                scratch_packed,
                scratch_scale,
                packed_ptrs_t,
                scale_ptrs_t,
            );
        }
        tracing::info!(
            "MoE down scratch: {:.0} MB packed + {:.0} MB scale, shared across {} layers",
            scratch_packed_bytes as f64 / (1024.0 * 1024.0),
            scratch_scale_bytes as f64 / (1024.0 * 1024.0),
            config.num_hidden_layers,
        );
    } else {
        tracing::warn!(
            "MoE transpose pass: cost {:.1} GB (full) / {:.1} GB (gate+up), \
             free {:.1} GB → SKIP (insufficient memory; prefill uses \
             uncoalesced fallback, TTFT will be ~2× slower)",
            gb(cost_full),
            gb(cost_gate_up),
            gb(free),
        );
    }
    Ok(())
}
