// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-26: The MoE steps of `load_layers`: the free-memory check that decides whether the
//! MoE prefill tables are transposed, the FP8-to-BF16 expert dequant, and the native-FP8
//! expert install.
//!
//! Owner: model-arch weight loader (Qwen3.5).
//! Invariants: none beyond the types.

use metrale_config::ModelConfig;
use metrale_gpu_runtime::gpu::GpuBackend;
use metrale_model_layers::layers::MoeLayer;
use metrale_model_layers::weight_map::{
    load_fp8_block_scaled_as_fp8weight, load_moe_qwen35_fp8_experts,
};

use super::load_cx::LoadCx;

/// 2026-09-26: Whether the transposed MoE prefill tables exceed free memory less 2 GiB;
/// warns when they do.
pub(super) fn moe_transpose_skipped(config: &ModelConfig, gpu: &dyn GpuBackend, h: usize) -> bool {
    let inter = config.moe_intermediate_size;
    let group_size = 16usize;
    let gu_bytes = inter * h / 2 + inter * h / group_size;
    let d_bytes = h * inter / 2 + h * inter / group_size;
    let per_layer = config.num_experts * (2 * gu_bytes + d_bytes);
    let total = per_layer * config.num_hidden_layers;
    let available = gpu.free_memory().unwrap_or(0);
    let headroom = 2 * 1024 * 1024 * 1024;
    let skip = total > available.saturating_sub(headroom);
    if skip {
        tracing::warn!(
            target: "metrale_model_arch::weight_loader::qwen35::load_layers",
            "Skipping MoE weight transposition ({:.1} GB needed, {:.1} GB available). \
             Prefill will use fallback grouped GEMM.",
            total as f64 / (1024.0 * 1024.0 * 1024.0),
            available as f64 / (1024.0 * 1024.0 * 1024.0),
        );
    }
    skip
}

/// 2026-09-26: `METRALE_FP8_DEQUANT_MOE_TO_BF16`: dequantizes layer `i`'s FP8 experts to BF16 and
/// installs them on `moe_layer`. A failure is logged, and the layer keeps its native-FP8 MoE.
pub(super) fn install_bf16_dequant_experts(
    cx: &LoadCx,
    lp: &str,
    i: usize,
    moe_layer: &mut MoeLayer,
) {
    let LoadCx {
        store, config, gpu, ..
    } = *cx;
    use metrale_model_layers::weight_map::quant_helpers::dequant_fp8_blockscaled_to_bf16;
    let p = format!("{lp}.mlp");
    let mut gate_bf16 = Vec::with_capacity(config.num_experts);
    let mut up_bf16 = Vec::with_capacity(config.num_experts);
    let mut down_bf16 = Vec::with_capacity(config.num_experts);
    let mut load_err: Option<anyhow::Error> = None;
    // 2026-09-25: Frees each expert's FP8 source once its dequant succeeds. The store
    // keeps the freed pointers; nothing below reads them, because the native-FP8 expert
    // load is skipped whenever `dequant_moe_to_bf16` is set.
    let free_src = |prefix: &str| {
        for suffix in ["weight", "weight_scale_inv"] {
            let k = format!("{prefix}.{suffix}");
            if let Ok(w) = store.get(&k) {
                let _ = gpu.free(w.ptr);
            }
        }
    };
    for e in 0..config.num_experts {
        let ep = format!("{p}.experts.{e}");
        let gate_key = format!("{ep}.gate_proj");
        let up_key = format!("{ep}.up_proj");
        let down_key = format!("{ep}.down_proj");
        let g = dequant_fp8_blockscaled_to_bf16(store, &gate_key, gpu);
        let u = dequant_fp8_blockscaled_to_bf16(store, &up_key, gpu);
        let d = dequant_fp8_blockscaled_to_bf16(store, &down_key, gpu);
        match (g, u, d) {
            (Ok(g), Ok(u), Ok(d)) => {
                gate_bf16.push(g);
                up_bf16.push(u);
                down_bf16.push(d);
                free_src(&gate_key);
                free_src(&up_key);
                free_src(&down_key);
            }
            (g, u, d) => {
                load_err = Some(anyhow::anyhow!(
                    "Layer {i} expert {e}: BF16 dequant failed (gate_ok={}, up_ok={}, down_ok={})",
                    g.is_ok(),
                    u.is_ok(),
                    d.is_ok(),
                ));
                break;
            }
        }
    }
    // 2026-09-25: The shared expert is optional: a tensor that fails to dequantize becomes
    // NULL.
    let sp = format!("{p}.shared_expert");
    let sh_gate_key = format!("{sp}.gate_proj");
    let sh_up_key = format!("{sp}.up_proj");
    let sh_down_key = format!("{sp}.down_proj");
    let sh_g = dequant_fp8_blockscaled_to_bf16(store, &sh_gate_key, gpu).ok();
    let sh_u = dequant_fp8_blockscaled_to_bf16(store, &sh_up_key, gpu).ok();
    let sh_d = dequant_fp8_blockscaled_to_bf16(store, &sh_down_key, gpu).ok();
    if sh_g.is_some() {
        free_src(&sh_gate_key);
    }
    if sh_u.is_some() {
        free_src(&sh_up_key);
    }
    if sh_d.is_some() {
        free_src(&sh_down_key);
    }
    let sh_g_ptr = sh_g
        .map(|w| w.weight)
        .unwrap_or(metrale_gpu_runtime::gpu::DevicePtr::NULL);
    let sh_u_ptr = sh_u
        .map(|w| w.weight)
        .unwrap_or(metrale_gpu_runtime::gpu::DevicePtr::NULL);
    let sh_d_ptr = sh_d
        .map(|w| w.weight)
        .unwrap_or(metrale_gpu_runtime::gpu::DevicePtr::NULL);
    match load_err {
        Some(e) => {
            tracing::error!(
                target: "metrale_model_arch::weight_loader::qwen35::load_layers",
                "Layer {i}: dequant-to-BF16 MoE load failed: {e:#}"
            );
            tracing::warn!(
                target: "metrale_model_arch::weight_loader::qwen35::load_layers",
                "Layer {i}: falling back to native FP8 MoE"
            );
        }
        None => {
            if let Err(e) = moe_layer.set_bf16_experts(
                &gate_bf16, &up_bf16, &down_bf16, sh_g_ptr, sh_u_ptr, sh_d_ptr, gpu,
            ) {
                tracing::error!(
                    target: "metrale_model_arch::weight_loader::qwen35::load_layers",
                    "Layer {i}: failed to build BF16 expert pointer tables: {e:#}"
                );
            } else {
                tracing::info!(
                    target: "metrale_model_arch::weight_loader::qwen35::load_layers",
                    "Layer {i}: MoE experts dequanted FP8→BF16 ({} routed + 1 shared)",
                    config.num_experts
                );
            }
        }
    }
}

/// 2026-09-26: Loads layer `i`'s native-FP8 routed and shared experts and installs them on
/// `moe_layer`. A failure is logged.
pub(super) fn install_native_fp8_experts(
    cx: &LoadCx,
    lp: &str,
    i: usize,
    moe_layer: &mut MoeLayer,
) {
    let LoadCx {
        store, config, gpu, ..
    } = *cx;
    let fp8_experts = match load_moe_qwen35_fp8_experts(store, lp, config.num_experts, gpu, config)
    {
        Ok(e) => Some(e),
        Err(e) => {
            tracing::error!(
                target: "metrale_model_arch::weight_loader::qwen35::load_layers",
                "Layer {i}: native-FP8 expert load failed: {e:#} — routed experts \
                 would be NULL (incoherent output). MoE left on its fallback path."
            );
            None
        }
    };
    if let Some(fp8_experts) = fp8_experts {
        let sp = format!("{lp}.mlp.shared_expert");
        use metrale_gpu_runtime::gpu::DevicePtr;
        use metrale_model_layers::weight_map::{Fp8ExpertWeight as FEW, Fp8Weight as FW};
        let null_fw = FW {
            weight: DevicePtr::NULL,
            row_scale: DevicePtr::NULL,
            n: 0,
            k: 0,
            // 2026-09-25: Stands in for a shared-expert tensor that failed to load, tagged
            // like the block-scaled loader's output.
            scale_format: metrale_model_layers::weight_map::WeightQuantFormat::Fp8BlockScaled,
        };
        let sh_gate = load_fp8_block_scaled_as_fp8weight(store, &format!("{sp}.gate_proj"), gpu);
        let sh_up = load_fp8_block_scaled_as_fp8weight(store, &format!("{sp}.up_proj"), gpu);
        let sh_down = load_fp8_block_scaled_as_fp8weight(store, &format!("{sp}.down_proj"), gpu);
        if sh_gate.is_err() || sh_up.is_err() || sh_down.is_err() {
            tracing::warn!(
                target: "metrale_model_arch::weight_loader::qwen35::load_layers",
                "Layer {i}: shared expert FP8 load failed (gate={}, up={}, down={})",
                sh_gate.is_ok(),
                sh_up.is_ok(),
                sh_down.is_ok(),
            );
        }
        let shared_fp8 = FEW {
            gate_proj: sh_gate.unwrap_or(null_fw),
            up_proj: sh_up.unwrap_or(null_fw),
            down_proj: sh_down.unwrap_or(null_fw),
        };
        if let Err(e) = moe_layer.set_fp8_experts(&fp8_experts, shared_fp8, gpu) {
            tracing::error!(
                target: "metrale_model_arch::weight_loader::qwen35::load_layers",
                "Layer {i}: failed to build FP8 expert pointer tables: {e:#}"
            );
            tracing::warn!(
                target: "metrale_model_arch::weight_loader::qwen35::load_layers",
                "Layer {i}: falling back to NVFP4-only decode for MoE experts"
            );
        } else {
            tracing::info!(
                target: "metrale_model_arch::weight_loader::qwen35::load_layers",
                "Layer {i}: MoE experts loaded as native FP8"
            );
        }
    }
}
