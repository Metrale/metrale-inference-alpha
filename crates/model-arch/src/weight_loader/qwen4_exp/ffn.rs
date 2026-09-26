// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-25: The per-layer MoE FFN of `qwen4_exp`: the weights from
//! `load_moe_qwen35`, an NVFP4 copy of the router for `MoeLayer::new`, and,
//! with `METRALE_HOLO_MOE_GROUPED_CUTLASS=1`, the CUTLASS grouped scale
//! tables.
//!
//! Owner: model-arch weight loader.
//! Invariants: none beyond the types.

use anyhow::{Context, Result};
use metrale_config::ModelConfig;
use metrale_gpu_runtime::gpu::GpuBackend;
use metrale_model_weights::weights::WeightStore;

use metrale_model_layers::layers::{FfnComponent, MoeLayer};
use metrale_model_layers::weight_map::{Nvfp4Variant, load_moe_qwen35, quantize_to_nvfp4};

pub(super) fn build_moe(
    store: &WeightStore,
    lp: &str,
    config: &ModelConfig,
    gpu: &dyn GpuBackend,
    variant: Nvfp4Variant,
) -> Result<FfnComponent> {
    let h = config.hidden_size;
    let absmax_k = gpu.kernel("quantize_nvfp4", "nvfp4_global_absmax")?;
    let quantize_k = gpu.kernel("quantize_nvfp4", "quantize_bf16_to_nvfp4")?;
    let stream = gpu.default_stream();

    let weights = load_moe_qwen35(
        store,
        lp,
        config.num_experts,
        gpu,
        config,
        variant,
        absmax_k,
        quantize_k,
        stream,
        false,
    )
    .with_context(|| format!("qwen4_exp: MoE block at {lp}"))?;

    let gate_nvfp4 = Some(quantize_to_nvfp4(
        &weights.gate,
        config.num_experts,
        h,
        gpu,
        absmax_k,
        quantize_k,
        stream,
    )?);

    let mut moe = MoeLayer::new(weights, config.num_experts, gate_nvfp4, gpu, config)?;

    // 2026-09-25: The grouped CUTLASS prefill arm runs only when
    // `cutlass_grouped_host` is set (`moe/forward_prefill_routed.rs`), and
    // `build_cutlass_grouped_sfb` is what sets it. Without transposed scales it
    // reads the original N-major ones.
    if std::env::var("METRALE_HOLO_MOE_GROUPED_CUTLASS")
        .ok()
        .as_deref()
        == Some("1")
    {
        moe.build_cutlass_grouped_sfb(gpu, config, stream)?;
    }

    Ok(FfnComponent::Moe(moe))
}
