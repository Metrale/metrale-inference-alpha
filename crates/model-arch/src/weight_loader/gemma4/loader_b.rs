// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-25: Gemma-4 loader helpers: the per-layer MoE FFN, the V-norm
//! weight and the optional BF16 MLP, plus the embedding, final norm, LM head,
//! MTP and KV-dimension methods.
//!
//! Owner: model-arch weight loader.
//! Invariants: none beyond the types.

use anyhow::Result;
use metrale_config::ModelConfig;
use metrale_gpu_runtime::gpu::GpuBackend;
use metrale_model_weights::weights::WeightStore;

use metrale_model_layers::layers::FfnComponent;
use metrale_model_layers::weight_map::{DenseWeight, MtpWeights, QuantizeCtx, dense};

/// 2026-09-25: Build one layer's MoE FFN and the three norms of the dual-FFN
/// path; `None` when `num_experts == 0`.
pub(super) fn build_moe_ffn(
    store: &WeightStore,
    lp: &str,
    i: usize,
    config: &ModelConfig,
    gpu: &dyn GpuBackend,
    variant: metrale_model_layers::weight_map::Nvfp4Variant,
    qctx: QuantizeCtx,
    h: usize,
    absmax_k: metrale_gpu_runtime::gpu::KernelHandle,
    quantize_k: metrale_gpu_runtime::gpu::KernelHandle,
    // 2026-09-25: Whether to build the MoE prefill copies. The caller decides
    // once, before any layer loads, because `gpu.free_memory()` shrinks as
    // layers load: a per-layer probe would give the early layers the copies
    // and the late ones none.
    moe_prefill_copies: bool,
    stream: u64,
) -> Result<Option<(FfnComponent, DenseWeight, DenseWeight, DenseWeight)>> {
    if config.num_experts == 0 {
        return Ok(None);
    }
    use metrale_model_layers::weight_map::ssm_qwen35_more::load_moe_gemma4;
    tracing::info!(
        "L{i}: loading MoE ({} experts, top-{})...",
        config.num_experts,
        config.num_experts_per_tok
    );
    let moe_weights = load_moe_gemma4(store, lp, config.num_experts, gpu, config, variant, qctx)?;
    gpu.synchronize(stream)?;
    let gate_nvfp4 = metrale_model_layers::weight_map::quantize_to_nvfp4(
        &moe_weights.gate,
        config.num_experts,
        h,
        gpu,
        absmax_k,
        quantize_k,
        stream,
    )?;
    let mut moe_layer = metrale_model_layers::layers::MoeLayer::new(
        moe_weights,
        config.num_experts,
        Some(gate_nvfp4),
        gpu,
        config,
    )?;
    moe_layer.set_gelu_activation(gpu)?;
    if moe_prefill_copies {
        moe_layer.transpose_for_prefill(gpu, config)?;
        moe_layer.predequant_for_prefill(gpu, config, stream)?;
    }
    // 2026-09-25: The pre-expert norm applies after routing, to the experts'
    // input only.
    let pre_expert_norm = dense(store, &format!("{lp}.pre_feedforward_layernorm_2.weight"))?;
    moe_layer.set_pre_expert_norm(pre_expert_norm);
    gpu.synchronize(stream)?;
    tracing::info!("L{i}: MoE layer built (GeGLU activation, pre-expert norm)");

    let pre_moe_norm = dense(store, &format!("{lp}.pre_feedforward_layernorm_2.weight"))?;
    let post_moe_out_norm = dense(store, &format!("{lp}.post_feedforward_layernorm_2.weight"))?;
    let post_dense_ffn_norm = dense(store, &format!("{lp}.post_feedforward_layernorm_1.weight"))?;
    Ok(Some((
        FfnComponent::Moe(moe_layer),
        pre_moe_norm,
        post_moe_out_norm,
        post_dense_ffn_norm,
    )))
}

/// 2026-09-25: A BF16 buffer of `head_dim` ones, the V-norm weight.
pub(super) fn make_v_norm_ones_bf16(gpu: &dyn GpuBackend, head_dim: usize) -> Result<DenseWeight> {
    let bytes = head_dim * 2;
    let ptr = gpu.alloc(bytes)?;
    // 2026-09-25: BF16 1.0 is 0x3F80, stored little-endian as 0x80, 0x3F.
    let ones_host: Vec<u8> = std::iter::repeat_with(|| [0x80u8, 0x3Fu8])
        .take(head_dim)
        .flatten()
        .collect();
    gpu.copy_h2d(&ones_host, ptr)?;
    Ok(DenseWeight { weight: ptr })
}

/// 2026-09-25: BF16 copies of the MLP gate/up/down, built only when `bf16_mlp`
/// is set and the MLP is NVFP4 (it has `gate_proj.weight_scale`); else `None`.
pub(super) fn build_bf16_mlp(
    store: &WeightStore,
    lp: &str,
    bf16_mlp: bool,
    config: &ModelConfig,
    gpu: &dyn GpuBackend,
    h: usize,
) -> Result<Option<(DenseWeight, DenseWeight, DenseWeight)>> {
    let mlp_is_nvfp4 = store.contains(&format!("{lp}.mlp.gate_proj.weight_scale"));
    if !(bf16_mlp && mlp_is_nvfp4) {
        return Ok(None);
    }
    use metrale_model_layers::weight_map::fp8_lut::dequant_nvfp4_to_bf16;
    let gate_bf16 = dequant_nvfp4_to_bf16(
        store,
        &format!("{lp}.mlp.gate_proj"),
        config.intermediate_size,
        h,
        gpu,
    )?;
    let up_bf16 = dequant_nvfp4_to_bf16(
        store,
        &format!("{lp}.mlp.up_proj"),
        config.intermediate_size,
        h,
        gpu,
    )?;
    let down_bf16 = dequant_nvfp4_to_bf16(
        store,
        &format!("{lp}.mlp.down_proj"),
        h,
        config.intermediate_size,
        gpu,
    )?;
    Ok(Some((gate_bf16, up_bf16, down_bf16)))
}

pub(super) fn load_embedding_impl(
    store: &WeightStore,
    config: &ModelConfig,
) -> Result<DenseWeight> {
    let prefix = &config.weight_prefix;
    dense(store, &format!("{prefix}.embed_tokens.weight"))
}

pub(super) fn load_final_norm_impl(
    store: &WeightStore,
    config: &ModelConfig,
) -> Result<DenseWeight> {
    // 2026-09-25: Gemma-4's `rms_norm` kernel scales by `w`, not `1 + w`, so the
    // norm weights load unchanged.
    let prefix = &config.weight_prefix;
    dense(store, &format!("{prefix}.norm.weight"))
}

pub(super) fn load_lm_head_impl(store: &WeightStore, config: &ModelConfig) -> Result<DenseWeight> {
    // 2026-09-25: An `lm_head` tensor under any of these names wins; otherwise
    // the embedding is the head.
    for pattern in &[
        "lm_head.weight",
        "language_model.lm_head.weight",
        "model.lm_head.weight",
    ] {
        if store.contains(pattern) {
            return dense(store, pattern);
        }
    }
    load_embedding_impl(store, config)
}

pub(super) fn load_mtp_weights_impl(
    _store: &WeightStore,
    _config: &ModelConfig,
    _gpu: &dyn GpuBackend,
) -> Result<Option<MtpWeights>> {
    Ok(None)
}

pub(super) fn kv_layer_dims_impl(config: &ModelConfig) -> Vec<(usize, usize)> {
    let sliding_nkv = config.num_key_value_heads;
    let sliding_hd = 256;
    let full_nkv = 4;
    let full_hd = 512;
    let mut dims = Vec::with_capacity(config.num_hidden_layers);
    for i in 0..config.num_hidden_layers {
        if (i + 1) % 6 == 0 {
            dims.push((full_nkv, full_hd));
        } else {
            dims.push((sliding_nkv, sliding_hd));
        }
    }
    dims
}
