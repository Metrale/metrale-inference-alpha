// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-28: Run the attention and GDN projections of a native FP8 Qwen3.5 checkpoint (HF
//! `fp8`: E4M3 weights in 128x128 blocks, activations declared dynamic per token and 128-wide
//! group) W8A8 at decode, on the block-scaled FP8 weights the loader already holds
//! (`adopt_fp8_block_w8a8`). The routed and shared experts are not touched here: their W8A8
//! decode is the MoE layer's own (`set_moe_expert_fp8_act`).
//!
//! Owner: model-arch weight loader (Qwen3.5).
//! Invariants: at TP > 1, or when the W8A8 kernels are not compiled into the target, nothing is
//! adopted; every adopted projection of the model shares one `W8a8Ctx`.

use anyhow::Result;
use metrale_config::ModelConfig;
use metrale_gpu_runtime::gpu::GpuBackend;
use metrale_model_layers::layer::TransformerLayer;
use metrale_model_layers::layers::W8a8Ctx;
use metrale_model_layers::layers::ops::W8a8Scale;
use metrale_model_layers::layers::qwen3_attention::Qwen3AttentionLayer;
use metrale_model_layers::layers::qwen3_ssm::Qwen3SsmLayer;

/// 2026-09-28: Adopt W8A8 on every layer that holds block-scaled FP8 attention or GDN weights and
/// whose first projection the declared-precision policy wants run W8A8
/// (`WeightQuantPolicy::fp8_block_scaled_decode_act(module) == Some(Fp8)`). While
/// `kernel_caps().w8a8_block_scaled_decode` is off and the experts decode W8A8, nothing is
/// adopted, and the log says once that the declared FP8 activations run W8A16. Returns `(attention layers, GDN layers)`
/// adopted.
pub(super) fn adopt_declared(
    layers: &mut [Box<dyn TransformerLayer>],
    config: &ModelConfig,
    gpu: &dyn GpuBackend,
) -> Result<(usize, usize)> {
    // 2026-10-02: The block-scaled W8A8 cap is off because, stacked with the expert W8A8 on
    // Qwen3.6-35B-A3B-FP8, it flipped a greedy tie in the ssm-state-poisoning gate; either family
    // alone passed (`layers::kernel_caps`). A model whose experts do not decode W8A8 (the NVFP4
    // checkpoint's W4A16 experts) has no such stack, so its declared FP8 attention and GDN
    // activations run W8A8 as declared.
    let mut caps = metrale_model_layers::layers::kernel_caps();
    caps.w8a8_block_scaled_decode |= !metrale_model_layers::layers::moe_expert_fp8_act();
    let policy = metrale_config::WeightQuantPolicy::for_checkpoint(
        metrale_model_layers::layers::weight_quantization(),
        config.quantization_config.as_ref(),
        caps,
    );
    if !policy.follows_plan() || config.tp_world_size > 1 {
        return Ok((0, 0));
    }
    let wants = |m: String| {
        policy.fp8_block_scaled_decode_act(&m)
            == Some(metrale_config::weight_quantization::ActFormat::Fp8)
    };
    let modules = |i: usize| {
        let lp = config.layer_prefix(i);
        [
            format!("{lp}.self_attn.q_proj"),
            format!("{lp}.linear_attn.in_proj_qkv"),
        ]
    };
    let any_wanted = (0..config.num_hidden_layers).any(|i| modules(i).into_iter().any(&wants));
    if !any_wanted {
        if (0..config.num_hidden_layers).any(|i| {
            modules(i)
                .iter()
                .any(|m| policy.declares_fp8_activations(m))
        }) {
            tracing::info!(
                "--weight-quantization declared: the checkpoint declares FP8 activations for \
                 its block-scaled attention and GDN projections; they decode W8A16 (above \
                 declared) until the block-scaled W8A8 path is re-validated (its cap is off: \
                 with it and the expert W8A8 both on, the ssm-state-poisoning gate failed on \
                 a greedy tie, 2026-09-29)"
            );
        }
        return Ok((0, 0));
    }
    let q_dim = (config.num_attention_heads * config.head_dim) as u32;
    let value_dim = (config.linear_num_value_heads * config.linear_value_head_dim) as u32;
    let h = config.hidden_size as u32;
    let ctx = W8a8Ctx::new(gpu, h.max(q_dim).max(value_dim))?;
    if !ctx.kernels.resolved(W8a8Scale::Block128) {
        return Ok((0, 0));
    }
    let (mut attn, mut gdn) = (0, 0);
    for (i, layer) in layers.iter_mut().enumerate() {
        let lp = config.layer_prefix(i);
        let Some(any) = layer.as_any_mut() else {
            continue;
        };
        if let Some(l) = any.downcast_mut::<Qwen3AttentionLayer>()
            && wants(format!("{lp}.self_attn.q_proj"))
        {
            attn += usize::from(l.adopt_fp8_block_w8a8(ctx, h)?);
        } else if let Some(l) = any.downcast_mut::<Qwen3SsmLayer>()
            && wants(format!("{lp}.linear_attn.in_proj_qkv"))
        {
            gdn += usize::from(l.adopt_fp8_block_w8a8(ctx, h)?);
        }
    }
    tracing::info!(
        "W8A8 decode (declared FP8 W8A8, 128x128 blocks): {attn} attention and {gdn} GDN layers"
    );
    Ok((attn, gdn))
}
