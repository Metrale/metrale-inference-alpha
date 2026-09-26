// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-25: Last load step: builds `MlaWeights` from the context, the null GQA `AttentionWeights`, the MoE FFN, and the layer.
//!
//! Owner: model-arch (Mistral loader).
//! Invariants:
//! - A required MLA tensor missing from the context is an error naming it, not a panic.
//! - A MoE load or construction failure is logged and the layer gets `FfnComponent::None`.

use anyhow::{Result, anyhow};
use metrale_cache::kv_cache::KvCacheDtype;
use metrale_gpu_runtime::gpu::{DevicePtr, GpuBackend};

use super::ctx::MistralLayerCtx;
use metrale_model_layers::layer::TransformerLayer;
use metrale_model_layers::layers::qwen3_attention::MlaWeights;
use metrale_model_layers::layers::{FfnComponent, MoeLayer, Qwen3AttentionLayer};
use metrale_model_layers::weight_map::loaders_moe::load_moe_mistral;
use metrale_model_layers::weight_map::{AttentionWeights, DenseWeight, QuantizedWeight, dense};

pub(super) fn assemble_layer(
    ctx: MistralLayerCtx<'_>,
    yarn_inv_freq: DevicePtr,
    layer_kv_dtypes: &[KvCacheDtype],
) -> Result<Box<dyn TransformerLayer>> {
    let i = ctx.layer_idx;
    let prefix = format!("layers.{i}");
    let gpu = ctx.gpu;
    let config = ctx.config;
    let q_lora = ctx.q_lora;
    let kv_lora = ctx.kv_lora;
    let nope = ctx.nope;
    let rope = ctx.rope;
    let v_dim = ctx.v_dim;

    // 2026-09-25: Every GQA projection is null; the layer is given
    // `mla_weights` below.
    let null = DenseWeight {
        weight: metrale_gpu_runtime::gpu::DevicePtr::NULL,
    };
    let o_dummy_quant = QuantizedWeight {
        weight: metrale_gpu_runtime::gpu::DevicePtr::NULL,
        weight_scale: metrale_gpu_runtime::gpu::DevicePtr::NULL,
        weight_scale_2: 0.0,
        input_scale: metrale_gpu_runtime::gpu::DevicePtr::NULL,
        weight_scale_2_vec: metrale_gpu_runtime::gpu::DevicePtr::NULL,
    };
    let attn = AttentionWeights {
        q_proj: null,
        k_proj: null,
        v_proj: null,
        o_proj: o_dummy_quant,
        q_norm: null,
        k_norm: null,
        q_norm_full: None,
        k_norm_full: None,
        k_scale: 1.0,
        v_scale: 1.0,
    };

    // 2026-09-25: The NVFP4 copies of wq_a, wq_b, wkv_a and wo are used
    // unless METRALE_NVFP4_MLA is 0, false, no or off (trimmed, any case);
    // then the layer gets the BF16 weights only.
    let disable_nvfp4_mla = std::env::var("METRALE_NVFP4_MLA")
        .map(|v| {
            let v = v.trim().to_ascii_lowercase();
            matches!(v.as_str(), "0" | "false" | "no" | "off")
        })
        .unwrap_or(false);

    // 2026-09-25: A field an earlier step left unset is an error naming the
    // field, not a panic.
    let require = |v: Option<DenseWeight>, name: &'static str| -> Result<DenseWeight> {
        v.ok_or_else(|| anyhow!("Mistral L{i} phase_assemble: missing required tensor `{name}`"))
    };

    let wq_a_dense = require(ctx.wq_a_dense, "wq_a_dense")?;
    let wkv_a_dense = require(ctx.wkv_a_dense, "wkv_a_dense")?;
    let mla_weights = MlaWeights {
        wq_a: wq_a_dense,
        wq_a_fp8: None,
        wq_a_nvfp4: if disable_nvfp4_mla {
            None
        } else {
            ctx.wq_a_nvfp4
        },
        wq_b: require(ctx.wq_b, "wq_b")?,
        wq_b_fp8: None,
        wq_b_nvfp4: if disable_nvfp4_mla {
            None
        } else {
            ctx.wq_b_nvfp4
        },
        q_a_norm: require(ctx.q_a_norm, "q_a_norm")?,
        wkv_a: wkv_a_dense,
        wkv_a_nvfp4: if disable_nvfp4_mla {
            None
        } else {
            ctx.wkv_a_nvfp4
        },
        wkv_b: require(ctx.wkv_b, "wkv_b")?,
        kv_a_norm: require(ctx.kv_a_norm, "kv_a_norm")?,
        wkv_a_rope: require(ctx.wkv_a_rope_dense, "wkv_a_rope_dense")?,
        wkv_a_merged: DenseWeight {
            weight: wkv_a_dense.weight,
        },
        wo: require(ctx.o_dense_bf16, "o_dense_bf16")?,
        wo_nvfp4: if disable_nvfp4_mla { None } else { ctx.o_nvfp4 },
        wo_a: null,
        wo_a_nvfp4: None,
        wo_b: null,
        wo_b_nvfp4: None,
        wo_b_fp8: None,
        wo_a_fp8: None,
        wkv_a_fp8: None,
        wq_b_rope: require(ctx.wq_b_rope, "wq_b_rope")?,
        w_uk_t: require(ctx.w_uk_t, "w_uk_t")?,
        w_uv: require(ctx.w_uv, "w_uv")?,
        w_qk_absorbed: require(ctx.w_qk_absorbed, "w_qk_absorbed")?,
        w_uk_block_diag: require(ctx.w_uk_block_diag, "w_uk_block_diag")?,
        w_uv_block_diag: require(ctx.w_uv_block_diag, "w_uv_block_diag")?,
        yarn_inv_freq,
        main_inv_freq: yarn_inv_freq,
        q_lora_rank: q_lora,
        kv_lora_rank: kv_lora,
        o_lora_rank: 0,
        nope,
        rope,
        v_dim,
        compressor: None,
        attn_sink: metrale_gpu_runtime::gpu::DevicePtr::NULL,
    };

    let input_norm = dense(ctx.store, &format!("{prefix}.attention_norm.weight"))?;
    let post_norm = dense(ctx.store, &format!("{prefix}.ffn_norm.weight"))?;
    let kv_dtype = layer_kv_dtypes.get(i).copied().unwrap_or(KvCacheDtype::Fp8);

    let ffn = build_moe_ffn(ctx.store, i, gpu, config);

    let mut layer = Qwen3AttentionLayer::new_ungated(
        input_norm, attn, post_norm, ffn, i, None, None, None, gpu, kv_dtype, 0, config,
    )?;
    layer.set_mla_weights(mla_weights);
    Ok(Box::new(layer))
}

fn build_moe_ffn(
    store: &metrale_model_weights::weights::WeightStore,
    i: usize,
    gpu: &dyn GpuBackend,
    config: &metrale_config::ModelConfig,
) -> FfnComponent {
    if config.num_experts == 0 {
        return FfnComponent::None;
    }
    match load_moe_mistral(store, i, config.num_experts, gpu, config) {
        Ok(moe_weights) => {
            match MoeLayer::new(moe_weights, config.num_experts, None, gpu, config) {
                Ok(mut moe) => {
                    // 2026-09-25: The prefill transpose runs only under expert
                    // parallelism; a single GPU keeps the experts untransposed.
                    // A failed transpose is logged and the layer runs untransposed.
                    if config.ep_world_size > 1
                        && let Err(e) = moe.transpose_for_prefill(gpu, config)
                    {
                        tracing::warn!("L{i}: MoE transpose failed: {e}, using untransposed");
                    }
                    FfnComponent::Moe(moe)
                }
                Err(e) => {
                    tracing::warn!("L{i}: MoE construction failed: {e}, using None");
                    FfnComponent::None
                }
            }
        }
        Err(e) => {
            tracing::warn!("L{i}: MoE weight load failed: {e}, using None");
            FfnComponent::None
        }
    }
}
