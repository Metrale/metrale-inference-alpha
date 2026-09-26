// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-25: `ModelWeights::from_store`, the whole-model loader for ModelOpt NVFP4 checkpoints with attention and linear-attention layers.
//!
//! Owner: model-layers (weight loading).
//! Invariants: none beyond the types.

#![allow(unused_imports)]

use anyhow::{Context, Result, bail, ensure};
use metrale_gpu_runtime::gpu::{DevicePtr, GpuBackend};
use metrale_model_weights::weights::{WeightDtype, WeightStore};

use super::*;

impl ModelWeights {
    /// 2026-09-25: Build typed weight references from a flat `WeightStore`,
    /// loading every layer as `Nvfp4Variant::Standard`.
    ///
    /// `layer_types[i]` must be `FullAttention` or `LinearAttention`: `SparseAttention`
    /// is an error, and `SlidingAttention` or `Moe` panics.
    pub fn from_store(
        store: &WeightStore,
        layer_types: &[metrale_config::LayerType],
        num_experts: usize,
        gpu: &dyn GpuBackend,
        config: &metrale_config::ModelConfig,
    ) -> Result<Self> {
        let embed_tokens = dense(store, "model.embed_tokens.weight")?;
        let final_norm = dense(store, "model.norm.weight")?;

        let lm_head = if store.contains("lm_head.weight") {
            dense(store, "lm_head.weight")?
        } else {
            embed_tokens
        };

        let mut layers = Vec::with_capacity(layer_types.len());
        for (i, lt) in layer_types.iter().enumerate() {
            let lp = config.layer_prefix(i);
            let input_norm = dense(store, &format!("{lp}.input_layernorm.weight"))?;
            let post_attn_norm = dense(store, &format!("{lp}.post_attention_layernorm.weight"))?;
            // 2026-09-25: Placeholder kernel handles (0): a key that `quantized_any`
            // routes to load-time quantization gets no usable kernel here.
            let dummy_qctx = QuantizeCtx {
                absmax_k: metrale_gpu_runtime::gpu::KernelHandle(0),
                quantize_k: metrale_gpu_runtime::gpu::KernelHandle(0),
                stream: 0,
            };
            let moe = load_moe(
                store,
                &lp,
                num_experts,
                gpu,
                config,
                Nvfp4Variant::Standard,
                dummy_qctx,
            )?;

            match lt {
                metrale_config::LayerType::FullAttention => {
                    let attn = load_attention(
                        store,
                        &lp,
                        gpu,
                        Nvfp4Variant::Standard,
                        dummy_qctx,
                        config,
                    )?;
                    layers.push(LayerWeights::FullAttention {
                        input_norm,
                        attn,
                        post_attn_norm,
                        moe,
                    });
                }
                metrale_config::LayerType::LinearAttention => {
                    let ssm =
                        load_ssm(store, &lp, gpu, Nvfp4Variant::Standard, dummy_qctx, config)?;
                    layers.push(LayerWeights::LinearAttention {
                        input_norm,
                        ssm,
                        post_attn_norm,
                        moe,
                    });
                }
                metrale_config::LayerType::SlidingAttention => {
                    unreachable!("unexpected SlidingAttention in this loader")
                }
                metrale_config::LayerType::Moe => {
                    unreachable!("Qwen3 has no standalone MoE layers")
                }
                // 2026-09-25: `LayerWeights` has no sparse variant, and a sparse layer
                // bound as dense attention would attend over the whole cache.
                metrale_config::LayerType::SparseAttention => anyhow::bail!(
                    "layer {i}: SparseAttention has no weight-map variant in this loader"
                ),
            }

            if (i + 1) % 12 == 0 {
                tracing::info!("Mapped weights for layers 0..{}", i + 1);
            }
        }

        tracing::info!(
            "Weight map: {} layers ({} attention, {} SSM)",
            layers.len(),
            layers
                .iter()
                .filter(|l| matches!(l, LayerWeights::FullAttention { .. }))
                .count(),
            layers
                .iter()
                .filter(|l| matches!(l, LayerWeights::LinearAttention { .. }))
                .count(),
        );

        Ok(Self {
            embed_tokens,
            final_norm,
            lm_head,
            layers,
        })
    }
}
