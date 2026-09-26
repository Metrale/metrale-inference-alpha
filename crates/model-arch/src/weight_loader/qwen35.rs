// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-25: `Qwen35WeightLoader`, the loader for the Qwen3.5/3.6 MoE family and
//! `holo3_1_moe`: layers (`load_layers`), embedding, final norm, LM head, MTP weights and
//! the vision tower.
//!
//! Owner: model-arch weight loader (Qwen3.5).
//! Invariants: none beyond the types.

pub(crate) mod load_layers;

use anyhow::{Context, Result};
use metrale_cache::kv_cache::KvCacheDtype;
use metrale_config::ModelConfig;
use metrale_gpu_runtime::gpu::GpuBackend;
use metrale_model_weights::weights::{WeightDtype, WeightStore};

use super::ModelWeightLoader;
use metrale_model_layers::layer::TransformerLayer;
use metrale_model_layers::weight_map::loaders_moe::load_mtp;
use metrale_model_layers::weight_map::quant_helpers::dense_auto;
use metrale_model_layers::weight_map::{DenseWeight, MtpWeights, detect_nvfp4_variant};

pub struct Qwen35WeightLoader;

fn vision_dense_auto(
    store: &WeightStore,
    prefix: &str,
    gpu: &dyn GpuBackend,
) -> Result<DenseWeight> {
    let name = format!("{prefix}.weight");
    let w = store.get(&name)?;
    match w.dtype {
        WeightDtype::BF16 | WeightDtype::FP8E4M3 => {
            metrale_model_layers::weight_map::dense_auto_fp8_or_bf16(store, prefix, gpu)
        }
        WeightDtype::FP32 => metrale_model_layers::weight_map::dense_f32_safe(store, &name, gpu),
        other => anyhow::bail!("vision_dense_auto: unsupported dtype {other:?} for {name}"),
    }
}

fn vision_tensor_dense_auto(
    store: &WeightStore,
    name: &str,
    gpu: &dyn GpuBackend,
) -> Result<DenseWeight> {
    let w = store.get(name)?;
    match w.dtype {
        WeightDtype::BF16 => Ok(DenseWeight { weight: w.ptr }),
        WeightDtype::FP32 => metrale_model_layers::weight_map::dense_f32_safe(store, name, gpu),
        other => anyhow::bail!("vision_tensor_dense_auto: unsupported dtype {other:?} for {name}"),
    }
}

impl ModelWeightLoader for Qwen35WeightLoader {
    fn supports_tp(&self) -> bool {
        // 2026-09-25: The native-FP8 and NVFP4 attention arms shard Q/K/V/O, and the BF16
        // dense and NVFP4 linear-attention builders shard by head. The BF16-dequant attention
        // arm and the native-FP8 linear-attention builder refuse TP > 1.
        true
    }

    fn load_layers(
        &self,
        store: &WeightStore,
        config: &ModelConfig,
        gpu: &dyn GpuBackend,
        layer_kv_dtypes: &[KvCacheDtype],
    ) -> Result<Vec<Box<dyn TransformerLayer>>> {
        load_layers::load_layers(self, store, config, gpu, layer_kv_dtypes)
    }

    fn load_embedding(
        &self,
        store: &WeightStore,
        config: &ModelConfig,
        gpu: &dyn GpuBackend,
    ) -> Result<DenseWeight> {
        let prefix = &config.weight_prefix;
        dense_auto(store, &format!("{prefix}.embed_tokens.weight"), gpu)
    }

    fn load_final_norm(
        &self,
        store: &WeightStore,
        config: &ModelConfig,
        gpu: &dyn GpuBackend,
    ) -> Result<DenseWeight> {
        let prefix = &config.weight_prefix;
        dense_auto(store, &format!("{prefix}.norm.weight"), gpu)
    }

    fn load_lm_head(
        &self,
        store: &WeightStore,
        config: &ModelConfig,
        gpu: &dyn GpuBackend,
    ) -> Result<DenseWeight> {
        // 2026-09-25: The head is the first of `lm_head.weight`, `language_model.lm_head.weight`
        // and `model.lm_head.weight` present. An FP8E4M3 head is dequantized to BF16
        // (`dense_auto_fp8_or_bf16`, which rejects any dtype but BF16 and FP8E4M3); every other
        // dtype is passed to `dense` unchecked, so an NVFP4 head's packed bytes load as they are.
        for prefix in ["lm_head", "language_model.lm_head", "model.lm_head"] {
            let key = format!("{prefix}.weight");
            if !store.contains(&key) {
                continue;
            }
            let is_fp8 = store
                .get(&key)
                .map(|w| w.dtype == WeightDtype::FP8E4M3)
                .unwrap_or(false);
            return if is_fp8 {
                metrale_model_layers::weight_map::dense_auto_fp8_or_bf16(store, prefix, gpu)
            } else {
                metrale_model_layers::weight_map::dense(store, &key)
            };
        }
        // 2026-09-25: No head tensor: the head is the embedding table (tied embeddings).
        let prefix = &config.weight_prefix;
        metrale_model_layers::weight_map::dense(store, &format!("{prefix}.embed_tokens.weight"))
    }

    fn load_mtp_weights(
        &self,
        store: &WeightStore,
        config: &ModelConfig,
        gpu: &dyn GpuBackend,
    ) -> Result<Option<MtpWeights>> {
        if !store.contains("mtp.fc.weight") {
            tracing::info!("No MTP weights found — speculative decoding disabled");
            return Ok(None);
        }
        let variant = detect_nvfp4_variant(store, config);
        tracing::info!(
            "Loading MTP weights ({} experts, variant={:?})...",
            config.num_experts,
            variant
        );
        let mtp = load_mtp(store, config.num_experts, gpu, variant)?;
        tracing::info!(
            "MTP weights loaded: fc=[2048,4096], {} experts, attn layer",
            mtp.experts.len(),
        );
        Ok(Some(mtp))
    }

    /// 2026-09-25: Loads the ViT tower from `model.visual` or `model.language_model.visual`.
    /// Returns `None` when `config.vision` is `None` or neither prefix holds
    /// `patch_embed.proj.weight`. Reads `depth` blocks, one deepstack merger per
    /// `deepstack_visual_indexes` entry, and the final merger. Linear weights may be BF16,
    /// FP32 or FP8E4M3 (dequantized to BF16); biases, norms and embeddings BF16 or FP32.
    fn load_vision_encoder(
        &self,
        store: &WeightStore,
        config: &ModelConfig,
        gpu: &dyn GpuBackend,
    ) -> Result<Option<metrale_model_layers::layers::VisionTower>> {
        let vcfg = match &config.vision {
            Some(v) => v.clone(),
            None => return Ok(None),
        };
        // 2026-09-25: The flat `model.visual.*` and the nested `model.language_model.visual.*`
        // layouts are both accepted; the first that holds `patch_embed.proj.weight` wins.
        let vp = if store.contains("model.visual.patch_embed.proj.weight") {
            "model.visual"
        } else if store.contains("model.language_model.visual.patch_embed.proj.weight") {
            "model.language_model.visual"
        } else {
            tracing::warn!(
                "Vision encoder tensors absent under both `model.visual.*` and \
                 `model.language_model.visual.*`; skipping vision tower (text-only mode)"
            );
            return Ok(None);
        };

        let patch_embed_w =
            vision_tensor_dense_auto(store, &format!("{vp}.patch_embed.proj.weight"), gpu)?;
        let patch_embed_b =
            vision_tensor_dense_auto(store, &format!("{vp}.patch_embed.proj.bias"), gpu)?;
        let pos_embed = vision_tensor_dense_auto(store, &format!("{vp}.pos_embed.weight"), gpu)?;
        let pos_embed_shape = store.get(&format!("{vp}.pos_embed.weight"))?.shape.clone();
        let num_position_embeddings = pos_embed_shape
            .first()
            .copied()
            .context("pos_embed shape missing rows")?;

        let mut blocks = Vec::with_capacity(vcfg.depth);
        for i in 0..vcfg.depth {
            let bp = format!("{vp}.blocks.{i}");
            blocks.push(metrale_model_layers::layers::ViTBlock {
                norm1_w: vision_tensor_dense_auto(store, &format!("{bp}.norm1.weight"), gpu)?
                    .weight,
                norm1_b: vision_tensor_dense_auto(store, &format!("{bp}.norm1.bias"), gpu)?.weight,
                qkv_w: vision_dense_auto(store, &format!("{bp}.attn.qkv"), gpu)?.weight,
                qkv_b: vision_tensor_dense_auto(store, &format!("{bp}.attn.qkv.bias"), gpu)?.weight,
                proj_w: vision_dense_auto(store, &format!("{bp}.attn.proj"), gpu)?.weight,
                proj_b: vision_tensor_dense_auto(store, &format!("{bp}.attn.proj.bias"), gpu)?
                    .weight,
                norm2_w: vision_tensor_dense_auto(store, &format!("{bp}.norm2.weight"), gpu)?
                    .weight,
                norm2_b: vision_tensor_dense_auto(store, &format!("{bp}.norm2.bias"), gpu)?.weight,
                fc1_w: vision_dense_auto(store, &format!("{bp}.mlp.linear_fc1"), gpu)?.weight,
                fc1_b: vision_tensor_dense_auto(store, &format!("{bp}.mlp.linear_fc1.bias"), gpu)?
                    .weight,
                fc2_w: vision_dense_auto(store, &format!("{bp}.mlp.linear_fc2"), gpu)?.weight,
                fc2_b: vision_tensor_dense_auto(store, &format!("{bp}.mlp.linear_fc2.bias"), gpu)?
                    .weight,
            });
        }

        let mut deepstack = Vec::with_capacity(vcfg.deepstack_visual_indexes.len());
        for i in 0..vcfg.deepstack_visual_indexes.len() {
            let mp = format!("{vp}.deepstack_merger_list.{i}");
            deepstack.push(metrale_model_layers::layers::MergerLayer {
                norm_w: vision_tensor_dense_auto(store, &format!("{mp}.norm.weight"), gpu)?.weight,
                norm_b: vision_tensor_dense_auto(store, &format!("{mp}.norm.bias"), gpu)?.weight,
                fc1_w: vision_dense_auto(store, &format!("{mp}.linear_fc1"), gpu)?.weight,
                fc1_b: vision_tensor_dense_auto(store, &format!("{mp}.linear_fc1.bias"), gpu)?
                    .weight,
                fc2_w: vision_dense_auto(store, &format!("{mp}.linear_fc2"), gpu)?.weight,
                fc2_b: vision_tensor_dense_auto(store, &format!("{mp}.linear_fc2.bias"), gpu)?
                    .weight,
            });
        }

        let mp = format!("{vp}.merger");
        let merger = metrale_model_layers::layers::MergerLayer {
            norm_w: vision_tensor_dense_auto(store, &format!("{mp}.norm.weight"), gpu)?.weight,
            norm_b: vision_tensor_dense_auto(store, &format!("{mp}.norm.bias"), gpu)?.weight,
            fc1_w: vision_dense_auto(store, &format!("{mp}.linear_fc1"), gpu)?.weight,
            fc1_b: vision_tensor_dense_auto(store, &format!("{mp}.linear_fc1.bias"), gpu)?.weight,
            fc2_w: vision_dense_auto(store, &format!("{mp}.linear_fc2"), gpu)?.weight,
            fc2_b: vision_tensor_dense_auto(store, &format!("{mp}.linear_fc2.bias"), gpu)?.weight,
        };

        let deepstack_indexes = vcfg.deepstack_visual_indexes.clone();
        let ve = metrale_model_layers::layers::VisionEncoder::new(
            patch_embed_w.weight,
            patch_embed_b.weight,
            pos_embed.weight,
            num_position_embeddings,
            blocks,
            deepstack,
            deepstack_indexes,
            merger,
            vcfg.hidden_size,
            vcfg.num_heads,
            vcfg.spatial_merge_size,
            vcfg.out_hidden_size,
            vcfg.intermediate_size,
            vcfg.patch_size,
            vcfg.max_pixels,
            gpu,
        )?;
        tracing::info!(
            "Qwen3.6 vision encoder loaded: depth={}, hidden={}, heads={}, FP8-blocks>=4",
            vcfg.depth,
            vcfg.hidden_size,
            vcfg.num_heads,
        );
        Ok(Some(metrale_model_layers::layers::VisionTower::qwen(ve)))
    }
}
