// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-25: `ModelWeightLoader` for `MistralWeightLoader`: every layer runs the same sequence of load steps over one `ctx::MistralLayerCtx`.
//!
//! The steps, in the order `load_layers_inner` runs them:
//! - `phase_lora_qkv`: wq_a, wq_b, wkv_a, wkv_b and their norms, NVFP4 copies, TP shard.
//! - `phase_per_head`: W_UK_T (per-head transpose), W_UV and wq_b_rope.
//! - `phase_qk_absorbed`: W_QK_absorbed, computed on the CPU.
//! - `phase_block_diag`: block-diagonal W_UK and W_UV.
//! - `phase_o_proj`: the output projection and its NVFP4 copy.
//! - `yarn`: the YaRN inv_freq table, computed for the first layer and shared.
//! - `phase_assemble`: `MlaWeights`, the MoE FFN and the `TransformerLayer`.
//!
//! Owner: model-arch (Mistral loader).
//! Invariants:
//! - Layers are built in index order, and the first error ends the load.
//! - Every layer gets the same YaRN inv_freq pointer.

use anyhow::{Context, Result};
use metrale_cache::kv_cache::KvCacheDtype;
use metrale_config::ModelConfig;
use metrale_gpu_runtime::gpu::GpuBackend;
use metrale_model_weights::weights::WeightStore;

use super::MistralWeightLoader;
use crate::weight_loader::ModelWeightLoader;
use metrale_model_layers::layer::TransformerLayer;
use metrale_model_layers::layers::VisionTower;
use metrale_model_layers::weight_map::{DenseWeight, MtpWeights, dense};

// 2026-09-25: `ctx` and the three steps that read no tensor names are
// `pub(crate)` because the LongCat loader runs them too. The steps that read
// Mistral tensor names stay private.
pub(crate) mod ctx;
mod phase_assemble;
pub(crate) mod phase_block_diag;
mod phase_lora_qkv;
mod phase_o_proj;
pub(crate) mod phase_per_head;
pub(crate) mod phase_qk_absorbed;
mod yarn;

impl ModelWeightLoader for MistralWeightLoader {
    fn supports_tp(&self) -> bool {
        // 2026-09-25: Under TP, `phase_lora_qkv` shards wq_b and wkv_b
        // column-parallel on the head axis; wq_a, wkv_a and the two latent
        // norms stay replicated. The per-head steps loop over
        // `num_key_value_heads`, which the server has already divided by the
        // TP size, so they build this rank's heads only.
        true
    }

    fn load_layers(
        &self,
        store: &WeightStore,
        config: &ModelConfig,
        gpu: &dyn GpuBackend,
        layer_kv_dtypes: &[KvCacheDtype],
    ) -> Result<Vec<Box<dyn TransformerLayer>>> {
        self.load_layers_inner(store, config, gpu, layer_kv_dtypes)
    }

    fn load_embedding(
        &self,
        store: &WeightStore,
        _config: &ModelConfig,
        _gpu: &dyn GpuBackend,
    ) -> Result<DenseWeight> {
        dense(store, "tok_embeddings.weight")
            .or_else(|_| dense(store, "model.embed_tokens.weight"))
            .context("Mistral: embedding not found")
    }

    fn load_final_norm(
        &self,
        store: &WeightStore,
        _config: &ModelConfig,
        _gpu: &dyn GpuBackend,
    ) -> Result<DenseWeight> {
        dense(store, "norm.weight")
            .or_else(|_| dense(store, "model.norm.weight"))
            .context("Mistral: final norm not found")
    }

    fn load_lm_head(
        &self,
        store: &WeightStore,
        config: &ModelConfig,
        _gpu: &dyn GpuBackend,
    ) -> Result<DenseWeight> {
        if store.contains("output.weight") {
            dense(store, "output.weight")
        } else if store.contains("lm_head.weight") {
            dense(store, "lm_head.weight")
        } else if config.tie_word_embeddings {
            dense(store, "tok_embeddings.weight")
                .or_else(|_| dense(store, "model.embed_tokens.weight"))
                .context("Mistral: tied embedding lm_head not found")
        } else {
            anyhow::bail!("Mistral: lm_head/output weight not found")
        }
    }

    fn load_mtp_weights(
        &self,
        _store: &WeightStore,
        _config: &ModelConfig,
        _gpu: &dyn GpuBackend,
    ) -> Result<Option<MtpWeights>> {
        Ok(None)
    }

    fn load_vision_encoder(
        &self,
        _store: &WeightStore,
        _config: &ModelConfig,
        _gpu: &dyn GpuBackend,
    ) -> Result<Option<VisionTower>> {
        Ok(None)
    }
}

impl MistralWeightLoader {
    pub(crate) fn load_layers_inner(
        &self,
        store: &WeightStore,
        config: &ModelConfig,
        gpu: &dyn GpuBackend,
        layer_kv_dtypes: &[KvCacheDtype],
    ) -> Result<Vec<Box<dyn TransformerLayer>>> {
        let n = config.num_hidden_layers;
        let q_lora = config.q_lora_rank;
        let kv_lora = config.kv_lora_rank;
        let nope = config.qk_nope_head_dim;
        let rope = config.qk_rope_head_dim;
        let v_dim = config.v_head_dim;

        tracing::info!(
            "Mistral MLA→GQA: expanding LoRA on GPU (q_lora={q_lora}, kv_lora={kv_lora}, \
             nope={nope}, rope={rope}, v_dim={v_dim})"
        );

        let absmax_k = gpu.kernel("quantize_nvfp4", "nvfp4_global_absmax")?;
        let quantize_k = gpu.kernel("quantize_nvfp4", "quantize_bf16_to_nvfp4")?;
        let stream = gpu.default_stream();

        let mut layers: Vec<Box<dyn TransformerLayer>> = Vec::with_capacity(n);
        let mut yarn_inv_freq_shared = metrale_gpu_runtime::gpu::DevicePtr::NULL;

        for i in 0..n {
            let mut ctx =
                ctx::MistralLayerCtx::new(store, config, gpu, absmax_k, quantize_k, stream, i);
            phase_lora_qkv::load_lora_qkv(&mut ctx)?;
            phase_per_head::build_per_head_views(&mut ctx)?;
            phase_qk_absorbed::build_w_qk_absorbed(&mut ctx)?;
            phase_block_diag::build_block_diagonals(&mut ctx)?;
            phase_o_proj::load_o_proj(&mut ctx)?;
            let yarn_inv_freq =
                ctx::ensure_yarn_inv_freq(&mut yarn_inv_freq_shared, config, rope, gpu)?;
            let layer = phase_assemble::assemble_layer(ctx, yarn_inv_freq, layer_kv_dtypes)?;
            layers.push(layer);

            if (i + 1) % 6 == 0 || i == n - 1 {
                let free = gpu.free_memory().unwrap_or(0);
                tracing::info!("L{}/{n} done — {:.1} GB free", i + 1, free as f64 / 1e9);
            }
        }
        Ok(layers)
    }
}
