// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-25: DeepSeek-V4 weight loader: layers, embedding, final norm and lm_head.
//!
//! Owner: model-arch weight loader.
//! Invariants:
//! - `supports_tp` is false: V4 layers shard experts by EP only.
//! - `load_mtp_weights` returns `None`; the V4 MTP module is loaded separately
//!   by `mtp::load_v4_mtp_module`.

mod assemble;
mod attn_sink;
mod compute;
mod csa_ape;
mod load_layers;
pub mod mtp;
pub(crate) use mtp::DeepseekV4MtpModule;

use anyhow::{Context, Result};
use metrale_cache::kv_cache::KvCacheDtype;
use metrale_config::ModelConfig;
use metrale_gpu_runtime::gpu::GpuBackend;
use metrale_model_weights::weights::WeightStore;

use super::ModelWeightLoader;
use metrale_model_layers::layer::TransformerLayer;
use metrale_model_layers::weight_map::quant_helpers::dense_auto;
use metrale_model_layers::weight_map::{DenseWeight, MtpWeights, dense};

pub struct DeepSeekV4WeightLoader;

impl ModelWeightLoader for DeepSeekV4WeightLoader {
    fn supports_tp(&self) -> bool {
        false
    }

    fn load_layers(
        &self,
        store: &WeightStore,
        config: &ModelConfig,
        gpu: &dyn GpuBackend,
        layer_kv_dtypes: &[KvCacheDtype],
    ) -> Result<Vec<Box<dyn TransformerLayer>>> {
        load_layers::load_all_layers(store, config, gpu, layer_kv_dtypes)
    }

    fn load_embedding(
        &self,
        store: &WeightStore,
        _config: &ModelConfig,
        _gpu: &dyn GpuBackend,
    ) -> Result<DenseWeight> {
        if let Ok(w) = dense(store, "embed.weight") {
            return Ok(w);
        }
        if let Ok(w) = dense(store, "model.embed_tokens.weight") {
            return Ok(w);
        }
        dense(store, "embed_tokens.weight")
            .context("DeepSeek-V4: no embedding tensor found (tried embed.weight, model.embed_tokens.weight, embed_tokens.weight)")
    }

    fn load_final_norm(
        &self,
        store: &WeightStore,
        _config: &ModelConfig,
        _gpu: &dyn GpuBackend,
    ) -> Result<DenseWeight> {
        // 2026-09-25: Loaded as stored, with no offset: `deepseek_v4` is in
        // `metrale_model_layers::ships_vanilla_norm_weights`, so the final norm runs
        // `rms_norm_vanilla` (`out = x * w / rms`).
        if let Ok(w) = dense_auto(store, "norm.weight", _gpu) {
            return Ok(w);
        }
        if let Ok(w) = dense_auto(store, "model.norm.weight", _gpu) {
            return Ok(w);
        }
        dense_auto(store, "final_norm.weight", _gpu)
            .context("DeepSeek-V4: no final norm tensor found (tried norm.weight, model.norm.weight, final_norm.weight)")
    }

    fn load_lm_head(
        &self,
        store: &WeightStore,
        config: &ModelConfig,
        _gpu: &dyn GpuBackend,
    ) -> Result<DenseWeight> {
        if store.contains("lm_head.weight") {
            return dense(store, "lm_head.weight");
        }
        if store.contains("output.weight") {
            return dense(store, "output.weight");
        }
        if store.contains("head.weight") {
            return dense(store, "head.weight");
        }
        if config.tie_word_embeddings
            || store.contains("embed.weight")
            || store.contains("model.embed_tokens.weight")
        {
            if let Ok(w) = dense(store, "embed.weight") {
                return Ok(w);
            }
            if let Ok(w) = dense(store, "model.embed_tokens.weight") {
                return Ok(w);
            }
            return dense(store, "embed_tokens.weight")
                .context("DeepSeek-V4: tied lm_head — no embedding tensor found");
        }
        anyhow::bail!(
            "DeepSeek-V4: lm_head not found (tried lm_head.weight, output.weight, head.weight, and tied embeddings)"
        )
    }

    fn load_mtp_weights(
        &self,
        _store: &WeightStore,
        _config: &ModelConfig,
        _gpu: &dyn GpuBackend,
    ) -> Result<Option<MtpWeights>> {
        Ok(None)
    }
}
