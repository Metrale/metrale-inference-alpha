// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-25: Weight loader for Gemma-4. Every layer is an ungated attention
//! layer with a GELU dense FFN, plus an MoE FFN when `num_experts > 0`.
//! Layers are sliding or full attention (RoPE θ 10,000 or 1,000,000), and
//! each has four norms and an optional `layer_scalar`. With no `lm_head`
//! tensor the embedding serves as the head.
//!
//! Owner: model-arch weight loader.
//! Invariants: none beyond the types.

use anyhow::Result;
use metrale_cache::kv_cache::KvCacheDtype;
use metrale_config::ModelConfig;
use metrale_gpu_runtime::gpu::GpuBackend;
use metrale_model_weights::weights::WeightStore;

use super::ModelWeightLoader;
use metrale_model_layers::layer::TransformerLayer;
use metrale_model_layers::weight_map::{DenseWeight, MtpWeights};

mod loader_a;
mod loader_b;

pub struct Gemma4WeightLoader;

impl ModelWeightLoader for Gemma4WeightLoader {
    fn supports_tp(&self) -> bool {
        // 2026-09-25: Under TP the loader shards the attention projections
        // (see `loader_a::load_layers_impl`); the dense FFN weights are loaded
        // whole on every rank.
        true
    }

    fn load_layers(
        &self,
        store: &WeightStore,
        config: &ModelConfig,
        gpu: &dyn GpuBackend,
        layer_kv_dtypes: &[KvCacheDtype],
    ) -> Result<Vec<Box<dyn TransformerLayer>>> {
        loader_a::load_layers_impl(store, config, gpu, layer_kv_dtypes)
    }

    fn load_embedding(
        &self,
        store: &WeightStore,
        config: &ModelConfig,
        _gpu: &dyn GpuBackend,
    ) -> Result<DenseWeight> {
        loader_b::load_embedding_impl(store, config)
    }

    fn load_final_norm(
        &self,
        store: &WeightStore,
        config: &ModelConfig,
        _gpu: &dyn GpuBackend,
    ) -> Result<DenseWeight> {
        loader_b::load_final_norm_impl(store, config)
    }

    fn load_lm_head(
        &self,
        store: &WeightStore,
        config: &ModelConfig,
        _gpu: &dyn GpuBackend,
    ) -> Result<DenseWeight> {
        loader_b::load_lm_head_impl(store, config)
    }

    fn load_mtp_weights(
        &self,
        store: &WeightStore,
        config: &ModelConfig,
        gpu: &dyn GpuBackend,
    ) -> Result<Option<MtpWeights>> {
        loader_b::load_mtp_weights_impl(store, config, gpu)
    }

    fn kv_layer_dims(&self, config: &ModelConfig) -> Vec<(usize, usize)> {
        loader_b::kv_layer_dims_impl(config)
    }
}
