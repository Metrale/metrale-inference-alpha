// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-25: The embedding, final-norm and LM-head loads that the
//! `Qwen35DenseWeightLoader` trait impl in `qwen35_dense.rs` calls.
//!
//! Owner: model-arch weight loader (Qwen3.5 dense).
//! Invariants: none beyond the types.

use anyhow::Result;
use metrale_config::ModelConfig;
use metrale_gpu_runtime::gpu::GpuBackend;
use metrale_model_weights::weights::{WeightDtype, WeightStore};

use metrale_model_layers::weight_map::{DenseWeight, dense, dense_auto_fp8_or_bf16};

pub(super) fn load_embedding(store: &WeightStore, config: &ModelConfig) -> Result<DenseWeight> {
    let prefix = &config.weight_prefix;
    dense(store, &format!("{prefix}.embed_tokens.weight"))
}

pub(super) fn load_final_norm(store: &WeightStore, config: &ModelConfig) -> Result<DenseWeight> {
    let prefix = &config.weight_prefix;
    dense(store, &format!("{prefix}.norm.weight"))
}

/// 2026-09-25: Load the LM head from the first of `lm_head`,
/// `language_model.lm_head` and `model.lm_head` present. An FP8 E4M3 head is
/// dequantized to BF16 (`dense_auto_fp8_or_bf16`); any other dtype is passed
/// through as the store pointer (`dense`), since `dense_auto_fp8_or_bf16`
/// rejects every dtype but BF16 and FP8 E4M3.
pub(super) fn load_lm_head(
    store: &WeightStore,
    config: &ModelConfig,
    gpu: &dyn GpuBackend,
) -> Result<DenseWeight> {
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
            dense_auto_fp8_or_bf16(store, prefix, gpu)
        } else {
            dense(store, &key)
        };
    }
    // 2026-09-25: No head tensor: the head is `embed_tokens`, passed through
    // with no dtype check.
    let prefix = &config.weight_prefix;
    dense(store, &format!("{prefix}.embed_tokens.weight"))
}
