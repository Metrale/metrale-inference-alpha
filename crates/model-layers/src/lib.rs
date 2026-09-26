// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-25: Model layers: the per-layer forward implementations, LoRA
//! adapters, weight maps, the SSM reserve, speculative helpers and the
//! vision/video preprocessing, plus the per-model-type dispatch predicates
//! below.
//!
//! Owner: model-layers.
//! Invariants: none beyond the types.

#![deny(warnings)]
#![deny(clippy::all)]
#![allow(clippy::too_many_arguments)]
#![allow(clippy::needless_range_loop)]
#![allow(clippy::if_same_then_else)]
#![allow(clippy::ptr_arg)]
#![allow(clippy::type_complexity)]

pub mod forward;
pub mod layer;
pub mod layers;
pub mod lora;
pub mod quant_format;
pub mod speculative;
pub mod ssm_reserve;
pub mod video_decode_ffmpeg;
pub mod video_preprocess;
pub mod vision_item;
pub mod vision_preprocess;
pub mod vision_preprocess_glm;
/// 2026-09-25: Marconi snapshot-restore threshold: the default, and the
/// setter `met serve` calls with `--marconi-min-tokens` before it builds the
/// model.
pub use crate::mtp_carry::{DEFAULT_MARCONI_MIN_TOKENS, set_marconi_min_tokens};
pub use vision_item::VisionItem;

pub mod weight_map;

/// 2026-09-26: True when the checkpoint's RMSNorm weights are plain
/// multipliers, `out = x * w / rms` (the `rms_norm_vanilla` kernel), rather
/// than offsets from 1, `out = x * (1 + w) / rms` (the `rms_norm` kernel).
/// It selects the final-norm kernel (model-engine `model/impl_a1.rs`) and the
/// attention norm kernels (`layers/qwen3_attention/init_proj_kernels.rs` and
/// `init_prefill_kernels.rs`). The answer comes from `model_type` alone, not
/// from the weight values.
pub fn ships_vanilla_norm_weights(config: &metrale_config::ModelConfig) -> bool {
    model_type_ships_vanilla_norm_weights(&config.model_type)
}

/// 2026-09-25: The predicate behind `ships_vanilla_norm_weights`, on the bare
/// `model_type`, so a test needs no `ModelConfig`.
pub fn model_type_ships_vanilla_norm_weights(model_type: &str) -> bool {
    // 2026-09-25: The final norm runs outside every layer, so a plain-weight
    // model missing from this list gets the `(1 + w)` offset on its final hidden
    // state. Nothing in the weight shapes reveals the convention.
    matches!(
        model_type,
        "deepseek_v4" | "deepseek_v41" | "laguna" | "glm5_next" | "kimi_k3"
    )
}

/// 2026-09-25: Must chunked prefill run as one chunk for this model?
///
/// True for every MLA model (`kv_lora_rank > 0`) except `glm5_next` and
/// `kimi_k3`. The Qwen3 attention layer's paged MLA prefill
/// (`layers/qwen3_attention/prefill/paged_mla.rs`) attends over the K/V of the
/// call's own tokens only and refuses `seq_len_start != 0`, so a second chunk
/// cannot run there.
///
/// `glm5_next` and `kimi_k3` do not prefill through that layer:
/// `Glm5NextLayer::prefill` (model-arch `glm5next_layer/mod.rs`) takes
/// `seq_len_start`, and `K3BoundLayer` (model-arch `kimi_k3/bound.rs`) keeps the
/// trait's default prefill, which calls `decode` once per token. Both keep the
/// scheduler's chunk budget.
pub fn requires_single_chunk_prefill(model_type: &str, kv_lora_rank: usize) -> bool {
    kv_lora_rank > 0 && !matches!(model_type, "glm5_next" | "kimi_k3")
}

#[cfg(test)]
mod single_chunk_prefill_tests {
    use super::requires_single_chunk_prefill as single;

    #[test]
    fn kimi_k3_keeps_prefill_bounded_beyond_two_chunks() {
        assert!(!single("kimi_k3", 512));
        assert!(!single("kimi_k3", 64));
        for model in ["deepseek_v3", "deepseek_v4", "mistral", "unknown_mla"] {
            assert!(
                single(model, 512),
                "{model} must retain its correctness gate"
            );
        }
    }

    #[test]
    fn glm5_next_is_mla_but_chunks_fine() {
        assert!(!single("glm5_next", 512));
        assert!(single("deepseek_v4", 512));
        assert!(single("mistral", 512));
        assert!(!single("qwen3_5_moe", 0));
    }
}

#[cfg(test)]
mod norm_convention_tests {
    use super::model_type_ships_vanilla_norm_weights as vanilla;

    /// 2026-09-25: Only the listed model types take the plain-weight norm.
    #[test]
    fn vanilla_norm_models_are_explicit() {
        assert!(vanilla("deepseek_v4"));
        assert!(vanilla("laguna"));
        assert!(vanilla("glm5_next"));
        assert!(vanilla("kimi_k3"));
        for other in [
            "qwen3_next",
            "qwen3_5_moe",
            "qwen3_moe",
            "deepseek_v3",
            "llama",
            "mistral",
            "nemotron",
            "",
        ] {
            assert!(!vanilla(other), "{other} must keep offset-from-1 semantics");
        }
    }
}

pub mod drafter_context;
pub mod mtp_carry;
