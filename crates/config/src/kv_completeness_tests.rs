// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-26: Tests for the two KV-completeness gates in `kv_completeness.rs`.
//!
//! Owner: config.
//! Invariants: none beyond the types.

use crate::ModelConfig;

/// 2026-09-26: The model types `per_sequence_state_is_kv_complete` refuses outright.
const NOT_KV_COMPLETE: [&str; 2] = ["glm5_next", "glm5_next_text"];

#[test]
fn any_compressed_deepseek_v4_layer_is_not_kv_cache_complete() {
    let mut config = ModelConfig::qwen3_next_80b_nvfp4();
    config.model_type = "deepseek_v4".to_string();

    for ratios in [vec![4, 0, 0], vec![0, 4, 0], vec![0, 0, 128]] {
        config.compress_ratios = ratios;
        assert!(!config.kv_only_prefix_cache_is_safe());
        assert!(!config.kv_only_swap_out_is_safe());
    }
}

#[test]
fn glm_prompt_built_dsa_state_is_not_kv_cache_complete() {
    let mut config = ModelConfig::qwen3_next_80b_nvfp4();

    for model_type in NOT_KV_COMPLETE {
        config.model_type = model_type.to_string();
        assert!(
            !config.kv_only_prefix_cache_is_safe(),
            "{model_type}: prefix cache must stay closed"
        );
        assert!(
            !config.kv_only_swap_out_is_safe(),
            "{model_type}: swap-out must stay closed"
        );
    }
}

/// 2026-09-26: The validation override opens the prefix-cache gate for GLM and nothing else.
/// The test calls the env-free core, so it never changes the process environment.
#[test]
fn glm53_validation_override_opens_only_the_prefix_cache_arm() {
    let mut config = ModelConfig::qwen3_next_80b_nvfp4();

    for model_type in NOT_KV_COMPLETE {
        config.model_type = model_type.to_string();
        assert!(
            config.kv_only_prefix_cache_is_safe_with(true),
            "{model_type}: the validation switch must open the prefix-cache arm"
        );
        assert!(
            !config.kv_only_swap_out_is_safe(),
            "{model_type}: the swap-out image cannot carry aux blobs, so the \
             validation switch must not reach it"
        );
        assert!(
            !config.kv_only_prefix_cache_is_safe_with(false),
            "{model_type}: unset is the rollback, and it must close the arm again"
        );
    }
}

#[test]
fn glm53_validation_override_does_not_reach_compressed_deepseek_v4() {
    let mut config = ModelConfig::qwen3_next_80b_nvfp4();
    config.model_type = "deepseek_v4".to_string();
    config.compress_ratios = vec![4, 0, 0];

    assert!(!config.kv_only_prefix_cache_is_safe_with(true));
    assert!(!config.kv_only_swap_out_is_safe());
}

#[test]
fn kv_complete_models_keep_both_capabilities() {
    let mut config = ModelConfig::qwen3_next_80b_nvfp4();
    assert!(config.kv_only_prefix_cache_is_safe());
    assert!(config.kv_only_swap_out_is_safe());

    config.model_type = "deepseek_v4".to_string();
    config.compress_ratios = vec![0; 3];
    assert!(config.kv_only_prefix_cache_is_safe());
    assert!(config.kv_only_swap_out_is_safe());
}
