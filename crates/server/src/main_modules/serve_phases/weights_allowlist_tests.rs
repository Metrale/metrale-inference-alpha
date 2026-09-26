// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-26: Tests of the per-model lists in [`super`] that leave checkpoint
//! tensors unloaded, asserting both members and non-members.
//!
//! Owner: server startup (`met serve`).
//! Invariants: none beyond the types.

use super::{skip_activation_scales, skip_mtp};
use metrale_config::ModelConfig;

fn cfg(model_type: &str) -> ModelConfig {
    let mut c = ModelConfig::qwen3_next_80b_nvfp4();
    c.model_type = model_type.to_string();
    c
}

#[test]
fn activation_scales_are_skipped_only_for_the_listed_models() {
    assert!(skip_activation_scales(&cfg("glm5_next")));
    assert!(skip_activation_scales(&cfg("qwen4_exp")));

    // 2026-09-26: `step3p7` reads `input_scale` on its own loader path.
    for keep in ["step3p7", "qwen3_5_moe", "minimax_m2", "kimi_k3", "llama"] {
        assert!(
            !skip_activation_scales(&cfg(keep)),
            "{keep} must keep its activation scales"
        );
    }
}

/// 2026-09-26: The two lists are independent: `glm5_next` skips activation
/// scales but keeps `mtp.*`.
#[test]
fn the_mtp_skip_list_is_unchanged_by_the_activation_scale_list() {
    assert!(skip_mtp(&cfg("qwen4_exp")));
    assert!(!skip_mtp(&cfg("glm5_next")));
}
