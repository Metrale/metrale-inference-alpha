// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-26: Tests for the scalar and array forms of `eos_token_id`.
//!
//! Owner: config.
//! Invariants: none beyond the types.

use super::*;

/// 2026-09-26: A minimal nested config whose only varying part is `eos_token_id`.
fn eos_fixture(eos: &str) -> String {
    format!(
        r#"{{
        "model_type": "qwen3_5_moe",
        "text_config": {{
            "model_type": "qwen3_5_moe_text",
            "hidden_size": 2048,
            "num_hidden_layers": 4,
            "num_attention_heads": 16,
            "num_key_value_heads": 2,
            "head_dim": 128,
            "vocab_size": 1000,
            "eos_token_id": {eos}
        }}
    }}"#
    )
}

/// 2026-09-26: A scalar `eos_token_id` gives that primary and a one-element stop set.
#[test]
fn scalar_eos_round_trips_unchanged() {
    let cfg = parse_config(&eos_fixture("248044")).unwrap();
    assert_eq!(cfg.eos_token_id, 248044);
    assert_eq!(cfg.eos_ids(), vec![248044]);
    assert!(cfg.is_eos(248044));
    assert!(!cfg.is_eos(1));
}

/// 2026-09-26: An array `eos_token_id` keeps every id, element 0 first as the primary.
#[test]
fn array_eos_preserves_every_id_primary_first() {
    let cfg = parse_config(&eos_fixture("[154820, 154827, 154829]")).unwrap();
    assert_eq!(cfg.eos_token_id, 154820, "primary is element 0");
    assert_eq!(cfg.eos_ids(), vec![154820, 154827, 154829]);
    for id in [154820u32, 154827, 154829] {
        assert!(cfg.is_eos(id), "generation must stop on {id}");
    }
    assert!(!cfg.is_eos(154821));
}

/// 2026-09-26: A `ModelConfig` built without `parse_config` has an empty `eos_token_ids`;
/// `eos_ids()` and `is_eos()` then use the scalar `eos_token_id`.
#[test]
fn an_unpopulated_eos_set_falls_back_to_the_scalar() {
    let mut cfg = ModelConfig::qwen3_next_80b_nvfp4();
    cfg.eos_token_id = 7;
    assert!(cfg.eos_token_ids.is_empty());
    assert_eq!(cfg.eos_ids(), vec![7]);
    assert!(cfg.is_eos(7));
    assert!(!cfg.is_eos(8));
}

/// 2026-09-26: Building the stop set keeps the family parser's primary id first; `step3p7`
/// takes the last array element as the primary.
#[test]
fn populating_the_set_never_overrides_a_parser_s_primary_choice() {
    let mut cfg = ModelConfig::qwen3_next_80b_nvfp4();
    cfg.eos_token_id = 128007;
    crate::dispatch::populate_eos_token_ids_for_test(
        &mut cfg,
        r#"{"eos_token_id": [1, 2, 128007]}"#,
    );
    assert_eq!(cfg.eos_token_id, 128007, "primary untouched");
    assert_eq!(
        cfg.eos_ids(),
        vec![128007, 1, 2],
        "primary first, then the rest, de-duplicated"
    );
}
