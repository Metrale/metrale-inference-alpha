// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-26: Tests for the Gemma-4 parser: a dense config with rope under
//! `{sliding,full}_attention_config`, and a MoE config with rope under `rope_parameters`.
//!
//! Owner: config (model parsers).
//! Invariants: none beyond the types.

use super::*;

#[test]
fn gemma4_legacy_config_maps_attention_and_embedding_controls() {
    let json = r#"{
        "model_type": "gemma4",
        "tie_word_embeddings": true,
        "final_logit_softcapping": 30.0,
        "text_config": {
            "hidden_size": 5376,
            "num_hidden_layers": 4,
            "num_attention_heads": 32,
            "num_key_value_heads": 16,
            "head_dim": 256,
            "intermediate_size": 21504,
            "vocab_size": 262144,
            "hidden_activation": "gelu_pytorch_tanh",
            "sliding_window": 1024,
            "attention_pattern": [
                "sliding_attention", "sliding_attention",
                "full_attention", "sliding_attention"
            ],
            "full_attention_config": {
                "rope_theta": 1000000.0,
                "partial_rotary_factor": 0.25
            },
            "sliding_attention_config": {
                "rope_theta": 10000.0
            },
            "rms_norm_eps": 1e-6,
            "max_position_embeddings": 262144
        }
    }"#;
    let cfg = parse_config(json).unwrap();
    assert_eq!(cfg.model_type, "gemma4");
    assert_eq!(cfg.hidden_size, 5376);
    assert_eq!(cfg.num_hidden_layers, 4);
    assert_eq!(cfg.num_attention_heads, 32);
    assert_eq!(cfg.num_key_value_heads, 16);
    assert_eq!(cfg.head_dim, 256);
    assert_eq!(cfg.intermediate_size, 21504);
    assert_eq!(cfg.vocab_size, 262144);
    assert_eq!(cfg.rms_norm_eps, 1e-6);
    assert_eq!(cfg.max_position_embeddings, 262144);
    assert_eq!(cfg.sliding_window, 1024);
    assert_eq!(cfg.rope_theta, 10000.0);
    assert_eq!(cfg.partial_rotary_factor, 0.25);
    assert_eq!(cfg.embed_scale, 5376_f32.sqrt());
    assert!(cfg.tie_word_embeddings);
    assert!(!cfg.attn_gated);
    assert!(cfg.nested_config);
    assert_eq!(cfg.layer_types.len(), 4);
    assert_eq!(cfg.num_attention_layers(), 4);
    assert_eq!(cfg.num_ssm_layers(), 0);
    assert_eq!(cfg.num_experts, 0);
    assert_eq!(cfg.mtp_num_hidden_layers, 0);
    assert_eq!(cfg.linear_num_key_heads, 0);
    assert_eq!(cfg.gqa_ratio(), 2);
    assert_eq!(cfg.rotary_dim(), 64);
}

#[test]
fn gemma4_moe_config_preserves_routing_and_canonical_rope_controls() {
    let json = r#"{
        "model_type": "gemma4",
        "tie_word_embeddings": true,
        "text_config": {
            "hidden_size": 2304,
            "num_hidden_layers": 1,
            "num_attention_heads": 8,
            "num_key_value_heads": 4,
            "head_dim": 256,
            "global_head_dim": 512,
            "intermediate_size": 9216,
            "vocab_size": 262144,
            "sliding_window": 512,
            "attention_pattern": ["full_attention"],
            "rope_parameters": {
                "full_attention": {
                    "rope_theta": 1000000.0,
                    "partial_rotary_factor": 0.25
                },
                "sliding_attention": {"rope_theta": 10000.0}
            },
            "num_experts": 128,
            "top_k_experts": 8,
            "moe_intermediate_size": 704,
            "rms_norm_eps": 1e-6,
            "max_position_embeddings": 131072
        }
    }"#;

    let cfg = parse_config(json).unwrap();
    assert_eq!(cfg.num_experts, 128);
    assert_eq!(cfg.num_experts_per_tok, 8);
    assert_eq!(cfg.moe_intermediate_size, 704);
    assert!(cfg.norm_topk_prob);
    assert_eq!(cfg.head_dim, 512);
    assert_eq!(cfg.rope_theta, 10000.0);
    assert_eq!(cfg.partial_rotary_factor, 0.25);
}
