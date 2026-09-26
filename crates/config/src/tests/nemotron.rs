// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-26: Nemotron-H config tests: the Mamba/MoE/attention layout, the refusal of
//! invalid Mamba geometry, and the Puzzle variant.
//!
//! Owner: config (model parsers).
//! Invariants: none beyond the types.

use super::*;

#[test]
fn nemotron_h_fixture_maps_mamba_moe_and_weight_layout() {
    let json = r#"{
        "model_type": "nemotron_h",
        "hidden_size": 2688,
        "num_hidden_layers": 52,
        "num_attention_heads": 32,
        "num_key_value_heads": 2,
        "head_dim": 128,
        "intermediate_size": 1856,
        "n_routed_experts": 128,
        "num_experts_per_tok": 6,
        "moe_intermediate_size": 1856,
        "moe_shared_expert_intermediate_size": 3712,
        "vocab_size": 131072,
        "hybrid_override_pattern": "MEMEM*EMEMEM*EMEMEM*EMEMEM*EMEMEM*EMEMEMEM*EMEMEMEME",
        "mamba_num_heads": 64,
        "mamba_head_dim": 64,
        "ssm_state_size": 128,
        "n_groups": 8,
        "expand": 2,
        "conv_kernel": 4,
        "norm_eps": 1e-5,
        "rope_theta": 10000,
        "routed_scaling_factor": 2.5,
        "norm_topk_prob": true
    }"#;
    let cfg = parse_config(json).unwrap();
    assert_eq!(cfg.model_type, "nemotron_h");
    assert_eq!(cfg.hidden_size, 2688);
    assert_eq!(cfg.num_hidden_layers, 52);
    assert_eq!(cfg.num_experts, 128);
    assert_eq!(cfg.num_experts_per_tok, 6);
    assert_eq!(cfg.shared_expert_intermediate_size, 3712);
    assert_eq!(cfg.rms_norm_eps, 1e-5);
    assert_eq!(cfg.linear_conv_kernel_dim, 4);
    assert_eq!(cfg.mamba_num_heads, 64);
    assert_eq!(cfg.mamba_head_dim, 64);
    assert_eq!(cfg.ssm_state_size, 128);
    assert_eq!(cfg.n_groups, 8);
    assert_eq!(cfg.mamba2_d_inner(), 4096);
    assert_eq!(cfg.layer_types.len(), 52);
    assert_eq!(cfg.num_ssm_layers(), 23);
    assert_eq!(cfg.num_moe_layers(), 23);
    assert_eq!(cfg.num_attention_layers(), 6);
    assert_eq!(cfg.layer_type(0), LayerType::LinearAttention);
    assert_eq!(cfg.layer_type(1), LayerType::Moe);
    assert_eq!(cfg.layer_type(5), LayerType::FullAttention);
    assert_eq!(cfg.gqa_ratio(), 16);
    assert_eq!(cfg.rotary_dim(), 128);
    assert_eq!(cfg.routed_scaling_factor, 2.5);
    assert!(cfg.norm_topk_prob);
    assert_eq!(cfg.weight_prefix, "backbone");
}

#[test]
fn nemotron_h_rejects_invalid_mamba_geometry() {
    let base = serde_json::json!({
        "model_type": "nemotron_h",
        "hidden_size": 2688,
        "num_hidden_layers": 1,
        "hybrid_override_pattern": "M",
        "mamba_num_heads": 64,
        "mamba_head_dim": 64,
        "ssm_state_size": 128,
        "n_groups": 8,
        "conv_kernel": 4
    });
    let mut accepted = Vec::new();

    for (field, value) in [
        ("mamba_head_dim", 0),
        ("ssm_state_size", 0),
        ("n_groups", 0),
        ("n_groups", 7),
    ] {
        let mut raw = base.clone();
        raw[field] = serde_json::json!(value);
        match parse_config(&raw.to_string()) {
            Ok(_) => accepted.push(field),
            Err(error) => assert!(
                error.to_string().contains(field),
                "error for {field} did not name the invalid field: {error}"
            ),
        }
    }

    assert!(accepted.is_empty(), "parser accepted invalid {accepted:?}");
}

#[test]
fn test_parse_nemotron_h_puzzle_config() {
    // 2026-09-26: A Puzzle schedule of 4 layers whose MoE layers differ in size and top-k.
    let json = r#"{
        "model_type": "nemotron_h_puzzle",
        "architectures": ["NemotronHPuzzleForCausalLM"],
        "hidden_size": 4096,
        "num_hidden_layers": null,
        "num_attention_heads": 32,
        "num_key_value_heads": 2,
        "head_dim": 128,
        "intermediate_size": 21504,
        "n_routed_experts": 512,
        "n_shared_experts": 1,
        "moe_latent_size": 1024,
        "moe_shared_expert_intermediate_size": 5376,
        "vocab_size": 131072,
        "layers_block_type": ["mamba", "moe", "attention", "moe"],
        "block_configs": [
            {"block_type": "mamba"},
            {"block_type": "moe", "moe_intermediate_size": 1280, "num_experts_per_tok": 4},
            {"block_type": "attention"},
            {"block_type": "moe", "moe_intermediate_size": 2688, "num_experts_per_tok": 22}
        ],
        "mamba_num_heads": 128,
        "mamba_head_dim": 64,
        "ssm_state_size": 96,
        "n_groups": 8,
        "expand": 2,
        "conv_kernel": 4,
        "norm_eps": 1e-5,
        "routed_scaling_factor": 5.0,
        "norm_topk_prob": true
    }"#;
    let cfg = parse_config(json).unwrap();
    assert_eq!(cfg.model_type, "nemotron_h_puzzle");
    assert_eq!(cfg.num_hidden_layers, 4);
    assert_eq!(cfg.num_experts, 512);
    assert_eq!(cfg.moe_latent_size, 1024);
    assert_eq!(cfg.shared_expert_intermediate_size, 5376);
    assert_eq!(cfg.layer_types.len(), 4);
    assert_eq!(cfg.layer_type(0), LayerType::LinearAttention);
    assert_eq!(cfg.layer_type(1), LayerType::Moe);
    assert_eq!(cfg.layer_type(2), LayerType::FullAttention);
    assert_eq!(cfg.layer_type(3), LayerType::Moe);
    assert_eq!(cfg.num_moe_layers(), 2);
    assert_eq!(cfg.moe_intermediate_size_for(1), 1280);
    assert_eq!(cfg.num_experts_per_tok_for(1), 4);
    assert_eq!(cfg.moe_intermediate_size_for(3), 2688);
    assert_eq!(cfg.num_experts_per_tok_for(3), 22);
    // 2026-09-26: The scalar fields, and the lookups for a layer without MoE, give the
    // largest per-layer value.
    assert_eq!(cfg.moe_intermediate_size, 2688);
    assert_eq!(cfg.num_experts_per_tok, 22);
    assert_eq!(cfg.moe_intermediate_size_for(0), 2688);
    assert_eq!(cfg.num_experts_per_tok_for(0), 22);
    assert_eq!(cfg.max_moe_intermediate_size(), 2688);
    assert_eq!(cfg.moe_input_size(), 1024);
    assert_eq!(cfg.weight_prefix, "backbone");
}
