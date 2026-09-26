// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-26: Tests for parsing a DeepSeek-V4 `config.json`.
//!
//! Owner: config (model parsers).
//! Invariants: none beyond the types.

use super::*;

#[test]
fn test_parse_deepseek_v4_config() {
    let json = r#"{
        "model_type": "deepseek_v4",
        "hidden_size": 4096,
        "num_hidden_layers": 43,
        "num_attention_heads": 64,
        "num_key_value_heads": 1,
        "head_dim": 512,
        "q_lora_rank": 1024,
        "o_lora_rank": 1024,
        "qk_rope_head_dim": 64,
        "n_routed_experts": 256,
        "n_shared_experts": 1,
        "num_experts_per_tok": 6,
        "moe_intermediate_size": 2048,
        "norm_topk_prob": true,
        "scoring_func": "sqrtsoftplus",
        "topk_method": "noaux_tc",
        "routed_scaling_factor": 1.5,
        "sliding_window": 128,
        "max_position_embeddings": 1048576,
        "rope_theta": 10000,
        "rms_norm_eps": 1e-06,
        "vocab_size": 129280,
        "bos_token_id": 0,
        "eos_token_id": 1,
        "tie_word_embeddings": false,
        "num_nextn_predict_layers": 1,
        "dspark_block_size": 5,
        "dspark_noise_token_id": 128799,
        "dspark_target_layer_ids": [40, 41, 42],
        "dspark_markov_rank": 256,
        "rope_scaling": {
            "type": "yarn",
            "factor": 16,
            "original_max_position_embeddings": 65536,
            "beta_fast": 32,
            "beta_slow": 1
        },
        "index_n_heads": 64,
        "index_head_dim": 128,
        "index_topk": 512,
        "compress_ratios": [0, 0, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 0],
        "num_hash_layers": 3
    }"#;
    let cfg = parse_config(json).unwrap();
    assert_eq!(cfg.model_type, "deepseek_v4");
    assert_eq!(cfg.hidden_size, 4096);
    assert_eq!(cfg.num_hidden_layers, 43);
    assert_eq!(cfg.num_attention_heads, 64);
    assert_eq!(cfg.num_key_value_heads, 1);
    assert_eq!(cfg.head_dim, 512);
    assert_eq!(cfg.q_lora_rank, 1024);
    assert_eq!(cfg.o_lora_rank, 1024);
    assert_eq!(cfg.qk_rope_head_dim, 64);
    assert_eq!(cfg.kv_lora_rank, 512);
    assert_eq!(cfg.qk_nope_head_dim, 448);
    assert_eq!(cfg.v_head_dim, 512);
    assert_eq!(cfg.partial_rotary_factor, 0.125);
    assert_eq!(cfg.num_experts, 256);
    assert_eq!(cfg.num_experts_per_tok, 6);
    assert_eq!(cfg.moe_intermediate_size, 2048);
    assert_eq!(cfg.shared_expert_intermediate_size, 2048);
    assert!(cfg.norm_topk_prob);
    assert_eq!(cfg.scoring_func, "sqrtsoftplus");
    assert!(cfg.use_routing_bias);
    assert_eq!(cfg.routed_scaling_factor, 1.5);
    assert_eq!(cfg.num_mtp_modules, 1);
    assert_eq!(cfg.mtp_transformer_layers, 1);
    assert_eq!(cfg.dspark_block_size, 5);
    assert_eq!(cfg.dspark_noise_token_id, 128799);
    assert_eq!(cfg.dspark_target_layer_ids, vec![40, 41, 42]);
    assert_eq!(cfg.dspark_markov_rank, 256);
    assert_eq!(cfg.sliding_window, 128);
    assert_eq!(cfg.max_position_embeddings, 1048576);
    assert_eq!(cfg.rope_theta, 10000.0);
    assert_eq!(cfg.rms_norm_eps, 1e-6);
    assert_eq!(cfg.vocab_size, 129280);
    assert_eq!(cfg.bos_token_id, 0);
    assert_eq!(cfg.eos_token_id, 1);
    assert!(!cfg.tie_word_embeddings);
    assert!(!cfg.attn_gated);
    assert_eq!(cfg.weight_prefix, "model");
    // 2026-09-26: The fixture has one `compress_ratios` entry more than it has layers; all
    // are kept.
    assert_eq!(cfg.compress_ratios.len(), 44);
    assert_eq!(cfg.index_n_heads, 64);
    assert_eq!(cfg.index_head_dim, 128);
    assert_eq!(cfg.index_topk, 512);
    assert_eq!(cfg.num_hash_layers, 3);
    assert_eq!(cfg.hc_mult, 4);
    assert_eq!(cfg.hc_sinkhorn_iters, 20);
    assert_eq!(cfg.hc_eps, 1e-6);
    assert_eq!(cfg.num_attention_layers(), 43);
    assert_eq!(cfg.num_ssm_layers(), 0);
    let caps = cfg.capabilities();
    assert!(caps.has_attention_layers);
    assert!(caps.has_moe_layers);
    assert!(!caps.has_ssm_layers);
    assert!(caps.has_mtp);
    assert_eq!(caps.attention_type, crate::capabilities::AttentionType::Mla);
}
