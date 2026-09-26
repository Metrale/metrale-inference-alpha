// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-26: Tests for `parse_config` and the `ModelConfig` methods; the per-family tests
//! are in `tests/`.
//!
//! Owner: config.
//! Invariants: none beyond the types.

#![allow(unused_imports)]

mod tests_b;
use super::*;

mod deepseek_v4;
mod eos;
mod gemma4;
mod holo;
mod nemotron;
mod nllb;

#[test]
fn qwen3_next_factory_preserves_attention_ssm_and_routing_shape() {
    let cfg = ModelConfig::qwen3_next_80b_nvfp4();
    assert_eq!(cfg.hidden_size, 2048);
    assert_eq!(cfg.num_hidden_layers, 48);
    assert_eq!(cfg.num_attention_heads, 16);
    assert_eq!(cfg.num_key_value_heads, 2);
    assert_eq!(cfg.num_experts, 512);
    assert_eq!(cfg.num_experts_per_tok, 10);
    assert_eq!(cfg.num_attention_layers(), 12);
    assert_eq!(cfg.num_ssm_layers(), 36);
    assert_eq!(cfg.gqa_ratio(), 8);
    assert_eq!(cfg.rotary_dim(), 64);
    assert_eq!(cfg.vocab_size, 151936);
    assert_eq!(cfg.layer_type(2), LayerType::LinearAttention);
    assert_eq!(cfg.layer_type(3), LayerType::FullAttention);
    assert_eq!(cfg.layer_type(47), LayerType::FullAttention);
    assert_eq!(cfg.linear_num_key_heads, 16);
    assert_eq!(cfg.linear_key_head_dim, 128);
    assert_eq!(cfg.linear_conv_kernel_dim, 4);
    assert_eq!(cfg.ssm_qkvz_size(), 2048 + 2048 + 4096 + 4096);
    assert_eq!(cfg.ssm_ba_size(), 64);
}

#[test]
fn qwen3_next_fixture_parses_layout_routing_and_quantization() {
    let json = include_str!(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../test_data/qwen3_config.json"
    ));
    let cfg = parse_config(json).unwrap();
    assert_eq!(cfg.hidden_size, 2048);
    assert_eq!(cfg.num_experts, 512);
    assert_eq!(cfg.num_experts_per_tok, 10);
    assert_eq!(cfg.num_hidden_layers, 48);
    assert_eq!(cfg.layer_types.len(), 48);
    assert_eq!(cfg.layer_types[0], LayerType::LinearAttention);
    assert_eq!(cfg.layer_types[3], LayerType::FullAttention);
    assert_eq!(cfg.vocab_size, 151936);
    assert_eq!(cfg.rope_theta, 10_000_000.0);
    assert!(cfg.norm_topk_prob);
    assert!(!cfg.tie_word_embeddings);
    assert_eq!(cfg.rms_norm_eps, 1e-6);
    assert_eq!(cfg.partial_rotary_factor, 0.25);
    assert_eq!(cfg.model_type, "qwen3_next");
    assert!(cfg.weight_prefix.is_empty());

    let quant = cfg
        .quantization_config
        .expect("NVIDIA NVFP4 fixture must retain quantization metadata");
    assert_eq!(quant.quant_method, "modelopt");
    assert_eq!(quant.quant_algo, "NVFP4");
    assert!(
        quant
            .ignore_modules
            .iter()
            .any(|module| module == "lm_head")
    );
}

#[test]
fn qwen35_moe_nested_config_maps_attention_ssm_and_layout_controls() {
    let json = r#"{
        "model_type": "qwen3_5_moe",
        "text_config": {
            "model_type": "qwen3_5_moe_text",
            "hidden_size": 2048,
            "num_hidden_layers": 40,
            "num_attention_heads": 16,
            "num_key_value_heads": 2,
            "head_dim": 256,
            "partial_rotary_factor": 0.25,
            "linear_num_key_heads": 16,
            "linear_key_head_dim": 128,
            "linear_num_value_heads": 32,
            "linear_value_head_dim": 128,
            "linear_conv_kernel_dim": 4,
            "num_experts": 256,
            "num_experts_per_tok": 8,
            "moe_intermediate_size": 512,
            "shared_expert_intermediate_size": 512,
            "vocab_size": 248320,
            "eos_token_id": 248044,
            "full_attention_interval": 4,
            "layer_types": [
                "linear_attention", "linear_attention", "linear_attention", "full_attention",
                "linear_attention", "linear_attention", "linear_attention", "full_attention",
                "linear_attention", "linear_attention", "linear_attention", "full_attention",
                "linear_attention", "linear_attention", "linear_attention", "full_attention",
                "linear_attention", "linear_attention", "linear_attention", "full_attention",
                "linear_attention", "linear_attention", "linear_attention", "full_attention",
                "linear_attention", "linear_attention", "linear_attention", "full_attention",
                "linear_attention", "linear_attention", "linear_attention", "full_attention",
                "linear_attention", "linear_attention", "linear_attention", "full_attention",
                "linear_attention", "linear_attention", "linear_attention", "full_attention"
            ],
            "rope_parameters": {
                "rope_theta": 10000000,
                "rope_type": "default"
            },
            "mtp_num_hidden_layers": 1
        }
    }"#;
    let cfg = parse_config(json).unwrap();
    assert_eq!(cfg.model_type, "qwen3_5_moe");
    assert_eq!(cfg.hidden_size, 2048);
    assert_eq!(cfg.num_hidden_layers, 40);
    assert_eq!(cfg.num_experts, 256);
    assert_eq!(cfg.num_experts_per_tok, 8);
    assert_eq!(cfg.vocab_size, 248320);
    assert_eq!(cfg.num_attention_layers(), 10);
    assert_eq!(cfg.num_ssm_layers(), 30);
    assert_eq!(cfg.layer_types.len(), 40);
    assert_eq!(cfg.eos_token_id, 248044);
    assert_eq!(cfg.rope_theta, 10_000_000.0);
    assert!(cfg.is_qwen35());
    assert!(cfg.nested_config);
    assert!(cfg.attn_gated);
    assert!(cfg.weight_prefix.is_empty());
    assert!(cfg.norm_topk_prob);
    assert_eq!(cfg.partial_rotary_factor, 0.25);
    assert_eq!(cfg.rotary_dim(), 64);
    assert_eq!(cfg.linear_conv_kernel_dim, 4);
    assert_eq!(cfg.ssm_qkv_size(), 2048 + 2048 + 4096);
    assert_eq!(cfg.ssm_z_size(), 4096);
    assert_eq!(cfg.mtp_num_hidden_layers, 1);
}

#[test]
fn qwen3_vl_moe_config_maps_attention_and_routing_controls() {
    let json = r#"{
        "model_type": "qwen3_vl_moe",
        "text_config": {
            "model_type": "qwen3_vl_moe_text",
            "hidden_size": 2048,
            "num_hidden_layers": 48,
            "num_attention_heads": 32,
            "num_key_value_heads": 4,
            "head_dim": 128,
            "num_experts": 128,
            "num_experts_per_tok": 8,
            "moe_intermediate_size": 768,
            "vocab_size": 151936,
            "rope_theta": 5000000,
            "norm_topk_prob": true
        }
    }"#;
    let cfg = parse_config(json).unwrap();
    assert_eq!(cfg.model_type, "qwen3_vl_moe");
    assert!(cfg.is_qwen3_vl());
    assert!(!cfg.is_qwen35());
    assert!(cfg.capabilities().has_nested_config);
    assert_eq!(cfg.hidden_size, 2048);
    assert_eq!(cfg.head_dim, 128);
    assert_eq!(cfg.num_attention_heads, 32);
    assert_eq!(cfg.num_key_value_heads, 4);
    assert_eq!(cfg.num_experts, 128);
    assert_eq!(cfg.num_experts_per_tok, 8);
    assert_eq!(cfg.moe_intermediate_size, 768);
    assert_eq!(cfg.num_hidden_layers, 48);
    assert!(!cfg.attn_gated);
    assert_eq!(cfg.num_attention_layers(), 48);
    assert_eq!(cfg.num_ssm_layers(), 0);
    assert_eq!(cfg.gqa_ratio(), 8);
    assert_eq!(cfg.rotary_dim(), 128);
    assert_eq!(cfg.rope_theta, 5_000_000.0);
    assert!(cfg.norm_topk_prob);
}

/// 2026-09-26: A `qwen3_5` config with a `vision_config` block is a vision-language model:
/// `is_qwen3_vl()` tells it from the text-only variant by the parsed `config.vision`.
#[test]
fn test_parse_qwen3_5_vl_config() {
    let json = r#"{
        "model_type": "qwen3_5",
        "architectures": ["Qwen3_5ForConditionalGeneration"],
        "text_config": {
            "model_type": "qwen3_5",
            "hidden_size": 2560,
            "num_hidden_layers": 32,
            "num_attention_heads": 16,
            "num_key_value_heads": 4,
            "head_dim": 256,
            "intermediate_size": 9216,
            "vocab_size": 248320,
            "rope_theta": 10000000.0
        },
        "vision_config": {
            "hidden_size": 1024,
            "num_hidden_layers": 27,
            "num_attention_heads": 16,
            "intermediate_size": 4096,
            "patch_size": 16,
            "spatial_merge_size": 2
        }
    }"#;
    let cfg = parse_config(json).unwrap();
    assert_eq!(cfg.model_type, "qwen3_5");
    assert!(
        cfg.is_qwen3_vl(),
        "Qwen3.5-VL detected via model_type=qwen3_5 + vision_config presence"
    );
    assert!(cfg.vision.is_some());
}

/// 2026-09-26: A `qwen3_5` config without `vision_config` is not vision-language.
#[test]
fn test_qwen3_5_text_only_not_vl() {
    let json = r#"{
        "model_type": "qwen3_5",
        "text_config": {
            "model_type": "qwen3_5",
            "hidden_size": 2560,
            "num_hidden_layers": 32,
            "num_attention_heads": 16,
            "num_key_value_heads": 4,
            "head_dim": 256,
            "vocab_size": 151936,
            "rope_theta": 10000000.0
        }
    }"#;
    let cfg = parse_config(json).unwrap();
    assert_eq!(cfg.model_type, "qwen3_5");
    assert!(
        !cfg.is_qwen3_vl(),
        "qwen3_5 without vision_config must not be classified as VL"
    );
}

/// 2026-09-26: A dense `qwen3_5` config with MRoPE, like the local
/// Kbenkhaled/Qwen3.5-27B-NVFP4 config, keeps `model_type = "qwen3_5"`; `parse_config`
/// renames only MoE configs with MRoPE to `qwen3_6_moe`. The MRoPE fields are still read.
#[test]
fn test_kbenkhaled_qwen35_27b_dense_mrope_no_rewrite() {
    let json = r#"{
        "model_type": "qwen3_5",
        "text_config": {
            "model_type": "qwen3_5_text",
            "hidden_size": 5120,
            "num_hidden_layers": 64,
            "num_attention_heads": 24,
            "num_key_value_heads": 4,
            "head_dim": 256,
            "intermediate_size": 17408,
            "partial_rotary_factor": 0.25,
            "linear_num_key_heads": 16,
            "linear_key_head_dim": 128,
            "linear_num_value_heads": 32,
            "linear_value_head_dim": 128,
            "linear_conv_kernel_dim": 4,
            "vocab_size": 248320,
            "eos_token_id": 248044,
            "full_attention_interval": 4,
            "rope_parameters": {
                "rope_theta": 10000000,
                "rope_type": "default",
                "mrope_interleaved": true,
                "mrope_section": [11, 11, 10]
            }
        }
    }"#;
    let cfg = parse_config(json).unwrap();
    assert_eq!(cfg.model_type, "qwen3_5");
    assert_eq!(cfg.hidden_size, 5120);
    assert_eq!(cfg.num_experts, 0);
    assert!(cfg.mrope_interleaved);
    assert_eq!(cfg.mrope_section, [11, 11, 10]);
}

#[test]
fn test_layer_prefix() {
    let cfg80b = ModelConfig::qwen3_next_80b_nvfp4();
    assert_eq!(cfg80b.layer_prefix(3), "model.layers.3");

    let mut cfg35 = ModelConfig::qwen3_next_80b_nvfp4();
    cfg35.weight_prefix = "model.language_model".to_string();
    assert_eq!(cfg35.layer_prefix(3), "model.language_model.layers.3");
}

#[test]
fn expert_parallelism_partitions_experts_without_dropping_remainder() {
    let mut cfg = ModelConfig::qwen3_next_80b_nvfp4();
    assert_eq!(cfg.local_expert_range(), (0, 512));
    assert!(cfg.is_local_expert(0));
    assert!(cfg.is_local_expert(511));

    cfg.ep_rank = 0;
    cfg.ep_world_size = 2;
    assert_eq!(cfg.local_expert_range(), (0, 256));
    assert!(cfg.is_local_expert(0));
    assert!(cfg.is_local_expert(255));
    assert!(!cfg.is_local_expert(256));
    assert!(!cfg.is_local_expert(511));

    cfg.ep_rank = 1;
    assert_eq!(cfg.local_expert_range(), (256, 512));
    assert!(!cfg.is_local_expert(0));
    assert!(!cfg.is_local_expert(255));
    assert!(cfg.is_local_expert(256));
    assert!(cfg.is_local_expert(511));

    cfg.num_experts = 513;
    assert_eq!(cfg.local_expert_range(), (256, 513));
    assert!(cfg.is_local_expert(512));
}

#[test]
fn test_tensor_parallelism_range() {
    let mut cfg = ModelConfig::qwen3_next_80b_nvfp4();

    assert_eq!(cfg.tp_shard_range(2048), (0, 2048));
    assert_eq!(cfg.tp_shard_dim(2048), 2048);

    cfg.tp_world_size = 2;
    cfg.tp_rank = 0;
    assert_eq!(cfg.tp_shard_range(2048), (0, 1024));
    assert_eq!(cfg.tp_shard_dim(2048), 1024);

    cfg.tp_rank = 1;
    assert_eq!(cfg.tp_shard_range(2048), (1024, 2048));
    assert_eq!(cfg.tp_shard_dim(2048), 1024);

    cfg.tp_world_size = 4;
    cfg.tp_rank = 2;
    assert_eq!(cfg.tp_shard_range(2048), (1024, 1536));
}

#[test]
fn test_num_attention_layers_counts_sliding_attention() {
    // 2026-09-26: Sliding layers hold paged KV cache like full-attention layers, so
    // `num_attention_layers()` counts both. The fixture has 45 layers with every 4th full
    // (12 full, 33 sliding), the counts in `kernels/gb10/step3p7-flash/MODEL.toml`.
    let mut layer_types = vec![LayerType::SlidingAttention; 45];
    for full in (0..45).step_by(4) {
        layer_types[full] = LayerType::FullAttention;
    }
    let mut cfg = ModelConfig::qwen3_next_80b_nvfp4();
    cfg.num_hidden_layers = 45;
    cfg.layer_types = layer_types;

    assert_eq!(cfg.num_attention_layers(), 45);
    assert_eq!(cfg.num_ssm_layers(), 0);
    assert!(!cfg.has_recurrent_state());
    assert_eq!(cfg.layer_type(0), LayerType::FullAttention);
    assert_eq!(cfg.layer_type(1), LayerType::SlidingAttention);
    assert_eq!(cfg.layer_type(43), LayerType::SlidingAttention);
    assert_eq!(cfg.layer_type(44), LayerType::FullAttention);
}

/// 2026-09-26: Sparse-attention layers hold paged KV cache and count as attention layers.
/// The fixture has GLM-5.3-Flash's pattern: 34 linear and 11 sparse layers.
#[test]
fn sparse_attention_layers_are_counted_as_attention() {
    use crate::LayerType::{LinearAttention, SparseAttention};
    let mut cfg = ModelConfig::qwen3_next_80b_nvfp4();
    cfg.num_hidden_layers = 45;
    cfg.layer_types = (0..45)
        .map(|i| {
            if i % 4 == 3 {
                SparseAttention
            } else {
                LinearAttention
            }
        })
        .collect();
    assert_eq!(
        cfg.layer_types
            .iter()
            .filter(|t| **t == SparseAttention)
            .count(),
        11
    );
    assert_eq!(cfg.num_attention_layers(), 11);
    assert_eq!(cfg.num_ssm_layers(), 34);
}
