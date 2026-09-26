// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-26: Tests for the `qwen4_exp` parser over a fixture with the values of the local
//! Qwen3.8-Flash-Next `config.json`.
//!
//! Owner: config (model parsers).
//! Invariants: none beyond the types.

use super::*;
use crate::LayerType;

/// 2026-09-26: The fixture, parsed.
fn raw_config() -> Value {
    serde_json::from_str(RAW).expect("fixture must be valid JSON")
}

/// 2026-09-26: The keys the parser reads, with the local Qwen3.8-Flash-Next config's values,
/// cut to 4 layers. A raw string rather than `serde_json::json!`, whose macro recursion
/// limit a literal this deep exceeds.
const RAW: &str = r#"{
  "architectures": ["Qwen4ExpForConditionalGeneration"],
  "model_type": "qwen4_exp",
  "text_config": {
    "model_type": "qwen4_exp_text",
    "hidden_size": 2560,
    "num_hidden_layers": 4,
    "num_attention_heads": 24,
    "num_key_value_heads": 2,
    "head_dim": 256,
    "vocab_size": 248320,
    "eos_token_id": 248044,
    "rms_norm_eps": 1e-06,
    "max_position_embeddings": 262144,
    "num_experts": 512,
    "num_experts_per_tok": 10,
    "moe_intermediate_size": 640,
    "shared_expert_intermediate_size": 640,
    "full_attention_interval": 4,
    "layer_types": ["linear_attention", "linear_attention",
                    "linear_attention", "full_attention"],
    "linear_num_key_heads": 16,
    "linear_num_value_heads": 48,
    "linear_key_head_dim": 128,
    "linear_value_head_dim": 128,
    "linear_conv_kernel_dim": 4,
    "hc_count": 4,
    "hc_lowrank": 320,
    "indexer_budget": 2048,
    "indexer_compress_ratio": 4,
    "indexer_head_dim": 128,
    "indexer_n_heads": 4,
    "indexer_kv_heads": 1,
    "ngram_size": 3,
    "ngram_vocab_size_base": 20000000,
    "heads_per_ngram": 8,
    "split_ngram_parts": 128,
    "make_ngram_vocab_size_divisible_by": 128,
    "ple_layer_ids": [2],
    "ple_embed_dim": 2560,
    "ple_conv_kernel_size": 4,
    "output_gate_type": "sigmoid",
    "partial_rotary_factor": 0.25,
    "rope_parameters": {
      "mrope_interleaved": true,
      "mrope_section": [11, 11, 10],
      "partial_rotary_factor": 0.25,
      "rope_theta": 10000000,
      "rope_type": "default"
    }
  },
  "vision_config": {
    "depth": 27,
    "hidden_size": 1152,
    "intermediate_size": 4304,
    "num_heads": 16,
    "out_hidden_size": 2560,
    "patch_size": 16,
    "spatial_merge_size": 2,
    "temporal_patch_size": 2
  }
}"#;

#[test]
fn parses_the_shipped_checkpoint_config() {
    let c = parse_qwen4_exp(&raw_config()).expect("parse");
    assert_eq!(c.model_type, "qwen4_exp", "top-level type, not *_text");
    assert!(c.nested_config);
    assert_eq!(c.hidden_size, 2560);
    assert_eq!(c.vocab_size, 248320);
    assert_eq!(c.head_dim, 256);
    assert_eq!(c.num_experts, 512);
    assert_eq!(c.num_experts_per_tok, 10);
    assert_eq!(c.moe_intermediate_size, 640);
    assert_eq!(c.shared_expert_intermediate_size, 640);
}

#[test]
fn hyper_connections_map_onto_the_deepseek_v4_fields() {
    let c = parse_qwen4_exp(&raw_config()).expect("parse");
    assert_eq!(c.hc_mult, 4, "hc_count -> hc_mult");
    assert_eq!(c.hc_lowrank, 320, "selects the low-rank mixer variant");
}

/// 2026-09-26: `hc_count` without `hc_lowrank` is refused.
#[test]
fn partial_hyper_connection_config_is_refused() {
    let mut raw = raw_config();
    raw["text_config"]["hc_lowrank"] = Value::from(0);
    let err = parse_qwen4_exp(&raw).unwrap_err().to_string();
    assert!(err.contains("hyper-connection config is partial"), "{err}");
}

#[test]
fn indexer_maps_onto_index_fields_and_only_full_attention_layers() {
    let c = parse_qwen4_exp(&raw_config()).expect("parse");
    assert_eq!(c.index_n_heads, 4);
    assert_eq!(c.index_head_dim, 128);
    assert_eq!(c.index_topk, 2048, "indexer_budget -> index_topk");
    assert_eq!(c.index_compress_ratio, 4);
    // 2026-09-26: `compress_ratios` stays empty; see `parse_indexer`.
    assert!(
        c.compress_ratios.is_empty(),
        "must not dispatch the V4 compressor for Qwen's indexer"
    );
}

#[test]
fn partial_indexer_config_is_refused() {
    let mut raw = raw_config();
    raw["text_config"]["indexer_head_dim"] = Value::from(0);
    let err = parse_qwen4_exp(&raw).unwrap_err().to_string();
    assert!(err.contains("indexer config is partial"), "{err}");
}

#[test]
fn ngram_geometry_matches_the_longcat_formula() {
    let c = parse_qwen4_exp(&raw_config()).expect("parse");
    assert_eq!(c.emb_neighbor_num, 3, "ngram_size");
    assert_eq!(c.emb_split_num, 8, "heads_per_ngram");
    let heads = c.emb_split_num * (c.emb_neighbor_num - 1);
    assert_eq!(heads, 16);
    assert_eq!(c.hidden_size / heads, 160);
    assert_eq!(c.ngram_vocab_size_base, 20_000_000);
    assert_eq!(c.ngram_split_parts, 128);
    assert_eq!(c.ple_layer_ids, vec![2]);
    assert_eq!(c.ple_conv_kernel_size, 4);
}

/// 2026-09-26: An n-gram head count that does not divide `hidden_size` is refused.
#[test]
fn ngram_head_count_must_divide_hidden_size() {
    let mut raw = raw_config();
    raw["text_config"]["heads_per_ngram"] = Value::from(7);
    let err = parse_qwen4_exp(&raw).unwrap_err().to_string();
    assert!(err.contains("must divide evenly"), "{err}");
    assert!(err.contains("concatenated, not projected"), "{err}");
}

#[test]
fn ple_layer_id_past_the_end_is_refused() {
    let mut raw = raw_config();
    raw["text_config"]["ple_layer_ids"] = Value::from(vec![99u64]);
    let err = parse_qwen4_exp(&raw).unwrap_err().to_string();
    assert!(err.contains("only 4 layers"), "{err}");
}

#[test]
fn rope_reads_through_the_nested_rope_parameters() {
    let c = parse_qwen4_exp(&raw_config()).expect("parse");
    assert_eq!(c.rope_theta, 10_000_000.0);
    assert_eq!(c.partial_rotary_factor, 0.25);
    assert!(c.mrope_interleaved);
    assert_eq!(c.mrope_section, [11, 11, 10]);
}

#[test]
fn attention_is_gated_and_layer_types_survive() {
    let c = parse_qwen4_exp(&raw_config()).expect("parse");
    assert!(c.attn_gated, "output_gate_type: sigmoid");
    assert_eq!(c.layer_types.len(), 4);
    assert_eq!(c.layer_types[0], LayerType::LinearAttention);
    assert_eq!(c.layer_types[3], LayerType::FullAttention);
}

#[test]
fn layer_types_length_must_match_num_hidden_layers() {
    let mut raw = raw_config();
    raw["text_config"]["num_hidden_layers"] = Value::from(48);
    let err = parse_qwen4_exp(&raw).unwrap_err().to_string();
    assert!(err.contains("layer_types has 4 entries"), "{err}");
}

#[test]
fn missing_text_config_is_refused() {
    let err = parse_qwen4_exp(&serde_json::json!({"model_type": "qwen4_exp"}))
        .unwrap_err()
        .to_string();
    assert!(err.contains("missing text_config"), "{err}");
}

#[test]
fn vision_tower_is_parsed() {
    let c = parse_qwen4_exp(&raw_config()).expect("parse");
    assert!(c.vision.is_some(), "vision_config sits at the TOP level");
}

/// 2026-09-26: Parses the `config.json` named by `METRALE_QWEN4_EXP_CONFIG`, and passes
/// without checking anything when the variable is unset:
///
///     METRALE_QWEN4_EXP_CONFIG=/path/to/snapshot/config.json \
///       cargo test -p metrale-config --lib qwen4_exp
#[test]
fn parses_the_real_config_json_when_present() {
    let Ok(path) = std::env::var("METRALE_QWEN4_EXP_CONFIG") else {
        eprintln!("METRALE_QWEN4_EXP_CONFIG unset — skipping real-checkpoint parse");
        return;
    };
    let text = std::fs::read_to_string(&path).expect("read config.json");
    let raw: Value = serde_json::from_str(&text).expect("config.json is valid JSON");
    let c = parse_qwen4_exp(&raw).expect("real config.json must parse");

    // 2026-09-26: The values of the local Qwen3.8-Flash-Next config.
    assert_eq!(c.num_hidden_layers, 48);
    assert_eq!(c.layer_types.len(), 48);
    assert_eq!(
        c.layer_types
            .iter()
            .filter(|t| **t == LayerType::FullAttention)
            .count(),
        12,
        "3 GDN : 1 full over 48 layers"
    );
    assert_eq!(c.hidden_size, 2560);
    assert_eq!(c.num_experts, 512);
    assert_eq!(c.hc_mult, 4);
    assert_eq!(c.hc_lowrank, 320);
    assert_eq!(c.index_topk, 2048);
    assert_eq!(
        c.emb_split_num * (c.emb_neighbor_num - 1),
        16,
        "n-gram heads"
    );
    assert_eq!(c.ple_layer_ids, vec![2]);
    assert!(c.vision.is_some());
}

/// 2026-09-26: Both top-level spellings, `qwen4_exp` and `qwen3_8_flash_next`, reach this
/// parser through `parse_config` and become `qwen4_exp`.
#[test]
fn both_naming_revisions_route_to_this_parser() {
    for (model_type, arch) in [
        ("qwen4_exp", "Qwen4ExpForConditionalGeneration"),
        (
            "qwen3_8_flash_next",
            "Qwen3_8FlashNextForConditionalGeneration",
        ),
    ] {
        let mut raw = raw_config();
        raw["model_type"] = Value::from(model_type);
        raw["architectures"] = Value::from(vec![arch]);
        raw["text_config"]["model_type"] = Value::from(format!("{model_type}_text"));

        let c = crate::parse_config(&raw.to_string())
            .unwrap_or_else(|e| panic!("{model_type} must parse: {e:#}"));
        assert_eq!(c.model_type, "qwen4_exp", "{model_type} normalizes");
        assert_eq!(c.hc_mult, 4);
        assert_eq!(c.num_experts, 512);
    }
}
