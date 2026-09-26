// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-26: Tests for parsing an NLLB / M2M-100 `config.json`.
//!
//! Owner: config (model parsers).
//! Invariants: none beyond the types.

use super::*;

#[test]
fn test_parse_nllb_m2m100_config() {
    let json = r#"{
        "activation_function": "relu",
        "architectures": ["M2M100ForConditionalGeneration"],
        "bos_token_id": 0,
        "d_model": 2048,
        "decoder_attention_heads": 16,
        "decoder_ffn_dim": 8192,
        "decoder_layers": 24,
        "encoder_attention_heads": 16,
        "encoder_ffn_dim": 8192,
        "encoder_layers": 24,
        "eos_token_id": 2,
        "is_encoder_decoder": true,
        "max_position_embeddings": 1024,
        "model_type": "m2m_100",
        "num_hidden_layers": 24,
        "pad_token_id": 1,
        "scale_embedding": true,
        "use_cache": true,
        "vocab_size": 256206
    }"#;

    let cfg = parse_config(json).unwrap();
    assert_eq!(cfg.model_type, "m2m_100");
    assert_eq!(cfg.hidden_size, 2048);
    assert_eq!(cfg.num_hidden_layers, 24);
    assert_eq!(cfg.intermediate_size, 8192);
    assert_eq!(cfg.num_attention_heads, 16);
    assert_eq!(cfg.num_key_value_heads, 16);
    assert_eq!(cfg.head_dim, 128);
    assert_eq!(cfg.max_position_embeddings, 1024);
    assert_eq!(cfg.vocab_size, 256206);
    assert_eq!(cfg.bos_token_id, 0);
    assert_eq!(cfg.eos_token_id, 2);
    assert!(cfg.tie_word_embeddings);
    assert_eq!(cfg.num_experts, 0);
    assert_eq!(cfg.mtp_num_hidden_layers, 0);
    assert_eq!(cfg.num_attention_layers(), 24);
    assert_eq!(cfg.num_ssm_layers(), 0);
    assert_eq!(cfg.weight_prefix, "model.decoder");
    assert!(!cfg.attn_gated);
}

#[test]
fn test_parse_nllb_rejects_missing_required_fields() {
    let json = r#"{
        "bos_token_id": 0,
        "d_model": 2048,
        "decoder_attention_heads": 16,
        "decoder_ffn_dim": 8192,
        "decoder_layers": 24,
        "eos_token_id": 2,
        "max_position_embeddings": 1024,
        "model_type": "nllb",
        "vocab_size": 256206
    }"#;

    let valid: serde_json::Value = serde_json::from_str(json).unwrap();
    for field in [
        "d_model",
        "decoder_layers",
        "decoder_ffn_dim",
        "vocab_size",
        "decoder_attention_heads",
        "max_position_embeddings",
        "bos_token_id",
        "eos_token_id",
    ] {
        let mut missing = valid.clone();
        missing.as_object_mut().unwrap().remove(field);
        let err = parse_config(&missing.to_string()).unwrap_err().to_string();
        assert!(
            err.contains(&format!("nllb config missing required field `{field}`")),
            "missing {field}: {err}"
        );
    }
}
