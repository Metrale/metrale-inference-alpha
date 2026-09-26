// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-26: Tests for `parse_glm5_next` over a GLM-5.3-Flash `config.json` fixture.
//!
//! Owner: config (model parsers).
//! Invariants: none beyond the types.

use super::*;

/// 2026-09-26: A GLM-5.3-Flash `config.json` fixture: 45 text layers, KDA except for the
/// sparse-attention layers at `i % 4 == 3`, listed explicitly in both `linear_attn_config`
/// index lists and `layer_types`, plus one MTP layer.
fn glm53_config_json() -> String {
    let kda: Vec<String> = (0..45)
        .filter(|i| i % 4 != 3)
        .map(|i| i.to_string())
        .collect();
    let full: Vec<String> = (0..45)
        .filter(|i| i % 4 == 3)
        .map(|i| i.to_string())
        .collect();
    let layer_types: Vec<String> = (0..45)
        .map(|i| {
            if i % 4 == 3 {
                "\"deepseek_sparse_attention\"".to_string()
            } else {
                "\"linear_attention\"".to_string()
            }
        })
        .collect();
    format!(
        r#"{{
  "architectures": ["Glm5NextForConditionalGeneration"],
  "model_type": "glm5_next",
  "text_config": {{
    "model_type": "glm5_next_text",
    "num_hidden_layers": 45,
    "num_nextn_predict_layers": 1,
    "hidden_size": 4096,
    "intermediate_size": 12288,
    "num_attention_heads": 64,
    "num_key_value_heads": 64,
    "head_dim": 0,
    "qk_head_dim": 256,
    "qk_nope_head_dim": 256,
    "qk_rope_head_dim": 0,
    "v_head_dim": 256,
    "kv_lora_rank": 512,
    "q_lora_rank": 1536,
    "mla_use_nope": true,
    "index_topk": 2048,
    "index_kpool": 4,
    "index_n_heads": 32,
    "index_head_dim": 128,
    "hc_mult": 4,
    "hc_sinkhorn_iters": 20,
    "hc_eps": 1e-06,
    "mhc": true,
    "n_routed_experts": 288,
    "n_shared_experts": 1,
    "num_experts_per_tok": 8,
    "moe_intermediate_size": 2048,
    "first_k_dense_replace": 3,
    "swiglu_limit": 10.0,
    "routed_scaling_factor": 2.5,
    "norm_topk_prob": true,
    "n_group": 1,
    "topk_group": 1,
    "scoring_func": "sigmoid",
    "topk_method": "noaux_tc",
    "rms_norm_eps": 1e-05,
    "vocab_size": 154880,
    "max_position_embeddings": 1048576,
    "linear_attn_config": {{
      "num_heads": 64,
      "head_dim": 128,
      "short_conv_kernel_size": 4,
      "gate_lower_bound": -5.0,
      "kda_layers": [{kda}],
      "full_attn_layers": [{full}]
    }},
    "layer_types": [{lt}]
  }}
}}"#,
        kda = kda.join(","),
        full = full.join(","),
        lt = layer_types.join(",")
    )
}

/// 2026-09-26: The fixture with one `text_config` key set to `value`.
fn with_text_key(key: &str, value: serde_json::Value) -> String {
    let mut raw: serde_json::Value =
        serde_json::from_str(&glm53_config_json()).expect("fixture json");
    raw["text_config"][key] = value;
    raw.to_string()
}

/// 2026-09-26: An explicit `indexer_types` array of 45 `"full"` entries parses.
#[test]
fn an_all_full_indexer_array_is_accepted() {
    let all_full = serde_json::Value::from(vec!["full"; 45]);
    let c = parse_glm5_next(&with_text_key("indexer_types", all_full)).expect("parse");
    let dsa: Vec<usize> = c
        .layer_types
        .iter()
        .enumerate()
        .filter(|(_, t)| **t != LayerType::LinearAttention)
        .map(|(i, _)| i)
        .collect();
    assert_eq!(dsa, vec![3, 7, 11, 15, 19, 23, 27, 31, 35, 39, 43]);
}

/// 2026-09-26: A DSA layer marked `"shared"` is refused, and the error names the layer.
#[test]
fn a_shared_dsa_layer_is_refused() {
    let mut modes = vec!["full"; 45];
    modes[7] = "shared";
    let e = parse_glm5_next(&with_text_key("indexer_types", modes.into()))
        .expect_err("shared DSA indexing must be refused");
    let msg = e.to_string() + &e.root_cause().to_string();
    assert!(msg.contains('7'), "the error must name the layer: {msg}");
}

/// 2026-09-26: A `"shared"` entry on a KDA layer is ignored: KDA layers have no indexer.
#[test]
fn a_shared_entry_on_a_linear_layer_is_inert() {
    let mut modes = vec!["full"; 45];
    modes[0] = "shared";
    assert!(parse_glm5_next(&with_text_key("indexer_types", modes.into())).is_ok());
}

/// 2026-09-26: An `indexer_types` array whose length is not `num_hidden_layers` is refused.
#[test]
fn a_wrong_length_indexer_array_is_refused() {
    let short = serde_json::Value::from(vec!["full"; 44]);
    assert!(parse_glm5_next(&with_text_key("indexer_types", short)).is_err());
}

/// 2026-09-26: Without `indexer_types` the modes are derived. The default `freq = 1` makes
/// every layer full, so the bare fixture parses; `freq = 4` makes DSA layers shared, which
/// is refused.
#[test]
fn an_absent_array_is_derived_not_assumed_full() {
    assert!(parse_glm5_next(&glm53_config_json()).is_ok());
    // 2026-09-26: freq=4, offset=2: full only where max(i-1, 0) % 4 == 0, so DSA layer 7 is
    // shared.
    let e = parse_glm5_next(&with_text_key("index_topk_freq", 4.into()))
        .expect_err("a freq schedule that shares DSA layers must be refused");
    assert!(e.root_cause().to_string().contains("SHARED"), "{e}");
}

/// 2026-09-26: Without `indexer_types`, an `index_topk_pattern` string (one `F` or `S` per
/// layer) decides the modes.
#[test]
fn an_index_topk_pattern_is_honoured() {
    let ok: String = "F".repeat(45);
    assert!(parse_glm5_next(&with_text_key("index_topk_pattern", ok.into())).is_ok());
    let mut bad: Vec<char> = "F".repeat(45).chars().collect();
    bad[43] = 'S';
    let bad: String = bad.into_iter().collect();
    assert!(parse_glm5_next(&with_text_key("index_topk_pattern", bad.into())).is_err());
}

#[test]
fn parses_glm5_next() {
    let c = parse_glm5_next(&glm53_config_json()).expect("parse");
    assert_eq!(c.model_type, "glm5_next");
    assert_eq!(c.num_hidden_layers, 45);
    assert_eq!(c.hidden_size, 4096);
    assert_eq!(c.n_routed_experts, 288);
    assert_eq!(c.num_experts_per_tok, 8);
    assert_eq!(c.moe_intermediate_size, 2048);
}

/// 2026-09-26: A zero `qk_rope_head_dim` is kept as read.
#[test]
fn nope_rope_dim_zero_survives_exactly() {
    let c = parse_glm5_next(&glm53_config_json()).expect("parse");
    assert_eq!(c.qk_rope_head_dim, 0, "NoPE zero must not be 'repaired'");
    assert_eq!(c.qk_nope_head_dim, 256, "nope dim must come from the file");
    assert_eq!(c.v_head_dim, 256);
    assert_eq!(c.partial_rotary_factor, 0.0);
}

/// 2026-09-26: GLM-5.3 is MLA (a 512-dim latent KV cache) and NoPE (rope dim 0) at once, so
/// a non-zero rope dim cannot stand in for "is MLA".
#[test]
fn is_mla_and_nope_simultaneously() {
    let c = parse_glm5_next(&glm53_config_json()).expect("parse");
    assert!(
        c.kv_lora_rank > 0,
        "GLM-5.3 is MLA: a latent KV cache of {} dims",
        c.kv_lora_rank
    );
    assert_eq!(c.kv_lora_rank, 512);
    assert_eq!(
        c.qk_rope_head_dim, 0,
        "...and simultaneously NoPE. `rope > 0` must never stand in for `is MLA`."
    );
}

/// 2026-09-26: The fixture's `head_dim: 0` resolves to `qk_head_dim` (256), not to
/// `hidden_size / num_attention_heads` (64).
#[test]
fn head_dim_resolves_to_mla_width_not_hidden_over_heads() {
    let c = parse_glm5_next(&glm53_config_json()).expect("parse");
    assert_eq!(c.head_dim, 256);
    assert_ne!(c.head_dim, 4096 / 64);
}

/// 2026-09-26: The fixture's text stack (layers 0..=44) has 34 KDA layers, 11 sparse-attention
/// layers and no plain full-attention layer; the MTP layer is not counted.
#[test]
fn layer_census_matches_reconciled_counts() {
    let c = parse_glm5_next(&glm53_config_json()).expect("parse");
    let kda = c
        .layer_types
        .iter()
        .filter(|t| **t == LayerType::LinearAttention)
        .count();
    let dsa = c
        .layer_types
        .iter()
        .filter(|t| **t == LayerType::SparseAttention)
        .count();
    let plain_full = c
        .layer_types
        .iter()
        .filter(|t| **t == LayerType::FullAttention)
        .count();
    assert_eq!(c.layer_types.len(), 45);
    assert_eq!(kda, 34, "KDA layers over text layers 0..44");
    assert_eq!(dsa, 11, "DSA layers over text layers 0..44");
    assert_eq!(plain_full, 0, "GLM-5.3 has no plain full-attention layer");
    assert_eq!(c.layer_types[0], LayerType::LinearAttention);
    assert_eq!(c.layer_types[3], LayerType::SparseAttention);
    assert_eq!(c.layer_types[43], LayerType::SparseAttention);
    assert_eq!(c.layer_types[44], LayerType::LinearAttention);
}

/// 2026-09-26: `hf_name()` of each parsed layer type gives back the fixture's `layer_types`
/// string.
#[test]
fn layer_types_round_trip_to_the_checkpoint_strings() {
    let c = parse_glm5_next(&glm53_config_json()).expect("parse");
    let raw: serde_json::Value = serde_json::from_str(&glm53_config_json()).unwrap();
    let want = raw["text_config"]["layer_types"]
        .as_array()
        .expect("layer_types");
    assert_eq!(want.len(), c.layer_types.len());
    for (i, w) in want.iter().enumerate() {
        assert_eq!(
            c.layer_types[i].hf_name(),
            w.as_str().unwrap(),
            "layer {i} does not round-trip"
        );
    }
}

/// 2026-09-26: The MTP layer (index 45) resolves through `mtp_layer_types` and is not
/// appended to `layer_types`.
#[test]
fn mtp_layer_is_represented_outside_the_text_stack() {
    let c = parse_glm5_next(&glm53_config_json()).expect("parse");
    assert_eq!(c.num_hidden_layers, 45);
    assert_eq!(c.layer_types.len(), 45, "text stack stays 0..=44");
    assert_eq!(c.mtp_layer_types, vec![LayerType::SparseAttention]);
    assert_eq!(c.layer_type_at(45), Some(LayerType::SparseAttention));
    assert_eq!(c.layer_type_at(46), None);
    assert!(c.has_sparse_attention());
    assert_eq!(c.sparse_attention_layers().len(), 11, "text stack only");
}

#[test]
fn kda_geometry_from_linear_attn_config() {
    let c = parse_glm5_next(&glm53_config_json()).expect("parse");
    assert_eq!(c.linear_num_key_heads, 64);
    assert_eq!(c.linear_key_head_dim, 128);
    assert_eq!(c.linear_conv_kernel_dim, 4);
}

#[test]
fn indexer_topk_is_2048_not_the_deepseek_default() {
    let c = parse_glm5_next(&glm53_config_json()).expect("parse");
    assert_eq!(c.index_topk, 2048);
}

#[test]
fn missing_rope_key_is_refused_not_guessed() {
    let mut v: serde_json::Value = serde_json::from_str(&glm53_config_json()).unwrap();
    v["text_config"]
        .as_object_mut()
        .unwrap()
        .remove("qk_rope_head_dim");
    let err = parse_glm5_next(&v.to_string()).unwrap_err();
    assert!(
        err.to_string().contains("refusing to guess"),
        "unexpected error: {err}"
    );
}

/// 2026-09-26: A config without `swiglu_limit` is refused, not read as "no clamp".
#[test]
fn a_missing_swiglu_limit_is_refused_not_defaulted() {
    let mut v: serde_json::Value = serde_json::from_str(&glm53_config_json()).unwrap();
    v["text_config"]
        .as_object_mut()
        .unwrap()
        .remove("swiglu_limit");
    let err = parse_glm5_next(&v.to_string()).unwrap_err();
    assert!(
        err.to_string().contains("swiglu_limit"),
        "unexpected error: {err}"
    );
}

#[test]
fn the_swiglu_limit_is_read_verbatim() {
    let c = parse_glm5_next(&glm53_config_json()).unwrap();
    assert_eq!(c.swiglu_limit, 10.0);
}

/// 2026-09-26: `n_group` or `topk_group` other than 1 is refused at parse time, because
/// `glm5next_router_topk` does not implement grouped routing.
#[test]
fn grouped_expert_routing_is_refused() {
    for key in ["n_group", "topk_group"] {
        let mut v: serde_json::Value = serde_json::from_str(&glm53_config_json()).unwrap();
        v["text_config"][key] = serde_json::json!(8);
        let err = parse_glm5_next(&v.to_string()).unwrap_err();
        assert!(
            err.to_string().contains("Grouped expert routing"),
            "{key}: unexpected error: {err}"
        );
    }
}

#[test]
fn contradictory_layer_maps_are_rejected() {
    let mut v: serde_json::Value = serde_json::from_str(&glm53_config_json()).unwrap();
    // 2026-09-26: Layer 0 is KDA in the index lists and sparse attention in `layer_types`.
    v["text_config"]["layer_types"][0] =
        serde_json::Value::String("deepseek_sparse_attention".into());
    let err = parse_glm5_next(&v.to_string()).unwrap_err();
    assert!(err.to_string().contains("disagrees"), "unexpected: {err}");
}

/// 2026-09-26: An absent `moe_router_dtype` selects `Glm5NextRouterMode::HfFp32`.
#[test]
fn glm_router_defaults_to_hf_fp32_when_the_config_is_silent() {
    let cfg = parse_glm5_next(&glm53_config_json()).unwrap();
    assert_eq!(cfg.glm5next_router_mode, Glm5NextRouterMode::HfFp32);
    assert!(cfg.glm5next_router_mode.is_fp32());
}

/// 2026-09-26: `moe_router_dtype` selects `VllmBf16` for `"bfloat16"` and `HfFp32` for
/// `"float32"`.
#[test]
fn glm_router_bf16_compat_mode_is_explicit_and_never_inferred() {
    let base = glm53_config_json();
    let with = base.replace(
        r#""hc_mult": 4,"#,
        r#""hc_mult": 4, "moe_router_dtype": "bfloat16","#,
    );
    assert_ne!(with, base, "fixture anchor moved");
    let cfg = parse_glm5_next(&with).unwrap();
    assert_eq!(cfg.glm5next_router_mode, Glm5NextRouterMode::VllmBf16);
    assert!(!cfg.glm5next_router_mode.is_fp32());

    let fp32 = base.replace(
        r#""hc_mult": 4,"#,
        r#""hc_mult": 4, "moe_router_dtype": "float32","#,
    );
    assert_eq!(
        parse_glm5_next(&fp32).unwrap().glm5next_router_mode,
        Glm5NextRouterMode::HfFp32
    );
}

/// 2026-09-26: An unrecognised `moe_router_dtype` is refused.
#[test]
fn an_unknown_router_dtype_is_refused_not_defaulted() {
    let bad = glm53_config_json().replace(
        r#""hc_mult": 4,"#,
        r#""hc_mult": 4, "moe_router_dtype": "fp8","#,
    );
    assert!(parse_glm5_next(&bad).is_err());
}
