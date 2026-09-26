// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-26: `config.json` parser for LongCat-Flash n-gram models
//! (`longcat_flash_ngram`, `longcat_flash`, or `architectures:
//! ["LongcatFlashNgramForCausalLM"]`).
//!
//! The config's own key names are renamed onto `ModelConfig` fields when the standard name
//! is absent: `num_layers`, `n_routed_experts`, `moe_topk`, `expert_ffn_hidden_size` and
//! `ffn_hidden_size`. `num_hidden_layers` is then doubled: each checkpoint layer is two
//! engine sublayers, as in `bench/ngram_ref/modeling_longcat_ngram.py`, and the LongCat
//! loader walks `num_hidden_layers / 2` checkpoint layers. The MLA fields, the n-gram
//! fields and `zero_expert_num` deserialize under their own names.
//!
//! Owner: config (model parsers).
//! Invariants:
//! - A parse that succeeds has non-zero `hidden_size`, `num_hidden_layers` and `vocab_size`,
//!   and the three n-gram fields are either all zero or all set with
//!   `emb_neighbor_num >= 2` and `hidden_size` a multiple of the table count.

use anyhow::{Context, Result, ensure};
use serde_json::Value;

use super::super::{ModelConfig, finalize_config};

pub(crate) fn parse_longcat_ngram(raw: &Value) -> Result<ModelConfig> {
    let mut normalized = raw.clone();
    let object = normalized
        .as_object_mut()
        .context("longcat config.json must be an object")?;

    for (from, to) in [
        ("num_layers", "num_hidden_layers"),
        ("n_routed_experts", "num_experts"),
        ("moe_topk", "num_experts_per_tok"),
        ("expert_ffn_hidden_size", "moe_intermediate_size"),
        ("ffn_hidden_size", "intermediate_size"),
    ] {
        if let Some(v) = object.get(from).cloned()
            && !object.contains_key(to)
        {
            object.insert(to.into(), v);
        }
    }

    if let Some(n) = object.get("num_hidden_layers").and_then(Value::as_u64) {
        object.insert("num_hidden_layers".into(), Value::from(n * 2));
    }

    // 2026-09-26: `eos_token_id` may be an array; the first id is the primary.
    if let Some(eos) = object.get("eos_token_id").cloned() {
        let primary = match &eos {
            Value::Number(n) => n.as_u64(),
            Value::Array(ids) => ids.first().and_then(Value::as_u64),
            _ => None,
        }
        .context("longcat eos_token_id must be an integer or non-empty integer array")?;
        object.insert("eos_token_id".into(), Value::from(primary));
    }

    let mut config: ModelConfig =
        serde_json::from_value(normalized).context("Failed to parse longcat config.json")?;

    ensure!(
        config.hidden_size > 0,
        "longcat hidden_size must be non-zero"
    );
    ensure!(
        config.num_hidden_layers > 0,
        "longcat num_layers must be non-zero"
    );
    ensure!(config.vocab_size > 0, "longcat vocab_size must be non-zero");

    let ngram_fields = [
        config.ngram_vocab_size_ratio,
        config.emb_neighbor_num,
        config.emb_split_num,
    ];
    let present = ngram_fields.iter().filter(|&&v| v > 0).count();
    ensure!(
        present == 0 || present == 3,
        "longcat n-gram config is partial (ratio={}, neighbor={}, split={}) — \
         all three of ngram_vocab_size_ratio / emb_neighbor_num / emb_split_num \
         must be present together",
        config.ngram_vocab_size_ratio,
        config.emb_neighbor_num,
        config.emb_split_num,
    );
    if present == 3 {
        ensure!(
            config.emb_neighbor_num >= 2,
            "emb_neighbor_num must be >= 2 (largest n-gram size)"
        );
        let num_tables = config.emb_split_num * (config.emb_neighbor_num - 1);
        ensure!(
            config.hidden_size.is_multiple_of(num_tables),
            "hidden_size {} must divide evenly by the {} n-gram tables \
             (emb_split_num {} x (emb_neighbor_num {} - 1))",
            config.hidden_size,
            num_tables,
            config.emb_split_num,
            config.emb_neighbor_num,
        );
    }

    // 2026-09-26: Unless the config sets them, `head_dim` is the full qk head width
    // (nope + rope) and there is one KV head per attention head.
    if config.head_dim == 0 {
        config.head_dim = config.qk_nope_head_dim + config.qk_rope_head_dim;
    }
    if config.num_key_value_heads == 0 {
        config.num_key_value_heads = config.num_attention_heads;
    }
    if config.layer_types.is_empty() {
        config.layer_types = vec![super::super::LayerType::FullAttention; config.num_hidden_layers];
    }
    // 2026-09-26: Plain rope. The config declares no rope scaling, but the MLA loader builds a
    // YaRN inv_freq table, and `compute_yarn_inv_freq` uses factor 128 when `yarn_factor` is
    // 0. At factor 1.0 the interpolated and extrapolated frequencies are equal, so the table
    // is the plain rope, and `yarn_rope_mscale` returns 1.0 for `factor <= 1.0`.
    if config.yarn_factor == 0.0 {
        config.yarn_factor = 1.0;
    }

    // 2026-09-26: The router is a softmax over `num_experts + zero_expert_num` logits; the
    // top-k is chosen with `e_score_correction_bias` added, weighted by the unbiased scores
    // times `routed_scaling_factor`, and not renormalized (`bench/ngram_ref/longcat_forward_ref.py`
    // `moe`).
    config.scoring_func = "softmax".to_string();
    config.norm_topk_prob = false;

    config.model_type = "longcat_flash_ngram".to_string();
    finalize_config(&mut config, raw)?;
    Ok(config)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// 2026-09-26: A LongCat-Flash-Lite `config.json` fixture: 14 checkpoint layers and the full
    /// n-gram trio.
    fn lite_config() -> Value {
        serde_json::json!({
            "architectures": ["LongcatFlashNgramForCausalLM"],
            "model_type": "longcat_flash_ngram",
            "vocab_size": 131072,
            "hidden_size": 3072,
            "ffn_hidden_size": 6144,
            "expert_ffn_hidden_size": 1024,
            "num_layers": 14,
            "num_attention_heads": 32,
            "kv_lora_rank": 512,
            "q_lora_rank": 1536,
            "qk_rope_head_dim": 64,
            "qk_nope_head_dim": 128,
            "v_head_dim": 128,
            "n_routed_experts": 256,
            "moe_topk": 12,
            "routed_scaling_factor": 6.0,
            "zero_expert_num": 128,
            "ngram_vocab_size_ratio": 78,
            "emb_neighbor_num": 4,
            "emb_split_num": 4,
            "rope_theta": 5000000.0,
            "rms_norm_eps": 1e-5,
            "max_position_embeddings": 327680,
            "eos_token_id": 2,
            "torch_dtype": "bfloat16"
        })
    }

    #[test]
    fn parses_lite_config() {
        let c = parse_longcat_ngram(&lite_config()).unwrap();
        assert_eq!(c.num_hidden_layers, 28);
        assert_eq!(c.layer_types.len(), 28);
        assert_eq!(c.head_dim, 192);
        assert_eq!(c.num_key_value_heads, 32);
        assert_eq!(c.zero_expert_num, 128);
        assert_eq!(c.scoring_func, "softmax");
        assert!(!c.norm_topk_prob);
        assert_eq!(c.routed_scaling_factor, 6.0);
        assert_eq!(c.yarn_factor, 1.0);
        assert_eq!(c.num_experts, 256);
        assert_eq!(c.num_experts_per_tok, 12);
        assert_eq!(c.moe_intermediate_size, 1024);
        assert_eq!(c.intermediate_size, 6144);
        assert_eq!(c.kv_lora_rank, 512);
        assert_eq!(c.q_lora_rank, 1536);
        assert_eq!(c.ngram_vocab_size_ratio, 78);
        assert_eq!(c.emb_neighbor_num, 4);
        assert_eq!(c.emb_split_num, 4);
        let tables = c.emb_split_num * (c.emb_neighbor_num - 1);
        assert_eq!(tables, 12);
        assert_eq!(c.hidden_size / tables, 256);
    }

    /// 2026-09-26: A config with `architectures` and no `model_type` is routed here by
    /// architecture name.
    #[test]
    fn architectures_only_config_routes_to_longcat() {
        let mut v = lite_config();
        v.as_object_mut().unwrap().remove("model_type");
        let json = serde_json::to_string(&v).unwrap();
        let c = crate::parse_config(&json).unwrap();
        assert_eq!(c.model_type, "longcat_flash_ngram");
        assert_eq!(c.num_hidden_layers, 28);
        assert_eq!(c.zero_expert_num, 128);
    }

    #[test]
    fn rejects_partial_ngram_trio() {
        let mut v = lite_config();
        v.as_object_mut().unwrap().remove("emb_split_num");
        assert!(parse_longcat_ngram(&v).is_err());
    }

    #[test]
    fn non_ngram_longcat_parses() {
        let mut v = lite_config();
        for k in [
            "ngram_vocab_size_ratio",
            "emb_neighbor_num",
            "emb_split_num",
        ] {
            v.as_object_mut().unwrap().remove(k);
        }
        let c = parse_longcat_ngram(&v).unwrap();
        assert_eq!(c.ngram_vocab_size_ratio, 0);
    }
}
