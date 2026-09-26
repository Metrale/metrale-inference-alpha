// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-26: `config.json` parser for `model_type = "gemma4"` (Gemma-4, dense and MoE).
//!
//! Owner: config (model parsers).
//! Invariants: none beyond the types.

#![allow(unused_imports)]

use anyhow::{Context, Result};
use serde_json::Value;

use super::super::{
    LayerType, ModelConfig, QuantizationConfig, VisionConfig, default_conv_kernel, default_one,
    default_one_f64, default_partial_rotary, default_rms_eps, default_rope_theta, finalize_config,
    parse_quantization_config, parse_vision_config, validate_config,
};

pub(crate) fn parse_gemma4_params(raw: &serde_json::Value) -> Result<ModelConfig> {
    let tc = raw
        .get("text_config")
        .context("gemma4 config missing text_config")?;

    let get_usize = |key: &str| -> usize {
        tc.get(key).and_then(serde_json::Value::as_u64).unwrap_or(0) as usize
    };
    let get_f64 = |key: &str, default: f64| -> f64 {
        tc.get(key)
            .and_then(serde_json::Value::as_f64)
            .unwrap_or(default)
    };

    let hidden_size = get_usize("hidden_size");
    let num_hidden_layers = get_usize("num_hidden_layers");
    let num_attention_heads = get_usize("num_attention_heads");
    let num_key_value_heads = get_usize("num_key_value_heads");
    let head_dim = get_usize("head_dim");
    let intermediate_size = get_usize("intermediate_size");
    let vocab_size = get_usize("vocab_size");
    let rms_norm_eps = get_f64("rms_norm_eps", 1e-6);
    let max_position_embeddings = get_usize("max_position_embeddings");

    // 2026-09-26: `rope_theta` is the sliding layers' theta; the Gemma-4 loader sets the full
    // layers' rope per layer (`weight_loader/gemma4/loader_a.rs`). The local Gemma-4 configs
    // keep both under `rope_parameters.{sliding_attention,full_attention}`.
    let rope_params = tc.get("rope_parameters");
    let sliding_rope = rope_params.and_then(|r| r.get("sliding_attention"));
    let full_rope = rope_params.and_then(|r| r.get("full_attention"));
    let sliding_config = tc.get("sliding_attention_config");
    let full_config = tc.get("full_attention_config");

    let rope_theta = sliding_rope
        .or(sliding_config)
        .and_then(|c| c.get("rope_theta"))
        .and_then(serde_json::Value::as_f64)
        .unwrap_or(10000.0);

    let partial_rotary_factor = full_rope
        .or(full_config)
        .and_then(|c| c.get("partial_rotary_factor"))
        .and_then(serde_json::Value::as_f64)
        .unwrap_or(default_partial_rotary());

    // 2026-09-26: Sliding and full layers are both `LayerType::FullAttention` here.
    let layer_types: Vec<LayerType> = if let Some(pattern) = tc
        .get("attention_pattern")
        .and_then(serde_json::Value::as_array)
    {
        pattern
            .iter()
            .map(|v| {
                // 2026-09-26: The local Gemma-4 configs list only these two layer kinds.
                match v.as_str().unwrap_or("full_attention") {
                    "sliding_attention" | "full_attention" => LayerType::FullAttention,
                    other => panic!("Unknown Gemma-4 attention_pattern entry: '{other}'"),
                }
            })
            .collect()
    } else {
        vec![LayerType::FullAttention; num_hidden_layers]
    };

    let tie_word_embeddings = raw
        .get("tie_word_embeddings")
        .and_then(serde_json::Value::as_bool)
        .unwrap_or(false);

    // 2026-09-26: The template's linear-attention, MoE and MTP fields are overwritten below.
    let mut config = ModelConfig::qwen3_next_80b_nvfp4();
    config.linear_num_key_heads = 0;
    config.linear_key_head_dim = 0;
    config.linear_num_value_heads = 0;
    config.linear_value_head_dim = 0;
    config.linear_conv_kernel_dim = 0;
    // 2026-09-26: Only the MoE variant declares experts (the local Gemma-4-26B-A4B config sets
    // `num_experts: 128`; the 31B config sets null).
    let num_experts = tc.get("num_experts").and_then(|v| v.as_u64()).unwrap_or(0) as usize;
    let top_k_experts = tc
        .get("top_k_experts")
        .and_then(|v| v.as_u64())
        .unwrap_or(0) as usize;
    let moe_intermediate_size = tc
        .get("moe_intermediate_size")
        .and_then(|v| v.as_u64())
        .unwrap_or(0) as usize;
    if num_experts > 0 {
        config.num_experts = num_experts;
        config.num_experts_per_tok = top_k_experts;
        config.moe_intermediate_size = moe_intermediate_size;
        config.shared_expert_intermediate_size = 0;
    } else {
        config.num_experts = 0;
        config.num_experts_per_tok = 1;
        config.moe_intermediate_size = 0;
        config.shared_expert_intermediate_size = 0;
    }
    config.mtp_num_hidden_layers = 0;

    config.hidden_size = hidden_size;
    config.num_hidden_layers = num_hidden_layers;
    config.intermediate_size = intermediate_size;
    config.vocab_size = vocab_size;
    // 2026-09-26: Sliding and full layers differ in head width and KV heads (31B config:
    // sliding 16 KV x `head_dim` 256, full `num_global_key_value_heads` 4 x `global_head_dim`
    // 512). `head_dim` takes `global_head_dim` when present, for buffer sizing; the loader
    // derives each layer's widths from its weight shapes.
    let global_head_dim = tc
        .get("global_head_dim")
        .and_then(|v| v.as_u64())
        .unwrap_or(0) as usize;
    config.num_attention_heads = num_attention_heads;
    config.num_key_value_heads = num_key_value_heads;
    config.head_dim = if global_head_dim > 0 {
        global_head_dim
    } else {
        head_dim
    };
    config.partial_rotary_factor = partial_rotary_factor;
    config.layer_types = layer_types;
    // 2026-09-26: The loader applies the window to sliding layers only.
    config.sliding_window = tc
        .get("sliding_window")
        .and_then(serde_json::Value::as_u64)
        .unwrap_or(0) as u32;
    config.max_position_embeddings = max_position_embeddings;
    config.rope_theta = rope_theta;
    config.rms_norm_eps = rms_norm_eps;
    config.tie_word_embeddings = tie_word_embeddings;
    config.model_type = "gemma4".to_string();
    config.attn_gated = false;
    config.nested_config = true;
    config.norm_topk_prob = num_experts > 0;

    config.embed_scale = (hidden_size as f32).sqrt();

    config.final_logit_softcapping = raw
        .get("text_config")
        .and_then(|tc| tc.get("final_logit_softcapping"))
        .or_else(|| raw.get("final_logit_softcapping"))
        .and_then(serde_json::Value::as_f64)
        .unwrap_or(30.0) as f32;
    // 2026-09-26: The MoE variant runs without softcapping even though its config sets 30.0
    // (the local Gemma-4-26B-A4B checkpoint).
    if config.num_experts > 0 && config.model_type == "gemma4" {
        config.final_logit_softcapping = 0.0;
    }
    if let Ok(v) = std::env::var("METRALE_SOFTCAP_OVERRIDE")
        && let Ok(cap) = v.parse::<f32>()
    {
        config.final_logit_softcapping = cap;
    }
    let _softcap_from_config = raw
        .get("text_config")
        .and_then(|tc| tc.get("final_logit_softcapping"))
        .or_else(|| raw.get("final_logit_softcapping"))
        .and_then(serde_json::Value::as_f64)
        .unwrap_or(30.0) as f32;

    finalize_config(&mut config, raw)?;
    Ok(config)
}
