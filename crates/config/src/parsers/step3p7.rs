// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-26: `config.json` parser for Step 3.7 Flash (`model_type = "step3p7"`), whose
//! language-model fields are under `text_config`. The geometry it serves is recorded in
//! `kernels/gb10/step3p7-flash/MODEL.toml`.
//!
//! Step 3.7 key names and the `ModelConfig` fields they fill:
//!   moe_num_experts                     → num_experts
//!   moe_top_k                           → num_experts_per_tok
//!   share_expert_dim / share_expert_dims → shared_expert_intermediate_size
//!   num_attention_groups                → num_key_value_heads
//!   moe_router_activation               → scoring_func
//!   moe_router_scaling_factor           → routed_scaling_factor
//!   use_moe_router_bias                 → use_routing_bias
//!   norm_expert_weight                  → norm_topk_prob
//!   num_nextn_predict_layers            → mtp_num_hidden_layers, num_mtp_modules
//!
//! Owner: config (model parsers).
//! Invariants: none beyond the types.

#![allow(unused_imports)]

use anyhow::{Context, Result};
use serde_json::Value;

use super::super::{
    LayerType, ModelConfig, default_conv_kernel, default_one, default_one_f64,
    default_partial_rotary, default_rms_eps, default_rope_theta, finalize_config,
    parse_quantization_config, parse_vision_config, validate_config,
};

pub(crate) fn parse_step3p7(raw: &serde_json::Value) -> Result<ModelConfig> {
    let text_config = raw
        .get("text_config")
        .context("step3p7 config missing text_config")?;

    let mut tc_value = text_config.clone();
    if let Some(obj) = tc_value.as_object_mut() {
        // 2026-09-26: An `eos_token_id` array gives its last id as the primary (the serde field
        // would take the first); `parse_config` keeps that choice (`tests/eos.rs`).
        if let Some(arr) = obj.get("eos_token_id").and_then(Value::as_array) {
            let last = arr.last().and_then(Value::as_u64).unwrap_or(1);
            obj.insert("eos_token_id".to_string(), Value::from(last));
        }
        // 2026-09-26: A per-layer `rope_theta` array gives its first element; see the rope
        // section below.
        if let Some(arr) = obj.get("rope_theta").and_then(Value::as_array) {
            let first = arr.first().and_then(Value::as_f64).unwrap_or(5000000.0);
            obj.insert("rope_theta".to_string(), Value::from(first));
        }
        // 2026-09-26: Keys serde cannot read into `ModelConfig` are removed.
        // `swiglu_limits` / `swiglu_limits_shared` are per-layer SwiGLU clamps with no
        // `ModelConfig` field; the Step 3.7 MoE kernel clamps at its own constant
        // `SWIGLU_LIMIT` (10.0, `kernels/gb10/step3p7-flash/nvfp4/moe_silu_mul.cu`).
        obj.remove("partial_rotary_factors");
        obj.remove("swiglu_limits");
        obj.remove("swiglu_limits_shared");
        obj.remove("use_rope_layers");
        obj.remove("yarn_only_types");
        obj.remove("architectures");
        obj.remove("moe_layers_enum");
        // 2026-09-26: `layer_types` is built below; its strings are not serde's `LayerType`
        // names.
        obj.remove("layer_types");
        if let Some(mne) = obj.get("moe_num_experts").cloned() {
            obj.entry("num_experts".to_string()).or_insert(mne);
        }
        if let Some(mtk) = obj.get("moe_top_k").cloned() {
            obj.entry("num_experts_per_tok".to_string()).or_insert(mtk);
        }
        if let Some(nag) = obj.get("num_attention_groups").cloned() {
            obj.entry("num_key_value_heads".to_string()).or_insert(nag);
        }
    }

    let mut config: ModelConfig =
        serde_json::from_value(tc_value).context("Failed to parse step3p7 text_config")?;

    config.model_type = "step3p7".to_string();
    config.nested_config = true;
    config.weight_prefix = "model.language_model".to_string();

    if config.num_experts == 0 {
        config.num_experts = text_config
            .get("moe_num_experts")
            .and_then(Value::as_u64)
            .unwrap_or(288) as usize;
    }
    if config.num_experts_per_tok <= 1 {
        config.num_experts_per_tok = text_config
            .get("moe_top_k")
            .and_then(Value::as_u64)
            .unwrap_or(8) as usize;
    }
    if config.moe_intermediate_size == 0 {
        config.moe_intermediate_size = text_config
            .get("moe_intermediate_size")
            .and_then(Value::as_u64)
            .unwrap_or(1280) as usize;
    }

    config.shared_expert_intermediate_size = text_config
        .get("share_expert_dim")
        .or_else(|| text_config.get("share_expert_dims"))
        .and_then(Value::as_u64)
        .unwrap_or(1280) as usize;

    if config.num_key_value_heads == 0 {
        config.num_key_value_heads = text_config
            .get("num_attention_groups")
            .and_then(Value::as_u64)
            .unwrap_or(8) as usize;
    }

    if config.head_dim == 0 {
        config.head_dim = text_config
            .get("head_dim")
            .and_then(Value::as_u64)
            .unwrap_or(128) as usize;
    }

    // 2026-09-26: `rope_theta` and `partial_rotary_factor` hold one value; a per-layer theta
    // array gives its first element. The loader overrides sliding layers with theta 10000 over
    // the whole head (`weight_loader/step3p7/load_layers.rs`).
    if let Some(rt) = text_config.get("rope_theta") {
        if let Some(theta) = rt.as_f64() {
            config.rope_theta = theta;
        } else if let Some(theta) = rt
            .as_array()
            .and_then(|a| a.first())
            .and_then(Value::as_f64)
        {
            config.rope_theta = theta;
        }
    }
    if let Some(rope_params) = text_config
        .get("rope_scaling")
        .or_else(|| text_config.get("rope_parameters"))
    {
        if config.rope_theta == default_rope_theta()
            && let Some(theta) = rope_params.get("rope_theta").and_then(Value::as_f64)
        {
            config.rope_theta = theta;
        }
        if config.partial_rotary_factor == default_partial_rotary()
            && let Some(prf) = rope_params
                .get("partial_rotary_factor")
                .and_then(Value::as_f64)
        {
            config.partial_rotary_factor = prf;
        }
    }
    if config.partial_rotary_factor == default_partial_rotary()
        && let Some(prf) = text_config
            .get("partial_rotary_factor")
            .and_then(Value::as_f64)
    {
        config.partial_rotary_factor = prf;
    }

    if config.rotary_dim == 0 && config.partial_rotary_factor < 1.0 {
        config.rotary_dim = (config.head_dim as f64 * config.partial_rotary_factor) as usize;
    }

    let router_activation = text_config
        .get("moe_router_activation")
        .and_then(Value::as_str)
        .unwrap_or("sigmoid");
    config.scoring_func = router_activation.to_string();

    config.routed_scaling_factor = text_config
        .get("moe_router_scaling_factor")
        .and_then(Value::as_f64)
        .unwrap_or(3.0);

    config.use_routing_bias = text_config
        .get("use_moe_router_bias")
        .and_then(Value::as_bool)
        .unwrap_or(true);

    config.norm_topk_prob = text_config
        .get("norm_expert_weight")
        .and_then(Value::as_bool)
        .unwrap_or(true);

    // 2026-09-26: Full and sliding layers both hold KV cache: `num_attention_layers()` counts
    // both (`LayerType::is_attention`), and the loader sets the sliding window on sliding
    // layers only.
    if config.layer_types.is_empty()
        && let Some(list) = text_config.get("layer_types").and_then(Value::as_array)
    {
        config.layer_types = list
            .iter()
            .map(|v| match v.as_str().unwrap_or("full_attention") {
                "full_attention" => LayerType::FullAttention,
                "sliding_attention" => LayerType::SlidingAttention,
                other => panic!(
                    "step3p7: unexpected layer_type '{other}' — \
                     only full_attention and sliding_attention are supported"
                ),
            })
            .collect();
    }

    // 2026-09-26: The array also covers the MTP layers (48 entries for 45 layers in
    // `MODEL.toml`); only the first `num_hidden_layers` are kept.
    if config.layer_types.len() > config.num_hidden_layers && config.num_hidden_layers > 0 {
        config.layer_types.truncate(config.num_hidden_layers);
    }

    if let Some(sw) = text_config.get("sliding_window").and_then(Value::as_u64) {
        config.sliding_window = sw as u32;
    }

    // 2026-09-26: Sliding layers may have more Q heads
    // (`attention_other_setting.num_attention_heads`). `num_attention_heads` takes the larger
    // count for buffer sizing; the loader derives each layer's count from its q_proj.
    if let Some(other_heads) = text_config
        .get("attention_other_setting")
        .and_then(|o| o.get("num_attention_heads"))
        .and_then(Value::as_u64)
    {
        let other_heads = other_heads as usize;
        if other_heads > config.num_attention_heads {
            config.num_attention_heads = other_heads;
        }
    }

    // 2026-09-26: `attn_gated` describes a gate interleaved into q_proj. Step 3.7's gate is a
    // separate per-head `g_proj`, which the loader installs as a sigmoid head gate when the
    // checkpoint has `g_proj.weight`.
    config.attn_gated = false;

    let mtp_layers = text_config
        .get("num_nextn_predict_layers")
        .and_then(Value::as_u64)
        .unwrap_or(3) as usize;
    config.mtp_num_hidden_layers = mtp_layers;
    config.num_mtp_modules = mtp_layers;
    config.mtp_transformer_layers = 1;

    if config.vocab_size == 0 {
        config.vocab_size = raw
            .get("vocab_size")
            .or_else(|| text_config.get("vocab_size"))
            .and_then(Value::as_u64)
            .unwrap_or(128896) as usize;
    }

    if config.eos_token_id == 0 {
        config.eos_token_id = text_config
            .get("eos_token_id")
            .and_then(|v| {
                v.as_u64()
                    .or_else(|| v.as_array().and_then(|a| a.first()).and_then(Value::as_u64))
            })
            .unwrap_or(1) as u32;
    }

    if raw.get("vision_config").is_some() || raw.get("image_token_id").is_some() {
        config.vision = parse_vision_config(raw);
    }

    finalize_config(&mut config, raw)?;
    Ok(config)
}
