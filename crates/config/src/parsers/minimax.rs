// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-26: `config.json` parser for `model_type = "minimax_m2"` (MiniMax M2).
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

pub(crate) fn parse_minimax_m2(raw: &serde_json::Value) -> Result<ModelConfig> {
    let mut config: ModelConfig =
        serde_json::from_value(raw.clone()).context("Failed to parse minimax_m2 config.json")?;

    // 2026-09-26: A config may carry rope_theta only under `rope_parameters`; it is read
    // there when the flat field kept its serde default.
    if config.rope_theta == default_rope_theta()
        && let Some(rp) = raw.get("rope_parameters")
        && let Some(theta) = rp.get("rope_theta").and_then(serde_json::Value::as_f64)
    {
        config.rope_theta = theta;
    }

    if config.num_experts == 0 {
        let n = raw
            .get("num_local_experts")
            .and_then(serde_json::Value::as_u64)
            .unwrap_or(0) as usize;
        config.num_experts = n;
    }
    // 2026-09-26: MiniMax's `intermediate_size` is the per-expert FFN width; the model has no
    // dense FFN (the MiniMax-M2.7 config.json in the local HF cache sets 1536 and
    // `shared_intermediate_size: 0`).
    if config.moe_intermediate_size == 0 {
        config.moe_intermediate_size = config.intermediate_size;
    }
    config.shared_expert_intermediate_size = 0;

    // 2026-09-26: `attn_type_list` holds one code per layer; only 1 (full attention) is
    // supported and any other code panics.
    if config.layer_types.is_empty()
        && let Some(list) = raw
            .get("attn_type_list")
            .and_then(serde_json::Value::as_array)
    {
        config.layer_types = list.iter().map(|v| {
                match v.as_u64().unwrap_or(1) {
                    1 => LayerType::FullAttention,
                    other => panic!(
                        "minimax_m2: unexpected attn_type_list entry {other} — only 1 (full) is supported in M1"
                    ),
                }
            }).collect();
    }

    let use_mtp = raw
        .get("use_mtp")
        .and_then(serde_json::Value::as_bool)
        .unwrap_or(false);
    if use_mtp && config.mtp_num_hidden_layers == 0 {
        let layers_per = config.mtp_transformer_layers.max(1);
        config.mtp_num_hidden_layers = config.num_mtp_modules.max(1) * layers_per;
    }

    config.attn_gated = false;
    config.nested_config = false;
    config.model_type = "minimax_m2".to_string();

    // 2026-09-26: Set here because the MiniMax-M2.7 config.json in the local HF cache has no
    // `norm_topk_prob` key, which the serde default would read as false.
    config.norm_topk_prob = true;

    finalize_config(&mut config, raw)?;
    Ok(config)
}
