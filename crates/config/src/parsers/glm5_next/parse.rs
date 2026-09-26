// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-26: `parse_glm5_next`, the GLM-5.3 `config.json` entry point. The layer maps,
//! the vision parser and the refusals it calls are in `glm5_next.rs`.
//!
//! Owner: config (model parsers).
//! Invariants: none beyond those of `glm5_next.rs`.

use super::*;

pub fn parse_glm5_next(json: &str) -> Result<ModelConfig> {
    let raw: serde_json::Value =
        serde_json::from_str(json).context("Invalid JSON in GLM-5.3 (glm5_next) config.json")?;

    // 2026-09-26: Model fields come from `text_config` (the top level when there is none);
    // the outer object supplies `quantization_config` (through `finalize_config`) and
    // `vision_config`.
    let text = text_config(&raw).clone();

    let mut text_for_struct = text.clone();
    if let Some(obj) = text_for_struct.as_object_mut() {
        obj.remove("layer_types");
        // 2026-09-26: `eos_token_id` may be an array. The serde field keeps element 0
        // (`eos_token_id_field`) and `parse_config` collects every id into `eos_token_ids`;
        // read stop tokens through `ModelConfig::eos_ids()` / `is_eos()`.
    }
    let text_json =
        serde_json::to_string(&text_for_struct).context("re-serialize glm5_next text_config")?;
    let mut config: ModelConfig =
        serde_json::from_str(&text_json).context("Failed to parse glm5_next text_config")?;
    let text = &text;

    // 2026-09-26: The inner object says `glm5_next_text`; the model skeleton accepts only
    // `glm5_next`.
    config.model_type = "glm5_next".to_string();

    // 2026-09-26: NoPE: `qk_rope_head_dim` is kept as read, 0 included, and
    // `qk_nope_head_dim` is not derived from it.
    let qk_rope = text
        .get("qk_rope_head_dim")
        .and_then(|v| v.as_u64())
        .map(|v| v as usize);
    match qk_rope {
        Some(v) => config.qk_rope_head_dim = v,
        None => bail!(
            "glm5_next config.json has no qk_rope_head_dim; refusing to guess \
             whether this checkpoint is NoPE"
        ),
    }
    if let Some(v) = text.get("qk_nope_head_dim").and_then(|v| v.as_u64()) {
        config.qk_nope_head_dim = v as usize;
    }
    if let Some(v) = text.get("v_head_dim").and_then(|v| v.as_u64()) {
        config.v_head_dim = v as usize;
    }
    // 2026-09-26: With `head_dim` 0 (as in the test fixture), the MLA per-head width
    // `qk_head_dim` is used, else nope + rope; `hidden_size / num_attention_heads` is not the
    // MLA width.
    if config.head_dim == 0 {
        config.head_dim = text
            .get("qk_head_dim")
            .and_then(|v| v.as_u64())
            .map(|v| v as usize)
            .unwrap_or(config.qk_nope_head_dim + config.qk_rope_head_dim);
    }
    config.partial_rotary_factor = if config.head_dim > 0 {
        config.qk_rope_head_dim as f64 / config.head_dim as f64
    } else {
        0.0
    };

    if config.num_experts == 0 && config.n_routed_experts > 0 {
        config.num_experts = config.n_routed_experts;
    }
    let n_shared = text
        .get("n_shared_experts")
        .and_then(|v| v.as_u64())
        .unwrap_or(0) as usize;
    if config.shared_expert_intermediate_size == 0 && n_shared > 0 {
        config.shared_expert_intermediate_size = n_shared * config.moe_intermediate_size;
    }

    // 2026-09-26: The KDA geometry is under `linear_attn_config`; it is copied into the flat
    // `linear_*` fields.
    let lac = text.get("linear_attn_config");
    if let Some(lac) = lac {
        let g = |k: &str| lac.get(k).and_then(|v| v.as_u64()).map(|v| v as usize);
        if let Some(v) = g("num_heads") {
            config.linear_num_key_heads = v;
            config.linear_num_value_heads = v;
        }
        if let Some(v) = g("head_dim") {
            config.linear_key_head_dim = v;
            config.linear_value_head_dim = v;
        }
        if let Some(v) = g("short_conv_kernel_size") {
            config.linear_conv_kernel_dim = v;
        }
        match lac.get("gate_lower_bound").and_then(|v| v.as_f64()) {
            Some(v) => config.linear_gate_lower_bound = v as f32,
            None => bail!(
                "glm5_next: linear_attn_config has no gate_lower_bound; refusing to guess the \
                 KDA decay bound (GLM-5.3-Flash declares -5.0)"
            ),
        }
    }

    if config.index_topk == 0
        && let Some(v) = text.get("index_topk").and_then(|v| v.as_u64())
    {
        config.index_topk = v as usize;
    }
    if let Some(v) = text.get("index_kpool").and_then(|v| v.as_u64()) {
        config.index_kpool = v as usize;
    }
    if let Some(v) = text
        .get("index_kpool_always_select_tail")
        .and_then(|v| v.as_bool())
    {
        config.index_kpool_always_select_tail = v;
    }

    config.layer_types = build_layer_types(text, config.num_hidden_layers)?;

    // 2026-09-26: The `num_nextn_predict_layers` MTP layers sit past the text stack, in
    // `mtp_layer_types` rather than `layer_types`, so loops over the text stack exclude them.
    // The config does not state their mixer: it is `SparseAttention` when the text stack has
    // one, else `FullAttention`.
    let n_mtp = text
        .get("num_nextn_predict_layers")
        .and_then(|v| v.as_u64())
        .unwrap_or(0) as usize;
    if n_mtp > 0 {
        let kind = if config.layer_types.contains(&LayerType::SparseAttention) {
            LayerType::SparseAttention
        } else {
            LayerType::FullAttention
        };
        config.mtp_layer_types = vec![kind; n_mtp];
    }

    config.mlp_only_layers = build_mlp_only_layers(text, config.num_hidden_layers)?;

    // 2026-09-26: GLM clamps its SwiGLU (`gate` from above only, `up` both ways; see
    // `glm5next_ffn.cu`). Dropping the clamp changes only activations beyond the limit, so a
    // missing or non-positive `swiglu_limit` is an error rather than "no clamp".
    config.swiglu_limit = match text.get("swiglu_limit").and_then(|v| v.as_f64()) {
        Some(v) if v > 0.0 => v as f32,
        Some(v) => bail!("glm5_next: swiglu_limit is {v}, which cannot bound anything"),
        None => bail!(
            "glm5_next config.json has no swiglu_limit; refusing to guess whether this \
             checkpoint clamps its SwiGLU (GLM-5.3-Flash declares 10.0)"
        ),
    };

    // 2026-09-26: Grouped expert routing is not implemented: `glm5next_router_topk` ranks
    // every expert as one group and returns without writing when `n_group != 1`. A value
    // other than 1 for either key is refused here, so the error names the config.
    for key in ["n_group", "topk_group"] {
        if let Some(v) = text.get(key).and_then(|v| v.as_u64())
            && v != 1
        {
            bail!(
                "glm5_next: {key} = {v}. Grouped expert routing is not implemented — \
                 glm5next_router_topk ranks every expert in one group."
            );
        }
    }

    config.glm5next_router_mode = match text.get("moe_router_dtype").and_then(|v| v.as_str()) {
        None => Glm5NextRouterMode::HfFp32,
        Some(s) => Glm5NextRouterMode::from_config_str(s).ok_or_else(|| {
            anyhow::anyhow!(
                "glm5_next: unknown moe_router_dtype {s:?}; expected float32 or bfloat16"
            )
        })?,
    };

    // 2026-09-26: `vision_config` is read from the outer object, a sibling of `text_config`,
    // and only when `glm_vision_enabled()` (`METRALE_GLM_VISION`) is on. Otherwise `vision`
    // is `None` and the chat API refuses image and video input.
    config.vision = match crate::glm_vision_enabled() {
        true => parse_glm5_next_vision(&raw),
        false => None,
    };

    finalize_config(&mut config, &raw).context("glm5_next: finalize_config")?;
    refuse_shared_indexer(text, &config).context("glm5_next: indexer_types")?;
    validate_glm5_next(&config)?;
    Ok(config)
}
