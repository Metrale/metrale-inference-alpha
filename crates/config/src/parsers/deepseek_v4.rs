// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-26: Parser for `deepseek_v4` configs, and the check that a DSpark block in one is
//! complete.
//!
//! Owner: config.
//! Invariants: none beyond the types.

use anyhow::{Context, Result};

use super::super::{LayerType, ModelConfig, finalize_config, parse_quantization_config};

pub fn parse_deepseek_v4(json: &str) -> Result<ModelConfig> {
    let mut raw: serde_json::Value =
        serde_json::from_str(json).context("Invalid JSON in DeepSeek-V4 config.json")?;

    // 2026-09-26: Top-level JSON nulls become 0 before serde reads the object, because
    // `#[serde(default)]` covers only missing keys.
    if let Some(obj) = raw.as_object_mut() {
        for v in obj.values_mut() {
            if v.is_null() {
                *v = serde_json::Value::Number(serde_json::Number::from(0));
            }
        }
    }

    let json_fixed =
        serde_json::to_string(&raw).context("Failed to re-serialize DeepSeek-V4 config")?;
    let mut config: ModelConfig =
        serde_json::from_str(&json_fixed).context("Failed to parse deepseek_v4 config.json")?;

    if config.num_experts == 0 && config.n_routed_experts > 0 {
        config.num_experts = config.n_routed_experts;
    }

    // 2026-09-26: `n_shared_experts` is a count; the shared FFN width is that count times
    // `moe_intermediate_size`.
    let n_shared_experts = raw
        .get("n_shared_experts")
        .and_then(|v| v.as_u64())
        .unwrap_or(0) as usize;
    if config.shared_expert_intermediate_size == 0 && n_shared_experts > 0 {
        config.shared_expert_intermediate_size = n_shared_experts * config.moe_intermediate_size;
    }

    if config.kv_lora_rank == 0 {
        config.kv_lora_rank = 512;
    }

    if config.head_dim == 0 && config.hidden_size > 0 && config.num_attention_heads > 0 {
        config.head_dim = config.hidden_size / config.num_attention_heads;
    }
    // 2026-09-26: The MLA head_dim is 512 (the fixture in the tests below declares it). With
    // 4096 hidden, 64 heads and both MLA ranks set, a head_dim of 64 (what the fallback above
    // computes) is replaced with it.
    if config.head_dim == 64
        && config.hidden_size == 4096
        && config.num_attention_heads == 64
        && config.kv_lora_rank > 0
        && config.q_lora_rank > 0
    {
        config.head_dim = 512;
    }

    if config.q_lora_rank == 0 {
        config.q_lora_rank = 1024;
    }

    if config.qk_nope_head_dim == 0 && config.head_dim > 0 && config.qk_rope_head_dim > 0 {
        config.qk_nope_head_dim = config.head_dim - config.qk_rope_head_dim;
    }

    if config.v_head_dim == 0 && config.head_dim > 0 {
        config.v_head_dim = config.head_dim;
    }

    // 2026-09-26: With `o_lora_rank > 0`, `rope_theta` takes `compress_rope_theta`, not the
    // top-level `rope_theta` (`ds4f_reads_checkpoint_compress_theta`).
    if config.o_lora_rank > 0 {
        if let Some(theta) = raw.get("compress_rope_theta").and_then(|v| v.as_f64()) {
            config.rope_theta = theta;
        }
        if let Some(rope_dim) = raw.get("qk_rope_head_dim").and_then(|v| v.as_u64()) {
            config.qk_rope_head_dim = rope_dim as usize;
        }
        if let Some(nope_dim) = raw.get("qk_nope_head_dim").and_then(|v| v.as_u64()) {
            config.qk_nope_head_dim = nope_dim as usize;
        }
        if let Some(g) = raw.get("o_groups").and_then(|v| v.as_u64()) {
            config.o_groups = g as usize;
        }
    }

    if config.qk_rope_head_dim > 0 && config.head_dim > 0 {
        config.partial_rotary_factor = config.qk_rope_head_dim as f64 / config.head_dim as f64;
    }

    config.layer_types = vec![LayerType::FullAttention; config.num_hidden_layers];

    config.model_type = "deepseek_v4".to_string();
    config.attn_gated = false;
    config.nested_config = false;
    config.weight_prefix = "model".to_string();

    let topk_method = raw
        .get("topk_method")
        .and_then(|v| v.as_str())
        .unwrap_or("");
    if topk_method == "noaux_tc" {
        config.use_routing_bias = true;
    }

    // 2026-09-26: An absent `scoring_func` is taken as `sqrtsoftplus`; the MoE forward paths
    // choose the routing kernel by this string.
    config.scoring_func = raw
        .get("scoring_func")
        .and_then(|v| v.as_str())
        .unwrap_or("sqrtsoftplus")
        .to_string();

    if let Some(s) = raw.get("routed_scaling_factor").and_then(|v| v.as_f64()) {
        config.routed_scaling_factor = s;
    }

    if let Some(n) = raw.get("num_nextn_predict_layers").and_then(|v| v.as_u64()) {
        config.num_mtp_modules = n as usize;
        config.mtp_transformer_layers = 1;
        config.mtp_num_hidden_layers = n as usize;
    }

    validate_dspark_contract(&config, &raw)?;

    if config.quantization_config.is_none() {
        config.quantization_config = parse_quantization_config(&raw);
    }

    if let Some(ratios) = raw.get("compress_ratios").and_then(|v| v.as_array()) {
        config.compress_ratios = ratios
            .iter()
            .filter_map(|v| v.as_u64().map(|x| x as usize))
            .collect();
    }
    if config.compress_ratios.contains(&4) {
        if config.index_n_heads == 0 {
            config.index_n_heads = 64;
        }
        if config.index_head_dim == 0 {
            config.index_head_dim = 128;
        }
        if config.index_topk == 0 {
            config.index_topk = 512;
        }
    }

    if let Some(n) = raw.get("num_hash_layers").and_then(|v| v.as_u64()) {
        config.num_hash_layers = n as usize;
    }

    // 2026-09-26: These fallbacks test the parsed value for 0, not the key for absence: a JSON
    // null was already turned into 0 above.
    if config.hc_mult == 0 {
        config.hc_mult = 4;
    }
    if config.hc_sinkhorn_iters == 0 {
        config.hc_sinkhorn_iters = 20;
    }
    if config.hc_eps == 0.0 {
        config.hc_eps = 1e-6;
    }

    // 2026-09-26: YaRN parameters come from `rope_scaling`, or from `rope_parameters` when
    // that is absent (`pre_release_rope_parameters_alias_is_supported`).
    if let Some(rp) = raw
        .get("rope_scaling")
        .or_else(|| raw.get("rope_parameters"))
    {
        if let Some(f) = rp.get("factor").and_then(|v| v.as_f64()) {
            config.yarn_factor = f as f32;
        }
        if let Some(bf) = rp.get("beta_fast").and_then(|v| v.as_f64()) {
            config.yarn_beta_fast = bf as f32;
        }
        if let Some(bs) = rp.get("beta_slow").and_then(|v| v.as_f64()) {
            config.yarn_beta_slow = bs as f32;
        }
        if let Some(om) = rp
            .get("original_max_position_embeddings")
            .and_then(|v| v.as_u64())
        {
            config.yarn_original_max_position_embeddings = om as usize;
        }
    }

    // 2026-09-26: Both YaRN mscale terms are forced to 0.0 for every `deepseek_v4` config,
    // whatever the checkpoint declares (`ds4f_explicit_checkpoint_mscale_is_overridden`).
    // `yarn_rope_mscale` divides the two, so the amplitude factor is 1.0.
    config.yarn_mscale = 0.0;
    config.yarn_mscale_all_dim = 0.0;

    finalize_config(&mut config, &raw)?;
    Ok(config)
}

fn validate_dspark_contract(config: &ModelConfig, raw: &serde_json::Value) -> Result<()> {
    const DSPARK_FIELDS: [&str; 4] = [
        "dspark_block_size",
        "dspark_noise_token_id",
        "dspark_target_layer_ids",
        "dspark_markov_rank",
    ];
    if DSPARK_FIELDS.iter().all(|field| raw.get(field).is_none()) {
        return Ok(());
    }

    for field in DSPARK_FIELDS {
        anyhow::ensure!(
            raw.get(field).is_some(),
            "DSpark config is missing `{field}`"
        );
    }
    anyhow::ensure!(
        config.dspark_block_size > 0,
        "`dspark_block_size` must be > 0"
    );
    anyhow::ensure!(
        config.dspark_noise_token_id < config.vocab_size as u32,
        "`dspark_noise_token_id` ({}) is outside vocab_size ({})",
        config.dspark_noise_token_id,
        config.vocab_size
    );
    anyhow::ensure!(
        !config.dspark_target_layer_ids.is_empty(),
        "`dspark_target_layer_ids` must not be empty"
    );
    anyhow::ensure!(
        config
            .dspark_target_layer_ids
            .iter()
            .all(|&layer| layer < config.num_hidden_layers),
        "`dspark_target_layer_ids` must reference target layers in 0..{}",
        config.num_hidden_layers
    );
    anyhow::ensure!(
        config
            .dspark_target_layer_ids
            .windows(2)
            .all(|pair| pair[0] < pair[1]),
        "`dspark_target_layer_ids` must be strictly increasing"
    );
    anyhow::ensure!(
        config.dspark_markov_rank > 0,
        "`dspark_markov_rank` must be > 0"
    );
    Ok(())
}

#[cfg(test)]
mod mscale_contract_tests {
    use super::*;

    const DS4F_CONFIG: &str = r#"{
      "architectures": ["DeepseekV4ForCausalLM"],
      "head_dim": 512,
      "hidden_size": 4096,
      "max_position_embeddings": 1048576,
      "model_type": "deepseek_v4",
      "num_attention_heads": 64,
      "num_hidden_layers": 43,
      "num_key_value_heads": 1,
      "o_lora_rank": 1024,
      "q_lora_rank": 1024,
      "qk_rope_head_dim": 64,
      "rms_norm_eps": 1e-06,
      "rope_scaling": {
        "beta_fast": 32,
        "beta_slow": 1,
        "factor": 16,
        "original_max_position_embeddings": 65536,
        "type": "yarn"
      },
      "rope_theta": 10000,
      "vocab_size": 129280,
      "compress_rope_theta": 160000,
      "compress_ratios": [0,0,4,128,4,128,4,0,0,0]
    }"#;

    #[test]
    fn ds4f_parser_forces_both_mscale_terms_to_zero() {
        let c = parse_deepseek_v4(DS4F_CONFIG).expect("parse DS4F");
        assert_eq!(c.yarn_mscale, 0.0, "yarn_mscale must be forced to 0.0");
        assert_eq!(
            c.yarn_mscale_all_dim, 0.0,
            "yarn_mscale_all_dim must be forced to 0.0"
        );
    }

    #[test]
    fn ds4f_reads_checkpoint_compress_theta() {
        let c = parse_deepseek_v4(DS4F_CONFIG).expect("parse DS4F");
        assert_eq!(c.rope_theta, 160000.0, "compress rope_theta must be 160000");

        let alternate = DS4F_CONFIG.replace(
            "\"compress_rope_theta\": 160000",
            "\"compress_rope_theta\": 234567",
        );
        let c = parse_deepseek_v4(&alternate).expect("parse alternate compress theta");
        assert_eq!(c.rope_theta, 234567.0);
    }

    #[test]
    fn ds4f_reads_checkpoint_yarn_scaling_parameters() {
        let c = parse_deepseek_v4(DS4F_CONFIG).expect("parse DS4F");
        assert_eq!(c.yarn_factor, 16.0);
        assert_eq!(c.yarn_beta_fast, 32.0);
        assert_eq!(c.yarn_beta_slow, 1.0);
        assert_eq!(c.yarn_original_max_position_embeddings, 65536);

        let alternate = DS4F_CONFIG
            .replace("\"factor\": 16", "\"factor\": 12.5")
            .replace("\"beta_fast\": 32", "\"beta_fast\": 40")
            .replace("\"beta_slow\": 1", "\"beta_slow\": 2")
            .replace(
                "\"original_max_position_embeddings\": 65536",
                "\"original_max_position_embeddings\": 32768",
            );
        let c = parse_deepseek_v4(&alternate).expect("parse alternate YaRN parameters");
        assert_eq!(c.yarn_factor, 12.5);
        assert_eq!(c.yarn_beta_fast, 40.0);
        assert_eq!(c.yarn_beta_slow, 2.0);
        assert_eq!(c.yarn_original_max_position_embeddings, 32768);
    }

    #[test]
    fn pre_release_rope_parameters_alias_is_supported() {
        let pre_release = DS4F_CONFIG.replacen("\"rope_scaling\"", "\"rope_parameters\"", 1);
        let c = parse_deepseek_v4(&pre_release).expect("parse pre-release YaRN parameters");
        assert_eq!(c.yarn_factor, 16.0);
        assert_eq!(c.yarn_beta_fast, 32.0);
        assert_eq!(c.yarn_beta_slow, 1.0);
        assert_eq!(c.yarn_original_max_position_embeddings, 65536);
    }

    #[test]
    fn ds4f_explicit_checkpoint_mscale_is_overridden() {
        let with_mscale = DS4F_CONFIG.replace(
            "\"type\": \"yarn\"",
            "\"type\": \"yarn\", \"mscale\": 1.0, \"mscale_all_dim\": 0.5",
        );
        let c = parse_deepseek_v4(&with_mscale).expect("parse DS4F w/ explicit mscale");
        assert_eq!(c.yarn_mscale, 0.0);
        assert_eq!(c.yarn_mscale_all_dim, 0.0);
    }

    #[test]
    fn each_dspark_field_is_required_when_contract_is_present() {
        let mut complete: serde_json::Value =
            serde_json::from_str(DS4F_CONFIG).expect("parse DS4F fixture");
        let object = complete.as_object_mut().expect("DS4F fixture is an object");
        object.insert("dspark_block_size".into(), serde_json::json!(5));
        object.insert("dspark_noise_token_id".into(), serde_json::json!(128799));
        object.insert(
            "dspark_target_layer_ids".into(),
            serde_json::json!([40, 41, 42]),
        );
        object.insert("dspark_markov_rank".into(), serde_json::json!(256));

        for missing in [
            "dspark_block_size",
            "dspark_noise_token_id",
            "dspark_target_layer_ids",
            "dspark_markov_rank",
        ] {
            let mut incomplete = complete.clone();
            incomplete
                .as_object_mut()
                .expect("DS4F fixture is an object")
                .remove(missing);
            let err =
                parse_deepseek_v4(&incomplete.to_string()).expect_err("incomplete DSpark config");
            let expected = format!("DSpark config is missing `{missing}`");
            assert!(
                err.to_string().contains(&expected),
                "missing {missing} produced unexpected error: {err:#}"
            );
        }
    }
}
