// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-26: Tests for the PEFT `adapter_config.json` parser: scaling, accepted forms, and
//! the named rejection of each unsupported setting.
//!
//! Owner: config (LoRA adapters).
//! Invariants: none beyond the types.

use super::*;

fn base_json() -> serde_json::Value {
    serde_json::json!({
        "peft_type": "LORA",
        "task_type": "CAUSAL_LM",
        "base_model_name_or_path": "Hcompany/Holo-3.1-0.8B",
        "r": 16,
        "lora_alpha": 32,
        "lora_dropout": 0.05,
        "bias": "none",
        "use_rslora": false,
        "use_dora": false,
        "rank_pattern": {},
        "alpha_pattern": {},
        "modules_to_save": null,
        "layers_to_transform": null,
        "target_modules": ["k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    })
}

#[test]
fn standard_lora_scaling_preserves_inputs_and_uses_alpha_over_rank() {
    let cfg = parse_peft_adapter_config(&base_json().to_string()).unwrap();
    assert_eq!(cfg.r, 16);
    assert_eq!(cfg.lora_alpha, 32.0);
    assert!(!cfg.use_rslora);
    assert_eq!(cfg.scaling(), 2.0);
}

#[test]
fn explicit_target_modules_are_preserved_verbatim() {
    let cfg = parse_peft_adapter_config(&base_json().to_string()).unwrap();
    assert_eq!(
        cfg.target_modules,
        [
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ]
    );
}

#[test]
fn rslora_scaling_preserves_inputs_and_uses_alpha_over_sqrt_rank() {
    let mut j = base_json();
    j["use_rslora"] = serde_json::json!(true);
    let cfg = parse_peft_adapter_config(&j.to_string()).unwrap();
    assert_eq!(cfg.r, 16);
    assert_eq!(cfg.lora_alpha, 32.0);
    assert!(cfg.use_rslora);
    assert_eq!(cfg.scaling(), 8.0);
}

#[test]
fn float_alpha_accepted() {
    let mut j = base_json();
    j["lora_alpha"] = serde_json::json!(16.5);
    let cfg = parse_peft_adapter_config(&j.to_string()).unwrap();
    assert_eq!(cfg.lora_alpha, 16.5);
}

#[test]
fn layers_to_transform_array_does_not_block_adapter_loading() {
    // 2026-09-26: `test_data/lora-holo-tiny/adapter_config.json` carries this list.
    let mut j = base_json();
    j["layers_to_transform"] = serde_json::json!([3, 7, 11, 15, 19, 23]);
    parse_peft_adapter_config(&j.to_string()).expect("supported PEFT layer list");
}

#[test]
fn missing_use_rslora_rejected_named() {
    let mut j = base_json();
    j.as_object_mut().unwrap().remove("use_rslora");
    let err = parse_peft_adapter_config(&j.to_string())
        .unwrap_err()
        .to_string();
    assert!(err.contains("REJECT(use_rslora)"), "{err}");
}

#[test]
fn q_proj_accepted() {
    let mut j = base_json();
    j["target_modules"] = serde_json::json!(["q_proj", "v_proj"]);
    let cfg = parse_peft_adapter_config(&j.to_string()).unwrap();
    assert!(cfg.target_modules.iter().any(|m| m == "q_proj"));
}

#[test]
fn gdn_module_rejected_named() {
    // 2026-09-26: The inputs that feed the recurrence; `out_proj` is accepted
    // (`gdn_out_proj_is_accepted`).
    for m in [
        "in_proj_qkvz",
        "in_proj_ba",
        "in_proj_qkv",
        "in_proj_z",
        "in_proj_a",
        "in_proj_b",
        "conv1d",
    ] {
        let mut j = base_json();
        j["target_modules"] = serde_json::json!([m]);
        let err = parse_peft_adapter_config(&j.to_string())
            .unwrap_err()
            .to_string();
        assert!(err.contains("REJECT(gdn)"), "{m}: {err}");
    }
}

#[test]
fn router_gate_target_accepted() {
    // 2026-09-26: The MoE router leaf `gate`, as a bare name and as a full path.
    let mut j = base_json();
    j["target_modules"] = serde_json::json!(["gate"]);
    let cfg = parse_peft_adapter_config(&j.to_string()).unwrap();
    assert_eq!(cfg.target_modules, vec!["gate"]);
    let mut j = base_json();
    j["target_modules"] = serde_json::json!(["model.layers.3.mlp.gate"]);
    assert!(parse_peft_adapter_config(&j.to_string()).is_ok());
}

#[test]
fn all_linear_rejected_named() {
    let mut j = base_json();
    j["target_modules"] = serde_json::json!("all-linear");
    let err = parse_peft_adapter_config(&j.to_string())
        .unwrap_err()
        .to_string();
    assert!(err.contains("REJECT(target_modules)"), "{err}");
}

#[test]
fn unsupported_inference_controls_are_rejected_by_name() {
    for (key, val, tag) in [
        ("use_dora", serde_json::json!(true), "REJECT(use_dora)"),
        ("bias", serde_json::json!("lora_only"), "REJECT(bias)"),
        (
            "rank_pattern",
            serde_json::json!({"k_proj": 8}),
            "REJECT(rank_pattern)",
        ),
        (
            "alpha_pattern",
            serde_json::json!({"k_proj": 8.0}),
            "REJECT(alpha_pattern)",
        ),
        (
            "target_parameters",
            serde_json::json!(["mlp.experts.gate_up_proj"]),
            "REJECT(target_parameters)",
        ),
        (
            "peft_type",
            serde_json::json!("ADALORA"),
            "REJECT(peft_type)",
        ),
    ] {
        let mut j = base_json();
        j[key] = val;
        let err = parse_peft_adapter_config(&j.to_string())
            .unwrap_err()
            .to_string();
        assert!(err.contains(tag), "{key}: {err}");
    }
}

#[test]
fn full_path_target_validates_on_leaf() {
    let mut j = base_json();
    j["target_modules"] = serde_json::json!(["model.layers.3.self_attn.k_proj"]);
    let cfg = parse_peft_adapter_config(&j.to_string()).unwrap();
    assert_eq!(cfg.target_modules, vec!["model.layers.3.self_attn.k_proj"]);
}

#[test]
fn zero_rank_rejected_by_name() {
    let mut j = base_json();
    j["r"] = serde_json::json!(0);
    let err = parse_peft_adapter_config(&j.to_string())
        .unwrap_err()
        .to_string();
    assert!(err.contains("REJECT(r)"), "{err}");
}

#[test]
fn nonpositive_alpha_rejected_by_name() {
    for alpha in [0.0, -1.0] {
        let mut j = base_json();
        j["lora_alpha"] = serde_json::json!(alpha);
        let err = parse_peft_adapter_config(&j.to_string())
            .unwrap_err()
            .to_string();
        assert!(err.contains("REJECT(lora_alpha)"), "{alpha}: {err}");
    }
}

#[test]
fn modules_to_save_embed_lmhead_accepted() {
    let mut j = base_json();
    j["modules_to_save"] = serde_json::json!(["embed_tokens", "lm_head"]);
    let cfg = parse_peft_adapter_config(&j.to_string()).unwrap();
    assert_eq!(cfg.modules_to_save, vec!["embed_tokens", "lm_head"]);
}

#[test]
fn modules_to_save_full_path_validates_on_leaf() {
    let mut j = base_json();
    j["modules_to_save"] = serde_json::json!(["base_model.model.model.embed_tokens"]);
    let cfg = parse_peft_adapter_config(&j.to_string()).unwrap();
    assert_eq!(cfg.modules_to_save, vec!["embed_tokens"]);
}

#[test]
fn modules_to_save_unknown_leaf_rejected() {
    let mut j = base_json();
    j["modules_to_save"] = serde_json::json!(["embed_tokens", "score"]);
    let err = parse_peft_adapter_config(&j.to_string())
        .unwrap_err()
        .to_string();
    assert!(err.contains("REJECT(modules_to_save)"), "{err}");
}

#[test]
fn trainable_token_indices_list_form_preserves_row_order() {
    let mut j = base_json();
    j["trainable_token_indices"] = serde_json::json!([7, 42, 256205]);
    let cfg = parse_peft_adapter_config(&j.to_string()).unwrap();
    assert_eq!(cfg.trainable_token_indices, vec![7, 42, 256205]);
}

#[test]
fn duplicate_trainable_token_index_rejected() {
    let mut j = base_json();
    j["trainable_token_indices"] = serde_json::json!([7, 42, 42, 256205]);
    let err = parse_peft_adapter_config(&j.to_string())
        .unwrap_err()
        .to_string();
    assert!(err.contains("REJECT(trainable_token_indices)"), "{err}");
    assert!(err.contains("duplicate id 42"), "{err}");
}

#[test]
fn nonascending_trainable_token_indices_rejected() {
    let mut j = base_json();
    j["trainable_token_indices"] = serde_json::json!([42, 7, 256205]);
    let err = parse_peft_adapter_config(&j.to_string())
        .unwrap_err()
        .to_string();
    assert!(err.contains("REJECT(trainable_token_indices)"), "{err}");
    assert!(err.contains("ids must be ascending"), "{err}");
}

#[test]
fn trainable_token_indices_dict_form_accepts_shared_order() {
    let mut j = base_json();
    j["trainable_token_indices"] = serde_json::json!({
        "embed_tokens": [10, 99, 256205],
        "lm_head": [10, 99, 256205]
    });
    let cfg = parse_peft_adapter_config(&j.to_string()).unwrap();
    assert_eq!(cfg.trainable_token_indices, vec![10, 99, 256205]);
}

#[test]
fn differing_per_module_trainable_token_indices_rejected() {
    let mut j = base_json();
    j["trainable_token_indices"] = serde_json::json!({
        "embed_tokens": [10, 99],
        "lm_head": [10, 256205]
    });
    let err = parse_peft_adapter_config(&j.to_string())
        .unwrap_err()
        .to_string();
    assert!(err.contains("REJECT(trainable_token_indices)"), "{err}");
    assert!(err.contains("per-module token lists differ"), "{err}");
}

#[test]
fn unknown_trainable_token_module_rejected() {
    let mut j = base_json();
    j["trainable_token_indices"] = serde_json::json!({"classifier": [10]});
    let err = parse_peft_adapter_config(&j.to_string())
        .unwrap_err()
        .to_string();
    assert!(err.contains("REJECT(trainable_token_indices)"), "{err}");
    assert!(err.contains("unsupported module 'classifier'"), "{err}");
}

#[test]
fn invalid_trainable_token_entries_rejected_by_name() {
    for (value, detail) in [
        (
            serde_json::json!(-1),
            "entries must be non-negative integers",
        ),
        (
            serde_json::json!(1.5),
            "entries must be non-negative integers",
        ),
        (
            serde_json::json!("7"),
            "entries must be non-negative integers",
        ),
        (serde_json::json!(4_294_967_296_u64), "exceeds u32 range"),
    ] {
        let mut j = base_json();
        j["trainable_token_indices"] = serde_json::json!([value]);
        let err = parse_peft_adapter_config(&j.to_string())
            .unwrap_err()
            .to_string();
        assert!(err.contains("REJECT(trainable_token_indices)"), "{err}");
        assert!(err.contains(detail), "{err}");
    }
}

#[test]
fn empty_target_modules_with_overlay_accepted() {
    let mut j = base_json();
    j["target_modules"] = serde_json::json!([]);
    j["trainable_token_indices"] = serde_json::json!([256205]);
    let cfg = parse_peft_adapter_config(&j.to_string()).unwrap();
    assert!(cfg.target_modules.is_empty());
    assert_eq!(cfg.trainable_token_indices, vec![256205]);
}

#[test]
fn empty_target_modules_without_overlay_rejected() {
    let mut j = base_json();
    j["target_modules"] = serde_json::json!([]);
    let err = parse_peft_adapter_config(&j.to_string())
        .unwrap_err()
        .to_string();
    assert!(err.contains("REJECT(target_modules)"), "{err}");
}

#[test]
fn absent_target_modules_with_modules_to_save_accepted() {
    let mut j = base_json();
    j.as_object_mut().unwrap().remove("target_modules");
    j["modules_to_save"] = serde_json::json!(["lm_head"]);
    let cfg = parse_peft_adapter_config(&j.to_string()).unwrap();
    assert!(cfg.target_modules.is_empty());
    assert_eq!(cfg.modules_to_save, vec!["lm_head"]);
}

/// 2026-09-26: The GDN block's output projection (value_dim -> hidden) is accepted; it runs
/// after the recurrence. `gdn_module_rejected_named` covers the rejected inputs.
#[test]
fn gdn_out_proj_is_accepted() {
    let mut j = base_json();
    j["target_modules"] = serde_json::json!(["out_proj"]);
    let cfg = parse_peft_adapter_config(&j.to_string())
        .expect("out_proj is supported since the GDN out_proj phase");
    assert_eq!(cfg.target_modules, vec!["out_proj".to_string()]);
}
