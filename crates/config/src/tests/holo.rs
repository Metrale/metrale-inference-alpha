// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-26: Tests for the Holo-3.1 detection in `parse_config`: a `qwen3_5_moe` config
//! with a vision tower, image token 248056 and no MTP layer becomes `holo3_1_moe`; with any
//! of those missing it stays `qwen3_6_moe`.
//!
//! Owner: config (model parsers).
//! Invariants: none beyond the types.

use super::*;

// 2026-09-26: A Holo-3.1-35B-shaped config: a Qwen3.6-35B-A3B config with a vision tower,
// image token 248056 and no `mtp_num_hidden_layers`.
const HOLO31_VLM_CONFIG: &str = r#"{
        "model_type": "qwen3_5_moe",
        "image_token_id": 248056,
        "vision_start_token_id": 248053,
        "vision_end_token_id": 248054,
        "text_config": {
            "model_type": "qwen3_5_moe_text",
            "hidden_size": 2048,
            "num_hidden_layers": 40,
            "num_attention_heads": 16,
            "num_key_value_heads": 2,
            "head_dim": 256,
            "partial_rotary_factor": 0.25,
            "linear_num_key_heads": 16,
            "linear_key_head_dim": 128,
            "linear_num_value_heads": 32,
            "linear_value_head_dim": 128,
            "linear_conv_kernel_dim": 4,
            "num_experts": 256,
            "num_experts_per_tok": 8,
            "moe_intermediate_size": 512,
            "shared_expert_intermediate_size": 512,
            "vocab_size": 248320,
            "eos_token_id": 248044,
            "full_attention_interval": 4,
            "layer_types": [
                "linear_attention", "linear_attention", "linear_attention", "full_attention",
                "linear_attention", "linear_attention", "linear_attention", "full_attention",
                "linear_attention", "linear_attention", "linear_attention", "full_attention",
                "linear_attention", "linear_attention", "linear_attention", "full_attention",
                "linear_attention", "linear_attention", "linear_attention", "full_attention",
                "linear_attention", "linear_attention", "linear_attention", "full_attention",
                "linear_attention", "linear_attention", "linear_attention", "full_attention",
                "linear_attention", "linear_attention", "linear_attention", "full_attention",
                "linear_attention", "linear_attention", "linear_attention", "full_attention",
                "linear_attention", "linear_attention", "linear_attention", "full_attention"
            ],
            "rope_parameters": {
                "mrope_interleaved": true,
                "mrope_section": [11, 11, 10],
                "rope_theta": 10000000,
                "rope_type": "default"
            }
        },
        "vision_config": {
            "deepstack_visual_indexes": [],
            "depth": 27,
            "hidden_size": 1152,
            "intermediate_size": 4304,
            "num_heads": 16,
            "out_hidden_size": 2048,
            "patch_size": 16,
            "spatial_merge_size": 2,
            "temporal_patch_size": 2
        }
    }"#;

#[test]
fn test_parse_holo31_vlm_config() {
    let cfg = parse_config(HOLO31_VLM_CONFIG).unwrap();
    assert_eq!(cfg.model_type, "holo3_1_moe");
    assert_eq!(cfg.hidden_size, 2048);
    assert_eq!(cfg.num_experts, 256);
    assert_eq!(cfg.num_attention_layers(), 10);
    assert_eq!(cfg.num_ssm_layers(), 30);
    assert_eq!(cfg.mrope_section, [11, 11, 10]);
    assert!(cfg.mrope_interleaved);

    let vision = cfg.vision.expect("Holo3.1 must parse vision_config");
    assert_eq!(vision.depth, 27);
    assert_eq!(vision.hidden_size, 1152);
    assert_eq!(vision.out_hidden_size, 2048);
    assert!(vision.deepstack_visual_indexes.is_empty());
    assert_eq!(vision.image_pad_token_id, 248056);
}

#[test]
fn test_holo31_discriminator_requires_all_signals() {
    let mut wrong_image_token: serde_json::Value = serde_json::from_str(HOLO31_VLM_CONFIG).unwrap();
    wrong_image_token["image_token_id"] = serde_json::json!(248_055);
    assert_eq!(
        parse_config(&wrong_image_token.to_string())
            .unwrap()
            .model_type,
        "qwen3_6_moe",
        "the Holo rewrite requires its checkpoint image token"
    );

    let mut no_vision: serde_json::Value = serde_json::from_str(HOLO31_VLM_CONFIG).unwrap();
    no_vision.as_object_mut().unwrap().remove("vision_config");
    assert_eq!(
        parse_config(&no_vision.to_string()).unwrap().model_type,
        "qwen3_6_moe",
        "a text-only config is not Holo-3.1 VLM"
    );

    let mut other_family: serde_json::Value = serde_json::from_str(HOLO31_VLM_CONFIG).unwrap();
    other_family["model_type"] = serde_json::json!("qwen3_vl_moe");
    assert_eq!(
        parse_config(&other_family.to_string()).unwrap().model_type,
        "qwen3_6_moe",
        "the Holo rewrite is limited to its qwen3_5_moe source family"
    );
}

// 2026-09-26: The local Qwen/Qwen3.6-35B-A3B-FP8 config has a vision tower and image token
// 248056 but also `mtp_num_hidden_layers: 1`, so it stays `qwen3_6_moe`.
#[test]
fn test_qwen36_35b_with_mtp_is_not_holo() {
    let json = HOLO31_VLM_CONFIG.replace(
        "\"model_type\": \"qwen3_5_moe_text\",",
        "\"model_type\": \"qwen3_5_moe_text\",\n            \"mtp_num_hidden_layers\": 1,",
    );
    let cfg = parse_config(&json).unwrap();
    assert_eq!(cfg.model_type, "qwen3_6_moe");
    assert_eq!(cfg.mtp_num_hidden_layers, 1);
    assert!(cfg.vision.is_some());
}
