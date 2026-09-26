// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-26: Parser for the `vision_config` block of a vision-language `config.json`.
//!
//! Owner: config (model parsers).
//! Invariants: none beyond the types.

#![allow(unused_imports)]

use anyhow::{Context, Result};
use serde_json::Value;

use super::super::{ModelConfig, VisionConfig};

pub(crate) fn parse_vision_config(raw: &serde_json::Value) -> Option<VisionConfig> {
    let vc = raw.get("vision_config")?;
    let get_usize = |key: &str| -> usize {
        vc.get(key).and_then(serde_json::Value::as_u64).unwrap_or(0) as usize
    };
    let deepstack_visual_indexes = vc
        .get("deepstack_visual_indexes")
        .and_then(serde_json::Value::as_array)
        .map(|arr| {
            arr.iter()
                .filter_map(serde_json::Value::as_u64)
                .map(|v| v as usize)
                .collect()
        })
        .unwrap_or_default();
    // 2026-09-26: The placeholder ids are read from the top level first, then from
    // `vision_config`. 0 means undeclared: the model engine then uses
    // `vision_encoder::IMAGE_PAD_TOKEN_ID` / `VIDEO_PAD_TOKEN_ID`, and the tokenizer looks
    // up `<|image_pad|>` / `<|video_pad|>`.
    let image_pad_token_id = raw
        .get("image_token_id")
        .and_then(serde_json::Value::as_u64)
        .or_else(|| vc.get("image_token_id").and_then(serde_json::Value::as_u64))
        .unwrap_or(0) as u32;
    let video_pad_token_id = raw
        .get("video_token_id")
        .and_then(serde_json::Value::as_u64)
        .or_else(|| vc.get("video_token_id").and_then(serde_json::Value::as_u64))
        .unwrap_or(0) as u32;
    Some(VisionConfig {
        depth: get_usize("depth"),
        hidden_size: get_usize("hidden_size"),
        num_heads: get_usize("num_heads"),
        patch_size: get_usize("patch_size"),
        temporal_patch_size: get_usize("temporal_patch_size"),
        spatial_merge_size: get_usize("spatial_merge_size"),
        intermediate_size: get_usize("intermediate_size"),
        out_hidden_size: get_usize("out_hidden_size"),
        deepstack_visual_indexes,
        image_pad_token_id,
        video_pad_token_id,
        // 2026-09-26: Not in config.json. The server fills it from `--vision-max-pixels` or
        // the checkpoint's processor config before `build_model` constructs the encoder.
        max_pixels: None,
        // 2026-09-26: The remaining fields keep `VisionConfig::default()`, the Qwen tower's
        // values; `parse_glm5_next` sets them for GLM-5.3.
        ..VisionConfig::default()
    })
}
