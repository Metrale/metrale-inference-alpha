// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-26: `config.json` parser for GLM-5.3-Flash (`model_type` `glm5_next` or
//! `glm5_next_text`): NoPE MLA (`qk_rope_head_dim == 0`) over a stack of KDA linear-attention
//! and DeepSeek sparse-attention (DSA) layers.
//!
//! The DeepSeek-V4 parser is not reused: it derives `qk_nope_head_dim` from
//! `head_dim - qk_rope_head_dim` when the rope dim is non-zero, and here a zero rope dim is a
//! value to keep.
//!
//! Owner: config (model parsers).
//! Invariants:
//! - A parse that succeeds has `qk_rope_head_dim == 0`, a non-zero `head_dim`, and at least
//!   one linear and one non-linear text layer (`validate_glm5_next`).
//! - When `linear_attn_config` has both index lists, they decide the layer map, and a
//!   `layer_types` array with one entry per layer must agree with it (`build_layer_types`).
//! - A DSA layer with shared indexing is refused (`refuse_shared_indexer`).

use anyhow::{Context, Result, bail};

use super::super::{Glm5NextRouterMode, LayerType, ModelConfig, finalize_config};
mod parse;
pub use parse::parse_glm5_next;

/// 2026-09-26: The layer index of the MTP layer: `num_hidden_layers`, one past the text
/// stack, which `num_hidden_layers` counts alone.
pub fn glm5_next_mtp_layer_index(config: &ModelConfig) -> usize {
    config.num_hidden_layers
}

fn text_config(raw: &serde_json::Value) -> &serde_json::Value {
    raw.get("text_config").unwrap_or(raw)
}

/// 2026-09-26: GLM's `layer_types` name for its sparse-MLA mixer. It parses to
/// [`LayerType::SparseAttention`], whose `hf_name()` returns this string.
pub const GLM5NEXT_SPARSE_ATTN: &str = "deepseek_sparse_attention";

/// 2026-09-26: GLM-5.3's `vision_config` as a `VisionConfig`.
///
/// `parse_vision_config` is not used: it leaves the GLM-specific fields at
/// `VisionConfig::default()`, i.e. `rms_norm_eps = 1e-6`, `swiglu_limit = 0` and no merger
/// width.
///
/// Returns `None` when a tower dimension or `projection_intermediate_size` is missing. The
/// other fields fall back to the constants below when absent.
fn parse_glm5_next_vision(raw: &serde_json::Value) -> Option<super::super::VisionConfig> {
    let vc = raw.get("vision_config")?;
    let u = |k: &str| {
        vc.get(k)
            .and_then(serde_json::Value::as_u64)
            .map(|v| v as usize)
    };
    let f = |k: &str| {
        vc.get(k)
            .and_then(serde_json::Value::as_f64)
            .map(|v| v as f32)
    };
    let rope_theta = vc
        .get("rope_parameters")
        .and_then(|r| r.get("rope_theta"))
        .and_then(serde_json::Value::as_f64)
        .map(|v| v as f32)
        .unwrap_or(10_000.0);
    let stats = |k: &str, fallback: [f32; 3]| -> [f32; 3] {
        let Some(arr) = vc.get(k).and_then(serde_json::Value::as_array) else {
            return fallback;
        };
        let v: Vec<f32> = arr
            .iter()
            .filter_map(serde_json::Value::as_f64)
            .map(|x| x as f32)
            .collect();
        match v.len() {
            3 => [v[0], v[1], v[2]],
            _ => fallback,
        }
    };
    Some(super::super::VisionConfig {
        depth: u("depth")?,
        hidden_size: u("hidden_size")?,
        num_heads: u("num_heads")?,
        patch_size: u("patch_size")?,
        temporal_patch_size: u("temporal_patch_size")?,
        spatial_merge_size: u("spatial_merge_size")?,
        intermediate_size: u("intermediate_size")?,
        out_hidden_size: u("out_hidden_size")?,
        deepstack_visual_indexes: Vec::new(),
        image_pad_token_id: raw
            .get("image_token_id")
            .and_then(serde_json::Value::as_u64)
            .unwrap_or(0) as u32,
        video_pad_token_id: raw
            .get("video_token_id")
            .and_then(serde_json::Value::as_u64)
            .unwrap_or(0) as u32,
        // 2026-09-26: Filled by the server before the encoder is built, as in
        // `parse_vision_config`.
        max_pixels: None,
        hidden_act: vc
            .get("hidden_act")
            .and_then(serde_json::Value::as_str)
            .unwrap_or("silu")
            .to_string(),
        rms_norm_eps: f("rms_norm_eps").unwrap_or(1e-5),
        attention_bias: vc
            .get("attention_bias")
            .and_then(serde_json::Value::as_bool)
            .unwrap_or(true),
        projection_intermediate_size: u("projection_intermediate_size")?,
        swiglu_limit: f("swiglu_limit").unwrap_or(10.0),
        rope_theta,
        // 2026-09-26: The fallback mean and std are 0.48145466, 0.4578275, 0.40821073 and
        // 0.26862954, 0.26130258, 0.27577711, written as the shortest decimals that give the
        // same `f32`. `VisionConfig::default()` holds 0.5 on every channel.
        image_mean: stats("image_mean", [0.481_454_67, 0.457_827_5, 0.408_210_72]),
        image_std: stats("image_std", [0.268_629_55, 0.261_302_6, 0.275_777_1]),
        min_image_tokens: u("min_image_tokens").unwrap_or(16),
        max_image_tokens: u("max_image_tokens").unwrap_or(8000),
        block_major_patches: true,
    })
}

/// 2026-09-26: The layers whose MLP is dense rather than routed.
///
/// With `first_k_dense_replace = k` the answer is layers `0..k`, and a `mlp_layer_types` array
/// that disagrees is an error. Without it, the indices of the array's non-`"sparse"` entries
/// are used. With neither, this is an error.
fn build_mlp_only_layers(text: &serde_json::Value, n_layers: usize) -> Result<Vec<usize>> {
    let first_k = text
        .get("first_k_dense_replace")
        .and_then(|v| v.as_u64())
        .map(|v| v as usize);
    let textual: Option<Vec<usize>> =
        text.get("mlp_layer_types")
            .and_then(|v| v.as_array())
            .map(|a| {
                a.iter()
                    .enumerate()
                    .filter(|(_, v)| v.as_str() != Some("sparse"))
                    .map(|(i, _)| i)
                    .collect()
            });

    match (first_k, textual) {
        (Some(k), t) => {
            if k > n_layers {
                bail!("glm5_next: first_k_dense_replace {k} exceeds num_hidden_layers {n_layers}");
            }
            let derived: Vec<usize> = (0..k).collect();
            if let Some(t) = t
                && t != derived
            {
                bail!(
                    "glm5_next: first_k_dense_replace={k} implies dense layers \
                     {derived:?}, but mlp_layer_types says {t:?}"
                );
            }
            Ok(derived)
        }
        (None, Some(t)) => Ok(t),
        (None, None) => bail!(
            "glm5_next: neither first_k_dense_replace nor mlp_layer_types present; \
             refusing to guess which layers are dense"
        ),
    }
}

/// 2026-09-26: The per-layer mixer map.
///
/// When `linear_attn_config` has both `kda_layers` and `full_attn_layers`, they decide it,
/// and a `layer_types` array with `n_layers` entries must agree. Otherwise `layer_types`
/// decides it alone. No index arithmetic (such as `layer % 4 == 3`) is assumed.
fn build_layer_types(text: &serde_json::Value, n_layers: usize) -> Result<Vec<LayerType>> {
    let idx_list = |key: &str| -> Option<Vec<usize>> {
        text.get("linear_attn_config")?
            .get(key)?
            .as_array()
            .map(|a| {
                a.iter()
                    .filter_map(|v| v.as_u64())
                    .map(|v| v as usize)
                    .collect()
            })
    };

    let kda = idx_list("kda_layers");
    let full = idx_list("full_attn_layers");

    let mut types = vec![LayerType::FullAttention; n_layers];
    match (kda, full) {
        (Some(kda), Some(full)) => {
            if kda.len() + full.len() != n_layers {
                bail!(
                    "glm5_next: kda_layers ({}) + full_attn_layers ({}) != num_hidden_layers ({})",
                    kda.len(),
                    full.len(),
                    n_layers
                );
            }
            for i in &kda {
                if *i >= n_layers {
                    bail!("glm5_next: kda_layers index {i} out of range for {n_layers} layers");
                }
                types[*i] = LayerType::LinearAttention;
            }
            // 2026-09-26: `full_attn_layers` lists the non-KDA layers; the index list cannot
            // tell sparse from dense attention, so the textual array decides the variant.
            let textual = text.get("layer_types").and_then(|v| v.as_array());
            for i in &full {
                if *i >= n_layers {
                    bail!("glm5_next: full_attn_layers index {i} out of range");
                }
                if types[*i] == LayerType::LinearAttention {
                    bail!("glm5_next: layer {i} listed as BOTH kda and full attention");
                }
                types[*i] = match textual.and_then(|a| a.get(*i)).and_then(|v| v.as_str()) {
                    Some(GLM5NEXT_SPARSE_ATTN) => LayerType::SparseAttention,
                    _ => LayerType::FullAttention,
                };
            }
        }
        _ => {
            let arr = text
                .get("layer_types")
                .and_then(|v| v.as_array())
                .context("glm5_next: neither linear_attn_config lists nor layer_types present")?;
            if arr.len() != n_layers {
                bail!(
                    "glm5_next: layer_types has {} entries, expected {n_layers}",
                    arr.len()
                );
            }
            for (i, v) in arr.iter().enumerate() {
                types[i] = match v.as_str().unwrap_or("") {
                    "linear_attention" => LayerType::LinearAttention,
                    GLM5NEXT_SPARSE_ATTN => LayerType::SparseAttention,
                    "full_attention" => LayerType::FullAttention,
                    other => bail!("glm5_next: unknown layer_type {other:?} at layer {i}"),
                };
            }
        }
    }

    if let Some(arr) = text.get("layer_types").and_then(|v| v.as_array())
        && arr.len() == n_layers
    {
        for (i, v) in arr.iter().enumerate() {
            let want = match v.as_str().unwrap_or("") {
                "linear_attention" => LayerType::LinearAttention,
                GLM5NEXT_SPARSE_ATTN => LayerType::SparseAttention,
                _ => LayerType::FullAttention,
            };
            if types[i] != want {
                bail!(
                    "glm5_next: layer {i} disagrees — index lists say {:?}, layer_types says {want:?}",
                    types[i]
                );
            }
        }
    }
    Ok(types)
}

/// 2026-09-26: Refuse a config whose DSA layers use shared indexing.
///
/// `indexer_types[i]` is `"full"` (the layer runs its own DSA indexer) or `"shared"` (it
/// reuses the previous full layer's top-k selection). The engine runs a per-layer indexer
/// and does not propagate selections, so a shared DSA layer would attend to the wrong token
/// set without any error; it is refused here. A `"shared"` entry on a KDA layer is ignored.
/// Nothing is stored in the config.
///
/// Without `indexer_types`, the modes come from `index_topk_pattern` (one `F` or `S` per
/// layer), else layer `i` is full when `max(i - offset + 1, 0) % freq == 0`, with
/// `freq = max(index_topk_freq, 1)` (default 1) and `offset = index_skip_topk_offset`
/// (default 2).
fn refuse_shared_indexer(text: &serde_json::Value, config: &ModelConfig) -> Result<()> {
    let n = config.num_hidden_layers;
    let modes: Vec<String> = if let Some(arr) = text.get("indexer_types").and_then(|v| v.as_array())
    {
        if arr.len() != n {
            bail!(
                "indexer_types has {} entries for {n} layers; a length mismatch would \
                 silently misalign every layer's indexer mode",
                arr.len()
            );
        }
        arr.iter()
            .map(|v| {
                v.as_str()
                    .map(str::to_string)
                    .context("indexer_types entry is not a string")
            })
            .collect::<Result<_>>()?
    } else if let Some(pat) = text.get("index_topk_pattern").and_then(|v| v.as_str()) {
        if pat.chars().count() != n {
            bail!(
                "index_topk_pattern has {} chars for {n} layers",
                pat.chars().count()
            );
        }
        pat.chars()
            .map(|c| match c {
                'F' => Ok("full".to_string()),
                'S' => Ok("shared".to_string()),
                other => bail!("index_topk_pattern: unknown char {other:?}, expected F or S"),
            })
            .collect::<Result<_>>()?
    } else {
        let freq = text
            .get("index_topk_freq")
            .and_then(|v| v.as_u64())
            .unwrap_or(1)
            .max(1) as i64;
        let offset = text
            .get("index_skip_topk_offset")
            .and_then(|v| v.as_i64())
            .unwrap_or(2);
        (0..n)
            .map(|i| {
                let shifted = ((i as i64) - offset + 1).max(0);
                if shifted % freq == 0 {
                    "full"
                } else {
                    "shared"
                }
                .to_string()
            })
            .collect()
    };

    for (i, m) in modes.iter().enumerate() {
        if m != "full" && m != "shared" {
            bail!("layer {i}: unknown indexer mode {m:?}, expected \"full\" or \"shared\"");
        }
    }

    let shared: Vec<usize> = config
        .layer_types
        .iter()
        .enumerate()
        .filter(|(i, t)| {
            **t != LayerType::LinearAttention && modes.get(*i).is_some_and(|m| m == "shared")
        })
        .map(|(i, _)| i)
        .collect();
    if !shared.is_empty() {
        bail!(
            "DSA layer(s) {shared:?} use SHARED indexing (reuse the previous full layer's \
             top-k). Metrale Engine runs a per-layer indexer and does not propagate selections, so \
             these layers would attend to the wrong token set — a wrong answer, not a \
             crash. GLM-5.3-Flash-NVFP4 is entirely \"full\"; implement prev_topk_indices \
             propagation before serving a checkpoint that is not."
        );
    }
    Ok(())
}

fn validate_glm5_next(config: &ModelConfig) -> Result<()> {
    if config.qk_rope_head_dim != 0 {
        bail!(
            "glm5_next: expected NoPE (qk_rope_head_dim == 0), got {}. \
             A non-zero value means this is not the GLM-5.3 geometry we support.",
            config.qk_rope_head_dim
        );
    }
    if config.head_dim == 0 {
        bail!("glm5_next: head_dim resolved to 0");
    }
    let linear = config
        .layer_types
        .iter()
        .filter(|t| **t == LayerType::LinearAttention)
        .count();
    let full = config.layer_types.len() - linear;
    if linear == 0 || full == 0 {
        bail!("glm5_next: degenerate layer map — {linear} linear / {full} full");
    }
    Ok(())
}

#[cfg(test)]
mod tests;
