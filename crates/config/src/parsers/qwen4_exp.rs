// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-26: `config.json` parser for Qwen3.8-Flash-Next (`model_type` `qwen4_exp` or
//! `qwen3_8_flash_next`). The reference implementation is vendored in `bench/qwen4_exp/ref/`.
//!
//! Beyond the fields the Qwen MoE models share, it reads: hyper-connections (`hc_count`
//! residual streams mixed at rank `hc_lowrank`), a QSA indexer on the full-attention layers
//! (`indexer_*` keys, stored in the `index_*` fields), n-gram PLE injection at the
//! `ple_layer_ids` layers, and an explicit `layer_types` array.
//!
//! Owner: config (model parsers).
//! Invariants:
//! - A parse that succeeds has non-zero `hidden_size`, `num_hidden_layers` and `vocab_size`,
//!   `layer_types` with `num_hidden_layers` entries, `hc_mult` and `hc_lowrank` both zero or
//!   both set, and the indexer's three sizes all zero or all set.

use anyhow::{Context, Result, ensure};
use serde_json::Value;

use super::super::ModelConfig;
use super::vision::parse_vision_config;

/// 2026-09-26: `make_ngram_vocab_size_divisible_by` when the config omits it; used only to
/// check `ngram_vocab_size_base`.
const NGRAM_VOCAB_ALIGN: usize = 128;

pub(crate) fn parse_qwen4_exp(raw: &Value) -> Result<ModelConfig> {
    let text = raw
        .get("text_config")
        .context("qwen4_exp config.json missing text_config")?;

    let mut config: ModelConfig =
        serde_json::from_value(text.clone()).context("Failed to parse qwen4_exp text_config")?;

    // 2026-09-26: The inner `model_type` is `qwen4_exp_text`; both top-level spellings become
    // `qwen4_exp`.
    config.model_type = "qwen4_exp".to_string();
    config.nested_config = true;
    // 2026-09-26: The reference router renormalizes the top-k probabilities when
    // `norm_topk_prob`, which defaults to true there (`ref/configuration_qwen4_exp.py`). The
    // local Qwen3.8-Flash-Next config omits the key, which serde would read as false.
    config.norm_topk_prob = true;
    // 2026-09-26: The local checkpoint has no final norm weight (`model.safetensors.index.json`).
    config.final_norm_identity = true;
    // 2026-09-26: Decoder tensors are under `model.language_model.` and the ViT under
    // `model.visual.` in the local checkpoint; `layer_prefix(i)` builds keys from this prefix.
    config.weight_prefix = "model.language_model".to_string();

    ensure!(
        config.hidden_size > 0,
        "qwen4_exp hidden_size must be non-zero"
    );
    ensure!(
        config.num_hidden_layers > 0,
        "qwen4_exp num_hidden_layers must be non-zero"
    );
    ensure!(
        config.vocab_size > 0,
        "qwen4_exp vocab_size must be non-zero"
    );

    // 2026-09-26: `eos_token_id` may be an array; the first id is the primary.
    if config.eos_token_id == 0
        && let Some(eos) = text.get("eos_token_id")
    {
        let primary = match eos {
            Value::Number(n) => n.as_u64(),
            Value::Array(ids) => ids.first().and_then(Value::as_u64),
            _ => None,
        };
        config.eos_token_id = primary.unwrap_or(0) as u32;
    }

    parse_rope(text, &mut config);
    parse_hyper_connections(text, &mut config)?;
    parse_indexer(text, &mut config)?;
    parse_ngram_ple(text, &mut config)?;

    // 2026-09-26: An `output_gate_type` other than empty or `none` gates the attention: q_proj
    // emits Q and the gate, `2 * num_attention_heads * head_dim` rows.
    config.attn_gated = text
        .get("output_gate_type")
        .and_then(Value::as_str)
        .is_some_and(|s| !s.is_empty() && s != "none");

    // 2026-09-26: The same field sets the GDN gated norm's activation: the reference passes
    // `output_gate_type or hidden_act` to its gated RMSNorm.
    config.gdn_norm_sigmoid = text
        .get("output_gate_type")
        .and_then(Value::as_str)
        .is_some_and(|s| s == "sigmoid");

    // 2026-09-26: `vision_config` is at the top level, not inside `text_config`.
    if raw.get("vision_config").is_some() {
        config.vision = parse_vision_config(raw);
    }

    ensure!(
        !config.layer_types.is_empty(),
        "qwen4_exp requires an explicit layer_types array (48 entries \
         interleaving linear_attention and full_attention); deriving it from \
         full_attention_interval would be a guess"
    );
    ensure!(
        config.layer_types.len() == config.num_hidden_layers,
        "qwen4_exp layer_types has {} entries but num_hidden_layers is {}",
        config.layer_types.len(),
        config.num_hidden_layers,
    );

    Ok(config)
}

/// 2026-09-26: Rope theta, partial rotary and the mRoPE fields are under `rope_parameters`,
/// which the serde pass over `text_config` does not read.
fn parse_rope(text: &Value, config: &mut ModelConfig) {
    let Some(rp) = text.get("rope_parameters") else {
        return;
    };
    if let Some(theta) = rp.get("rope_theta").and_then(Value::as_f64) {
        config.rope_theta = theta;
    }
    if let Some(prf) = rp.get("partial_rotary_factor").and_then(Value::as_f64) {
        config.partial_rotary_factor = prf;
    }
    config.mrope_interleaved = rp
        .get("mrope_interleaved")
        .and_then(Value::as_bool)
        .unwrap_or(false);
    if let Some(sec) = rp.get("mrope_section").and_then(Value::as_array) {
        for (i, v) in sec.iter().take(3).enumerate() {
            config.mrope_section[i] = v.as_u64().unwrap_or(0) as usize;
        }
    }
}

/// 2026-09-26: `hc_count` / `hc_lowrank` into `hc_mult` / `hc_lowrank`; both are set or
/// neither is.
fn parse_hyper_connections(text: &Value, config: &mut ModelConfig) -> Result<()> {
    config.hc_mult = text.get("hc_count").and_then(Value::as_u64).unwrap_or(0) as usize;
    config.hc_lowrank = text.get("hc_lowrank").and_then(Value::as_u64).unwrap_or(0) as usize;
    ensure!(
        (config.hc_mult == 0) == (config.hc_lowrank == 0),
        "qwen4_exp hyper-connection config is partial (hc_count={}, \
         hc_lowrank={}) — both must be present together",
        config.hc_mult,
        config.hc_lowrank,
    );
    Ok(())
}

/// 2026-09-26: `indexer_n_heads`, `indexer_head_dim` and `indexer_budget` into the `index_*`
/// fields (all set or all zero), and `indexer_compress_ratio` into `index_compress_ratio`.
fn parse_indexer(text: &Value, config: &mut ModelConfig) -> Result<()> {
    let g = |k: &str| text.get(k).and_then(Value::as_u64).unwrap_or(0) as usize;
    config.index_n_heads = g("indexer_n_heads");
    config.index_head_dim = g("indexer_head_dim");
    config.index_topk = g("indexer_budget");
    let ratio = g("indexer_compress_ratio");

    let present = [
        config.index_n_heads,
        config.index_head_dim,
        config.index_topk,
    ]
    .iter()
    .filter(|&&v| v > 0)
    .count();
    ensure!(
        present == 0 || present == 3,
        "qwen4_exp indexer config is partial (n_heads={}, head_dim={}, \
         budget={}) — all three must be present together",
        config.index_n_heads,
        config.index_head_dim,
        config.index_topk,
    );

    // 2026-09-26: `compress_ratios` stays empty: a non-empty value turns on
    // `compressed_attn` in the attention layer's gates, DeepSeek-V4's compressor. Up to the
    // budget, dense attention equals the reference, which selects
    // `topk(min(indexer_budget / compress_ratio, num_complete_blocks))` blocks, i.e. every
    // block when `seq_len <= indexer_budget`. See `bench/qwen4_exp/ARCHITECTURE.md` §3.
    config.index_compress_ratio = ratio;
    Ok(())
}

/// 2026-09-26: The n-gram / PLE geometry. `ngram_size` and `heads_per_ngram` are stored as
/// LongCat's `emb_neighbor_num` and `emb_split_num`; the head count is
/// `heads_per_ngram * (ngram_size - 1)`.
fn parse_ngram_ple(text: &Value, config: &mut ModelConfig) -> Result<()> {
    let g = |k: &str| text.get(k).and_then(Value::as_u64).unwrap_or(0) as usize;
    let ngram_size = g("ngram_size");
    let heads_per_ngram = g("heads_per_ngram");
    config.ngram_vocab_size_base = g("ngram_vocab_size_base");
    config.ngram_split_parts = g("split_ngram_parts");
    config.ple_conv_kernel_size = g("ple_conv_kernel_size");
    if let Some(ids) = text.get("ple_layer_ids").and_then(Value::as_array) {
        config.ple_layer_ids = ids
            .iter()
            .filter_map(Value::as_u64)
            .map(|v| v as usize)
            .collect();
    }

    if ngram_size == 0 && heads_per_ngram == 0 && config.ple_layer_ids.is_empty() {
        return Ok(());
    }

    ensure!(
        ngram_size >= 2,
        "qwen4_exp ngram_size must be >= 2, got {ngram_size}"
    );
    ensure!(
        heads_per_ngram > 0,
        "qwen4_exp heads_per_ngram must be non-zero when ngram_size is set"
    );
    config.emb_neighbor_num = ngram_size;
    config.emb_split_num = heads_per_ngram;

    // 2026-09-26: In the reference each head holds `ple_embed_dim / heads` dims
    // (`head_dim_per_ngram`). This check uses `hidden_size`, which equals `ple_embed_dim`
    // (2560, 16 heads of 160) in the local config.
    let heads = heads_per_ngram * (ngram_size - 1);
    ensure!(
        config.hidden_size.is_multiple_of(heads),
        "qwen4_exp hidden_size {} must divide evenly by the {} n-gram heads \
         (heads_per_ngram {} x (ngram_size {} - 1)) — the per-head slices are \
         concatenated, not projected",
        config.hidden_size,
        heads,
        heads_per_ngram,
        ngram_size,
    );

    ensure!(
        !config.ple_layer_ids.is_empty(),
        "qwen4_exp declares n-gram tables but no ple_layer_ids — nothing \
         would consume them"
    );
    for &l in &config.ple_layer_ids {
        ensure!(
            l < config.num_hidden_layers,
            "qwen4_exp ple_layer_ids contains layer {l} but the model has \
             only {} layers",
            config.num_hidden_layers,
        );
    }

    if config.ngram_vocab_size_base > 0 {
        let align = text
            .get("make_ngram_vocab_size_divisible_by")
            .and_then(Value::as_u64)
            .unwrap_or(NGRAM_VOCAB_ALIGN as u64) as usize;
        ensure!(
            align > 0 && config.ngram_vocab_size_base.is_multiple_of(align),
            "qwen4_exp ngram_vocab_size_base {} is not a multiple of \
             make_ngram_vocab_size_divisible_by {}",
            config.ngram_vocab_size_base,
            align,
        );
    }
    Ok(())
}

#[cfg(test)]
#[path = "qwen4_exp_tests.rs"]
mod qwen4_exp_tests;
