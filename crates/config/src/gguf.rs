// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-26: Build a [`ModelConfig`] from GGUF metadata key/values.
//!
//! The keys become an HF-config-shaped JSON object that serde reads into `ModelConfig`. The
//! fields the JSON cannot carry (`model_type`, `attn_gated`, `weight_prefix`, the gemma and
//! DeepSeek-V4.1 fields) are then set directly, and `finalize_config` validates the result.
//! `arch_to_model_type` lists the accepted `general.architecture` values. The metadata is read
//! through [`GgufMeta`], which metrale-model-weights implements for its `GgufFile`.
//!
//! Owner: config.
//! Invariants: none beyond the types.

use anyhow::{Context, Result, bail};
use serde_json::{Map, Value, json};

use super::{ModelConfig, finalize_config};

/// 2026-09-26: Typed read access to GGUF metadata. Integer getters widen to u64 and float
/// getters to f64. A getter returns `None` when the key is absent or holds another value
/// type; the builder decides whether absence is an error or has a fallback.
pub trait GgufMeta {
    fn get_u64(&self, key: &str) -> Option<u64>;
    fn get_f64(&self, key: &str) -> Option<f64>;
    fn get_str(&self, key: &str) -> Option<&str>;
    fn get_arr_len(&self, key: &str) -> Option<usize>;
    /// 2026-09-26: Default `None`. A reader without it cannot build a DeepSeek-V4.1 config,
    /// which requires `attention.compress_ratios` and the `engram.*` integer arrays.
    fn get_u64_arr(&self, _key: &str) -> Option<Vec<u64>> {
        None
    }
    /// 2026-09-26: Default `None`. DeepSeek-V4.1 reads `swiglu_clamp_exp` through it when the
    /// key is present.
    fn get_f64_arr(&self, _key: &str) -> Option<Vec<f64>> {
        None
    }
}

/// 2026-09-26: Inputs to [`config_from_gguf`]: the metadata reader plus two facts from the
/// tensor section.
pub struct GgufConfigInputs<'a> {
    pub meta: &'a dyn GgufMeta,
    /// 2026-09-26: Rows of `token_embd.weight`: the vocab size when `{arch}.vocab_size` is
    /// absent, and checked against it when present. `None` if the loader could not read
    /// the tensor shape.
    pub token_embd_vocab: Option<usize>,
    /// 2026-09-26: An `output.weight` tensor means an untied LM head; without one the head
    /// ties to the input embeddings.
    pub has_output_weight: bool,
}

/// 2026-09-26: Map `general.architecture` to a `model_type` and whether attention Q is gated.
/// An unmapped architecture is an error.
fn arch_to_model_type(arch: &str) -> Result<(&'static str, bool)> {
    Ok(match arch {
        "llama" => ("mistral", false),
        "qwen2" => ("mistral", false),
        "qwen3" => ("qwen3_5", false),
        "qwen3moe" => ("qwen3_5_moe", false),
        "gemma" | "gemma2" | "gemma3" | "gemma4" => ("gemma4", false),
        "deepseek41" => ("deepseek_v41", false),
        other => bail!(
            "GGUF general.architecture '{other}' has no Metrale Engine model_type mapping. \
             Supported GGUF arches: llama, qwen2, qwen3, qwen3moe, gemma/gemma2/gemma3/gemma4, deepseek41."
        ),
    })
}

/// 2026-09-26: Build a [`ModelConfig`] from GGUF metadata and validate it with
/// `finalize_config`.
pub fn config_from_gguf(inputs: &GgufConfigInputs) -> Result<ModelConfig> {
    let meta = inputs.meta;

    let arch = meta
        .get_str("general.architecture")
        .context("GGUF metadata missing required key 'general.architecture'")?
        .to_string();
    let (model_type, attn_gated) = arch_to_model_type(&arch)?;

    let k = |suffix: &str| format!("{arch}.{suffix}");
    let req_u64 = |suffix: &str| -> Result<u64> {
        meta.get_u64(&k(suffix))
            .with_context(|| format!("GGUF metadata missing required key '{arch}.{suffix}'"))
    };

    let hidden_size = req_u64("embedding_length")? as usize;
    let num_hidden_layers = req_u64("block_count")? as usize;
    // 2026-09-26: `feed_forward_length` is the dense FFN width. It is required only when
    // the file declares no experts; a MoE file without it gets 0.
    let has_experts = meta.get_u64(&k("expert_count")).unwrap_or(0) > 0;
    let intermediate_size = match meta.get_u64(&k("feed_forward_length")) {
        Some(v) => v as usize,
        None if has_experts => 0,
        None => bail!("GGUF metadata missing required key '{arch}.feed_forward_length'"),
    };
    let num_attention_heads = req_u64("attention.head_count")? as usize;

    // 2026-09-26: An absent `attention.head_count_kv` means one KV head per attention head.
    let num_key_value_heads = meta
        .get_u64(&k("attention.head_count_kv"))
        .map(|v| v as usize)
        .unwrap_or(num_attention_heads);
    if num_attention_heads > 0
        && (num_key_value_heads == 0 || !num_attention_heads.is_multiple_of(num_key_value_heads))
    {
        bail!(
            "GGUF metadata key '{}.attention.head_count_kv' ({num_key_value_heads}) must be a non-zero divisor of attention.head_count ({num_attention_heads})",
            arch
        );
    }

    let head_dim = match meta.get_u64(&k("attention.key_length")) {
        Some(0) => bail!(
            "GGUF metadata key '{}.attention.key_length' must be greater than zero",
            arch
        ),
        Some(v) => v as usize,
        None => {
            if num_attention_heads == 0 || !hidden_size.is_multiple_of(num_attention_heads) {
                bail!(
                    "GGUF: cannot derive head_dim — '{arch}.attention.key_length' absent and \
                     hidden_size ({hidden_size}) not divisible by head_count ({num_attention_heads})"
                );
            }
            hidden_size / num_attention_heads
        }
    };

    // 2026-09-26: vocab_size is `{arch}.vocab_size`, else the `token_embd.weight` rows, else
    // the `tokenizer.ggml.tokens` length. The first two must agree when both exist.
    let metadata_vocab = meta.get_u64(&k("vocab_size")).map(|v| v as usize);
    if let (Some(metadata_vocab), Some(tensor_vocab)) = (metadata_vocab, inputs.token_embd_vocab)
        && metadata_vocab != tensor_vocab
    {
        bail!(
            "GGUF: '{arch}.vocab_size' ({metadata_vocab}) does not match token_embd.weight rows \
             ({tensor_vocab})"
        );
    }
    let vocab_size = metadata_vocab
        .or(inputs.token_embd_vocab)
        .or_else(|| meta.get_arr_len("tokenizer.ggml.tokens"))
        .context(
            "GGUF: could not determine vocab_size (no '{arch}.vocab_size', no token_embd rows, \
             no 'tokenizer.ggml.tokens')",
        )?;
    if vocab_size == 0 {
        bail!("GGUF: vocab_size must be non-zero");
    }

    // 2026-09-26: An absent `attention.layer_norm_rms_epsilon` gives 1e-5, set here instead
    // of the serde default `default_rms_eps()` (1e-6).
    let rms_norm_eps = meta
        .get_f64(&k("attention.layer_norm_rms_epsilon"))
        .unwrap_or(1e-5);
    let rope_theta = meta.get_f64(&k("rope.freq_base")).unwrap_or(10_000.0);
    let max_position_embeddings = req_u64("context_length")? as usize;

    let bos_token_id = meta.get_u64("tokenizer.ggml.bos_token_id").unwrap_or(0);
    let eos_token_id = meta.get_u64("tokenizer.ggml.eos_token_id").unwrap_or(0);

    let tie_word_embeddings = !inputs.has_output_weight;

    let num_experts = if arch == "qwen3moe" {
        req_u64("expert_count")? as usize
    } else {
        meta.get_u64(&k("expert_count"))
            .map(|v| v as usize)
            .unwrap_or(0)
    };
    if arch == "qwen3moe" && num_experts == 0 {
        bail!("GGUF metadata key '{arch}.expert_count' must be greater than zero");
    }

    let mut body: Map<String, Value> = json!({
        "hidden_size": hidden_size,
        "num_hidden_layers": num_hidden_layers,
        "intermediate_size": intermediate_size,
        "vocab_size": vocab_size,
        "num_attention_heads": num_attention_heads,
        "num_key_value_heads": num_key_value_heads,
        "head_dim": head_dim,
        "rms_norm_eps": rms_norm_eps,
        "rope_theta": rope_theta,
        "max_position_embeddings": max_position_embeddings,
        "bos_token_id": bos_token_id,
        "eos_token_id": eos_token_id,
        "tie_word_embeddings": tie_word_embeddings,
        "model_type": model_type,
    })
    .as_object()
    .expect("json! object literal")
    .clone();

    if num_experts > 0 {
        let experts_per_tok = req_u64("expert_used_count").with_context(|| {
            format!("GGUF: MoE arch '{arch}' has expert_count>0 but no '{arch}.expert_used_count'")
        })? as usize;
        let moe_ffn = req_u64("expert_feed_forward_length").with_context(|| {
            format!("GGUF: MoE arch '{arch}' missing '{arch}.expert_feed_forward_length'")
        })? as usize;
        if experts_per_tok == 0 || experts_per_tok > num_experts {
            bail!(
                "GGUF metadata key '{arch}.expert_used_count' must be in 1..={num_experts}, \
                 found {experts_per_tok}"
            );
        }
        if moe_ffn == 0 {
            bail!(
                "GGUF metadata key '{arch}.expert_feed_forward_length' must be greater than zero"
            );
        }
        body.insert("num_experts".into(), json!(num_experts));
        body.insert("num_experts_per_tok".into(), json!(experts_per_tok));
        body.insert("moe_intermediate_size".into(), json!(moe_ffn));
    }

    if let Some(sw) = meta.get_u64(&k("attention.sliding_window")) {
        body.insert("sliding_window".into(), json!(sw));
    }

    let raw = Value::Object(body);
    let json_str = serde_json::to_string(&raw).context("serialize synthesized GGUF config")?;
    let mut config: ModelConfig =
        serde_json::from_str(&json_str).context("deserialize synthesized GGUF config")?;

    config.model_type = model_type.to_string();
    config.attn_gated = attn_gated;
    // 2026-09-26: The GGUF name map emits HF names under `model.` (`model.embed_tokens.weight`,
    // `model.layers.N.*`). `layer_prefix()` gives `model.layers.N` for both "" and "model",
    // but the embedding and norm loaders format `weight_prefix` as is, so it must be "model".
    config.weight_prefix = "model".to_string();

    if model_type == "gemma4" {
        config.embed_scale = (hidden_size as f32).sqrt();
        config.final_logit_softcapping = match meta.get_f64(&k("final_logit_softcapping")) {
            Some(v) if v >= 0.0 && v <= f32::MAX as f64 => v as f32,
            Some(v) => bail!(
                "GGUF metadata key '{}.final_logit_softcapping' must be non-negative and representable as a finite f32 (got {v})",
                arch
            ),
            None => 0.0,
        };
    }

    // 2026-09-26: DeepSeek-V4.1 fields, from the `deepseek41.*` keys. An absent key is an
    // error, except `hash_layer_count` (0) and `swiglu_clamp_exp` (the clamp stays unset);
    // the `engram.*` keys past `engram.layer_ids` are read only when that list is non-empty.
    if model_type == "deepseek_v41" {
        let req = |suffix: &str| -> Result<u64> {
            meta.get_u64(&k(suffix))
                .with_context(|| format!("DeepSeek-V4.1 GGUF missing '{arch}.{suffix}'"))
        };
        let req_arr = |suffix: &str| -> Result<Vec<u64>> {
            meta.get_u64_arr(&k(suffix))
                .with_context(|| format!("DeepSeek-V4.1 GGUF missing array '{arch}.{suffix}'"))
        };

        config.q_lora_rank = req("attention.q_lora_rank")? as usize;
        config.o_lora_rank = req("attention.output_lora_rank")? as usize;
        config.o_groups = req("attention.output_group_count")? as usize;
        config.kv_lora_rank = req("attention.value_length")? as usize;
        config.rotary_dim = req("rope.dimension_count")? as usize;

        config.index_n_heads = req("attention.indexer.head_count")? as usize;
        config.index_head_dim = req("attention.indexer.key_length")? as usize;
        config.index_topk = req("attention.indexer.top_k")? as usize;
        config.compress_ratios = req_arr("attention.compress_ratios")?
            .into_iter()
            .map(|v| v as usize)
            .collect();
        // 2026-09-26: `compress_ratios` may have more entries than `block_count`; only fewer
        // is an error.
        if config.compress_ratios.len() < num_hidden_layers {
            bail!(
                "DeepSeek-V4.1: '{arch}.attention.compress_ratios' has {} entries, fewer than \
                 block_count ({num_hidden_layers})",
                config.compress_ratios.len()
            );
        }
        config.compress_rope_theta = meta
            .get_f64(&k("attention.compress_rope_freq_base"))
            .context("DeepSeek-V4.1 GGUF missing 'attention.compress_rope_freq_base'")?
            as f32;

        config.hc_mult = req("hyper_connection.count")? as usize;
        config.hc_sinkhorn_iters = req("hyper_connection.sinkhorn_iterations")? as usize;
        config.hc_eps = meta
            .get_f64(&k("hyper_connection.epsilon"))
            .context("DeepSeek-V4.1 GGUF missing 'hyper_connection.epsilon'")?
            as f32;

        config.n_routed_experts = num_experts;
        config.norm_topk_prob = req("expert_weights_norm")? != 0;
        config.routed_scaling_factor = meta
            .get_f64(&k("expert_weights_scale"))
            .context("DeepSeek-V4.1 GGUF missing 'expert_weights_scale'")?;
        config.scoring_func = match req("expert_gating_func")? {
            4 => "sqrtsoftplus".to_string(),
            other => bail!(
                "DeepSeek-V4.1: unsupported '{arch}.expert_gating_func' ({other}); \
                 only 4 (sqrt-softplus) is implemented"
            ),
        };
        config.num_hash_layers = meta.get_u64(&k("hash_layer_count")).unwrap_or(0) as usize;

        // 2026-09-26: `swiglu_clamp_exp` is per layer and `ModelConfig` carries one
        // `swiglu_limit`, so a non-uniform array is an error rather than element 0.
        if let Some(clamps) = meta.get_f64_arr(&k("swiglu_clamp_exp")) {
            let first = clamps.first().copied().unwrap_or(0.0);
            if clamps.iter().any(|v| (v - first).abs() > f64::EPSILON) {
                bail!(
                    "DeepSeek-V4.1: '{arch}.swiglu_clamp_exp' is not uniform across layers; \
                     ModelConfig carries a single `swiglu_limit`"
                );
            }
            config.swiglu_limit = first as f32;
        }

        config.engram_layer_ids = req_arr("engram.layer_ids")?
            .into_iter()
            .map(|v| v as usize)
            .collect();
        if !config.engram_layer_ids.is_empty() {
            config.engram_max_ngram_size = req("engram.max_ngram_size")? as usize;
            config.engram_n_heads = req("engram.head_count")? as usize;
            config.engram_head_dim = req("engram.key_length")? as usize;
            config.engram_pad_token_id = req("engram.pad_id")? as u32;
            config.engram_multipliers = req_arr("engram.multipliers")?;
            config.engram_primes = req_arr("engram.primes")?;
            config.engram_offsets = req_arr("engram.offsets")?;
            if meta.get_arr_len(&k("engram.token_map")).unwrap_or(0) != vocab_size {
                bail!(
                    "DeepSeek-V4.1: '{arch}.engram.token_map' length does not match vocab_size \
                     ({vocab_size})"
                );
            }
        }
    }

    finalize_config(&mut config, &raw)?;
    Ok(config)
}

#[cfg(test)]
mod tests;
