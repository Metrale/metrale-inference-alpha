// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-25: `Qwen4ExpWeightLoader`, the weight loader for `qwen4_exp`
//! (Qwen3.8-Flash-Next) checkpoints.
//!
//! Owner: model-arch weight loader.
//! Invariants:
//! - `load_layers` runs `audit_namespace` and `ensure_loadable` before it
//!   builds any layer.
//! - When `hc_mult > 0`, a load that succeeds has attached both mHC sites to
//!   every layer.
//!
//! The GDN and full-attention layers are built by the qwen35 loader's arms
//! (`linear_attn_arms`, `attention_arms`), keyed through
//! `config.layer_prefix(i)`. On top of them this loader adds ones-filled
//! placeholders for the per-layer norms, the multi-hyperconnection (mHC)
//! sites (`hc`), the QSA indexer (`aux::attach_qsa`), the PLE layer (`ple`)
//! and the MoE FFN (`ffn`).

use anyhow::{Context, Result};
use metrale_cache::kv_cache::KvCacheDtype;
use metrale_config::{LayerType, ModelConfig};
use metrale_gpu_runtime::gpu::GpuBackend;
use metrale_model_weights::weights::WeightStore;

use crate::weight_loader::ModelWeightLoader;
use metrale_model_layers::layer::TransformerLayer;
use metrale_model_layers::weight_map::{DenseWeight, MtpWeights, dense};

// 2026-09-25: The file is `aux_sites.rs` because `aux` is a reserved file
// name on Windows, where a checkout of `aux.rs` fails.
#[path = "qwen4_exp/aux_sites.rs"]
mod aux;
mod ffn;
mod hc;
mod ple;
mod probe;

pub use probe::audit_namespace;

/// 2026-09-25: The PLE table's shard layout, `(file, byte offset)` per shard
/// plus the rows per shard, read from a checkpoint's safetensors index and
/// headers. `ple_tests.rs` uses it to build the segmented row cache without
/// loading the model.
#[cfg(test)]
pub fn ple_shard_layout(snapshot: &str) -> Result<(Vec<(std::path::PathBuf, u64)>, u64)> {
    use std::io::{Read, Seek, SeekFrom};
    let idx: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(
        std::path::Path::new(snapshot).join("model.safetensors.index.json"),
    )?)?;
    let map = idx["weight_map"].as_object().context("weight_map")?;
    let mut names: Vec<(usize, &String)> = map
        .keys()
        .filter(|k| k.contains(".ngram_embedding.shard_"))
        .map(|k| {
            let n = k
                .rsplit("shard_")
                .next()
                .and_then(|r| r.split('.').next())
                .and_then(|r| r.parse().ok())
                .unwrap_or(usize::MAX);
            (n, k)
        })
        .collect();
    names.sort();
    anyhow::ensure!(!names.is_empty(), "no PLE shards in {snapshot}");

    // 2026-09-25: Each file's header is read once. A shard's offset is taken
    // against its own file's data start, because shards may be spread over
    // several files.
    let mut headers: std::collections::HashMap<String, (serde_json::Value, u64)> =
        std::collections::HashMap::new();
    let mut shards = Vec::with_capacity(names.len());
    let mut rows_per = 0u64;
    for (i, name) in &names {
        let file = map[name.as_str()].as_str().context("shard file")?;
        if !headers.contains_key(file) {
            let path = std::path::Path::new(snapshot).join(file);
            let mut fh = std::fs::File::open(&path)?;
            let mut len = [0u8; 8];
            fh.read_exact(&mut len)?;
            let hlen = u64::from_le_bytes(len);
            let mut hdr = vec![0u8; hlen as usize];
            fh.seek(SeekFrom::Start(8))?;
            fh.read_exact(&mut hdr)?;
            headers.insert(file.to_owned(), (serde_json::from_slice(&hdr)?, 8 + hlen));
        }
        let (hdr, data_start) = &headers[file];
        let e = &hdr[name.as_str()];
        let off = e["data_offsets"][0].as_u64().context("data_offsets")?;
        let rows = e["shape"][0].as_u64().context("shape")?;
        if *i == 0 {
            rows_per = rows;
        }
        anyhow::ensure!(
            rows == rows_per,
            "shard {i} has {rows} rows, not {rows_per}"
        );
        shards.push((std::path::Path::new(snapshot).join(file), data_start + off));
    }
    Ok((shards, rows_per))
}

pub struct Qwen4ExpWeightLoader;

impl ModelWeightLoader for Qwen4ExpWeightLoader {
    fn supports_tp(&self) -> bool {
        false
    }

    fn load_layers(
        &self,
        store: &WeightStore,
        config: &ModelConfig,
        gpu: &dyn GpuBackend,
        layer_kv_dtypes: &[KvCacheDtype],
    ) -> Result<Vec<Box<dyn TransformerLayer>>> {
        let report = audit_namespace(store, config);
        report.log();
        report.ensure_loadable()?;

        let h = config.hidden_size;
        let variant = metrale_model_layers::weight_map::detect_nvfp4_variant(store, config);
        let absmax_k = gpu.kernel("quantize_nvfp4", "nvfp4_global_absmax")?;
        let quantize_k = gpu.kernel("quantize_nvfp4", "quantize_bf16_to_nvfp4")?;
        let stream = gpu.default_stream();

        tracing::info!(
            "Qwen3.8-Flash-Next: {} layers ({} GDN + {} full attention), \
             {} experts top-{}, hc {} streams x rank {}, indexer budget {}, \
             PLE at {:?}; NVFP4 variant {:?}",
            config.num_hidden_layers,
            config
                .layer_types
                .iter()
                .filter(|t| **t == LayerType::LinearAttention)
                .count(),
            config
                .layer_types
                .iter()
                .filter(|t| **t == LayerType::FullAttention)
                .count(),
            config.num_experts,
            config.num_experts_per_tok,
            config.hc_mult,
            config.hc_lowrank,
            config.index_topk,
            config.ple_layer_ids,
            variant,
        );

        // 2026-09-25: The model-level mixer, which collapses the streams before
        // `lm_head`. Every layer gets a copy; only the last model layer uses it.
        let hc_head = if config.hc_mult > 0 {
            Some(hc::load_head(store, config)?)
        } else {
            None
        };

        // 2026-09-25: The PLE scratch is sized once, for the largest prefill
        // chunk, not the context: `METRALE_PLE_MAX_TOKENS`, default 2048. A
        // larger chunk is refused by the PLE layer with a message naming the
        // variable (`ple/layer.rs`).
        let max_ple_tokens: usize = std::env::var("METRALE_PLE_MAX_TOKENS")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(2048);
        // 2026-09-25: With `METRALE_QWEN4EXP_NO_PLE=1`, `ple::load` is not
        // called, so no row cache is opened.
        let ple_off = std::env::var("METRALE_QWEN4EXP_NO_PLE").as_deref() == Ok("1");
        // 2026-09-25: GDN layers are built with BF16 projections
        // (`build_linear_attention_dense_bf16`) unless
        // `METRALE_QWEN4EXP_BF16_GDN=0`, which requantizes them to NVFP4.
        let bf16_gdn = std::env::var("METRALE_QWEN4EXP_BF16_GDN").as_deref() != Ok("0");
        tracing::info!(
            "GDN projections: {} on the {} linear-attention layers",
            if bf16_gdn {
                "BF16 as shipped (no runtime NVFP4 requantization)"
            } else {
                "requantized to NVFP4 (METRALE_QWEN4EXP_BF16_GDN=0)"
            },
            config
                .layer_types
                .iter()
                .filter(|t| **t == LayerType::LinearAttention)
                .count(),
        );

        let mut layers: Vec<Box<dyn TransformerLayer>> =
            Vec::with_capacity(config.num_hidden_layers);
        let mut attn_idx = 0usize;

        // 2026-09-25: Free-memory deltas per part (MoE, attention/GDN arm,
        // mHC+PLE), summed over layers and logged once.
        let (mut moe_bytes, mut arm_bytes, mut hc_bytes) = (0u64, 0u64, 0u64);
        let free_now = |g: &dyn GpuBackend| g.free_memory().unwrap_or(0) as u64;

        for i in 0..config.num_hidden_layers {
            let lp = config.layer_prefix(i);
            let f0 = free_now(gpu);
            let ffn = ffn::build_moe(store, &lp, config, gpu, variant)?;
            let f1 = free_now(gpu);
            moe_bytes += f0.saturating_sub(f1);

            // 2026-09-25: Ones-filled per-layer norms for the shared arms'
            // signatures. `ensure_loadable` refuses a checkpoint that has
            // per-layer norms, and the low-rank mHC forward paths do not apply
            // these (`HcVariant::applies_block_input_norm`,
            // `qwen3_ssm/trait_decode_hc.rs`).
            let input_norm = ones_norm(h, gpu)?;
            let post_attn_norm = ones_norm(h, gpu)?;

            let layer = match config.layer_types[i] {
                LayerType::LinearAttention if bf16_gdn => {
                    crate::weight_loader::qwen35::load_layers::linear_attn_arms::build_linear_attention_dense_bf16(
                        i, store, &lp, gpu, variant, config, h,
                        input_norm, post_attn_norm, ffn,
                    )
                    .with_context(|| format!("qwen4_exp: GDN layer {i} (BF16)"))?
                }
                LayerType::LinearAttention => {
                    crate::weight_loader::qwen35::load_layers::linear_attn_arms::build_linear_attention_nvfp4(
                        store, &lp, gpu, variant, config, h, absmax_k, quantize_k, stream,
                        input_norm, post_attn_norm, ffn,
                    )
                    .with_context(|| format!("qwen4_exp: GDN layer {i}"))?
                }
                LayerType::FullAttention => {
                    let kv_dtype = layer_kv_dtypes
                        .get(attn_idx)
                        .copied()
                        .unwrap_or(KvCacheDtype::Bf16);
                    let l = crate::weight_loader::qwen35::load_layers::attention_arms::build_full_attention_nvfp4(
                        i, store, &lp, gpu, variant, config, h, absmax_k, quantize_k, stream,
                        kv_dtype, attn_idx, input_norm, post_attn_norm, ffn,
                    )
                    .with_context(|| format!("qwen4_exp: full-attention layer {i}"))?;
                    attn_idx += 1;
                    l
                }
                other => anyhow::bail!(
                    "qwen4_exp layer {i} has type {other:?}; this architecture is \
                     only linear_attention / full_attention"
                ),
            };
            let f2 = free_now(gpu);
            arm_bytes += f1.saturating_sub(f2);

            // 2026-09-25: Two mHC sites per layer, around attention and around
            // the MoE.
            let mut layer = layer;
            if config.hc_mult > 0 {
                let (attn, ffn) = hc::load_layer_sites(store, &lp, config)?;
                aux::attach_hc(&mut layer, i, attn, ffn, hc_head.clone(), config)?;
            }
            aux::attach_qsa(&mut layer, i, &lp, store, config, gpu)?;
            // 2026-09-25: `ple::load` returns `None` for a layer that
            // `ple_layer_ids` does not list; `attach_ple` refuses a non-GDN host.
            let ple_layer = if ple_off {
                None
            } else {
                ple::load(store, config, i, max_ple_tokens, gpu)?
            };
            if let Some(p) = ple_layer {
                aux::attach_ple(&mut layer, i, p)?;
            }
            layers.push(layer);
            hc_bytes += f2.saturating_sub(free_now(gpu));
        }
        tracing::info!(
            "qwen4_exp layer construction: MoE {:.2} GB ({:.1} MB/layer), \
             attn/GDN arms {:.2} GB ({:.1} MB/layer), mHC+PLE {:.2} GB",
            moe_bytes as f64 / 1e9,
            moe_bytes as f64 / 1e6 / config.num_hidden_layers as f64,
            arm_bytes as f64 / 1e9,
            arm_bytes as f64 / 1e6 / config.num_hidden_layers as f64,
            hc_bytes as f64 / 1e9,
        );

        if !config.ple_layer_ids.is_empty()
            && std::env::var("METRALE_QWEN4EXP_NO_PLE").as_deref() == Ok("1")
        {
            tracing::warn!(
                "METRALE_QWEN4EXP_NO_PLE=1: PLE n-gram injection at model layer {} \
                 is DISABLED. Output is wrong by construction — this arm exists \
                 to bisect the mHC spine, nothing else.",
                config.ple_layer_ids[0].saturating_sub(1),
            );
        }
        tracing::info!(
            "Qwen3.8-Flash-Next loaded {} layers with the mHC highway live on \
             all of them ({} GDN + {} full-attention).",
            layers.len(),
            layers.len()
                - config
                    .layer_types
                    .iter()
                    .filter(|t| **t == LayerType::FullAttention)
                    .count(),
            config
                .layer_types
                .iter()
                .filter(|t| **t == LayerType::FullAttention)
                .count(),
        );
        Ok(layers)
    }

    fn load_embedding(
        &self,
        store: &WeightStore,
        config: &ModelConfig,
        _gpu: &dyn GpuBackend,
    ) -> Result<DenseWeight> {
        let pfx = embed_prefix(config);
        dense(store, &format!("{pfx}.embed_tokens.weight")).context("qwen4_exp: embedding")
    }

    /// 2026-09-25: A ones-filled `[hidden]` placeholder
    /// (`aux::final_norm_placeholder`), which errors unless the mixer's
    /// `hc_norm` exists. The model does not apply it: the qwen4_exp config
    /// parser sets `final_norm_identity`, and the final-norm step then copies
    /// the hidden state (`impl_a3_norm.rs`).
    fn load_final_norm(
        &self,
        store: &WeightStore,
        config: &ModelConfig,
        gpu: &dyn GpuBackend,
    ) -> Result<DenseWeight> {
        aux::final_norm_placeholder(store, config, gpu)
    }

    fn load_lm_head(
        &self,
        store: &WeightStore,
        config: &ModelConfig,
        _gpu: &dyn GpuBackend,
    ) -> Result<DenseWeight> {
        if store.contains("lm_head.weight") {
            return dense(store, "lm_head.weight");
        }
        anyhow::ensure!(
            config.tie_word_embeddings,
            "qwen4_exp: no lm_head.weight and tie_word_embeddings is false"
        );
        let pfx = embed_prefix(config);
        dense(store, &format!("{pfx}.embed_tokens.weight")).context("qwen4_exp: tied lm_head")
    }

    fn load_vision_encoder(
        &self,
        store: &WeightStore,
        config: &ModelConfig,
        gpu: &dyn GpuBackend,
    ) -> Result<Option<metrale_model_layers::layers::VisionTower>> {
        crate::weight_loader::qwen35::Qwen35WeightLoader.load_vision_encoder(store, config, gpu)
    }

    fn load_mtp_weights(
        &self,
        _store: &WeightStore,
        _config: &ModelConfig,
        _gpu: &dyn GpuBackend,
    ) -> Result<Option<MtpWeights>> {
        Ok(None)
    }
}

/// 2026-09-25: A ones-filled `[n]` BF16 norm scale. BF16 1.0 is `0x3F80`, two
/// different bytes, so a byte `memset` cannot produce it.
fn ones_norm(n: usize, gpu: &dyn GpuBackend) -> Result<DenseWeight> {
    let host: Vec<u8> = std::iter::repeat_n([0x80u8, 0x3Fu8], n).flatten().collect();
    let ptr = gpu.alloc(host.len())?;
    gpu.copy_h2d(&host, ptr)?;
    Ok(DenseWeight { weight: ptr })
}

/// 2026-09-25: `config.weight_prefix`, or `model` when it is empty.
fn embed_prefix(config: &ModelConfig) -> String {
    if config.weight_prefix.is_empty() {
        "model".to_string()
    } else {
        config.weight_prefix.clone()
    }
}

/// 2026-09-25: The prefix of the model-level hyper-connection mixer.
fn mixer_prefix(config: &ModelConfig) -> String {
    format!("{}.hyper_connection_mixer", embed_prefix(config))
}
