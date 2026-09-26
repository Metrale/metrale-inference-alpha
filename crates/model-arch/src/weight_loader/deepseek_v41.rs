// SPDX-License-Identifier: MIT OR Apache-2.0
// provenance-id: 526f6e616c6420522e205374657369616b

//! 2026-09-25: DeepSeek-V4.1 loader: assembles
//! [`DeepSeekV41Layer`](crate::deepseek_v41_layer::DeepSeekV41Layer)s from a
//! GGUF checkpoint.
//!
//! Attention, the compressor and indexer projections, the routers, the shared
//! experts, the engram projections, the norms, the embedding and the head are
//! read from the weight store. The projections read through `resident_mat` may
//! stay raw Q2_K / Q3_K, and a Q6_K head stays raw. The routed expert stacks
//! must be deferred by the GGUF loader; they are streamed from the shard files
//! through `expert_stream`, as are the engram rows.
//!
//! The loader widens to f32 the tensors the kernels read as f32: the attention
//! sinks, the `q_norm`/`kv_norm`/compressor/indexer norm weights, the
//! compressor projections of layers with ratio > 1, and the HC mixes. It forms
//! the engram `q * k` product on the host.
//!
//! The KV and index source layers are the layers whose tensors include
//! `compressor.wkv.weight` and `indexer.wq_b.weight`. Candidate settings the
//! config leaves unset fall back to `DEFAULT_CANDIDATE_SOURCE` (after
//! `METRALE_DS41_CANDIDATE_SOURCE`), `DEFAULT_CANDIDATE_TOPK_BLOCKS` and
//! `DEFAULT_CANDIDATE_BLOCK`.
//!
//! Owner: model-arch weight loader.
//! Invariants: none beyond the types.

mod engram_q2k;
pub mod load_layers;

use anyhow::{Context, Result, ensure};
use metrale_config::ModelConfig;
use metrale_gpu_runtime::gpu::{DevicePtr, GpuBackend};
use metrale_model_weights::weights::{WeightDtype, WeightStore};

use crate::weight_loader::ModelWeightLoader;
use metrale_model_layers::layer::TransformerLayer;
use metrale_model_layers::layers::ops::ResidentMat;
use metrale_model_layers::layers::qwen3_attention::HcSiteWeights;
use metrale_model_layers::weight_map::quant_helpers::dense_auto;
use metrale_model_layers::weight_map::{DenseWeight, MtpWeights};

pub struct DeepSeekV41WeightLoader;

pub(super) const DEFAULT_CANDIDATE_SOURCE: usize = 20;
pub(super) const DEFAULT_CANDIDATE_TOPK_BLOCKS: usize = 2048;
pub(super) const DEFAULT_CANDIDATE_BLOCK: usize = 8;

pub(super) fn env_usize(key: &str, default: usize) -> usize {
    std::env::var(key)
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(default)
}

pub fn bf16_ptr(store: &WeightStore, name: &str) -> Result<DevicePtr> {
    let t = store.get(name)?;
    ensure!(
        t.dtype == WeightDtype::BF16,
        "{name}: expected bf16, got {:?}",
        t.dtype
    );
    Ok(t.ptr)
}

/// 2026-09-25: A resident projection as the store holds it: bf16, or raw
/// Q2_K / Q3_K blocks. Any other dtype is an error.
pub(super) fn resident_mat(store: &WeightStore, name: &str) -> Result<ResidentMat> {
    let t = store.get(name)?;
    match t.dtype {
        WeightDtype::BF16 => Ok(ResidentMat::Bf16(t.ptr)),
        WeightDtype::Q2K => Ok(ResidentMat::Q2K(t.ptr)),
        WeightDtype::Q3K => Ok(ResidentMat::Q3K(t.ptr)),
        d => anyhow::bail!("{name}: expected bf16, Q2_K or Q3_K, got {d:?}"),
    }
}

pub fn download_f32(gpu: &dyn GpuBackend, store: &WeightStore, name: &str) -> Result<Vec<f32>> {
    let t = store.get(name)?;
    let n = t.num_elements();
    match t.dtype {
        WeightDtype::BF16 => {
            let mut b = vec![0u8; n * 2];
            gpu.copy_d2h(t.ptr, &mut b)?;
            Ok(b.chunks_exact(2)
                .map(|c| f32::from_bits((u16::from_le_bytes([c[0], c[1]]) as u32) << 16))
                .collect())
        }
        WeightDtype::FP32 => {
            let mut b = vec![0u8; n * 4];
            gpu.copy_d2h(t.ptr, &mut b)?;
            Ok(b.chunks_exact(4)
                .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                .collect())
        }
        WeightDtype::Q2K | WeightDtype::Q3K => {
            use metrale_model_weights::weights::dequant_cpu::{GgmlType, dequant_to_f32};
            let gt = if t.dtype == WeightDtype::Q2K {
                GgmlType::Q2K
            } else {
                GgmlType::Q3K
            };
            let mut b = vec![0u8; t.byte_size()];
            gpu.copy_d2h(t.ptr, &mut b)?;
            let mut out = vec![0f32; n];
            dequant_to_f32(gt, &b, n, &mut out)
                .with_context(|| format!("{name}: CPU dequant of the resident {gt:?} blocks"))?;
            Ok(out)
        }
        other => anyhow::bail!("{name}: cannot widen {other:?} to f32"),
    }
}

pub fn upload_f32(gpu: &dyn GpuBackend, v: &[f32]) -> Result<DevicePtr> {
    let bytes: Vec<u8> = v.iter().flat_map(|x| x.to_le_bytes()).collect();
    let p = gpu.alloc(bytes.len().max(4))?;
    gpu.copy_h2d(&bytes, p)?;
    Ok(p)
}

/// 2026-09-25: A new f32 device copy of `name`, which must hold `expect`
/// elements. The store's tensor is widened on the host (`download_f32`) and
/// uploaded.
pub fn f32_ptr(
    gpu: &dyn GpuBackend,
    store: &WeightStore,
    name: &str,
    expect: usize,
) -> Result<DevicePtr> {
    let v = download_f32(gpu, store, name)?;
    ensure!(
        v.len() == expect,
        "{name}: {} elements, expected {expect}",
        v.len()
    );
    upload_f32(gpu, &v)
}

pub(super) fn hc_site(
    gpu: &dyn GpuBackend,
    store: &WeightStore,
    lp: &str,
    site: &str,
    c: &ModelConfig,
) -> Result<HcSiteWeights> {
    let hc = c.hc_mult;
    let mix_hc = (2 + hc) * hc;
    Ok(HcSiteWeights {
        hc_fn: f32_ptr(
            gpu,
            store,
            &format!("{lp}.hc_{site}_fn"),
            mix_hc * hc * c.hidden_size,
        )?,
        hc_base: f32_ptr(gpu, store, &format!("{lp}.hc_{site}_base"), mix_hc)?,
        hc_scale: f32_ptr(gpu, store, &format!("{lp}.hc_{site}_scale"), 3)?,
        lowrank: None,
    })
}

impl ModelWeightLoader for DeepSeekV41WeightLoader {
    fn supports_tp(&self) -> bool {
        false
    }

    fn load_layers(
        &self,
        store: &WeightStore,
        config: &ModelConfig,
        gpu: &dyn GpuBackend,
        layer_kv_dtypes: &[metrale_cache::kv_cache::KvCacheDtype],
    ) -> Result<Vec<Box<dyn TransformerLayer>>> {
        load_layers::load_layers(store, config, gpu, layer_kv_dtypes)
    }

    fn load_embedding(
        &self,
        store: &WeightStore,
        _config: &ModelConfig,
        gpu: &dyn GpuBackend,
    ) -> Result<DenseWeight> {
        dense_auto(store, "model.embed_tokens.weight", gpu)
    }

    fn load_final_norm(
        &self,
        store: &WeightStore,
        _config: &ModelConfig,
        gpu: &dyn GpuBackend,
    ) -> Result<DenseWeight> {
        dense_auto(store, "model.norm.weight", gpu)
    }

    fn load_lm_head(
        &self,
        store: &WeightStore,
        _config: &ModelConfig,
        gpu: &dyn GpuBackend,
    ) -> Result<DenseWeight> {
        let t = store.get("lm_head.weight")?;
        if t.dtype == WeightDtype::Q6K {
            // 2026-09-25: A Q6_K head is returned as the raw blocks, unconverted.
            return Ok(DenseWeight { weight: t.ptr });
        }
        dense_auto(store, "lm_head.weight", gpu)
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

#[cfg(all(test, feature = "cuda"))]
mod real_file_tests;
