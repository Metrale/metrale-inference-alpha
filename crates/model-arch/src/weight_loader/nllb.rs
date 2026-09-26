// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-25: Marker loader for NLLB / M2M-100 (`model_type` `m2m_100` or `nllb`), an
//! encoder-decoder model this decoder-only loader interface cannot build.
//!
//! In a `cuda` build, model-engine's `build_model` (`factory/build.rs`) serves these model
//! types with `NllbGpuModel` and returns before `loader_for_config`; there this loader is
//! reached only when that path is bypassed.
//!
//! Owner: model-arch weight loader.
//! Invariants:
//! - `load_layers`, `load_embedding`, `load_final_norm` and `load_lm_head` always return an
//!   error; `load_mtp_weights` returns `Ok(None)` and `supports_tp` is false.

use anyhow::{Result, bail};
use metrale_cache::kv_cache::KvCacheDtype;
use metrale_config::ModelConfig;
use metrale_gpu_runtime::gpu::GpuBackend;
use metrale_model_weights::weights::WeightStore;

use crate::weight_loader::ModelWeightLoader;
use metrale_model_layers::layer::TransformerLayer;
use metrale_model_layers::weight_map::{DenseWeight, MtpWeights};

pub struct NllbWeightLoader;

impl NllbWeightLoader {
    fn unsupported() -> anyhow::Error {
        anyhow::anyhow!(
            "NLLB / m2m_100 is served by the dedicated GPU encoder-decoder runtime (metrale_model_engine::model::nllb::NllbGpuModel), which build_model selects before this loader; the generic decoder-only ModelWeightLoader pipeline cannot serve it. Reaching this loader means the dedicated serve path was bypassed."
        )
    }
}

impl ModelWeightLoader for NllbWeightLoader {
    fn supports_tp(&self) -> bool {
        false
    }

    fn load_layers(
        &self,
        _store: &WeightStore,
        _config: &ModelConfig,
        _gpu: &dyn GpuBackend,
        _layer_kv_dtypes: &[KvCacheDtype],
    ) -> Result<Vec<Box<dyn TransformerLayer>>> {
        bail!(Self::unsupported())
    }

    fn load_embedding(
        &self,
        _store: &WeightStore,
        _config: &ModelConfig,
        _gpu: &dyn GpuBackend,
    ) -> Result<DenseWeight> {
        bail!(Self::unsupported())
    }

    fn load_final_norm(
        &self,
        _store: &WeightStore,
        _config: &ModelConfig,
        _gpu: &dyn GpuBackend,
    ) -> Result<DenseWeight> {
        bail!(Self::unsupported())
    }

    fn load_lm_head(
        &self,
        _store: &WeightStore,
        _config: &ModelConfig,
        _gpu: &dyn GpuBackend,
    ) -> Result<DenseWeight> {
        bail!(Self::unsupported())
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
