// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-26: The inputs `Qwen35DenseWeightLoader::load_layers` hands to its per-layer
//! helpers (`ffn_arm`, `attn_layer`, `gdn_layer`): the load-wide values in [`LoadCx`], one
//! layer's norms and FFN in [`LayerIn`], and [`Flow`], which tells the layer loop whether
//! to run its end-of-layer progress step.
//!
//! Owner: model-arch weight loader (Qwen3.5 dense).
//! Invariants:
//! - Every field of [`LoadCx`] is computed once per load, before the layer loop, and is
//!   not changed inside it.

use metrale_cache::kv_cache::KvCacheDtype;
use metrale_config::ModelConfig;
use metrale_gpu_runtime::gpu::{GpuBackend, KernelHandle};
use metrale_model_layers::layers::FfnComponent;
use metrale_model_layers::weight_map::{DenseWeight, Nvfp4Variant};
use metrale_model_weights::weights::WeightStore;

use super::fp8_residency::RouteEnv;

/// 2026-09-26: The values `load_layers` computes before its layer loop and every layer
/// reads.
pub(super) struct LoadCx<'a> {
    pub(super) store: &'a WeightStore,
    pub(super) config: &'a ModelConfig,
    pub(super) gpu: &'a dyn GpuBackend,
    pub(super) layer_kv_dtypes: &'a [KvCacheDtype],
    pub(super) variant: Nvfp4Variant,
    pub(super) absmax_k: KernelHandle,
    pub(super) quantize_k: KernelHandle,
    pub(super) stream: u64,
    pub(super) h: usize,
    pub(super) bf16_to_fp8_k: Option<KernelHandle>,
    pub(super) route_env: &'a RouteEnv,
}

/// 2026-09-26: One layer's index, prefix, norms and built FFN, which every layer arm
/// passes to its layer constructor.
pub(super) struct LayerIn<'a> {
    pub(super) i: usize,
    pub(super) lp: &'a str,
    pub(super) input_norm: DenseWeight,
    pub(super) post_attn_norm: DenseWeight,
    pub(super) ffn: FfnComponent,
}

/// 2026-09-26: `Continue` makes the layer loop skip its end-of-layer progress step for
/// this layer; `Proceed` runs it.
pub(super) enum Flow {
    Continue,
    Proceed,
}
