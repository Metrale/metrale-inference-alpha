// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-26: `LoadCx`, the loop-invariant inputs of `load_layers` that its per-layer
//! builders read, and `LayerIn`, the per-layer parts a builder consumes.
//!
//! Owner: model-arch weight loader (Qwen3.5).
//! Invariants: none beyond the types.

use metrale_cache::kv_cache::KvCacheDtype;
use metrale_config::ModelConfig;
use metrale_gpu_runtime::gpu::{GpuBackend, KernelHandle};
use metrale_model_layers::layers::FfnComponent;
use metrale_model_layers::weight_map::{DenseWeight, Nvfp4Variant};
use metrale_model_weights::weights::WeightStore;

/// 2026-09-26: The values `load_layers` resolves before its layer loop, under the names it
/// gives them there.
pub(super) struct LoadCx<'a> {
    pub(super) store: &'a WeightStore,
    pub(super) config: &'a ModelConfig,
    pub(super) gpu: &'a dyn GpuBackend,
    pub(super) layer_kv_dtypes: &'a [KvCacheDtype],
    pub(super) variant: Nvfp4Variant,
    pub(super) h: usize,
    pub(super) absmax_k: KernelHandle,
    pub(super) quantize_k: KernelHandle,
    pub(super) stream: u64,
    pub(super) modelopt_mixed_precision: bool,
    pub(super) native_modelopt_ssm: bool,
}

/// 2026-09-26: The layer's norms and FFN, which the layer builder takes by value.
pub(super) struct LayerIn {
    pub(super) input_norm: DenseWeight,
    pub(super) post_attn_norm: DenseWeight,
    pub(super) ffn: FfnComponent,
}
