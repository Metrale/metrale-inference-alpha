// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-25: `Qwen3SsmLayer`, the Gated DeltaNet (linear-attention) layer,
//! and the submodules its constructor, forward paths and tests live in.
//!
//! Owner: model-layers (qwen3_ssm).
//! Invariants: none beyond the types.

use anyhow::Result;
use metrale_cache::kv_cache::PagedKvCache;
use metrale_gpu_runtime::gpu::{DevicePtr, GpuBackend, KernelHandle};

use crate::layer::{ForwardContext, GdnPrefillBuffers, LayerState, SsmLayerState};
use crate::layers::FfnComponent;
use crate::layers::ops;
use crate::weight_map::{DenseWeight, QuantizedWeight, SsmWeights};

mod layer_struct;
mod ple_seq;
pub use layer_struct::Qwen3SsmLayer;

mod debug;
mod decode_w8a8_proj;
pub mod gdn_flags;
mod init;
mod init_fp8;
mod init_q2;
mod kernel_select;
mod lora;
mod prefill_out_w8a8;
mod prefill_w8a8;
mod rowwise_bf16;
mod ssm_forward;
pub(crate) mod ssm_h_fp16;
mod trait_decode;
mod trait_decode_batched;
mod trait_decode_batched_conv_gdn;
mod trait_decode_batched_conv_gdn_exact;
mod trait_decode_batched_conv_gdn_multi;
mod trait_decode_batched_conv_gdn_multi_exact;
mod trait_decode_batched_conv_gdn_wyn;
mod trait_decode_hc;
mod trait_decode_multi_seq;
mod trait_layer;
mod trait_prefill;
mod trait_prefill_block;
mod trait_prefill_gdn;
mod trait_prefill_hc;
mod trait_prefill_helper;
mod trait_prefill_phase1;
mod trait_prefill_phase3;
mod trait_prefill_proj;
mod trait_prefill_recur;
mod woa;

pub use gdn_flags::{
    GdnFlags, MAX_F16_TWIN_DFLASH_GAMMA, MAX_F16_TWIN_K, default_dflash_gamma,
    gdn_fused_norm_enabled, ssm_batched_recurrent_enabled, ssm_h_dtype_bits,
    ssm_h_f16_pool_enabled, ssm_h_fp16_enabled, verify_exact_enabled,
};

#[cfg(test)]
#[path = "prefill_alloc_tests.rs"]
mod prefill_alloc_tests;
#[cfg(test)]
#[path = "rowwise_alloc_tests.rs"]
mod rowwise_alloc_tests;
#[cfg(test)]
mod tests;

#[path = "hc.rs"]
mod hc;
