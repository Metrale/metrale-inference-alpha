// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-25: The embedding-scale and logit-softcap launches, and the
//! accessors for the single-token decode logits buffer.
//!
//! Owner: model-engine.
//! Invariants: none beyond the types.

#![allow(unused_imports, dead_code)]

use parking_lot::Mutex;
use std::collections::HashMap;
use std::sync::Arc;

use anyhow::{Result, bail};
use metrale_cache::kv_cache::PagedKvCache;
use metrale_config::{LayerType, ModelConfig};
use metrale_gpu_runtime::buffers::BufferArena;
use metrale_gpu_runtime::gpu::{DevicePtr, GpuBackend, GraphHandle, KernelHandle};

use super::block_mgmt::{
    apply_evicted_blocks, ensure_blocks_through_decode, ensure_blocks_through_prefill,
    extract_layer_refs, reuse_prefix_match_disk_ids,
};
use super::ssm_pool::SsmStatePool;
use super::ssm_snapshot::SsmSnapshotPool;
use super::types::{PinnedMetaStaging, TransformerModel};
use crate::traits::{ChunkedPrefillPageMetadata, Model, SequenceState};
use metrale_model_layers::layer::{
    AttnMetadataDev, ForwardContext, GdnPrefillBuffers, LayerState, SsmLayerState, TransformerLayer,
};
use metrale_model_layers::layers::ops;
use metrale_model_layers::speculative::DraftProposer;
use metrale_model_layers::weight_map::{DenseWeight, MtpWeights, QuantizedWeight};

impl TransformerModel {
    /// 2026-09-25: Scale `num_tokens` BF16 embedding rows in place by
    /// `config.embed_scale`; a no-op when the scale kernel is not loaded.
    pub(super) fn scale_embeddings(
        &self,
        data: DevicePtr,
        num_tokens: usize,
        stream: u64,
    ) -> Result<()> {
        self.scale_embeddings_bf16(data, num_tokens, stream)
    }

    pub(super) fn scale_embeddings_bf16(
        &self,
        data: DevicePtr,
        num_tokens: usize,
        stream: u64,
    ) -> Result<()> {
        if self.embed_scale_kernel.0 == 0 {
            return Ok(());
        }
        use metrale_gpu_runtime::kernel_args::KernelLaunch;
        let n = (num_tokens * self.config.hidden_size) as u32;
        KernelLaunch::new(self.gpu.as_ref(), self.embed_scale_kernel)
            .grid([n.div_ceil(256), 1, 1])
            .block([256, 1, 1])
            .arg_ptr(data)
            .arg_u32(n)
            .arg_f32(self.config.embed_scale)
            .launch(stream)
    }

    /// 2026-09-25: Softcap BF16 logits in place: `logits[i] = cap * tanh(logits[i] / cap)`.
    /// The caller must hold a loaded `logit_softcap_kernel`;
    /// `apply_logit_softcap_dtype` checks instead.
    pub(super) fn apply_logit_softcap(
        &self,
        logits: DevicePtr,
        num_elements: u32,
        cap: f32,
        stream: u64,
    ) -> Result<()> {
        use metrale_gpu_runtime::kernel_args::KernelLaunch;
        let inv_cap = 1.0f32 / cap;
        KernelLaunch::new(self.gpu.as_ref(), self.logit_softcap_kernel)
            .grid([num_elements.div_ceil(256), 1, 1])
            .block([256, 1, 1])
            .arg_ptr(logits)
            .arg_u32(num_elements)
            .arg_f32(inv_cap)
            .arg_f32(cap)
            .launch(stream)
    }

    /// 2026-09-25: Softcap `logits` in place with the BF16 or the FP32 kernel,
    /// per `is_fp32`. A no-op when that kernel is not loaded: the BF16 one is
    /// loaded only when the config sets `final_logit_softcapping`, the FP32 one
    /// never.
    pub(super) fn apply_logit_softcap_dtype(
        &self,
        logits: DevicePtr,
        num_elements: u32,
        cap: f32,
        is_fp32: bool,
        stream: u64,
    ) -> Result<()> {
        use metrale_gpu_runtime::kernel_args::KernelLaunch;
        let kernel = if is_fp32 {
            self.logit_softcap_fp32_kernel
        } else {
            self.logit_softcap_kernel
        };
        if kernel.0 == 0 {
            return Ok(());
        }
        let inv_cap = 1.0f32 / cap;
        KernelLaunch::new(self.gpu.as_ref(), kernel)
            .grid([num_elements.div_ceil(256), 1, 1])
            .block([256, 1, 1])
            .arg_ptr(logits)
            .arg_u32(num_elements)
            .arg_f32(inv_cap)
            .arg_f32(cap)
            .launch(stream)
    }

    /// 2026-09-25: Whether the single-token decode `lm_head` writes FP32 logits
    /// to `logits_fp32_buf` (`use_fp32_logits`, which `TransformerModel::new`
    /// sets to false). Other lm_head paths always write BF16.
    pub fn decode_logits_fp32(&self) -> bool {
        self.use_fp32_logits
    }

    /// 2026-09-25: The buffer the single-token decode `lm_head` writes:
    /// `logits_fp32_buf` when `use_fp32_logits`, else the shared BF16 logits
    /// buffer. Read it with the dtype `decode_logits_fp32` reports.
    pub fn decode_logits_ptr(&self) -> DevicePtr {
        if self.use_fp32_logits {
            self.logits_fp32_buf
        } else {
            self.buffers.logits()
        }
    }
}
