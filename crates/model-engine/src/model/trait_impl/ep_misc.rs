// SPDX-License-Identifier: MIT OR Apache-2.0

#![allow(unused_imports, dead_code, clippy::too_many_arguments)]

//! 2026-09-25: Thin `Model` dispatch helpers: multi-rank protocol queries, EP broadcasts, and stream and event forwarding.
//!
//! Owner: model-engine.
//! Invariants: none beyond the types.

use parking_lot::Mutex;
use std::collections::HashMap;
use std::sync::Arc;

use anyhow::{Result, bail};
use metrale_cache::kv_cache::PagedKvCache;
use metrale_config::{LayerType, ModelConfig};
use metrale_gpu_runtime::buffers::BufferArena;
use metrale_gpu_runtime::gpu::{DevicePtr, GpuBackend, GraphHandle, KernelHandle};

use super::super::block_mgmt::{
    apply_evicted_blocks, ensure_blocks_through_decode, ensure_blocks_through_prefill,
    extract_layer_refs, reuse_prefix_match_disk_ids,
};
use super::super::ssm_pool::SsmStatePool;
use super::super::ssm_snapshot::SsmSnapshotPool;
use super::super::types::{PinnedMetaStaging, TransformerModel};
use crate::traits::{ChunkedPrefillPageMetadata, Model, SequenceState};
use metrale_model_layers::layer::{
    AttnMetadataDev, ForwardContext, GdnPrefillBuffers, LayerState, SsmLayerState, TransformerLayer,
};
use metrale_model_layers::layers::ops;
use metrale_model_layers::speculative::DraftProposer;
use metrale_model_layers::weight_map::{DenseWeight, MtpWeights, QuantizedWeight};

impl TransformerModel {
    pub(super) fn ep_worker_step_dispatch(
        &self,
        slots: &mut [Option<SequenceState>],
    ) -> Result<bool> {
        self.ep_worker_step_impl(slots)
    }

    pub(super) fn is_ep_dispatch(&self) -> bool {
        // 2026-09-25: True when the multi-rank head/worker command protocol is active, for EP
        // sharding and for pure TP alike. The scheduler then skips the fused mixed forward and
        // the batched multi-stream prefill, which have no worker wire protocol.
        self.multi_rank_protocol_active()
    }

    pub(super) fn is_mla_dispatch(&self) -> bool {
        // 2026-09-25: Answers "must chunked prefill run as one chunk". The rule and its reasons
        // are on `requires_single_chunk_prefill`.
        metrale_model_layers::requires_single_chunk_prefill(
            &self.config.model_type,
            self.config.kv_lora_rank,
        )
    }

    pub(super) fn decode_logits_fp32_dispatch(&self) -> bool {
        // 2026-09-25: Gated on `use_fp32_logits`, which model construction sets to false.
        TransformerModel::decode_logits_fp32(self)
    }

    pub(super) fn decode_logits_ptr_dispatch(&self) -> DevicePtr {
        TransformerModel::decode_logits_ptr(self)
    }

    pub(super) fn ep_broadcast_cmd_dispatch(&self, cmd: u32) -> Result<()> {
        // 2026-09-25: Same gate as `ep_broadcast_seq_and_cmd`: live for EP and for pure TP.
        if self.multi_rank_protocol_active() {
            self.ep_broadcast_u32(cmd)?;
        }
        Ok(())
    }

    pub(super) fn ep_broadcast_tokens_dispatch(&self, tokens: &[u32]) -> Result<Vec<u32>> {
        // 2026-09-25: One broadcast of the whole slice; it errors when the payload exceeds the
        // scratch buffer.
        TransformerModel::ep_broadcast_tokens(self, tokens)
    }

    pub(super) fn default_stream_dispatch(&self) -> u64 {
        self.gpu.default_stream()
    }

    pub(super) fn create_stream_dispatch(&self) -> Result<u64> {
        self.gpu.create_stream()
    }

    pub(super) fn create_event_dispatch(&self) -> Result<u64> {
        self.gpu.create_event()
    }

    pub(super) fn record_event_dispatch(&self, event: u64, stream: u64) -> Result<()> {
        self.gpu.record_event(event, stream)
    }

    pub(super) fn stream_wait_event_dispatch(&self, stream: u64, event: u64) -> Result<()> {
        self.gpu.stream_wait_event(stream, event)
    }

    pub(super) fn synchronize_dispatch(&self, stream: u64) -> Result<()> {
        self.gpu.synchronize(stream)
    }
}
