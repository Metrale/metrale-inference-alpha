// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-25: The decode-rollback ring of the SSM snapshot pool (`save_decode`,
//! `restore_decode`).
//!
//! Owner: model-engine SSM snapshot pool.
//! Invariants:
//! - Ring slot `r` of pool slot `s` is flat index `s * decode_ring_slots + r`; an
//!   index outside `decode_max_seqs` x `decode_ring_slots`, or a disabled ring, is an
//!   error before any copy is queued.
//! - Save and restore copy the same widths: `min(h_stored_bytes, h_bytes)` of h and
//!   `conv_bytes` of conv per layer.

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

impl SsmSnapshotPool {
    pub(super) fn decode_rollback_enabled(&self) -> bool {
        self.decode_ring_slots > 0 && !self.decode_h_snapshots.is_empty()
    }

    /// 2026-09-25: Queue a copy of pool slot `ssm_slot`'s state into ring slot
    /// `(ssm_slot, ring_slot)`. Errors when the ring is disabled or an index is out
    /// of range, since a skipped save would leave nothing to roll back to.
    pub(super) fn save_decode(
        &self,
        ssm_slot: usize,
        ring_slot: usize,
        main_pool: &SsmStatePool,
        gpu: &dyn GpuBackend,
        stream: u64,
    ) -> Result<()> {
        let flat = self.decode_flat_index(ssm_slot, ring_slot)?;
        for i in 0..self.num_ssm_layers {
            // 2026-09-25: Copy the pool's storage width bit for bit; the ring itself
            // is strided by the FP32 width `h_bytes`.
            gpu.copy_d2d_async(
                main_pool.h_state(i, ssm_slot),
                self.decode_h_snapshots[i].offset(flat * self.h_bytes),
                main_pool.h_stored_bytes.min(self.h_bytes),
                stream,
            )?;
            gpu.copy_d2d_async(
                main_pool.conv_state(i, ssm_slot),
                self.decode_conv_snapshots[i].offset(flat * self.conv_bytes),
                self.conv_bytes,
                stream,
            )?;
        }
        Ok(())
    }

    /// 2026-09-25: Queue a copy of ring slot `(ssm_slot, ring_slot)` back into pool
    /// slot `ssm_slot`, with the same widths and errors as `save_decode`.
    pub(super) fn restore_decode(
        &self,
        ssm_slot: usize,
        ring_slot: usize,
        main_pool: &SsmStatePool,
        gpu: &dyn GpuBackend,
        stream: u64,
    ) -> Result<()> {
        let flat = self.decode_flat_index(ssm_slot, ring_slot)?;
        for i in 0..self.num_ssm_layers {
            gpu.copy_d2d_async(
                self.decode_h_snapshots[i].offset(flat * self.h_bytes),
                main_pool.h_state(i, ssm_slot),
                main_pool.h_stored_bytes.min(self.h_bytes),
                stream,
            )?;
            gpu.copy_d2d_async(
                self.decode_conv_snapshots[i].offset(flat * self.conv_bytes),
                main_pool.conv_state(i, ssm_slot),
                self.conv_bytes,
                stream,
            )?;
        }
        Ok(())
    }

    fn decode_flat_index(&self, ssm_slot: usize, ring_slot: usize) -> Result<usize> {
        if !self.decode_rollback_enabled() {
            bail!("SSM decode-rollback region not allocated");
        }
        if ssm_slot >= self.decode_max_seqs {
            bail!(
                "SSM decode-rollback: ssm_slot {ssm_slot} >= reserved {} seqs",
                self.decode_max_seqs
            );
        }
        if ring_slot >= self.decode_ring_slots {
            bail!(
                "SSM decode-rollback: ring_slot {ring_slot} >= reserved {} slots",
                self.decode_ring_slots
            );
        }
        Ok(ssm_slot * self.decode_ring_slots + ring_slot)
    }
}
