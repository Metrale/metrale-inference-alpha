// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-25: Batched-decode metadata upload (positions, KV slots, seq_lens,
//! block table and the LoRA routing buffers) and the BF16/FP32 readback
//! helpers.
//!
//! Owner: model-engine.
//! Invariants:
//! - `upload_batch_metadata_fixed` and `upload_batch_metadata_at` refuse a
//!   `padded_n` above `decode_meta().rows()`, and every block-table entry they
//!   do not set holds `dummy_kv_block`.

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
    /// 2026-09-25: Upload batch metadata at `scratch + 32768` with fixed
    /// strides, so captured CUDA graphs stay valid: `max_blocks_per_seq` is the
    /// block-table stride, and rows `seqs.len()..padded_n` are padding that
    /// points at `dummy_kv_block`.
    pub(super) fn upload_batch_metadata_fixed(
        &self,
        seqs: &[&mut SequenceState],
        padded_n: usize,
        kv_cache: &mut PagedKvCache,
        stream: u64,
    ) -> Result<AttnMetadataDev> {
        // 2026-09-25: Offsets come from `buffers.decode_meta()`, with
        // `R = max(32, max_batch_size)` rows: positions u32 at 0, LoRA seq_slot
        // i32 at 4R, KV slots i64 at 8R, seq_lens i32 at 16R, the block table
        // at 24R. More than R rows would overrun the next region, hence the
        // check.
        let lay = self.buffers.decode_meta();
        anyhow::ensure!(
            padded_n <= lay.rows(),
            "upload_batch_metadata_fixed: padded_n={padded_n} exceeds the {}-row \
             derived metadata layout (rows = max(32, --max-batch-size))",
            lay.rows()
        );
        let n = seqs.len();
        let block_size = kv_cache.block_size();
        let max_blocks = self.max_blocks_per_seq;

        let mut positions = Vec::with_capacity(padded_n);
        let mut slots = Vec::with_capacity(padded_n);
        let mut seq_lens_host = Vec::with_capacity(padded_n);
        // 2026-09-25: Unset block-table entries point at the zeroed
        // `dummy_kv_block`, so a read past a sequence's blocks lands on zeros.
        let mut block_table_flat: Vec<i32> =
            vec![self.dummy_kv_block as i32; padded_n * max_blocks as usize];

        for (i, seq) in seqs.iter().enumerate() {
            let pos = seq.seq_len as u32;
            positions.push(pos);

            let block_idx = pos as usize / block_size;
            let block_offset = pos as usize % block_size;
            let physical_block = seq
                .physical_block_for(block_idx)
                .unwrap_or(self.dummy_kv_block);
            let slot = (physical_block as i64) * (block_size as i64) + (block_offset as i64);
            slots.push(slot);

            seq_lens_host.push((seq.seq_len + 1) as i32);

            // 2026-09-25: A real sequence's block table must cover its
            // `seq_len + 1` tokens, or attention reads the dummy block in place
            // of its own KV. Checked only in debug builds.
            debug_assert!(
                seq.block_table.len() > (seq.seq_len / block_size),
                "seq slot={} seq_len={} block_table.len={} (need >= {})",
                seq.slot_idx,
                seq.seq_len,
                seq.block_table.len(),
                (seq.seq_len / block_size) + 1,
            );

            for (j, &block) in seq.block_table.iter().take(max_blocks as usize).enumerate() {
                block_table_flat[i * max_blocks as usize + j] = block as i32;
            }
        }

        // 2026-09-25: Padding rows write to the dummy block, with seq_len 1 at
        // position 0.
        let dummy_slot = (self.dummy_kv_block as i64) * (block_size as i64);
        for i in n..padded_n {
            positions.push(0);
            slots.push(dummy_slot);
            seq_lens_host.push(1);
            block_table_flat[i * max_blocks as usize] = self.dummy_kv_block as i32;
        }

        let meta_base = self.buffers.scratch().offset(32768);
        let pos_bytes: Vec<u8> = positions.iter().flat_map(|p| p.to_le_bytes()).collect();
        let slot_bytes: Vec<u8> = slots.iter().flat_map(|s| s.to_le_bytes()).collect();
        let sl_bytes: Vec<u8> = seq_lens_host.iter().flat_map(|s| s.to_le_bytes()).collect();
        let bt_bytes: Vec<u8> = block_table_flat
            .iter()
            .flat_map(|b| b.to_le_bytes())
            .collect();

        self.gpu.copy_h2d_async(&pos_bytes, meta_base, stream)?;
        self.gpu
            .copy_h2d_async(&slot_bytes, meta_base.offset(lay.slots_off()), stream)?;
        self.gpu
            .copy_h2d_async(&sl_bytes, meta_base.offset(lay.seq_lens_off()), stream)?;
        self.gpu
            .copy_h2d_async(&bt_bytes, meta_base.offset(lay.block_table_off()), stream)?;

        // 2026-09-25: Per-request LoRA routing slots, in the seq_slot region
        // `[4R, 8R)` between positions and KV slots: a fixed address with
        // per-step contents, so captured graphs stay valid. `DevicePtr(0)` when
        // no LoRA weights are loaded.
        let seq_slot =
            self.upload_seq_slots(seqs, padded_n, meta_base.offset(lay.seq_slot_off()), stream)?;
        // 2026-09-25: The batched-decode MoE per-row adapter map, in its own
        // fixed-address buffer (`moe_row_adapter_buf`). `DevicePtr(0)` when no
        // LoRA weights are loaded.
        let moe_row_adapter =
            self.upload_moe_row_adapter(seqs, padded_n, self.moe_row_adapter_buf, stream)?;

        Ok(AttnMetadataDev {
            positions: meta_base,
            positions_h: meta_base,
            positions_w: meta_base,
            slot: meta_base.offset(lay.slots_off()),
            seq_len: meta_base.offset(lay.seq_lens_off()),
            block_table: meta_base.offset(lay.block_table_off()),
            max_blocks_per_seq: max_blocks,
            num_seqs: padded_n as u32,
            seq_slot,
            moe_row_adapter,
        })
    }

    /// 2026-09-25: Build and upload the `[padded_n]` i32 adapter-slot buffer for
    /// per-request LoRA routing at `dst`; each row is resolved by
    /// `metrale_model_layers::lora::build_seq_slot_host`. Returns `dst`, or
    /// `DevicePtr(0)` without uploading when no LoRA weights are loaded.
    fn upload_seq_slots(
        &self,
        seqs: &[&mut SequenceState],
        padded_n: usize,
        dst: DevicePtr,
        stream: u64,
    ) -> Result<DevicePtr> {
        let active = match self.lora.as_ref() {
            Some(lw) => lw.active as i32,
            None => return Ok(DevicePtr(0)),
        };
        let adapter_slots: Vec<i32> = seqs.iter().map(|s| s.adapter_slot).collect();
        let host =
            metrale_model_layers::lora::build_seq_slot_host(&adapter_slots, padded_n, active);
        let bytes: Vec<u8> = host.iter().flat_map(|v| v.to_le_bytes()).collect();
        self.gpu.copy_h2d_async(&bytes, dst, stream)?;
        Ok(dst)
    }

    /// 2026-09-25: Build and upload the `[padded_n]` i32 per-row MoE adapter map
    /// to `dst` (`< 0` skips the row, `>= 0` folds the active adapter); rows are
    /// resolved by `metrale_model_layers::lora::build_moe_row_adapter_decode`.
    /// Returns `dst`, or `DevicePtr(0)` without uploading when no LoRA weights
    /// are loaded; with LoRA weights loaded, errors when `padded_n > 32`.
    fn upload_moe_row_adapter(
        &self,
        seqs: &[&mut SequenceState],
        padded_n: usize,
        dst: DevicePtr,
        stream: u64,
    ) -> Result<DevicePtr> {
        let active = match self.lora.as_ref() {
            Some(lw) => lw.active as i32,
            None => return Ok(DevicePtr(0)),
        };
        anyhow::ensure!(
            padded_n <= 32,
            "concurrent LoRA decode is limited to batch<=32 (shared decode \
             metadata layout: positions/seq_slot/slot each hold 32 rows); got \
             padded_n={padded_n}. Use --max-num-seqs <=32 with a resident MoE \
             adapter."
        );
        let adapter_slots: Vec<i32> = seqs.iter().map(|s| s.adapter_slot).collect();
        let host = metrale_model_layers::lora::build_moe_row_adapter_decode(
            &adapter_slots,
            padded_n,
            active,
            true,
        );
        let bytes: Vec<u8> = host.iter().flat_map(|v| v.to_le_bytes()).collect();
        self.gpu.copy_h2d_async(&bytes, dst, stream)?;
        Ok(dst)
    }

    /// 2026-09-25: Upload a `[count]` i32 adapter-slot buffer whose rows all
    /// carry one request's `adapter_slot` (`-1` resolves to the active
    /// adapter), for the single-request decode, prefill and verify paths.
    /// Returns `DevicePtr(0)` without uploading when no LoRA weights are loaded
    /// or the request resolves to the active adapter; otherwise uploads and
    /// returns `dst`.
    pub(crate) fn upload_seq_slot_uniform(
        &self,
        adapter_slot: i32,
        count: usize,
        dst: DevicePtr,
        stream: u64,
    ) -> Result<DevicePtr> {
        let active = match self.lora.as_ref() {
            Some(lw) => lw.active as i32,
            None => return Ok(DevicePtr(0)),
        };
        // 2026-09-25: A request that resolves to the active adapter keeps the
        // installed-pair path (`apply_lora_delta`) instead of the per-row bgmv,
        // whose numerics differ; only a request routed to another adapter
        // uploads a slot buffer.
        let resolved = if adapter_slot >= 0 {
            adapter_slot
        } else {
            active
        };
        if resolved == active {
            return Ok(DevicePtr(0));
        }
        let slots = vec![adapter_slot; count];
        let host = metrale_model_layers::lora::build_seq_slot_host(&slots, count, active);
        let bytes: Vec<u8> = host.iter().flat_map(|v| v.to_le_bytes()).collect();
        self.gpu.copy_h2d_async(&bytes, dst, stream)?;
        Ok(dst)
    }

    /// 2026-09-25: The request slot's LoRA pairs, indexed by global layer, for
    /// a prefill's `ForwardContext.routed_lora_layers`. `Some` only when
    /// `adapter_slot` routes to a non-active, in-range slot
    /// (`LoraWeights::routed_prefill_slot`); `None` for requests on the active
    /// adapter and when no LoRA weights are loaded.
    pub(crate) fn routed_slot_layers(
        &self,
        adapter_slot: i32,
    ) -> Option<&[Option<metrale_model_layers::lora::LoraLayerWeights>]> {
        let lw = self.lora.as_ref()?;
        let resolved = lw.routed_prefill_slot(adapter_slot)?;
        Some(lw.slots[resolved].layers.as_slice())
    }

    /// 2026-09-25: `upload_batch_metadata_fixed` at a caller-chosen `meta_base`
    /// instead of `scratch + 32768`, with the same layout and row check. The
    /// mixed forward uses it to place its decode metadata elsewhere.
    pub(super) fn upload_batch_metadata_at(
        &self,
        seqs: &[&mut SequenceState],
        padded_n: usize,
        kv_cache: &mut PagedKvCache,
        meta_base: DevicePtr,
        stream: u64,
    ) -> Result<AttnMetadataDev> {
        let n = seqs.len();
        let block_size = kv_cache.block_size();
        let max_blocks = self.max_blocks_per_seq;

        let lay = self.buffers.decode_meta();
        anyhow::ensure!(
            padded_n <= lay.rows(),
            "upload_batch_metadata_at: padded_n={padded_n} exceeds the {}-row \
             derived metadata layout (rows = max(32, --max-batch-size))",
            lay.rows()
        );

        let mut positions = Vec::with_capacity(padded_n);
        let mut slots = Vec::with_capacity(padded_n);
        let mut seq_lens_host = Vec::with_capacity(padded_n);
        let mut block_table_flat: Vec<i32> =
            vec![self.dummy_kv_block as i32; padded_n * max_blocks as usize];

        for seq in seqs.iter() {
            let pos = seq.seq_len as u32;
            positions.push(pos);

            let block_idx = pos as usize / block_size;
            let block_offset = pos as usize % block_size;
            let physical_block = seq
                .physical_block_for(block_idx)
                .unwrap_or(self.dummy_kv_block);
            let slot = (physical_block as i64) * (block_size as i64) + (block_offset as i64);
            slots.push(slot);

            seq_lens_host.push((seq.seq_len + 1) as i32);
        }

        for (i, seq) in seqs.iter().enumerate() {
            for (j, &block) in seq.block_table.iter().take(max_blocks as usize).enumerate() {
                block_table_flat[i * max_blocks as usize + j] = block as i32;
            }
        }

        let dummy_slot = (self.dummy_kv_block as i64) * (block_size as i64);
        for i in n..padded_n {
            positions.push(0);
            slots.push(dummy_slot);
            seq_lens_host.push(1);
            block_table_flat[i * max_blocks as usize] = self.dummy_kv_block as i32;
        }

        let pos_bytes: Vec<u8> = positions.iter().flat_map(|p| p.to_le_bytes()).collect();
        let slot_bytes: Vec<u8> = slots.iter().flat_map(|s| s.to_le_bytes()).collect();
        let sl_bytes: Vec<u8> = seq_lens_host.iter().flat_map(|s| s.to_le_bytes()).collect();
        let bt_bytes: Vec<u8> = block_table_flat
            .iter()
            .flat_map(|b| b.to_le_bytes())
            .collect();

        self.gpu.copy_h2d_async(&pos_bytes, meta_base, stream)?;
        self.gpu
            .copy_h2d_async(&slot_bytes, meta_base.offset(lay.slots_off()), stream)?;
        self.gpu
            .copy_h2d_async(&sl_bytes, meta_base.offset(lay.seq_lens_off()), stream)?;
        self.gpu
            .copy_h2d_async(&bt_bytes, meta_base.offset(lay.block_table_off()), stream)?;

        let seq_slot =
            self.upload_seq_slots(seqs, padded_n, meta_base.offset(lay.seq_slot_off()), stream)?;
        let moe_row_adapter =
            self.upload_moe_row_adapter(seqs, padded_n, self.moe_row_adapter_buf, stream)?;

        Ok(AttnMetadataDev {
            positions: meta_base,
            positions_h: meta_base,
            positions_w: meta_base,
            slot: meta_base.offset(lay.slots_off()),
            seq_len: meta_base.offset(lay.seq_lens_off()),
            block_table: meta_base.offset(lay.block_table_off()),
            max_blocks_per_seq: max_blocks,
            num_seqs: padded_n as u32,
            seq_slot,
            moe_row_adapter,
        })
    }

    /// 2026-09-25: Read the first `n` BF16 values at `ptr` back as f32, with
    /// their L2 norm.
    pub(super) fn readback_bf16(&self, ptr: DevicePtr, n: usize) -> Result<(Vec<f32>, f32)> {
        let bytes = n * 2;
        let mut buf = vec![0u8; bytes];
        self.gpu.copy_d2h(ptr, &mut buf)?;
        let vals: Vec<f32> = buf
            .chunks_exact(2)
            .map(|c| {
                let bits = u16::from_le_bytes([c[0], c[1]]);
                f32::from_bits((bits as u32) << 16)
            })
            .collect();
        let norm = vals.iter().map(|v| v * v).sum::<f32>().sqrt();
        Ok((vals, norm))
    }

    /// 2026-09-25: Read the first `n` FP32 values at `ptr`, with their L2 norm.
    pub(super) fn readback_f32(&self, ptr: DevicePtr, n: usize) -> Result<(Vec<f32>, f32)> {
        let bytes = n * 4;
        let mut buf = vec![0u8; bytes];
        self.gpu.copy_d2h(ptr, &mut buf)?;
        let vals: Vec<f32> = buf
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect();
        let norm = vals.iter().map(|v| v * v).sum::<f32>().sqrt();
        Ok((vals, norm))
    }
}
