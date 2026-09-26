// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-25: Chunked-prefill paged metadata: upload the block-table delta and the seq_len,
//! then fill the chunk's KV slots from the block table on the device.
//!
//! Owner: model-engine.
//! Invariants: `uploaded_blocks` advances only after its block-table copy was enqueued.

#![allow(unused_imports, dead_code, clippy::too_many_arguments)]

use anyhow::Result;
use metrale_cache::kv_cache::PagedKvCache;
use metrale_gpu_runtime::gpu::DevicePtr;

use super::super::super::types::TransformerModel;
use crate::traits::SequenceState;
use metrale_model_layers::layers::ops;

impl TransformerModel {
    pub(in crate::model) fn prefill_b_upload_paged(
        &self,
        seq: &mut SequenceState,
        total: usize,
        proc_start: usize,
        proc_count: usize,
        meta_base: DevicePtr,
        slot_offset: usize,
        kv_cache: &PagedKvCache,
        stream: u64,
    ) -> Result<()> {
        let bs = kv_cache.block_size();
        let current_blocks = seq.block_table.len();
        let upload_start = self
            .ensure_chunked_prefill_meta(seq, total, bs)?
            .uploaded_blocks;
        // 2026-09-25: Once the HSS window has slid, `block_table[i]` is absolute
        // block `hss_window_start() + i` (`physical_block_for`), so the delta
        // cannot be written at its absolute offset; the block-table upload is
        // skipped. The seq_len upload and slot fill below still run.
        if upload_start < current_blocks && seq.hss_window_start() == 0 {
            let new_blocks = &seq.block_table[upload_start..];
            // 2026-09-25: SAFETY: the length is `size_of_val(new_blocks)`, derived from the
            // slice itself, so it can never exceed it, over a live `&[u32]`
            // sub-slice of `seq.block_table`.
            let bt_bytes = unsafe {
                std::slice::from_raw_parts(
                    new_blocks.as_ptr() as *const u8,
                    std::mem::size_of_val(new_blocks),
                )
            };
            let block_table_base = seq.chunked_prefill_meta.as_ref().unwrap().block_table;
            self.gpu.copy_h2d_async(
                bt_bytes,
                block_table_base.offset(upload_start * std::mem::size_of::<u32>()),
                stream,
            )?;
            seq.chunked_prefill_meta.as_mut().unwrap().uploaded_blocks = current_blocks;
        }

        let seq_len_val = (proc_start + proc_count) as u32;
        // 2026-09-25: SAFETY: exactly `size_of::<u32>()` bytes over the live, fully
        // initialised `seq_len_val` local on the line above.
        let seq_len_bytes = unsafe {
            std::slice::from_raw_parts(
                &seq_len_val as *const u32 as *const u8,
                std::mem::size_of::<u32>(),
            )
        };
        let seq_len_base = seq.chunked_prefill_meta.as_ref().unwrap().seq_len;
        self.gpu
            .copy_h2d_async(seq_len_bytes, seq_len_base, stream)?;

        let block_table_base = seq.chunked_prefill_meta.as_ref().unwrap().block_table;
        ops::fill_slots_from_block_table(
            self.gpu.as_ref(),
            self.fill_slots_kernel,
            meta_base.offset(slot_offset),
            block_table_base,
            proc_start as u32,
            proc_count as u32,
            bs as u32,
            stream,
        )?;

        Ok(())
    }
}
