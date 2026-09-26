// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-25: Batched prefill metadata: builds the [`BatchedAttnMetadata`] that the
//! batched per-layer dispatchers (`prefill_attn_batched_layer`,
//! `prefill_ssm_batched_layer`) read. Called once per batched pass from
//! `batch_kernel.rs`, after the per-stream setup has uploaded each stream's
//! block table and `seq_len`.
//!
//! Layout in `scratch()` from `scratch_offset_bytes`, each field rounded up to 8 bytes:
//!   - positions `[total_tokens]` u32, then H and W `[total_tokens]` u32 under MRoPE
//!   - slots `[total_tokens]` i64
//!   - block-table pointers `[batch_size]` u64
//!   - `seq_len` pointers `[batch_size]` u64
//!   - `cu_seqlens` `[batch_size + 1]` i32
//!   - `kv_lens` `[batch_size]` i32
//!
//! The per-layer `h_state_ptrs` table is staged separately (`h_state_ptrs.rs`).
//!
//! Owner: model-engine prefill (batched).
//! Invariants: none beyond the types.

#![allow(unused_imports, dead_code, clippy::too_many_arguments)]

use anyhow::Result;
use metrale_cache::kv_cache::PagedKvCache;
use metrale_gpu_runtime::gpu::DevicePtr;

use super::super::super::types::TransformerModel;
use crate::traits::{PrefillSlice, SequenceState};
use metrale_model_layers::layer::BatchedAttnMetadata;

// 2026-09-25: `block_ptrs_bytes` is computed from `size_of::<DevicePtr>()` while the
// host staging vectors (`bt_ptrs`, `sl_ptrs`) hold `u64`s; that byte count bounds
// them only if the two widths agree.
const _: () = assert!(std::mem::size_of::<DevicePtr>() == std::mem::size_of::<u64>());

/// 2026-09-25: One stream's entry in the batched arrays, built in `batch_kernel.rs`
/// from its per-stream setup.
pub(in crate::model) struct PerStreamStageInfo<'a> {
    /// 2026-09-25: Absolute position of this stream's first processed token.
    pub proc_start: usize,
    /// 2026-09-25: Tokens this stream contributes; streams may differ (`cu_seqlens`).
    pub proc_count: usize,
    /// 2026-09-25: This stream's uploaded block-table device pointer.
    pub block_table_dev: DevicePtr,
    /// 2026-09-25: This stream's uploaded `seq_len` device pointer.
    pub seq_len_dev: DevicePtr,
    /// 2026-09-25: Block-table entries this stream uses; the maximum over streams is
    /// `max_blocks_per_seq`.
    pub num_blocks: usize,
    /// 2026-09-25: The stream's sequence, read for the slot table.
    pub seq: &'a SequenceState,
}

impl TransformerModel {
    /// 2026-09-25: Build and upload the `BatchedAttnMetadata` for `streams_info`, laid out
    /// as the module docs describe from `scratch_offset_bytes` in `scratch()`. Errors
    /// when `streams_info` is empty, when the layout does not fit in `scratch()`, or
    /// when the pack or the upload fails.
    pub(in crate::model) fn stage_batched_attn_metadata(
        &self,
        streams_info: &[PerStreamStageInfo<'_>],
        kv_cache: &PagedKvCache,
        use_mrope: bool,
        scratch_offset_bytes: usize,
        stream: u64,
    ) -> Result<BatchedAttnMetadata> {
        let n = streams_info.len();
        if n == 0 {
            anyhow::bail!("stage_batched_attn_metadata called with empty streams_info");
        }
        // 2026-09-25: Per-stream token counts become the `cu_seqlens` prefix sum;
        // `chunk_len` is the largest count.
        let mut cu_seqlens_host: Vec<i32> = Vec::with_capacity(n + 1);
        cu_seqlens_host.push(0);
        let mut acc = 0i32;
        let mut chunk_len = 0usize;
        // 2026-09-25: Per-stream K/V extent (`proc_start + proc_count`), so a shorter
        // stream is not given the longest stream's causal bound and block count.
        let mut kv_lens_host: Vec<i32> = Vec::with_capacity(n);
        for info in streams_info.iter() {
            acc += info.proc_count as i32;
            cu_seqlens_host.push(acc);
            chunk_len = chunk_len.max(info.proc_count);
            kv_lens_host.push((info.proc_start + info.proc_count) as i32);
        }
        let total_tokens = acc as usize;

        // 2026-09-25: Byte offsets within `scratch()`, starting at `scratch_offset_bytes`.
        let pos_bytes = total_tokens * 4;
        let pos_aligned = (pos_bytes + 7) & !7;
        let positions_off = scratch_offset_bytes;
        let positions_h_off = if use_mrope {
            positions_off + pos_aligned
        } else {
            positions_off
        };
        let positions_w_off = if use_mrope {
            positions_h_off + pos_aligned
        } else {
            positions_off
        };
        let after_positions = if use_mrope {
            positions_w_off + pos_aligned
        } else {
            positions_off + pos_aligned
        };
        let slot_off = after_positions;
        let slot_bytes = total_tokens * 8;
        let slot_aligned = (slot_bytes + 7) & !7;
        let block_ptrs_off = slot_off + slot_aligned;
        let block_ptrs_bytes = n * std::mem::size_of::<DevicePtr>();
        let block_ptrs_aligned = (block_ptrs_bytes + 7) & !7;
        let seq_len_ptrs_off = block_ptrs_off + block_ptrs_aligned;
        let seq_len_ptrs_aligned = (block_ptrs_bytes + 7) & !7;
        let cu_seqlens_off = seq_len_ptrs_off + seq_len_ptrs_aligned;
        let cu_seqlens_bytes = (n + 1) * 4;
        let cu_seqlens_aligned = (cu_seqlens_bytes + 7) & !7;
        let kv_lens_off = cu_seqlens_off + cu_seqlens_aligned;
        let kv_lens_bytes = n * 4;
        let kv_lens_aligned = (kv_lens_bytes + 7) & !7;
        let total_meta_bytes = kv_lens_off + kv_lens_aligned - scratch_offset_bytes;

        // 2026-09-25: `check_kernel_batched_eligible` sizes the batched footprint
        // (`q12_batched_scratch_bytes` or its varlen form) against `scratch()` at
        // admission. This check covers the layout actually built here and errors
        // instead of writing past `scratch()`.
        let scratch_cap = self.buffers.scratch_bytes();
        if scratch_offset_bytes + total_meta_bytes > scratch_cap {
            tracing::error!(
                "stage_batched_attn_metadata SSOT violation: n={n} chunk_len={chunk_len} \
                 meta_bytes={total_meta_bytes} scratch_offset={scratch_offset_bytes} \
                 scratch_cap={scratch_cap} — eligibility pre-flight should have prevented this"
            );
            anyhow::bail!(
                "batched attn metadata footprint {} B at offset {} exceeds scratch \
                 capacity {} B (n={n}, chunk_len={chunk_len}) — fall back to per-stream",
                total_meta_bytes,
                scratch_offset_bytes,
                scratch_cap,
            );
        }

        // 2026-09-25: SAFETY: only the scheduler thread touches the model (see the
        // `unsafe impl Sync for TransformerModel` note in types.rs).
        let stg = unsafe { &mut *self.pinned_staging.get() };
        stg.positions.clear();
        let mut max_blocks: u32 = 0;
        for info in streams_info.iter() {
            for t in 0..info.proc_count {
                stg.positions.push((info.proc_start + t) as u32);
            }
            max_blocks = max_blocks.max(info.num_blocks as u32);
        }
        // 2026-09-25: Without MRoPE the H and W pointers alias `positions_stacked`, so
        // only MRoPE stages them. The batched path stages text positions only: H and W
        // copy T, and `mrope_pos::build` is not applied.
        if use_mrope {
            stg.positions_h.clear();
            stg.positions_w.clear();
            stg.positions_h.extend_from_slice(&stg.positions);
            stg.positions_w.extend_from_slice(&stg.positions);
        }

        // 2026-09-25: Slot = block index * block_size + offset in block; a position with
        // no physical block gets `dummy_kv_block`.
        let bs = kv_cache.block_size();
        stg.slots.clear();
        for info in streams_info.iter() {
            for t in 0..info.proc_count {
                let pos = info.proc_start + t;
                let block_idx = info
                    .seq
                    .physical_block_for(pos / bs)
                    .unwrap_or(self.dummy_kv_block);
                let slot = (block_idx as i64) * (bs as i64) + ((pos % bs) as i64);
                stg.slots.push(slot);
            }
        }

        let mut bt_ptrs: Vec<u64> = Vec::with_capacity(n);
        let mut sl_ptrs: Vec<u64> = Vec::with_capacity(n);
        for info in streams_info.iter() {
            bt_ptrs.push(info.block_table_dev.0);
            sl_ptrs.push(info.seq_len_dev.0);
        }

        // 2026-09-25: One H2D copy of the whole layout.
        let scratch_base = self.buffers.scratch().offset(scratch_offset_bytes);

        // 2026-09-25: `put_prefix_at` refuses a source shorter than the layout expects,
        // and the packer checks every field against the destination before writing.
        // The `*_aligned` round-ups leave up to 4 bytes per field that no copy writes;
        // `pad_to` includes them in the upload, and they are initialised because the
        // staging allocation is zeroed (`pinned_pack` module docs).
        let mut pack = stg.packer_for(scratch_cap.saturating_sub(scratch_offset_bytes));
        let mut cursor = 0usize;
        pack.put_prefix_at("positions", cursor, &stg.positions, total_tokens)?;
        cursor = pos_aligned;
        if use_mrope {
            pack.put_prefix_at("positions_h", cursor, &stg.positions_h, total_tokens)?;
            cursor += pos_aligned;
            pack.put_prefix_at("positions_w", cursor, &stg.positions_w, total_tokens)?;
            cursor += pos_aligned;
        }
        pack.put_prefix_at("slots", cursor, &stg.slots, total_tokens)?;
        cursor += slot_aligned;
        pack.put_prefix_at("block_table_ptrs", cursor, &bt_ptrs, n)?;
        cursor += block_ptrs_aligned;
        pack.put_prefix_at("seq_len_ptrs", cursor, &sl_ptrs, n)?;
        cursor += seq_len_ptrs_aligned;
        pack.put_prefix_at("cu_seqlens", cursor, &cu_seqlens_host, n + 1)?;
        cursor += cu_seqlens_aligned;
        pack.put_prefix_at("kv_lens", cursor, &kv_lens_host, n)?;
        cursor += kv_lens_aligned;
        pack.pad_to(cursor)?;
        self.gpu
            .copy_h2d_async_retained(pack.packed(), scratch_base, stream)?;

        Ok(BatchedAttnMetadata {
            positions_stacked: scratch_base,
            positions_h_stacked: scratch_base.offset(positions_h_off - positions_off),
            positions_w_stacked: scratch_base.offset(positions_w_off - positions_off),
            slot_stacked: scratch_base.offset(slot_off - positions_off),
            block_table_ptrs: scratch_base.offset(block_ptrs_off - positions_off),
            seq_len_ptrs: scratch_base.offset(seq_len_ptrs_off - positions_off),
            batch_size: n as u32,
            chunk_len: chunk_len as u32,
            total_tokens: total_tokens as u32,
            cu_seqlens: scratch_base.offset(cu_seqlens_off - positions_off),
            cu_seqlens_host,
            kv_lens: scratch_base.offset(kv_lens_off - positions_off),
            kv_lens_host,
            max_blocks_per_seq: max_blocks,
            staged_bytes: total_meta_bytes,
        })
    }
}
