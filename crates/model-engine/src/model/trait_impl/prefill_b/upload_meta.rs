// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-25: Per-pass prefill metadata: positions (three streams under MRoPE) and,
//! when not paged, the K/V slot table, packed in pinned staging and uploaded into a
//! region of `scratch()`. Returns the `MetaLayout` that `upload_paged` and
//! `forward_layers` read.
//!
//! Owner: model-engine prefill.
//! Invariants: none beyond the types.

#![allow(unused_imports, dead_code, clippy::too_many_arguments)]

use anyhow::Result;
use metrale_cache::kv_cache::PagedKvCache;
use metrale_gpu_runtime::gpu::DevicePtr;

use super::super::super::types::TransformerModel;
use crate::traits::SequenceState;

pub(in crate::model) struct MetaLayout {
    pub meta_base: DevicePtr,
    pub slot_offset: usize,
    pub pos_stream_bytes: usize,
    pub use_mrope: bool,
    pub needs_paged: bool,
}

impl TransformerModel {
    pub(super) fn prefill_b_upload_meta(
        &self,
        tokens: &[u32],
        seq: &mut SequenceState,
        chunk_start: usize,
        chunk_len: usize,
        proc_start: usize,
        proc_count: usize,
        effective_seq_len_start: usize,
        kv_cache: &PagedKvCache,
        stream: u64,
    ) -> Result<MetaLayout> {
        // 2026-09-25: The metadata goes after the MoE top-k staging area at the start of
        // `scratch()` (`proc_count * num_experts_per_tok * 8` bytes, rounded up to 8).
        let moe_scratch_bytes = proc_count * self.config.num_experts_per_tok * 4 * 2;
        let meta_offset = (moe_scratch_bytes + 7) & !7;
        let meta_base = self.buffers.scratch().offset(meta_offset);
        self.prefill_b_upload_meta_at(
            tokens,
            seq,
            chunk_start,
            chunk_len,
            proc_start,
            proc_count,
            effective_seq_len_start,
            kv_cache,
            meta_base,
            self.buffers.scratch_bytes().saturating_sub(meta_offset),
            stream,
        )
    }

    /// 2026-09-25: Build positions and slots for `proc_count` tokens and upload them to
    /// `meta_base`.
    ///
    /// `meta_region_bytes` is the room the caller gives this block at `meta_base`: both
    /// callers pass the rest of `scratch()` from `meta_base` (`batch_kernel.rs` then
    /// advances its cursor by `per_stream_meta_bytes`). `meta_base` carries no size, so
    /// the packer bounds every write by this value as well as by the host staging buffer
    /// (`packer_for`).
    pub(in crate::model) fn prefill_b_upload_meta_at(
        &self,
        tokens: &[u32],
        seq: &mut SequenceState,
        chunk_start: usize,
        chunk_len: usize,
        proc_start: usize,
        proc_count: usize,
        effective_seq_len_start: usize,
        kv_cache: &PagedKvCache,
        meta_base: DevicePtr,
        meta_region_bytes: usize,
        stream: u64,
    ) -> Result<MetaLayout> {
        // 2026-09-25: MRoPE-interleaved packs three u32 position streams (T, H, W).
        let use_mrope = self.config.mrope_interleaved;
        let pos_stream_bytes = proc_count * 4;
        let slot_offset = if use_mrope {
            (pos_stream_bytes * 3 + 7) & !7
        } else {
            (pos_stream_bytes + 7) & !7
        };
        let needs_paged = effective_seq_len_start > 0;

        // 2026-09-25: Build positions and, when not paged, slots in the pinned staging
        // buffer, then upload.
        {
            // 2026-09-25: SAFETY: only the scheduler thread touches the model (see the
            // `unsafe impl Sync for TransformerModel` note in types.rs).
            let stg = unsafe { &mut *self.pinned_staging.get() };
            stg.positions.clear();
            stg.positions
                .extend(proc_start as u32..(proc_start + proc_count) as u32);

            // 2026-09-25: MRoPE: when the chunk holds vision pad tokens, `mrope_pos::build`
            // rebuilds all three streams (the rule is documented there); otherwise H and
            // W copy the linear T stream.
            if use_mrope {
                stg.positions_h.clear();
                stg.positions_w.clear();
                let grids = self.vision_image_grids.lock().clone();
                let (pad_id, video_pad_id) = self.vision_pad_ids();
                let is_pad = |tok: u32| tok == pad_id || tok == video_pad_id;
                let chunk_tokens = &tokens[chunk_start..chunk_start + chunk_len];
                let have_vision = !grids.is_empty() && chunk_tokens.iter().copied().any(is_pad);

                if have_vision {
                    stg.positions.clear();
                    let current_pos: u32 = proc_start as u32;
                    // 2026-09-25: This request owns `grids[grid_base..grid_base + owned]` of the
                    // shared `vision_image_grids`; `owned == 0` means all of them.
                    let grid_base = *self.vision_grid_base.lock();
                    let owned = *self.vision_owned_images.lock();
                    let grid_hi = if owned > 0 {
                        (grid_base + owned).min(grids.len())
                    } else {
                        grids.len()
                    };
                    mrope_pos::build(
                        chunk_tokens,
                        &grids,
                        grid_base,
                        grid_hi,
                        current_pos,
                        pad_id,
                        video_pad_id,
                        &mut stg.positions,
                        &mut stg.positions_h,
                        &mut stg.positions_w,
                    );
                } else {
                    stg.positions_h.extend_from_slice(&stg.positions);
                    stg.positions_w.extend_from_slice(&stg.positions);
                }
            }

            // 2026-09-25: Build the slot table before packing: the packer borrows the
            // staging struct, so every reusable `Vec` has to be final first.
            if !needs_paged {
                let bs = kv_cache.block_size();
                stg.slots.clear();
                stg.slots
                    .extend((proc_start..proc_start + proc_count).map(|i| {
                        let block_idx = seq
                            .physical_block_for(i / bs)
                            .unwrap_or(self.dummy_kv_block);
                        (block_idx as i64) * (bs as i64) + ((i % bs) as i64)
                    }));
            }

            // 2026-09-25: The MRoPE vision arm rebuilds `positions` from the chunk's tokens,
            // not from `proc_count`, so its length depends on the data. `put_prefix_at`
            // takes the first `proc_count` elements of each stream and refuses a shorter
            // one; the packer checks every field against the destination before writing.
            //
            // Rounding `slot_offset` up to 8 leaves up to 4 bytes after the position
            // streams that no copy writes; they are initialised because the staging
            // allocation is zeroed (`pinned_pack` module docs).
            let mut pack = stg.packer_for(meta_region_bytes);
            pack.put_prefix_at("positions", 0, &stg.positions, proc_count)?;
            if use_mrope {
                let h_at = pos_stream_bytes;
                let w_at = h_at + pos_stream_bytes;
                pack.put_prefix_at("positions_h", h_at, &stg.positions_h, proc_count)?;
                pack.put_prefix_at("positions_w", w_at, &stg.positions_w, proc_count)?;
            }
            if !needs_paged {
                pack.put_prefix_at("slots", slot_offset, &stg.slots, proc_count)?;
            }
            self.gpu
                .copy_h2d_async_retained(pack.packed(), meta_base, stream)?;
        }

        Ok(MetaLayout {
            meta_base,
            slot_offset,
            pos_stream_bytes,
            use_mrope,
            needs_paged,
        })
    }
}

#[path = "mrope_pos.rs"]
pub(crate) mod mrope_pos;
