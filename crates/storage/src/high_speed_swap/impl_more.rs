// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-25: `HighSpeedSwap` offload and attention. `offload_block*` writes a
//! KV block to the tier, one K and one V group per kv_head, and can project its
//! K for the predictor; `attend_layer*` streams a sequence's blocks through the
//! scratch pool into tiled attention, one tile of `resident_blocks` at a time.
//!
//! Owner: metrale-storage high-speed swap.
//! Invariants:
//! - A successful offload removes the block's scratch-pool copy, so a later
//!   attend reads the block from the tier.

use anyhow::Result;
use std::ffi::c_void;

use super::HighSpeedSwap;
use crate::backend::{ReadRequest, StorageBackend};
use crate::config::HighSpeedSwapConfig;
use crate::cuda_min::{CudaCtx, copy_d_to_h_async, copy_h_to_d_async, stream_sync};
use crate::group::{GroupKey, KvKind};
use crate::predictor::Predictor;
use crate::scratch_pool::{ResidentKey, ScratchPool};

impl HighSpeedSwap {
    /// 2026-09-25: Write a KV block to the tier and project its K for the
    /// predictor. `k_block_dev`, `k_block_host` and `v_block_host` are BF16
    /// `[block_size, num_kv_heads, head_dim]`; the device K feeds the projection,
    /// the host copies the per-kv_head groups.
    pub fn offload_block(
        &mut self,
        ctx: &CudaCtx,
        layer: u32,
        block: u32,
        k_block_dev: u64,
        k_block_host: &[half::bf16],
        v_block_host: &[half::bf16],
    ) -> Result<()> {
        self.offload_block_on_stream(
            ctx.stream,
            layer,
            block,
            k_block_dev,
            k_block_host,
            v_block_host,
        )
    }

    /// 2026-09-25: `offload_block` on `stream`, which must belong to the current
    /// thread's CUDA context.
    pub fn offload_block_on_stream(
        &mut self,
        stream: u64,
        layer: u32,
        block: u32,
        k_block_dev: u64,
        k_block_host: &[half::bf16],
        v_block_host: &[half::bf16],
    ) -> Result<()> {
        // 2026-09-25: `true` runs the projection, which reads `k_block_dev` as
        // BF16; callers with another K layout use
        // `offload_block_no_predict_on_stream`.
        self.offload_block_inner_on_stream(
            stream,
            layer,
            block,
            k_block_dev,
            k_block_host,
            v_block_host,
            true,
        )
    }

    /// 2026-09-25: `offload_block_on_stream` without the predictor projection,
    /// for callers whose device K is not BF16. The host buffers are still BF16.
    pub fn offload_block_no_predict_on_stream(
        &mut self,
        stream: u64,
        layer: u32,
        block: u32,
        k_block_host: &[half::bf16],
        v_block_host: &[half::bf16],
    ) -> Result<()> {
        self.offload_block_inner_on_stream(
            stream,
            layer,
            block,
            0,
            k_block_host,
            v_block_host,
            false,
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn offload_block_inner_on_stream(
        &mut self,
        stream: u64,
        layer: u32,
        block: u32,
        k_block_dev: u64,
        k_block_host: &[half::bf16],
        v_block_host: &[half::bf16],
        do_predict: bool,
    ) -> Result<()> {
        if do_predict {
            self.predictor.project_kv_block_on_stream(
                stream,
                layer as usize,
                block as usize,
                k_block_dev,
            )?;
        }
        let bs = self.model.block_size as usize;
        let nkv = self.model.num_kv_heads as usize;
        let hd = self.model.head_dim as usize;
        if k_block_host.len() != bs * nkv * hd || v_block_host.len() != bs * nkv * hd {
            anyhow::bail!(
                "offload_block: host buffers must be {} BF16 elements",
                bs * nkv * hd
            );
        }
        for kh in 0..nkv {
            let mut k_stripe = Vec::with_capacity(bs * hd * 2);
            let mut v_stripe = Vec::with_capacity(bs * hd * 2);
            for tok in 0..bs {
                let base = (tok * nkv + kh) * hd;
                for x in &k_block_host[base..base + hd] {
                    k_stripe.extend_from_slice(&x.to_le_bytes());
                }
                for x in &v_block_host[base..base + hd] {
                    v_stripe.extend_from_slice(&x.to_le_bytes());
                }
            }
            self.backend
                .write_from_host(GroupKey::new(layer, block, kh as u16, KvKind::K), &k_stripe)?;
            self.backend
                .write_from_host(GroupKey::new(layer, block, kh as u16, KvKind::V), &v_stripe)?;
        }
        // 2026-09-25: The tier copy was just rewritten; drop any scratch-pool copy
        // so attention does not read a stale slot.
        self.pool.invalidate(ResidentKey { layer, block });
        Ok(())
    }

    /// 2026-09-25: Streaming attention for one (layer, sequence). `q_dev` is the
    /// step's BF16 query `[num_q_heads × head_dim]`, `seq_block_ids` the
    /// sequence's blocks, and `output_dev` receives the BF16 output
    /// `[num_q_heads × head_dim]`.
    pub fn attend_layer(
        &mut self,
        ctx: &CudaCtx,
        layer: u32,
        seq_block_ids: &[u32],
        q_dev: u64,
        output_dev: u64,
    ) -> Result<()> {
        self.attend_layer_on_stream(ctx.stream, layer, seq_block_ids, q_dev, output_dev)
    }

    /// 2026-09-25: `attend_layer` on `stream`, which must belong to the current
    /// thread's CUDA context, with no causal mask: every slot of the last block
    /// is attended. A query that must not see later tokens of its own block uses
    /// `attend_layer_on_stream_with_q_pos`.
    pub fn attend_layer_on_stream(
        &mut self,
        stream: u64,
        layer: u32,
        seq_block_ids: &[u32],
        q_dev: u64,
        output_dev: u64,
    ) -> Result<()> {
        let bs = self.model.block_size as i32;
        self.attend_layer_on_stream_with_q_pos(stream, layer, seq_block_ids, q_dev, output_dev, bs)
    }

    /// 2026-09-25: Only the first `last_block_valid_slots` token slots of the
    /// last block in `seq_block_ids` are attended; for a query at position
    /// `q_pos`, pass `(q_pos % block_size) + 1`.
    pub fn attend_layer_on_stream_with_q_pos(
        &mut self,
        stream: u64,
        layer: u32,
        seq_block_ids: &[u32],
        q_dev: u64,
        output_dev: u64,
        last_block_valid_slots: i32,
    ) -> Result<()> {
        // 2026-09-25: All `max_blocks_per_layer` blocks of the layer are scored;
        // only the sequence's blocks are used.
        self.predictor
            .project_q_on_stream(stream, q_dev, self.q_proj.ptr)?;
        let m = &self.model;
        let layer_a_g = self.predictor.a_g_dev_ptr()
            + (layer as u64)
                * (m.max_blocks_per_layer as u64)
                * (m.num_kv_heads as u64)
                * (m.block_size as u64)
                * (self.cfg.rank as u64)
                * 2;
        self.predictor.score_blocks_on_stream(
            stream,
            self.q_proj.ptr,
            layer_a_g,
            self.block_scores_dev.ptr,
            m.max_blocks_per_layer as usize,
        )?;
        copy_d_to_h_async(
            self.score_host_buf.as_mut_ptr() as *mut c_void,
            self.block_scores_dev.ptr,
            self.score_host_buf.len() * 4,
            stream,
        )?;
        stream_sync(stream)?;

        self.attn.begin_step_on_stream(stream, 1)?;
        let tile_cap = self.cfg.resident_blocks as usize;
        let mut tile_idx = 0;
        while tile_idx < seq_block_ids.len() {
            let tile_end = (tile_idx + tile_cap).min(seq_block_ids.len());
            let tile = &seq_block_ids[tile_idx..tile_end];

            let mut block_table = vec![0_i32; tile_cap];
            let mut pinned: Vec<u32> = Vec::new();
            let mut missing: Vec<u32> = Vec::new();
            for (i, &blk) in tile.iter().enumerate() {
                let key = ResidentKey { layer, block: blk };
                if let Some(slot) = self.pool.lookup(key) {
                    block_table[i] = slot as i32;
                    pinned.push(slot);
                    self.eviction.touch(slot);
                } else {
                    missing.push(blk);
                }
            }
            let mut reqs: Vec<ReadRequest> = Vec::new();
            for &blk in &missing {
                let key = ResidentKey { layer, block: blk };
                let candidates = self.eviction.rank(&pinned);
                let slot = self.pool.assign(key, &candidates)?;
                pinned.push(slot);
                self.eviction.touch(slot);
                self.eviction
                    .record_score(slot, self.score_host_buf[blk as usize]);
                let idx = tile.iter().position(|&x| x == blk).unwrap();
                block_table[idx] = slot as i32;
                for kh in 0..self.model.num_kv_heads {
                    reqs.push(ReadRequest {
                        group: GroupKey::new(layer, blk, kh, KvKind::K),
                        dst_dev_ptr: self.pool.slot_k_ptr(slot, kh),
                    });
                    reqs.push(ReadRequest {
                        group: GroupKey::new(layer, blk, kh, KvKind::V),
                        dst_dev_ptr: self.pool.slot_v_ptr(slot, kh),
                    });
                }
            }
            self.backend.read(&reqs, stream)?;

            let counts = [(tile.len()) as i32];
            copy_h_to_d_async(
                self.block_table_dev.ptr,
                block_table.as_ptr() as *const c_void,
                tile_cap * 4,
                stream,
            )?;
            copy_h_to_d_async(
                self.counts_dev.ptr,
                counts.as_ptr() as *const c_void,
                4,
                stream,
            )?;
            let (s_blk, s_tok, s_kvh) = self.attn.scratch_pool_strides();
            let v_off = (self.model.num_kv_heads as u64)
                * (self.model.block_size as u64)
                * (self.model.head_dim as u64)
                * 2;
            // 2026-09-25: The mask applies only to the tile that holds the
            // sequence's last block.
            let lbvs = if tile_end == seq_block_ids.len() {
                last_block_valid_slots
            } else {
                self.model.block_size as i32
            };
            self.attn.step_tile_on_stream(
                stream,
                q_dev,
                self.pool.pool_dev_ptr(),
                self.pool.pool_dev_ptr() + v_off,
                self.block_table_dev.ptr,
                self.counts_dev.ptr,
                1,
                s_blk,
                s_tok,
                s_kvh,
                lbvs,
            )?;
            tile_idx = tile_end;
        }
        self.attn.finalize_on_stream(stream, output_dev, 1)?;
        Ok(())
    }

    pub fn pool(&self) -> &ScratchPool {
        &self.pool
    }
    pub fn predictor(&self) -> &Predictor {
        &self.predictor
    }
    pub fn config(&self) -> &HighSpeedSwapConfig {
        &self.cfg
    }
}
