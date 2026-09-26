// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-26: The per-forward sizes and inputs of `forward_block`: row counts, projection
//! widths, the ctx rows the non-paged path projects, and the paged-path block table.
//!
//! Owner: model-arch (DFlash drafter).
//! Invariants:
//! - `block_dims` reads no device state and launches nothing.

use metrale_gpu_runtime::gpu::{DevicePtr, GpuBackend};
use metrale_model_layers::layer::ForwardContext;

use crate::dflash_head::levers::DFlashLevers;
use crate::dflash_head::{BlockDiffusionDraftHead, DflashBatch};

/// 2026-09-26: The locals `forward_block` computes before its first launch, under the same
/// names, plus its `option_b`, `last_token`, `position` and `batch` arguments.
pub(super) struct BlockDims<'a> {
    pub(super) gpu: &'a dyn GpuBackend,
    pub(super) n_seq: usize,
    pub(super) width: usize,
    pub(super) block_g: u32,
    pub(super) g: u32,
    pub(super) rows_total: usize,
    pub(super) h: u32,
    pub(super) q_dim: u32,
    pub(super) kv_dim: u32,
    pub(super) inter: u32,
    pub(super) bf16: usize,
    pub(super) inv_sqrt_d: f32,
    pub(super) levers: DFlashLevers,
    pub(super) ctx_base_ptr: Option<DevicePtr>,
    pub(super) ctx_total: usize,
    pub(super) eff_ctx: usize,
    pub(super) option_b: Option<(DevicePtr, u32)>,
    pub(super) option_b_block_table: Option<DevicePtr>,
    pub(super) option_b_ctx_count: u32,
    pub(super) option_b_on: bool,
    pub(super) n_attn: u32,
    pub(super) target_hidden_dim: usize,
    pub(super) ctx_slot_bytes: usize,
    pub(super) last_token: u32,
    pub(super) position: usize,
    pub(super) batch: Option<&'a DflashBatch<'a>>,
}

impl BlockDiffusionDraftHead {
    /// 2026-09-26: The `BlockDims` of one `forward_block` call.
    pub(super) fn block_dims<'a>(
        &self,
        last_token: u32,
        position: usize,
        ctx: &'a ForwardContext<'_>,
        ctx_buffer: Option<(DevicePtr, usize)>,
        option_b: Option<(DevicePtr, u32)>,
        batch: Option<&'a DflashBatch<'a>>,
    ) -> BlockDims<'a> {
        let n_seq = batch.map_or(1usize, |b| b.last_tokens.len().max(1));
        // 2026-09-25: `width` is the rows per sequence and `g` the rows in this forward.
        // Weight-bearing ops take `g`; attention, the KV slot writes and the selector's
        // chain seed work per sequence band. The layer helpers and the DFlash2 calls read
        // `block_g()` again; only `propose_drafts` and `propose_batch` call
        // `set_block_g`, both before this forward.
        let width = self.block_g();
        let block_g = width as u32;
        let g = block_g * n_seq as u32;
        let rows_total = width * n_seq;
        let h = self.hidden_size as u32;
        let q_dim = (self.num_q_heads * self.head_dim) as u32;
        let kv_dim = (self.num_kv_heads * self.head_dim) as u32;
        let inter = self.intermediate_size as u32;
        let bf16 = 2usize;
        let inv_sqrt_d = 1.0f32 / (self.head_dim as f32).sqrt();
        let gpu = ctx.gpu;

        // 2026-09-25: `eff_ctx` is how many of the most recent ctx rows the non-paged
        // path projects: the accumulator's fill capped at `ctx_window`, or
        // `METRALE_DFLASH_DEBUG_CTX_USED` (also capped), or 0 under
        // `METRALE_DFLASH_DEBUG_CTX_OFF=1`. Every lever here comes from `self.levers`,
        // resolved when the head was built; this per-step path reads no environment
        // (hot_path_env_guards.rs checks that).
        let levers = self.levers;
        let force_no_ctx = levers.force_no_ctx;
        let force_ctx_used = levers.force_ctx_used;
        let (ctx_base_ptr, ctx_total, eff_ctx) = match ctx_buffer {
            Some(_) if force_no_ctx => (None, 0, 0),
            Some((p, n)) => {
                let eff = match force_ctx_used {
                    Some(forced) => forced.min(n).min(self.ctx_window),
                    None => n.min(self.ctx_window),
                };
                (Some(p), n, eff)
            }
            None => (None, 0, 0),
        };

        // 2026-09-25: On the paged path the caller has already written the ctx K/V into
        // the paged cache, so `eff_ctx` is 0: the embed, the position ids and the layers
        // cover only the block rows.
        let (option_b_block_table, option_b_ctx_count) = match option_b {
            Some((bt, cc)) => (Some(bt), cc),
            None => (None, 0),
        };
        let option_b_on = option_b_block_table.is_some();
        let eff_ctx = if option_b_on { 0 } else { eff_ctx };
        let _ = ctx_base_ptr;
        // 2026-09-25: Rows the embed and the non-paged layers cover: the `eff_ctx` ctx rows
        // plus `width` rows per sequence.
        let n_attn = (eff_ctx + width * n_seq) as u32;
        let target_hidden_dim = self.target_layer_ids.len() * self.target_hidden_size;
        let ctx_slot_bytes = target_hidden_dim * bf16;
        BlockDims {
            gpu,
            n_seq,
            width,
            block_g,
            g,
            rows_total,
            h,
            q_dim,
            kv_dim,
            inter,
            bf16,
            inv_sqrt_d,
            levers,
            ctx_base_ptr,
            ctx_total,
            eff_ctx,
            option_b,
            option_b_block_table,
            option_b_ctx_count,
            option_b_on,
            n_attn,
            target_hidden_dim,
            ctx_slot_bytes,
            last_token,
            position,
            batch,
        }
    }
}
