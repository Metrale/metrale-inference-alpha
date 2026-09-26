// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-26: Two passes `ms_phase_qkv` runs after every projection route, before the
//! q/k RMS norms: the per-request Q/K/V LoRA delta and the deferred Q/gate split.
//!
//! Owner: model-layers (qwen3 attention).
//! Invariants:
//! - They read and write `qkv_buf` in the row layout the parent module states: Q at
//!   offset 0, K at `q_proj_bytes`, V after K, rows `per_seq_qkv` bytes apart.

use anyhow::Result;

use super::super::ctx::MultiSeqCtx;
use crate::layers::ops;
use crate::layers::qwen3_attention::Qwen3AttentionLayer;

impl Qwen3AttentionLayer {
    /// 2026-09-25: Split each row's interleaved `[Q | gate]` in place with
    /// `deinterleave_qg`. Runs only on a gated layer with a q adapter, after
    /// `ms_qkv_apply_lora` has folded the delta onto the interleaved values.
    pub(super) fn ms_qkv_deinterleave_q(&self, c: &MultiSeqCtx<'_>) -> Result<()> {
        if !self.gated || !self.q_lora_active() {
            return Ok(());
        }
        for i in 0..c.n {
            let q_out_i = c.qkv_buf.offset(i * c.per_seq_qkv);
            ops::deinterleave_qg(
                c.fwd.gpu,
                self.deinterleave_qg_k,
                q_out_i,
                1,
                c.nq,
                c.hd,
                c.q_proj_dim,
                c.stream,
            )?;
        }
        Ok(())
    }

    /// 2026-09-25: Per-request Q/K/V LoRA delta: each row gets its own
    /// adapter's delta, folded in place into its `qkv_buf` Q, K and V segments
    /// by one bgmv launch per projection. Does nothing unless the layer has
    /// LoRA weights and `seq_slot` is non-null; a projection without a route
    /// is skipped.
    pub(super) fn ms_qkv_apply_lora(&self, c: &MultiSeqCtx<'_>) -> Result<()> {
        let Some(ref lw) = self.lora else {
            return Ok(());
        };
        if c.seq_slot.0 == 0 {
            return Ok(());
        }
        let bf16 = c.bf16;
        let out_row_stride = (c.per_seq_qkv / bf16) as u32; // 2026-09-25: strided qkv_buf rows
        let x_row_stride = c.h as u32; // 2026-09-25: normed rows are contiguous [n, h]
        let kv_bytes = (c.nkv * c.hd) as usize * bf16;
        // 2026-09-25: Q delta onto the segment at offset 0, `q_proj_dim` wide
        // (raw interleaved `[Q | gate]` on a gated layer), before
        // `ms_qkv_deinterleave_q`.
        if let Some(ref route) = lw.q_route {
            let q_out0 = c.qkv_buf;
            ops::lora_delta::apply_lora_bgmv(
                c.fwd.gpu,
                &lw.kernels,
                route,
                c.normed,
                q_out0,
                c.seq_slot,
                c.n as u32,
                x_row_stride,
                out_row_stride,
                c.fwd.buffers.lora_xa(),
                c.stream,
            )?;
        }
        // 2026-09-25: K delta onto the K segment, after Q.
        if let Some(ref route) = lw.k_route {
            let k_out0 = c.qkv_buf.offset(c.q_proj_bytes);
            ops::lora_delta::apply_lora_bgmv(
                c.fwd.gpu,
                &lw.kernels,
                route,
                c.normed,
                k_out0,
                c.seq_slot,
                c.n as u32,
                x_row_stride,
                out_row_stride,
                c.fwd.buffers.lora_xa(),
                c.stream,
            )?;
        }
        // 2026-09-25: V delta onto the V segment, after Q and K.
        if let Some(ref route) = lw.v_route {
            let v_out0 = c.qkv_buf.offset(c.q_proj_bytes + kv_bytes);
            ops::lora_delta::apply_lora_bgmv(
                c.fwd.gpu,
                &lw.kernels,
                route,
                c.normed,
                v_out0,
                c.seq_slot,
                c.n as u32,
                x_row_stride,
                out_row_stride,
                c.fwd.buffers.lora_xa(),
                c.stream,
            )?;
        }
        Ok(())
    }
}
