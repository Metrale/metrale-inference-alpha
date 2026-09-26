// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-25: The O projection of single-token decode, then its LoRA delta. By weight: the MLA
//! `wo` (NVFP4 or dense), a BF16 `o_dense_bf16`, packed Q2_0, FP8 (`w8a16_gemv`), or NVFP4.
//! `attention_forward`, the only caller, returns through the MLA path before calling this, so the
//! MLA arm is not reached from it.
//!
//! Owner: model-layers attention decode.
//! Invariants: none beyond the types.

use anyhow::Result;
use metrale_gpu_runtime::gpu::DevicePtr;

use super::super::Qwen3AttentionLayer;
use crate::layer::ForwardContext;
use crate::layers::ops;

impl Qwen3AttentionLayer {
    #[allow(clippy::too_many_arguments)]
    pub(super) fn attention_forward_oproj(
        &self,
        attn_out: DevicePtr,
        nq: u32,
        hd: u32,
        h: u32,
        ctx: &ForwardContext,
        stream: u64,
    ) -> Result<DevicePtr> {
        let o_out = ctx.buffers.norm_output();
        if let Some(ref mla) = self.mla {
            if let Some(ref wo_nvfp4) = mla.wo_nvfp4 {
                self.nvfp4_decode_gemv(
                    ctx.gpu,
                    ctx.levers.gemv_sw,
                    attn_out,
                    wo_nvfp4,
                    o_out,
                    h,
                    nq * hd,
                    stream,
                )?;
            } else {
                ops::dense_gemv(
                    ctx.gpu,
                    self.dense_gemv_k,
                    attn_out,
                    &mla.wo,
                    o_out,
                    h,
                    nq * hd,
                    stream,
                )?;
            }
        } else if let Some(o_bf16) = self.o_dense_bf16.as_ref() {
            ops::dense_gemv(
                ctx.gpu,
                self.dense_gemv_k,
                attn_out,
                o_bf16,
                o_out,
                h,
                nq * hd,
                stream,
            )?;
        } else if let Some(q2) = self.o_weight.as_ref().and_then(|w| w.as_packed_q2()) {
            ops::q2_0_gemv_vec(ctx.gpu, self.q2_0_gemv_k, attn_out, q2, o_out, stream)?;
        } else if let Some(fp8) = self.o_weight.as_ref().and_then(|w| w.as_fp8()) {
            ops::w8a16_gemv(
                ctx.gpu,
                self.w8a16_gemv_k,
                attn_out,
                fp8.weight,
                fp8.row_scale,
                o_out,
                h,
                nq * hd,
                stream,
            )?;
        } else {
            self.nvfp4_decode_gemv(
                ctx.gpu,
                ctx.levers.gemv_sw,
                attn_out,
                &self.attn.o_proj,
                o_out,
                h,
                nq * hd,
                stream,
            )?;
        }
        // 2026-09-25: LoRA delta on o_proj. `attn_out` already carries the caller's gates, so the
        // delta sees the o_proj input.
        if let Some(ref lw) = self.lora
            && let Some(ref pair) = lw.o
        {
            debug_assert_eq!(pair.k_in, nq * hd);
            debug_assert_eq!(pair.n_out, h);
            // 2026-09-25: The bgmv for this request's adapter when a per-sequence slot and a route
            // exist, else the installed active pair.
            let seq_slot = ctx
                .attn_metadata
                .map(|m| m.seq_slot)
                .unwrap_or(DevicePtr(0));
            if seq_slot.0 != 0
                && let Some(ref route) = lw.o_route
            {
                ops::lora_delta::apply_lora_bgmv(
                    ctx.gpu,
                    &lw.kernels,
                    route,
                    attn_out,
                    o_out,
                    seq_slot,
                    1,
                    pair.k_in,
                    pair.n_out,
                    ctx.buffers.lora_xa(),
                    stream,
                )?;
            } else {
                ops::lora_delta::apply_lora_delta(
                    ctx.gpu,
                    &lw.kernels,
                    pair,
                    attn_out,
                    o_out,
                    1,
                    ctx.buffers.lora_xa(),
                    ctx.buffers.lora_delta(),
                    stream,
                )?;
            }
        }
        Ok(o_out)
    }
}
