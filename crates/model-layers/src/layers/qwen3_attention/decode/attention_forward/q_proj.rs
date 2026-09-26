// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-26: The Q projection of single-token `attention_forward`, and its layer-0 profile
//! log.
//!
//! Owner: model-layers attention decode.
//! Invariants:
//! - On a gated layer `q_out` ends as `[Q_all | Gate_all]`, with the q LoRA delta folded in
//!   before the deinterleave; on an ungated layer the delta is folded onto the final `q_out`.

use anyhow::Result;
use metrale_gpu_runtime::gpu::DevicePtr;

use super::super::super::Qwen3AttentionLayer;
use crate::layer::ForwardContext;
use crate::layers::ops;

impl Qwen3AttentionLayer {
    /// 2026-09-26: `normed` through q_proj (the Q+gate projection on a gated layer) into `q_out`,
    /// dispatched on the weight format: packed Q2_0, FP8, NVFP4 or dense BF16.
    pub(super) fn attention_forward_q_proj(
        &self,
        ctx: &ForwardContext,
        normed: DevicePtr,
        q_out: DevicePtr,
        q_dim: u32,
        q_proj_dim: u32,
        nq: u32,
        hd: u32,
        h: u32,
        stream: u64,
    ) -> Result<()> {
        if self.gated {
            // 2026-09-25: Q+gate projection, then deinterleave to `[Q_all | Gate_all]`.
            if let Some(q2) = self.q_weight.as_ref().and_then(|w| w.as_packed_q2()) {
                // 2026-09-25: Packed Q2_0 weight: 2-bit GEMV into the interleaved `[Q|gate]`, then
                // the same deinterleave the dense arm uses.
                ops::q2_0_gemv_vec(ctx.gpu, self.q2_0_gemv_k, normed, q2, q_out, stream)?;
                ops::deinterleave_qg(
                    ctx.gpu,
                    self.deinterleave_qg_k,
                    q_out,
                    1,
                    nq,
                    hd,
                    nq * hd * 2,
                    stream,
                )?;
            } else if let Some(fp8) = self.q_weight.as_ref().and_then(|w| w.as_fp8()) {
                ops::w8a16_gemv(
                    ctx.gpu,
                    self.w8a16_gemv_k,
                    normed,
                    fp8.weight,
                    fp8.row_scale,
                    q_out,
                    q_proj_dim,
                    h,
                    stream,
                )?;
                // 2026-09-25: q_proj LoRA on the raw interleaved `[Q|gate]`, before the
                // deinterleave.
                self.apply_q_lora(ctx, normed, q_out, stream)?;
                ops::deinterleave_qg(
                    ctx.gpu,
                    self.deinterleave_qg_k,
                    q_out,
                    1,
                    nq,
                    hd,
                    nq * hd * 2,
                    stream,
                )?;
            } else if let Some(nvfp4) = self.q_weight.as_ref().and_then(|w| w.as_nvfp4()) {
                if self.lora.as_ref().and_then(|lw| lw.q.as_ref()).is_some() {
                    // 2026-09-25: With a q adapter resident, the fused GEMV+deinterleave is split
                    // into GEMV, q LoRA fold, deinterleave, so the delta is added in the
                    // interleaved layout of the q_proj output.
                    self.nvfp4_decode_gemv(
                        ctx.gpu,
                        ctx.levers.gemv_sw,
                        normed,
                        nvfp4,
                        q_out,
                        q_proj_dim,
                        h,
                        stream,
                    )?;
                    self.apply_q_lora(ctx, normed, q_out, stream)?;
                    ops::deinterleave_qg(
                        ctx.gpu,
                        self.deinterleave_qg_k,
                        q_out,
                        1,
                        nq,
                        hd,
                        nq * hd * 2,
                        stream,
                    )?;
                } else {
                    ops::w4a16_gemv_qg(
                        ctx.gpu,
                        self.w4a16_gemv_qg_k,
                        normed,
                        nvfp4,
                        q_out,
                        q_proj_dim,
                        h,
                        nq,
                        hd,
                        stream,
                    )?;
                }
            } else {
                ops::dense_gemv(
                    ctx.gpu,
                    self.dense_gemv_k,
                    normed,
                    &self.attn.q_proj,
                    q_out,
                    q_proj_dim,
                    h,
                    stream,
                )?;
                // 2026-09-25: q_proj LoRA on the raw interleaved `[Q|gate]`, before the
                // deinterleave.
                self.apply_q_lora(ctx, normed, q_out, stream)?;
                ops::deinterleave_qg(
                    ctx.gpu,
                    self.deinterleave_qg_k,
                    q_out,
                    1,
                    nq,
                    hd,
                    nq * hd * 2,
                    stream,
                )?;
            }
        } else {
            if let Some(q2) = self.q_weight.as_ref().and_then(|w| w.as_packed_q2()) {
                ops::q2_0_gemv_vec(ctx.gpu, self.q2_0_gemv_k, normed, q2, q_out, stream)?;
            } else if let Some(fp8) = self.q_weight.as_ref().and_then(|w| w.as_fp8()) {
                ops::w8a16_gemv(
                    ctx.gpu,
                    self.w8a16_gemv_k,
                    normed,
                    fp8.weight,
                    fp8.row_scale,
                    q_out,
                    q_dim,
                    h,
                    stream,
                )?;
            } else if let Some(nvfp4) = self.q_weight.as_ref().and_then(|w| w.as_nvfp4()) {
                self.nvfp4_decode_gemv(
                    ctx.gpu,
                    ctx.levers.gemv_sw,
                    normed,
                    nvfp4,
                    q_out,
                    q_dim,
                    h,
                    stream,
                )?;
            } else {
                ops::dense_gemv(
                    ctx.gpu,
                    self.dense_gemv_k,
                    normed,
                    &self.attn.q_proj,
                    q_out,
                    q_dim,
                    h,
                    stream,
                )?;
            }
            // 2026-09-25: Ungated: no deinterleave, so the q LoRA folds onto the final `q_out`.
            self.apply_q_lora(ctx, normed, q_out, stream)?;
        }
        Ok(())
    }

    /// 2026-09-26: The profile log `attention_forward` calls after the Q projection.
    pub(super) fn attention_forward_q_diag(
        &self,
        ctx: &ForwardContext,
        normed: DevicePtr,
        q_out: DevicePtr,
        nq: u32,
        hd: u32,
        h: u32,
        stream: u64,
    ) -> Result<()> {
        if self.attn_layer_idx == 0 && ctx.profile {
            ctx.gpu.synchronize(stream)?;
            let mut input_buf = vec![0u8; 16];
            ctx.gpu.copy_d2h(normed, &mut input_buf)?;
            let input_vals: Vec<f32> = input_buf
                .chunks_exact(2)
                .map(|c| {
                    let bits = u16::from_le_bytes([c[0], c[1]]);
                    f32::from_bits((bits as u32) << 16)
                })
                .collect();
            let mut q_buf = vec![0u8; 16];
            ctx.gpu.copy_d2h(q_out, &mut q_buf)?;
            let q_vals: Vec<f32> = q_buf
                .chunks_exact(2)
                .map(|c| {
                    let bits = u16::from_le_bytes([c[0], c[1]]);
                    f32::from_bits((bits as u32) << 16)
                })
                .collect();
            tracing::info!(target: "metrale_model_layers::layers::qwen3_attention::decode::attention_forward", "GEMV_DIAG L0: input[0:8]={:.4?} q_out[0:8]={:.4?} nq={nq} hd={hd} h={h}",
                input_vals,
                q_vals
            );
        }
        Ok(())
    }
}
