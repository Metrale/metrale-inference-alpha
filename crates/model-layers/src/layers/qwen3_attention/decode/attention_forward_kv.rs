// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-25: The K and V projections of single-token decode. By weight format: packed Q2_0, FP8
//! (`w8a16_gemv` per projection), NVFP4 on both (one `w4a16_gemv_dual`), or per projection NVFP4
//! or dense.
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
    pub(super) fn attention_forward_kv(
        &self,
        normed: DevicePtr,
        k_out: DevicePtr,
        v_out: DevicePtr,
        nkv: u32,
        hd: u32,
        h: u32,
        ctx: &ForwardContext,
        stream: u64,
    ) -> Result<()> {
        if self.mla.is_some() {
            // 2026-09-25: Unreachable from `attention_forward`, which returns through the MLA path
            // before calling this.
            return Ok(());
        }

        if let (Some(k_q2), Some(v_q2)) = (
            self.k_weight.as_ref().and_then(|w| w.as_packed_q2()),
            self.v_weight.as_ref().and_then(|w| w.as_packed_q2()),
        ) {
            ops::q2_0_gemv_vec(ctx.gpu, self.q2_0_gemv_k, normed, k_q2, k_out, stream)?;
            ops::q2_0_gemv_vec(ctx.gpu, self.q2_0_gemv_k, normed, v_q2, v_out, stream)?;
            return Ok(());
        }

        if let (Some(k_fp8), Some(v_fp8)) = (
            self.k_weight.as_ref().and_then(|w| w.as_fp8()),
            self.v_weight.as_ref().and_then(|w| w.as_fp8()),
        ) {
            ops::w8a16_gemv(
                ctx.gpu,
                self.w8a16_gemv_k,
                normed,
                k_fp8.weight,
                k_fp8.row_scale,
                k_out,
                nkv * hd,
                h,
                stream,
            )?;
            ops::w8a16_gemv(
                ctx.gpu,
                self.w8a16_gemv_k,
                normed,
                v_fp8.weight,
                v_fp8.row_scale,
                v_out,
                nkv * hd,
                h,
                stream,
            )?;
            return Ok(());
        }

        match (
            self.k_weight.as_ref().and_then(|w| w.as_nvfp4()),
            self.v_weight.as_ref().and_then(|w| w.as_nvfp4()),
        ) {
            (Some(k_fp4), Some(v_fp4)) => {
                ops::w4a16_gemv_dual(
                    ctx.gpu,
                    self.w4a16_gemv_dual_k,
                    normed,
                    k_fp4,
                    k_out,
                    v_fp4,
                    v_out,
                    nkv * hd,
                    h,
                    stream,
                )?;
            }
            _ => {
                if let Some(nvfp4) = self.k_weight.as_ref().and_then(|w| w.as_nvfp4()) {
                    self.nvfp4_decode_gemv(
                        ctx.gpu,
                        ctx.levers.gemv_sw,
                        normed,
                        nvfp4,
                        k_out,
                        nkv * hd,
                        h,
                        stream,
                    )?;
                } else {
                    ops::dense_gemv(
                        ctx.gpu,
                        self.dense_gemv_k,
                        normed,
                        &self.attn.k_proj,
                        k_out,
                        nkv * hd,
                        h,
                        stream,
                    )?;
                }
                if let Some(nvfp4) = self.v_weight.as_ref().and_then(|w| w.as_nvfp4()) {
                    self.nvfp4_decode_gemv(
                        ctx.gpu,
                        ctx.levers.gemv_sw,
                        normed,
                        nvfp4,
                        v_out,
                        nkv * hd,
                        h,
                        stream,
                    )?;
                } else {
                    ops::dense_gemv(
                        ctx.gpu,
                        self.dense_gemv_k,
                        normed,
                        &self.attn.v_proj,
                        v_out,
                        nkv * hd,
                        h,
                        stream,
                    )?;
                }
            }
        }
        Ok(())
    }
}
