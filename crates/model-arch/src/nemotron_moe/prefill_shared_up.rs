// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-25: Shared-expert up projection dispatch for
//! `NemotronMoeLayer::prefill`, in precedence order native FP8, W4A4,
//! pre-dequant FP8, transposed NVFP4, plain W4A16.
//!
//! Owner: model-arch (Nemotron-H).
//! Invariants: none beyond the types.

use anyhow::Result;
use metrale_gpu_runtime::gpu::DevicePtr;

use super::NemotronMoeLayer;
use metrale_model_layers::layer::ForwardContext;
use metrale_model_layers::layers::ops;

impl NemotronMoeLayer {
    /// 2026-09-25: Shared expert up GEMM: `[N, h] × [h, shared_inter] → [N, shared_inter]`.
    ///
    /// W4A4 runs `shared_up` in its NVFP4 form with the activations quantized
    /// to NVFP4, from 512 tokens, unless `METRALE_NO_SHARED_W4A4` is set (any
    /// value). The native FP8 arm goes first because W4A4 also quantizes the
    /// activations to 4 bits.
    pub(super) fn prefill_shared_up(
        &self,
        normed: DevicePtr,
        shared_up_out_base: DevicePtr,
        n: u32,
        h: usize,
        shared_inter: u32,
        ctx: &ForwardContext,
        stream: u64,
    ) -> Result<()> {
        let native_shared_up = self.weights.shared_up_fp8.is_some()
            && (self.w8a16_gemm_pipelined_k.0 != 0 || self.w8a16_gemm_k.0 != 0);
        let w4a4 = !native_shared_up
            && n >= 512
            && self.w4a4_gemm_k.0 != 0
            && self.quantize_nvfp4_k.0 != 0
            && ctx.buffers.fp8_act_bytes() >= (shared_inter as usize).max(h) * (n as usize)
            && ctx.levers.shared_w4a4;
        if w4a4 {
            let a4 = ctx.buffers.fp8_act();
            let a4_sf = a4.offset((n as usize) * h / 2);
            ops::quantize_bf16_to_nvfp4(
                ctx.gpu,
                self.quantize_nvfp4_k,
                normed,
                a4,
                a4_sf,
                n,
                h as u32,
                stream,
            )?;
            ops::w4a4_gemm_mfast(
                ctx.gpu,
                self.w4a4_gemm_k,
                a4,
                a4_sf,
                &self.weights.shared_up,
                shared_up_out_base,
                n,
                shared_inter,
                h as u32,
                stream,
            )?;
        } else if let Some(fp8w) = self
            .weights
            .shared_up_fp8
            .as_ref()
            .filter(|_| self.w8a16_gemm_pipelined_k.0 != 0 || self.w8a16_gemm_k.0 != 0)
        {
            let (kern, pipelined) = if self.w8a16_gemm_pipelined_k.0 != 0 {
                (self.w8a16_gemm_pipelined_k, true)
            } else {
                (self.w8a16_gemm_k, false)
            };
            let f = if pipelined {
                ops::w8a16_gemm_pipelined
            } else {
                ops::w8a16_gemm
            };
            f(
                ctx.gpu,
                kern,
                normed,
                fp8w.weight,
                fp8w.row_scale,
                shared_up_out_base,
                n,
                shared_inter,
                h as u32,
                stream,
            )?;
        } else if let Some(w_fp8) = self.shared_up_pd_fp8 {
            ops::fp8_gemm_m128_mfast(
                ctx.gpu,
                self.fp8_gemm_m128_k,
                normed,
                w_fp8,
                shared_up_out_base,
                n,
                shared_inter,
                h as u32,
                stream,
            )?;
        } else if let Some(ref sut) = self.shared_up_t {
            if n > 128 && self.w4a16_gemm_t_m128_k.0 != 0 {
                ops::w4a16_gemm_n128_m128(
                    ctx.gpu,
                    self.w4a16_gemm_t_m128_k,
                    normed,
                    sut,
                    shared_up_out_base,
                    n,
                    shared_inter,
                    h as u32,
                    stream,
                )?;
            } else {
                ops::w4a16_gemm_n128(
                    ctx.gpu,
                    self.w4a16_gemm_t_k,
                    normed,
                    sut,
                    shared_up_out_base,
                    n,
                    shared_inter,
                    h as u32,
                    stream,
                )?;
            }
        } else {
            ops::w4a16_gemm(
                ctx.gpu,
                self.w4a16_gemm_k,
                normed,
                &self.weights.shared_up,
                shared_up_out_base,
                n,
                shared_inter,
                h as u32,
                stream,
            )?;
        }
        Ok(())
    }
}
