// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-25: The shared-expert phase of the NVFP4 `MoeLayer::forward_prefill`.
//!
//! Owner: model-layers (MoE).
//! Invariants: none beyond the types.

use super::*;

impl MoeLayer {
    /// 2026-09-25: The shared expert for `n` prefill rows, launched on `aux`, with
    /// the result in `attn_output`. The installed BF16 shared expert wins;
    /// otherwise gate/up and down each use the predequantized FP8 copy, the
    /// transposed W4A16 weights or the W4A16 weights, with `moe_act_mul` between.
    /// With `use_overlap`, `aux` first waits on `stream` (`event_a`) and records
    /// `event_b` when done. Returns at once when `shared_inter == 0`.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn run_shared_expert_prefill(
        &self,
        input: DevicePtr,
        n: u32,
        h: u32,
        shared_inter: u32,
        aux: u64,
        stream: u64,
        use_overlap: bool,
        ctx: &ForwardContext,
    ) -> Result<()> {
        if shared_inter == 0 {
            return Ok(());
        }
        if use_overlap {
            ctx.gpu.record_event(self.event_a, stream)?;
            ctx.gpu.stream_wait_event(aux, self.event_a)?;
        }

        let shared_gate_out = ctx.buffers.ssm_deinterleaved();
        let shared_up_out = ctx.buffers.ssm_qkvz();
        let shared_down_out = ctx.buffers.attn_output();
        if self.run_bf16_shared_expert(
            input,
            n,
            h,
            shared_inter,
            shared_gate_out,
            shared_up_out,
            shared_down_out,
            ctx,
            aux,
        )? {
            if use_overlap {
                ctx.gpu.record_event(self.event_b, aux)?;
            }
            return Ok(());
        }

        if let (Some(sg_fp8), Some(su_fp8)) = (self.shared_gate_fp8, self.shared_up_fp8) {
            ops::fp8_gemm_n128(
                ctx.gpu,
                self.fp8_gemm_k,
                input,
                sg_fp8,
                shared_gate_out,
                n,
                shared_inter,
                h,
                aux,
            )?;
            ops::fp8_gemm_n128(
                ctx.gpu,
                self.fp8_gemm_k,
                input,
                su_fp8,
                shared_up_out,
                n,
                shared_inter,
                h,
                aux,
            )?;
        } else if let (Some(sg), Some(su), Some(_sd)) =
            (&self.shared_gate_t, &self.shared_up_t, &self.shared_down_t)
        {
            ops::w4a16_gemm_n128(
                ctx.gpu,
                self.w4a16_gemm_t,
                input,
                sg,
                shared_gate_out,
                n,
                shared_inter,
                h,
                aux,
            )?;
            ops::w4a16_gemm_n128(
                ctx.gpu,
                self.w4a16_gemm_t,
                input,
                su,
                shared_up_out,
                n,
                shared_inter,
                h,
                aux,
            )?;
        } else {
            ops::w4a16_gemm(
                ctx.gpu,
                self.w4a16_gemm,
                input,
                &self.weights.shared_expert.gate_proj,
                shared_gate_out,
                n,
                shared_inter,
                h,
                aux,
            )?;
            ops::w4a16_gemm(
                ctx.gpu,
                self.w4a16_gemm,
                input,
                &self.weights.shared_expert.up_proj,
                shared_up_out,
                n,
                shared_inter,
                h,
                aux,
            )?;
        }

        ops::silu_mul(
            ctx.gpu,
            self.moe_act_mul,
            shared_gate_out,
            shared_up_out,
            shared_gate_out,
            n * shared_inter,
            aux,
        )?;
        if let Some(sd_fp8) = self.shared_down_fp8 {
            ops::fp8_gemm_n128(
                ctx.gpu,
                self.fp8_gemm_k,
                shared_gate_out,
                sd_fp8,
                shared_down_out,
                n,
                h,
                shared_inter,
                aux,
            )?;
        } else if let Some(sd) = &self.shared_down_t {
            ops::w4a16_gemm_n128(
                ctx.gpu,
                self.w4a16_gemm_t,
                shared_gate_out,
                sd,
                shared_down_out,
                n,
                h,
                shared_inter,
                aux,
            )?;
        } else {
            ops::w4a16_gemm(
                ctx.gpu,
                self.w4a16_gemm,
                shared_gate_out,
                &self.weights.shared_expert.down_proj,
                shared_down_out,
                n,
                h,
                shared_inter,
                aux,
            )?;
        }

        if use_overlap {
            ctx.gpu.record_event(self.event_b, aux)?;
        }
        Ok(())
    }
}
