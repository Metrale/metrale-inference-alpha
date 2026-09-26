// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-26: The shared-expert GEMMs of `forward_prefill_fp8` when no BF16 shared-expert
//! copy is installed: W8A8 (block-scaled) or W8A16. Each leaves the shared expert's output
//! in `ctx.buffers.attn_output()`, which the shared blend reads.
//!
//! Owner: model-layers (MoE).
//! Invariants: none beyond the types.

use super::*;

impl MoeLayer {
    /// 2026-09-26: W8A8 shared expert: `input` quantised to FP8 per row and 128-column
    /// group, then `fp8_gemm_t_blockscaled` for gate, up and down. Every launch goes to
    /// `shared_stream`; `stream`, the caller's, is what the profile timer synchronizes.
    pub(super) fn fp8_prefill_shared_w8a8(
        &self,
        input: DevicePtr,
        sh: &Fp8ExpertWeight,
        n: u32,
        h: u32,
        shared_inter: u32,
        fp8_scratch: &MoeFp8Scratch,
        ctx: &ForwardContext,
        stream: u64,
        shared_stream: u64,
        mt: &mut Option<std::time::Instant>,
    ) -> Result<()> {
        macro_rules! mprof {
            ($label:expr) => {
                mprof_step!(*mt, ctx, stream, n, $label)
            };
        }
        let shared_gate_out = ctx.buffers.ssm_deinterleaved();
        let shared_up_out = ctx.buffers.ssm_qkvz();
        let input_fp8 = fp8_scratch.activation;
        let input_scale = fp8_scratch.scales;
        ops::per_token_group_quant_fp8(
            ctx.gpu,
            self.per_token_group_quant_fp8_k,
            input,
            input_fp8,
            input_scale,
            n,
            h,
            shared_stream,
        )?;
        ops::fp8_gemm_t_blockscaled(
            ctx.gpu,
            self.fp8_gemm_t_blockscaled_k,
            input_fp8,
            input_scale,
            sh.gate_proj.weight,
            sh.gate_proj.row_scale,
            shared_gate_out,
            n,
            shared_inter,
            h,
            shared_stream,
        )?;
        ops::fp8_gemm_t_blockscaled(
            ctx.gpu,
            self.fp8_gemm_t_blockscaled_k,
            input_fp8,
            input_scale,
            sh.up_proj.weight,
            sh.up_proj.row_scale,
            shared_up_out,
            n,
            shared_inter,
            h,
            shared_stream,
        )?;
        let shared_down_out = ctx.buffers.attn_output();
        let down_in_fp8 = fp8_scratch.activation;
        let down_in_scale = fp8_scratch.scales;
        if self.fused_silu_quant_ok(shared_inter) {
            // 2026-09-25: Nothing after this reads the BF16 post-SiLU shared
            // intermediate, so the BF16 output pointer is NULL.
            ops::silu_mul_quant_fp8(
                ctx.gpu,
                self.silu_mul_quant_fp8_k,
                shared_gate_out,
                shared_up_out,
                down_in_fp8,
                down_in_scale,
                metrale_gpu_runtime::gpu::DevicePtr::NULL,
                n,
                shared_inter,
                shared_stream,
            )?;
        } else {
            ops::silu_mul(
                ctx.gpu,
                self.moe_act_mul,
                shared_gate_out,
                shared_up_out,
                shared_gate_out,
                n * shared_inter,
                shared_stream,
            )?;
            ops::per_token_group_quant_fp8(
                ctx.gpu,
                self.per_token_group_quant_fp8_k,
                shared_gate_out,
                down_in_fp8,
                down_in_scale,
                n,
                shared_inter,
                shared_stream,
            )?;
        }
        mprof!("silu_mul_quant");
        ops::fp8_gemm_t_blockscaled(
            ctx.gpu,
            self.fp8_gemm_t_blockscaled_k,
            down_in_fp8,
            down_in_scale,
            sh.down_proj.weight,
            sh.down_proj.row_scale,
            shared_down_out,
            n,
            h,
            shared_inter,
            shared_stream,
        )?;
        Ok(())
    }

    /// 2026-09-26: W8A16 shared expert: gate, up and down through `sh_gemm`.
    pub(super) fn fp8_prefill_shared_w8a16(
        &self,
        input: DevicePtr,
        sh: &Fp8ExpertWeight,
        n: u32,
        h: u32,
        shared_inter: u32,
        ctx: &ForwardContext,
        stream: u64,
        mt: &mut Option<std::time::Instant>,
    ) -> Result<()> {
        macro_rules! mprof {
            ($label:expr) => {
                mprof_step!(*mt, ctx, stream, n, $label)
            };
        }
        let shared_gate_out = ctx.buffers.ssm_deinterleaved();
        let shared_up_out = ctx.buffers.ssm_qkvz();
        // 2026-09-25: The pipelined W8A16 GEMM when it resolved, else
        // `w8a16_gemm` (kernels/strix-hip ships only the latter).
        let use_pipelined = self.w8a16_gemm_pipelined_k.0 != 0;
        let sh_gemm = |inp, w, sc, outp, mm, nn, kk| -> anyhow::Result<()> {
            if use_pipelined {
                ops::w8a16_gemm_pipelined(
                    ctx.gpu,
                    self.w8a16_gemm_pipelined_k,
                    inp,
                    w,
                    sc,
                    outp,
                    mm,
                    nn,
                    kk,
                    stream,
                )
            } else {
                ops::w8a16_gemm(
                    ctx.gpu,
                    self.w8a16_gemm_k,
                    inp,
                    w,
                    sc,
                    outp,
                    mm,
                    nn,
                    kk,
                    stream,
                )
            }
        };
        sh_gemm(
            input,
            sh.gate_proj.weight,
            sh.gate_proj.row_scale,
            shared_gate_out,
            n,
            shared_inter,
            h,
        )?;
        sh_gemm(
            input,
            sh.up_proj.weight,
            sh.up_proj.row_scale,
            shared_up_out,
            n,
            shared_inter,
            h,
        )?;
        ops::silu_mul(
            ctx.gpu,
            self.moe_act_mul,
            shared_gate_out,
            shared_up_out,
            shared_gate_out,
            n * shared_inter,
            stream,
        )?;
        mprof!("silu_mul");
        let shared_down_out = ctx.buffers.attn_output();
        sh_gemm(
            shared_gate_out,
            sh.down_proj.weight,
            sh.down_proj.row_scale,
            shared_down_out,
            n,
            h,
            shared_inter,
        )?;
        Ok(())
    }
}
