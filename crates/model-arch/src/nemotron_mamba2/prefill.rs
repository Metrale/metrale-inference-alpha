// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-25: `NemotronMamba2Layer` prefill: RMS norm, in_proj GEMM, causal
//! conv1d, the Mamba-2 scan (SSD chunked, persistent or plain sequential),
//! gated RMS norm, out_proj and the residual add, over all prompt tokens.
//!
//! Owner: model-arch (Nemotron-H).
//! Invariants: none beyond the types.

use anyhow::Result;
use metrale_gpu_runtime::gpu::DevicePtr;

use super::NemotronMamba2Layer;
use metrale_model_layers::layer::{ForwardContext, LayerState, SsmLayerState};
use metrale_model_layers::layers::ops;

impl NemotronMamba2Layer {
    #[allow(clippy::too_many_arguments)]
    pub(super) fn prefill_ssm(
        &self,
        hidden: DevicePtr,
        residual: DevicePtr,
        num_tokens: usize,
        state: &mut dyn LayerState,
        ctx: &ForwardContext,
        stream: u64,
    ) -> Result<()> {
        let h = ctx.config.hidden_size;
        let eps = ctx.config.rms_norm_eps as f32;
        let n = num_tokens as u32;
        let bf16 = 2usize;

        let ssm_state = state
            .as_any_mut()
            .downcast_mut::<SsmLayerState>()
            .ok_or_else(|| anyhow::anyhow!("Expected SsmLayerState"))?;

        let gs = self.n_groups * self.state_size;

        let normed = ctx.buffers.norm_output();
        ops::rms_norm_residual(
            ctx.gpu,
            self.rms_norm_residual_k,
            hidden,
            &self.input_norm,
            normed,
            residual,
            n,
            h as u32,
            eps,
            stream,
        )?;

        // 2026-09-25: Each token's `proj` row is
        // `[z (d_inner) | xBC (d_xbc) | dt (num_heads)]`.
        let proj = ctx.buffers.ssm_qkvz();
        // 2026-09-25: `fp8_a`: the pre-dequant FP8 arms cast the activations to
        // FP8 once (`bf16_to_fp8`) and run the FP8 x FP8 kernel, from 512
        // tokens, when both kernels resolved and `fp8_act` holds
        // n * max(d_inner, h) bytes.
        let fp8_a = n >= 512
            && self.fp8_fp8_gemm_t_k.0 != 0
            && self.bf16_to_fp8_k.0 != 0
            && ctx.buffers.fp8_act_bytes() >= (n as usize) * self.d_inner.max(h);
        // 2026-09-25: `w4a4`: the NVFP4 weights with the activations quantized
        // to NVFP4 per call (`quantize_bf16_to_nvfp4`), which changes the
        // activation numerics; setting `METRALE_NO_SSM_W4A4` (any value) turns
        // it off. Scratch: packed A at fp8_act[0], scales at fp8_act[n*K/2];
        // n*K*9/16 bytes in all, within the n*K this gate requires.
        let w4a4 = n >= 512
            && self.w4a4_gemm_k.0 != 0
            && self.quantize_nvfp4_k.0 != 0
            && ctx.buffers.fp8_act_bytes() >= (n as usize) * self.d_inner.max(h)
            && ctx.levers.ssm_w4a4;
        // 2026-09-25: The pre-dequant FP8 arms launch `fp8_fp8_gemm_t_m128_mfast`
        // (with `fp8_a`) or `fp8_gemm_t_m128_mfast`. Both come from `try_kernel`,
        // which returns a null handle when the model's `w4a16` module lacks the
        // entry (the gb10 Nano and Super trees take `w4a16_gemm.cu` from
        // deepseek-v4-flash, which defines `fp8_gemm_t` and `fp8_fp8_gemm_t`
        // only). `fp8_a` already requires its kernel, so only the other is
        // checked; without it prefill falls through to the transposed NVFP4 and
        // plain `w4a16_gemm` arms.
        let pd_fp8_ok = fp8_a || self.fp8_gemm_t_k.0 != 0;
        self.prefill_in_proj(normed, proj, n, h, fp8_a, w4a4, pd_fp8_ok, ctx, stream)?;

        let xbc_ptr = proj.offset(self.d_inner * bf16);
        let conv_out = ctx.buffers.ssm_deinterleaved();
        ops::conv1d_update_prefill(
            ctx.gpu,
            self.conv1d_prefill_k,
            self.conv1d_prefill_tp_k,
            ssm_state.conv_state,
            xbc_ptr,
            &self.ssm.conv1d_weight,
            self.ssm.conv1d_bias.weight,
            conv_out,
            self.d_xbc as u32,
            self.d_conv as u32,
            n,
            self.in_proj_size as u32,
            self.d_xbc as u32,
            stream,
        )?;

        let x_ptr = conv_out;
        let b_ptr = conv_out.offset(self.d_inner * bf16);
        let c_ptr = conv_out.offset((self.d_inner + gs) * bf16);
        let dt_ptr = proj.offset((self.d_inner + self.d_xbc) * bf16);
        let y_out = ctx.buffers.attn_output();
        // 2026-09-25: The scan: the SSD chunked scan (chunks of `SSD_L` tokens)
        // when its kernels and scratch exist, the shapes divide, its shared
        // memory fits and `METRALE_NO_SSD` is unset; else the persistent
        // sequential kernel when it resolved and `METRALE_NO_SSM_PERSISTENT` is
        // unset; else the plain sequential kernel.
        let ssd_ok = self.ssd_cumsum_k.0 != 0
            && self.ssd_bmm_k.0 != 0
            && self.ssd_scan_k.0 != 0
            && ctx.buffers.ssd_scratch() != metrale_gpu_runtime::gpu::DevicePtr::NULL
            && self.head_dim.is_multiple_of(ops::SSD_PT as usize)
            && self.state_size.is_multiple_of(8)
            && (self.state_size / 8).is_multiple_of(4)
            // 2026-09-25: The scan's dynamic shared memory grows with
            // `state_size` (`ssd_scan_smem`: 91,392 B at 96, 115,968 B at 128)
            // and must fit `MAX_DYNAMIC_SMEM` (101,376 B); over it the launch
            // returns an error instead of falling back, so the fit is checked
            // here.
            && ops::ssd_scan_fits(self.state_size as u32)
            && ctx.levers.ssd;

        if ssd_ok {
            let l = ops::SSD_L;
            let nchunks = n.div_ceil(l);
            let heads = self.num_heads as u32;
            let groups = self.n_groups as u32;
            let scratch = ctx.buffers.ssd_scratch();
            let dt_bytes = (heads * nchunks * l * 4) as usize;
            let dt_f32 = scratch;
            let da_cs = scratch.offset(dt_bytes);
            let cb = scratch.offset(2 * dt_bytes);

            ops::mamba2_ssd_cumsum(
                ctx.gpu,
                self.ssd_cumsum_k,
                dt_ptr,
                self.ssm.a_log.weight,
                self.ssm.dt_bias.weight,
                dt_f32,
                da_cs,
                n,
                heads,
                nchunks,
                1,
                self.in_proj_size as u32,
                1e-9,
                1e9,
                stream,
            )?;
            ops::mamba2_ssd_bmm(
                ctx.gpu,
                self.ssd_bmm_k,
                b_ptr,
                c_ptr,
                cb,
                n,
                nchunks,
                groups,
                self.state_size as u32,
                1,
                self.d_xbc as u32,
                stream,
            )?;
            ops::mamba2_ssd_scan(
                ctx.gpu,
                self.ssd_scan_k,
                ssm_state.h_state,
                x_ptr,
                b_ptr,
                c_ptr,
                self.ssm.d_param.weight,
                dt_f32,
                da_cs,
                cb,
                y_out,
                n,
                heads,
                self.head_dim as u32,
                self.state_size as u32,
                groups,
                nchunks,
                1,
                self.d_xbc as u32,
                self.d_xbc as u32,
                self.d_inner as u32,
                stream,
            )?;
        } else if self.mamba2_ssm_prefill_persistent_k.0 != 0
            // 2026-09-25: `ssm_persistent` is off when `METRALE_NO_SSM_PERSISTENT` is set.
            && ctx.levers.ssm_persistent
        {
            ops::mamba2_ssm_prefill_persistent(
                ctx.gpu,
                self.mamba2_ssm_prefill_persistent_k,
                ssm_state.h_state,
                x_ptr,
                b_ptr,
                c_ptr,
                dt_ptr,
                self.ssm.a_log.weight,
                self.ssm.d_param.weight,
                self.ssm.dt_bias.weight,
                y_out,
                1,
                n,
                self.num_heads as u32,
                self.head_dim as u32,
                self.state_size as u32,
                self.n_groups as u32,
                1e-9,
                1e9,
                self.d_xbc as u32,
                self.d_xbc as u32,
                self.in_proj_size as u32,
                self.d_inner as u32,
                stream,
            )?;
        } else {
            ops::mamba2_ssm_prefill(
                ctx.gpu,
                self.mamba2_ssm_prefill_k,
                ssm_state.h_state,
                x_ptr,
                b_ptr,
                c_ptr,
                dt_ptr,
                self.ssm.a_log.weight,
                self.ssm.d_param.weight,
                self.ssm.dt_bias.weight,
                y_out,
                1,
                n,
                self.num_heads as u32,
                self.head_dim as u32,
                self.state_size as u32,
                self.n_groups as u32,
                1e-9,
                1e9,
                self.d_xbc as u32,
                self.d_xbc as u32,
                self.in_proj_size as u32,
                self.d_inner as u32,
                stream,
            )?;
        }

        let gated_out = ctx.buffers.norm_output();
        let group_size = (self.d_inner / self.n_groups) as u32;
        ops::gated_rms_norm(
            ctx.gpu,
            self.gated_rms_norm_k,
            y_out,
            proj,
            &self.ssm.ssm_norm,
            gated_out,
            n,
            self.d_inner as u32,
            self.in_proj_size as u32,
            eps,
            group_size,
            stream,
        )?;

        let out = ctx.buffers.ssm_qkvz();
        self.prefill_out_proj(gated_out, out, n, h, fp8_a, w4a4, pd_fp8_ok, ctx, stream)?;

        ops::residual_add(
            ctx.gpu,
            self.residual_add_k,
            hidden,
            out,
            (num_tokens * h) as u32,
            stream,
        )?;

        Ok(())
    }
}
