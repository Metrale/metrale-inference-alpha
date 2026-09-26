// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-26: Two per-row phases of `decode_batched_inner`: the BA projection with the GDN
//! gates, and the gated RMS norm after the conv/GDN body.
//!
//! Owner: model-layers (Qwen3 SSM layer).
//! Invariants:
//! - The gates phase writes one [gate(nv) | beta(nv)] FP32 row per token into `gates_buf`,
//!   `gate_beta_stride` bytes apart.
//! - The norm phase writes `num_tokens` normed rows, `value_dim` BF16 apart, into
//!   `normed_out_buf`, or leaves them to the exact arm when `verify_exact_enabled()`.

use super::*;

impl Qwen3SsmLayer {
    /// 2026-09-26: BA projection and GDN gates for all `num_tokens` rows into `gates_buf`.
    pub(super) fn batched_ba_gates(
        &self,
        ctx: &ForwardContext,
        d: &BatchedDims,
        normed: DevicePtr,
        gates_buf: DevicePtr,
        gate_beta_stride: usize,
        ba_size: usize,
    ) -> Result<()> {
        let BatchedDims {
            num_tokens,
            h,
            bf16,
            fp32,
            nk,
            nv,
            vpg,
            stream,
            ..
        } = *d;
        if batched_ba_gates_enabled() {
            // 2026-09-25: One launch for all rows: `dense_gemm_ba_gates_prefill`, which the
            // SSM prefill and the multi-seq batched-recurrent decode also call. It is not
            // bitwise the per-token pair below: that pair rounds the BA projection to BF16
            // in `ssm_ba` before the gate transforms, and this kernel does not.
            ops::dense_gemm_ba_gates_prefill(
                ctx.gpu,
                self.ba_gates_prefill_k,
                self.ba_gates_prefill_hopper_k,
                normed,
                &self.ssm.in_proj_ba,
                self.ssm.a_log.weight,
                self.ssm.dt_bias.weight,
                gates_buf,
                num_tokens as u32,
                ba_size as u32,
                h as u32,
                h as u32,
                (nv * 2) as u32,
                nv as u32,
                vpg as u32,
                stream,
            )?;
        } else {
            for t in 0..(num_tokens as u32) {
                let normed_t = normed.offset(t as usize * h * bf16);
                let ba_out = ctx.buffers.ssm_ba().offset(t as usize * ba_size * bf16);
                ops::dense_gemv(
                    ctx.gpu,
                    self.dense_gemv_k,
                    normed_t,
                    &self.ssm.in_proj_ba,
                    ba_out,
                    ba_size as u32,
                    h as u32,
                    stream,
                )?;
                let gate_t = gates_buf.offset(t as usize * gate_beta_stride);
                let beta_t = gates_buf.offset(t as usize * gate_beta_stride + nv * fp32);
                ops::compute_gdn_gates(
                    ctx.gpu,
                    self.compute_gdn_gates_k,
                    ba_out,
                    self.ssm.a_log.weight,
                    self.ssm.dt_bias.weight,
                    gate_t,
                    beta_t,
                    1,
                    nv as u32,
                    nk as u32,
                    vpg as u32,
                    ba_size as u32,
                    stream,
                )?;
            }
        }
        Ok(())
    }

    /// 2026-09-26: Gated RMS norm of the `num_tokens` GDN rows at `gdn_out_buf` with the Z
    /// gate at `z_offset` of each deinterleaved row, into `normed_out_buf`.
    pub(super) fn batched_gated_norm(
        &self,
        ctx: &ForwardContext,
        d: &BatchedDims,
        gdn_out_buf: DevicePtr,
        deinterleaved: DevicePtr,
        normed_out_buf: DevicePtr,
        z_offset: usize,
    ) -> Result<()> {
        let BatchedDims {
            num_tokens,
            eps,
            nv,
            vd,
            value_dim,
            qkvz_size,
            bf16,
            stream,
            ..
        } = *d;
        if super::verify_exact_enabled() {
            // 2026-09-25: The exact arm (`decode_batched_conv_gdn_exact` and its batched
            // form) already wrote the normed rows to `normed_out_buf` at `value_dim`
            // stride, so no norm runs here.
        } else if num_tokens == 2 && self.fused_verify_k2_enabled() {
            ops::gdn_verify_fused_norm_k2(
                ctx.gpu,
                self.gdn_verify_fused_norm_k2_k,
                gdn_out_buf,
                deinterleaved,
                &self.ssm.norm,
                normed_out_buf,
                nv as u32,
                vd as u32,
                eps,
                qkvz_size as u32,
                z_offset as u32,
                value_dim as u32,
                stream,
            )?;
        } else if batched_norm_enabled() {
            // 2026-09-25: One launch for every (head, row) pair. `gated_rms_norm_prefill`
            // is `gated_rms_norm` with the row on blockIdx.y and explicit row strides; the
            // per-head arithmetic and block size are the same (kernels/gb10/common/
            // rms_norm.cu, ops launchers).
            ops::gated_rms_norm_prefill(
                ctx.gpu,
                self.gated_rms_norm_prefill_k,
                gdn_out_buf,
                deinterleaved.offset(z_offset * bf16),
                &self.ssm.norm,
                normed_out_buf,
                nv as u32,
                vd as u32,
                eps,
                num_tokens as u32,
                value_dim as u32,
                qkvz_size as u32,
                stream,
            )?;
        } else {
            for t in 0..(num_tokens as u32) {
                let gdn_t = gdn_out_buf.offset(t as usize * value_dim * bf16);
                let z_t = deinterleaved.offset(t as usize * qkvz_size * bf16 + z_offset * bf16);
                let normed_t = normed_out_buf.offset(t as usize * value_dim * bf16);
                ops::gated_rms_norm(
                    ctx.gpu,
                    self.gated_rms_norm_k,
                    gdn_t,
                    z_t,
                    &self.ssm.norm,
                    normed_t,
                    nv as u32,
                    vd as u32,
                    vd as u32,
                    eps,
                    vd as u32,
                    stream,
                )?;
            }
        }
        Ok(())
    }
}
