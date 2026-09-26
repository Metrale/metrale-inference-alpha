// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-26: The per-sequence arm of `decode_ms_ssm_recurrent`: BA gates, conv1d, GDN and
//! gated norm for one sequence at a time, taken when the batched recurrence is not.
//!
//! Owner: model-layers, GDN/SSM layer (`qwen3_ssm`).
//! Invariants: iteration `i` reads `normed_base` row `i` and `deinterleaved` row `i`, writes
//! `normed_out_base` row `i`, and updates only `states[i]`'s h and conv state.

use super::super::super::*;

impl Qwen3SsmLayer {
    /// 2026-09-26: Run the recurrence for `n` sequences, one launch chain each, adding the
    /// per-phase times to `rec_*_us` when `detail_profile` is set.
    pub(super) fn decode_ms_ssm_recurrent_per_seq(
        &self,
        states: &mut [&mut (dyn LayerState + 'static)],
        n: usize,
        normed_base: DevicePtr,
        deinterleaved: DevicePtr,
        normed_out_base: DevicePtr,
        qkvz_size: usize,
        key_dim: usize,
        value_dim: usize,
        conv_dim: u32,
        qk_channels: u32,
        d_conv: u32,
        nk: usize,
        nv: usize,
        kd: usize,
        vd: usize,
        vpg: usize,
        ba_size: u32,
        h: usize,
        bf16: usize,
        eps: f32,
        detail_profile: bool,
        h_f16: bool,
        rec_ba_us: &mut u128,
        rec_conv_us: &mut u128,
        rec_gdn_us: &mut u128,
        rec_norm_us: &mut u128,
        ctx: &ForwardContext,
        stream: u64,
    ) -> Result<()> {
        for i in 0..n {
            let normed_i = normed_base.offset(i * h * bf16);
            let deint_i = deinterleaved.offset(i * qkvz_size * bf16);
            let z_i = deint_i.offset((key_dim * 2 + value_dim) * bf16);
            let normed_out_i = normed_out_base.offset(i * value_dim * bf16);

            let ssm_state = states[i]
                .as_any_mut()
                .downcast_mut::<SsmLayerState>()
                .ok_or_else(|| anyhow::anyhow!("Expected SsmLayerState for seq {i}"))?;

            let gates = ctx.buffers.ssm_gates();
            let beta_fp32 = gates.offset(nv * 4);
            let sub_t0 = if detail_profile {
                ctx.gpu.synchronize(stream).ok();
                Some(std::time::Instant::now())
            } else {
                None
            };
            ops::dense_gemv_ba_gates(
                ctx.gpu,
                self.ba_gates_k,
                normed_i,
                &self.ssm.in_proj_ba,
                self.ssm.a_log.weight,
                self.ssm.dt_bias.weight,
                gates,
                beta_fp32,
                ba_size,
                h as u32,
                vpg as u32,
                stream,
            )?;
            if let Some(t0) = sub_t0 {
                ctx.gpu.synchronize(stream).ok();
                *rec_ba_us += t0.elapsed().as_micros();
            }

            let conv_out = ctx.buffers.ssm_conv_out_f32();
            // 2026-09-25: METRALE_GDN_FUSED_CONV: one kernel runs conv, GDN
            // and gated norm, admitted only when nv == 2 * nk and both head
            // dims are 128. The standalone conv is then skipped.
            let use_fused_conv = self.gdn_f32_conv_norm_k.0 != 0
                && nv == nk * 2
                && kd == 128
                && vd == 128
                && crate::layers::ops::ModelLevers::get().gdn_fused_conv;
            let sub_t0 = if detail_profile {
                Some(std::time::Instant::now())
            } else {
                None
            };
            if !use_fused_conv {
                ops::conv1d_update_l2norm(
                    ctx.gpu,
                    self.conv1d_l2norm_f32_k,
                    ssm_state.conv_state,
                    deint_i,
                    &self.ssm.conv1d,
                    conv_out,
                    conv_dim,
                    d_conv,
                    1,
                    qk_channels,
                    kd as u32,
                    1e-6,
                    stream,
                )?;
            }
            if let Some(t0) = sub_t0 {
                ctx.gpu.synchronize(stream).ok();
                *rec_conv_us += t0.elapsed().as_micros();
            }

            let gdn_out = conv_out.offset((key_dim * 2 + value_dim) * 4);
            let q_conv = conv_out;
            let k_conv = conv_out.offset(key_dim * 4);
            let v_conv = conv_out.offset(key_dim * 2 * 4);
            let sub_t0 = if detail_profile {
                Some(std::time::Instant::now())
            } else {
                None
            };
            if h_f16 && (use_fused_conv || self.gdn_f32_norm_k.0 == 0) {
                anyhow::bail!(
                    "METRALE_SSM_H_FP16: the per-seq decode arm selected an FP32-only kernel                          (fused_conv={use_fused_conv}, gdn_f32_norm={}). That would read the FP16                          pool as FP32. Unset METRALE_GDN_FUSED_CONV and set METRALE_GDN_FUSED_NORM=1.",
                    self.gdn_f32_norm_k.0
                );
            }
            if use_fused_conv {
                ops::gdn_decode_f32_conv_norm(
                    ctx.gpu,
                    self.gdn_f32_conv_norm_k,
                    ssm_state.h_state,
                    ssm_state.conv_state,
                    deint_i,
                    self.ssm.conv1d.weight,
                    gates,
                    beta_fp32,
                    z_i,
                    self.ssm.norm.weight,
                    normed_out_i,
                    1,
                    nk as u32,
                    nv as u32,
                    kd as u32,
                    vd as u32,
                    conv_dim,
                    d_conv,
                    1e-6,
                    eps,
                    stream,
                )?;
                if let Some(t0) = sub_t0 {
                    ctx.gpu.synchronize(stream).ok();
                    *rec_gdn_us += t0.elapsed().as_micros();
                }
            } else if self.gdn_f32_norm_k.0 != 0
                && crate::layers::qwen3_ssm::gdn_fused_norm_enabled()
            {
                ops::gdn_decode_f32_norm(
                    ctx.gpu,
                    if h_f16 {
                        self.gdn_f16_norm_k
                    } else {
                        self.gdn_f32_norm_k
                    },
                    ssm_state.h_state,
                    q_conv,
                    k_conv,
                    v_conv,
                    gates,
                    beta_fp32,
                    z_i,
                    self.ssm.norm.weight,
                    normed_out_i,
                    1,
                    nk as u32,
                    nv as u32,
                    kd as u32,
                    vd as u32,
                    eps,
                    stream,
                )?;
                if let Some(t0) = sub_t0 {
                    ctx.gpu.synchronize(stream).ok();
                    *rec_gdn_us += t0.elapsed().as_micros();
                }
            } else {
                ops::gdn_decode(
                    ctx.gpu,
                    self.gdn_f32_k,
                    ssm_state.h_state,
                    q_conv,
                    k_conv,
                    v_conv,
                    gates,
                    beta_fp32,
                    gdn_out,
                    1,
                    nk as u32,
                    nv as u32,
                    kd as u32,
                    vd as u32,
                    stream,
                )?;
                if let Some(t0) = sub_t0 {
                    ctx.gpu.synchronize(stream).ok();
                    *rec_gdn_us += t0.elapsed().as_micros();
                }

                let sub_t0 = if detail_profile {
                    Some(std::time::Instant::now())
                } else {
                    None
                };
                ops::gated_rms_norm(
                    ctx.gpu,
                    self.gated_rms_norm_f32_k,
                    gdn_out,
                    z_i,
                    &self.ssm.norm,
                    normed_out_i,
                    nv as u32,
                    vd as u32,
                    vd as u32,
                    eps,
                    vd as u32,
                    stream,
                )?;
                if let Some(t0) = sub_t0 {
                    ctx.gpu.synchronize(stream).ok();
                    *rec_norm_us += t0.elapsed().as_micros();
                }
            }
        }
        Ok(())
    }
}
