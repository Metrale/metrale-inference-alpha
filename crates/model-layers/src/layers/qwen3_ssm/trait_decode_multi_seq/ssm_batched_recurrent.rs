// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-25: The recurrence of the batched-projection GDN mixer
//! (`decode_ms_ssm_recurrent`): BA/gates, conv1d, GDN and gated norm.
//!
//! Owner: model-layers, GDN/SSM layer (`qwen3_ssm`).
//! Invariants: none beyond the types.

use super::super::*;

mod per_seq;

/// 2026-09-25: Batched-recurrence engagements. `FALLBACK` counts the
/// per-sequence fallbacks taken because the pool slots were not contiguous in
/// slice order.
static BATCHED_OK: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
static FALLBACK: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
static FALLBACK_ONCE: std::sync::Once = std::sync::Once::new();

impl Qwen3SsmLayer {
    /// 2026-09-25: Recurrence of the batched-projection GDN mixer, reading
    /// `deinterleaved` and writing the gated-norm rows to `normed_out_base`.
    ///
    /// Batched launches when `ssm_batched_recurrent_enabled()`, the strided GDN
    /// kernel is loaded, `n > 1`, and the sequences' pool slots are contiguous
    /// in slice order; otherwise a per-sequence loop.
    ///
    /// `detail_t0` / `detail_parts` carry the caller's profiling state, so the
    /// `ssm_detail_profile` summary (METRALE_SSM_DETAIL_PROFILE=1) spans the
    /// whole mixer.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn decode_ms_ssm_recurrent<'a, 'b: 'a>(
        &self,
        states: &'a mut [&'b mut (dyn LayerState + 'static)],
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
        detail_parts: &mut Vec<(&'static str, u128)>,
        detail_t0: &mut Option<std::time::Instant>,
        ctx: &ForwardContext,
        stream: u64,
    ) -> Result<()> {
        let mut rec_ba_us = 0u128;
        let mut rec_conv_us = 0u128;
        let mut rec_gdn_us = 0u128;
        let mut rec_norm_us = 0u128;
        macro_rules! detail_step {
            ($label:expr) => {
                if let Some(t0) = detail_t0.take() {
                    ctx.gpu.synchronize(stream).ok();
                    detail_parts.push(($label, t0.elapsed().as_micros()));
                    *detail_t0 = Some(std::time::Instant::now());
                }
            };
        }

        // 2026-09-25: FP16 h-state: `ssm_h_to_f16_dispatch` (model-engine)
        // converts the states at decode entry. Here every state must already be
        // FP16, and both head dims must be 128.
        let h_f16 = super::super::ssm_h_fp16_enabled();
        if h_f16 {
            for st in states.iter_mut().take(n) {
                let ssm = st
                    .as_any_mut()
                    .downcast_mut::<SsmLayerState>()
                    .ok_or_else(|| anyhow::anyhow!("Expected SsmLayerState for FP16 h-state"))?;
                super::super::ssm_h_fp16::require_h_f16(ssm)?;
            }
            if kd != 128 || vd != 128 {
                anyhow::bail!(
                    "METRALE_SSM_H_FP16 needs linear head dims 128/128 (the FP16 twins size their                      smem for k_dim==128); this model is {kd}/{vd}"
                );
            }
        }

        let batched_recurrent = if crate::layers::qwen3_ssm::ssm_batched_recurrent_enabled()
            && self.gdn_f32_strided_k.0 != 0
            && n > 1
        {
            let mut h_base = DevicePtr::NULL;
            let mut conv_base = DevicePtr::NULL;
            let mut contiguous = true;
            for i in 0..n {
                let ssm_state = states[i]
                    .as_any_mut()
                    .downcast_mut::<SsmLayerState>()
                    .ok_or_else(|| anyhow::anyhow!("Expected SsmLayerState for seq {i}"))?;
                if i == 0 {
                    h_base = ssm_state.h_state;
                    conv_base = ssm_state.conv_state;
                } else {
                    contiguous &=
                        ssm_state.h_state.0 == h_base.0 + (i * self.h_slot_stride_bytes()) as u64;
                    contiguous &=
                        ssm_state.conv_state.0 == conv_base.0 + (i * self.conv_state_bytes) as u64;
                }
            }
            if contiguous {
                BATCHED_OK.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                Some((h_base, conv_base))
            } else {
                // 2026-09-25: Log the first fallback with the first slot out of
                // place, and number every fallback at debug level.
                let n_fb = FALLBACK.fetch_add(1, std::sync::atomic::Ordering::Relaxed) + 1;
                FALLBACK_ONCE.call_once(|| {
                    let mut broke_at = usize::MAX;
                    let mut delta = 0i64;
                    for i in 1..n {
                        if let Some(st) = states[i].as_any_mut().downcast_mut::<SsmLayerState>() {
                            let want = h_base.0 + (i * self.h_slot_stride_bytes()) as u64;
                            if st.h_state.0 != want {
                                broke_at = i;
                                delta = st.h_state.0 as i64 - want as i64;
                                break;
                            }
                        }
                    }
                    tracing::info!(
                        "SSM batched recurrent DECLINED (n={n}): pool slots are not contiguous in \
                         slice order — seq {broke_at} sits {} slot(s) from where the batch axis \
                         expects it. The per-seq loop costs ~28% more on this block. Slots \
                         fragment as sequences finish, so this is expected to recur; count is \
                         logged at debug on every occurrence.",
                        delta / self.h_slot_stride_bytes().max(1) as i64
                    );
                });
                tracing::debug!("SSM batched recurrent fallback #{n_fb} (n={n})");
                None
            }
        } else {
            None
        };

        if let Some((h_state_base, conv_state_base)) = batched_recurrent {
            let gates = ctx.buffers.ssm_gates();
            let beta_fp32 = gates.offset(nv * 4);
            let gate_stride = (nv * 2) as u32;
            ops::dense_gemm_ba_gates_prefill(
                ctx.gpu,
                self.ba_gates_prefill_k,
                self.ba_gates_prefill_hopper_k,
                normed_base,
                &self.ssm.in_proj_ba,
                self.ssm.a_log.weight,
                self.ssm.dt_bias.weight,
                gates,
                n as u32,
                ba_size,
                h as u32,
                h as u32,
                gate_stride,
                nv as u32,
                vpg as u32,
                stream,
            )?;
            detail_step!("recurrent_batched_ba");

            let conv_out = ctx.buffers.ssm_conv_out_f32();
            // 2026-09-25: The conv reads `deinterleaved` rows `qkvz_size` apart
            // and writes FP32 rows `conv_dim` apart. The plain kernel uses
            // `dim` (= conv_dim) for both strides, so a batch launch of it would
            // read row b >= 1 from the wrong offset. The strided kernel takes
            // both strides and runs the batch in one launch; without it (handle
            // 0) each row runs alone with pre-offset pointers.
            if self.conv1d_l2norm_f32_strided_k.0 != 0 {
                ops::conv1d_update_l2norm_strided(
                    ctx.gpu,
                    self.conv1d_l2norm_f32_strided_k,
                    conv_state_base,
                    deinterleaved,
                    &self.ssm.conv1d,
                    conv_out,
                    conv_dim,
                    d_conv,
                    n as u32,
                    qk_channels,
                    kd as u32,
                    1e-6,
                    qkvz_size as u32,
                    conv_dim,
                    stream,
                )?;
            } else {
                for i in 0..n {
                    ops::conv1d_update_l2norm(
                        ctx.gpu,
                        self.conv1d_l2norm_f32_k,
                        conv_state_base.offset(i * self.conv_state_bytes),
                        deinterleaved.offset(i * qkvz_size * bf16),
                        &self.ssm.conv1d,
                        conv_out.offset(i * conv_dim as usize * 4),
                        conv_dim,
                        d_conv,
                        1,
                        qk_channels,
                        kd as u32,
                        1e-6,
                        stream,
                    )?;
                }
            }
            detail_step!("recurrent_batched_conv");

            if self.gdn_f32_strided_norm_k.0 != 0
                && crate::layers::qwen3_ssm::gdn_fused_norm_enabled()
            {
                let z_base = deinterleaved.offset((key_dim * 2 + value_dim) * bf16);
                fn gdn_half_reg_enabled() -> bool {
                    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
                    *ON.get_or_init(|| {
                        std::env::var("METRALE_NO_GDN_HALF_REG").ok().as_deref() != Some("1")
                    })
                }
                // 2026-09-25: The SMEM-staged twin keeps the un-retained H
                // columns in shared memory instead of re-reading them; it must
                // match the half-register kernel bit for bit
                // (`gdn_strided_norm_microtest` compares output and H state).
                // Opt-in: METRALE_GDN_SMEM_STAGE is a presence check, so any
                // value, `0` included, turns it on.
                fn gdn_smem_stage_enabled() -> bool {
                    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
                    *ON.get_or_init(|| std::env::var("METRALE_GDN_SMEM_STAGE").is_ok())
                }
                let gdn_norm_k = if kd == 128
                    && vd == 128
                    && self.gdn_f32_strided_norm_smem_k.0 != 0
                    && gdn_half_reg_enabled()
                    && gdn_smem_stage_enabled()
                {
                    // 2026-09-25: The kernel's staging buffer is sized for
                    // k_dim == 128, which the kd/vd check above establishes.
                    self.gdn_f32_strided_norm_smem_k
                } else if kd == 128
                    && vd == 128
                    && self.gdn_f32_strided_norm_half_k.0 != 0
                    && gdn_half_reg_enabled()
                {
                    self.gdn_f32_strided_norm_half_k
                } else {
                    self.gdn_f32_strided_norm_k
                };
                if h_f16 {
                    // 2026-09-25: Under the FP16 h-state this arm always runs
                    // the f16 kernel. Its stride is the pool slot pitch in
                    // `__half` elements, from `h_slot_stride_bytes`: an
                    // FP32-sized pool spaces slots `h_state_bytes` apart, twice
                    // the FP16 footprint; an f16-sized pool spaces them
                    // `h_state_bytes / 2` apart.
                    ops::gdn_decode_f16_strided_norm(
                        ctx.gpu,
                        self.gdn_f16_strided_norm_half_k,
                        h_state_base,
                        conv_out,
                        conv_out.offset(key_dim * 4),
                        conv_out.offset(key_dim * 2 * 4),
                        gates,
                        beta_fp32,
                        z_base,
                        self.ssm.norm.weight,
                        normed_out_base,
                        n as u32,
                        nk as u32,
                        nv as u32,
                        kd as u32,
                        vd as u32,
                        conv_dim,
                        conv_dim,
                        gate_stride,
                        qkvz_size as u32,
                        value_dim as u32,
                        (self.h_slot_stride_bytes() / 2) as u64,
                        eps,
                        stream,
                    )?;
                } else {
                    ops::gdn_decode_f32_strided_norm(
                        ctx.gpu,
                        gdn_norm_k,
                        h_state_base,
                        conv_out,
                        conv_out.offset(key_dim * 4),
                        conv_out.offset(key_dim * 2 * 4),
                        gates,
                        beta_fp32,
                        z_base,
                        self.ssm.norm.weight,
                        normed_out_base,
                        n as u32,
                        nk as u32,
                        nv as u32,
                        kd as u32,
                        vd as u32,
                        conv_dim,
                        conv_dim,
                        gate_stride,
                        qkvz_size as u32,
                        value_dim as u32,
                        eps,
                        stream,
                    )?;
                }
                detail_step!("recurrent_batched_gdn_norm");
            } else {
                if h_f16 {
                    anyhow::bail!(
                        "METRALE_SSM_H_FP16: the batched decode arm selected the FP32-only \
                         gated_delta_rule_decode_f32_strided (METRALE_GDN_FUSED_NORM is not 1)"
                    );
                }
                let gdn_out = conv_out.offset(n * conv_dim as usize * 4);
                ops::gdn_decode_f32_strided(
                    ctx.gpu,
                    self.gdn_f32_strided_k,
                    h_state_base,
                    conv_out,
                    conv_out.offset(key_dim * 4),
                    conv_out.offset(key_dim * 2 * 4),
                    gates,
                    beta_fp32,
                    gdn_out,
                    n as u32,
                    nk as u32,
                    nv as u32,
                    kd as u32,
                    vd as u32,
                    conv_dim,
                    conv_dim,
                    gate_stride,
                    value_dim as u32,
                    stream,
                )?;
                detail_step!("recurrent_batched_gdn");

                // 2026-09-25: One launch for all `n` rows when the strided
                // gated-norm kernel is loaded, else one launch per row.
                let z_base = deinterleaved.offset((key_dim * 2 + value_dim) * bf16);
                if self.gated_rms_norm_f32_strided_k.0 != 0 {
                    ops::gated_rms_norm_strided(
                        ctx.gpu,
                        self.gated_rms_norm_f32_strided_k,
                        gdn_out,
                        z_base,
                        &self.ssm.norm,
                        normed_out_base,
                        nv as u32,
                        n as u32,
                        vd as u32,
                        vd as u32,
                        eps,
                        vd as u32,
                        value_dim as u32,
                        qkvz_size as u32,
                        value_dim as u32,
                        stream,
                    )?;
                } else {
                    for i in 0..n {
                        let z_i = z_base.offset(i * qkvz_size * bf16);
                        let gdn_out_i = gdn_out.offset(i * value_dim * 4);
                        let normed_out_i = normed_out_base.offset(i * value_dim * bf16);
                        ops::gated_rms_norm(
                            ctx.gpu,
                            self.gated_rms_norm_f32_k,
                            gdn_out_i,
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
                    }
                }
                detail_step!("recurrent_batched_norm");
            }
        } else {
            self.decode_ms_ssm_recurrent_per_seq(
                states,
                n,
                normed_base,
                deinterleaved,
                normed_out_base,
                qkvz_size,
                key_dim,
                value_dim,
                conv_dim,
                qk_channels,
                d_conv,
                nk,
                nv,
                kd,
                vd,
                vpg,
                ba_size,
                h,
                bf16,
                eps,
                detail_profile,
                h_f16,
                &mut rec_ba_us,
                &mut rec_conv_us,
                &mut rec_gdn_us,
                &mut rec_norm_us,
                ctx,
                stream,
            )?;
            if detail_profile {
                detail_parts.push(("recurrent_ba", rec_ba_us));
                detail_parts.push(("recurrent_conv", rec_conv_us));
                detail_parts.push(("recurrent_gdn", rec_gdn_us));
                if rec_norm_us > 0 {
                    detail_parts.push(("recurrent_norm", rec_norm_us));
                }
                *detail_t0 = Some(std::time::Instant::now());
            }
        }

        Ok(())
    }
}
