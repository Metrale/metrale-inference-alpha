// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-25: `Qwen3SsmLayer::ssm_forward`, the one-token GDN mixer. Callers:
//! `trait_decode.rs`, `trait_decode_hc.rs`, and the per-sequence loops of
//! `trait_decode_multi_seq.rs` and `trait_decode_multi_seq/hc.rs`.
//!
//! Order: QKVZ projection (plus `deinterleave_qkvz` unless `sequential_qkvz`),
//! BA projection with the GDN gates, conv1d update with SiLU and per-head L2
//! norm on Q and K, the GDN recurrence, gated RMS norm with the Z gate (fused
//! into the recurrence kernel when `gdn_fused_norm_enabled()` and the FP32
//! kernels resolved), `out_proj`, then `ssm_tp_all_reduce` (TP all-reduce,
//! then the `out_proj` LoRA delta).
//!
//! Owner: model-layers (qwen3_ssm).
//! Invariants:
//! - With `ssm_h_fp16_enabled()`, the recurrence launches only
//!   `gdn_f16_norm_k`, and only on a state with `h_is_f16`; any other case
//!   returns an error before the recurrence launches.

use super::*;

impl Qwen3SsmLayer {
    pub(super) fn ssm_forward(
        &self,
        normed: DevicePtr,
        state: &mut SsmLayerState,
        ctx: &ForwardContext,
        stream: u64,
        trace: bool,
    ) -> Result<DevicePtr> {
        let h = ctx.config.hidden_size as u32;
        let nk = ctx.config.linear_num_key_heads;
        let kd = ctx.config.linear_key_head_dim;
        let nv = ctx.config.linear_num_value_heads;
        let vd = ctx.config.linear_value_head_dim;
        let vpg = nv / nk;
        // 2026-09-25: No debug synchronize or dump during CUDA-graph capture,
        // where a stream synchronize fails with
        // `CUDA_ERROR_STREAM_CAPTURE_UNSUPPORTED`.
        let debug = tracing::enabled!(tracing::Level::DEBUG) && !ctx.graph_capture;
        let profile = ctx.profile;

        macro_rules! prof {
            ($label:expr, $body:expr) => {{
                if profile {
                    let t = std::time::Instant::now();
                    let r = $body;
                    ctx.gpu.synchronize(stream)?;
                    tracing::info!("    SSM {}: {:.0}μs", $label, t.elapsed().as_micros());
                    r
                } else {
                    $body
                }
            }};
        }

        let deinterleaved = ctx.buffers.ssm_deinterleaved();
        let qkvz_size = ctx.config.ssm_qkvz_size() as u32;
        prof!("qkvz", {
            if let Some(ref fp8) = self.qkvz_fp8w {
                // 2026-09-25: `w8a16_gemv` reads `[N/128, K/128]` FP32 block
                // scales (`kernels/gb10/common/w8a16_gemv.cu`); the assert panics
                // on a per-row or single-scale weight.
                fp8.scale_format.expect(
                    crate::weight_map::WeightQuantFormat::Fp8BlockScaled,
                    "ssm_forward::qkvz_fp8w → w8a16_gemv",
                );
                if self.sequential_qkvz {
                    ops::w8a16_gemv(
                        ctx.gpu,
                        self.w8a16_gemv_k,
                        normed,
                        fp8.weight,
                        fp8.row_scale,
                        deinterleaved,
                        qkvz_size,
                        h,
                        stream,
                    )
                } else {
                    let qkvz_out = ctx.buffers.ssm_qkvz();
                    ops::w8a16_gemv(
                        ctx.gpu,
                        self.w8a16_gemv_k,
                        normed,
                        fp8.weight,
                        fp8.row_scale,
                        qkvz_out,
                        qkvz_size,
                        h,
                        stream,
                    )?;
                    ops::deinterleave_qkvz(
                        ctx.gpu,
                        self.deinterleave_k,
                        qkvz_out,
                        deinterleaved,
                        1,
                        nk as u32,
                        kd as u32,
                        vpg as u32,
                        vd as u32,
                        stream,
                    )
                }
            } else if self.sequential_qkvz {
                if let Some(ref q2) = self.qkvz_q2 {
                    ops::q2_0_gemv_vec(ctx.gpu, self.q2_0_gemv_k, normed, q2, deinterleaved, stream)
                } else if let Some(ref nvfp4) = self.qkvz_nvfp4 {
                    ops::w4a16_decode_gemv(
                        ctx.gpu,
                        self.w4a16_gemv_k,
                        self.w4a16_gemv_sw_k,
                        ctx.levers.gemv_sw,
                        normed,
                        nvfp4,
                        deinterleaved,
                        qkvz_size,
                        h,
                        stream,
                    )
                } else {
                    ops::dense_gemv(
                        ctx.gpu,
                        self.dense_gemv_k,
                        normed,
                        &self.ssm.in_proj_qkvz,
                        deinterleaved,
                        qkvz_size,
                        h,
                        stream,
                    )
                }
            } else if let Some(ref nvfp4) = self.qkvz_nvfp4 {
                // 2026-09-25: `w4a16_gemv_qkvz` writes the deinterleaved layout
                // itself.
                ops::w4a16_gemv_qkvz(
                    ctx.gpu,
                    self.w4a16_gemv_qkvz_k,
                    normed,
                    nvfp4,
                    deinterleaved,
                    qkvz_size,
                    h,
                    nk as u32,
                    kd as u32,
                    vpg as u32,
                    vd as u32,
                    stream,
                )
            } else {
                let qkvz_out = ctx.buffers.ssm_qkvz();
                ops::dense_gemv(
                    ctx.gpu,
                    self.dense_gemv_k,
                    normed,
                    &self.ssm.in_proj_qkvz,
                    qkvz_out,
                    qkvz_size,
                    h,
                    stream,
                )?;
                ops::deinterleave_qkvz(
                    ctx.gpu,
                    self.deinterleave_k,
                    qkvz_out,
                    deinterleaved,
                    1,
                    nk as u32,
                    kd as u32,
                    vpg as u32,
                    vd as u32,
                    stream,
                )
            }
        })?;
        if trace {
            ctx.gpu.synchronize(stream).inspect_err(|_e| {
                tracing::error!("CRASH at qkvz_proj");
            })?;
        }
        if debug {
            ctx.gpu.synchronize(stream)?;
            Self::debug_bf16(ctx.gpu, "deinterleaved-Q", deinterleaved, 4);
        }

        // 2026-09-25: The deinterleaved buffer is `[Q | K | V | Z]` BF16, Q and K
        // `key_dim` wide, V and Z `value_dim` wide.
        let key_dim = nk * kd;
        let value_dim = nv * vd;
        let qkv_ptr = deinterleaved;
        let z_ptr = deinterleaved.offset((key_dim * 2 + value_dim) * 2);

        let ba_size = ctx.config.ssm_ba_size() as u32;
        let gates = ctx.buffers.ssm_gates();
        let beta_fp32 = gates.offset(nv * 4);
        prof!("ba_gates", {
            ops::dense_gemv_ba_gates(
                ctx.gpu,
                self.ba_gates_k,
                normed,
                &self.ssm.in_proj_ba,
                self.ssm.a_log.weight,
                self.ssm.dt_bias.weight,
                gates,
                beta_fp32,
                ba_size,
                h,
                vpg as u32,
                stream,
            )
        })?;
        if trace {
            ctx.gpu.synchronize(stream).inspect_err(|_e| {
                tracing::error!("CRASH at ba_gates");
            })?;
        }
        if debug {
            ctx.gpu.synchronize(stream)?;
            Self::debug_f32(ctx.gpu, "gate", gates, 4);
            Self::debug_f32(ctx.gpu, "beta", beta_fp32, 4);
        }

        // 2026-09-25: One launch: conv1d state update and SiLU on all `conv_dim`
        // channels, then per-head L2 norm on the first `qk_channels` (Q and K).
        let conv_dim = (key_dim * 2 + value_dim) as u32;
        let d_conv = ctx.config.linear_conv_kernel_dim as u32;
        let qk_channels = (key_dim * 2) as u32;
        let (conv_out, use_f32_conv) = if self.conv1d_l2norm_f32_k.0 != 0 {
            (ctx.buffers.ssm_conv_out_f32(), true)
        } else {
            (ctx.buffers.ssm_qkvz(), false)
        };
        if use_f32_conv {
            ops::conv1d_update_l2norm(
                ctx.gpu,
                self.conv1d_l2norm_f32_k,
                state.conv_state,
                qkv_ptr,
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
        } else {
            ops::conv1d_update_l2norm(
                ctx.gpu,
                self.conv1d_l2norm_k,
                state.conv_state,
                qkv_ptr,
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
        if trace {
            ctx.gpu.synchronize(stream).inspect_err(|_e| {
                tracing::error!("CRASH at conv1d_l2norm");
            })?;
        }
        if debug {
            ctx.gpu.synchronize(stream)?;
            Self::debug_bf16(ctx.gpu, "conv1d-l2norm-out", conv_out, 4);
        }

        let elem_size = if use_f32_conv { 4 } else { 2 };
        let q_conv = conv_out;
        let k_conv = conv_out.offset(key_dim * elem_size);
        let v_conv = conv_out.offset(key_dim * 2 * elem_size);

        let use_f32_gdn = self.gdn_f32_k.0 != 0 && self.gated_rms_norm_f32_k.0 != 0;
        let gdn_out = if use_f32_gdn {
            // 2026-09-25: The FP32 recurrence output goes after the conv output
            // in `ssm_conv_out_f32`, which is sized for `ssm_qkvz_size()` FP32
            // values per row, so the `value_dim` values past
            // `2 * key_dim + value_dim` fit.
            ctx.buffers
                .ssm_conv_out_f32()
                .offset((key_dim * 2 + value_dim) * 4)
        } else {
            ctx.buffers.attn_output()
        };
        let normed_out = ctx.buffers.ssm_qkvz();
        let gdn_kernel = if use_f32_gdn {
            self.gdn_f32_k
        } else {
            self.gdn_k
        };
        let fused_gdn_norm = use_f32_gdn
            && self.gdn_f32_norm_k.0 != 0
            && crate::layers::qwen3_ssm::gdn_fused_norm_enabled();
        // 2026-09-25: Of these arms only the fused-norm one has an FP16 h-state
        // kernel (`gdn_f16_norm_k`); the others would read the FP16 state as
        // FP32, so with the FP16 h-state on they error instead.
        let h_f16 = super::ssm_h_fp16_enabled();
        if h_f16 {
            super::ssm_h_fp16::require_h_f16(state)?;
            if !fused_gdn_norm {
                anyhow::bail!(
                    "METRALE_SSM_H_FP16: single-seq decode fell through to the FP32-only                      gated_delta_rule_decode arm (use_f32_gdn={use_f32_gdn},                      gdn_f32_norm={}). Set METRALE_GDN_FUSED_NORM=1.",
                    self.gdn_f32_norm_k.0
                );
            }
        }
        if fused_gdn_norm {
            ops::gdn_decode_f32_norm(
                ctx.gpu,
                if h_f16 {
                    self.gdn_f16_norm_k
                } else {
                    self.gdn_f32_norm_k
                },
                state.h_state,
                q_conv,
                k_conv,
                v_conv,
                gates,
                beta_fp32,
                z_ptr,
                self.ssm.norm.weight,
                normed_out,
                1,
                nk as u32,
                nv as u32,
                kd as u32,
                vd as u32,
                ctx.config.rms_norm_eps as f32,
                stream,
            )?;
            if trace {
                ctx.gpu.synchronize(stream).inspect_err(|_e| {
                    tracing::error!("CRASH at gdn_decode_f32_norm");
                })?;
            }
        } else {
            ops::gdn_decode(
                ctx.gpu,
                gdn_kernel,
                state.h_state,
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
            if trace {
                ctx.gpu.synchronize(stream).inspect_err(|_e| {
                    tracing::error!("CRASH at gdn_decode");
                })?;
            }

            let norm_kernel = if use_f32_gdn {
                self.gated_rms_norm_f32_k
            } else {
                self.gated_rms_norm_k
            };
            ops::gated_rms_norm(
                ctx.gpu,
                norm_kernel,
                gdn_out,
                z_ptr,
                &self.ssm.norm,
                normed_out,
                nv as u32,
                vd as u32,
                vd as u32,
                ctx.config.rms_norm_eps as f32,
                vd as u32,
                stream,
            )?;
            if trace {
                ctx.gpu.synchronize(stream).inspect_err(|_e| {
                    tracing::error!("CRASH at gated_rms_norm");
                })?;
            }
        }
        if trace && fused_gdn_norm {
            ctx.gpu.synchronize(stream).inspect_err(|_e| {
                tracing::error!("CRASH after fused gdn norm");
            })?;
        }
        if debug {
            ctx.gpu.synchronize(stream)?;
            Self::debug_bf16(ctx.gpu, "gated-norm-out", normed_out, 4);
        }

        let out = ctx.buffers.moe_output();
        if let Some(ref fp8) = self.out_proj_fp8w {
            fp8.scale_format.expect(
                crate::weight_map::WeightQuantFormat::Fp8BlockScaled,
                "ssm_forward::out_proj_fp8w → w8a16_gemv",
            );
            ops::w8a16_gemv(
                ctx.gpu,
                self.w8a16_gemv_k,
                normed_out,
                fp8.weight,
                fp8.row_scale,
                out,
                h,
                value_dim as u32,
                stream,
            )?;
        } else if let Some(ref dense_out) = self.out_proj_dense {
            ops::dense_gemv(
                ctx.gpu,
                self.dense_gemv_k,
                normed_out,
                dense_out,
                out,
                h,
                value_dim as u32,
                stream,
            )?;
        } else {
            ops::w4a16_decode_gemv(
                ctx.gpu,
                self.w4a16_gemv_k,
                self.w4a16_gemv_sw_k,
                ctx.levers.gemv_sw,
                normed_out,
                &self.ssm.out_proj,
                out,
                h,
                value_dim as u32,
                stream,
            )?;
        }
        if trace {
            ctx.gpu.synchronize(stream).inspect_err(|_e| {
                tracing::error!("CRASH at out_proj");
            })?;
        }
        if debug {
            ctx.gpu.synchronize(stream)?;
            Self::debug_bf16(ctx.gpu, "out-proj", out, 4);
        }

        // 2026-09-25: All-reduce `out` across TP ranks (no-op at TP 1), then add
        // the `out_proj` LoRA delta computed from `normed_out`, the activation
        // `out_proj` consumed. One token, so `num_tokens` is 1.
        self.ssm_tp_all_reduce(out, normed_out, 1, ctx, stream)?;

        Ok(out)
    }
}
