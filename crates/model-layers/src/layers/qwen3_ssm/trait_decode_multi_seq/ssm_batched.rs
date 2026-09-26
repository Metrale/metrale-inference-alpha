// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-25: Batched-projection GDN mixer for multi-sequence decode
//! (`try_decode_multi_seq_ssm_batched`).
//!
//! Owner: model-layers, GDN/SSM layer (`qwen3_ssm`).
//! Invariants:
//! - When the layer is not eligible it returns `Ok(false)` and has launched
//!   nothing.

use super::super::*;

/// 2026-09-25: Smallest batch whose NVFP4 mixer projections run as a tile GEMM
/// on the transposed twins (`qkvz_nvfp4_t`, `out_proj_nvfp4_t`) instead of the
/// batched GEMV. `METRALE_SSM_TC_PROJ`: unset or `1` gives 9, `0` disables,
/// a number >= 2 sets the threshold, and any other value disables. Read once
/// per process.
///
/// Measured 2026-07-27 on GB10, Qwen3.6-27B NVFP4, decode tok/s, 2 reps per
/// cell: GEMV gave C=8 57.8, C=16 79.6; threshold 9 gave 57.6, 86.9;
/// threshold 5 gave 54.9, 86.4.
pub(super) fn ssm_tc_proj_min_n() -> Option<usize> {
    static N: std::sync::OnceLock<Option<usize>> = std::sync::OnceLock::new();
    *N.get_or_init(
        || match std::env::var("METRALE_SSM_TC_PROJ").ok().as_deref() {
            None => Some(9),
            Some("0") => None,
            Some("1") => Some(9),
            Some(v) => v.parse::<usize>().ok().filter(|&x| x >= 2),
        },
    )
}

impl Qwen3SsmLayer {
    /// 2026-09-25: Batched-projection GDN mixer for `n` concurrent decode rows.
    ///
    /// Returns `Ok(false)` unless `n >= 2`, QKVZ is sequential, the FP32 conv
    /// and GDN kernels are loaded, a batched QKVZ path exists for this weight
    /// format and row count, an out_proj weight exists, and QKVZ is not packed
    /// Q2_0; the caller then runs its per-sequence loop. Otherwise QKVZ and
    /// out_proj run as one GEMM each over all `n` rows, and the recurrence runs
    /// in `decode_ms_ssm_recurrent`.
    #[allow(clippy::too_many_arguments)]
    /// 2026-09-25: With `hc`, the input rows arrive mixed in `norm_output`
    /// (from `hc_pre`), both norm steps are skipped, the out_proj rows stay in
    /// `moe_output` for the caller's `hc_post`, and `hidden`/`residual` are not
    /// read.
    pub(super) fn try_decode_multi_seq_ssm_batched<'a, 'b: 'a>(
        &self,
        hidden: DevicePtr,
        residual: DevicePtr,
        n: usize,
        states: &'a mut [&'b mut (dyn LayerState + 'static)],
        hc: bool,
        ctx: &ForwardContext,
        stream: u64,
    ) -> Result<bool> {
        let use_f32_conv = self.conv1d_l2norm_f32_k.0 != 0;
        let use_f32_gdn = self.gdn_f32_k.0 != 0 && self.gated_rms_norm_f32_k.0 != 0;
        // 2026-09-25: The NVFP4 batched GEMV serves at most 16 rows; the tile
        // GEMM on the transposed twins lifts that cap when both twins exist
        // and `ssm_tc_proj_min_n` admits `n`.
        let tc_wide_ok = self.qkvz_nvfp4_t.is_some()
            && self.out_proj_nvfp4_t.is_some()
            && ssm_tc_proj_min_n().is_some_and(|min| n >= min);
        // 2026-09-25: Without NVFP4 QKVZ, an FP8 build (`qkvz_fp8w`) needs
        // `w8a16_gemm_k`, the last arm `ms_batched_qkvz` can fall to. A BF16
        // build passes on `dense_gemm_k`, which `init.rs` always loads; its
        // projection runs on cuBLASLt.
        let qkvz_ok = (self.qkvz_nvfp4.is_none()
            && ((self.qkvz_fp8w.is_some() && self.w8a16_gemm_k.0 != 0)
                || (self.qkvz_fp8w.is_none() && self.dense_gemm_k.0 != 0)))
            || (self.qkvz_nvfp4.is_some()
                && ((self.w4a16_batchm.has_base() && n <= 16) || tc_wide_ok));
        let out_ok = self.out_proj_fp8w.is_some()
            || self.out_proj_dense.is_some()
            || self.qkvz_nvfp4.is_some();
        // 2026-09-25: Packed Q2_0 QKVZ (`qkvz_q2`) has no batched GEMM here;
        // the per-sequence `ssm_forward` serves it with `q2_0_gemv_vec`.
        if n < 2
            || !self.sequential_qkvz
            || !use_f32_conv
            || !use_f32_gdn
            || !qkvz_ok
            || !out_ok
            || self.qkvz_q2.is_some()
        {
            // 2026-09-25: Log the failing conditions once per process: the
            // per-sequence fallback reads the QKVZ and out_proj weights once per
            // sequence.
            if n >= 2 {
                static WHY: std::sync::Once = std::sync::Once::new();
                let (sq, fc, fg, qk, op) = (
                    self.sequential_qkvz,
                    use_f32_conv,
                    use_f32_gdn,
                    qkvz_ok,
                    out_ok,
                );
                let nvfp4 = self.qkvz_nvfp4.is_some();
                let b4 = self.w4a16_batchm.has_base();
                let tct = self.qkvz_nvfp4_t.is_some() && self.out_proj_nvfp4_t.is_some();
                WHY.call_once(|| {
                    tracing::info!(
                        "SSM batched projections DECLINED (n={n}): sequential_qkvz={sq} \
                         f32_conv={fc} f32_gdn={fg} qkvz_ok={qk} out_ok={op} \
                         [qkvz_nvfp4={nvfp4} w4a16_gemv_batch4={b4} tc_twins={tct} \
                         tc_wide_ok={tc_wide_ok}] — falling back to the \
                         per-seq loop, which re-reads QKVZ/out_proj weights n times"
                    );
                });
            }
            return Ok(false);
        }
        {
            static ON: std::sync::Once = std::sync::Once::new();
            ON.call_once(|| {
                tracing::info!("SSM batched projections ACTIVE — QKVZ/out_proj read once per step");
            });
        }

        let h = ctx.config.hidden_size;
        let bf16 = 2usize;
        let eps = ctx.config.rms_norm_eps as f32;
        let nk = ctx.config.linear_num_key_heads;
        let kd = ctx.config.linear_key_head_dim;
        let nv = ctx.config.linear_num_value_heads;
        let vd = ctx.config.linear_value_head_dim;
        let vpg = nv / nk;
        let key_dim = nk * kd;
        let value_dim = nv * vd;
        let conv_dim = (key_dim * 2 + value_dim) as u32;
        let qk_channels = (key_dim * 2) as u32;
        let d_conv = ctx.config.linear_conv_kernel_dim as u32;
        let qkvz_size = ctx.config.ssm_qkvz_size();
        let ba_size = ctx.config.ssm_ba_size() as u32;

        let normed_base = ctx.buffers.norm_output();
        let deinterleaved = ctx.buffers.ssm_deinterleaved();
        // 2026-09-25: The gated-norm output `[n, value_dim]` BF16 goes in the
        // `ssm_qkvz` scratch, which is free here: sequential QKVZ projects into
        // `deinterleaved`, and the FP32 conv writes `ssm_conv_out_f32`.
        let normed_out_base = ctx.buffers.ssm_qkvz();
        let ssm_out_base = ctx.buffers.moe_output();
        let detail_profile =
            crate::layers::ops::ModelLevers::get().ssm_detail_profile && !ctx.graph_capture;
        let mut detail_parts: Vec<(&'static str, u128)> = Vec::new();
        let mut detail_t0 = if detail_profile {
            ctx.gpu.synchronize(stream).ok();
            Some(std::time::Instant::now())
        } else {
            None
        };
        macro_rules! detail_step {
            ($label:expr) => {
                if let Some(t0) = detail_t0.take() {
                    ctx.gpu.synchronize(stream).ok();
                    detail_parts.push(($label, t0.elapsed().as_micros()));
                    detail_t0 = Some(std::time::Instant::now());
                }
            };
            ($label:expr, final) => {
                if let Some(t0) = detail_t0.take() {
                    ctx.gpu.synchronize(stream).ok();
                    detail_parts.push(($label, t0.elapsed().as_micros()));
                }
            };
        }

        if !hc {
            ops::rms_norm_residual(
                ctx.gpu,
                self.rms_norm_residual_k,
                hidden,
                &self.input_norm,
                normed_base,
                residual,
                n as u32,
                h as u32,
                eps,
                stream,
            )?;
        }
        detail_step!("input_norm");

        // 2026-09-25: `ssm_batched_proj.rs` picks the GEMM for this row count.
        let tier = self.batched_proj_tier(n);
        self.ms_batched_qkvz(
            ctx,
            &tier,
            n,
            normed_base,
            deinterleaved,
            qkvz_size,
            h,
            stream,
        )?;
        detail_step!("qkvz");

        // 2026-09-25: Batched when `ssm_batched_recurrent_enabled()` and the
        // pool slots are contiguous, else per sequence (see the callee).
        self.decode_ms_ssm_recurrent(
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
            &mut detail_parts,
            &mut detail_t0,
            ctx,
            stream,
        )?;
        detail_step!("recurrent_total_tail");

        self.ms_batched_out_proj(
            ctx,
            &tier,
            n,
            normed_out_base,
            ssm_out_base,
            h,
            value_dim,
            stream,
        )?;
        detail_step!("out_proj");

        // 2026-09-25: Sum the out_proj partials across TP ranks and apply the
        // out_proj LoRA delta, before the residual add.
        self.ssm_tp_all_reduce(ssm_out_base, normed_out_base, n, ctx, stream)?;

        if !hc {
            ops::residual_add_rms_norm(
                ctx.gpu,
                self.residual_add_rms_norm_k,
                hidden,
                ssm_out_base,
                &self.post_attn_norm,
                normed_base,
                residual,
                n as u32,
                h as u32,
                eps,
                stream,
            )?;
        }
        detail_step!("post_norm", final);
        if detail_profile {
            let summary = detail_parts
                .iter()
                .map(|(label, us)| format!("{label}={us}us"))
                .collect::<Vec<_>>()
                .join(" ");
            tracing::info!("METRALE_SSM_DETAIL n={n}: {summary}");
        }

        Ok(true)
    }
}
