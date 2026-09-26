// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-25: The GDN block body, steps 2-10 of the SSM prefill
//! (`prefill_block`).
//!
//! Owner: model-layers, GDN/SSM layer (`qwen3_ssm`).
//! Invariants:
//! - The block takes neither `hidden` nor `residual`; its output is
//!   `ctx.buffers.moe_output()`.
//!
//! Two entry paths call it, and neither wraps the other. `prefill_inner`
//! folds its residual bookkeeping into its norms:
//!
//! ```text
//! rms_norm_residual(hidden, input_norm) -> normed, residual      # step 1
//!   ... THIS FILE ...                   -> out_proj_buf          # steps 2-10
//! residual_add_rms_norm(hidden, out_proj_buf, post_attn_norm)    # step 11
//! ffn.forward_prefill(norm_output)                               # step 12
//! residual_add(hidden, moe_output)                               # step 13
//! ```
//!
//! Under an mHC highway (Qwen3.8-Flash-Next) the highway is the residual, so
//! `prefill_inner_hc` replaces steps 1, 11 and 13 and calls this body
//! between its `hc_pre` and `hc_post`.

use super::*;

impl Qwen3SsmLayer {
    /// 2026-09-25: Steps 2-10: QKVZ projection, BA gates, conv1d, the Q/K L2
    /// norm, the delta-rule recurrence, the gated norm, `out_proj`, and the TP
    /// reduce. Returns the buffer holding `out_proj`'s output.
    ///
    /// `ssm_layer_idx` is passed in rather than re-fetched: it comes from a
    /// global call counter that is bumped once per layer call, and both entry
    /// paths bump it before calling here.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn prefill_block(
        &self,
        normed: DevicePtr,
        num_tokens: usize,
        state: &mut dyn LayerState,
        ssm_layer_idx: usize,
        ctx: &ForwardContext,
        stream: u64,
    ) -> Result<DevicePtr> {
        let h = ctx.config.hidden_size;
        let eps = ctx.config.rms_norm_eps as f32;
        let k = num_tokens as u32;
        let bf16 = 2usize;
        #[allow(unused_variables)]
        let fp32 = 4usize;

        let ssm_state = state
            .as_any_mut()
            .downcast_mut::<SsmLayerState>()
            .ok_or_else(|| anyhow::anyhow!("Expected SsmLayerState"))?;

        let nk = ctx.config.linear_num_key_heads;
        let kd = ctx.config.linear_key_head_dim;
        let nv = ctx.config.linear_num_value_heads;
        let vd = ctx.config.linear_value_head_dim;
        let vpg = nv / nk;
        let key_dim = nk * kd;
        let value_dim = nv * vd;
        #[allow(unused_variables)]
        let conv_dim = key_dim * 2 + value_dim;
        let d_conv = ctx.config.linear_conv_kernel_dim;
        let qkvz_size = ctx.config.ssm_qkvz_size();

        macro_rules! prof {
            ($label:expr, $t0:expr) => {
                if ctx.profile {
                    if let Some(t0) = $t0 {
                        ctx.gpu.synchronize(stream)?;
                        let elapsed = t0.elapsed().as_micros();
                        tracing::info!("  SSM prefill [{}] N={}: {}\u{b5}s", $label, k, elapsed);
                    }
                }
            };
        }
        let mut t0 = if ctx.profile {
            ctx.gpu.synchronize(stream)?;
            Some(std::time::Instant::now())
        } else {
            None
        };

        let deinterleaved = ctx.buffers.ssm_deinterleaved();
        self.prefill_qkvz_proj(
            normed,
            deinterleaved,
            k,
            qkvz_size,
            h,
            nk,
            kd,
            vpg,
            vd,
            ctx,
            stream,
        )?;
        // 2026-09-25: METRALE_GDN_DUMP tag `post_qkvz`: the deinterleaved
        // projection, `qkvz_size` (Q, K, V and Z) elements per token.
        super::debug::maybe_dump_gdn_buf(
            ctx.gpu,
            deinterleaved,
            (num_tokens - 1) * qkvz_size * bf16,
            qkvz_size,
            ssm_layer_idx,
            "post_qkvz",
            &super::debug::DUMP_GDN,
            stream,
        )?;

        // 2026-09-25: METRALE_QWEN4EXP_DUMP tap `qkvz_preconv`: the QKVZ
        // projection before the conv.
        crate::layers::ple::dump::tap_bf16(
            ctx.gpu,
            deinterleaved,
            ssm_layer_idx,
            "qkvz_preconv",
            num_tokens * qkvz_size,
            stream,
        );

        prof!("qkvz_gemm", t0);
        t0 = if ctx.profile {
            ctx.gpu.synchronize(stream)?;
            Some(std::time::Instant::now())
        } else {
            None
        };

        // 2026-09-25: Fused BA GEMM and GDN gates. Per token: gate[nv] then
        // beta[nv], FP32, `gate_stride` = 2 * nv.
        let ba_size = ctx.config.ssm_ba_size();
        let gates_buf = ctx.buffers.ssm_gates();
        let gate_stride = nv * 2;
        ops::dense_gemm_ba_gates_prefill(
            ctx.gpu,
            self.ba_gates_prefill_k,
            self.ba_gates_prefill_hopper_k,
            normed,
            &self.ssm.in_proj_ba,
            self.ssm.a_log.weight,
            self.ssm.dt_bias.weight,
            gates_buf,
            k,
            ba_size as u32,
            h as u32,
            h as u32,
            gate_stride as u32,
            nv as u32,
            vpg as u32,
            stream,
        )?;
        // 2026-09-25: Tap `gates`: the gates as the recurrence reads them.
        crate::layers::ple::dump::tap_f32(
            ctx.gpu,
            gates_buf,
            ssm_layer_idx,
            "gates",
            num_tokens * gate_stride,
            stream,
        );
        prof!("ba+gates", t0);
        t0 = if ctx.profile {
            ctx.gpu.synchronize(stream)?;
            Some(std::time::Instant::now())
        } else {
            None
        };

        // 2026-09-25: The conv output goes in `ssm_qkvz`, which is free: the
        // QKVZ projection is already in `deinterleaved`.
        let conv_out_buf = ctx.buffers.ssm_qkvz();
        let gdn_out_buf = ctx.buffers.attn_output();

        // 2026-09-25: Conv input rows are `qkvz_size` wide; the output keeps
        // the first `conv_dim` channels (Q, K, V). `midcap_idx` is this SSM
        // layer's ordinal for a mid-chunk tail capture, shared by the conv and
        // recurrence splits; `None` unless `ctx.midchunk_capture` is set.
        let midcap_idx = ctx.midchunk_capture.as_ref().map(|c| {
            c.ssm_layer_counter
                .fetch_add(1, std::sync::atomic::Ordering::Relaxed)
        });
        self.conv1d_prefill_capture(
            ctx,
            ssm_state.conv_state,
            deinterleaved,
            conv_out_buf,
            conv_dim,
            d_conv,
            k,
            qkvz_size,
            midcap_idx,
            stream,
        )?;
        // 2026-09-25: Tap `post_conv`: the conv output. The gates computed
        // above are not read until the recurrence.
        crate::layers::ple::dump::tap_bf16(
            ctx.gpu,
            conv_out_buf,
            ssm_layer_idx,
            "post_conv",
            num_tokens * conv_dim,
            stream,
        );

        // 2026-09-25: METRALE_GDN_DUMP tag `conv`: the last token's conv
        // output, `[conv_dim]` BF16.
        super::debug::maybe_dump_gdn_buf(
            ctx.gpu,
            conv_out_buf,
            (num_tokens - 1) * conv_dim * bf16,
            conv_dim,
            ssm_layer_idx,
            "conv",
            &super::debug::DUMP_CONV,
            stream,
        )?;
        prof!("conv1d", t0);
        t0 = if ctx.profile {
            ctx.gpu.synchronize(stream)?;
            Some(std::time::Instant::now())
        } else {
            None
        };

        // 2026-09-25: L2 norm over the Q and K heads, the first `2 * key_dim`
        // elements of each `conv_dim` row, in place.
        ops::l2_norm(
            ctx.gpu,
            self.l2_norm_k,
            conv_out_buf,
            (nk * 2) as u32,
            kd as u32,
            1e-6,
            k,
            conv_dim as u32,
            stream,
        )?;
        // 2026-09-25: METRALE_GDN_DUMP tag `l2`: the same rows after the L2
        // norm.
        super::debug::maybe_dump_gdn_buf(
            ctx.gpu,
            conv_out_buf,
            (num_tokens - 1) * conv_dim * bf16,
            conv_dim,
            ssm_layer_idx,
            "l2",
            &super::debug::DUMP_L2,
            stream,
        )?;
        prof!("l2_norm", t0);
        t0 = if ctx.profile {
            ctx.gpu.synchronize(stream)?;
            Some(std::time::Instant::now())
        } else {
            None
        };

        // 2026-09-25: `prefill_gdn_recurrence_staged` picks the recurrence
        // kernel (FLA chunked, register-resident, WY4, persistent or split4).
        let q_ptr = conv_out_buf;
        let k_ptr = conv_out_buf.offset(key_dim * bf16);
        let v_ptr = conv_out_buf.offset(key_dim * 2 * bf16);

        self.prefill_gdn_recurrence_staged(
            ssm_state,
            q_ptr,
            k_ptr,
            v_ptr,
            gates_buf,
            gdn_out_buf,
            k,
            nk,
            nv,
            kd,
            vd,
            conv_dim,
            midcap_idx,
            ctx,
            stream,
        )?;

        // 2026-09-25: Tap `raw_recur`: the recurrence output before the gated
        // norm.
        crate::layers::ple::dump::tap_bf16(
            ctx.gpu,
            gdn_out_buf,
            ssm_layer_idx,
            "raw_recur",
            num_tokens * value_dim,
            stream,
        );

        // 2026-09-25: METRALE_GDN_DUMP tag `gdn`: the last token's recurrence
        // output, `[value_dim]` BF16.
        super::debug::maybe_dump_gdn_buf(
            ctx.gpu,
            gdn_out_buf,
            (num_tokens - 1) * value_dim * bf16,
            value_dim,
            ssm_layer_idx,
            "gdn",
            &super::debug::DUMP_GDN,
            stream,
        )?;
        prof!("gdn_prefill", t0);
        t0 = if ctx.profile {
            ctx.gpu.synchronize(stream)?;
            Some(std::time::Instant::now())
        } else {
            None
        };

        let normed_out_buf = conv_out_buf;
        let z_base = deinterleaved.offset((key_dim * 2 + value_dim) * bf16);
        ops::gated_rms_norm_prefill(
            ctx.gpu,
            self.gated_rms_norm_prefill_k,
            gdn_out_buf,
            z_base,
            &self.ssm.norm,
            normed_out_buf,
            nv as u32,
            vd as u32,
            eps,
            k,
            value_dim as u32,
            qkvz_size as u32,
            stream,
        )?;
        // 2026-09-25: METRALE_GDN_DUMP tag `gnorm`: the gated-norm output. It
        // reuses `conv_out_buf`, whose conv rows are no longer needed, with rows
        // `value_dim` apart as `prefill_out_proj_dispatch` reads them.
        super::debug::maybe_dump_gdn_buf(
            ctx.gpu,
            normed_out_buf,
            (num_tokens - 1) * value_dim * bf16,
            value_dim,
            ssm_layer_idx,
            "gnorm",
            &super::debug::DUMP_GNORM,
            stream,
        )?;
        prof!("gated_rms_norm", t0);
        t0 = if ctx.profile {
            ctx.gpu.synchronize(stream)?;
            Some(std::time::Instant::now())
        } else {
            None
        };

        crate::layers::ple::dump::tap_bf16(
            ctx.gpu,
            normed_out_buf,
            ssm_layer_idx,
            "pre_out_proj",
            num_tokens * value_dim,
            stream,
        );

        let out_proj_buf = ctx.buffers.moe_output();
        self.prefill_out_proj_dispatch(ctx, normed_out_buf, out_proj_buf, k, h, value_dim, stream)?;
        // 2026-09-25: Sum the out_proj partials across TP ranks and apply the
        // out_proj LoRA delta.
        self.ssm_tp_all_reduce(out_proj_buf, normed_out_buf, num_tokens, ctx, stream)?;
        // 2026-09-25: METRALE_GDN_DUMP tag `out_proj`: the out_proj output.
        super::debug::maybe_dump_gdn_buf(
            ctx.gpu,
            out_proj_buf,
            (num_tokens - 1) * h * bf16,
            h,
            ssm_layer_idx,
            "out_proj",
            &super::debug::DUMP_GDN,
            stream,
        )?;

        prof!("out_proj", t0);
        Ok(out_proj_buf)
    }
}
