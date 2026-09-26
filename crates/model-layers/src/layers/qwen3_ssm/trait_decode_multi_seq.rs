// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-25: Multi-sequence decode for the GDN layer
//! (`decode_multi_seq_inner`): the SSM mixer, then the FFN dispatched by
//! batch width.
//!
//! Owner: model-layers, GDN/SSM layer (`qwen3_ssm`).
//! Invariants: none beyond the types.

use super::*;

mod hc;
mod ssm_batched;
mod ssm_batched_proj;
mod ssm_batched_recurrent;

/// 2026-09-25: Smallest batch that takes the batched dense FFN
/// (`forward_prefill`): `METRALE_SSM_FFN_PREFILL_MIN_N`, default 5. A value
/// below 2 or one that does not parse gives 5. Read once per process.
///
/// Measured 2026-07-27 on GB10, Qwen3.6-27B NVFP4, decode tok/s, 2 reps per
/// cell: MIN_N=9 gave C=4 37.7, C=8 53.4; MIN_N=5 gave 37.8, 57.8; MIN_N=4
/// gave 36.2, 57.8.
fn ssm_ffn_prefill_min_n() -> usize {
    static N: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    *N.get_or_init(|| {
        std::env::var("METRALE_SSM_FFN_PREFILL_MIN_N")
            .ok()
            .and_then(|v| v.parse().ok())
            .filter(|&v| v >= 2)
            .unwrap_or(5)
    })
}

/// 2026-09-25: The batched dense FFN arm is on unless
/// `METRALE_NO_SSM_FFN_PREFILL` is exactly `1`. Read once per process.
fn ssm_ffn_prefill_enabled() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var("METRALE_NO_SSM_FFN_PREFILL").as_deref() != Ok("1"))
}

impl Qwen3SsmLayer {
    #[allow(clippy::too_many_arguments)]
    /// 2026-09-25: Multi-sequence decode for a GDN layer.
    ///
    /// The mixer carries per-sequence recurrent state. It runs through
    /// `try_decode_multi_seq_ssm_batched` when that accepts the layer, else as
    /// a per-sequence loop over `ssm_forward`. Either way row `i` of
    /// `norm_output` ends up holding sequence `i`'s FFN input, and the FFN
    /// then runs over all `n` rows, dispatched by `n` and the FFN kind.
    ///
    /// In the per-sequence loop, `ssm_forward` writes no `norm_output` row and
    /// returns its output in `moe_output`, which the same iteration's
    /// `residual_add_rms_norm` consumes before the next sequence runs.
    pub(super) fn decode_multi_seq_inner<'a, 'b: 'a>(
        &self,
        hidden: DevicePtr,
        residual: DevicePtr,
        num_seqs: usize,
        states: &'a mut [&'b mut (dyn LayerState + 'static)],
        _kv_cache: &mut PagedKvCache,
        _seq_lens: &[usize],
        _block_tables: &[Vec<u32>],
        ctx: &ForwardContext,
        stream: u64,
    ) -> Result<()> {
        let h = ctx.config.hidden_size;
        let bf16 = 2usize;
        let eps = ctx.config.rms_norm_eps as f32;
        let n = num_seqs;
        let ssm_ms_profile =
            crate::layers::ops::ModelLevers::get().ssm_ms_profile && !ctx.graph_capture;
        let phase_a_t0 = if ssm_ms_profile {
            ctx.gpu.synchronize(stream).ok();
            Some(std::time::Instant::now())
        } else {
            None
        };

        let residual_elem = 2usize;

        // 2026-09-25: Mixer. The batched path runs QKVZ and out_proj as one
        // GEMM each over all n rows; the per-sequence loop below reads those
        // weights once per sequence.
        if !self
            .try_decode_multi_seq_ssm_batched(hidden, residual, n, states, false, ctx, stream)?
        {
            for i in 0..n {
                let hidden_i = hidden.offset(i * h * residual_elem);
                let residual_i = residual.offset(i * h * residual_elem);
                let normed_i = ctx.buffers.norm_output().offset(i * h * bf16);

                let ssm_state = states[i]
                    .as_any_mut()
                    .downcast_mut::<SsmLayerState>()
                    .ok_or_else(|| anyhow::anyhow!("Expected SsmLayerState for seq {i}"))?;

                ops::rms_norm_residual(
                    ctx.gpu,
                    self.rms_norm_residual_k,
                    hidden_i,
                    &self.input_norm,
                    normed_i,
                    residual_i,
                    1,
                    h as u32,
                    eps,
                    stream,
                )?;

                let ssm_out = self.ssm_forward(normed_i, ssm_state, ctx, stream, false)?;

                ops::residual_add_rms_norm(
                    ctx.gpu,
                    self.residual_add_rms_norm_k,
                    hidden_i,
                    ssm_out,
                    &self.post_attn_norm,
                    normed_i,
                    residual_i,
                    1,
                    h as u32,
                    eps,
                    stream,
                )?;
            }
        }
        let phase_a_us = if let Some(t0) = phase_a_t0 {
            ctx.gpu.synchronize(stream).ok();
            t0.elapsed().as_micros()
        } else {
            0
        };
        let phase_b_t0 = if ssm_ms_profile {
            Some(std::time::Instant::now())
        } else {
            None
        };

        let normed_base = ctx.buffers.norm_output();
        match n {
            2 | 3 => {
                if n == 2 {
                    self.ffn.forward_k2(normed_base, ctx, stream)?;
                } else {
                    self.ffn.forward_k3(normed_base, ctx, stream)?;
                }
                for i in 0..n {
                    let hidden_i = hidden.offset(i * h * residual_elem);
                    let moe_out_i = ctx.buffers.moe_output().offset(i * h * bf16);
                    ops::residual_add(
                        ctx.gpu,
                        self.residual_add_k,
                        hidden_i,
                        moe_out_i,
                        h as u32,
                        stream,
                    )?;
                }
            }
            // 2026-09-25: Dense FFN at n >= `ssm_ffn_prefill_min_n()`: one
            // batched `forward_prefill` over all n rows, where the chunked arm
            // below reads gate/up/down once per 8 rows. Measured
            // 2026-07-27 on GB10 (METRALE_SSM_MS_PROFILE, Qwen3.6-27B): the
            // chunked arm cost 1023 us per layer at n=8 and 2022 us at n=16.
            n if n >= ssm_ffn_prefill_min_n()
                && self.ffn.is_dense()
                && ssm_ffn_prefill_enabled() =>
            {
                self.ffn.forward_prefill(normed_base, n, ctx, stream)?;
                ops::residual_add(
                    ctx.gpu,
                    self.residual_add_k,
                    hidden,
                    ctx.buffers.moe_output(),
                    (n * h) as u32,
                    stream,
                )?;
            }
            4.. if self.ffn.can_forward_km(8.min(n) as u32) => {
                // 2026-09-25: Dense layers at n >= 4 that the arm above did not
                // take: `try_forward_km` runs a batched GEMV over chunks of at
                // most 8 rows, reading the weights ceil(n/8) times.
                // `can_forward_km` is false for MoE, which falls through.
                // Each chunk reuses `moe_output[0..m]`, so its residual add
                // runs before the next chunk overwrites it.
                let mut done = 0usize;
                while done < n {
                    let m = (n - done).min(8);
                    let normed_c = normed_base.offset(done * h * bf16);
                    let used = self.ffn.try_forward_km(normed_c, m as u32, ctx, stream)?;
                    debug_assert!(used, "can_forward_km checked at branch entry");
                    let hidden_c = hidden.offset(done * h * residual_elem);
                    ops::residual_add(
                        ctx.gpu,
                        self.residual_add_k,
                        hidden_c,
                        ctx.buffers.moe_output(),
                        (m * h) as u32,
                        stream,
                    )?;
                    done += m;
                }
            }
            _ => {
                // 2026-09-25: Every other case, in order: grouped FP8 MoE,
                // grouped-GEMM MoE, the C=4 atomic MoE kernel, token-major MoE
                // (the default), `forward_batched`, and a per-row `forward`
                // loop.
                if self.ffn.fp8_grouped_decode_ok(n, ctx) {
                    // 2026-09-25: METRALE_FP8_MOE_GROUPED_DECODE: FP8
                    // block-scaled experts grouped across all rows in one
                    // dispatch (`forward_token_major_decode` hands FP8 experts
                    // to `forward_batched`).
                    self.ffn
                        .forward_fp8_grouped_decode(normed_base, n, ctx, stream)?;
                    let moe_out = ctx.buffers.moe_output();
                    ops::residual_add(
                        ctx.gpu,
                        self.residual_add_k,
                        hidden,
                        moe_out,
                        (n * h) as u32,
                        stream,
                    )?;
                } else if crate::layers::moe_grouped_decode_for(n) {
                    // 2026-09-25: Grouped-GEMM MoE over all rows.
                    // `moe_grouped_decode_for` turns it on at n >= 16 unless
                    // METRALE_NO_MOE_GROUPED_DECODE is set;
                    // METRALE_MOE_GROUPED_DECODE=1 also runs it below 16.
                    self.ffn.forward_prefill(normed_base, n, ctx, stream)?;
                    let moe_out = ctx.buffers.moe_output();
                    ops::residual_add(
                        ctx.gpu,
                        self.residual_add_k,
                        hidden,
                        moe_out,
                        (n * h) as u32,
                        stream,
                    )?;
                } else if n == 4
                    && std::env::var("METRALE_MOE_ATOMIC_C4_DECODE")
                        .ok()
                        .as_deref()
                        == Some("1")
                {
                    // 2026-09-25: C=4 MoE decode that accumulates the routed
                    // down projections with FP32 atomic adds.
                    self.ffn
                        .forward_atomic_c4_decode(normed_base, n, ctx, stream)?;
                    let moe_out = ctx.buffers.moe_output();
                    ops::residual_add(
                        ctx.gpu,
                        self.residual_add_k,
                        hidden,
                        moe_out,
                        (n * h) as u32,
                        stream,
                    )?;
                } else if !crate::layers::ops::ModelLevers::get().moe_legacy_pertoken_decode {
                    // 2026-09-25: Token-major MoE decode, the default
                    // (METRALE_MOE_LEGACY_PERTOKEN_DECODE=1 turns it off). With a
                    // resident MoE adapter or unsupported expert weights,
                    // `forward_token_major_decode` runs `forward_batched` itself;
                    // any error it returns also falls back to `forward_batched`.
                    if self
                        .ffn
                        .forward_token_major_decode(normed_base, n, ctx, stream)
                        .is_err()
                    {
                        self.ffn.forward_batched(normed_base, n, ctx, stream)?;
                    }
                    let moe_out = ctx.buffers.moe_output();
                    ops::residual_add(
                        ctx.gpu,
                        self.residual_add_k,
                        hidden,
                        moe_out,
                        (n * h) as u32,
                        stream,
                    )?;
                } else if std::env::var("METRALE_MOE_BATCHED_DECODE").ok().as_deref() == Some("1") {
                    self.ffn.forward_batched(normed_base, n, ctx, stream)?;
                    let moe_out = ctx.buffers.moe_output();
                    ops::residual_add(
                        ctx.gpu,
                        self.residual_add_k,
                        hidden,
                        moe_out,
                        (n * h) as u32,
                        stream,
                    )?;
                } else {
                    for i in 0..n {
                        let hidden_i = hidden.offset(i * h * residual_elem);
                        let normed_i = normed_base.offset(i * h * bf16);
                        let moe_out = self.ffn.forward(normed_i, ctx, stream)?;
                        ops::residual_add(
                            ctx.gpu,
                            self.residual_add_k,
                            hidden_i,
                            moe_out,
                            h as u32,
                            stream,
                        )?;
                    }
                }
            }
        }
        if let Some(t0) = phase_b_t0 {
            ctx.gpu.synchronize(stream).ok();
            tracing::info!(
                "METRALE_SSM_MS_PROFILE n={n}: mixer={}us moe_residual={}us",
                phase_a_us,
                t0.elapsed().as_micros(),
            );
        }

        Ok(())
    }
}
