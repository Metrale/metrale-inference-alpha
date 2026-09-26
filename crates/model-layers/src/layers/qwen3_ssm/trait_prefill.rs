// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-25: Single-stream GDN layer prefill without mHC (`prefill_inner`):
//! input norm, the GDN block (`prefill_block`), post-norm, FFN, and the
//! residual adds.
//!
//! Owner: model-layers, GDN/SSM layer (`qwen3_ssm`).
//! Invariants:
//! - Each call bumps `SSM_LAYER_CALL_COUNTER` exactly once, before anything
//!   that can fail.

use super::*;

impl Qwen3SsmLayer {
    #[allow(clippy::too_many_arguments)]
    pub(super) fn prefill_inner(
        &self,
        hidden: DevicePtr,
        residual: DevicePtr,
        num_tokens: usize,
        state: &mut dyn LayerState,
        _kv_cache: &mut PagedKvCache,
        _seq_len_start: usize,
        _block_table: &mut Vec<u32>,
        _disk_block_ids: &mut Vec<u32>,
        _disk_last_offloaded_per_layer: &mut Vec<u32>,
        _kv_write_start: usize,
        ctx: &ForwardContext,
        stream: u64,
    ) -> Result<()> {
        let h = ctx.config.hidden_size;
        let eps = ctx.config.rms_norm_eps as f32;
        let k = num_tokens as u32;
        let bf16 = 2usize;
        let fp32 = 4usize;

        // 2026-09-25: The value labels this layer's `METRALE_GDN_DUMP` output;
        // `maybe_dump_gdn_buf` reduces it modulo METRALE_GDN_DUMP_N_SSM
        // (default 30).
        let ssm_layer_idx =
            super::debug::SSM_LAYER_CALL_COUNTER.fetch_add(1, std::sync::atomic::Ordering::Relaxed);

        macro_rules! prof {
            ($label:expr, $t0:expr) => {
                if ctx.profile {
                    if let Some(t0) = $t0 {
                        ctx.gpu.synchronize(stream)?;
                        let elapsed = t0.elapsed().as_micros();
                        tracing::info!("  SSM prefill [{}] N={}: {}µs", $label, k, elapsed);
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

        // 2026-09-25: For k > 4096, synchronise at entry so an earlier layer's
        // fault is reported here.
        if k > 4096 {
            tracing::info!("SSM prefill ENTRY: k={k} h={h}");
            ctx.gpu
                .synchronize(stream)
                .map_err(|e| anyhow::anyhow!("SSM prefill ENTRY: stream broken (k={k}): {e}"))?;
        }

        // 2026-09-25: METRALE_GDN_DUMP tag `pre_norm`: this layer's input hidden
        // state.
        super::debug::maybe_dump_gdn_buf(
            ctx.gpu,
            hidden,
            (num_tokens - 1) * h * fp32,
            h,
            ssm_layer_idx,
            "pre_norm",
            &super::debug::DUMP_CONV,
            stream,
        )?;

        let normed = ctx.buffers.norm_output();
        ops::rms_norm_residual(
            ctx.gpu,
            self.rms_norm_residual_k,
            hidden,
            &self.input_norm,
            normed,
            residual,
            k,
            h as u32,
            eps,
            stream,
        )?;
        // 2026-09-25: METRALE_GDN_DUMP tag `post_norm`: the normed input to the
        // QKVZ projection.
        super::debug::maybe_dump_gdn_buf(
            ctx.gpu,
            normed,
            (num_tokens - 1) * h * 2,
            h,
            ssm_layer_idx,
            "post_norm",
            &super::debug::DUMP_L2,
            stream,
        )?;
        if k > 4096 {
            ctx.gpu
                .synchronize(stream)
                .map_err(|e| anyhow::anyhow!("SSM prefill: SYNC after rms_norm (k={k}): {e}"))?;
        }

        prof!("rms_norm_residual", t0);
        t0 = if ctx.profile {
            ctx.gpu.synchronize(stream)?;
            Some(std::time::Instant::now())
        } else {
            None
        };

        let out_proj_buf =
            self.prefill_block(normed, num_tokens, state, ssm_layer_idx, ctx, stream)?;

        // 2026-09-25: METRALE_DUMP_EXPERT_IDS=1: log the norm and first five
        // values of the last token's `hidden`, `out_proj_buf` and their sum.
        if std::env::var("METRALE_DUMP_EXPERT_IDS").ok().as_deref() == Some("1") {
            ctx.gpu.synchronize(stream)?;
            let offset = (num_tokens - 1) * h * 2;
            let mut buf_h = vec![0u8; h * 2];
            let _ = ctx.gpu.copy_d2h(hidden.offset(offset), &mut buf_h);
            let v_h: Vec<f32> = buf_h
                .chunks_exact(2)
                .map(|c| {
                    let bits = u16::from_le_bytes([c[0], c[1]]);
                    f32::from_bits((bits as u32) << 16)
                })
                .collect();
            let n_h = v_h.iter().map(|x| x * x).sum::<f32>().sqrt();
            let mut buf_o = vec![0u8; h * 2];
            let _ = ctx.gpu.copy_d2h(out_proj_buf.offset(offset), &mut buf_o);
            let v_o: Vec<f32> = buf_o
                .chunks_exact(2)
                .map(|c| {
                    let bits = u16::from_le_bytes([c[0], c[1]]);
                    f32::from_bits((bits as u32) << 16)
                })
                .collect();
            let n_o = v_o.iter().map(|x| x * x).sum::<f32>().sqrt();
            tracing::info!(
                "METRALE_PRENORM_HIDDEN last_tok: |x|={:.4} first5={:?}",
                n_h,
                &v_h[..5]
            );
            tracing::info!(
                "METRALE_PRENORM_OUTPROJ last_tok: |x|={:.4} first5={:?}",
                n_o,
                &v_o[..5]
            );
            let v_sum: Vec<f32> = v_h.iter().zip(v_o.iter()).map(|(a, b)| a + b).collect();
            let n_sum = v_sum.iter().map(|x| x * x).sum::<f32>().sqrt();
            tracing::info!(
                "METRALE_PRENORM_SUM (hidden+out_proj): |x|={:.4} first5={:?}",
                n_sum,
                &v_sum[..5]
            );
        }

        ops::residual_add_rms_norm(
            ctx.gpu,
            self.residual_add_rms_norm_k,
            hidden,
            out_proj_buf,
            &self.post_attn_norm,
            ctx.buffers.norm_output(),
            residual,
            num_tokens as u32,
            h as u32,
            eps,
            stream,
        )?;
        self.ffn
            .forward_prefill(ctx.buffers.norm_output(), num_tokens, ctx, stream)?;
        // 2026-09-25: METRALE_GDN_DUMP tag `moe_out`: the FFN output.
        super::debug::maybe_dump_gdn_buf(
            ctx.gpu,
            ctx.buffers.moe_output(),
            (num_tokens - 1) * h * bf16,
            h,
            ssm_layer_idx,
            "moe_out",
            &super::debug::DUMP_GNORM,
            stream,
        )?;
        ops::residual_add(
            ctx.gpu,
            self.residual_add_k,
            hidden,
            ctx.buffers.moe_output(),
            (num_tokens * h) as u32,
            stream,
        )?;

        prof!("moe_ffn", t0);

        Ok(())
    }
}
