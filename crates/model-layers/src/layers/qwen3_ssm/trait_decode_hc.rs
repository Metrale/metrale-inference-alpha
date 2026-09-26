// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-25: Single-token GDN decode under an mHC highway (`decode_inner_hc`).
//!
//! Owner: model-layers, GDN/SSM layer (`qwen3_ssm`).
//! Invariants:
//! - Returns an error before any launch when the FFN's `fp32_routing_active`
//!   is true.
//!
//! The highway is the residual: `hc_post_site` adds each sublayer's output
//! into the streams, so this path runs none of `decode_inner`'s
//! `rms_norm_residual`, `residual_add_rms_norm` or `residual_add` steps.
//! `ssm_forward` takes no residual, so it runs unchanged on `hc_pre`'s
//! output. `trait_prefill_hc.rs` is the prefill counterpart.

use super::*;

impl Qwen3SsmLayer {
    pub(super) fn decode_inner_hc(
        &self,
        hidden: DevicePtr,
        state: &mut dyn LayerState,
        ctx: &ForwardContext,
        stream: u64,
    ) -> Result<()> {
        let hc = self
            .hc
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("decode_inner_hc without mHC weights"))?;
        let h = ctx.config.hidden_size;
        let eps = ctx.config.rms_norm_eps as f32;
        let hc_mult = hc.hc_mult as u32;

        let ssm_state = state
            .as_any_mut()
            .downcast_mut::<SsmLayerState>()
            .ok_or_else(|| anyhow::anyhow!("Expected SsmLayerState"))?;

        // 2026-09-25: Under METRALE_FP32_ROUTING the MoE gate reads
        // `moe_router_in_f32`, which only `residual_add_rms_norm_gatef32` in the
        // non-hc `decode_inner` paths writes. `hc_pre` replaces that norm here,
        // so the buffer would hold an earlier layer's rows; refuse the lever.
        anyhow::ensure!(
            !self.ffn.fp32_routing_active(ctx.levers),
            "qwen3_ssm mHC: METRALE_FP32_ROUTING needs the fused gate-f32 norm, \
             which the highway path replaces. The router would read a stale \
             moe_router_in_f32. Unset it."
        );

        let streams = ctx.buffers.hc_streams();
        let post = ctx.buffers.hc_post();
        let comb = ctx.buffers.hc_comb();

        // 2026-09-25: METRALE_QWEN4EXP_DECODE_PROF=1 logs per-stage wall clock
        // for the first 150 calls (`PROF_LEFT`). Each probe synchronises the
        // stream, so the mode is for measurement, not serving.
        static PROF: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
        static PROF_LEFT: std::sync::atomic::AtomicI32 = std::sync::atomic::AtomicI32::new(150);
        let prof = *PROF
            .get_or_init(|| std::env::var("METRALE_QWEN4EXP_DECODE_PROF").as_deref() == Ok("1"))
            && PROF_LEFT.fetch_sub(1, std::sync::atomic::Ordering::Relaxed) > 0;
        let mut t = if prof {
            ctx.gpu.synchronize(stream).ok();
            Some(std::time::Instant::now())
        } else {
            None
        };
        macro_rules! stage {
            ($name:expr) => {
                if let Some(t0) = t.as_mut() {
                    ctx.gpu.synchronize(stream).ok();
                    tracing::info!("hc-decode [{}]: {}us", $name, t0.elapsed().as_micros());
                    *t0 = std::time::Instant::now();
                }
            };
        }

        if hc.is_first_model_layer {
            ops::hc_expand(
                ctx.gpu,
                self.hc_expand_k,
                hidden,
                streams,
                1,
                h as u32,
                hc_mult,
                stream,
            )?;
        }

        // 2026-09-25: `fresh` is false: prefill created this sequence's PLE
        // state, and decode carries its conv state and token history.
        if let Some(ple) = self.ple.as_ref() {
            let st = ssm_state
                .ple
                .as_mut()
                .ok_or_else(|| anyhow::anyhow!("PLE decode before prefill: no seq state"))?;
            ple.forward(st, streams, 1, false, ctx, stream)?;
        }
        stage!("ple");

        // 2026-09-25: From here `hidden` is scratch; the highway carries the
        // state between layers.
        ops::hc_pre_site(
            ctx.gpu,
            self.hc_pre_k,
            streams,
            &hc.attn,
            hc,
            hidden,
            post,
            comb,
            ctx.buffers.hc_lowrank_scratch(),
            1,
            h as u32,
            eps,
            stream,
        )?;
        stage!("hc_pre_attn");
        // 2026-09-25: No `input_norm`: `hc_norm` inside `hc_pre` is this
        // layer's norm. The loader fills `input_norm` with ones, and a second
        // RMS norm with unit weights still rescales, so it is not applied.
        let ssm_out = self.ssm_forward(hidden, ssm_state, ctx, stream, false)?;
        stage!("ssm_forward");
        ops::hc_post_site(
            ctx.gpu,
            self.hc_post_k,
            hc,
            ssm_out,
            streams,
            post,
            comb,
            streams,
            1,
            h as u32,
            stream,
        )?;

        ops::hc_pre_site(
            ctx.gpu,
            self.hc_pre_k,
            streams,
            &hc.ffn,
            hc,
            hidden,
            post,
            comb,
            ctx.buffers.hc_lowrank_scratch(),
            1,
            h as u32,
            eps,
            stream,
        )?;
        stage!("hc_post+hc_pre_ffn");
        let moe_out = self.ffn.forward(hidden, ctx, stream)?;
        stage!("moe");
        ops::hc_post_site(
            ctx.gpu,
            self.hc_post_k,
            hc,
            moe_out,
            streams,
            post,
            comb,
            streams,
            1,
            h as u32,
            stream,
        )?;

        Ok(())
    }
}
