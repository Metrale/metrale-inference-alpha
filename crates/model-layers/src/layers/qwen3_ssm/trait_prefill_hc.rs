// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-25: GDN prefill under an mHC highway (`prefill_inner_hc`).
//!
//! Owner: model-layers, GDN/SSM layer (`qwen3_ssm`).
//! Invariants:
//! - Returns an error before any launch when the FFN's `fp32_routing_active`
//!   is true.
//!
//! Qwen3.8-Flash-Next is the only model whose loader attaches mHC weights to
//! this layer. This is not a wrapper around `prefill_inner`, which folds its
//! residual bookkeeping into its norms:
//!
//! ```text
//! rms_norm_residual(hidden, input_norm) -> normed, residual      # step 1
//!   prefill_block(normed)               -> out_proj_buf          # steps 2-10
//! residual_add_rms_norm(hidden, out_proj_buf, post_attn_norm)    # step 11
//! ffn.forward_prefill(norm_output)                               # step 12
//! residual_add(hidden, moe_output)                               # step 13
//! ```
//!
//! Under mHC the highway is the residual: the block output reaches it through
//! `hc_post`, weighted per stream by the `post` vector `hc_pre` emitted. This
//! path therefore replaces steps 1, 11 and 13, in the same
//! `hc_expand`/`hc_pre`/`hc_post` order as `qwen3_attention`'s
//! `prefill_inner_hc`:
//!
//! ```text
//! hc_expand(hidden -> streams)                    # MODEL layer 0 only
//! hc_pre(streams, attn_site) -> hidden, inj       # `hidden` is scratch here
//!   prefill_block(hidden)    -> out_proj_buf
//! hc_post(out_proj_buf, streams, inj) -> streams
//! hc_pre(streams, ffn_site)  -> hidden, inj
//!   ffn.forward_prefill(hidden) -> moe_output
//! hc_post(moe_output, streams, inj) -> streams
//! ```
//!
//! No `input_norm`, no `post_attn_norm`, no `residual_add`: the checkpoint
//! has no per-layer norms, the loader fills those slots with ones, and
//! `hc_norm` inside `hc_pre` is the norm.
//!
//! `hc_expand` runs where `hc.is_first_model_layer` is set, which the loader
//! sets on model layer 0. The checkpoint's `layer_types` makes layers 0-2 GDN
//! and layer 3 the first full-attention layer, so the highway is seeded on
//! this path. There is no `hc_head` here: the last model layer (47) is full
//! attention, so the final collapse runs on the attention path.

use super::*;

impl Qwen3SsmLayer {
    #[allow(clippy::too_many_arguments)]
    pub(super) fn prefill_inner_hc(
        &self,
        hidden: DevicePtr,
        num_tokens: usize,
        state: &mut dyn LayerState,
        seq_len_start: usize,
        ctx: &ForwardContext,
        stream: u64,
    ) -> Result<()> {
        let hc = self
            .hc
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("prefill_inner_hc without mHC weights"))?;
        let h = ctx.config.hidden_size;
        let eps = ctx.config.rms_norm_eps as f32;
        let n = num_tokens as u32;

        // 2026-09-25: One increment per call, as in `prefill_inner`; the value
        // labels this layer's `METRALE_GDN_DUMP` and `METRALE_QWEN4EXP_DUMP` taps.
        let ssm_layer_idx =
            super::debug::SSM_LAYER_CALL_COUNTER.fetch_add(1, std::sync::atomic::Ordering::Relaxed);

        // 2026-09-25: METRALE_QWEN4EXP_PREFILL_PROF=1 logs per-stage µs for the
        // first 400 calls (`PROF_LEFT`), synchronising the stream at each stage.
        static PROF: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
        static PROF_LEFT: std::sync::atomic::AtomicI32 = std::sync::atomic::AtomicI32::new(400);
        let prof = *PROF
            .get_or_init(|| std::env::var("METRALE_QWEN4EXP_PREFILL_PROF").as_deref() == Ok("1"))
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
                    tracing::info!(
                        "hc-prefill L{ssm_layer_idx} T={num_tokens} [{}]: {}us",
                        $name,
                        t0.elapsed().as_micros()
                    );
                    *t0 = std::time::Instant::now();
                }
            };
        }

        // 2026-09-25: Under METRALE_FP32_ROUTING the MoE gate reads
        // `moe_router_in_f32`, which only `residual_add_rms_norm_gatef32` in the
        // non-hc `decode_inner` paths writes. This path never writes it, so the
        // gate would read an earlier layer's rows; refuse the lever.
        anyhow::ensure!(
            !self.ffn.fp32_routing_active(ctx.levers),
            "qwen3_ssm mHC: METRALE_FP32_ROUTING needs the fused gate-f32 norm, \
             which the highway path replaces. The router would read a stale \
             moe_router_in_f32. Unset it."
        );

        // 2026-09-25: In a mixed step this chunk's highway rows sit above the
        // padded decode rows; `ctx.hc_row_offset` counts those rows.
        let streams = ctx
            .buffers
            .hc_streams()
            .offset(ctx.hc_row_offset * hc.hc_mult * h * 4);
        let post = ctx.buffers.hc_post();
        let comb = ctx.buffers.hc_comb();

        if hc.is_first_model_layer {
            ops::hc_expand(
                ctx.gpu,
                self.hc_expand_k,
                hidden,
                streams,
                n,
                h as u32,
                hc.hc_mult as u32,
                stream,
            )?;
        }

        // 2026-09-25: PLE injects into the highway before this layer's
        // `hc_pre`, the reference order (bench/qwen4_exp/ARCHITECTURE.md §2).
        // `fresh` is a prefill from position 0; it resets the PLE conv state
        // and token history.
        if let Some(ple) = self.ple.as_ref() {
            let ssm = state
                .as_any_mut()
                .downcast_mut::<crate::layer::SsmLayerState>()
                .ok_or_else(|| anyhow::anyhow!("PLE host layer state is not SsmLayerState"))?;
            if ssm.ple.is_none() {
                ssm.ple = Some(ple.new_seq_state(ctx.gpu)?);
            }
            let st = ssm.ple.as_mut().expect("just created");
            ple.forward(st, streams, num_tokens, seq_len_start == 0, ctx, stream)?;
        }
        stage!("ple");

        // 2026-09-25: `hidden` is scratch from here on: the highway carries the
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
            n,
            h as u32,
            eps,
            stream,
        )?;
        stage!("hc_pre_attn");
        let hc_dim = hc.hc_mult * h;
        crate::layers::ple::dump::tap_highway(
            ctx.gpu,
            streams,
            ssm_layer_idx,
            "in",
            num_tokens,
            hc_dim,
            stream,
        );
        crate::layers::ple::dump::tap_bf16(
            ctx.gpu,
            hidden,
            ssm_layer_idx,
            "hc_pre_mixed",
            num_tokens * h,
            stream,
        );
        crate::layers::ple::dump::tap_f32(
            ctx.gpu,
            post,
            ssm_layer_idx,
            "hc_pre_inj",
            num_tokens * hc.hc_mult,
            stream,
        );
        let out_proj_buf =
            self.prefill_block(hidden, num_tokens, state, ssm_layer_idx, ctx, stream)?;
        stage!("gdn_block");
        crate::layers::ple::dump::tap_bf16(
            ctx.gpu,
            out_proj_buf,
            ssm_layer_idx,
            "block_out",
            num_tokens * h,
            stream,
        );
        ops::hc_post_site(
            ctx.gpu,
            self.hc_post_k,
            hc,
            out_proj_buf,
            streams,
            post,
            comb,
            streams,
            n,
            h as u32,
            stream,
        )?;

        stage!("hc_post_attn");
        // 2026-09-25: Tapped before the MoE: at layer 0 a reference reproduces
        // this point from the GDN block alone, without any experts.
        crate::layers::ple::dump::tap_highway(
            ctx.gpu,
            streams,
            ssm_layer_idx,
            "post_gdn",
            num_tokens,
            hc_dim,
            stream,
        );

        // 2026-09-25: `prefill_block` returned `ctx.buffers.moe_output()`, which
        // the FFN overwrites next. The `hc_post_site` above has already read it
        // into the highway, so that call must stay before the FFN.
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
            n,
            h as u32,
            eps,
            stream,
        )?;
        stage!("hc_pre_ffn");
        self.ffn.forward_prefill(hidden, num_tokens, ctx, stream)?;
        stage!("moe");
        ops::hc_post_site(
            ctx.gpu,
            self.hc_post_k,
            hc,
            ctx.buffers.moe_output(),
            streams,
            post,
            comb,
            streams,
            n,
            h as u32,
            stream,
        )?;
        crate::layers::ple::dump::tap_highway(
            ctx.gpu,
            streams,
            ssm_layer_idx,
            "post_moe",
            num_tokens,
            hc_dim,
            stream,
        );
        stage!("hc_post_ffn");

        Ok(())
    }
}
