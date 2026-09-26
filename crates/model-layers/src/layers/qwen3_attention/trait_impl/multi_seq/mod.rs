// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-25: Multi-sequence batched decode for `Qwen3AttentionLayer`: one
//! decode token per sequence, `n` sequences per call.
//!
//! Owner: model-layers (qwen3 attention).
//! Invariants: none beyond the types.
//!
//! `decode_multi_seq_inner` builds the `ctx::MultiSeqCtx`, runs the input RMS
//! norm, then the phases in order: `ms_phase_qkv` (`qkv`), `ms_phase_rope`,
//! `ms_phase_cache_write`, `ms_phase_paged_decode`, `ms_phase_o_proj` (`attn`),
//! and `ms_phase_ffn` (`ffn`). MLA layers replace the attention phases with
//! `ms_mla_decode` (`mla`, `mla_gemv`). Hyper-connection layers take
//! `decode_multi_seq_inner_hc`.

use anyhow::Result;
use metrale_cache::kv_cache::PagedKvCache;
use metrale_gpu_runtime::gpu::DevicePtr;

use super::super::Qwen3AttentionLayer;
use crate::layer::{ForwardContext, LayerState};
use crate::layers::ops;

mod attn;
mod ctx;
mod ffn;
mod mla;
mod mla_gemv;
mod qkv;
mod qkv_fp8_batch;
mod w8a8_decode;

impl Qwen3AttentionLayer {
    #[allow(clippy::too_many_arguments)]
    pub(in crate::layers::qwen3_attention) fn decode_multi_seq_inner<'a, 'b: 'a>(
        &self,
        hidden: DevicePtr,
        residual: DevicePtr,
        num_seqs: usize,
        states: &'a mut [&'b mut (dyn LayerState + 'static)],
        kv_cache: &mut PagedKvCache,
        _seq_lens: &[usize],
        _block_tables: &[Vec<u32>],
        ctx: &ForwardContext,
        stream: u64,
    ) -> Result<()> {
        let bs = kv_cache.block_size() as u32;
        let mut c = ctx::MultiSeqCtx::new(self, ctx, hidden, residual, num_seqs, bs, stream);
        if let Some(m) = ctx.attn_metadata.as_ref() {
            c.seq_slot = m.seq_slot;
        }

        if self.hc.is_some() {
            return self.decode_multi_seq_inner_hc(c, states, _seq_lens, kv_cache, ctx, stream);
        }
        let _ = states;

        ops::rms_norm_residual(
            ctx.gpu,
            self.rms_norm_residual_k,
            c.hidden,
            &self.input_norm,
            c.normed,
            c.residual,
            c.n as u32,
            c.h as u32,
            c.eps,
            c.stream,
        )?;

        let meta = ctx
            .attn_metadata
            .expect("attention layer requires metadata");

        // 2026-09-25: An MLA layer's standard projections are placeholders (the
        // Mistral MLA loader sets `attn.q_proj` to `DevicePtr::NULL`), so MLA
        // must not reach `ms_phase_qkv`.
        let o_out = if let Some(ref _mla) = self.mla {
            self.ms_mla_decode(&c, kv_cache, meta)?
        } else {
            self.ms_phase_qkv(&c)?;

            self.ms_phase_rope(&c, meta)?;

            self.ms_phase_cache_write(&c, kv_cache, meta)?;

            let attn_out = self.ms_phase_paged_decode(&c, kv_cache, meta)?;

            self.ms_phase_o_proj(&c, attn_out)?
        };

        // 2026-09-25: Under tensor parallelism each rank's o_proj output is a
        // partial sum; the all-reduce completes it before the FFN, as in
        // `decode_inner.rs` and `prefill_inner.rs`.
        if c.fwd.config.tp_world_size > 1
            && let Some(comm) = c.fwd.comm
        {
            let bytes = c.n * c.h * c.bf16;
            comm.all_reduce_async(o_out.0, bytes, c.stream)?;
        }

        self.ms_phase_ffn(&c, o_out)?;

        Ok(())
    }

    /// 2026-09-25: Multi-sequence decode for a hyper-connection layer. The
    /// attention runs batched; the FFN runs one sequence at a time.
    fn decode_multi_seq_inner_hc<'a, 'b: 'a>(
        &self,
        c: ctx::MultiSeqCtx<'_>,
        states: &'a mut [&'b mut (dyn LayerState + 'static)],
        seq_lens: &[usize],
        kv_cache: &mut PagedKvCache,
        ctx: &ForwardContext,
        stream: u64,
    ) -> Result<()> {
        let h = ctx.config.hidden_size;
        let eps = ctx.config.rms_norm_eps as f32;
        let n = c.n;
        let hc = self.hc.as_ref().unwrap();
        let hc_mult = hc.hc_mult as u32;
        // 2026-09-25: Model layer indices, not `attn_layer_idx`; see `prefill_inner.rs`.
        let is_first_layer = hc.is_first_model_layer;
        let is_last_layer = hc.is_last_model_layer;
        let hc_streams = ctx.buffers.hc_streams();
        let post = ctx.buffers.hc_post();
        let comb = ctx.buffers.hc_comb();
        let diag_this =
            std::env::var("METRALE_DIAG_V4_ALL_LAYERS").is_ok_and(|v| v == "1" || v == "true");

        if is_first_layer {
            ops::hc_expand(
                ctx.gpu,
                self.hc_expand_k,
                c.hidden,
                hc_streams,
                n as u32,
                h as u32,
                hc_mult,
                stream,
            )?;
        }

        ops::hc_pre_site(
            ctx.gpu,
            self.hc_pre_k,
            hc_streams,
            &hc.attn,
            hc,
            c.hidden,
            post,
            comb,
            ctx.buffers.hc_lowrank_scratch(),
            n as u32,
            h as u32,
            eps,
            stream,
        )?;
        if diag_this {
            super::diag_norm(
                ctx.gpu,
                c.hidden,
                n * h,
                stream,
                &format!("V4-msdecode L{} hc_pre-attn", self.attn_layer_idx),
            );
            super::diag_norm_f32(
                ctx.gpu,
                post,
                n * (hc_mult as usize),
                stream,
                &format!("V4-msdecode L{} post-attn", self.attn_layer_idx),
            );
            super::diag_norm_f32(
                ctx.gpu,
                comb,
                n * (hc_mult as usize) * (hc_mult as usize),
                stream,
                &format!("V4-msdecode L{} comb-attn", self.attn_layer_idx),
            );
        }
        if ops::HcVariant::of(hc).applies_block_input_norm() {
            ops::rms_norm(
                ctx.gpu,
                self.rms_norm_w_k,
                c.hidden,
                &self.input_norm,
                c.normed,
                n as u32,
                h as u32,
                eps,
                stream,
            )?;
        } else {
            // 2026-09-25: `hc_pre`'s `hc_norm` is this layer's input norm here; see
            // `prefill_inner.rs`.
            ctx.gpu
                .copy_d2d_async(c.hidden, c.normed, n * h * 2, stream)?;
        }

        let meta = ctx
            .attn_metadata
            .expect("attention layer requires metadata");

        let o_out = if let Some(ref _mla) = self.mla {
            self.ms_mla_decode(&c, kv_cache, meta)?
        } else {
            self.ms_phase_qkv(&c)?;
            self.ms_phase_rope(&c, meta)?;
            self.ms_phase_cache_write(&c, kv_cache, meta)?;
            let attn_out = self.ms_phase_paged_decode(&c, kv_cache, meta)?;
            self.ms_phase_o_proj(&c, attn_out)?
        };

        if c.fwd.config.tp_world_size > 1
            && let Some(comm) = c.fwd.comm
        {
            let bytes = c.n * c.h * c.bf16;
            comm.all_reduce_async(o_out.0, bytes, c.stream)?;
        }

        // 2026-09-25: QSA ingest for each sequence. Below the inert bound
        // `decode_select` only ingests and returns `None`. The model's decode
        // dispatch (`decode_a2.rs`, `qsa_active`) sends a hyper-connection batch
        // to the per-sequence loop once any sequence reaches the bound, so a
        // `Some` here means the two disagree, and the step fails instead of
        // running dense attention past the budget.
        if let Some(qsa) = self.qsa.as_ref() {
            for (i, state) in states.iter_mut().enumerate().take(n) {
                let st =
                    crate::layers::qwen3_attention::helpers::qsa_seq_state(qsa, *state, ctx.gpu)?;
                let sel = qsa.decode_select(
                    st,
                    c.normed.offset(i * h * c.bf16),
                    seq_lens[i],
                    kv_cache.k_pool_ptr(self.attn_layer_idx),
                    kv_cache.v_pool_ptr(self.attn_layer_idx),
                    meta.block_table
                        .offset(i * meta.max_blocks_per_seq as usize * 4),
                    c.bs,
                    ctx.gpu,
                    stream,
                )?;
                anyhow::ensure!(
                    sel.is_none(),
                    "QSA selection active for seq {i} on the batched ms path; \
                     the dispatch gate should have routed this batch per-seq"
                );
            }
        }

        ops::hc_post_site(
            ctx.gpu,
            self.hc_post_k,
            hc,
            o_out,
            hc_streams,
            post,
            comb,
            hc_streams,
            n as u32,
            h as u32,
            stream,
        )?;
        if diag_this {
            super::diag_norm_f32(
                ctx.gpu,
                hc_streams,
                h,
                stream,
                &format!("V4-msdecode L{} hc_post-attn", self.attn_layer_idx),
            );
            super::diag_norm_f32(
                ctx.gpu,
                hc_streams,
                n * (hc_mult as usize) * h,
                stream,
                &format!(
                    "V4-msdecode L{} hc_post-attn ALL_STREAMS",
                    self.attn_layer_idx
                ),
            );
        }

        if self.ffn.is_none() {
            if is_last_layer && let Some(ref head) = hc.head {
                ops::hc_head_site(
                    ctx.gpu,
                    self.hc_head_k,
                    hc_streams,
                    head,
                    hc,
                    c.hidden,
                    ctx.buffers.hc_lowrank_scratch(),
                    n as u32,
                    h as u32,
                    eps,
                    stream,
                )?;
                if diag_this {
                    super::diag_norm(
                        ctx.gpu,
                        c.hidden,
                        n * h,
                        stream,
                        &format!("V4-msdecode L{} hc_head", self.attn_layer_idx),
                    );
                }
            } else if is_last_layer {
                tracing::warn!(
                    "V4-msdecode L{}: hc_head SKIPPED (no head weights)",
                    self.attn_layer_idx
                );
            }
            return Ok(());
        }

        ops::hc_pre_site(
            ctx.gpu,
            self.hc_pre_k,
            hc_streams,
            &hc.ffn,
            hc,
            c.hidden,
            post,
            comb,
            ctx.buffers.hc_lowrank_scratch(),
            n as u32,
            h as u32,
            eps,
            stream,
        )?;
        if diag_this {
            super::diag_norm(
                ctx.gpu,
                c.hidden,
                n * h,
                stream,
                &format!("V4-msdecode L{} hc_pre-ffn", self.attn_layer_idx),
            );
            super::diag_norm_f32(
                ctx.gpu,
                post,
                n * (hc_mult as usize),
                stream,
                &format!("V4-msdecode L{} post-ffn", self.attn_layer_idx),
            );
            super::diag_norm_f32(
                ctx.gpu,
                comb,
                n * (hc_mult as usize) * (hc_mult as usize),
                stream,
                &format!("V4-msdecode L{} comb-ffn", self.attn_layer_idx),
            );
        }
        if ops::HcVariant::of(hc).applies_block_input_norm() {
            ops::rms_norm(
                ctx.gpu,
                self.rms_norm_w_k,
                c.hidden,
                &self.post_attn_norm,
                c.normed,
                n as u32,
                h as u32,
                eps,
                stream,
            )?;
        } else {
            ctx.gpu
                .copy_d2d_async(c.hidden, c.normed, n * h * 2, stream)?;
        }

        for i in 0..n {
            let normed2_i = c.normed.offset(i * c.h * c.bf16);
            let moe_out = self.ffn.forward(normed2_i, ctx, stream)?;
            // 2026-09-25: `hc_streams`, `post` and `comb` hold FP32, 4 bytes per element.
            let hc_streams_i = hc_streams.offset(i * hc.hc_mult * c.h * 4);
            let post_i = post.offset(i * hc.hc_mult * 4);
            let comb_i = comb.offset(i * hc.hc_mult * hc.hc_mult * 4);
            ops::hc_post_site(
                ctx.gpu,
                self.hc_post_k,
                hc,
                moe_out,
                hc_streams_i,
                post_i,
                comb_i,
                hc_streams_i,
                1,
                h as u32,
                stream,
            )?;
        }
        if diag_this {
            super::diag_norm_f32(
                ctx.gpu,
                hc_streams,
                h,
                stream,
                &format!("V4-msdecode L{} hc_post-ffn", self.attn_layer_idx),
            );
            super::diag_norm_f32(
                ctx.gpu,
                hc_streams,
                n * (hc_mult as usize) * h,
                stream,
                &format!(
                    "V4-msdecode L{} hc_post-ffn ALL_STREAMS",
                    self.attn_layer_idx
                ),
            );
        }

        if is_last_layer && let Some(ref head) = hc.head {
            ops::hc_head_site(
                ctx.gpu,
                self.hc_head_k,
                hc_streams,
                head,
                hc,
                c.hidden,
                ctx.buffers.hc_lowrank_scratch(),
                n as u32,
                h as u32,
                eps,
                stream,
            )?;
            if diag_this {
                super::diag_norm(
                    ctx.gpu,
                    c.hidden,
                    n * h,
                    stream,
                    &format!("V4-msdecode L{} hc_head", self.attn_layer_idx),
                );
            }
        } else if is_last_layer {
            tracing::warn!(
                "V4-msdecode L{}: hc_head SKIPPED (no head weights)",
                self.attn_layer_idx
            );
        }

        Ok(())
    }
}
