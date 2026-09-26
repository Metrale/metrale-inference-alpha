// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-26: `prefill_inner_hc`, the prefill body of a hyper-connection (mHC) layer.
//! `prefill_inner` calls it when the layer has `hc` weights.
//!
//! Owner: model-layers (qwen3 attention).
//! Invariants:
//! - A batched call (`batched_meta` is `Some`) at `seq_len_start == 0`, or on a layer with
//!   high-speed swap engaged, returns an error.

use anyhow::Result;
use metrale_cache::kv_cache::PagedKvCache;
use metrale_gpu_runtime::gpu::DevicePtr;

use super::super::super::Qwen3AttentionLayer;
use crate::layer::{BatchedAttnMetadata, ForwardContext, LayerState};
use crate::layers::ops;

impl Qwen3AttentionLayer {
    /// 2026-09-25: Prefill body for a hyper-connection (mHC) layer. The highway
    /// (`hc_streams`, `hc_mult` streams per token) carries the residual state
    /// across layers; `hidden` is single-stream scratch.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn prefill_inner_hc(
        &self,
        hidden: DevicePtr,
        _residual: DevicePtr,
        num_tokens: usize,
        state: &mut dyn LayerState,
        kv_cache: &mut PagedKvCache,
        seq_len_start: usize,
        block_table: &mut Vec<u32>,
        disk_block_ids: &mut Vec<u32>,
        disk_last_offloaded_per_layer: &mut Vec<u32>,
        kv_write_start: usize,
        batched_meta: Option<&BatchedAttnMetadata>,
        ctx: &ForwardContext,
        stream: u64,
    ) -> Result<()> {
        let h = ctx.config.hidden_size;
        let eps = ctx.config.rms_norm_eps as f32;
        let n = num_tokens as u32;
        let hc = self.hc.as_ref().unwrap();
        let hc_mult = hc.hc_mult as u32;
        // 2026-09-25: Model layer indices, carried on the HC weights.
        // `attn_layer_idx` counts attention layers only, so on a model that
        // interleaves GDN and attention layers it is not the model index.
        let is_first_layer = hc.is_first_model_layer;
        let is_last_layer = hc.is_last_model_layer;
        // 2026-09-25: In a mixed decode+prefill step this chunk's highway rows
        // start at `ctx.hc_row_offset`, after the decode rows.
        let hc_streams = ctx
            .buffers
            .hc_streams()
            .offset(ctx.hc_row_offset * hc.hc_mult * h * 4);
        let post = ctx.buffers.hc_post();
        let comb = ctx.buffers.hc_comb();
        // 2026-09-25: Opt-in (`METRALE_DIAG_V4_ALL_LAYERS`): each diag
        // synchronizes the stream and copies to the host.
        let diag_all =
            std::env::var("METRALE_DIAG_V4_ALL_LAYERS").is_ok_and(|v| v == "1" || v == "true");
        let diag_this = diag_all;

        if is_first_layer {
            ops::hc_expand(
                ctx.gpu,
                self.hc_expand_k,
                hidden,
                hc_streams,
                n,
                h as u32,
                hc_mult,
                stream,
            )?;
        }

        // 2026-09-25: Highway taps (`METRALE_QWEN4EXP_DUMP`). `model_layer`
        // assumes three GDN layers before each attention layer (attention layer
        // k is model layer 4k+3).
        let model_layer = self.attn_layer_idx * 4 + 3;
        crate::layers::ple::dump::tap_highway(
            ctx.gpu,
            hc_streams,
            model_layer,
            "attn_in",
            num_tokens,
            (hc_mult as usize) * h,
            stream,
        );

        ops::hc_pre_site(
            ctx.gpu,
            self.hc_pre_k,
            hc_streams,
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
        if diag_this {
            super::diag_norm(
                ctx.gpu,
                hidden,
                h,
                stream,
                &format!("V4-prefill L{} hc_pre-attn", self.attn_layer_idx),
            );
            super::diag_norm_f32(
                ctx.gpu,
                post,
                (n as usize) * (hc_mult as usize),
                stream,
                &format!("V4-prefill L{} post-attn", self.attn_layer_idx),
            );
            super::diag_norm_f32(
                ctx.gpu,
                comb,
                (n as usize) * (hc_mult as usize) * (hc_mult as usize),
                stream,
                &format!("V4-prefill L{} comb-attn", self.attn_layer_idx),
            );
        }

        let normed = ctx.buffers.norm_output();
        if ops::HcVariant::of(hc).applies_block_input_norm() {
            ops::rms_norm(
                ctx.gpu,
                self.rms_norm_w_k,
                hidden,
                &self.input_norm,
                normed,
                n,
                h as u32,
                eps,
                stream,
            )?;
        } else {
            // 2026-09-25: Low-rank (qwen4_exp): `hc_pre`'s grouped `hc_norm` is this
            // layer's input norm. The checkpoint has no per-layer
            // `input_layernorm`, and a second RMS norm with the loader's
            // ones-filled placeholder would not be an identity. Hand `hc_pre`'s
            // output straight to the block.
            ctx.gpu
                .copy_d2d_async(hidden, normed, num_tokens * h * 2, stream)?;
        }

        // 2026-09-25: QSA indexer ingest: park this chunk's raw indexer keys,
        // projected from the same block input the attention reads.
        if let Some(ref qsa) = self.qsa {
            let st = crate::layers::qwen3_attention::helpers::qsa_seq_state(qsa, state, ctx.gpu)?;
            qsa.prefill_ingest(st, normed, num_tokens, seq_len_start, ctx.gpu, stream)?;
        }

        if batched_meta.is_some() && seq_len_start == 0 {
            anyhow::bail!(
                "prefill_inner_hc: batched mode requires seq_len_start > 0; \
                 got seq_len_start=0."
            );
        }
        let attn_out = if seq_len_start == 0 {
            self.prefill_attention_with_cache_skip(
                state,
                normed,
                num_tokens,
                kv_write_start,
                block_table,
                kv_cache,
                None,
                ctx,
                stream,
            )?
        } else {
            self.prefill_attention_paged(
                state,
                normed,
                num_tokens,
                seq_len_start,
                kv_cache,
                block_table,
                disk_block_ids,
                disk_last_offloaded_per_layer,
                batched_meta,
                kv_write_start,
                ctx,
                stream,
            )?
        };

        if ctx.config.tp_world_size > 1
            && let Some(comm) = ctx.comm
        {
            let bytes = num_tokens * h * 2;
            comm.all_reduce_async(attn_out.0, bytes, stream)?;
        }

        if batched_meta.is_some() && self.high_speed_swap_engaged(kv_cache) {
            anyhow::bail!(
                "prefill_inner_hc: batched mode does not support HSS-engaged layers \
                 (layer {})",
                self.attn_layer_idx
            );
        }
        if self.high_speed_swap_engaged(kv_cache) {
            let nkv = self
                .num_kv_heads_override
                .unwrap_or(ctx.config.num_key_value_heads) as u32;
            let hd = self.head_dim_override.unwrap_or(ctx.config.head_dim) as u32;
            let bs = kv_cache.block_size();
            self.high_speed_swap_offload_new_blocks(
                kv_cache,
                block_table,
                disk_block_ids,
                disk_last_offloaded_per_layer,
                ctx,
                stream,
                nkv,
                hd,
                bs,
            )?;
        }

        if let Some(ref post_norm) = self.post_attn_out_norm {
            ops::rms_norm(
                ctx.gpu,
                self.rms_norm_w_k,
                attn_out,
                post_norm,
                attn_out,
                n,
                h as u32,
                eps,
                stream,
            )?;
        }

        if self.ffn.is_none() {
            ops::hc_post_site(
                ctx.gpu,
                self.hc_post_k,
                hc,
                attn_out,
                hc_streams,
                post,
                comb,
                hc_streams,
                n,
                h as u32,
                stream,
            )?;
            if is_last_layer && let Some(ref head) = hc.head {
                ops::hc_head_site(
                    ctx.gpu,
                    self.hc_head_k,
                    hc_streams,
                    head,
                    hc,
                    hidden,
                    ctx.buffers.hc_lowrank_scratch(),
                    n,
                    h as u32,
                    eps,
                    stream,
                )?;
            }
            return Ok(());
        }

        ops::hc_post_site(
            ctx.gpu,
            self.hc_post_k,
            hc,
            attn_out,
            hc_streams,
            post,
            comb,
            hc_streams,
            n,
            h as u32,
            stream,
        )?;
        if diag_this {
            super::diag_norm_f32(
                ctx.gpu,
                hc_streams,
                h,
                stream,
                &format!("V4-prefill L{} hc_post-attn", self.attn_layer_idx),
            );
            super::diag_norm_f32(
                ctx.gpu,
                hc_streams,
                (n as usize) * (hc_mult as usize) * h,
                stream,
                &format!(
                    "V4-prefill L{} hc_post-attn ALL_STREAMS",
                    self.attn_layer_idx
                ),
            );
        }

        crate::layers::ple::dump::tap_highway(
            ctx.gpu,
            hc_streams,
            model_layer,
            "post_attn",
            num_tokens,
            (hc_mult as usize) * h,
            stream,
        );

        ops::hc_pre_site(
            ctx.gpu,
            self.hc_pre_k,
            hc_streams,
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
        if diag_this {
            super::diag_norm(
                ctx.gpu,
                hidden,
                h,
                stream,
                &format!("V4-prefill L{} hc_pre-ffn", self.attn_layer_idx),
            );
            super::diag_norm_f32(
                ctx.gpu,
                post,
                (n as usize) * (hc_mult as usize),
                stream,
                &format!("V4-prefill L{} post-ffn", self.attn_layer_idx),
            );
            super::diag_norm_f32(
                ctx.gpu,
                comb,
                (n as usize) * (hc_mult as usize) * (hc_mult as usize),
                stream,
                &format!("V4-prefill L{} comb-ffn", self.attn_layer_idx),
            );
        }

        let normed2 = ctx.buffers.norm_output();
        if ops::HcVariant::of(hc).applies_block_input_norm() {
            ops::rms_norm(
                ctx.gpu,
                self.rms_norm_w_k,
                hidden,
                &self.post_attn_norm,
                normed2,
                n,
                h as u32,
                eps,
                stream,
            )?;
        } else {
            // 2026-09-25: Low-rank (qwen4_exp): the FFN site's own `hc_pre` already
            // normed this, as the attention site's did. The checkpoint has no
            // `post_attention_layernorm`.
            ctx.gpu
                .copy_d2d_async(hidden, normed2, num_tokens * h * 2, stream)?;
        }

        self.ffn
            .forward_prefill(normed2, num_tokens, ctx, stream)
            .map_err(|e| anyhow::anyhow!("ffn.forward_prefill (HC) failed: {e}"))?;

        let dense_out = ctx.buffers.moe_output();

        if let Some(ref post_norm) = self.post_ffn_out_norm {
            ops::rms_norm(
                ctx.gpu,
                self.rms_norm_w_k,
                dense_out,
                post_norm,
                dense_out,
                n,
                h as u32,
                eps,
                stream,
            )?;
        }

        ops::hc_post_site(
            ctx.gpu,
            self.hc_post_k,
            hc,
            dense_out,
            hc_streams,
            post,
            comb,
            hc_streams,
            n,
            h as u32,
            stream,
        )?;
        if diag_this {
            super::diag_norm_f32(
                ctx.gpu,
                hc_streams,
                h,
                stream,
                &format!("V4-prefill L{} hc_post-ffn", self.attn_layer_idx),
            );
            super::diag_norm_f32(
                ctx.gpu,
                hc_streams,
                (n as usize) * (hc_mult as usize) * h,
                stream,
                &format!(
                    "V4-prefill L{} hc_post-ffn ALL_STREAMS",
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
                hidden,
                ctx.buffers.hc_lowrank_scratch(),
                n,
                h as u32,
                eps,
                stream,
            )?;
            if diag_this {
                super::diag_norm(
                    ctx.gpu,
                    hidden,
                    (n as usize) * h,
                    stream,
                    &format!("V4-prefill L{} hc_head", self.attn_layer_idx),
                );
            }
        } else if is_last_layer {
            tracing::warn!(target: "metrale_model_layers::layers::qwen3_attention::trait_impl::prefill_inner", "V4-prefill L{}: hc_head SKIPPED (no head weights)",
                self.attn_layer_idx
            );
        }

        Ok(())
    }
}
