// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-26: `decode_inner_hc`, the single-token decode of a hyper-connection layer.
//! `decode_inner` calls it when the layer has `hc` weights.
//!
//! Owner: model-layers (qwen3 attention).
//! Invariants:
//! - The `diag_norm` diagnostics never run during graph capture: `diag_this` includes
//!   `!ctx.graph_capture`.

use anyhow::Result;
use metrale_cache::kv_cache::PagedKvCache;
use metrale_gpu_runtime::gpu::DevicePtr;

use super::super::super::Qwen3AttentionLayer;
use crate::layer::{ForwardContext, LayerState};
use crate::layers::ops;

impl Qwen3AttentionLayer {
    /// 2026-09-25: Single-token decode for a hyper-connection layer. The
    /// persistent state is `hc_streams` (`[1, hc_mult, H]`, FP32); `hidden` is
    /// single-stream scratch.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn decode_inner_hc(
        &self,
        hidden: DevicePtr,
        _residual: DevicePtr,
        state: &mut dyn LayerState,
        kv_cache: &mut PagedKvCache,
        seq_len: usize,
        block_table: &mut Vec<u32>,
        disk_block_ids: &mut Vec<u32>,
        disk_last_offloaded_per_layer: &mut Vec<u32>,
        ctx: &ForwardContext,
        stream: u64,
    ) -> Result<()> {
        let h = ctx.config.hidden_size;
        let eps = ctx.config.rms_norm_eps as f32;
        let hc = self.hc.as_ref().unwrap();
        let hc_mult = hc.hc_mult as u32;
        // 2026-09-25: First and last model layer come from the loader's
        // `HcWeights`, not from `attn_layer_idx`, which counts attention layers
        // only and so differs from the model index on a hybrid model.
        let is_first_layer = hc.is_first_model_layer;
        let is_last_layer = hc.is_last_model_layer;
        let hc_streams = ctx.buffers.hc_streams();
        let post = ctx.buffers.hc_post();
        let comb = ctx.buffers.hc_comb();
        // 2026-09-25: Opt-in (`METRALE_DIAG_V4_ALL_LAYERS`), never during capture:
        // `diag_norm` synchronises and copies to the host and ignores the
        // errors, so inside a capture it would invalidate the graph silently.
        let diag_all =
            std::env::var("METRALE_DIAG_V4_ALL_LAYERS").is_ok_and(|v| v == "1" || v == "true");
        let diag_this = diag_all && !ctx.graph_capture;

        // 2026-09-25: The first model layer expands the single stream into
        // `hc_mult` copies.
        if is_first_layer {
            ops::hc_expand(
                ctx.gpu,
                self.hc_expand_k,
                hidden,
                hc_streams,
                1,
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
            hidden,
            post,
            comb,
            ctx.buffers.hc_lowrank_scratch(),
            1,
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
                &format!("V4-decode L{} hc_pre-attn", self.attn_layer_idx),
            );
            super::diag_norm_f32(
                ctx.gpu,
                post,
                hc_mult as usize,
                stream,
                &format!("V4-decode L{} post-attn", self.attn_layer_idx),
            );
            super::diag_norm_f32(
                ctx.gpu,
                comb,
                (hc_mult as usize) * (hc_mult as usize),
                stream,
                &format!("V4-decode L{} comb-attn", self.attn_layer_idx),
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
                1,
                h as u32,
                eps,
                stream,
            )?;
        } else {
            // 2026-09-25: `hc_pre`'s `hc_norm` is this layer's input norm here; see
            // `prefill_inner.rs`.
            ctx.gpu.copy_d2d_async(hidden, normed, h * 2, stream)?;
        }

        let attn_out = self.attention_forward(
            state,
            normed,
            seq_len,
            block_table,
            disk_block_ids,
            disk_last_offloaded_per_layer,
            kv_cache,
            ctx,
            stream,
        )?;

        if ctx.config.tp_world_size > 1
            && let Some(comm) = ctx.comm
        {
            let bytes = h * 2;
            comm.all_reduce_async(attn_out.0, bytes, stream)?;
        }

        if let Some(ref post_norm) = self.post_attn_out_norm {
            ops::rms_norm(
                ctx.gpu,
                self.rms_norm_w_k,
                attn_out,
                post_norm,
                attn_out,
                1,
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
                1,
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
                    1,
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
            1,
            h as u32,
            stream,
        )?;
        if diag_this {
            super::diag_norm_f32(
                ctx.gpu,
                hc_streams,
                h,
                stream,
                &format!("V4-decode L{} hc_post-attn", self.attn_layer_idx),
            );
            super::diag_norm_f32(
                ctx.gpu,
                hc_streams,
                (hc_mult as usize) * (h),
                stream,
                &format!(
                    "V4-decode L{} hc_post-attn ALL_STREAMS",
                    self.attn_layer_idx
                ),
            );
        }

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
            1,
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
                &format!("V4-decode L{} hc_pre-ffn", self.attn_layer_idx),
            );
            super::diag_norm_f32(
                ctx.gpu,
                post,
                hc_mult as usize,
                stream,
                &format!("V4-decode L{} post-ffn", self.attn_layer_idx),
            );
            super::diag_norm_f32(
                ctx.gpu,
                comb,
                (hc_mult as usize) * (hc_mult as usize),
                stream,
                &format!("V4-decode L{} comb-ffn", self.attn_layer_idx),
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
                1,
                h as u32,
                eps,
                stream,
            )?;
        } else {
            ctx.gpu.copy_d2d_async(hidden, normed2, h * 2, stream)?;
        }

        let ffn_out = self.ffn.forward(normed2, ctx, stream)?;

        if let Some(ref post_norm) = self.post_ffn_out_norm {
            ops::rms_norm(
                ctx.gpu,
                self.rms_norm_w_k,
                ffn_out,
                post_norm,
                ffn_out,
                1,
                h as u32,
                eps,
                stream,
            )?;
        }

        if let Some(scalar) = self.layer_scalar {
            self.apply_layer_scalar(ctx.gpu, ffn_out, h, scalar, stream)?;
        }

        ops::hc_post_site(
            ctx.gpu,
            self.hc_post_k,
            hc,
            ffn_out,
            hc_streams,
            post,
            comb,
            hc_streams,
            1,
            h as u32,
            stream,
        )?;
        if diag_this {
            super::diag_norm_f32(
                ctx.gpu,
                hc_streams,
                h,
                stream,
                &format!("V4-decode L{} hc_post-ffn", self.attn_layer_idx),
            );
            super::diag_norm_f32(
                ctx.gpu,
                hc_streams,
                (hc_mult as usize) * (h),
                stream,
                &format!("V4-decode L{} hc_post-ffn ALL_STREAMS", self.attn_layer_idx),
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
                1,
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
                    &format!("V4-decode L{} hc_head", self.attn_layer_idx),
                );
            }
        } else if is_last_layer {
            tracing::warn!(
                target: "metrale_model_layers::layers::qwen3_attention::trait_impl::decode_inner",
                "V4-decode L{}: hc_head SKIPPED (no head weights)",
                self.attn_layer_idx
            );
        }

        Ok(())
    }
}
