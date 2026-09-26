// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-25: The N-token prefill body of `Qwen3AttentionLayer`: input norm, attention, the FFN or MoE sublayer and the residual adds, plus the hyper-connection (mHC) variant.
//!
//! Owner: model-layers (qwen3 attention).
//! Invariants:
//! - A batched call (`batched_meta` is `Some`) never takes the contiguous
//!   first-chunk attention: at `seq_len_start == 0` it returns an error unless
//!   `prefill_batched_first_chunk_enabled()` (always, on the mHC body).
//! - A batched call on a layer with high-speed swap engaged returns an error.

use anyhow::Result;
use metrale_cache::kv_cache::PagedKvCache;
use metrale_gpu_runtime::gpu::DevicePtr;

use super::super::Qwen3AttentionLayer;
use super::{diag_norm, diag_norm_f32};
use crate::layer::{BatchedAttnMetadata, ForwardContext, LayerState};
use crate::layers::ops;

mod ffn_residual;
mod hc;

impl Qwen3AttentionLayer {
    #[allow(clippy::too_many_arguments)]
    pub(super) fn prefill_inner(
        &self,
        hidden: DevicePtr,
        residual: DevicePtr,
        num_tokens: usize,
        state: &mut dyn LayerState,
        kv_cache: &mut PagedKvCache,
        seq_len_start: usize,
        block_table: &mut Vec<u32>,
        disk_block_ids: &mut Vec<u32>,
        disk_last_offloaded_per_layer: &mut Vec<u32>,
        kv_write_start: usize,
        // 2026-09-25: When `Some`, the attention step runs the batched paged
        // kernels: `num_tokens` is `batched_meta.total_tokens`, and
        // hidden/residual hold the streams' rows back to back, at the prefix
        // sums in `cu_seqlens` (or `b * chunk_len` when it is null).
        batched_meta: Option<&BatchedAttnMetadata>,
        ctx: &ForwardContext,
        stream: u64,
    ) -> Result<()> {
        // 2026-09-25: Hyper-connection (mHC) layers (DeepSeek-V4, qwen4_exp) take
        // their own body.
        if self.hc.is_some() {
            return self.prefill_inner_hc(
                hidden,
                residual,
                num_tokens,
                state,
                kv_cache,
                seq_len_start,
                block_table,
                disk_block_ids,
                disk_last_offloaded_per_layer,
                kv_write_start,
                batched_meta,
                ctx,
                stream,
            );
        }

        let h = ctx.config.hidden_size;
        let eps = ctx.config.rms_norm_eps as f32;
        let n = num_tokens as u32;
        let bf16 = 2usize;

        // 2026-09-25: `METRALE_OP_DUMP` hook: the layer's input hidden state (last token).
        if num_tokens > 0 {
            super::super::op_dump::dump_bf16(
                ctx.gpu,
                hidden,
                (num_tokens - 1) * h * bf16,
                h,
                self.attn_layer_idx,
                "input_norm_in",
                stream,
            )?;
        }

        let normed = ctx.buffers.norm_output();
        ops::rms_norm_residual(
            ctx.gpu,
            self.rms_norm_residual_k,
            hidden,
            &self.input_norm,
            normed,
            residual,
            n,
            h as u32,
            eps,
            stream,
        )
        .map_err(|e| anyhow::anyhow!("rms_norm_residual failed: {e}"))?;
        // 2026-09-25: `METRALE_OP_DUMP` hook: the input-norm output (last token),
        // the Q/K/V projections' input.
        if num_tokens > 0 {
            super::super::op_dump::dump_bf16(
                ctx.gpu,
                normed,
                (num_tokens - 1) * h * bf16,
                h,
                self.attn_layer_idx,
                "input_norm_out",
                stream,
            )?;
        }

        let is_mistral_diag = ctx.profile
            && ctx.config.model_type == "mistral"
            && (self.attn_layer_idx == 0 || self.attn_layer_idx == 35);
        if is_mistral_diag {
            diag_norm(
                ctx.gpu,
                hidden,
                h,
                stream,
                &format!("L{} hidden_in", self.attn_layer_idx),
            );
            diag_norm(
                ctx.gpu,
                normed,
                h,
                stream,
                &format!("L{} normed", self.attn_layer_idx),
            );
        }

        // 2026-09-25: Batched mode runs the paged path; at a first chunk
        // (`seq_len_start == 0`) it is refused unless
        // `prefill_batched_first_chunk_enabled()`.
        let allow_batched_first_chunk =
            batched_meta.is_some() && crate::layers::ops::prefill_batched_first_chunk_enabled();
        if batched_meta.is_some() && seq_len_start == 0 && !allow_batched_first_chunk {
            anyhow::bail!(
                "prefill_inner: batched mode requires seq_len_start > 0 (paged path); \
                 got seq_len_start=0. Caller must fall back to per-stream for this chunk."
            );
        }
        let attn_out = if seq_len_start == 0 && !allow_batched_first_chunk {
            // 2026-09-25: First chunk of a single stream: attention over this
            // chunk's contiguous Q/K/V, writing K/V to the cache from
            // `kv_write_start`.
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
            // 2026-09-25: Later chunks, and batched first chunks when enabled:
            // paged attention over the cache. With `batched_meta` it runs the
            // batched kernels over each stream's `block_table_ptrs`.
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

        // 2026-09-25: Under tensor parallelism (`tp_world_size > 1` with a
        // communicator), sum the attention output across ranks.
        if ctx.config.tp_world_size > 1
            && let Some(comm) = ctx.comm
        {
            let bytes = num_tokens * h * 2;
            let _t0 = if ctx.profile {
                ctx.gpu.synchronize(stream)?;
                Some(std::time::Instant::now())
            } else {
                None
            };
            comm.all_reduce_async(attn_out.0, bytes, stream)?;
            if let Some(t0) = _t0 {
                ctx.gpu.synchronize(stream)?;
                tracing::info!(
                    "  TP allreduce (attn out) N={} L{:02}: {}µs",
                    num_tokens,
                    self.attn_layer_idx,
                    t0.elapsed().as_micros(),
                );
            }
        }

        // 2026-09-25: High-speed swap: after prefill writes K/V, copy every
        // block that has no disk copy yet to disk. Batched mode refuses a layer
        // with it engaged.
        if batched_meta.is_some() && self.high_speed_swap_engaged(kv_cache) {
            anyhow::bail!(
                "prefill_inner: batched mode does not support HSS-engaged layers \
                 (layer {}). Caller should fall back to per-stream for this chunk.",
                self.attn_layer_idx
            );
        }
        if self.high_speed_swap_engaged(kv_cache) {
            let nq = self
                .num_q_heads_override
                .unwrap_or(ctx.config.num_attention_heads) as u32;
            let nkv = self
                .num_kv_heads_override
                .unwrap_or(ctx.config.num_key_value_heads) as u32;
            let hd = self.head_dim_override.unwrap_or(ctx.config.head_dim) as u32;
            let bs = kv_cache.block_size();
            let _ = nq;
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
            let _ = nq;
        }

        if is_mistral_diag {
            diag_norm(
                ctx.gpu,
                attn_out,
                h,
                stream,
                &format!("L{} attn_out", self.attn_layer_idx),
            );
        }

        // 2026-09-25: Gemma-4 `post_attention_layernorm`, applied to the attention
        // output before the residual add.
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
            ops::residual_add(
                ctx.gpu,
                self.residual_add_k,
                hidden,
                attn_out,
                (num_tokens * h) as u32,
                stream,
            )?;
            return Ok(());
        }

        ops::residual_add_rms_norm(
            ctx.gpu,
            self.residual_add_rms_norm_k,
            hidden,
            attn_out,
            &self.post_attn_norm,
            ctx.buffers.norm_output(),
            residual,
            n,
            h as u32,
            eps,
            stream,
        )
        .map_err(|e| anyhow::anyhow!("residual_add_rms_norm failed: n={n} h={h}: {e}"))?;
        // 2026-09-25: `METRALE_OP_DUMP` hook: the post-attention norm output (last
        // token), the FFN's input.
        if num_tokens > 0 {
            super::super::op_dump::dump_bf16(
                ctx.gpu,
                ctx.buffers.norm_output(),
                (num_tokens - 1) * h * bf16,
                h,
                self.attn_layer_idx,
                "post_attn_norm_out",
                stream,
            )?;
        }

        // 2026-09-25: `METRALE_PREFILL_HOST_TIMING=1`: host wall-clock time of this
        // layer's FFN half, taken with no synchronize.
        let t_ffn = (std::env::var("METRALE_PREFILL_HOST_TIMING").as_deref() == Ok("1"))
            .then(std::time::Instant::now);
        // 2026-09-25: LongCat shortcut MoE (producer): runs before the dense FFN
        // (both write `moe_output`), folds the zero experts in, and stashes the
        // result in the carry buffer.
        if let (Some(moe_ffn), Some((carry, cap))) = (&self.moe_ffn, self.shortcut_carry_out)
            && self.pre_moe_norm.is_none()
        {
            anyhow::ensure!(
                num_tokens <= cap,
                "shortcut carry capacity {cap} < prefill chunk {num_tokens}"
            );
            moe_ffn
                .forward_prefill(ctx.buffers.norm_output(), num_tokens, ctx, stream)
                .map_err(|e| anyhow::anyhow!("shortcut moe forward_prefill failed: {e}"))?;
            let moe_out = ctx.buffers.moe_output();
            if let crate::layers::FfnComponent::Moe(m) = moe_ffn {
                m.apply_zero_expert(
                    moe_out,
                    ctx.buffers.norm_output(),
                    num_tokens as u32,
                    ctx,
                    stream,
                )?;
            }
            // 2026-09-25: `METRALE_OP_DUMP` hook: the shortcut MoE output (zero
            // experts folded in), captured before the dense FFN reuses this
            // buffer. Not the same as "moe_out" below, the dense FFN output.
            if num_tokens > 0 {
                super::super::op_dump::dump_bf16(
                    ctx.gpu,
                    moe_out,
                    (num_tokens - 1) * h * bf16,
                    h,
                    self.attn_layer_idx,
                    "shortcut_moe_out",
                    stream,
                )?;
            }
            ctx.gpu
                .copy_d2d_async(moe_out, carry, num_tokens * h * 2, stream)?;
        }
        self.ffn
            .forward_prefill(ctx.buffers.norm_output(), num_tokens, ctx, stream)
            .map_err(|e| anyhow::anyhow!("ffn.forward_prefill failed: {e}"))?;
        if let Some(t) = t_ffn {
            crate::layers::qwen3_attention::add_ffn_host_us(t.elapsed().as_micros() as u64);
        }

        let dense_out = ctx.buffers.moe_output();
        // 2026-09-25: `METRALE_OP_DUMP` hook: the FFN output (last token), before
        // any post-FFN norm or residual add.
        if num_tokens > 0 {
            super::super::op_dump::dump_bf16(
                ctx.gpu,
                dense_out,
                (num_tokens - 1) * h * bf16,
                h,
                self.attn_layer_idx,
                "moe_out",
                stream,
            )?;
        }

        if is_mistral_diag {
            diag_norm(
                ctx.gpu,
                dense_out,
                h,
                stream,
                &format!("L{} moe_out", self.attn_layer_idx),
            );
        }

        self.prefill_ffn_residual(hidden, dense_out, num_tokens, n, h, eps, ctx, stream)?;

        // 2026-09-25: Gemma-4 `layer_scalar`: scale the whole hidden state at the
        // end of the layer.
        if let Some(scalar) = self.layer_scalar {
            self.apply_layer_scalar(ctx.gpu, hidden, num_tokens * h, scalar, stream)?;
        }

        if is_mistral_diag {
            diag_norm(
                ctx.gpu,
                hidden,
                h,
                stream,
                &format!("L{} residual", self.attn_layer_idx),
            );
        }

        Ok(())
    }
}
