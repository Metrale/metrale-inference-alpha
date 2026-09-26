// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-25: Batched MTP drafter context prefill: one drafter KV row per
//! prompt position, written before the first propose.
//!
//! The drafter is a single decoder layer, so its K/V at position i depend
//! only on its input `x_i = fc(concat(norm(embed(t_{i+1})), norm(hidden_i)))`
//! and the position, not on its own attention outputs. The pass therefore
//! runs no attention: embedding gather, norms, concat, fc, input_layernorm,
//! k/v projections, k_norm, RoPE and `reshape_and_cache`, over chunks of
//! [`PREFILL_CHUNK`] rows. Drafter row i pairs `embed(t_{i+1})` with
//! `hidden_i` at RoPE position `i + 1` (pair key i).
//!
//! Owner: model-layers (MTP head).
//! Invariants:
//! - Rows are written only when `row_base` equals the drafter's `seq_len`;
//!   otherwise the pass writes nothing and returns 0.
//! - Only a head with BF16 fc/k/v, BF16 KV and a resolved `dense_gemm_bf16`
//!   runs the pass; any other head returns 0.

use anyhow::Result;
use metrale_gpu_runtime::gpu::DevicePtr;

use super::{MtpHead, MtpProposerState, ProjectionWeight};
use crate::layer::ForwardContext;
use crate::layers::ops;
use crate::speculative::ProposerState;

/// 2026-09-25: Rows per batched pass, and the row count of the
/// `MtpPrefillScratch` buffers.
pub(crate) const PREFILL_CHUNK: usize = 512;

impl MtpHead {
    /// 2026-09-25: Batch-prefill the drafter KV over `prompt_tokens` from the
    /// per-position target hiddens. Returns the rows written (P - 1), or 0
    /// when unsupported or already prefilled.
    pub(crate) fn prefill_drafter_impl(
        &self,
        prompt_tokens: &[u32],
        hiddens: DevicePtr,
        state: &mut dyn ProposerState,
        ctx: &ForwardContext,
        stream: u64,
    ) -> Result<usize> {
        self.drafter_rows_impl(prompt_tokens, hiddens, 0, 1, state, ctx, stream)
    }

    /// 2026-09-25: Append drafter rows at KV slots `row_base..` with RoPE
    /// positions `pos_base..`: row r pairs `embed(prompt_tokens[r + 1])` with
    /// `hiddens` row r at RoPE `pos_base + r`. Slots and positions are
    /// separate arguments because a drafter row's slot and its sequence
    /// position can differ (`MtpProposerState::last_pair_key`).
    /// `row_base = 0, pos_base = 1` is the whole-prompt prefill;
    /// `catchup_drafter` appends at `row_base = seq_len`. Returns the rows
    /// written, or 0 when nothing was written.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn drafter_rows_impl(
        &self,
        prompt_tokens: &[u32],
        hiddens: DevicePtr,
        row_base: usize,
        pos_base: usize,
        state: &mut dyn ProposerState,
        ctx: &ForwardContext,
        stream: u64,
    ) -> Result<usize> {
        let mtp_state = match state.as_any_mut().downcast_mut::<MtpProposerState>() {
            Some(s) => s,
            None => return Ok(0),
        };
        // 2026-09-25: Rows append only at the drafter's current length; any
        // other `row_base` would leave a hole or overwrite live rows.
        if mtp_state.seq_len != row_base || prompt_tokens.len() < 2 {
            return Ok(0);
        }
        let scratch = match self.prefill_scratch.as_ref() {
            Some(s) => s,
            None => return Ok(0),
        };
        let (fc_w, k_w, v_w) = match (&self.fc, &self.k_proj, &self.v_proj) {
            (ProjectionWeight::Bf16(fc), ProjectionWeight::Bf16(k), ProjectionWeight::Bf16(v))
                if self.kv_bf16 && self.dense_gemm_k.0 != 0 =>
            {
                (fc, k, v)
            }
            _ => {
                if ctx.stats.once("log:mtp_prefill_unsupported") {
                    tracing::warn!(
                        "MTP drafter context: the batched drafter prefill supports \
                         the BF16 MTP head (--mtp-quantization bf16) with BF16 KV \
                         only; continuing WITHOUT drafter context prefill."
                    );
                }
                return Ok(0);
            }
        };

        let t0 = std::time::Instant::now();
        let h = ctx.config.hidden_size;
        let nq = ctx.config.num_attention_heads as u32;
        let nkv = ctx.config.num_key_value_heads as u32;
        let hd = ctx.config.head_dim as u32;
        let eps = ctx.config.rms_norm_eps as f32;
        let kv_dim = (nkv * hd) as usize;
        let bf16 = 2usize;
        let rows_total = prompt_tokens.len() - 1;

        let mut kv_cache = self.kv_cache.lock();
        let bs = kv_cache.block_size();
        let blocks_needed = (row_base + rows_total - 1) / bs + 1;
        while mtp_state.block_table.len() < blocks_needed {
            mtp_state.block_table.push(kv_cache.alloc_block()?);
        }

        // 2026-09-25: `METRALE_MTP_PREFILL_PROFILE=1` logs per-phase wall time.
        // Each profiled phase synchronizes the stream, which slows the pass.
        let profile = std::env::var("METRALE_MTP_PREFILL_PROFILE").ok().as_deref() == Some("1");
        let mut t_embed = 0f64;
        let mut t_concat = 0f64;
        let mut t_rest = 0f64;
        let mut t_fc = 0f64;
        let mut t_kv = 0f64;
        macro_rules! phase {
            ($acc:expr, $body:block) => {{
                // 2026-09-25: The immediately-invoked closures scope `?` to
                // `$body`, so the time is recorded before the error propagates;
                // the pattern trips `redundant_closure_call`, allowed here.
                #[allow(clippy::redundant_closure_call)]
                {
                    if profile {
                        let s = std::time::Instant::now();
                        let r = (|| -> Result<()> { $body })();
                        ctx.gpu.synchronize(stream)?;
                        $acc += s.elapsed().as_secs_f64() * 1e3;
                        r?;
                    } else {
                        (|| -> Result<()> { $body })()?;
                    }
                }
            }};
        }

        let mut done = 0usize;
        while done < rows_total {
            let c = (rows_total - done).min(PREFILL_CHUNK);

            phase!(t_embed, {
                for r in 0..c {
                    let tok = prompt_tokens[done + r + 1] as usize;
                    self_copy_embed_row(self, ctx, tok, scratch.embed, r, h, stream)?;
                }
                Ok(())
            });

            ops::rms_norm(
                ctx.gpu,
                self.rms_norm_k,
                scratch.embed,
                &self.pre_fc_norm_embedding,
                scratch.normed_embed,
                c as u32,
                h as u32,
                eps,
                stream,
            )?;
            // 2026-09-25: `scratch.concat` is not written until the concat
            // below, so it can hold the target-final-normed rows until then.
            let hidden_rows = self.target_postnorm_rows(
                ctx,
                true,
                hiddens.offset(done * h * bf16),
                scratch.concat,
                c as u32,
                stream,
            )?;
            ops::rms_norm(
                ctx.gpu,
                self.rms_norm_k,
                hidden_rows,
                &self.pre_fc_norm_hidden,
                scratch.normed_hidden,
                c as u32,
                h as u32,
                eps,
                stream,
            )?;

            phase!(t_concat, {
                for r in 0..c {
                    ops::bf16_concat(
                        ctx.gpu,
                        self.bf16_concat_k,
                        scratch.normed_embed.offset(r * h * bf16),
                        scratch.normed_hidden.offset(r * h * bf16),
                        scratch.concat.offset(r * 2 * h * bf16),
                        h as u32,
                        stream,
                    )?;
                }
                Ok(())
            });

            phase!(t_fc, {
                ops::dense_gemm(
                    ctx.gpu,
                    self.dense_gemm_k,
                    scratch.concat,
                    fc_w,
                    scratch.fc_out,
                    c as u32,
                    h as u32,
                    (2 * h) as u32,
                    stream,
                )
            });
            ops::rms_norm(
                ctx.gpu,
                self.rms_norm_k,
                scratch.fc_out,
                &self.input_layernorm,
                scratch.normed2,
                c as u32,
                h as u32,
                eps,
                stream,
            )?;

            // 2026-09-25: No Q projection: the pass runs no attention.
            phase!(t_kv, {
                ops::dense_gemm(
                    ctx.gpu,
                    self.dense_gemm_k,
                    scratch.normed2,
                    k_w,
                    scratch.k_out,
                    c as u32,
                    nkv * hd,
                    h as u32,
                    stream,
                )?;
                ops::dense_gemm(
                    ctx.gpu,
                    self.dense_gemm_k,
                    scratch.normed2,
                    v_w,
                    scratch.v_out,
                    c as u32,
                    nkv * hd,
                    h as u32,
                    stream,
                )
            });
            if !self.k_norm.weight.is_null() {
                ops::rms_norm(
                    ctx.gpu,
                    self.rms_norm_k,
                    scratch.k_out,
                    &self.k_norm,
                    scratch.k_out,
                    c as u32 * nkv,
                    hd,
                    eps,
                    stream,
                )?;
            }

            let positions: Vec<u32> = (0..c).map(|r| (pos_base + done + r) as u32).collect();
            // 2026-09-25: SAFETY: `positions` is `(0..c).map(..).collect()`, so
            // its length is `c` and all `c` elements are initialised, and
            // `c * 4 == positions.len() * size_of::<u32>()`: the span is exactly
            // the Vec's buffer. Shared borrow only.
            let pos_bytes =
                unsafe { std::slice::from_raw_parts(positions.as_ptr() as *const u8, c * 4) };
            ctx.gpu.copy_h2d_async(pos_bytes, scratch.pos_dev, stream)?;
            let slots: Vec<i64> = (0..c)
                .map(|r| {
                    let i = row_base + done + r;
                    (mtp_state.block_table[i / bs] as i64) * (bs as i64) + (i % bs) as i64
                })
                .collect();
            // 2026-09-25: SAFETY: `slots` is `(0..c).map(..).collect()`, so its
            // length is `c` with all `c` elements initialised, and
            // `c * 8 == slots.len() * size_of::<i64>()`: the span is exactly the
            // Vec's buffer. Shared borrow only.
            let slot_bytes =
                unsafe { std::slice::from_raw_parts(slots.as_ptr() as *const u8, c * 8) };
            ctx.gpu
                .copy_h2d_async(slot_bytes, scratch.slot_dev, stream)?;

            ops::rope(
                ctx.gpu,
                self.rope_k,
                scratch.q_scratch,
                scratch.k_out,
                scratch.pos_dev,
                c as u32,
                nq,
                nkv,
                hd,
                ctx.config.rotary_dim() as u32,
                ctx.config.rope_theta as f32,
                stream,
            )?;

            ops::reshape_and_cache(
                ctx.gpu,
                self.reshape_cache_k,
                scratch.k_out,
                scratch.v_out,
                kv_cache.k_pool_ptr(self.attn_layer_idx),
                kv_cache.v_pool_ptr(self.attn_layer_idx),
                scratch.slot_dev,
                c as u32,
                nkv,
                hd,
                bs as u32,
                kv_dim as u32,
                kv_dim as u32,
                kv_cache.cache_stride() as u64,
                stream,
            )?;
            // 2026-09-25: Synchronize before `positions` and `slots` drop: they
            // are the sources of the async H2D copies above.
            let t_sync = std::time::Instant::now();
            ctx.gpu.synchronize(stream)?;
            if profile {
                t_rest += t_sync.elapsed().as_secs_f64() * 1e3;
            }

            done += c;
        }

        mtp_state.seq_len = row_base + rows_total;
        // 2026-09-25: The last row has RoPE `pos_base + rows_total - 1`, which
        // is its pair key + 1.
        mtp_state.last_pair_key = Some(pos_base + rows_total - 2);
        tracing::info!(
            "MTP drafter prefill: {} positions ({} prompt tokens) in {:.1} ms",
            rows_total,
            prompt_tokens.len(),
            t0.elapsed().as_secs_f64() * 1e3,
        );
        if profile {
            tracing::info!(
                "MTP drafter prefill PROFILE: embed_loop={t_embed:.1} ms \
                 concat_loop={t_concat:.1} ms fc_gemm={t_fc:.1} ms kv_gemm={t_kv:.1} ms \
                 tail_sync={t_rest:.1} ms \
                 (rows={rows_total}, chunk={PREFILL_CHUNK})"
            );
        }
        Ok(rows_total)
    }
}

/// 2026-09-25: Copy the embedding row of `token` into row `r` of `dst`.
fn self_copy_embed_row(
    head: &MtpHead,
    ctx: &ForwardContext,
    token: usize,
    dst: DevicePtr,
    r: usize,
    h: usize,
    stream: u64,
) -> Result<()> {
    let row_bytes = h * 2;
    let src = head.embed_tokens.weight.offset(token * row_bytes);
    ctx.gpu
        .copy_d2d_async(src, dst.offset(r * row_bytes), row_bytes, stream)
}
