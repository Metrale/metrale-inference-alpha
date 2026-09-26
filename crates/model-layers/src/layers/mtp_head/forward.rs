// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-25: The single-row MTP drafter forward, `MtpHead::forward_one`.
//!
//! Owner: model-layers (MTP head).
//! Invariants:
//! - A `forward_one` that returns `Ok` wrote one drafter KV row: `state.seq_len`
//!   grew by 1 and `state.last_pair_key` is `position - 1` (saturating).

use anyhow::Result;
use metrale_gpu_runtime::gpu::DevicePtr;

use super::{MtpHead, MtpProposerState, MtpQuantization, ProjectionWeight};
use crate::layer::ForwardContext;
use crate::layers::mtp_meta::{MTP_META_OFFSET, pack_mtp_attn_meta};
use crate::layers::ops;

mod attend;
mod host_logits;

/// 2026-09-25: L2 norm of a BF16 device buffer, for the `mtp_debug_norms` lever
/// (`METRALE_MTP_DEBUG_NORMS`). A failed copy returns NaN.
fn mtp_dbg_l2(gpu: &dyn metrale_gpu_runtime::gpu::GpuBackend, p: DevicePtr, n: usize) -> f64 {
    let mut b = vec![0u8; n * 2];
    if gpu.copy_d2h(p, &mut b).is_err() {
        return f64::NAN;
    }
    b.chunks_exact(2)
        .map(|c| {
            let f = f32::from_bits((u16::from_le_bytes([c[0], c[1]]) as u32) << 16) as f64;
            f * f
        })
        .sum::<f64>()
        .sqrt()
}

impl MtpHead {
    /// 2026-09-25: Draft one token from `token` and the target hidden row.
    ///
    /// With `draft_embed_target = Some(ptr)`, `embed_from_argmax` writes the
    /// draft's embedding to `ptr` and its id to `draft_token_id_dev`; the
    /// unmasked path then returns 0 and the caller reads the id later with
    /// `read_deferred_draft_token`. With `None`, the unmasked path copies the id
    /// back and returns it. With a `grammar_bitmask`, the argmax runs on the host
    /// and the chosen id is returned. `mtp_multi` calls it for each of its heads.
    pub(crate) fn forward_one(
        &self,
        token: u32,
        target_hidden: DevicePtr,
        position: usize,
        state: &mut MtpProposerState,
        ctx: &ForwardContext,
        stream: u64,
        draft_embed_target: Option<DevicePtr>,
        grammar_bitmask: Option<&[i32]>,
        target_row: bool,
    ) -> Result<u32> {
        let debug_norms = ctx.levers.mtp_debug_norms;
        let h = ctx.config.hidden_size as u32;
        let nq = ctx.config.num_attention_heads as u32;
        let nkv = ctx.config.num_key_value_heads as u32;
        let hd = ctx.config.head_dim as u32;
        let eps = ctx.config.rms_norm_eps as f32;

        let embed_out = ctx.buffers.ssm_qkvz();
        let row_bytes = h as usize * 2;
        let src = self.embed_tokens.weight.offset(token as usize * row_bytes);
        ctx.gpu.copy_d2d_async(src, embed_out, row_bytes, stream)?;

        let normed_embed = ctx.buffers.ssm_deinterleaved();
        ops::rms_norm(
            ctx.gpu,
            self.rms_norm_k,
            embed_out,
            &self.pre_fc_norm_embedding,
            normed_embed,
            1,
            h,
            eps,
            stream,
        )?;

        let normed_hidden = ctx.buffers.ssm_gates();
        // 2026-09-25: `ssm_ba` is the concat destination below, so it can hold the
        // target-final-normed row until the concat overwrites it.
        let target_hidden =
            self.target_postnorm_row(ctx, target_row, target_hidden, ctx.buffers.ssm_ba(), stream)?;
        ops::rms_norm(
            ctx.gpu,
            self.rms_norm_k,
            target_hidden,
            &self.pre_fc_norm_hidden,
            normed_hidden,
            1,
            h,
            eps,
            stream,
        )?;

        let concat_out = ctx.buffers.ssm_ba();
        ops::bf16_concat(
            ctx.gpu,
            self.bf16_concat_k,
            normed_embed,
            normed_hidden,
            concat_out,
            h,
            stream,
        )?;

        if debug_norms {
            ctx.gpu.synchronize(stream).ok();
            tracing::warn!(
                "MTP_DBG s1-embed ||={:.4} s2-n_embed ||={:.4} s2-n_hidden ||={:.4} s3-concat ||={:.4}",
                mtp_dbg_l2(ctx.gpu, embed_out, h as usize),
                mtp_dbg_l2(ctx.gpu, normed_embed, h as usize),
                mtp_dbg_l2(ctx.gpu, normed_hidden, h as usize),
                mtp_dbg_l2(ctx.gpu, concat_out, (h * 2) as usize),
            );
        }

        let hidden = ctx.buffers.hidden_states();
        self.gemv(ctx.gpu, concat_out, &self.fc, hidden, h, h * 2, stream)?;
        if debug_norms {
            ctx.gpu.synchronize(stream).ok();
            tracing::warn!(
                "MTP_DBG s4-fc_hidden ||={:.4}",
                mtp_dbg_l2(ctx.gpu, hidden, h as usize)
            );
        }

        let residual = ctx.buffers.residual();
        ctx.gpu
            .copy_d2d_async(hidden, residual, row_bytes, stream)?;

        let normed = ctx.buffers.norm_output();
        ops::rms_norm(
            ctx.gpu,
            self.rms_norm_k,
            hidden,
            &self.input_layernorm,
            normed,
            1,
            h,
            eps,
            stream,
        )?;

        let q_out = ctx.buffers.qkv_output();
        let q_dim = nq * hd;
        let qg_dim = q_dim * 2;
        let qg_bytes = qg_dim as usize * 2;

        match self.quant {
            MtpQuantization::Nvfp4 => {
                if let ProjectionWeight::Nvfp4(ref w) = self.q_proj {
                    ops::w4a16_gemv_qg(
                        ctx.gpu,
                        self.w4a16_gemv_qg_k,
                        normed,
                        w,
                        q_out,
                        qg_dim,
                        h,
                        nq,
                        hd,
                        stream,
                    )?;
                }
            }
            MtpQuantization::Fp8 | MtpQuantization::Bf16 => {
                self.gemv(ctx.gpu, normed, &self.q_proj, q_out, qg_dim, h, stream)?;
                ops::deinterleave_qg(
                    ctx.gpu,
                    self.deinterleave_qg_k.unwrap(),
                    q_out,
                    1,
                    nq,
                    hd,
                    nq * hd * 2,
                    stream,
                )?;
            }
        }
        let gate_ptr = q_out.offset(q_dim as usize * 2);

        let k_out = q_out.offset(qg_bytes);
        let v_out = k_out.offset((nkv * hd) as usize * 2);

        match self.quant {
            MtpQuantization::Nvfp4 => {
                if let (ProjectionWeight::Nvfp4(kw), ProjectionWeight::Nvfp4(vw)) =
                    (&self.k_proj, &self.v_proj)
                {
                    ops::w4a16_gemv_dual(
                        ctx.gpu,
                        self.w4a16_gemv_dual_k,
                        normed,
                        kw,
                        k_out,
                        vw,
                        v_out,
                        nkv * hd,
                        h,
                        stream,
                    )?;
                }
            }
            MtpQuantization::Fp8 | MtpQuantization::Bf16 => {
                self.gemv(ctx.gpu, normed, &self.k_proj, k_out, nkv * hd, h, stream)?;
                self.gemv(ctx.gpu, normed, &self.v_proj, v_out, nkv * hd, h, stream)?;
            }
        }

        ops::rms_norm(
            ctx.gpu,
            self.rms_norm_k,
            q_out,
            &self.q_norm,
            q_out,
            nq,
            hd,
            eps,
            stream,
        )?;
        ops::rms_norm(
            ctx.gpu,
            self.rms_norm_k,
            k_out,
            &self.k_norm,
            k_out,
            nkv,
            hd,
            eps,
            stream,
        )?;

        let mut kv_cache = self.kv_cache.lock();
        let bs = kv_cache.block_size();
        let blocks_needed = (state.seq_len / bs) + 1;
        while state.block_table.len() < blocks_needed {
            state.block_table.push(kv_cache.alloc_block()?);
        }

        let meta_base = ctx.buffers.scratch().offset(MTP_META_OFFSET);
        let max_blocks = state.block_table.len() as u32;

        let block_idx = state.block_table[state.seq_len / bs];
        let global_slot = (block_idx as i64) * (bs as i64) + ((state.seq_len % bs) as i64);
        let actual_seq_len = (state.seq_len + 1) as i32;

        // 2026-09-25: The metadata grows with the block table, so its bound is the rest
        // of the scratch arena past `MTP_META_OFFSET`; `pack_mtp_attn_meta` fails
        // rather than write past it.
        let meta_buf = pack_mtp_attn_meta(
            position as u32,
            global_slot,
            actual_seq_len,
            &state.block_table,
            ctx.buffers.scratch_bytes().saturating_sub(MTP_META_OFFSET),
        )?;
        ctx.gpu.copy_h2d_async(&meta_buf, meta_base, stream)?;

        ops::rope(
            ctx.gpu,
            self.rope_k,
            q_out,
            k_out,
            meta_base,
            1,
            nq,
            nkv,
            hd,
            ctx.config.rotary_dim() as u32,
            ctx.config.rope_theta as f32,
            stream,
        )?;

        // 2026-09-25: The FP8 KV path runs with unit K and V scales; no scales are
        // computed for this head.
        let kv_stride = nkv * hd;
        let attn_out = ctx.buffers.attn_output();
        let inv_sqrt_d = 1.0f32 / (hd as f32).sqrt();
        self.mtp_attend(
            ctx, &kv_cache, q_out, k_out, v_out, attn_out, meta_base, max_blocks, bs, nq, nkv, hd,
            kv_stride, inv_sqrt_d, stream,
        )?;

        if debug_norms {
            ctx.gpu.synchronize(stream).ok();
            tracing::warn!(
                "MTP_DBG s7-attn_out(pre-gate) ||={:.4}  gate ||={:.4}",
                mtp_dbg_l2(ctx.gpu, attn_out, (nq * hd) as usize),
                mtp_dbg_l2(ctx.gpu, gate_ptr, (nq * hd) as usize)
            );
        }

        ops::sigmoid_gate_mul(
            ctx.gpu,
            self.sigmoid_gate_mul_k,
            attn_out,
            gate_ptr,
            attn_out,
            nq * hd,
            stream,
        )?;

        let o_out = ctx.buffers.norm_output();
        self.gemv(ctx.gpu, attn_out, &self.o_proj, o_out, h, nq * hd, stream)?;

        let normed2 = ctx.buffers.norm_output();
        ops::residual_add_rms_norm(
            ctx.gpu,
            self.residual_add_rms_norm_k,
            hidden,
            o_out,
            &self.post_attn_layernorm,
            normed2,
            residual,
            1,
            h,
            eps,
            stream,
        )?;

        let ffn_out = if self.dense_ffn_generic.is_some() {
            self.dense_ffn_forward_generic(normed2, ctx, stream)?
        } else {
            match self.quant {
                MtpQuantization::Nvfp4 => self
                    .moe_nvfp4
                    .as_ref()
                    .unwrap()
                    .forward(normed2, ctx, stream)?,
                MtpQuantization::Fp8 | MtpQuantization::Bf16 => match self.moe_fp8.as_ref() {
                    // 2026-09-25: The same `MoeLayer` the batched propose runs
                    // grouped, here for one row.
                    Some(moe) => moe.forward(normed2, ctx, stream)?,
                    None => self.moe_forward_generic(normed2, ctx, stream)?,
                },
            }
        };
        ops::residual_add(ctx.gpu, self.residual_add_k, hidden, ffn_out, h, stream)?;

        let final_normed = ctx.buffers.norm_output();
        ops::rms_norm(
            ctx.gpu,
            self.rms_norm_k,
            hidden,
            &self.norm,
            final_normed,
            1,
            h,
            eps,
            stream,
        )?;

        // 2026-09-25: The LM head covers the first `mtp_vocab_size` rows when that is
        // non-zero, else the full vocab.
        let v = if self.mtp_vocab_size > 0 {
            self.mtp_vocab_size.min(ctx.config.vocab_size as u32)
        } else {
            ctx.config.vocab_size as u32
        };
        let logits = ctx.buffers.logits();
        ops::w4a16_decode_gemv(
            ctx.gpu,
            self.w4a16_gemv_k,
            self.w4a16_gemv_sw_k,
            ctx.levers.gemv_sw,
            final_normed,
            &self.lm_head_nvfp4,
            logits,
            v,
            h,
            stream,
        )?;

        // 2026-09-25: With `mtp_debug_norms`, log the L2 norms of the target hidden
        // input, the final-normed row and the logits.
        if debug_norms {
            ctx.gpu.synchronize(stream).ok();
            let bf16_norm = |p: DevicePtr, n: usize| -> f64 {
                let mut b = vec![0u8; n * 2];
                if ctx.gpu.copy_d2h(p, &mut b).is_err() {
                    return -1.0;
                }
                b.chunks_exact(2)
                    .map(|c| {
                        let f =
                            f32::from_bits((u16::from_le_bytes([c[0], c[1]]) as u32) << 16) as f64;
                        f * f
                    })
                    .sum::<f64>()
                    .sqrt()
            };
            let hin = bf16_norm(target_hidden, h as usize);
            tracing::warn!(
                "MTP_DEBUG_NORMS: ||input_hidden||={:.4} ||final_normed||={:.4} ||logits||={:.4}",
                hin,
                bf16_norm(final_normed, h as usize),
                bf16_norm(logits, v as usize)
            );
        }

        // 2026-09-25: Drafter chain confidence (`draft_conf_tau > 0`,
        // `METRALE_MTP_DRAFT_CONF`). Token selection below is unchanged: this
        // copies the logits back and folds the draft's top-1 softmax probability
        // into the running minimum `last_conf_bits`, which `propose` resets. The
        // model's `run_mtp_propose_inner` drops the drafts when it is below tau.
        if ctx.levers.draft_conf_tau > 0.0 {
            self.fold_draft_conf(ctx, logits, v);
        }

        // 2026-09-25: Shadow top-k (`shadow_topk`, `METRALE_MTP_SHADOW_TOPK`, at most
        // 8). Logs this position's top-k ids and softmax probabilities; token
        // selection is unchanged.
        let shadow_k = ctx.levers.shadow_topk;
        if shadow_k > 0 {
            host_logits::log_shadow_topk(ctx, logits, v, shadow_k, position);
        }

        let out_ptr = ctx.buffers.scratch();

        let token_id = if let Some(bitmask) = grammar_bitmask {
            self.grammar_masked_argmax(
                ctx,
                logits,
                v,
                bitmask,
                out_ptr,
                draft_embed_target,
                h,
                position,
                stream,
            )?
        } else {
            ops::argmax_bf16(ctx.gpu, self.argmax_k, logits, out_ptr, v, stream)?;
            if let Some(embed_target) = draft_embed_target {
                ops::embed_from_argmax(
                    ctx.gpu,
                    self.embed_from_argmax_k,
                    out_ptr,
                    self.embed_tokens.weight,
                    embed_target,
                    self.draft_token_id_dev,
                    h,
                    stream,
                )?;
                0u32
            } else {
                let mut buf = [0u8; 4];
                ctx.gpu.copy_d2h(out_ptr, &mut buf)?;
                u32::from_le_bytes(buf)
            }
        };

        state.seq_len += 1;
        // 2026-09-25: This call wrote the drafter row for sequence key `position - 1`;
        // the catch-up path reads `last_pair_key` to find missing rows.
        state.last_pair_key = Some(position.saturating_sub(1));
        Ok(token_id)
    }
}
