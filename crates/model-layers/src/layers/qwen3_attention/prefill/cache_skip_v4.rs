// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-25: The DeepSeek-V4 cache-skip prefill, reached only for MLA
//! layers with `o_lora_rank > 0`: Q latent projection with a per-head unit RMS
//! norm, one KV projection used as both K and V, interleaved RoPE, windowed
//! raw attention with a per-head sink (plus a compressed-KV arm on compressor
//! layers), de-rotation of the output, and a grouped low-rank O projection.
//!
//! Owner: model-layers (attention).
//! Invariants: none beyond the types.

use anyhow::Result;
use metrale_cache::kv_cache::PagedKvCache;
use metrale_gpu_runtime::gpu::DevicePtr;

use super::super::Qwen3AttentionLayer;
use crate::layer::ForwardContext;
use crate::layers::ops;

// 2026-09-25: Both attention kernels limit the raw arm to the last
// `V4_WINDOW` keys, on every layer.
pub(super) const V4_WINDOW: u32 = 128;

impl Qwen3AttentionLayer {
    pub(super) fn prefill_attention_cache_skip_v4(
        &self,
        kv_cache: &mut PagedKvCache,
        ctx: &ForwardContext,
        args: &super::cache_skip_mla::CacheSkipMlaArgs,
    ) -> Result<DevicePtr> {
        let super::cache_skip_mla::CacheSkipMlaArgs {
            normed,
            num_tokens: _,
            n,
            h,
            nq,
            nkv,
            hd: _,
            kv_dim: _,
            eps,
            bf16: _,
            stream,
        } = *args;
        let mla = self.mla.as_ref().expect("V4-Flash prefill requires MLA");
        let meta = ctx
            .attn_metadata
            .expect("V4-Flash prefill requires metadata");

        let nope = mla.nope as u32;
        let rope = mla.rope as u32;
        let kv_lora = mla.kv_lora_rank as u32;
        let _v_dim = mla.v_dim as u32;
        let q_lora = mla.q_lora_rank as u32;
        let o_lora = mla.o_lora_rank as u32;
        let mla_cache_dim = kv_lora + rope;
        let hd_mla = nope + rope;
        let use_tc = self.dense_gemm_tc_k.0 != 0;
        let diag_all =
            std::env::var("METRALE_DIAG_V4_ALL_LAYERS").is_ok_and(|v| v == "1" || v == "true");
        let diag_this = self.attn_layer_idx == 0 || diag_all;

        // 2026-09-25: Diagnostics, on layer 0 or on every layer with
        // `METRALE_DIAG_V4_ALL_LAYERS`: the first token of `normed` holding a
        // non-finite value.
        if diag_this {
            let _ = ctx.gpu.synchronize(stream);
            let hh = h as usize;
            let mut buf = vec![0u8; (n as usize) * hh * 2];
            if ctx.gpu.copy_d2h(normed, &mut buf).is_ok() {
                let mut bad_tok = -1i64;
                for t in 0..n as usize {
                    let off = t * hh * 2;
                    if (0..hh).any(|i| {
                        let c = &buf[off + i * 2..off + i * 2 + 2];
                        let v = f32::from_bits((u16::from_le_bytes([c[0], c[1]]) as u32) << 16);
                        !v.is_finite()
                    }) {
                        bad_tok = t as i64;
                        break;
                    }
                }
                tracing::info!(
                    "DIAG V4-prefill L{} NORMED first non-finite (nan/inf) token = {}",
                    self.attn_layer_idx,
                    bad_tok
                );
            }
        }

        let (q_latent, q_full) = self.cache_skip_v4_q_proj(
            ctx, mla, normed, n, h, nq, q_lora, hd_mla, eps, use_tc, diag_this, stream,
        )?;

        // 2026-09-25: One KV projection serves as both K and V. `qkv_output`
        // holds `[Q | K | V]`.
        let q_dim = nq * hd_mla;
        let kv_dim = nkv * hd_mla;
        let k_out = q_full.offset((n * q_dim) as usize * 2);
        let v_out = k_out.offset((n * kv_dim) as usize * 2);
        let kv_latent = ctx.buffers.expert_gate_out(); // 2026-09-25: cache assembly reads it too
        #[allow(clippy::overly_complex_bool_expr)]
        if false && use_tc {
            ops::dense_gemm_tc(
                ctx.gpu,
                self.dense_gemm_tc_k,
                normed,
                &mla.wkv_a,
                kv_latent,
                n,
                kv_lora,
                h,
                stream,
            )?;
        } else {
            ops::dense_gemm(
                ctx.gpu,
                self.dense_gemm_k,
                normed,
                &mla.wkv_a,
                kv_latent,
                n,
                kv_lora,
                h,
                stream,
            )?;
        }
        if diag_this {
            let _ = ctx.gpu.synchronize(stream);
            let kl = kv_lora as usize;
            let mut wbuf = vec![0u8; kl * 2];
            let _ = ctx.gpu.copy_d2h(mla.kv_a_norm.weight, &mut wbuf);
            let wnan = (0..kl).any(|i| {
                let c = &wbuf[i * 2..i * 2 + 2];
                f32::from_bits((u16::from_le_bytes([c[0], c[1]]) as u32) << 16).is_nan()
            });
            let mut lbuf = vec![0u8; (n as usize) * kl * 2];
            let mut lnan = -1i64;
            if ctx.gpu.copy_d2h(kv_latent, &mut lbuf).is_ok() {
                for t in 0..n as usize {
                    if (0..kl).any(|i| {
                        let c = &lbuf[t * kl * 2 + i * 2..t * kl * 2 + i * 2 + 2];
                        f32::from_bits((u16::from_le_bytes([c[0], c[1]]) as u32) << 16).is_nan()
                    }) {
                        lnan = t as i64;
                        break;
                    }
                }
            }
            tracing::info!(
                "DIAG V4-prefill L{} PRE-kvnorm: kv_a_norm has NaN={}, kv_latent first NaN token={}",
                self.attn_layer_idx,
                wnan,
                lnan
            );
        }
        // 2026-09-25: The weighted `kv_a_norm` runs on `kv_latent` before the
        // copies to `k_out`/`v_out`, before RoPE and before cache assembly, so
        // all of them see the normed latent.
        ops::rms_norm(
            ctx.gpu,
            self.rms_norm_w_k,
            kv_latent,
            &mla.kv_a_norm,
            kv_latent,
            n * nkv,
            kv_lora,
            eps,
            stream,
        )?;
        // 2026-09-25: `k_out` = the normed latent.
        ctx.gpu
            .copy_d2d_async(kv_latent, k_out, n as usize * kv_lora as usize * 2, stream)?;
        ctx.gpu
            .synchronize(stream)
            .map_err(|e| anyhow::anyhow!("V4 attn: k_out gemm sync failed: {e}"))?;
        if diag_this {
            // 2026-09-25: K norms over all `n` tokens, then for token 0.
            super::super::trait_impl::diag_norm(
                ctx.gpu,
                k_out,
                (n * kv_lora) as usize,
                stream,
                &format!("V4-prefill L{} K FULL ({} tokens)", self.attn_layer_idx, n),
            );
            super::super::trait_impl::diag_norm(
                ctx.gpu,
                k_out,
                kv_dim as usize,
                stream,
                &format!("V4-prefill L{} K after proj", self.attn_layer_idx),
            );
        }
        // 2026-09-25: `v_out` = K before RoPE. The attention calls below pass
        // `k_out` as V, so only the diagnostics read `v_out`.
        ctx.gpu
            .copy_d2d_async(k_out, v_out, (n * kv_dim) as usize * 2, stream)?;
        if diag_this {
            super::super::trait_impl::diag_norm(
                ctx.gpu,
                v_out,
                kv_dim as usize,
                stream,
                &format!("V4-prefill L{} V after copy", self.attn_layer_idx),
            );
        }

        let k_rope_tmp = self.cache_skip_v4_rope(
            ctx, mla, &meta, q_latent, q_full, k_out, n, nq, nkv, hd_mla, nope, rope, kv_dim,
            diag_this, stream,
        )?;

        // 2026-09-25: Core attention. A compressor layer with at least one full
        // window (`n / ratio > 0`) and the `csa_compress` kernel attends over the
        // windowed raw KV plus the compressed KV (`prefill_attn_compressed`);
        // every other layer runs `prefill_attention_512_sink`. Both pass the
        // per-head sink `attn_sink`.
        let attn_out = ctx.buffers.attn_output();
        // 2026-09-25: `v4_comp_pool_filled` is a field of the layer, shared by
        // every sequence, and the compressed branch below stores to it only when
        // `n / ratio > 0`. Storing `n / ratio` here first keeps a prompt shorter
        // than one window from inheriting the previous prefill's count on this
        // layer. Sequences in flight at the same time still share the counter.
        if let Some(c) = mla.compressor.as_ref() {
            self.v4_comp_pool_filled
                .store(n / c.ratio as u32, std::sync::atomic::Ordering::Relaxed);
        }
        let csa = match mla.compressor {
            Some(c) if self.csa_compress_k.0 != 0 && (n / c.ratio as u32) > 0 => Some(c),
            _ => None,
        };
        let did_csa = if let Some(comp) = csa {
            self.cache_skip_v4_csa(
                ctx, mla, comp, normed, q_full, k_out, attn_out, n, h, nq, nkv, hd_mla, nope, rope,
                eps, stream,
            )?;
            true
        } else {
            false
        };
        // 2026-09-25: Seed decode's compression state from the prompt, on every
        // compressor layer whether or not the compressed branch ran: reset
        // `v4_decode_started`; for a CSA layer with a full window, copy the last
        // full window's `normed` rows to `prev_win`; copy the rows after the last
        // full window to the head of `ring`. These are per-layer fields, shared by
        // every sequence.
        if let Some(comp) = mla.compressor.as_ref() {
            use std::sync::atomic::Ordering::Relaxed;
            let cratio = comp.ratio as u32;
            let cnwin = n / cratio;
            let crem = n % cratio;
            let hbytes = h as usize * 2;
            self.v4_decode_started.store(false, Relaxed);
            if comp.is_csa && cnwin > 0 {
                ctx.gpu.copy_d2d_async(
                    normed.offset(((cnwin - 1) * cratio) as usize * hbytes),
                    comp.prev_win,
                    cratio as usize * hbytes,
                    stream,
                )?;
                self.v4_comp_prev_valid.store(true, Relaxed);
            } else {
                self.v4_comp_prev_valid.store(false, Relaxed);
            }
            if crem > 0 {
                ctx.gpu.copy_d2d_async(
                    normed.offset((cnwin * cratio) as usize * hbytes),
                    comp.ring,
                    crem as usize * hbytes,
                    stream,
                )?;
            }
        }
        if !did_csa {
            // 2026-09-25: `prefill_attention_512_sink` with `k_out` as both K and
            // V, the `V4_WINDOW` window and the per-head sink `attn_sink`.
            ops::prefill_attention_512_sink(
                ctx.gpu,
                self.prefill_attn_512_k,
                q_full,
                k_out,
                k_out,
                attn_out,
                n,
                1,
                nq,
                nkv,
                hd_mla,
                1.0f32 / (hd_mla as f32).sqrt(),
                true,
                V4_WINDOW,
                mla.attn_sink,
                stream,
            )
            .map_err(|e| anyhow::anyhow!("V4 attn: prefill_attention_512_sink failed: {e}"))?;
        }
        ctx.gpu
            .synchronize(stream)
            .map_err(|e| anyhow::anyhow!("V4 attn: prefill_attention sync failed: {e}"))?;

        self.cache_skip_v4_derotate(ctx, mla, &meta, attn_out, n, nq, hd_mla, nope, rope, stream)?;
        if diag_this {
            super::super::trait_impl::diag_norm(
                ctx.gpu,
                attn_out,
                (nq * hd_mla) as usize,
                stream,
                &format!("V4-prefill L{} attn_out token0", self.attn_layer_idx),
            );
            let last_token_offset = ((n - 1) * nq * hd_mla * 2) as usize;
            super::super::trait_impl::diag_norm(
                ctx.gpu,
                attn_out.offset(last_token_offset),
                (nq * hd_mla) as usize,
                stream,
                &format!("V4-prefill L{} attn_out last", self.attn_layer_idx),
            );
        }

        // 2026-09-25: The cache row is the normed latent (`kv_lora`) followed by
        // K's rotated rope dims (`rope`), `mla_cache_dim` wide in all.
        let k_cache_assembled = ctx.buffers.expert_up_out();
        let v_cache_assembled = ctx.buffers.expert_down_out();
        ops::mla_cache_assemble_batched(
            ctx.gpu,
            self.mla_cache_assemble_batched_k,
            kv_latent,
            k_rope_tmp,
            k_cache_assembled,
            v_cache_assembled,
            n,
            kv_lora,
            rope,
            mla_cache_dim,
            stream,
        )?;
        self.write_kv_cache(
            ctx.gpu,
            k_cache_assembled,
            v_cache_assembled,
            kv_cache,
            meta.slot,
            n,
            1,
            mla_cache_dim,
            kv_cache.block_size() as u32,
            mla_cache_dim,
            mla_cache_dim,
            stream,
            ctx.graph_capture,
        )?;
        ctx.gpu
            .synchronize(stream)
            .map_err(|e| anyhow::anyhow!("V4 attn: write_kv_cache sync failed: {e}"))?;

        let o_out = self.cache_skip_v4_o_proj(
            ctx, mla, attn_out, n, h, nq, hd_mla, o_lora, diag_this, stream,
        )?;

        Ok(o_out)
    }
}
