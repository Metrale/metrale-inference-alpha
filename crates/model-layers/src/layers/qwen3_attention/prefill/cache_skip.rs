// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-25: `prefill_attention_with_cache_skip`: full-attention prefill
//! that computes Q/K/V for every row of the chunk, writes the paged KV cache
//! only for positions from `kv_write_start` on, and runs flash attention on
//! the contiguous Q/K/V.
//!
//! Owner: model-layers (attention).
//! Invariants: none beyond the types.

use anyhow::Result;
use metrale_cache::kv_cache::PagedKvCache;
use metrale_gpu_runtime::gpu::DevicePtr;

use super::super::Qwen3AttentionLayer;
use crate::layer::{BatchedAttnMetadata, ForwardContext};
use crate::layers::ops;

impl Qwen3AttentionLayer {
    /// 2026-09-25: Prefill attention that skips the KV-cache write for
    /// positions already cached.
    ///
    /// `kv_write_start`: the number of leading positions whose KV is already
    /// in the cache; the cache write covers only positions from it on. MLA
    /// layers return through `cache_skip_mla.rs` (or `cache_skip_v4.rs` when
    /// `o_lora_rank > 0`).
    #[allow(unreachable_code, unused_variables, unused_assignments)]
    pub(in crate::layers::qwen3_attention) fn prefill_attention_with_cache_skip(
        &self,
        state: &mut dyn crate::layer::LayerState,
        normed: DevicePtr,
        num_tokens: usize,
        kv_write_start: usize,
        // 2026-09-25: The sequence's host block table; the QSA hook below reads
        // the paged cache by physical block through it.
        seq_block_table: &[u32],
        kv_cache: &mut PagedKvCache,
        // 2026-09-25: `Some`: a batched chunk. Positions and slots come from
        // the stacked metadata, and flash attention runs
        // `num_tokens / chunk_len` independent causal sequences of `chunk_len`
        // rows. `None`: one sequence, metadata from `ctx.attn_metadata`.
        batched_meta: Option<&BatchedAttnMetadata>,
        ctx: &ForwardContext,
        stream: u64,
    ) -> Result<DevicePtr> {
        let h = ctx.config.hidden_size as u32;
        let nq = self
            .num_q_heads_override
            .unwrap_or(ctx.config.num_attention_heads) as u32;
        let nkv = self
            .num_kv_heads_override
            .unwrap_or(ctx.config.num_key_value_heads) as u32;
        let hd = self.head_dim_override.unwrap_or(ctx.config.head_dim) as u32;
        let eps = ctx.config.rms_norm_eps as f32;
        let bs = kv_cache.block_size();
        let n = num_tokens as u32;
        let bf16 = 2usize;

        // 2026-09-25: A batched chunk must be non-MLA with head_dim <= 256; the
        // model-engine caller (`check_kernel_batched_eligible`) already refuses
        // the others.
        if batched_meta.is_some() {
            anyhow::ensure!(
                self.mla.is_none() && hd <= 256,
                "batched cache-skip flash: MLA/hd>256 unsupported (gate to paged)"
            );
        }
        // 2026-09-25: Position and slot sources, and the flash launch's
        // sequence length and batch.
        let (positions, positions_h, positions_w, kv_slot, flash_seq_len, flash_batch) =
            if let Some(mb) = batched_meta {
                // 2026-09-25: The batch comes from the stacked token count, which
                // must be a whole number of `chunk_len` sequences, so the flash
                // kernel never reads past the rows it was given.
                anyhow::ensure!(
                    mb.chunk_len > 0 && num_tokens.is_multiple_of(mb.chunk_len as usize),
                    "batched flash: num_tokens {num_tokens} not a multiple of chunk_len {}",
                    mb.chunk_len
                );
                let derived_batch = (num_tokens / mb.chunk_len as usize) as u32;
                (
                    mb.positions_stacked,
                    mb.positions_h_stacked,
                    mb.positions_w_stacked,
                    mb.slot_stacked,
                    mb.chunk_len,
                    derived_batch,
                )
            } else {
                let meta = ctx
                    .attn_metadata
                    .expect("attention prefill requires metadata");
                (
                    meta.positions,
                    meta.positions_h,
                    meta.positions_w,
                    meta.slot,
                    n,
                    1u32,
                )
            };

        let q_dim = (nq * hd) as usize;
        let q_proj_dim = if self.gated { q_dim * 2 } else { q_dim };
        let kv_dim = (nkv * hd) as usize;

        let _qg_out = ctx.buffers.qkv_output();
        let _k_contiguous = ctx.buffers.ssm_qkvz();
        let _v_contiguous = ctx.buffers.attn_output();

        macro_rules! aprof {
            ($label:expr, $t0:expr) => {
                if ctx.profile {
                    if let Some(t0) = $t0 {
                        ctx.gpu.synchronize(stream)?;
                        let elapsed = t0.elapsed().as_micros();
                        tracing::info!("  ATTN prefill [{}] N={}: {}µs", $label, n, elapsed);
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

        // 2026-09-25: With FP8xFP8 weights (`q_fp8`), convert the activations
        // to FP8 once for all three projections, into `attn_output` as scratch.
        let use_fp8_act = self.q_fp8.is_some();
        let normed_fp8 = if use_fp8_act {
            let act_fp8 = ctx.buffers.attn_output();
            ops::bf16_to_fp8(ctx.gpu, self.bf16_to_fp8_k, normed, act_fp8, n * h, stream)?;
            act_fp8
        } else {
            DevicePtr::NULL
        };
        aprof!("bf16_to_fp8", t0);
        t0 = if ctx.profile {
            ctx.gpu.synchronize(stream)?;
            Some(std::time::Instant::now())
        } else {
            None
        };

        if let Some(ref mla) = self.mla {
            let args = super::cache_skip_mla::CacheSkipMlaArgs {
                normed,
                num_tokens,
                n,
                h,
                nq,
                nkv,
                hd,
                kv_dim,
                eps,
                bf16,
                stream,
            };
            if mla.o_lora_rank > 0 {
                return self.prefill_attention_cache_skip_v4(kv_cache, ctx, &args);
            }
            return self.prefill_attention_cache_skip_mla(kv_cache, ctx, &args);
        }

        // 2026-09-25: Q/K/V projection for non-MLA layers.
        let ht = std::env::var("METRALE_PREFILL_HOST_TIMING").as_deref() == Ok("1");
        let tp0 = ht.then(std::time::Instant::now);
        if self.mla.is_none() {
            self.prefill_attention_cache_skip_qkv(
                normed, normed_fp8, n, h, nkv, hd, q_proj_dim, kv_dim, num_tokens, bf16, ctx,
                stream,
            )?;
        }
        if let Some(t) = tp0 {
            crate::layers::qwen3_attention::add_attn_phase_us(0, t.elapsed().as_micros() as u64);
        }
        let tp1 = ht.then(std::time::Instant::now);

        // 2026-09-25: Separate Q from the gate and apply the Q/K RMS norms.
        let qg_out = ctx.buffers.qkv_output();
        let k_contiguous = ctx.buffers.ssm_qkvz();
        let v_contiguous = k_contiguous.offset(num_tokens * kv_dim * bf16);
        // 2026-09-25: Ungated layers: the q_proj output already is Q in the
        // `[n, nq*hd]` layout (`q_proj_dim == q_dim`), so `q_contiguous` aliases
        // `qg_out` instead of copying it; the gate step below reads `qg_out`
        // only `if self.gated`. Gated layers' `qg_out` holds Q and the gate
        // together, so they deinterleave into a separate buffer.
        let q_contiguous = if self.gated {
            ctx.buffers.ssm_deinterleaved()
        } else {
            qg_out
        };
        let q_rope_fused = self.gated
            && !self.attn.q_norm.weight.is_null()
            && self.mrope_interleaved
            && !self.rope_proportional
            && std::env::var("METRALE_ATTN_PREFILL_FUSED_QROPE")
                .ok()
                .as_deref()
                == Some("1")
            && self.deinterleave_qg_split_qnorm_mrope_k.0 != 0
            && self.rope_mrope_interleaved_k_only_k.0 != 0;
        self.cache_skip_qk_norms(
            ctx,
            q_rope_fused,
            qg_out,
            q_contiguous,
            k_contiguous,
            v_contiguous,
            positions,
            positions_h,
            positions_w,
            n,
            nq,
            nkv,
            hd,
            q_proj_dim,
            q_dim,
            kv_dim,
            num_tokens,
            bf16,
            eps,
            stream,
        )?;

        aprof!("deinterleave+norms", t0);
        t0 = if ctx.profile {
            ctx.gpu.synchronize(stream)?;
            Some(std::time::Instant::now())
        } else {
            None
        };

        self.cache_skip_rope(
            ctx,
            q_rope_fused,
            q_contiguous,
            k_contiguous,
            positions,
            positions_h,
            positions_w,
            n,
            nq,
            nkv,
            hd,
            num_tokens,
            bf16,
            stream,
        )?;

        aprof!("rope", t0);
        t0 = if ctx.profile {
            ctx.gpu.synchronize(stream)?;
            Some(std::time::Instant::now())
        } else {
            None
        };

        // 2026-09-25: Write K/V for positions from `kv_write_start` on to the
        // paged cache.
        let write_start = kv_write_start;
        let write_count = num_tokens.saturating_sub(write_start);
        if write_count > 0 {
            let k_offset = write_start * kv_dim * bf16;
            let v_offset = write_start * kv_dim * bf16;
            let slot_offset = write_start * 8;
            self.write_kv_cache(
                ctx.gpu,
                k_contiguous.offset(k_offset),
                v_contiguous.offset(v_offset),
                kv_cache,
                kv_slot.offset(slot_offset),
                write_count as u32,
                nkv,
                hd,
                bs as u32,
                nkv * hd,
                nkv * hd,
                stream,
                ctx.graph_capture,
            )?;
        }
        aprof!("kv_cache_write", t0);
        t0 = if ctx.profile {
            ctx.gpu.synchronize(stream)?;
            Some(std::time::Instant::now())
        } else {
            None
        };

        if self.attn_layer_idx == 0 && ctx.config.model_type == "mistral" {
            tracing::info!(
                "DIAG ADDRS: q_contiguous={:?} k_contiguous={:?} v_at_offset={:?} ssm_deinterleaved={:?} ssm_qkvz={:?} attn_output={:?}",
                q_contiguous.0,
                k_contiguous.0,
                k_contiguous.offset(num_tokens * kv_dim * bf16).0,
                ctx.buffers.ssm_deinterleaved().0,
                ctx.buffers.ssm_qkvz().0,
                ctx.buffers.attn_output().0
            );
            crate::layers::qwen3_attention::trait_impl::diag_norm(
                ctx.gpu,
                q_contiguous,
                (nq * hd) as usize,
                stream,
                "L0 Q[0] pre-attn",
            );
            crate::layers::qwen3_attention::trait_impl::diag_norm(
                ctx.gpu,
                k_contiguous,
                (nkv * hd) as usize,
                stream,
                "L0 K[0] pre-attn",
            );
            let v_check = k_contiguous.offset(num_tokens * kv_dim * bf16);
            crate::layers::qwen3_attention::trait_impl::diag_norm(
                ctx.gpu,
                v_check,
                (nkv * hd) as usize,
                stream,
                "L0 V[0] pre-attn",
            );
        }

        // 2026-09-25: Flash attention on the contiguous Q/K/V.
        let attn_out = ctx.buffers.attn_output();
        let inv_sqrt_d = self.effective_attn_scale(hd);

        let wide_head_path = self.cache_skip_flash(
            ctx,
            kv_write_start,
            q_contiguous,
            k_contiguous,
            v_contiguous,
            attn_out,
            flash_seq_len,
            flash_batch,
            n,
            nq,
            nkv,
            hd,
            inv_sqrt_d,
            tp1,
            stream,
        )?;
        // 2026-09-25: The profile label names the kernel that ran: the wide-head
        // branch's tensor-core or scalar kernel, or `prefill_attention_64`.
        aprof!(
            match (wide_head_path, self.prefill_attn_512_is_tc) {
                (true, true) => "flash_attn_512_tc",
                (true, false) => "flash_attn_512_scalar",
                _ => "flash_attn_64",
            },
            t0
        );
        t0 = if ctx.profile {
            ctx.gpu.synchronize(stream)?;
            Some(std::time::Instant::now())
        } else {
            None
        };

        self.cache_skip_gates(
            state,
            ctx,
            kv_cache,
            seq_block_table,
            normed,
            qg_out,
            q_contiguous,
            attn_out,
            n,
            h,
            nq,
            hd,
            q_proj_dim,
            q_dim,
            num_tokens,
            bs,
            bf16,
            inv_sqrt_d,
            stream,
        )?;
        aprof!("sigmoid_gate", t0);
        t0 = if ctx.profile {
            ctx.gpu.synchronize(stream)?;
            Some(std::time::Instant::now())
        } else {
            None
        };

        // 2026-09-25: METRALE_OP_DUMP: the attention output after the gates,
        // which is the O-projection input.
        if num_tokens > 0 {
            let nq_hd = (nq * hd) as usize;
            super::super::op_dump::dump_bf16(
                ctx.gpu,
                attn_out,
                (num_tokens - 1) * nq_hd * bf16,
                nq_hd,
                self.attn_layer_idx,
                "attn_out_post_gate",
                stream,
            )?;
        }

        // 2026-09-25: O projection (`paged_oproj.rs`).
        let o_out = self.prefill_attention_paged_oproj(attn_out, n, h, nq, hd, ctx, stream)?;
        aprof!("o_proj", t0);
        Ok(o_out)
    }
}
