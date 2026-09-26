// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-25: The DeepSeek-V4-Flash branch of `prefill_attention_paged`
//! (`paged.rs` sends MLA layers with `o_lora_rank > 0` here): Q and direct KV
//! projections, interleaved RoPE, GQA flash attention with a per-head sink,
//! de-rotation of the output, the compressed cache write and the grouped
//! low-rank O projection.
//!
//! Owner: model-layers (qwen3 attention).
//! Invariants: none beyond the types.
//!
//! Attention covers only this call's tokens: the paged cache is not read, and,
//! unlike `paged_mla.rs`, a call with `seq_len_start > 0` is not refused.

use anyhow::Result;
use metrale_cache::kv_cache::PagedKvCache;
use metrale_gpu_runtime::gpu::DevicePtr;

use super::super::Qwen3AttentionLayer;
use super::paged_mla::MlaPrefillArgs;
use crate::layer::ForwardContext;
use crate::layers::ops;

impl Qwen3AttentionLayer {
    pub(super) fn prefill_attention_paged_v4(
        &self,
        kv_cache: &mut PagedKvCache,
        ctx: &ForwardContext,
        args: &MlaPrefillArgs,
        _seq_len_start: usize,
    ) -> Result<DevicePtr> {
        let MlaPrefillArgs {
            normed,
            num_tokens: _,
            n,
            h,
            nq,
            nkv,
            hd: _,
            seq_len_start: _,
            kv_dim: _,
            eps,
            bf16: _,
            bs,
            stream,
        } = *args;
        let mla = self
            .mla
            .as_ref()
            .expect("V4-Flash paged prefill requires MLA");
        let meta = ctx
            .attn_metadata
            .expect("V4-Flash paged prefill requires metadata");

        let nope = mla.nope as u32;
        let rope = mla.rope as u32;
        let kv_lora = mla.kv_lora_rank as u32;
        let _v_dim = mla.v_dim as u32;
        let q_lora = mla.q_lora_rank as u32;
        let o_lora = mla.o_lora_rank as u32;
        let _mla_cache_dim = kv_lora + rope;
        let hd_mla = nope + rope;

        // 2026-09-25: Q: latent, norm, expand.
        let q_latent = ctx.buffers.ssm_ba();
        ops::dense_gemm(
            ctx.gpu,
            self.dense_gemm_k,
            normed,
            &mla.wq_a,
            q_latent,
            n,
            q_lora,
            h,
            stream,
        )?;
        ops::rms_norm(
            ctx.gpu,
            self.rms_norm_w_k,
            q_latent,
            &mla.q_a_norm,
            q_latent,
            n,
            q_lora,
            eps,
            stream,
        )?;
        let q_full = ctx.buffers.qkv_output();
        ops::dense_gemm(
            ctx.gpu,
            self.dense_gemm_k,
            q_latent,
            &mla.wq_b,
            q_full,
            n,
            nq * hd_mla,
            q_lora,
            stream,
        )?;
        // 2026-09-25: RMS norm of each of the `n * nq` Q head vectors with the
        // unit weight `norm_unit_w`, before RoPE.
        ops::rms_norm(
            ctx.gpu,
            self.rms_norm_k,
            q_full,
            &crate::weight_map::DenseWeight {
                weight: ctx.buffers.norm_unit_w(),
            },
            q_full,
            n * nq,
            hd_mla,
            eps,
            stream,
        )?;

        // 2026-09-25: Direct KV projection, no absorption; K and V start equal.
        // `qkv_output` holds Q `[n, q_dim]`, then K `[n, kv_dim]`, then V.
        let q_dim = nq * hd_mla;
        let kv_dim = nkv * hd_mla;
        let k_out = q_full.offset((n * q_dim) as usize * 2);
        let v_out = k_out.offset((n * kv_dim) as usize * 2);
        ops::dense_gemm(
            ctx.gpu,
            self.dense_gemm_k,
            normed,
            &mla.wkv_a,
            k_out,
            n,
            kv_lora,
            h,
            stream,
        )?;
        // 2026-09-25: `kv_a_norm` over each KV latent row, before RoPE.
        ops::rms_norm(
            ctx.gpu,
            self.rms_norm_w_k,
            k_out,
            &mla.kv_a_norm,
            k_out,
            n * nkv,
            kv_lora,
            eps,
            stream,
        )?;
        // 2026-09-25: V is a copy of K, taken before RoPE.
        ctx.gpu
            .copy_d2d_async(k_out, v_out, (n * kv_dim) as usize * 2, stream)?;

        // 2026-09-25: RoPE on Q and K, not V. The rope dims sit at offset `nope`
        // in each head, so they are extracted, rotated and written back.
        let q_rope_tmp = ctx.buffers.ssm_conv_out_f32();
        let k_rope_tmp = q_latent; // 2026-09-25: free: the wq_b GEMM was its last reader
        ops::mla_q_rope_extract_batched(
            ctx.gpu,
            self.mla_q_rope_extract_batched_k,
            q_full,
            q_rope_tmp,
            n,
            nq,
            hd_mla,
            nope,
            rope,
            nq * hd_mla,
            stream,
        )?;
        ops::mla_q_rope_extract_batched(
            ctx.gpu,
            self.mla_q_rope_extract_batched_k,
            k_out,
            k_rope_tmp,
            n,
            nkv,
            hd_mla,
            nope,
            rope,
            nkv * hd_mla,
            stream,
        )?;
        ops::rope_yarn(
            ctx.gpu,
            // 2026-09-25: Interleaved RoPE: adjacent pairs (2i, 2i+1).
            self.rope_yarn_interleaved_k,
            q_rope_tmp,
            k_rope_tmp,
            meta.positions,
            n,
            nq,
            nkv,
            rope,
            rope,
            // 2026-09-25: A layer without a compressor uses `main_inv_freq` and
            // mscale 1; one with a compressor uses `yarn_inv_freq` and
            // `yarn_rope_mscale`. The de-rotation below makes the same choice.
            if mla.compressor.is_none() {
                mla.main_inv_freq
            } else {
                mla.yarn_inv_freq
            },
            if mla.compressor.is_none() {
                1.0f32
            } else {
                super::super::helpers::yarn_rope_mscale(ctx.config)
            },
            stream,
        )?;
        ops::mla_q_rope_writeback_batched(
            ctx.gpu,
            self.mla_q_rope_writeback_batched_k,
            q_rope_tmp,
            q_full,
            n,
            nq,
            hd_mla,
            nope,
            rope,
            nq * hd_mla,
            stream,
        )?;
        ops::mla_q_rope_writeback_batched(
            ctx.gpu,
            self.mla_q_rope_writeback_batched_k,
            k_rope_tmp,
            k_out,
            n,
            nkv,
            hd_mla,
            nope,
            rope,
            nkv * hd_mla,
            stream,
        )?;

        // 2026-09-25: Causal GQA flash attention over this call's tokens.
        let attn_out = ctx.buffers.attn_output();
        let prefill_k = if hd_mla > 256 {
            if self.prefill_attn_512_k.0 == 0 {
                anyhow::bail!(
                    "V4-Flash paged prefill: hd_mla={} > 256 but prefill_attn_512_k is not loaded (handle=0). \
                     The attn_prefill_512 kernel must be present in the PTX.",
                    hd_mla
                );
            }
            tracing::info!(
                "V4-Flash paged prefill: using prefill_attn_512_k (hd_mla={})",
                hd_mla
            );
            self.prefill_attn_512_k
        } else {
            tracing::info!(
                "V4-Flash paged prefill: using prefill_attn_64_k (hd_mla={})",
                hd_mla
            );
            self.prefill_attn_64_k
        };
        // 2026-09-25: `k_out`, with its rotated rope part, is passed as both key
        // and value; `v_out`, the unrotated copy, is kept for the cache below. The
        // per-head sink `attn_sink` is the one the MLA decode passes
        // (`decode/run_paged_decode.rs`).
        ops::prefill_attention_512_sink(
            ctx.gpu,
            prefill_k,
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
            0,
            mla.attn_sink,
            stream,
        )
        .map_err(|e| anyhow::anyhow!("V4 paged: prefill_attention failed: {e}"))?;

        // 2026-09-25: De-rotate the output's rope dims at each query position
        // (inverse interleaved RoPE) before the grouped O projection.
        {
            let o_rope_tmp = ctx.buffers.ssm_conv_out_f32();
            ops::mla_q_rope_extract_batched(
                ctx.gpu,
                self.mla_q_rope_extract_batched_k,
                attn_out,
                o_rope_tmp,
                n,
                nq,
                hd_mla,
                nope,
                rope,
                nq * hd_mla,
                stream,
            )?;
            ops::rope_yarn(
                ctx.gpu,
                self.rope_yarn_interleaved_inv_k,
                o_rope_tmp,
                o_rope_tmp,
                meta.positions,
                n,
                nq,
                0,
                rope,
                rope,
                // 2026-09-25: The same table and mscale as the RoPE above.
                if mla.compressor.is_none() {
                    mla.main_inv_freq
                } else {
                    mla.yarn_inv_freq
                },
                if mla.compressor.is_none() {
                    1.0f32
                } else {
                    super::super::helpers::yarn_rope_mscale(ctx.config)
                },
                stream,
            )?;
            ops::mla_q_rope_writeback_batched(
                ctx.gpu,
                self.mla_q_rope_writeback_batched_k,
                o_rope_tmp,
                attn_out,
                n,
                nq,
                hd_mla,
                nope,
                rope,
                nq * hd_mla,
                stream,
            )?;
        }

        // 2026-09-25: Compressed cache rows, `kv_lora + rope` wide, from `v_out`
        // (the latent before RoPE) and `k_rope_tmp` (the rotated rope values).
        let k_cache_assembled = ctx.buffers.expert_up_out();
        let v_cache_assembled = ctx.buffers.expert_down_out();
        let mla_cache_dim = kv_lora + rope;
        ops::mla_cache_assemble_batched(
            ctx.gpu,
            self.mla_cache_assemble_batched_k,
            v_out,
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
            bs,
            mla_cache_dim,
            mla_cache_dim,
            stream,
            ctx.graph_capture,
        )?;

        // 2026-09-25: Grouped low-rank O projection. `wo_a` is block-diagonal
        // over `o_groups`: each `group_in` slice of a row projects to `o_lora`,
        // giving an `o_groups * o_lora` latent that `wo_b` maps to hidden size.
        // The slices are strided, so `wo_a` runs as one GEMV per token and group;
        // `o_latent` is contiguous, so `wo_b` is one GEMM.
        let o_groups = ctx.config.o_groups.max(1) as u32;
        let group_in = (nq * hd_mla) / o_groups;
        let latent_dim = o_groups * o_lora;
        let o_latent = ctx.buffers.o_latent();
        let o_out = ctx.buffers.qkv_output();
        for t in 0..n {
            for g in 0..o_groups {
                let in_g = attn_out.offset(((t * nq * hd_mla) + g * group_in) as usize * 2);
                let w_g = crate::weight_map::DenseWeight {
                    weight: mla
                        .wo_a
                        .weight
                        .offset((g as usize) * (o_lora as usize) * (group_in as usize) * 2),
                };
                let out_g = o_latent.offset(((t * latent_dim) + g * o_lora) as usize * 2);
                ops::dense_gemv(
                    ctx.gpu,
                    self.dense_gemv_k,
                    in_g,
                    &w_g,
                    out_g,
                    o_lora,
                    group_in,
                    stream,
                )?;
            }
        }
        ops::dense_gemm(
            ctx.gpu,
            self.dense_gemm_k,
            o_latent,
            &mla.wo_b,
            o_out,
            n,
            h,
            latent_dim,
            stream,
        )?;

        Ok(o_out)
    }
}
