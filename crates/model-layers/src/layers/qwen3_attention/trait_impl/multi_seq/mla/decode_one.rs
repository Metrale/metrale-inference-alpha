// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-26: The absorbed-MLA decode chain for one sequence (`ms_mla_decode_one`), which
//! `ms_mla_decode` runs once per row when the layer's `o_lora_rank` is 0.
//!
//! Owner: model-layers (qwen3 attention).
//! Invariants: the call reads only its `normed` row and `meta` row, and writes only its
//! `o_out` row and its own slot of the latent-KV cache.

use anyhow::Result;
use metrale_cache::kv_cache::PagedKvCache;
use metrale_gpu_runtime::gpu::DevicePtr;

use super::super::ctx::MultiSeqCtx;
use super::super::mla_gemv::MlaDims;
use crate::layer::AttnMetadataDev;
use crate::layers::ops;
use crate::layers::qwen3_attention::Qwen3AttentionLayer;

impl Qwen3AttentionLayer {
    /// 2026-09-25: The absorbed-MLA decode chain for one sequence, with an
    /// explicit `normed` input row and `o_out` destination row.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn ms_mla_decode_one(
        &self,
        c: &MultiSeqCtx<'_>,
        kv_cache: &mut PagedKvCache,
        meta: &AttnMetadataDev,
        normed: DevicePtr,
        o_out: DevicePtr,
        mla: &crate::layers::qwen3_attention::types::MlaWeights,
        d: MlaDims,
        stream: u64,
    ) -> Result<()> {
        let gpu = c.fwd.gpu;
        let buffers = c.fwd.buffers;

        let q_latent = buffers.ssm_ba();
        if let Some(ref wqa_nvfp4) = mla.wq_a_nvfp4 {
            self.nvfp4_decode_gemv(
                gpu,
                c.fwd.levers.gemv_sw,
                normed,
                wqa_nvfp4,
                q_latent,
                d.q_lora,
                d.h,
                stream,
            )?;
        } else {
            ops::dense_gemv(
                gpu,
                self.dense_gemv_k,
                normed,
                &mla.wq_a,
                q_latent,
                d.q_lora,
                d.h,
                stream,
            )?;
        }
        ops::rms_norm(
            gpu,
            self.rms_norm_w_k,
            q_latent,
            &mla.q_a_norm,
            q_latent,
            1,
            d.q_lora,
            d.eps,
            stream,
        )?;
        let q_full = buffers.ssm_deinterleaved();
        if let Some(ref wqb_nvfp4) = mla.wq_b_nvfp4 {
            self.nvfp4_decode_gemv(
                gpu,
                c.fwd.levers.gemv_sw,
                q_latent,
                wqb_nvfp4,
                q_full,
                d.q_dim,
                d.q_lora,
                stream,
            )?;
        } else {
            ops::dense_gemv(
                gpu,
                self.dense_gemv_k,
                q_latent,
                &mla.wq_b,
                q_full,
                d.q_dim,
                d.q_lora,
                stream,
            )?;
        }

        // 2026-09-25: Q_absorbed = Q_nope @ W_UK_T.
        let q_absorbed_buf = buffers.expert_up_out();
        self.ms_mla_q_absorb(c, mla, &d, q_full, q_absorbed_buf, stream)?;

        // 2026-09-25: Copy each head's RoPE part of `q_full` to `q_rope_direct`
        // and to offset `kv_lora` of that head's `mla_cache_dim` slot in the
        // absorbed Q.
        let q_rope_direct = buffers.ssm_conv_out_f32();
        if self.mla_q_rope_scatter_k.0 != 0 {
            ops::mla_q_rope_scatter(
                gpu,
                self.mla_q_rope_scatter_k,
                q_full,
                q_absorbed_buf,
                q_rope_direct,
                d.nq,
                d.hd,
                d.mla_nope,
                d.mla_rope,
                d.kv_lora,
                d.mla_cache_dim,
                stream,
            )?;
        } else {
            for head_idx in 0..d.nq as usize {
                let src = q_full.offset((head_idx * d.hd as usize + mla.nope) * 2);
                gpu.copy_d2d_async(
                    src,
                    q_rope_direct.offset(head_idx * mla.rope * 2),
                    mla.rope * 2,
                    stream,
                )?;
                gpu.copy_d2d_async(
                    src,
                    q_absorbed_buf
                        .offset((head_idx * d.mla_cache_dim as usize + mla.kv_lora_rank) * 2),
                    mla.rope * 2,
                    stream,
                )?;
            }
        }

        let kv_latent = buffers.expert_gate_out();
        if let Some(ref wkva_nvfp4) = mla.wkv_a_nvfp4 {
            self.nvfp4_decode_gemv(
                gpu,
                c.fwd.levers.gemv_sw,
                normed,
                wkva_nvfp4,
                kv_latent,
                d.kv_lora,
                d.h,
                stream,
            )?;
        } else {
            ops::dense_gemv(
                gpu,
                self.dense_gemv_k,
                normed,
                &mla.wkv_a,
                kv_latent,
                d.kv_lora,
                d.h,
                stream,
            )?;
        }
        ops::rms_norm(
            gpu,
            self.rms_norm_w_k,
            kv_latent,
            &mla.kv_a_norm,
            kv_latent,
            1,
            d.kv_lora,
            d.eps,
            stream,
        )?;

        // 2026-09-25: `k_rope_single` reuses `ssm_ba`: `q_latent`, its previous
        // content, was last read by the `wq_b` GEMV above, on the same stream.
        let k_rope_single = buffers.ssm_ba();
        ops::dense_gemv(
            gpu,
            self.dense_gemv_k,
            normed,
            &mla.wkv_a_rope,
            k_rope_single,
            d.mla_rope,
            d.h,
            stream,
        )?;
        ops::rope_yarn(
            gpu,
            self.rope_yarn_k,
            q_rope_direct,
            k_rope_single,
            meta.positions,
            1,
            d.nq,
            1,
            d.mla_rope,
            d.mla_rope,
            mla.yarn_inv_freq,
            c.fwd.config.rope_theta as f32,
            stream,
        )?;
        if self.mla_q_rope_writeback_k.0 != 0 {
            ops::mla_q_rope_writeback(
                gpu,
                self.mla_q_rope_writeback_k,
                q_rope_direct,
                q_absorbed_buf,
                d.nq,
                d.mla_rope,
                d.kv_lora,
                d.mla_cache_dim,
                stream,
            )?;
        } else {
            for head_idx in 0..d.nq as usize {
                let src = q_rope_direct.offset(head_idx * mla.rope * 2);
                let dst = q_absorbed_buf
                    .offset((head_idx * d.mla_cache_dim as usize + mla.kv_lora_rank) * 2);
                gpu.copy_d2d_async(src, dst, mla.rope * 2, stream)?;
            }
        }

        // 2026-09-25: The K and V cache entries are assembled in `qkv_output`,
        // then written to this sequence's slot.
        let k_cache_entry = buffers.qkv_output();
        let v_cache_entry = k_cache_entry.offset(d.mla_cache_dim as usize * 2);
        if self.mla_cache_assemble_k.0 != 0 {
            ops::mla_cache_assemble(
                gpu,
                self.mla_cache_assemble_k,
                kv_latent,
                k_rope_single,
                k_cache_entry,
                v_cache_entry,
                d.kv_lora,
                d.mla_rope,
                d.mla_cache_dim,
                stream,
            )?;
        } else {
            gpu.copy_d2d_async(kv_latent, k_cache_entry, mla.kv_lora_rank * 2, stream)?;
            gpu.copy_d2d_async(
                k_rope_single,
                k_cache_entry.offset(mla.kv_lora_rank * 2),
                mla.rope * 2,
                stream,
            )?;
            gpu.copy_d2d_async(kv_latent, v_cache_entry, mla.kv_lora_rank * 2, stream)?;
            gpu.memset_async(
                v_cache_entry.offset(mla.kv_lora_rank * 2),
                0,
                mla.rope * 2,
                stream,
            )?;
        }
        self.write_kv_cache(
            gpu,
            k_cache_entry,
            v_cache_entry,
            kv_cache,
            meta.slot,
            1,
            1,
            d.mla_cache_dim,
            d.bs as u32,
            d.mla_cache_dim,
            d.mla_cache_dim,
            stream,
            c.fwd.graph_capture,
        )?;

        let attn_out = buffers.attn_output();
        ops::paged_decode_attn_bf16(
            gpu,
            self.paged_decode_mla_k,
            q_absorbed_buf,
            kv_cache.k_pool_ptr(self.attn_layer_idx),
            kv_cache.v_pool_ptr(self.attn_layer_idx),
            attn_out,
            meta.block_table,
            meta.seq_len,
            meta.max_blocks_per_seq,
            1,
            d.nq,
            1,
            d.mla_cache_dim,
            d.bs as u32,
            d.inv_sqrt_d,
            d.nq * d.mla_cache_dim,
            0,
            stream,
        )?;

        // 2026-09-25: V extraction, attn_latent @ W_UV, into `ssm_qkvz`:
        // `norm_output` holds the `normed` rows that later iterations still read.
        let v_extracted = buffers.ssm_qkvz();
        self.ms_mla_v_extract(c, mla, &d, attn_out, v_extracted, stream)?;

        if d.o_lora_rank > 0 {
            // 2026-09-25: Low-rank O projection (wo_a, then wo_b). The only
            // caller, `ms_mla_decode`, sends `o_lora_rank > 0` to
            // `attention_forward_v4` first, so this arm is not reached.
            let o_latent = buffers.attn_output();
            if let Some(ref woa_nvfp4) = mla.wo_a_nvfp4 {
                self.nvfp4_decode_gemv(
                    gpu,
                    c.fwd.levers.gemv_sw,
                    v_extracted,
                    woa_nvfp4,
                    o_latent,
                    d.o_lora_rank,
                    d.nq * d.mla_v_dim,
                    stream,
                )?;
            } else {
                ops::dense_gemv(
                    gpu,
                    self.dense_gemv_k,
                    v_extracted,
                    &mla.wo_a,
                    o_latent,
                    d.o_lora_rank,
                    d.nq * d.mla_v_dim,
                    stream,
                )?;
            }
            if let Some(ref wob_nvfp4) = mla.wo_b_nvfp4 {
                self.nvfp4_decode_gemv(
                    gpu,
                    c.fwd.levers.gemv_sw,
                    o_latent,
                    wob_nvfp4,
                    o_out,
                    d.h,
                    d.o_lora_rank,
                    stream,
                )?;
            } else {
                ops::dense_gemv(
                    gpu,
                    self.dense_gemv_k,
                    o_latent,
                    &mla.wo_b,
                    o_out,
                    d.h,
                    d.o_lora_rank,
                    stream,
                )?;
            }
        } else if let Some(ref wo_nvfp4) = mla.wo_nvfp4 {
            self.nvfp4_decode_gemv(
                gpu,
                c.fwd.levers.gemv_sw,
                v_extracted,
                wo_nvfp4,
                o_out,
                d.h,
                d.nq * d.mla_v_dim,
                stream,
            )?;
        } else {
            ops::dense_gemv(
                gpu,
                self.dense_gemv_k,
                v_extracted,
                &mla.wo,
                o_out,
                d.h,
                d.nq * d.mla_v_dim,
                stream,
            )?;
        }
        Ok(())
    }
}
