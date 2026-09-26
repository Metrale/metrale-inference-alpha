// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-26: The Q projection and the Q/K RoPE of the DeepSeek-V4 cache-skip
//! prefill (`cache_skip_v4.rs`).
//!
//! Owner: model-layers (attention).
//! Invariants: `prefill_attention_cache_skip_v4` calls these helpers where
//! their statements run, so every launch keeps its order and arguments.

use anyhow::Result;
use metrale_gpu_runtime::gpu::DevicePtr;

use super::super::Qwen3AttentionLayer;
use crate::layer::ForwardContext;
use crate::layers::ops;

impl Qwen3AttentionLayer {
    /// 2026-09-26: The V4 Q path: latent projection (`wq_a`), `q_a_norm`,
    /// expansion (`wq_b`) and the per-head unit RMS norm. Returns
    /// `(q_latent, q_full)`.
    pub(super) fn cache_skip_v4_q_proj(
        &self,
        ctx: &ForwardContext,
        mla: &crate::layers::qwen3_attention::MlaWeights,
        normed: DevicePtr,
        n: u32,
        h: u32,
        nq: u32,
        q_lora: u32,
        hd_mla: u32,
        eps: f32,
        use_tc: bool,
        diag_this: bool,
        stream: u64,
    ) -> Result<(DevicePtr, DevicePtr)> {
        // 2026-09-25: Q: latent projection, norm, expansion.
        let q_latent = ctx.buffers.ssm_ba();
        if use_tc {
            ops::dense_gemm_tc(
                ctx.gpu,
                self.dense_gemm_tc_k,
                normed,
                &mla.wq_a,
                q_latent,
                n,
                q_lora,
                h,
                stream,
            )?;
        } else {
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
        }
        ctx.gpu
            .synchronize(stream)
            .map_err(|e| anyhow::anyhow!("V4 attn: q_latent gemm sync failed: {e}"))?;
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
        ctx.gpu
            .synchronize(stream)
            .map_err(|e| anyhow::anyhow!("V4 attn: q_a_norm sync failed: {e}"))?;
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
        ctx.gpu
            .synchronize(stream)
            .map_err(|e| anyhow::anyhow!("V4 attn: q_full gemm sync failed: {e}"))?;
        // 2026-09-25: An RMS norm with the unit weight `norm_unit_w` over each of
        // the `n * nq` Q head vectors, before RoPE.
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
        if diag_this {
            super::super::trait_impl::diag_norm(
                ctx.gpu,
                q_full,
                (nq * hd_mla) as usize,
                stream,
                &format!(
                    "V4-prefill L{} Q after q_b_norm token0",
                    self.attn_layer_idx
                ),
            );
            let q_last_off = ((n - 1) * nq * hd_mla * 2) as usize;
            super::super::trait_impl::diag_norm(
                ctx.gpu,
                q_full.offset(q_last_off),
                (nq * hd_mla) as usize,
                stream,
                &format!("V4-prefill L{} Q after q_b_norm last", self.attn_layer_idx),
            );
        }
        Ok((q_latent, q_full))
    }

    /// 2026-09-26: The V4 interleaved RoPE on the last `rope` dims of each Q and K
    /// head: extract, rotate, write back. Returns `k_rope_tmp`, which cache
    /// assembly reads.
    pub(super) fn cache_skip_v4_rope(
        &self,
        ctx: &ForwardContext,
        mla: &crate::layers::qwen3_attention::MlaWeights,
        meta: &crate::layer::AttnMetadataDev,
        q_latent: DevicePtr,
        q_full: DevicePtr,
        k_out: DevicePtr,
        n: u32,
        nq: u32,
        nkv: u32,
        hd_mla: u32,
        nope: u32,
        rope: u32,
        kv_dim: u32,
        diag_this: bool,
        stream: u64,
    ) -> Result<DevicePtr> {
        // 2026-09-25: RoPE on Q and K. The rope dims are the last `rope` of each
        // head (offset `nope`): extract, rotate, write back.
        let q_rope_tmp = ctx.buffers.ssm_conv_out_f32();
        let k_rope_tmp = q_latent; // 2026-09-25: free after the wq_b GEMM
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
            // 2026-09-25: Layers without a compressor use `main_inv_freq` and
            // mscale 1; compressor layers use `yarn_inv_freq` and
            // `yarn_rope_mscale`.
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
        ctx.gpu
            .synchronize(stream)
            .map_err(|e| anyhow::anyhow!("V4 attn: rope_yarn sync failed: {e}"))?;
        if diag_this {
            super::super::trait_impl::diag_norm(
                ctx.gpu,
                k_out,
                kv_dim as usize,
                stream,
                &format!("V4-prefill L{} K after RoPE token0", self.attn_layer_idx),
            );
            super::super::trait_impl::diag_norm(
                ctx.gpu,
                k_out.offset((nope * 2) as usize),
                (kv_dim - nope) as usize,
                stream,
                &format!(
                    "V4-prefill L{} K rope after RoPE token0",
                    self.attn_layer_idx
                ),
            );
            let last_k_offset = ((n - 1) * kv_dim * 2) as usize;
            super::super::trait_impl::diag_norm(
                ctx.gpu,
                k_out.offset(last_k_offset),
                kv_dim as usize,
                stream,
                &format!("V4-prefill L{} K after RoPE last", self.attn_layer_idx),
            );
            super::super::trait_impl::diag_norm(
                ctx.gpu,
                k_out.offset(last_k_offset + (nope * 2) as usize),
                (kv_dim - nope) as usize,
                stream,
                &format!("V4-prefill L{} K rope after RoPE last", self.attn_layer_idx),
            );
        }
        Ok(k_rope_tmp)
    }
}
