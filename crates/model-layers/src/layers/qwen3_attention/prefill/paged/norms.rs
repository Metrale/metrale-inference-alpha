// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-26: The Q split and the Q/K/V norms of `prefill_attention_paged`: de-interleave Q from
//! the gated projection or copy it out, then the Q, K and V RMS norms the layer was loaded
//! with.
//!
//! Owner: model-layers (attention).
//! Invariants: none beyond the types.

use anyhow::Result;
use metrale_gpu_runtime::gpu::DevicePtr;

use super::super::super::Qwen3AttentionLayer;
use crate::layer::ForwardContext;
use crate::layers::ops;

impl Qwen3AttentionLayer {
    /// 2026-09-26: Called by `prefill_attention_paged` after the Q/K/V projection and the
    /// fused-K raw copy, before RoPE.
    pub(super) fn prefill_paged_split_and_norm(
        &self,
        ctx: &ForwardContext,
        qg_out: DevicePtr,
        q_contiguous: DevicePtr,
        k_contiguous: DevicePtr,
        v_contiguous: DevicePtr,
        n: u32,
        nq: u32,
        nkv: u32,
        hd: u32,
        q_proj_dim: usize,
        eps: f32,
        num_tokens: usize,
        q_dim: usize,
        bf16: usize,
        stream: u64,
    ) -> Result<()> {
        if self.gated && !self.attn.q_norm.weight.is_null() {
            ops::deinterleave_qg_split_qnorm(
                ctx.gpu,
                self.deinterleave_qg_split_qnorm_k,
                qg_out,
                q_contiguous,
                self.attn.q_norm.weight,
                n,
                nq,
                hd,
                q_proj_dim as u32,
                eps,
                stream,
            )?;
        } else if self.gated {
            ops::deinterleave_qg_split(
                ctx.gpu,
                self.deinterleave_qg_split_k,
                qg_out,
                q_contiguous,
                n,
                nq,
                hd,
                q_proj_dim as u32,
                stream,
            )?;
        } else if let Some(mla_ref) = self.mla.as_ref() {
            // 2026-09-25: Not reached: MLA layers returned above.
            let mla_nope_sz = mla_ref.nope;
            let mla_rope_sz = mla_ref.rope;
            for t in 0..num_tokens {
                for head_idx in 0..nq as usize {
                    let src = qg_out.offset((t * q_dim + head_idx * hd as usize) * bf16);
                    let dst = q_contiguous.offset((t * q_dim + head_idx * hd as usize) * bf16);
                    ctx.gpu.copy_d2d_async(
                        src.offset(mla_nope_sz * bf16),
                        dst,
                        mla_rope_sz * bf16,
                        stream,
                    )?;
                    ctx.gpu.copy_d2d_async(
                        src,
                        dst.offset(mla_rope_sz * bf16),
                        mla_nope_sz * bf16,
                        stream,
                    )?;
                }
            }
        } else {
            ctx.gpu
                .copy_d2d_async(qg_out, q_contiguous, num_tokens * q_dim * bf16, stream)?;
            if let Some(ref q_norm_full) = self.attn.q_norm_full {
                // 2026-09-25: `q_norm_full` (set by the MiniMax loader): one RMS
                // norm over each token's whole `nq * hd` Q row.
                ops::rms_norm(
                    ctx.gpu,
                    self.rms_norm_w_k,
                    q_contiguous,
                    q_norm_full,
                    q_contiguous,
                    n,
                    nq * hd,
                    eps,
                    stream,
                )?;
            } else if !self.attn.q_norm.weight.is_null() {
                ops::rms_norm(
                    ctx.gpu,
                    self.rms_norm_w_k,
                    q_contiguous,
                    &self.attn.q_norm,
                    q_contiguous,
                    nq * n,
                    hd,
                    eps,
                    stream,
                )?;
            }
        }
        if let Some(ref k_norm_full) = self.attn.k_norm_full {
            ops::rms_norm(
                ctx.gpu,
                self.rms_norm_w_k,
                k_contiguous,
                k_norm_full,
                k_contiguous,
                n,
                nkv * hd,
                eps,
                stream,
            )?;
        } else if !self.attn.k_norm.weight.is_null() {
            ops::rms_norm(
                ctx.gpu,
                self.rms_norm_w_k,
                k_contiguous,
                &self.attn.k_norm,
                k_contiguous,
                nkv * n,
                hd,
                eps,
                stream,
            )?;
        }

        // 2026-09-25: The V norm, when `set_k_eq_v` or `set_v_norm` installed
        // one: an RMS norm of each V head with `v_norm_weight`. V gets no RoPE.
        if let Some(v_norm_w) = self.v_norm_weight.as_ref() {
            ops::rms_norm(
                ctx.gpu,
                self.rms_norm_w_k,
                v_contiguous,
                v_norm_w,
                v_contiguous,
                nkv * n,
                hd,
                eps,
                stream,
            )?;
        }
        Ok(())
    }
}
