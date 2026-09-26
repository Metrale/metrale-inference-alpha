// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-26: The FFN-output half of `prefill_inner`: the post-FFN norms and residual adds,
//! the Gemma-4 dual FFN (dense MLP plus MoE) and the LongCat shortcut carry (consumer).
//!
//! Owner: model-layers (qwen3 attention).
//! Invariants: none beyond the types.

use anyhow::Result;
use metrale_gpu_runtime::gpu::DevicePtr;

use super::super::super::Qwen3AttentionLayer;
use crate::layer::ForwardContext;
use crate::layers::ops;

impl Qwen3AttentionLayer {
    /// 2026-09-26: Called by `prefill_inner` after the FFN wrote `dense_out`; ends with the
    /// layer's output in `hidden`.
    pub(super) fn prefill_ffn_residual(
        &self,
        hidden: DevicePtr,
        dense_out: DevicePtr,
        num_tokens: usize,
        n: u32,
        h: usize,
        eps: f32,
        ctx: &ForwardContext,
        stream: u64,
    ) -> Result<()> {
        // 2026-09-25: Gemma-4 dual FFN: a dense MLP and a MoE in the same layer.
        if let (Some(moe_ffn), Some(_pre_norm), Some(post_norm), Some(dense_norm)) = (
            &self.moe_ffn,
            &self.pre_moe_norm,
            &self.post_moe_out_norm,
            &self.post_dense_ffn_norm,
        ) {
            // 2026-09-25: The dense MLP output through `post_feedforward_layernorm_1`.
            ops::rms_norm(
                ctx.gpu,
                self.rms_norm_w_k,
                dense_out,
                dense_norm,
                dense_out,
                n,
                h as u32,
                eps,
                stream,
            )?;

            let scratch = ctx.buffers.attn_output();
            let nbytes = num_tokens * h * 2;
            ctx.gpu.copy_d2d_async(dense_out, scratch, nbytes, stream)?;

            // 2026-09-25: The MoE takes the residual stream: its router reads it
            // raw, and its experts apply `pre_feedforward_layernorm_2` themselves.
            moe_ffn
                .forward_prefill(hidden, num_tokens, ctx, stream)
                .map_err(|e| anyhow::anyhow!("moe_ffn.forward_prefill failed: {e}"))?;
            let moe_out = ctx.buffers.moe_output();
            ops::rms_norm(
                ctx.gpu,
                self.rms_norm_w_k,
                moe_out,
                post_norm,
                moe_out,
                n,
                h as u32,
                eps,
                stream,
            )?;

            ops::residual_add(
                ctx.gpu,
                self.residual_add_k,
                moe_out,
                scratch,
                (num_tokens * h) as u32,
                stream,
            )?;

            // 2026-09-25: `post_feedforward_layernorm` on the sum.
            if let Some(ref combined_norm) = self.post_ffn_out_norm {
                ops::rms_norm(
                    ctx.gpu,
                    self.rms_norm_w_k,
                    moe_out,
                    combined_norm,
                    moe_out,
                    n,
                    h as u32,
                    eps,
                    stream,
                )?;
            }

            ops::residual_add(
                ctx.gpu,
                self.residual_add_k,
                hidden,
                moe_out,
                (num_tokens * h) as u32,
                stream,
            )?;
        } else {
            if let Some(ref post_norm) = self.post_ffn_out_norm {
                ops::rms_norm(
                    ctx.gpu,
                    self.rms_norm_w_k,
                    dense_out,
                    post_norm,
                    dense_out,
                    n,
                    h as u32,
                    eps,
                    stream,
                )?;
            }
            ops::residual_add(
                ctx.gpu,
                self.residual_add_k,
                hidden,
                dense_out,
                (num_tokens * h) as u32,
                stream,
            )
            .map_err(|e| anyhow::anyhow!("residual_add failed: n={num_tokens} h={h}: {e}"))?;
            // 2026-09-25: LongCat shortcut MoE (consumer): add the paired previous
            // sublayer's stashed output at the end of this sublayer.
            if let Some((carry, cap)) = self.shortcut_carry_in {
                anyhow::ensure!(
                    num_tokens <= cap,
                    "shortcut carry capacity {cap} < prefill chunk {num_tokens}"
                );
                ops::residual_add(
                    ctx.gpu,
                    self.residual_add_k,
                    hidden,
                    carry,
                    (num_tokens * h) as u32,
                    stream,
                )?;
            }
        }
        Ok(())
    }
}
