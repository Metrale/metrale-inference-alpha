// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-26: De-rotation of the `attention_forward_v4` attention output: V = K carries rotated
//! rope, so the output's rope part is rotated back before the O projection.
//!
//! Owner: model-layers attention decode.
//! Invariants: none beyond the types.

use anyhow::Result;
use metrale_gpu_runtime::gpu::DevicePtr;

use super::super::super::{MlaWeights, Qwen3AttentionLayer};
use crate::layer::{AttnMetadataDev, ForwardContext};
use crate::layers::ops;

impl Qwen3AttentionLayer {
    /// 2026-09-26: Extract, conjugate-rotate at `meta.positions`, and write back the rope part
    /// of each head of `attn_out`.
    pub(super) fn v4_derotate_attn_out(
        &self,
        ctx: &ForwardContext,
        mla: &MlaWeights,
        meta: &AttnMetadataDev,
        attn_out: DevicePtr,
        nq: u32,
        hd: u32,
        mla_rope: u32,
        stream: u64,
    ) -> Result<()> {
        let o_rope_tmp = ctx.buffers.ssm_conv_out_f32();
        ops::mla_q_rope_extract_batched(
            ctx.gpu,
            self.mla_q_rope_extract_batched_k,
            attn_out,
            o_rope_tmp,
            1,
            nq,
            hd,
            mla.nope as u32,
            mla_rope,
            nq * hd,
            stream,
        )?;
        ops::rope_yarn(
            ctx.gpu,
            self.rope_yarn_interleaved_inv_k,
            o_rope_tmp,
            o_rope_tmp,
            meta.positions,
            1,
            nq,
            0,
            mla_rope,
            mla_rope,
            // 2026-09-25: The same frequencies and mscale as the Q/K RoPE above.
            if mla.compressor.is_none() {
                mla.main_inv_freq
            } else {
                mla.yarn_inv_freq
            },
            if mla.compressor.is_none() {
                1.0f32
            } else {
                super::super::super::helpers::yarn_rope_mscale(ctx.config)
            },
            stream,
        )?;
        ops::mla_q_rope_writeback_batched(
            ctx.gpu,
            self.mla_q_rope_writeback_batched_k,
            o_rope_tmp,
            attn_out,
            1,
            nq,
            hd,
            mla.nope as u32,
            mla_rope,
            nq * hd,
            stream,
        )?;
        Ok(())
    }
}
