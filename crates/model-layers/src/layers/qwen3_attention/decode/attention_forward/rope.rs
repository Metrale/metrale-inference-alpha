// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-26: RoPE on Q and K for single-token `attention_forward`.
//!
//! Owner: model-layers attention decode.
//! Invariants: none beyond the types.

use anyhow::Result;
use metrale_gpu_runtime::gpu::DevicePtr;

use super::super::super::Qwen3AttentionLayer;
use crate::layer::{AttnMetadataDev, ForwardContext};
use crate::layers::ops;

impl Qwen3AttentionLayer {
    /// 2026-09-26: The RoPE variant this layer was loaded with: YaRN-scaled, proportional,
    /// interleaved MRoPE, or plain RoPE (Q only when `fused_k_fp8`, which rotates K itself).
    pub(super) fn attention_forward_rope(
        &self,
        ctx: &ForwardContext,
        meta: &AttnMetadataDev,
        q_out: DevicePtr,
        k_out: DevicePtr,
        nq: u32,
        nkv: u32,
        hd: u32,
        rotary_dim: u32,
        fused_k_fp8: bool,
        stream: u64,
    ) -> Result<()> {
        if self.mla.is_some() {
            // 2026-09-25: Unreachable: MLA layers returned above.
        } else if !self.yarn_inv_freq.is_null() {
            ops::rope_yarn_scaled(
                ctx.gpu,
                self.rope_yarn_scaled_k,
                q_out,
                k_out,
                meta.positions,
                1,
                nq,
                nkv,
                hd,
                self.rotary_dim_override
                    .unwrap_or(ctx.config.rotary_dim() as u32),
                self.yarn_inv_freq,
                self.yarn_attention_factor,
                stream,
            )?;
        } else if self.rope_proportional && self.rope_proportional_k.0 != 0 {
            // 2026-09-25: Proportional RoPE: `rotary_dim_override` carries `rope_angles`.
            let rope_angles = self
                .rotary_dim_override
                .unwrap_or(ctx.config.rotary_dim() as u32);
            ops::rope_proportional(
                ctx.gpu,
                self.rope_proportional_k,
                q_out,
                k_out,
                meta.positions,
                1,
                nq,
                nkv,
                hd,
                rope_angles,
                self.rope_theta_override
                    .unwrap_or(ctx.config.rope_theta as f32),
                stream,
            )?;
        } else if self.mrope_interleaved && self.rope_mrope_interleaved_k.0 != 0 {
            ops::rope_mrope_interleaved(
                ctx.gpu,
                self.rope_mrope_interleaved_k,
                q_out,
                k_out,
                meta.positions,
                meta.positions_h,
                meta.positions_w,
                1,
                nq,
                nkv,
                hd,
                self.rotary_dim_override
                    .unwrap_or(ctx.config.rotary_dim() as u32),
                self.rope_theta_override
                    .unwrap_or(ctx.config.rope_theta as f32),
                stream,
            )?;
        } else {
            // 2026-09-25: On the fused FP8 path `num_kv_heads = 0` makes the launch Q-only, and K
            // stays unrotated because the fused writer rotates the raw projection itself.
            ops::rope(
                ctx.gpu,
                self.rope_k,
                q_out,
                k_out,
                meta.positions,
                1,
                nq,
                if fused_k_fp8 { 0 } else { nkv },
                hd,
                rotary_dim,
                self.rope_theta_override
                    .unwrap_or(ctx.config.rope_theta as f32),
                stream,
            )?;
        }
        Ok(())
    }
}
