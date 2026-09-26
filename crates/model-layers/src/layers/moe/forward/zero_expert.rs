// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-26: `MoeLayer::fp32_routing_active` and `MoeLayer::apply_zero_expert`, moved whole from
//! `forward.rs`.
//!
//! Owner: model-layers (MoE).
//! Invariants: none beyond the types.

use super::*;

impl MoeLayer {
    /// 2026-09-25: Whether the FP32 routing path is on for this layer: the pre-MoE
    /// norm then also writes an FP32 copy of the router input
    /// (`residual_add_rms_norm_gatef32` into `moe_router_in_f32`), which
    /// `batched_gate_logits` reads through `dense_gemm_f32in`. Needs a dense gate
    /// (no `gate_nvfp4`), no correction bias, the `dense_gemm_f32in` and
    /// `moe_topk_f32` kernels, and, checked last, the `fp32_routing` lever (on
    /// only when `METRALE_FP32_ROUTING=1`).
    pub fn fp32_routing_active(&self, levers: &crate::layers::ops::ModelLevers) -> bool {
        self.gate_nvfp4.is_none()
            && self.correction_bias_dev.is_none()
            && self.dense_gemm_f32in.0 != 0
            && self.moe_topk_f32.0 != 0
            && levers.fp32_routing
    }

    /// 2026-09-25: Add the zero-expert term `out[t,:] += zero_accum[t] * x[t,:]`,
    /// where `zero_accum` holds the weights of the selected zero experts that
    /// the softmax+bias router folded out. It adds into `out` in place, so it
    /// runs after the routed blend has written `out`, for the tokens whose
    /// routing wrote `zero_accum`. No launch when `router_logits_n` equals
    /// `num_experts`; an error when the kernel is missing.
    pub fn apply_zero_expert(
        &self,
        out: metrale_gpu_runtime::gpu::DevicePtr,
        x: metrale_gpu_runtime::gpu::DevicePtr,
        n: u32,
        ctx: &ForwardContext,
        stream: u64,
    ) -> Result<()> {
        if self.router_logits_n as usize == ctx.config.num_experts {
            return Ok(());
        }
        anyhow::ensure!(
            self.moe_zero_expert_add_k.0 != 0,
            "zero-expert model but moe_zero_expert_add kernel is absent from this build"
        );
        ops::moe_zero_expert_add(
            ctx.gpu,
            self.moe_zero_expert_add_k,
            out,
            x,
            self.zero_accum_dev,
            n,
            ctx.config.hidden_size as u32,
            stream,
        )
    }
}
