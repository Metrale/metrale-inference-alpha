// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-25: Correction-bias routers: the batched softmax + bias router for
//! prefill (LongCat: softmax over `num_experts + zero_expert_num` logits, the
//! bias used for selection only, zero-computation experts folded into
//! `zero_accum`) and the per-row dispatch for `forward_batched`. The
//! single-token decode calls the same kernel from `forward.rs`.
//!
//! Owner: model-layers (MoE).
//! Invariants: none beyond the types.

use anyhow::Result;
use metrale_gpu_runtime::gpu::DevicePtr;

use super::MoeLayer;
use crate::layer::ForwardContext;
use crate::layers::ops;

impl MoeLayer {
    #[allow(clippy::too_many_arguments)]
    pub(super) fn router_softmax_bias_batched(
        &self,
        gate_logits: DevicePtr,
        bias: DevicePtr,
        indices_dev: DevicePtr,
        weights_dev: DevicePtr,
        num_experts: u32,
        top_k: u32,
        n: u32,
        ctx: &ForwardContext,
        stream: u64,
    ) -> Result<()> {
        ops::moe_topk_softmax_bias_batched(
            ctx.gpu,
            self.moe_topk_softmax_bias_batched_k,
            gate_logits,
            bias,
            indices_dev,
            weights_dev,
            self.zero_accum_dev,
            self.router_logits_n,
            num_experts,
            top_k,
            ctx.config.norm_topk_prob,
            ctx.config.routed_scaling_factor as f32,
            n,
            stream,
        )
    }

    /// 2026-09-25: Correction-bias routing for row `t` of `forward_batched`, by
    /// `scoring_func`: `"sqrtsoftplus"`, `"softmax"` (softmax + bias, via
    /// `router_softmax_bias_one`), and sigmoid + bias for any other value.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn router_bias_one(
        &self,
        gate_t: DevicePtr,
        bias: DevicePtr,
        indices_dev: DevicePtr,
        weights_dev: DevicePtr,
        num_experts: u32,
        top_k: u32,
        t: usize,
        ctx: &ForwardContext,
        stream: u64,
    ) -> Result<()> {
        match ctx.config.scoring_func.as_str() {
            "sqrtsoftplus" => ops::moe_topk_sqrtsoftplus(
                ctx.gpu,
                self.moe_topk_sqrtsoftplus_k,
                gate_t,
                bias,
                indices_dev,
                weights_dev,
                num_experts,
                top_k,
                ctx.config.norm_topk_prob,
                ctx.config.routed_scaling_factor as f32,
                stream,
            ),
            "softmax" => self.router_softmax_bias_one(
                gate_t,
                bias,
                indices_dev,
                weights_dev,
                num_experts,
                top_k,
                t,
                ctx,
                stream,
            ),
            _ => ops::moe_topk_sigmoid(
                ctx.gpu,
                self.moe_topk_sigmoid_k,
                gate_t,
                bias,
                indices_dev,
                weights_dev,
                num_experts,
                top_k,
                ctx.config.norm_topk_prob,
                ctx.config.routed_scaling_factor as f32,
                stream,
            ),
        }
    }

    /// 2026-09-25: Softmax + bias routing for row `t` of `forward_batched`, which
    /// also serves FP8/BF16-expert prefills of 64 rows or fewer.
    ///
    /// `zero_accum` holds one f32 per row, and the single-token kernel writes
    /// element 0 of the pointer it gets, so row `t` passes `zero_accum + t`.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn router_softmax_bias_one(
        &self,
        gate_t: DevicePtr,
        bias: DevicePtr,
        indices_dev: DevicePtr,
        weights_dev: DevicePtr,
        num_experts: u32,
        top_k: u32,
        t: usize,
        ctx: &ForwardContext,
        stream: u64,
    ) -> Result<()> {
        ops::moe_topk_softmax_bias(
            ctx.gpu,
            self.moe_topk_softmax_bias_k,
            gate_t,
            bias,
            indices_dev,
            weights_dev,
            self.zero_accum_dev.offset(t * 4),
            self.router_logits_n,
            num_experts,
            top_k,
            ctx.config.norm_topk_prob,
            ctx.config.routed_scaling_factor as f32,
            stream,
        )
    }
}
