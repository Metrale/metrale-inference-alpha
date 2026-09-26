// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-26: The router steps of `MoeLayer::forward`: the gate GEMV into `gate_logits`, and
//! the scored top-k (correction-bias sqrtsoftplus, softmax or sigmoid, else plain softmax)
//! for a layer without hash routing.
//!
//! Owner: model-layers (MoE).
//! Invariants: none beyond the types.

use super::*;

impl MoeLayer {
    pub(super) fn decode_gate_gemv(
        &self,
        router_in: DevicePtr,
        gate_logits: DevicePtr,
        h: u32,
        ctx: &ForwardContext,
        stream: u64,
    ) -> Result<()> {
        if let Some(ref nvfp4) = self.gate_nvfp4 {
            ops::w4a16_decode_gemv(
                ctx.gpu,
                self.w4a16_gemv,
                self.w4a16_gemv_sw,
                ctx.levers.gemv_sw,
                router_in,
                nvfp4,
                gate_logits,
                // 2026-09-25: `num_experts + zero_expert_num` (`init.rs`).
                self.router_logits_n,
                h,
                stream,
            )
        } else {
            ops::dense_gemv(
                ctx.gpu,
                self.dense_gemv,
                router_in,
                &self.weights.gate,
                gate_logits,
                self.router_logits_n,
                h,
                stream,
            )
        }
    }

    pub(super) fn decode_topk_scored(
        &self,
        gate_logits: DevicePtr,
        indices_dev: DevicePtr,
        weights_dev: DevicePtr,
        num_experts: u32,
        top_k: u32,
        ctx: &ForwardContext,
        stream: u64,
    ) -> Result<()> {
        if let Some(bias) = self.correction_bias_dev {
            if ctx.config.scoring_func == "sqrtsoftplus" {
                // 2026-09-25: scores = sqrtsoftplus(logits), indices =
                // topk(scores + bias), weights = scores[indices],
                // divided by their sum when `norm_topk_prob`, then
                // times `routed_scaling_factor`.
                ops::moe_topk_sqrtsoftplus(
                    ctx.gpu,
                    self.moe_topk_sqrtsoftplus_k,
                    gate_logits,
                    bias,
                    indices_dev,
                    weights_dev,
                    num_experts,
                    top_k,
                    ctx.config.norm_topk_prob,
                    ctx.config.routed_scaling_factor as f32,
                    stream,
                )
            } else if ctx.config.scoring_func == "softmax" {
                // 2026-09-25: Softmax scores plus the bias select the
                // experts; the unbiased scores weight them. Selected
                // zero experts fold into `zero_accum`, which the caller
                // applies with `apply_zero_expert`.
                ops::moe_topk_softmax_bias(
                    ctx.gpu,
                    self.moe_topk_softmax_bias_k,
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
                    stream,
                )
            } else {
                // 2026-09-25: scores = sigmoid(logits), indices =
                // topk(scores + bias), weights = scores[indices],
                // divided by their sum when `norm_topk_prob`, then
                // times `routed_scaling_factor`.
                ops::moe_topk_sigmoid(
                    ctx.gpu,
                    self.moe_topk_sigmoid_k,
                    gate_logits,
                    bias,
                    indices_dev,
                    weights_dev,
                    num_experts,
                    top_k,
                    ctx.config.norm_topk_prob,
                    ctx.config.routed_scaling_factor as f32,
                    stream,
                )
            }
        } else {
            ops::moe_topk_softmax(
                ctx.gpu,
                self.moe_topk,
                gate_logits,
                indices_dev,
                weights_dev,
                num_experts,
                top_k,
                ctx.config.norm_topk_prob,
                stream,
            )
        }
    }
}
