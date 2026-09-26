// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-25: `MoeLayer::forward_k3`, the MoE block for three rows at once (K=3 verify).
//!
//! Owner: model-layers (MoE).
//! Invariants: none beyond the types.

use super::*;

impl MoeLayer {
    /// 2026-09-25: MoE for three rows: `input` is the normed [3, H] BF16 MoE input.
    ///
    /// Gate projection for all rows, top-K, then the fused expert gate+up,
    /// silu+down and weighted-sum+blend kernels. Output: `moe_output()` [3, H].
    /// LoRA, BF16 experts, E8M0 experts and an unsupported mixed shared expert
    /// take `forward_batched` instead.
    pub fn forward_k3(
        &self,
        input: DevicePtr, // 2026-09-25: [3, H] BF16
        ctx: &ForwardContext,
        stream: u64,
    ) -> Result<()> {
        // 2026-09-25: The batch3 kernels do not handle zero-computation experts
        // (router width above num_experts), so refuse instead of mis-routing.
        anyhow::ensure!(
            self.router_logits_n as usize == ctx.config.num_experts,
            "zero-expert MoE routing is not wired on this dispatch variant yet (forward_k3)"
        );

        // 2026-09-25: The fused batch3 kernels have no LoRA fold. With a MoE
        // adapter installed, `forward_batched` applies the router and expert
        // deltas per row and writes the same moe_output [3, H].
        if self.lora.is_some() {
            return self.forward_batched(input, 3, ctx, stream);
        }
        // 2026-09-25: BF16 (FP8-dequant-on-load) experts have no batch3 kernel,
        // and the loader frees their FP8 sources, so they always take
        // `forward_batched`.
        if self.bf16_gate_weight_ptrs.is_some() {
            return self.forward_batched(input, 3, ctx, stream);
        }
        // 2026-09-25: Mixed NVFP4-routed / BF16-shared experts: only the
        // unified-T branch (both batch3_t kernels present, no EP) serves them,
        // with NULL shared weights and a BF16 shared pass afterwards.
        let mixed_bf16_shared = self.has_mixed_bf16_shared_expert();
        if mixed_bf16_shared
            && !(self.use_t_layout_for_decode()
                && self.moe_expert_gate_up_shared_batch3_t_k.0 != 0
                && self.moe_expert_silu_down_shared_batch3_t_k.0 != 0
                && !(ctx.comm.is_some() && ctx.config.ep_world_size > 1))
        {
            return self.forward_batched(input, 3, ctx, stream);
        }
        // 2026-09-25: E8M0 (native MXFP4, per-32 scale) routed experts must not
        // reach `moe_expert_gate_up_shared_batch3_t`: that NVFP4 kernel has
        // GROUP_SIZE 16, so it reads K/16 scale rows from a K/32-row E8M0 buffer
        // and decodes them as E4M3. `forward_batched` selects the `_e8m0`
        // kernels through `e8m0_or` on its `use_t_layout_for_prefill` branch.
        if k3_e8m0_needs_per_token(self.experts_scale_kind) {
            return self.forward_batched(input, 3, ctx, stream);
        }

        let h = ctx.config.hidden_size as u32;
        let inter = ctx.config.moe_intermediate_size as u32;
        let num_experts = ctx.config.num_experts as u32;
        let top_k = ctx.config.num_experts_per_tok as u32;

        let router_in = self.router_input(input, 3, h, ctx, stream)?;
        let gate_logits = ctx.buffers.gate_logits();
        if let Some(ref nvfp4) = self.gate_nvfp4 {
            ops::w4a16_gemv_batch3(
                ctx.gpu,
                self.w4a16_gemv_batch3,
                router_in,
                nvfp4,
                gate_logits,
                num_experts,
                h,
                stream,
            )?;
        } else {
            ops::dense_gemm(
                ctx.gpu,
                self.dense_gemm,
                router_in,
                &self.weights.gate,
                gate_logits,
                3,
                num_experts,
                h,
                stream,
            )?;
        }

        // 2026-09-25: scratch holds indices [3, top_k] u32, then weights [3, top_k]
        // f32. With a correction bias this path always uses sigmoid scoring; it
        // has no sqrtsoftplus branch.
        let scratch = ctx.buffers.scratch();
        let indices_dev = scratch;
        let weights_dev = scratch.offset(3 * top_k as usize * 4);
        if let Some(bias) = self.correction_bias_dev {
            ops::moe_topk_sigmoid_batched(
                ctx.gpu,
                self.moe_topk_sigmoid_batched_k,
                gate_logits,
                bias,
                indices_dev,
                weights_dev,
                num_experts,
                top_k,
                ctx.config.norm_topk_prob,
                ctx.config.routed_scaling_factor as f32,
                3,
                stream,
            )?;
        } else {
            ops::moe_topk_softmax_batched(
                ctx.gpu,
                self.moe_topk_batched,
                gate_logits,
                indices_dev,
                weights_dev,
                num_experts,
                top_k,
                ctx.config.norm_topk_prob,
                3,
                stream,
            )?;
        }

        super::union_stats::maybe_sample_expert_union(ctx, indices_dev, 3, top_k as usize, stream);

        let expert_gate_out = ctx.buffers.expert_gate_out();
        let expert_up_out = ctx.buffers.expert_up_out();
        let shared_gate_scratch = ctx.buffers.logits();
        let shared_up_scratch = ctx.buffers.ssm_qkvz();
        let expert_down_out = ctx.buffers.expert_down_out();
        let shared_down_out = ctx.buffers.attn_output();
        let output = ctx.buffers.moe_output();

        let is_ep = ctx.comm.is_some() && ctx.config.ep_world_size > 1;

        if let (Some(gp), Some(up), Some(dp), Some(sh)) = (
            &self.fp8_gate_weight_ptrs,
            &self.fp8_up_weight_ptrs,
            &self.fp8_down_weight_ptrs,
            &self.fp8_shared_expert,
        ) {
            ops::moe_expert_gate_up_shared_fp8_batch3(
                ctx.gpu,
                self.moe_expert_gate_up_shared_fp8_batch3,
                input,
                gp.weight_ptrs,
                gp.scale_ptrs,
                expert_gate_out,
                up.weight_ptrs,
                up.scale_ptrs,
                expert_up_out,
                indices_dev,
                &sh.gate_proj,
                shared_gate_scratch,
                &sh.up_proj,
                shared_up_scratch,
                inter,
                h,
                top_k,
                stream,
            )?;
            ops::moe_expert_silu_down_shared_fp8_batch3(
                ctx.gpu,
                self.moe_expert_silu_down_shared_fp8_batch3,
                expert_gate_out,
                expert_up_out,
                dp.weight_ptrs,
                dp.scale_ptrs,
                expert_down_out,
                indices_dev,
                shared_gate_scratch,
                shared_up_scratch,
                &sh.down_proj,
                shared_down_out,
                h,
                inter,
                top_k,
                stream,
            )?;
            // 2026-09-25: With EP, blend a zeroed shared term (expert_gate_out is
            // not read after silu_down); the shared expert is added once after
            // the all-reduce below.
            let shared_for_blend = if is_ep && !shared_down_out.is_null() {
                ctx.gpu
                    .memset_async(expert_gate_out, 0, 3 * h as usize * 2, stream)?;
                expert_gate_out
            } else {
                shared_down_out
            };
            ops::moe_weighted_sum_blend_batch3(
                ctx.gpu,
                self.moe_weighted_sum_blend_fp8_batch3,
                output,
                expert_down_out,
                weights_dev,
                shared_for_blend,
                input,
                self.weights.shared_expert_gate.weight,
                h,
                top_k,
                h,
                stream,
            )?;
        } else if self.use_t_layout_for_decode() {
            // 2026-09-25: Transposed `[K/2, N]` tables; `use_t_layout_for_decode()`
            // is false in hybrid mode.
            let gate_t = self
                .gate_ptrs_t
                .as_ref()
                .expect("gate_ptrs_t under unified_t");
            let up_t = self.up_ptrs_t.as_ref().expect("up_ptrs_t under unified_t");
            let down_t = self
                .down_ptrs_t
                .as_ref()
                .expect("down_ptrs_t under unified_t");
            let null_qw = QuantizedWeight::null();
            // 2026-09-25: Mixed config: NULL shared weights make the kernels write
            // zeros to the shared outputs; the BF16 shared pass after silu_down_t
            // replaces them.
            let (sh_gate_t, sh_up_t, sh_down_t) = if mixed_bf16_shared {
                (&null_qw, &null_qw, &null_qw)
            } else {
                (
                    self.shared_gate_t.as_ref().unwrap_or(&null_qw),
                    self.shared_up_t.as_ref().unwrap_or(&null_qw),
                    self.shared_down_t.as_ref().unwrap_or(&null_qw),
                )
            };
            ops::moe_expert_gate_up_shared_batch3_t(
                ctx.gpu,
                self.moe_expert_gate_up_shared_batch3_t_k,
                input,
                gate_t.packed_ptrs,
                gate_t.scale_ptrs,
                gate_t.scale2_vals,
                expert_gate_out,
                up_t.packed_ptrs,
                up_t.scale_ptrs,
                up_t.scale2_vals,
                expert_up_out,
                indices_dev,
                sh_gate_t,
                shared_gate_scratch,
                sh_up_t,
                shared_up_scratch,
                inter,
                h,
                top_k,
                stream,
            )?;
            ops::moe_expert_silu_down_shared_batch3_t(
                ctx.gpu,
                self.moe_expert_silu_down_shared_batch3_t_k,
                expert_gate_out,
                expert_up_out,
                down_t.packed_ptrs,
                down_t.scale_ptrs,
                down_t.scale2_vals,
                expert_down_out,
                indices_dev,
                shared_gate_scratch,
                shared_up_scratch,
                sh_down_t,
                shared_down_out,
                h,
                inter,
                top_k,
                stream,
            )?;
            if mixed_bf16_shared {
                let shared_inter = ctx.config.shared_expert_intermediate_size as u32;
                self.run_bf16_shared_expert(
                    input,
                    3,
                    h,
                    shared_inter,
                    shared_gate_scratch,
                    shared_up_scratch,
                    shared_down_out,
                    ctx,
                    stream,
                )?;
            }
            // 2026-09-25: With EP, blend a zeroed shared term; the shared expert
            // is added after the all-reduce below.
            let shared_for_blend = if is_ep && !shared_down_out.is_null() {
                ctx.gpu
                    .memset_async(expert_gate_out, 0, 3 * h as usize * 2, stream)?;
                expert_gate_out
            } else {
                shared_down_out
            };
            ops::moe_weighted_sum_blend_batch3(
                ctx.gpu,
                self.moe_weighted_sum_blend_batch3,
                output,
                expert_down_out,
                weights_dev,
                shared_for_blend,
                input,
                self.weights.shared_expert_gate.weight,
                h,
                top_k,
                h,
                stream,
            )?;
        } else {
            ops::moe_expert_gate_up_shared_batch3(
                ctx.gpu,
                self.moe_expert_gate_up_shared_batch3,
                input,
                self.gate_ptrs.packed_ptrs,
                self.gate_ptrs.scale_ptrs,
                self.gate_ptrs.scale2_vals,
                expert_gate_out,
                self.up_ptrs.packed_ptrs,
                self.up_ptrs.scale_ptrs,
                self.up_ptrs.scale2_vals,
                expert_up_out,
                indices_dev,
                &self.weights.shared_expert.gate_proj,
                shared_gate_scratch,
                &self.weights.shared_expert.up_proj,
                shared_up_scratch,
                inter,
                h,
                top_k,
                stream,
            )?;
            ops::moe_expert_silu_down_shared_batch3(
                ctx.gpu,
                self.moe_expert_silu_down_shared_batch3,
                expert_gate_out,
                expert_up_out,
                self.down_ptrs.packed_ptrs,
                self.down_ptrs.scale_ptrs,
                self.down_ptrs.scale2_vals,
                expert_down_out,
                indices_dev,
                shared_gate_scratch,
                shared_up_scratch,
                &self.weights.shared_expert.down_proj,
                shared_down_out,
                h,
                inter,
                top_k,
                stream,
            )?;
            // 2026-09-25: With EP, blend a zeroed shared term; the shared expert
            // is added after the all-reduce below.
            let shared_for_blend = if is_ep && !shared_down_out.is_null() {
                ctx.gpu
                    .memset_async(expert_gate_out, 0, 3 * h as usize * 2, stream)?;
                expert_gate_out
            } else {
                shared_down_out
            };
            ops::moe_weighted_sum_blend_batch3(
                ctx.gpu,
                self.moe_weighted_sum_blend_batch3,
                output,
                expert_down_out,
                weights_dev,
                shared_for_blend,
                input,
                self.weights.shared_expert_gate.weight,
                h,
                top_k,
                h,
                stream,
            )?;
        }

        if let Some(comm) = ctx.comm
            && ctx.config.ep_world_size > 1
        {
            if ctx.graph_capture {
                comm.all_reduce(output.0, 3 * h as usize * 2)?;
            } else {
                comm.all_reduce_async(output.0, 3 * h as usize * 2, stream)?;
            }
            // 2026-09-25: Add the shared expert once, after the reduce; gated
            // when the layer has a shared_expert_gate weight.
            if !shared_down_out.is_null() {
                if self.weights.shared_expert_gate.weight.0 == 0 {
                    ops::residual_add(
                        ctx.gpu,
                        self.residual_add,
                        output,
                        shared_down_out,
                        3 * h,
                        stream,
                    )?;
                } else {
                    ops::moe_batched_blend(
                        ctx.gpu,
                        self.moe_batched_blend,
                        output,
                        shared_down_out,
                        input,
                        self.weights.shared_expert_gate.weight,
                        h,
                        3,
                        stream,
                    )?;
                }
            }
        }

        Ok(())
    }
}

/// 2026-09-25: True when `forward_k3` must send the rows to `forward_batched`:
/// E8M0 routed experts cannot use the GROUP_SIZE-16 batch3 kernel.
pub(crate) fn k3_e8m0_needs_per_token(scale_kind: crate::weight_map::WeightQuantFormat) -> bool {
    matches!(scale_kind, crate::weight_map::WeightQuantFormat::Mxfp4E8m0)
}

#[cfg(test)]
#[path = "forward_k3_dispatch_tests.rs"]
mod k3_dispatch_tests;
