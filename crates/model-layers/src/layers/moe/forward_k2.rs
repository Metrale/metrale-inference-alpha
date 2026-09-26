// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-25: `MoeLayer::forward_k2`, the MoE block for two rows at once (K=2 verify).
//!
//! Owner: model-layers (MoE).
//! Invariants: none beyond the types.

use anyhow::Context as _;

use super::*;

mod originals;
mod unified_t;

impl MoeLayer {
    /// 2026-09-25: MoE for two rows: `input` is the normed [2, H] BF16 MoE input.
    ///
    /// Gate projection for both rows, top-K, then the fused expert gate+up,
    /// silu+down and weighted-sum+blend kernels. The shared expert's scratch
    /// reuses the `logits` and `ssm_qkvz` buffers. Output: `moe_output()` [2, H].
    /// LoRA, BF16 experts that cannot use the BF16 batch2 kernels, E8M0 experts
    /// and an unsupported mixed shared expert take `forward_batched` instead.
    pub fn forward_k2(
        &self,
        input: DevicePtr, // 2026-09-25: [2, H] BF16
        ctx: &ForwardContext,
        stream: u64,
    ) -> Result<()> {
        // 2026-09-25: The batch2 kernels do not handle zero-computation experts
        // (router width above num_experts), so refuse instead of mis-routing.
        anyhow::ensure!(
            self.router_logits_n as usize == ctx.config.num_experts,
            "zero-expert MoE routing is not wired on this dispatch variant yet (forward_k2)"
        );

        // 2026-09-25: The fused batch2 kernels have no LoRA fold. With a MoE
        // adapter installed, `forward_batched` applies the router and expert
        // deltas per row and writes the same moe_output [2, H].
        if self.lora.is_some() {
            return self.forward_batched(input, 2, ctx, stream);
        }
        // 2026-09-25: BF16 (FP8-dequant-on-load) experts run on the BF16 batch2
        // kernels when both are present and there is no EP, else on
        // `forward_batched`. The loader frees their FP8 sources, so the FP8 and
        // NVFP4 branches below must not see them.
        let is_ep = ctx.comm.is_some() && ctx.config.ep_world_size > 1;
        let use_bf16_batch2 = self.bf16_gate_weight_ptrs.is_some()
            && self.moe_expert_gate_up_shared_bf16_batch2_k.0 != 0
            && self.moe_expert_silu_down_shared_bf16_batch2_k.0 != 0
            && !is_ep;
        if self.bf16_gate_weight_ptrs.is_some() && !use_bf16_batch2 {
            return self.forward_batched(input, 2, ctx, stream);
        }
        // 2026-09-25: E8M0 (native MXFP4, per-32 scale) routed experts must not
        // reach `moe_expert_gate_up_shared_batch2_t`: that NVFP4 kernel has
        // GROUP_SIZE 16, so it reads K/16 scale rows from a K/32-row E8M0 buffer
        // and decodes them as E4M3. There is no E8M0 batch2 kernel;
        // `forward_batched` selects the `_e8m0` kernels through `e8m0_or` on its
        // `use_t_layout_for_prefill` branch.
        if k2_e8m0_needs_per_token(self.experts_scale_kind) {
            return self.forward_batched(input, 2, ctx, stream);
        }
        // 2026-09-25: Mixed NVFP4-routed / BF16-shared experts (the Laguna
        // loader): the batch2 kernels cannot compute a BF16 shared expert. With
        // either fused layout (unified-T or originals) and no EP, the routed half
        // runs on those kernels with NULL shared weights and the shared expert
        // runs afterwards as one batched BF16 pass; otherwise `forward_batched`.
        let mixed_bf16_shared = self.has_mixed_bf16_shared_expert();
        let mixed_t_ok = self.use_t_layout_for_decode()
            && self.moe_expert_gate_up_shared_batch2_t_k.0 != 0
            && self.moe_expert_silu_down_shared_batch2_t_k.0 != 0;
        let mixed_orig_ok = !self.use_t_layout_for_decode()
            && self.moe_expert_gate_up_shared_batch2.0 != 0
            && self.moe_expert_silu_down_shared_batch2.0 != 0
            && !self.gate_ptrs.packed_ptrs.is_null();
        if mixed_bf16_shared && !((mixed_t_ok || mixed_orig_ok) && !is_ep) {
            return self.forward_batched(input, 2, ctx, stream);
        }

        let h = ctx.config.hidden_size as u32;
        let inter = ctx.config.moe_intermediate_size as u32;
        let num_experts = ctx.config.num_experts as u32;
        let top_k = ctx.config.num_experts_per_tok as u32;

        // 2026-09-25: With `k2_diag` (METRALE_K2_DIAG=1) the stream is synchronised
        // after each stage; the context of the first failing sync names the stage.
        let k2_diag = ctx.levers.k2_diag;
        if k2_diag {
            ctx.gpu
                .synchronize(stream)
                .context("K2 ENTRY: attention+norm BEFORE forward_k2")?;
        }

        let router_in = self.router_input(input, 2, h, ctx, stream)?;
        let gate_logits = ctx.buffers.gate_logits();
        if let Some(ref nvfp4) = self.gate_nvfp4 {
            ops::w4a16_gemv_batch2(
                ctx.gpu,
                self.w4a16_gemv_batch2,
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
                2,
                num_experts,
                h,
                stream,
            )?;
        }

        // 2026-09-25: scratch holds indices [2, top_k] u32, then weights [2, top_k] f32.
        let scratch = ctx.buffers.scratch();
        let indices_dev = scratch;
        let weights_dev = scratch.offset(2 * top_k as usize * 4);
        if let Some(bias) = self.correction_bias_dev {
            // 2026-09-25: With a correction bias: sqrt(softplus) scoring for
            // `scoring_func == "sqrtsoftplus"`, sigmoid otherwise, the same
            // branch as the single-token and prefill routers.
            if ctx.config.scoring_func == "sqrtsoftplus" {
                // 2026-09-25: One non-batched launch per row; gate_logits rows
                // are num_experts BF16 apart.
                for t in 0..2usize {
                    ops::moe_topk_sqrtsoftplus(
                        ctx.gpu,
                        self.moe_topk_sqrtsoftplus_k,
                        gate_logits.offset(t * num_experts as usize * 2),
                        bias,
                        indices_dev.offset(t * top_k as usize * 4),
                        weights_dev.offset(t * top_k as usize * 4),
                        num_experts,
                        top_k,
                        ctx.config.norm_topk_prob,
                        ctx.config.routed_scaling_factor as f32,
                        stream,
                    )?;
                }
            } else {
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
                    2,
                    stream,
                )?;
            }
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
                2,
                stream,
            )?;
        }
        super::union_stats::maybe_sample_expert_union(ctx, indices_dev, 2, top_k as usize, stream);

        if k2_diag {
            ctx.gpu
                .synchronize(stream)
                .context("K2: gate-GEMV + topk")?;
        }

        let expert_gate_out = ctx.buffers.expert_gate_out();
        let expert_up_out = ctx.buffers.expert_up_out();
        let shared_gate_scratch = ctx.buffers.logits();
        let shared_up_scratch = ctx.buffers.ssm_qkvz();
        let expert_down_out = ctx.buffers.expert_down_out();
        let shared_down_out = ctx.buffers.attn_output();
        let output = ctx.buffers.moe_output();

        if use_bf16_batch2
            && let (Some(gp), Some(up), Some(dp), Some(shared)) = (
                self.bf16_gate_weight_ptrs,
                self.bf16_up_weight_ptrs,
                self.bf16_down_weight_ptrs,
                self.bf16_shared_expert,
            )
        {
            // 2026-09-25: Non-EP only: `use_bf16_batch2` requires `!is_ep`.
            ops::moe_expert_gate_up_shared_bf16_batch2(
                ctx.gpu,
                self.moe_expert_gate_up_shared_bf16_batch2_k,
                input,
                gp,
                expert_gate_out,
                up,
                expert_up_out,
                indices_dev,
                shared.gate_proj.weight,
                shared_gate_scratch,
                shared.up_proj.weight,
                shared_up_scratch,
                inter,
                h,
                top_k,
                stream,
            )?;
            ops::moe_expert_silu_down_shared_bf16_batch2(
                ctx.gpu,
                self.moe_expert_silu_down_shared_bf16_batch2_k,
                expert_gate_out,
                expert_up_out,
                dp,
                expert_down_out,
                indices_dev,
                shared_gate_scratch,
                shared_up_scratch,
                shared.down_proj.weight,
                shared_down_out,
                h,
                inter,
                top_k,
                stream,
            )?;
            ops::moe_weighted_sum_blend_batch2(
                ctx.gpu,
                self.moe_weighted_sum_blend_batch2,
                output,
                expert_down_out,
                weights_dev,
                shared_down_out,
                input,
                self.weights.shared_expert_gate.weight,
                h,
                top_k,
                h,
                stream,
            )?;
        } else if let (Some(gp), Some(up), Some(dp), Some(sh)) = (
            &self.fp8_gate_weight_ptrs,
            &self.fp8_up_weight_ptrs,
            &self.fp8_down_weight_ptrs,
            &self.fp8_shared_expert,
        ) {
            ops::moe_expert_gate_up_shared_fp8_batch2(
                ctx.gpu,
                self.moe_expert_gate_up_shared_fp8_batch2,
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
            ops::moe_expert_silu_down_shared_fp8_batch2(
                ctx.gpu,
                self.moe_expert_silu_down_shared_fp8_batch2,
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
                    .memset_async(expert_gate_out, 0, 2 * h as usize * 2, stream)?;
                expert_gate_out
            } else {
                shared_down_out
            };
            ops::moe_weighted_sum_blend_batch2(
                ctx.gpu,
                self.moe_weighted_sum_blend_fp8_batch2,
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
            self.forward_k2_unified_t(
                input,
                indices_dev,
                weights_dev,
                expert_gate_out,
                expert_up_out,
                expert_down_out,
                shared_gate_scratch,
                shared_up_scratch,
                shared_down_out,
                output,
                inter,
                h,
                top_k,
                is_ep,
                mixed_bf16_shared,
                ctx,
                stream,
            )?;
        } else {
            self.forward_k2_originals(
                input,
                indices_dev,
                weights_dev,
                expert_gate_out,
                expert_up_out,
                expert_down_out,
                shared_gate_scratch,
                shared_up_scratch,
                shared_down_out,
                output,
                inter,
                h,
                top_k,
                is_ep,
                mixed_bf16_shared,
                ctx,
                stream,
            )?;
        }

        if k2_diag {
            ctx.gpu
                .synchronize(stream)
                .context("K2: expert dispatch (gate_up/silu_down/blend)")?;
        }

        if let Some(comm) = ctx.comm
            && ctx.config.ep_world_size > 1
        {
            if ctx.graph_capture {
                comm.all_reduce(output.0, 2 * h as usize * 2)?;
            } else {
                comm.all_reduce_async(output.0, 2 * h as usize * 2, stream)?;
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
                        2 * h,
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
                        2,
                        stream,
                    )?;
                }
            }
        }

        Ok(())
    }
}

mod forward_k2_helpers;
pub(crate) use forward_k2_helpers::{batch2_block_width, k2_e8m0_needs_per_token};

#[cfg(test)]
#[path = "forward_k2_dispatch_tests.rs"]
mod k2_dispatch_tests;
