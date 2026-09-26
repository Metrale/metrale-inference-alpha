// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-25: `MoeLayer::forward_prefill_bf16`, the grouped prefill for BF16 routed
//! experts (FP8 checkpoints dequantized to BF16 at load): `moe_bf16_grouped_gemm`
//! for gate, up and down, and the BF16 shared expert via `run_bf16_shared_expert`.
//!
//! Owner: model-layers (MoE).
//! Invariants: none beyond the types.

use super::*;

impl MoeLayer {
    pub(super) fn forward_prefill_bf16(
        &self,
        input: DevicePtr,
        num_tokens: usize,
        ctx: &ForwardContext,
        stream: u64,
    ) -> Result<()> {
        let h = ctx.config.hidden_size as u32;
        let inter = ctx.config.moe_intermediate_size as u32;
        let shared_inter = ctx.config.shared_expert_intermediate_size as u32;
        let num_experts = ctx.config.num_experts as u32;
        let top_k = ctx.config.num_experts_per_tok as u32;
        let n = num_tokens as u32;
        let total_expanded = n * top_k;
        let ne = num_experts as usize;

        let (gp, up, dp) = match (
            self.bf16_gate_weight_ptrs,
            self.bf16_up_weight_ptrs,
            self.bf16_down_weight_ptrs,
        ) {
            (Some(g), Some(u), Some(d)) => (g, u, d),
            _ => anyhow::bail!("BF16 expert pointer tables not set"),
        };

        let has_shared = shared_inter > 0 && self.bf16_shared_expert.is_some();
        if has_shared {
            self.run_bf16_shared_expert(
                input,
                n,
                h,
                shared_inter,
                ctx.buffers.ssm_deinterleaved(),
                ctx.buffers.ssm_qkvz(),
                ctx.buffers.attn_output(),
                ctx,
                stream,
            )?;
        }

        let router_in = self.router_input(input, n, h, ctx, stream)?;
        super::dump::dump_gate_input(ctx.gpu, stream, router_in, n, h)?;

        let gate_logits = ctx.buffers.gate_logits();
        if let Some(ref nvfp4) = self.gate_nvfp4 {
            ops::w4a16_gemm(
                ctx.gpu,
                self.w4a16_gemm,
                router_in,
                nvfp4,
                gate_logits,
                n,
                num_experts,
                h,
                stream,
            )?;
        } else {
            // 2026-09-25: The router GEMM keeps the scalar kernel's accumulation
            // order (`router_gate_gemm_dense`).
            self.router_gate_gemm_dense(router_in, gate_logits, n, num_experts, h, ctx, stream)?;
        }

        super::dump::dump_gate_logits(ctx.gpu, stream, gate_logits, n, num_experts)?;

        // 2026-09-25: Fold the router LoRA delta onto the logits before top-k; a
        // no-op without an installed router delta.
        self.apply_router_lora_prefill(router_in, gate_logits, n, ctx, stream)?;

        let scratch = ctx.buffers.scratch();
        let indices_dev = scratch;
        let weights_dev = scratch.offset(total_expanded as usize * 4);
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
                n,
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
                n,
                stream,
            )?;
        }

        super::dump::dump_expert_ids(ctx.gpu, stream, indices_dev, weights_dev, n, top_k)?;

        let te = total_expanded as usize;
        let sorted_token_ids = gate_logits;
        let sorted_expert_ids = gate_logits.offset(te * 4);
        let expert_offsets = gate_logits.offset(te * 4 * 2);
        let token_to_perm = gate_logits.offset(te * 4 * 2 + (ne + 1) * 4);
        ops::moe_sort_by_expert(
            ctx.gpu,
            self.moe_sort_by_expert,
            indices_dev,
            sorted_token_ids,
            sorted_expert_ids,
            expert_offsets,
            token_to_perm,
            total_expanded,
            num_experts,
            top_k,
            stream,
        )?;

        let max_m_tiles = (num_tokens * top_k as usize).div_ceil(64).max(1) as u32;

        let expert_gate_out = ctx.buffers.expert_gate_out();
        let expert_up_out = ctx.buffers.expert_up_out();
        let expert_down_out = ctx.buffers.expert_down_out();
        {
            let gu_bytes = te * inter as usize * 2;
            ctx.gpu.memset_async(expert_gate_out, 0, gu_bytes, stream)?;
            ctx.gpu.memset_async(expert_up_out, 0, gu_bytes, stream)?;
            ctx.gpu
                .memset_async(expert_down_out, 0, te * h as usize * 2, stream)?;
        }
        if max_m_tiles > 0 {
            ops::moe_bf16_grouped_gemm(
                ctx.gpu,
                self.moe_bf16_grouped_gemm_k,
                input,
                gp,
                expert_gate_out,
                expert_offsets,
                sorted_token_ids,
                num_experts,
                inter,
                h,
                max_m_tiles,
                stream,
            )?;
            ops::moe_bf16_grouped_gemm(
                ctx.gpu,
                self.moe_bf16_grouped_gemm_k,
                input,
                up,
                expert_up_out,
                expert_offsets,
                sorted_token_ids,
                num_experts,
                inter,
                h,
                max_m_tiles,
                stream,
            )?;
            // 2026-09-25: Fold gate/up LoRA deltas onto the sorted
            // `expert_gate_out`/`expert_up_out` before `silu_mul` overwrites
            // `expert_gate_out`.
            self.apply_expert_lora_prefill_gateup(
                expert_gate_out,
                expert_up_out,
                input,
                expert_offsets,
                sorted_token_ids,
                total_expanded,
                ctx,
                stream,
            )?;
            ops::silu_mul(
                ctx.gpu,
                self.moe_act_mul,
                expert_gate_out,
                expert_up_out,
                expert_gate_out,
                total_expanded * inter,
                stream,
            )?;
            ops::moe_bf16_grouped_gemm(
                ctx.gpu,
                self.moe_bf16_grouped_gemm_k,
                expert_gate_out,
                dp,
                expert_down_out,
                expert_offsets,
                metrale_gpu_runtime::gpu::DevicePtr(0),
                num_experts,
                h,
                inter,
                max_m_tiles,
                stream,
            )?;
        }

        // 2026-09-25: Fold the routed-expert down LoRA delta onto the sorted
        // `expert_down_out` before the weighted reduce; a no-op without deltas.
        self.apply_expert_lora_prefill_down(
            expert_gate_out,
            expert_down_out,
            expert_offsets,
            sorted_token_ids,
            total_expanded,
            ctx,
            stream,
        )?;

        let output = ctx.buffers.moe_output();
        ops::moe_unpermute_reduce_indexed(
            ctx.gpu,
            self.moe_unpermute_reduce,
            expert_down_out,
            output,
            token_to_perm,
            weights_dev,
            h,
            n,
            top_k,
            stream,
        )?;

        if let Some(comm) = ctx.comm
            && ctx.config.ep_world_size > 1
        {
            comm.all_reduce_async(output.0, num_tokens * h as usize * 2, stream)?;
        }

        if has_shared {
            let shared_down_out = ctx.buffers.attn_output();
            ops::moe_batched_blend(
                ctx.gpu,
                self.moe_batched_blend,
                output,
                shared_down_out,
                input,
                self.weights.shared_expert_gate.weight,
                h,
                n,
                stream,
            )?;
        }

        super::dump::dump_moe_out(ctx.gpu, stream, output, n, h)?;

        Ok(())
    }
}
