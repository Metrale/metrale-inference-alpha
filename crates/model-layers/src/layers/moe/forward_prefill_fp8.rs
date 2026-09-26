// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-25: `MoeLayer::forward_prefill_fp8`, the grouped prefill for FP8 routed experts.
//!
//! Owner: model-layers (MoE).
//! Invariants: none beyond the types.

use super::*;
use metrale_gpu_runtime::buffers::MoeFp8Scratch;

// 2026-09-26: One `ctx.profile` timing step of the FP8 prefill: when the timer `$mt` holds a
// start time, synchronise `$stream`, log the step's elapsed time under `$label` and restart the
// timer. Each function in this module wraps it in a local `mprof!` over its own timer.
macro_rules! mprof_step {
    ($mt:expr, $ctx:expr, $stream:expr, $n:expr, $label:expr) => {
        if let Some(t) = $mt.take() {
            $ctx.gpu.synchronize($stream)?;
            tracing::info!(
                target: "metrale_model_layers::layers::moe::forward_prefill_fp8",
                "  MOE prefill [{}] N={}: {}\u{b5}s",
                $label,
                $n,
                t.elapsed().as_micros()
            );
            $mt = Some(std::time::Instant::now());
        }
    };
}

// 2026-09-25: Copies of `PM4_M_TILE` / `PM4_N_TILE` in moe_fp8_grouped_gemm.cu;
// `moe_build_tile_worklist` packs `(m_tile << 6) | n_tile`.
const PM4_N_TILE: u32 = 64;
const PM4_M_TILE: u32 = 128;

mod down;
mod gate_up;
mod shared;

impl MoeLayer {
    /// 2026-09-25: Whether `silu_mul_quant_fp8` replaces the `silu_mul` then
    /// `per_token_group_quant_fp8` pair for K = `k`: the handle resolved, the
    /// activation is SiLU (the fused kernel computes SiLU only), `k % 128 == 0`
    /// and `k / 128 <= 16` (`SILU_QUANT_MAX_GROUPS` in moe_silu_mul.cu).
    fn fused_silu_quant_ok(&self, k: u32) -> bool {
        self.silu_mul_quant_fp8_k.0 != 0
            && !self.gelu_activation
            && k.is_multiple_of(128)
            && k / 128 <= 16
    }

    /// 2026-09-25: Grouped prefill over the FP8 pointer tables: shared expert,
    /// gate GEMM, top-K, sort by expert, grouped gate/up GEMM, activation,
    /// grouped down GEMM, unpermute + weighted reduce, shared blend. Errors if
    /// the FP8 tables are not set.
    // 2026-09-25: The last `mprof!` assigns `mt`, which is not read again.
    #[allow(unused_assignments)]
    pub(super) fn forward_prefill_fp8(
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
        // 2026-09-25: Every FP8 activation, scale and work-list temporary below is a
        // region of the arena's grouped-MoE slab, so nothing here allocates, frees or
        // synchronizes, and the addresses stay valid across CUDA graph replay.
        let fp8_scratch = ctx.buffers.moe_fp8_scratch(ctx.config, num_tokens)?;

        // 2026-09-25: Per-step timing only with `ctx.profile` outside graph capture;
        // each step then synchronises the stream.
        let profile = ctx.profile && !ctx.graph_capture;
        let mut mt = if profile {
            ctx.gpu.synchronize(stream)?;
            Some(std::time::Instant::now())
        } else {
            None
        };
        macro_rules! mprof {
            ($label:expr) => {
                mprof_step!(mt, ctx, stream, n, $label)
            };
        }

        let (gp, up, dp, sh) = match (
            &self.fp8_gate_weight_ptrs,
            &self.fp8_up_weight_ptrs,
            &self.fp8_down_weight_ptrs,
            &self.fp8_shared_expert,
        ) {
            (Some(g), Some(u), Some(d), Some(s)) => (g, u, d, s),
            _ => anyhow::bail!("FP8 expert pointer tables not set"),
        };

        // 2026-09-25: Shared expert: the installed BF16 copy first; else W8A8 when
        // `fp8_blockscaled_prefill` (on unless METRALE_FP8_SINGLE_SCALE) and its
        // kernels resolved (activations quantised to FP8 per row and 128-column
        // group, then `fp8_gemm_t_blockscaled`); else W8A16.
        let force_w8a8_sh = ctx.dispatch.fp8_blockscaled_prefill
            && self.fp8_gemm_t_blockscaled_k.0 != 0
            && self.per_token_group_quant_fp8_k.available();
        let has_shared = shared_inter > 0;
        let bf16_shared = has_shared
            && self.run_bf16_shared_expert(
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
        // 2026-09-25: Held across the router so shared W8A8 can leave the main stream.
        // Dropped after sort, before routed quant reuses `fp8_scratch`.
        let mut shared_join: Option<super::adaptive_fp8::SideJoin<'_>> = None;
        if !bf16_shared && has_shared && force_w8a8_sh {
            let shared_stream =
                if super::adaptive_fp8::overlap_shared_router(num_tokens, num_experts, top_k) {
                    let join = super::adaptive_fp8::begin_shared_side(ctx.gpu, stream)?;
                    let side = join.side();
                    shared_join = Some(join);
                    side
                } else {
                    stream
                };
            self.fp8_prefill_shared_w8a8(
                input,
                sh,
                n,
                h,
                shared_inter,
                &fp8_scratch,
                ctx,
                stream,
                shared_stream,
                &mut mt,
            )?;
        } else if !bf16_shared && has_shared {
            self.fp8_prefill_shared_w8a16(input, sh, n, h, shared_inter, ctx, stream, &mut mt)?;
        }

        let router_in = self.router_input(input, n, h, ctx, stream)?;
        super::dump::dump_gate_input(ctx.gpu, stream, router_in, n, h)?;
        let gate_logits = ctx.buffers.gate_logits();
        // 2026-09-25: The gate width is `router_logits_n`, which exceeds
        // num_experts when the router also scores zero-computation experts.
        if let Some(ref nvfp4) = self.gate_nvfp4 {
            ops::w4a16_gemm(
                ctx.gpu,
                self.w4a16_gemm,
                router_in,
                nvfp4,
                gate_logits,
                n,
                self.router_logits_n,
                h,
                stream,
            )?;
        } else {
            // 2026-09-25: The router GEMM keeps the scalar kernel's accumulation
            // order (`router_gate_gemm_dense`).
            self.router_gate_gemm_dense(
                router_in,
                gate_logits,
                n,
                self.router_logits_n,
                h,
                ctx,
                stream,
            )?;
        }

        super::dump::dump_gate_logits(ctx.gpu, stream, gate_logits, n, num_experts)?;

        // 2026-09-25: Fold the router LoRA delta onto the logits before top-k; a
        // no-op without an installed router delta.
        self.apply_router_lora_prefill(router_in, gate_logits, n, ctx, stream)?;

        // 2026-09-25: Routing: with a correction bias, softmax + bias for
        // `scoring_func == "softmax"` and sigmoid otherwise (this path has no
        // sqrtsoftplus or hash arm); without one, softmax top-K.
        let scratch = ctx.buffers.scratch();
        let indices_dev = scratch;
        let weights_dev = scratch.offset(total_expanded as usize * 4);
        if let Some(bias) = self.correction_bias_dev {
            if ctx.config.scoring_func == "softmax" {
                self.router_softmax_bias_batched(
                    gate_logits,
                    bias,
                    indices_dev,
                    weights_dev,
                    num_experts,
                    top_k,
                    n,
                    ctx,
                    stream,
                )?;
                mprof!("routing_topk");
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
                    n,
                    stream,
                )?;
                mprof!("routing_topk");
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
        mprof!("sort_by_expert");
        // 2026-09-25: Router, top-k, and sort do not read shared outputs or fp8 scratch.
        // Routed quant below reuses that scratch, so the side stream joins here.
        drop(shared_join);

        // 2026-09-25: `max_m_tiles` covers all te rows in one expert, so no
        // expert's rows are cut off; the dense W8A8 kernel takes it as its M-tile
        // count, and it is at least 1.
        let avg_per_expert = (num_tokens * top_k as usize).div_ceil(ne);
        let max_m_tiles = (num_tokens * top_k as usize).div_ceil(64).max(1) as u32;
        super::dump::dump_expert_load(
            ctx.gpu,
            stream,
            expert_offsets,
            ne,
            num_tokens,
            avg_per_expert,
            max_m_tiles,
        );

        let expert_gate_out = ctx.buffers.expert_gate_out();
        let expert_up_out = ctx.buffers.expert_up_out();
        // 2026-09-25: Zero the expert buffers before the grouped GEMMs, on every
        // path. The work-list builder skips an expert whose weight pointer is
        // NULL, so its sorted rows are never written, and the unpermute still
        // reads every sorted row.
        {
            let gu_bytes = te * inter as usize * 2;
            ctx.gpu.memset_async(expert_gate_out, 0, gu_bytes, stream)?;
            ctx.gpu.memset_async(expert_up_out, 0, gu_bytes, stream)?;
            ctx.gpu.memset_async(
                ctx.buffers.expert_down_out(),
                0,
                te * h as usize * 2,
                stream,
            )?;
        }
        // 2026-09-25: W8A8 when `fp8_blockscaled_prefill` and its kernels
        // resolved: activations quantised to FP8 per row and 128-column group
        // with FP32 scales.
        let force_w8a8 = ctx.dispatch.fp8_blockscaled_prefill
            && self.moe_w8a8_grouped_gemm_k.0 != 0
            && self.per_token_group_quant_fp8_k.available();

        if force_w8a8 && max_m_tiles > 0 {
            self.fp8_prefill_gate_up_w8a8(
                input,
                gp,
                up,
                expert_gate_out,
                expert_up_out,
                expert_offsets,
                sorted_token_ids,
                num_tokens,
                num_experts,
                h,
                inter,
                n,
                te,
                ne,
                max_m_tiles,
                &fp8_scratch,
                ctx,
                stream,
                &mut mt,
            )?;
        } else if max_m_tiles > 0 {
            self.fp8_prefill_gate_up_fp8(
                input,
                gp,
                up,
                expert_gate_out,
                expert_up_out,
                expert_offsets,
                sorted_token_ids,
                num_experts,
                h,
                inter,
                n,
                te,
                ne,
                &fp8_scratch,
                ctx,
                stream,
                &mut mt,
            )?;
        }

        // 2026-09-25: Fold gate/up LoRA deltas onto the sorted BF16
        // `expert_gate_out`/`expert_up_out` after either gate/up branch and
        // before the activation; the LoRA input is the BF16 `input`.
        if max_m_tiles > 0 {
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
        }

        let expert_down_out = ctx.buffers.expert_down_out();
        if force_w8a8 && max_m_tiles > 0 {
            self.fp8_prefill_down_w8a8(
                dp,
                expert_gate_out,
                expert_up_out,
                expert_down_out,
                expert_offsets,
                total_expanded,
                num_experts,
                h,
                inter,
                n,
                te,
                ne,
                max_m_tiles,
                &fp8_scratch,
                ctx,
                stream,
                &mut mt,
            )?;
        } else if max_m_tiles > 0 {
            self.fp8_prefill_down_fp8(
                dp,
                expert_gate_out,
                expert_up_out,
                expert_down_out,
                expert_offsets,
                total_expanded,
                num_experts,
                h,
                inter,
                n,
                te,
                ne,
                &fp8_scratch,
                ctx,
                stream,
                &mut mt,
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
        mprof!("unpermute_reduce");

        // 2026-09-25: With EP, the all-reduce covers only the routed output; the
        // shared blend below runs after it.
        if let Some(comm) = ctx.comm
            && ctx.config.ep_world_size > 1
        {
            comm.all_reduce_async(output.0, num_tokens * h as usize * 2, stream)?;
        }

        if has_shared {
            let shared_down_out = ctx.buffers.attn_output();
            super::dump::dump_routed_only(ctx.gpu, stream, output, n, h)?;
            super::dump::dump_shared_out(ctx.gpu, stream, shared_down_out, n, h)?;
            super::dump::dump_shared_gate(
                ctx.gpu,
                stream,
                input,
                self.weights.shared_expert_gate.weight,
                n,
                h,
            )?;
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
            mprof!("blend");
        }

        super::dump::dump_moe_out(ctx.gpu, stream, output, n, h)?;

        Ok(())
    }
}
