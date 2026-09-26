// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-25: `MoeLayer::forward_prefill`, the MoE block for N prefill rows.
//!
//! Owner: model-layers (MoE).
//! Invariants: none beyond the types.

use super::*;

impl MoeLayer {
    /// 2026-09-25: MoE for `num_tokens` rows of the normed [num_tokens, H] BF16
    /// `input`, with grouped GEMMs over the rows sorted by expert.
    ///
    /// BF16 and FP8 experts go to `forward_prefill_bf16` / `forward_prefill_fp8`
    /// above 64 rows when that grouped kernel resolved, and to `forward_batched`
    /// otherwise. The NVFP4 body here:
    /// shared expert, gate GEMM, top-K, sort by expert, grouped gate/up GEMM,
    /// activation, grouped down GEMM, unpermute + weighted reduce, shared blend.
    #[allow(unused_assignments)]
    pub fn forward_prefill(
        &self,
        input: DevicePtr,
        num_tokens: usize,
        ctx: &ForwardContext,
        stream: u64,
    ) -> Result<()> {
        // 2026-09-25: Native HIP (`metrale_hip`, the strix-hip target) has no BF16
        // grouped MoE GEMM (kernels/strix-hip/common ships the FP8 and W4A16 ones
        // only), so BF16 experts always take `forward_batched` there. FP8 experts
        // keep the grouped path.
        let hip_force_batched = cfg!(metrale_hip);
        let hip_force_batched_fp8 = false;

        // 2026-09-25: MoE LoRA needs no refusal here. Each grouped body (this one,
        // `forward_prefill_bf16`, `forward_prefill_fp8`) folds the router delta
        // before top-k and the routed-expert gate/up and down deltas on the sorted
        // buffers; `forward_batched` folds them per row.

        // 2026-09-25: BF16 and FP8 experts: the grouped GEMM above 64 rows when its
        // kernel resolved, else `forward_batched`.
        if self.bf16_gate_weight_ptrs.is_some() {
            if self.moe_bf16_grouped_gemm_k.0 != 0 && num_tokens > 64 && !hip_force_batched {
                return self.forward_prefill_bf16(input, num_tokens, ctx, stream);
            }
            return self.forward_batched(input, num_tokens, ctx, stream);
        }

        if self.fp8_gate_weight_ptrs.is_some() {
            if self.moe_fp8_grouped_gemm_k.0 != 0 && num_tokens > 64 && !hip_force_batched_fp8 {
                return self.forward_prefill_fp8(input, num_tokens, ctx, stream);
            }
            return self.forward_batched(input, num_tokens, ctx, stream);
        }

        // 2026-09-25: Transpose this layer's down_proj into the shared scratch on
        // the compute stream; a no-op unless the lazy scratch is wired.
        let _t_xpose = if ctx.profile && self.down_t_scratch_packed.is_some() {
            ctx.gpu.synchronize(stream)?;
            Some(std::time::Instant::now())
        } else {
            None
        };
        self.transpose_down_into_scratch(ctx, stream)?;
        if let Some(t0) = _t_xpose {
            ctx.gpu.synchronize(stream)?;
            tracing::info!(
                "  MoE prefill [lazy_transpose_down] N={}: {}µs",
                num_tokens,
                t0.elapsed().as_micros(),
            );
        }

        let h = ctx.config.hidden_size as u32;
        let inter = ctx.config.moe_intermediate_size as u32;
        let shared_inter = ctx.config.shared_expert_intermediate_size as u32;
        let num_experts = ctx.config.num_experts as u32;
        let top_k = ctx.config.num_experts_per_tok as u32;
        let n = num_tokens as u32;
        let total_expanded = n * top_k;

        #[allow(unused_macros)]
        macro_rules! prof {
            ($label:expr) => {
                if ctx.profile {
                    ctx.gpu.synchronize(stream)?;
                    let _t = std::time::Instant::now();
                    tracing::info!("  MoE prefill [{}] N={}", $label, num_tokens);
                }
            };
        }
        #[allow(unused_assignments)]
        let mut t0 = if ctx.profile {
            ctx.gpu.synchronize(stream)?;
            Some(std::time::Instant::now())
        } else {
            None
        };
        macro_rules! prof_step {
            ($label:expr) => {
                if let Some(t) = t0.take() {
                    ctx.gpu.synchronize(stream)?;
                    let elapsed = t.elapsed().as_micros();
                    tracing::info!("  MoE prefill [{}] N={}: {}µs", $label, num_tokens, elapsed);
                    t0 = Some(std::time::Instant::now());
                }
            };
        }

        // 2026-09-25: The shared expert runs first. It reads `input` and writes
        // only ssm_deinterleaved, ssm_qkvz and attn_output. `use_overlap` is
        // false, so it runs on `stream`. Skipped when shared_inter == 0.
        let has_shared = shared_inter > 0;
        let use_overlap = false;
        let aux = if use_overlap {
            self.prefill_stream
        } else {
            stream
        };

        if has_shared {
            self.run_shared_expert_prefill(
                input,
                n,
                h,
                shared_inter,
                aux,
                stream,
                use_overlap,
                ctx,
            )?;
        }
        prof_step!("shared_expert");

        let router_in = self.router_input(input, n, h, ctx, stream)?;
        super::dump::dump_gate_input(ctx.gpu, stream, router_in, n, h)?;
        let gate_logits = ctx.buffers.gate_logits();
        if let Some(fp8) = self.gate_fp8 {
            ops::fp8_gemm_n128(
                ctx.gpu,
                self.fp8_gemm_k,
                router_in,
                fp8,
                gate_logits,
                n,
                self.router_logits_n,
                h,
                stream,
            )?;
        } else if let Some(ref nvfp4) = self.gate_nvfp4 {
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
            // order (`router_gate_gemm_dense`); it has no cuBLAS arm.
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
        prof_step!("gate_gemm");

        // 2026-09-25: Fold the router LoRA delta onto the logits before top-k. A
        // no-op without an installed router delta (loaded only with
        // METRALE_LORA_EXPERTS=1) or when the request is not routed to it.
        self.apply_router_lora_prefill(router_in, gate_logits, n, ctx, stream)?;

        // 2026-09-25: Routing: the static hash table when `tid2eid_dev` is set,
        // else a correction bias with sqrtsoftplus, softmax or sigmoid scoring by
        // `scoring_func`, else plain softmax top-K.
        let scratch = ctx.buffers.scratch();
        let indices_dev = scratch;
        let weights_dev = scratch.offset(total_expanded as usize * 4);
        if let Some(tid2eid) = self.tid2eid_dev {
            let token_ids = ctx.token_ids.ok_or_else(|| {
                anyhow::anyhow!(
                    "DeepSeek-V4 hash-MoE layer requires ForwardContext.token_ids (prefill grouped)"
                )
            })?;
            ops::moe_hash_route_batched(
                ctx.gpu,
                self.moe_hash_route_batched_k,
                gate_logits,
                tid2eid,
                token_ids,
                indices_dev,
                weights_dev,
                num_experts,
                top_k,
                ctx.config.norm_topk_prob,
                ctx.config.routed_scaling_factor as f32,
                n,
                stream,
            )?;
        } else if let Some(bias) = self.correction_bias_dev {
            if ctx.config.scoring_func == "sqrtsoftplus" {
                ops::moe_topk_sqrtsoftplus_batched(
                    ctx.gpu,
                    self.moe_topk_sqrtsoftplus_batched_k,
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
            } else if ctx.config.scoring_func == "softmax" {
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
        prof_step!("topk");

        // 2026-09-25: Sort the (row, slot) pairs by expert. The sort outputs reuse
        // `gate_logits`, which top-K has consumed: sorted_token_ids [te],
        // sorted_expert_ids [te], expert_offsets [E + 1], token_to_perm [te], u32.
        let te = total_expanded as usize;
        let ne = num_experts as usize;
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
        prof_step!("sort");

        // 2026-09-25: With `pre_expert_norm`, the router has read the un-normed
        // `input` and the experts read a normed copy in ssm_deinterleaved. `input`
        // itself stays intact: the shared-expert blend reads it.
        let expert_input = if let Some(ref norm_w) = self.pre_expert_norm {
            let normed_buf = ctx.buffers.ssm_deinterleaved();
            let n_tokens = num_tokens as u32;
            let eps = ctx.config.rms_norm_eps as f32;
            ops::rms_norm(
                ctx.gpu,
                self.pre_expert_norm_k,
                input,
                norm_w,
                normed_buf,
                n_tokens,
                h,
                eps,
                stream,
            )?;
            normed_buf
        } else {
            input
        };
        prof_step!("pre_expert_norm");

        // 2026-09-25: Grouped gate+up GEMM, activation and grouped down GEMM; the
        // sorted result lands in `expert_down_out`.
        self.run_routed_grouped_gemm(
            expert_input,
            expert_offsets,
            sorted_token_ids,
            n,
            h,
            inter,
            num_experts,
            top_k,
            num_tokens,
            ne,
            &mut t0,
            ctx,
            stream,
        )?;
        let expert_down_out = ctx.buffers.expert_down_out();

        // 2026-09-25: Fold the routed-expert down LoRA delta onto the sorted
        // `expert_down_out` before the weighted reduce, so the routing weight
        // scales base + delta. A no-op without installed down deltas.
        self.apply_expert_lora_prefill_down(
            ctx.buffers.expert_gate_out(),
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

        // 2026-09-25: Shared-expert blend: output += sigmoid(input . gate) * shared,
        // weight 1 when the layer has no shared_expert_gate. With EP it runs
        // once after the all-reduce instead, so the reduce does not sum it per rank.
        let is_ep_prefill = ctx.comm.is_some() && ctx.config.ep_world_size > 1;
        if has_shared && !is_ep_prefill {
            let shared_down_out = ctx.buffers.attn_output();
            if use_overlap {
                ctx.gpu.stream_wait_event(stream, self.event_b)?;
            }
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
        }
        super::dump::dump_moe_out(ctx.gpu, stream, output, n, h)?;
        prof_step!("unpermute_blend");

        if let Some(comm) = ctx.comm
            && ctx.config.ep_world_size > 1
        {
            let _t0 = if ctx.profile {
                ctx.gpu.synchronize(stream)?;
                Some(std::time::Instant::now())
            } else {
                None
            };
            if ctx.graph_capture {
                comm.all_reduce(output.0, num_tokens * h as usize * 2)?;
            } else {
                comm.all_reduce_async(output.0, num_tokens * h as usize * 2, stream)?;
            }
            if let Some(t0) = _t0 {
                ctx.gpu.synchronize(stream)?;
                tracing::info!(
                    "  EP allreduce (moe out) N={}: {}µs",
                    num_tokens,
                    t0.elapsed().as_micros(),
                );
            }
            if has_shared {
                let shared_down_out = ctx.buffers.attn_output();
                if use_overlap {
                    ctx.gpu.stream_wait_event(stream, self.event_b)?;
                }
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
        }

        Ok(())
    }
}
