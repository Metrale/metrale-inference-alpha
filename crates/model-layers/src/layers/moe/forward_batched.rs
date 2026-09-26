// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-25: `MoeLayer::forward_batched`: one gate GEMM for N tokens, then
//! per-token routing and expert dispatch.
//!
//! Owner: model-layers (MoE).
//! Invariants: none beyond the types.

use super::*;

impl MoeLayer {
    /// 2026-09-25: The gate projection is one GEMM with M = N; top-k, the
    /// expert kernels and the blend then run once per token.
    pub fn forward_batched(
        &self,
        input: DevicePtr,
        num_tokens: usize,
        ctx: &ForwardContext,
        stream: u64,
    ) -> Result<()> {
        // 2026-09-25: A zero-expert router is served only through the
        // softmax+bias arm (`router_bias_one`). `forward_prefill` sends BF16
        // and FP8 expert prefills here unless it takes their grouped GEMM
        // (more than 64 rows and the kernel resolved).
        anyhow::ensure!(
            self.router_logits_n as usize == ctx.config.num_experts
                || (self.correction_bias_dev.is_some() && ctx.config.scoring_func == "softmax"),
            "zero-expert MoE routing is not wired on this dispatch variant yet (forward_batched)"
        );

        // 2026-09-25: The router LoRA delta folds onto the whole batch's
        // `gate_logits` before top-k (`batched_gate_logits`), and the expert
        // gate/up and down deltas fold per token below. `row_adapter_base` is
        // the per-step `[padded_n]` i32 adapter map (`moe_row_adapter`), with
        // which base rows are skipped on the device. It is null when no LoRA
        // weights are loaded or `ctx.attn_metadata` carries none, and the
        // fold hooks then use `moe_route_gate`, which refuses a `Refuse` route.
        // In batched decode a batch routed to a non-active adapter (`Refuse`) is
        // refused host-side by `ensure_decode_route_servable` before this pass.
        let row_adapter_base = ctx
            .attn_metadata
            .as_ref()
            .map_or(DevicePtr::NULL, |m| m.moe_row_adapter);
        let h = ctx.config.hidden_size as u32;
        let inter = ctx.config.moe_intermediate_size as u32;
        let shared_inter = ctx.config.shared_expert_intermediate_size as u32;
        let num_experts = ctx.config.num_experts as u32;
        let top_k = ctx.config.num_experts_per_tok as u32;
        let n = num_tokens as u32;
        let bf16 = 2usize;

        // 2026-09-25: The router width (`num_experts + zero_expert_num`), not the
        // expert count. It is also the row stride of `gate_t` below; a wrong
        // stride would read each token's logits from the wrong offset without
        // failing.
        let router_n = self.router_logits_n;
        let (gate_logits, fp32_gate, gate_elem) =
            self.batched_gate_logits(input, n, h, router_n, row_adapter_base, ctx, stream)?;

        let h_usize = h as usize;
        let expert_gate_out = ctx.buffers.expert_gate_out();
        let expert_up_out = ctx.buffers.expert_up_out();
        let expert_down_out = ctx.buffers.expert_down_out();
        // 2026-09-25: The start of `logits()` is the shared-expert gate scratch,
        // as in `forward.rs`.
        let shared_gate_scratch = ctx.buffers.logits();
        let shared_up_scratch = ctx.buffers.ssm_qkvz();

        for t in 0..num_tokens {
            let input_t = input.offset(t * h_usize * bf16);
            let gate_t = gate_logits.offset(t * router_n as usize * gate_elem);
            let output_t = ctx.buffers.moe_output().offset(t * h_usize * bf16);

            let scratch = ctx.buffers.scratch();
            let indices_dev = scratch;
            let weights_dev = scratch.offset(top_k as usize * 4);

            // 2026-09-25: The fold sees `n_slots == top_k` rows, all of token `t`,
            // so its map is token `t`'s i32 entry at `row_adapter_base + 4t`, a
            // fixed address for a captured graph. A null map stays null.
            let ra_t = if row_adapter_base.0 != 0 {
                row_adapter_base.offset(t * 4)
            } else {
                DevicePtr::NULL
            };

            if let Some(tid2eid) = self.tid2eid_dev {
                // 2026-09-25: Hash routing (`moe_hash_route.cu`), reading token
                // `t`'s u32 id at `token_ids + 4t`.
                let token_ids = ctx.token_ids.ok_or_else(|| {
                    anyhow::anyhow!(
                        "DeepSeek-V4 hash-MoE layer requires ForwardContext.token_ids (prefill)"
                    )
                })?;
                ops::moe_hash_route(
                    ctx.gpu,
                    self.moe_hash_route_k,
                    gate_t,
                    tid2eid,
                    token_ids.offset(t * 4),
                    indices_dev,
                    weights_dev,
                    num_experts,
                    top_k,
                    ctx.config.norm_topk_prob,
                    ctx.config.routed_scaling_factor as f32,
                    stream,
                )?;
            } else if let Some(bias) = self.correction_bias_dev {
                self.router_bias_one(
                    gate_t,
                    bias,
                    indices_dev,
                    weights_dev,
                    num_experts,
                    top_k,
                    t,
                    ctx,
                    stream,
                )?;
            } else {
                ops::moe_topk_softmax(
                    ctx.gpu,
                    if fp32_gate {
                        self.moe_topk_f32
                    } else {
                        self.moe_topk
                    },
                    gate_t,
                    indices_dev,
                    weights_dev,
                    num_experts,
                    top_k,
                    ctx.config.norm_topk_prob,
                    stream,
                )?;
            }
            // 2026-09-25: Dump the last token's routing (no-op unless
            // `METRALE_DUMP_EXPERT_IDS=1`).
            if t == num_tokens - 1 {
                super::dump::dump_expert_ids(ctx.gpu, stream, indices_dev, weights_dev, 1, top_k)?;
            }

            let shared_out = ctx.buffers.attn_output();
            if let (Some(gp), Some(up), Some(dp), Some(shared)) = (
                self.bf16_gate_weight_ptrs,
                self.bf16_up_weight_ptrs,
                self.bf16_down_weight_ptrs,
                self.bf16_shared_expert,
            ) {
                // 2026-09-25: BF16 experts, with the same kernels as `forward`.
                ops::moe_expert_gate_up_shared_bf16(
                    ctx.gpu,
                    self.moe_expert_gate_up_shared_bf16_k,
                    input_t,
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
                self.apply_expert_lora_decode_gateup(
                    expert_gate_out,
                    expert_up_out,
                    input_t,
                    indices_dev,
                    top_k,
                    top_k,
                    ra_t,
                    ctx,
                    stream,
                )?;
                ops::moe_expert_silu_down_shared_bf16(
                    ctx.gpu,
                    self.moe_expert_silu_down_shared_bf16_k,
                    expert_gate_out,
                    expert_up_out,
                    dp,
                    expert_down_out,
                    indices_dev,
                    shared_gate_scratch,
                    shared_up_scratch,
                    shared.down_proj.weight,
                    shared_out,
                    h,
                    inter,
                    top_k,
                    stream,
                )?;
            } else if let (Some(gp), Some(up), Some(dp), Some(sh)) = (
                &self.fp8_gate_weight_ptrs,
                &self.fp8_up_weight_ptrs,
                &self.fp8_down_weight_ptrs,
                &self.fp8_shared_expert,
            ) {
                ops::moe_expert_gate_up_shared_fp8(
                    ctx.gpu,
                    self.moe_expert_gate_up_shared_fp8,
                    input_t,
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
                self.apply_expert_lora_decode_gateup(
                    expert_gate_out,
                    expert_up_out,
                    input_t,
                    indices_dev,
                    top_k,
                    top_k,
                    ra_t,
                    ctx,
                    stream,
                )?;
                ops::moe_expert_silu_down_shared_fp8(
                    ctx.gpu,
                    self.moe_expert_silu_down_shared_fp8,
                    expert_gate_out,
                    expert_up_out,
                    dp.weight_ptrs,
                    dp.scale_ptrs,
                    expert_down_out,
                    indices_dev,
                    shared_gate_scratch,
                    shared_up_scratch,
                    &sh.down_proj,
                    shared_out,
                    h,
                    inter,
                    top_k,
                    stream,
                )?;
            } else if self.use_t_layout_for_prefill() {
                // 2026-09-25: Transposed-layout kernels; `use_t_layout_for_prefill`
                // holds in both the unified and the hybrid layout.
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
                let sh_gate_t = self.shared_gate_t.as_ref().unwrap_or(&null_qw);
                let sh_up_t = self.shared_up_t.as_ref().unwrap_or(&null_qw);
                let sh_down_t = self.shared_down_t.as_ref().unwrap_or(&null_qw);
                // 2026-09-25: The `_e8m0` kernels are `<32, true, GROUP_SIZE, false>`
                // (`moe_shared_expert_fused_t.cu`): routed experts E8M0, shared
                // expert NVFP4. Check the shared expert's format before using them.
                if self.experts_scale_kind == crate::weight_map::WeightQuantFormat::Mxfp4E8m0 {
                    self.shared_experts_scale_kind.expect(
                        crate::weight_map::WeightQuantFormat::Nvfp4,
                        "decode fused _e8m0 kernel assumes an NVFP4 shared expert",
                    );
                }
                ops::moe_expert_gate_up_shared_t(
                    ctx.gpu,
                    self.e8m0_or(
                        self.moe_expert_gate_up_shared_t_k,
                        self.moe_expert_gate_up_shared_t_e8m0_k,
                        "decode gate_up_shared_t",
                    ),
                    input_t,
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
                self.apply_expert_lora_decode_gateup(
                    expert_gate_out,
                    expert_up_out,
                    input_t,
                    indices_dev,
                    top_k,
                    top_k,
                    ra_t,
                    ctx,
                    stream,
                )?;
                ops::moe_expert_silu_down_shared_t(
                    ctx.gpu,
                    self.e8m0_or(
                        self.moe_expert_silu_down_shared_t_k,
                        self.moe_expert_silu_down_shared_t_e8m0_k,
                        "decode silu_down_shared_t",
                    ),
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
                    shared_out,
                    h,
                    inter,
                    top_k,
                    stream,
                )?;
            } else {
                ops::moe_expert_gate_up_shared(
                    ctx.gpu,
                    self.moe_expert_gate_up_shared,
                    input_t,
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
                self.apply_expert_lora_decode_gateup(
                    expert_gate_out,
                    expert_up_out,
                    input_t,
                    indices_dev,
                    top_k,
                    top_k,
                    ra_t,
                    ctx,
                    stream,
                )?;
                ops::moe_expert_silu_down_shared(
                    ctx.gpu,
                    self.moe_expert_silu_down_shared,
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
                    shared_out,
                    h,
                    inter,
                    top_k,
                    stream,
                )?;
            }

            // 2026-09-25: Fold the expert down_proj delta into this token's
            // `expert_down_out` (`[top_k, hidden]`) in place, before the blend, so
            // the routing weight scales base plus delta.
            self.apply_expert_lora_decode_down(
                expert_gate_out,
                expert_up_out,
                expert_down_out,
                indices_dev,
                top_k,
                top_k,
                ra_t,
                ctx,
                stream,
            )?;

            if self.has_mixed_bf16_shared_expert() {
                self.run_bf16_shared_expert(
                    input_t,
                    1,
                    h,
                    shared_inter,
                    shared_gate_scratch,
                    shared_up_scratch,
                    shared_out,
                    ctx,
                    stream,
                )?;
            }

            ops::moe_weighted_sum_blend(
                ctx.gpu,
                self.moe_weighted_sum_blend,
                output_t,
                expert_down_out,
                weights_dev,
                shared_out,
                input_t,
                self.weights.shared_expert_gate.weight,
                h,
                top_k,
                h,
                stream,
            )?;

            if let Some(comm) = ctx.comm
                && ctx.config.ep_world_size > 1
            {
                if ctx.graph_capture {
                    comm.all_reduce(output_t.0, h as usize * 2)?;
                } else {
                    comm.all_reduce_async(output_t.0, h as usize * 2, stream)?;
                }
            }
        }

        Ok(())
    }
}
