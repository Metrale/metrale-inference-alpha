// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-02: `MoeLayer::forward_bf16_grouped_decode`: the MoE of `m` rows over BF16 experts
//! (`set_bf16_experts`) in one routed+shared dispatch, on the BF16 point of the tensor-core grouped
//! family (`moe_bf16_grouped_tc.cu`). The steps are those of the grouped NVFP4 decode
//! (`forward_nvfp4_grouped_decode.rs`): the per-row router, `moe_fp8_grouped_sort`, gate+up and
//! SiLU, down, the grouped blend. Its user is the MTP drafter of a checkpoint whose MTP head is
//! BF16 (nvidia/Qwen3.6-35B-A3B-NVFP4), so a batched propose runs the drafter's MoE once for all
//! its sequences instead of per-expert GEMVs per sequence.
//!
//! Owner: model-layers (MoE).
//! Invariants: `forward_bf16_grouped_decode` launches nothing unless
//! `bf16_grouped_decode_ok(m, ctx)` holds; a row's bits do not depend on `m`.

use super::forward_fp8_grouped_decode::grouped_decode_buffer_need;
use super::forward_nvfp4_grouped_decode::NVFP4_GROUPED_DECODE_TC_MAX_ROWS;
use super::*;

impl MoeLayer {
    /// 2026-10-02: The weight, kernel, shape and arena terms of
    /// [`Self::bf16_grouped_decode_ok`], decidable without a `ForwardContext` (the MTP batch cap
    /// sizes a batch with it).
    pub fn bf16_grouped_decode_arena_ok(
        &self,
        m: usize,
        cfg: &metrale_config::ModelConfig,
        b: &metrale_gpu_runtime::buffers::BufferArena,
    ) -> bool {
        let (h, inter) = (cfg.hidden_size, cfg.moe_intermediate_size);
        let need =
            grouped_decode_buffer_need(m, h, inter, cfg.num_experts, cfg.num_experts_per_tok);
        let k = &self.nvfp4_grouped;
        (1..=NVFP4_GROUPED_DECODE_TC_MAX_ROWS).contains(&m)
            && ops::bf16_grouped_tc_shape_ok(inter as u32, h as u32, ops::BF16_GROUPED_GATE_UP_TC)
            && ops::bf16_grouped_tc_shape_ok(h as u32, inter as u32, ops::BF16_GROUPED_DOWN_TC)
            && k.bf16_gate_up_tc.0 != 0
            && k.bf16_down_tc.0 != 0
            && self.bf16_gate_weight_ptrs.is_some()
            && self.bf16_up_weight_ptrs.is_some()
            && self.bf16_down_weight_ptrs.is_some()
            && self.bf16_shared_expert.is_some()
            && self.moe_weighted_sum_blend_fp8_grouped_k.0 != 0
            && self.moe_fp8_grouped_sort_k.0 != 0
            && self.moe_topk_softmax_rows_k.0 != 0
            && self.router_gemv_batchm_k.0 != 0
            && cfg.num_experts <= ops::FP8_GROUPED_SORT_MAX_EXPERTS as usize
            && self.router_logits_n as usize == cfg.num_experts
            && self.gate_nvfp4.is_none()
            && self.correction_bias_dev.is_none()
            && self.weights.router_pre_norm.is_none()
            && self.lora.is_none()
            && self.pre_expert_norm.is_none()
            && self.tid2eid_dev.is_none()
            && h.is_multiple_of(8)
            && arena_fits(&need, b)
    }

    /// 2026-10-02: Whether `forward_bf16_grouped_decode` serves `m` rows under `ctx`: the arena
    /// terms, the per-row router's context terms and no expert parallelism.
    pub fn bf16_grouped_decode_ok(&self, m: usize, ctx: &ForwardContext) -> bool {
        let cfg = ctx.config;
        let refused = ctx.levers.fp32_gate
            || self.fp32_routing_active(ctx.levers)
            || (self.is_dflash_capture_layer && ctx.levers.frankenstein_decode_via_prefill)
            || (ctx.comm.is_some() && cfg.ep_world_size > 1);
        self.bf16_grouped_decode_arena_ok(m, cfg, ctx.buffers) && !refused
    }

    /// 2026-10-02: The BF16 MoE of `m` rows: `input` is `[m, H]` BF16, the output lands in rows
    /// `0..m` of `moe_output()`. Returns an error, launching nothing, when
    /// `bf16_grouped_decode_ok(m, ctx)` is false.
    pub fn forward_bf16_grouped_decode(
        &self,
        input: DevicePtr,
        m: usize,
        ctx: &ForwardContext,
        stream: u64,
    ) -> Result<()> {
        anyhow::ensure!(
            self.bf16_grouped_decode_ok(m, ctx),
            "forward_bf16_grouped_decode: predicate false for m={m} (caller must gate on it)"
        );
        let (Some(gp), Some(up), Some(dp), Some(sh)) = (
            self.bf16_gate_weight_ptrs,
            self.bf16_up_weight_ptrs,
            self.bf16_down_weight_ptrs,
            self.bf16_shared_expert.as_ref(),
        ) else {
            anyhow::bail!("forward_bf16_grouped_decode: BF16 experts missing");
        };
        let h = ctx.config.hidden_size as u32;
        let inter = ctx.config.moe_intermediate_size as u32;
        let num_experts = ctx.config.num_experts as u32;
        let top_k = ctx.config.num_experts_per_tok as u32;
        let n = m as u32;
        let te = m * top_k as usize;
        if ctx.stats.once("log:moe_bf16_grouped_decode") {
            tracing::info!(
                "MoE BF16 grouped decode active (first use M={m}, top_k={top_k}, \
                 experts={num_experts}; one-time log)"
            );
        }
        let scratch = ctx.buffers.scratch();
        let indices_dev = scratch;
        let weights_dev = scratch.offset(te * 4);
        self.grouped_route(
            input,
            m,
            GroupedRouting::PerRow,
            indices_dev,
            weights_dev,
            ctx,
            stream,
        )?;
        // 2026-10-02: The sort scratch in `gate_logits`, as the grouped NVFP4 decode lays it out.
        let ne = num_experts as usize;
        let gate_logits = ctx.buffers.gate_logits();
        let sorted_token_ids = gate_logits;
        let sorted_expert_ids = gate_logits.offset(te * 4);
        let expert_offsets = gate_logits.offset(te * 4 * 2);
        let token_to_perm = gate_logits.offset(te * 4 * 2 + (ne + 1) * 4);
        let cap = ops::fp8_grouped_active_cap(n, top_k, num_experts);
        let active_experts = token_to_perm.offset(te * 4);
        let active_count = active_experts.offset(cap as usize * 4);
        ops::moe_fp8_grouped_sort(
            ctx.gpu,
            self.moe_fp8_grouped_sort_k,
            ops::Fp8GroupedSortOut {
                sorted_token_ids,
                sorted_expert_ids,
                expert_offsets,
                token_to_perm,
                active_experts,
                active_count,
            },
            indices_dev,
            te as u32,
            num_experts,
            top_k,
            stream,
        )?;
        let act = ctx.buffers.expert_gate_out();
        let expert_down_out = ctx.buffers.expert_down_out();
        let shared_act = ctx.buffers.logits();
        let shared_out = ctx.buffers.attn_output();
        let k = &self.nvfp4_grouped;
        ops::moe_expert_gate_up_act_bf16_grouped(
            ctx.gpu,
            k.bf16_gate_up_tc,
            input,
            gp,
            up,
            act,
            expert_offsets,
            sorted_token_ids,
            active_experts,
            active_count,
            sh.gate_proj.weight,
            sh.up_proj.weight,
            shared_act,
            inter,
            h,
            cap,
            n,
            stream,
        )?;
        ops::moe_expert_down_act_bf16_grouped(
            ctx.gpu,
            k.bf16_down_tc,
            act,
            dp,
            expert_down_out,
            expert_offsets,
            active_experts,
            active_count,
            shared_act,
            sh.down_proj.weight,
            shared_out,
            h,
            inter,
            cap,
            n,
            stream,
        )?;
        ops::moe_weighted_sum_blend_fp8_grouped(
            ctx.gpu,
            self.moe_weighted_sum_blend_fp8_grouped_k,
            ctx.buffers.moe_output(),
            expert_down_out,
            weights_dev,
            token_to_perm,
            shared_out,
            input,
            self.weights.shared_expert_gate.weight,
            h,
            top_k,
            h,
            n,
            stream,
        )
    }

    /// 2026-10-02: The grouped decode of `m` rows on whichever expert tables this layer holds for
    /// it: FP8 (`forward_fp8_grouped_decode`) or BF16 (`forward_bf16_grouped_decode`).
    pub fn any_grouped_decode_ok(&self, m: usize, ctx: &ForwardContext) -> bool {
        if self.nvfp4_grouped.declared_experts {
            self.nvfp4_grouped_decode_ok(m, ctx)
        } else if self.fp8_gate_weight_ptrs.is_some() {
            self.fp8_grouped_decode_ok(m, ctx)
        } else {
            self.bf16_grouped_decode_ok(m, ctx)
        }
    }

    /// 2026-10-02: [`Self::any_grouped_decode_ok`]'s arena terms.
    pub fn any_grouped_decode_arena_ok(
        &self,
        m: usize,
        cfg: &metrale_config::ModelConfig,
        b: &metrale_gpu_runtime::buffers::BufferArena,
    ) -> bool {
        if self.nvfp4_grouped.declared_experts {
            // 2026-10-02: The NVFP4 grouped decode's arena needs are the FP8 grouped decode's.
            (1..=NVFP4_GROUPED_DECODE_TC_MAX_ROWS).contains(&m)
                && arena_fits(
                    &grouped_decode_buffer_need(
                        m,
                        cfg.hidden_size,
                        cfg.moe_intermediate_size,
                        cfg.num_experts,
                        cfg.num_experts_per_tok,
                    ),
                    b,
                )
        } else if self.fp8_gate_weight_ptrs.is_some() {
            self.fp8_grouped_decode_arena_ok(m, cfg, b)
        } else {
            self.bf16_grouped_decode_arena_ok(m, cfg, b)
        }
    }

    /// 2026-10-02: Run [`Self::any_grouped_decode_ok`]'s path.
    pub fn forward_any_grouped_decode(
        &self,
        input: DevicePtr,
        m: usize,
        ctx: &ForwardContext,
        stream: u64,
    ) -> Result<()> {
        if self.nvfp4_grouped.declared_experts {
            self.forward_nvfp4_grouped_decode(input, m, ctx, stream)
        } else if self.fp8_gate_weight_ptrs.is_some() {
            self.forward_fp8_grouped_decode(input, m, ctx, stream)
        } else {
            self.forward_bf16_grouped_decode(input, m, ctx, stream)
        }
    }
}

/// 2026-10-02: Whether the arena holds a grouped decode's buffers (`grouped_decode_buffer_need`).
fn arena_fits(
    need: &super::forward_fp8_grouped_decode::GroupedDecodeBufferNeed,
    b: &metrale_gpu_runtime::buffers::BufferArena,
) -> bool {
    b.scratch_bytes() >= need.scratch
        && b.gate_logits_bytes() >= need.gate_logits
        && b.expert_gate_out_bytes() >= need.expert_gate_out
        && b.expert_down_out_bytes() >= need.expert_down_out
        && b.logits_bytes() >= need.shared_act
        && b.attn_output_bytes() >= need.row_hidden
        && b.moe_output_bytes() >= need.row_hidden
}
