// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-26: `MoeLayer::forward`, the MoE of one decode token.
//! `fp32_routing_active` and `apply_zero_expert` are in `forward/zero_expert.rs`.
//!
//! Owner: model-layers (MoE).
//! Invariants: none beyond the types.

use super::*;

mod debug_dumps;
mod ep_reduce;
mod route;
mod zero_expert;

impl MoeLayer {
    pub fn forward(
        &self,
        input: DevicePtr,
        ctx: &ForwardContext,
        stream: u64,
    ) -> Result<DevicePtr> {
        // 2026-09-25: With one sequence this pass folds the MoE LoRA deltas
        // below, with no per-row adapter map. With more (a per-token call inside
        // a batched decode), `reject_decode_lora` refuses an adapter-routed batch
        // instead.
        let single_seq_decode = ctx.attn_metadata.as_ref().map_or(1, |m| m.num_seqs) <= 1;
        if !single_seq_decode {
            self.reject_decode_lora(ctx, "forward")?;
        }
        // 2026-09-25: With one sequence, the router delta folds onto
        // `gate_logits` before top-k and the expert gate/up/down deltas onto
        // their intermediates; each fold's `moe_route_gate` still refuses a
        // `Refuse` route.
        //
        // DFlash capture layers route this token through `forward_prefill(1)`
        // when the `frankenstein_decode_via_prefill` lever
        // (`METRALE_FRANKENSTEIN_DECODE_VIA_PREFILL`) is on; every other layer
        // takes the decode path below.
        if self.is_dflash_capture_layer && ctx.levers.frankenstein_decode_via_prefill {
            if ctx.stats.once("log:moe_route") {
                tracing::info!(
                    "FRANKENSTEIN: routing DFlash capture-layer MoE decode through forward_prefill(M=1) (one-time log)"
                );
            }
            self.forward_prefill(input, 1, ctx, stream)?;
            return Ok(ctx.buffers.moe_output());
        }

        let h = ctx.config.hidden_size as u32;
        let inter = ctx.config.moe_intermediate_size as u32;
        let shared_inter = ctx.config.shared_expert_intermediate_size as u32;
        let num_experts = ctx.config.num_experts as u32;
        let top_k = ctx.config.num_experts_per_tok as u32;
        let profile = ctx.profile;

        macro_rules! prof {
            ($label:expr, $body:expr) => {{
                if profile {
                    let t = std::time::Instant::now();
                    let r = $body;
                    ctx.gpu.synchronize(stream)?;
                    tracing::info!("    MoE {}: {:.0}μs", $label, t.elapsed().as_micros());
                    r
                } else {
                    $body
                }
            }};
        }

        let scratch = ctx.buffers.scratch();
        let indices_dev = scratch;
        let weights_dev = scratch.offset(top_k as usize * 4);

        {
            let router_in = self.router_input(input, 1, h, ctx, stream)?;
            let gate_logits = ctx.buffers.gate_logits();
            prof!("gate", {
                self.decode_gate_gemv(router_in, gate_logits, h, ctx, stream)
            })?;

            // 2026-09-25: Fold the router LoRA delta onto `gate_logits` before
            // top-k, with the prefill hook at n=1. It launches only device
            // kernels, so it can be captured; a no-op without a router delta.
            if single_seq_decode {
                self.apply_router_lora_prefill(router_in, gate_logits, 1, ctx, stream)?;
            }

            prof!("topk", {
                if let Some(tid2eid) = self.tid2eid_dev {
                    // 2026-09-25: Hash routing: the experts are the static
                    // `tid2eid[token_id]` row; the gate's sqrtsoftplus scores
                    // weight them (`moe_hash_route.cu`).
                    let token_ids = ctx.token_ids.ok_or_else(|| {
                        anyhow::anyhow!(
                            "DeepSeek-V4 hash-MoE layer requires ForwardContext.token_ids (decode)"
                        )
                    })?;
                    ops::moe_hash_route(
                        ctx.gpu,
                        self.moe_hash_route_k,
                        gate_logits,
                        tid2eid,
                        token_ids,
                        indices_dev,
                        weights_dev,
                        num_experts,
                        top_k,
                        ctx.config.norm_topk_prob,
                        ctx.config.routed_scaling_factor as f32,
                        stream,
                    )
                } else {
                    self.decode_topk_scored(
                        gate_logits,
                        indices_dev,
                        weights_dev,
                        num_experts,
                        top_k,
                        ctx,
                        stream,
                    )
                }
            })?;
        }

        if tracing::enabled!(tracing::Level::DEBUG) && !ctx.graph_capture {
            debug_dumps::debug_routing(ctx, indices_dev, weights_dev, top_k, stream)?;
        }

        // 2026-09-25: `pre_expert_norm`, when the layer has one, runs after
        // routing into `ssm_deinterleaved`, leaving `input` unchanged for the
        // blend's shared-expert gate below.
        let expert_input = if let Some(ref norm_w) = self.pre_expert_norm {
            let normed = ctx.buffers.ssm_deinterleaved();
            let eps = ctx.config.rms_norm_eps as f32;
            prof!("pre_expert_norm", {
                ops::rms_norm(
                    ctx.gpu,
                    self.pre_expert_norm_k,
                    input,
                    norm_w,
                    normed,
                    1,
                    h,
                    eps,
                    stream,
                )
            })?;
            normed
        } else {
            input
        };

        let expert_gate_out = ctx.buffers.expert_gate_out();
        let expert_up_out = ctx.buffers.expert_up_out();
        let expert_down_out = ctx.buffers.expert_down_out();
        // 2026-09-25: The start of `logits()` is the shared-expert gate scratch
        // here. Another user of `logits()` during the MoE must start past it
        // (`trait_impl/decode_b.rs` puts decode metadata at offset 65536).
        let shared_gate_scratch = ctx.buffers.logits();
        let shared_up_scratch = ctx.buffers.ssm_qkvz();
        let shared_out = ctx.buffers.attn_output();

        if let (Some(gp), Some(up), Some(dp), Some(shared)) = (
            self.bf16_gate_weight_ptrs,
            self.bf16_up_weight_ptrs,
            self.bf16_down_weight_ptrs,
            self.bf16_shared_expert,
        ) {
            // 2026-09-25: BF16 experts (`set_bf16_experts`).
            prof!("exp_gate_up_bf16", {
                ops::moe_expert_gate_up_shared_bf16(
                    ctx.gpu,
                    self.moe_expert_gate_up_shared_bf16_k,
                    expert_input,
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
                )
            })?;
            if single_seq_decode {
                self.apply_expert_lora_decode_gateup(
                    expert_gate_out,
                    expert_up_out,
                    expert_input,
                    indices_dev,
                    top_k,
                    top_k,
                    DevicePtr::NULL,
                    ctx,
                    stream,
                )?;
            }
            prof!("exp_silu_down_bf16", {
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
                )
            })?;
        } else if let (Some(gp), Some(up), Some(dp), Some(sh)) = (
            &self.fp8_gate_weight_ptrs,
            &self.fp8_up_weight_ptrs,
            &self.fp8_down_weight_ptrs,
            &self.fp8_shared_expert,
        ) {
            prof!("exp_gate_up_fp8", {
                ops::moe_expert_gate_up_shared_fp8(
                    ctx.gpu,
                    self.moe_expert_gate_up_shared_fp8,
                    expert_input,
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
                )
            })?;

            if single_seq_decode {
                self.apply_expert_lora_decode_gateup(
                    expert_gate_out,
                    expert_up_out,
                    expert_input,
                    indices_dev,
                    top_k,
                    top_k,
                    DevicePtr::NULL,
                    ctx,
                    stream,
                )?;
            }
            prof!("exp_silu_down_fp8", {
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
                )
            })?;
        } else if self.use_t_layout_for_decode() {
            prof!("exp_unified_t", {
                self.dispatch_unified_t_decode(
                    ctx,
                    expert_input,
                    expert_gate_out,
                    expert_up_out,
                    expert_down_out,
                    shared_gate_scratch,
                    shared_up_scratch,
                    shared_out,
                    indices_dev,
                    h,
                    inter,
                    top_k,
                    single_seq_decode,
                    stream,
                )
            })?;
        } else {
            prof!("exp_gate_up", {
                ops::moe_expert_gate_up_shared(
                    ctx.gpu,
                    self.moe_expert_gate_up_shared,
                    expert_input,
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
                )
            })?;

            if single_seq_decode {
                self.apply_expert_lora_decode_gateup(
                    expert_gate_out,
                    expert_up_out,
                    expert_input,
                    indices_dev,
                    top_k,
                    top_k,
                    DevicePtr::NULL,
                    ctx,
                    stream,
                )?;
            }

            if tracing::enabled!(tracing::Level::DEBUG) && !ctx.graph_capture {
                debug_dumps::debug_expert_intermediates(
                    ctx,
                    expert_gate_out,
                    expert_up_out,
                    shared_gate_scratch,
                    shared_up_scratch,
                    stream,
                )?;
            }

            prof!("exp_silu_down", {
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
                )
            })?;
        }

        // 2026-09-25: Fold the expert down_proj LoRA delta into `expert_down_out`
        // (`[top_k, hidden]`) in place, recomputing `act(gate) * up` from
        // `expert_gate_out`/`expert_up_out`. It runs before the blend, so the
        // routing weight scales base plus delta, and before the EP memset below
        // reuses `expert_gate_out`. A null row map leaves the decision to
        // `moe_route_gate`. A no-op without an expert adapter.
        if single_seq_decode {
            self.apply_expert_lora_decode_down(
                expert_gate_out,
                expert_up_out,
                expert_down_out,
                indices_dev,
                top_k,
                top_k,
                DevicePtr::NULL,
                ctx,
                stream,
            )?;
        }

        if self.has_mixed_bf16_shared_expert() {
            self.run_bf16_shared_expert(
                input,
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

        if tracing::enabled!(tracing::Level::DEBUG) && !ctx.graph_capture {
            ctx.gpu.synchronize(stream)?;
            let mut down_buf = vec![0u8; 16];
            ctx.gpu.copy_d2h(expert_down_out, &mut down_buf)?;
            let down_vals: Vec<f32> = (0..8)
                .map(|i| {
                    let bits = u16::from_le_bytes([down_buf[i * 2], down_buf[i * 2 + 1]]);
                    f32::from_bits((bits as u32) << 16)
                })
                .collect();
            tracing::info!("  MoE down_out[slot0,0..8]: {:?}", down_vals);
            let mut sh_buf = vec![0u8; 16];
            ctx.gpu.copy_d2h(shared_out, &mut sh_buf)?;
            let sh_vals: Vec<f32> = (0..8)
                .map(|i| {
                    let bits = u16::from_le_bytes([sh_buf[i * 2], sh_buf[i * 2 + 1]]);
                    f32::from_bits((bits as u32) << 16)
                })
                .collect();
            tracing::info!("  MoE shared_out[0..8]: {:?}", sh_vals);
        }

        // 2026-09-25: `moe_weighted_sum_blend` writes `output = Σ w·expert_out +
        // sigmoid(dot(input, gate_w)) · shared_out` (sigmoid taken as 1 with no
        // gate weight). Every rank computes the same shared expert, so under EP
        // a zeroed buffer stands in for `shared_out` and the shared expert is
        // added once after the all-reduce; inside the blend it would be summed
        // `ep_world_size` times.
        let output = ctx.buffers.moe_output();
        let is_ep = ctx.comm.is_some() && ctx.config.ep_world_size > 1;
        let shared_for_blend = if is_ep && !shared_out.is_null() {
            let zero_buf = ctx.buffers.expert_gate_out();
            ctx.gpu.memset_async(zero_buf, 0, h as usize * 2, stream)?;
            zero_buf
        } else {
            shared_out
        };
        prof!("wsum_blend", {
            ops::moe_weighted_sum_blend(
                ctx.gpu,
                self.moe_weighted_sum_blend,
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
            )
        })?;

        // 2026-09-25: Each rank's partial holds only its local experts, so the
        // sum over ranks is the full routed output.
        self.ep_reduce_shared(output, shared_out, input, h, ctx, stream)?;

        if tracing::enabled!(tracing::Level::DEBUG) && !ctx.graph_capture {
            ctx.gpu.synchronize(stream)?;
            let mut buf = vec![0u8; 8];
            ctx.gpu.copy_d2h(output, &mut buf)?;
            let vals: Vec<f32> = (0..4)
                .map(|i| {
                    let lo = buf[i * 2];
                    let hi = buf[i * 2 + 1];
                    f32::from_bits(((lo as u32) | ((hi as u32) << 8)) << 16)
                })
                .collect();
            tracing::info!("  MoE output: {:?}", vals);
        }

        Ok(output)
    }
}
