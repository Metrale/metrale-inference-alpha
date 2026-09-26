// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-25: `MoeLayer::batched_gate_logits`, the routing-logits phase of
//! `forward_batched`: the gate GEMM over N rows and the router (`mlp.gate`)
//! LoRA fold, in place, before top-k.
//!
//! Owner: model-layers (MoE).
//! Invariants: none beyond the types.

use super::*;

impl MoeLayer {
    /// 2026-09-25: Compute the batched routing logits and fold the router LoRA
    /// delta in place. Returns `(gate_logits, fp32_gate, gate_elem)`:
    /// `gate_logits` is `gate_logits_f32()` when `fp32_gate`, else
    /// `gate_logits()`, and `gate_elem` is its element size (4 or 2).
    #[allow(clippy::too_many_arguments)]
    pub(super) fn batched_gate_logits(
        &self,
        input: DevicePtr,
        n: u32,
        h: u32,
        num_experts: u32,
        row_adapter_base: DevicePtr,
        ctx: &ForwardContext,
        stream: u64,
    ) -> Result<(DevicePtr, bool, usize)> {
        // 2026-09-25: FP32 gate logits, so two experts whose logits differ by
        // less than a BF16 ULP are ranked by their FP32 values. Two levers:
        // - `fp32_routing` (`METRALE_FP32_ROUTING`, see `fp32_routing_active`):
        //   the pre-MoE norm wrote an FP32 `moe_router_in_f32`, which the gate
        //   GEMM reads through `dense_gemm_f32in`.
        // - `fp32_gate` (`METRALE_FP32_GATE`): BF16 `router_in`, FP32 GEMM output
        //   (`dense_gemm_f32out`).
        // Both need a dense gate, no correction bias, `moe_topk_f32` and their
        // GEMM kernel; otherwise the logits stay BF16.
        let fp32_routing = self.fp32_routing_active(ctx.levers);
        let fp32_gate = fp32_routing
            || (self.gate_nvfp4.is_none()
                && self.correction_bias_dev.is_none()
                && self.dense_gemm_f32out.0 != 0
                && self.moe_topk_f32.0 != 0
                && ctx.levers.fp32_gate);
        let gate_elem = if fp32_gate { 4usize } else { 2usize };

        let router_in = self.router_input(input, n, h, ctx, stream)?;
        let gate_logits = if fp32_gate {
            ctx.buffers.gate_logits_f32()
        } else {
            ctx.buffers.gate_logits()
        };
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
        } else if fp32_routing {
            ops::dense_gemm(
                ctx.gpu,
                self.dense_gemm_f32in,
                ctx.buffers.moe_router_in_f32(),
                &self.weights.gate,
                gate_logits,
                n,
                num_experts,
                h,
                stream,
            )?;
        } else {
            ops::dense_gemm(
                ctx.gpu,
                if fp32_gate {
                    self.dense_gemm_f32out
                } else {
                    self.dense_gemm
                },
                router_in,
                &self.weights.gate,
                gate_logits,
                n,
                num_experts,
                h,
                stream,
            )?;
        }
        // 2026-09-25: `dump_gate_logits` reads BF16, so the FP32-gate path skips it.
        if !fp32_gate {
            super::dump::dump_gate_logits(ctx.gpu, stream, gate_logits, n, num_experts)?;
        }

        // 2026-09-25: Fold the router LoRA delta onto all N rows of `gate_logits`
        // before top-k. With a `row_adapter` map, base rows are skipped on the
        // device; without one, `moe_route_gate` decides for the whole batch. A
        // no-op without a router adapter; refused on the FP32-gate path when the
        // batch folds (`apply_router_lora_batched`).
        self.apply_router_lora_batched(
            router_in,
            gate_logits,
            n,
            row_adapter_base,
            fp32_gate,
            ctx,
            stream,
        )?;

        Ok((gate_logits, fp32_gate, gate_elem))
    }
}
