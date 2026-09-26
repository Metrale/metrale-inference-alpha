// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-26: The expert-parallel tail of `MoeLayer::forward`: all-reduce the routed output
//! over ranks, then add the shared expert once. A no-op without EP.
//!
//! Owner: model-layers (MoE).
//! Invariants: none beyond the types.

use super::*;

impl MoeLayer {
    pub(super) fn ep_reduce_shared(
        &self,
        output: DevicePtr,
        shared_out: DevicePtr,
        input: DevicePtr,
        h: u32,
        ctx: &ForwardContext,
        stream: u64,
    ) -> Result<()> {
        if let Some(comm) = ctx.comm
            && ctx.config.ep_world_size > 1
        {
            if ctx.graph_capture {
                comm.all_reduce(output.0, h as usize * 2)?;
            } else {
                comm.all_reduce_async(output.0, h as usize * 2, stream)?;
            }
            // 2026-09-25: Add the shared expert once, through its sigmoid gate:
            // `output += sigmoid(dot(input, gate_w)) * shared_out`.
            if !shared_out.is_null() {
                if self.weights.shared_expert_gate.weight.0 == 0 {
                    // 2026-09-25: No gate weight: the shared expert is added at full strength.
                    ops::residual_add(ctx.gpu, self.residual_add, output, shared_out, h, stream)?;
                } else {
                    ops::moe_batched_blend(
                        ctx.gpu,
                        self.moe_batched_blend,
                        output,
                        shared_out,
                        input,
                        self.weights.shared_expert_gate.weight,
                        h,
                        1,
                        stream,
                    )?;
                }
            }
        }
        Ok(())
    }
}
