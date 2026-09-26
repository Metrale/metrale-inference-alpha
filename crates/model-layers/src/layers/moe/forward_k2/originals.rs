// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-25: The originals-layout (untransposed) NVFP4 branch of `forward_k2`,
//! including the mixed NVFP4-routed / BF16-shared expert.
//!
//! Owner: model-layers (MoE).
//! Invariants: none beyond the types.

use super::super::*;
use super::batch2_block_width;

impl MoeLayer {
    /// 2026-09-25: NVFP4 batch2 over the originals `[N, K/2]` layout. With
    /// `mixed_bf16_shared` the kernels get NULL shared weights, write zeros to the
    /// shared outputs, and the BF16 shared expert runs after silu_down.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn forward_k2_originals(
        &self,
        input: DevicePtr,
        indices_dev: DevicePtr,
        weights_dev: DevicePtr,
        expert_gate_out: DevicePtr,
        expert_up_out: DevicePtr,
        expert_down_out: DevicePtr,
        shared_gate_scratch: DevicePtr,
        shared_up_scratch: DevicePtr,
        shared_down_out: DevicePtr,
        output: DevicePtr,
        inter: u32,
        h: u32,
        top_k: u32,
        is_ep: bool,
        mixed_bf16_shared: bool,
        ctx: &ForwardContext,
        stream: u64,
    ) -> Result<()> {
        let null_shared = QuantizedWeight::null();
        let batch2_block = batch2_block_width(ctx.config.hidden_size);
        ops::moe_expert_gate_up_shared_batch2(
            ctx.gpu,
            self.moe_expert_gate_up_shared_batch2,
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
            if mixed_bf16_shared {
                &null_shared
            } else {
                &self.weights.shared_expert.gate_proj
            },
            shared_gate_scratch,
            if mixed_bf16_shared {
                &null_shared
            } else {
                &self.weights.shared_expert.up_proj
            },
            shared_up_scratch,
            inter,
            h,
            top_k,
            batch2_block,
            stream,
        )?;
        ops::moe_expert_silu_down_shared_batch2(
            ctx.gpu,
            self.moe_expert_silu_down_shared_batch2,
            expert_gate_out,
            expert_up_out,
            self.down_ptrs.packed_ptrs,
            self.down_ptrs.scale_ptrs,
            self.down_ptrs.scale2_vals,
            expert_down_out,
            indices_dev,
            shared_gate_scratch,
            shared_up_scratch,
            if mixed_bf16_shared {
                &null_shared
            } else {
                &self.weights.shared_expert.down_proj
            },
            shared_down_out,
            h,
            inter,
            top_k,
            batch2_block,
            stream,
        )?;
        if mixed_bf16_shared {
            let shared_inter = ctx.config.shared_expert_intermediate_size as u32;
            self.run_bf16_shared_expert(
                input,
                2,
                h,
                shared_inter,
                shared_gate_scratch,
                shared_up_scratch,
                shared_down_out,
                ctx,
                stream,
            )?;
        }
        // 2026-09-25: With EP, blend a zeroed shared term (expert_gate_out is
        // not read after silu_down); `forward_k2` adds the shared expert after
        // the all-reduce.
        let shared_for_blend = if is_ep && !shared_down_out.is_null() {
            ctx.gpu
                .memset_async(expert_gate_out, 0, 2 * h as usize * 2, stream)?;
            expert_gate_out
        } else {
            shared_down_out
        };
        ops::moe_weighted_sum_blend_batch2(
            ctx.gpu,
            self.moe_weighted_sum_blend_batch2,
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
        Ok(())
    }
}
