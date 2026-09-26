// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-25: Unified-layout single-token decode: `dispatch_unified_t_decode` runs
//! the gate+up and silu+down kernels on the transposed expert tables
//! (gate_t/up_t/down_t plus shared_*_t). `forward` calls it when
//! `use_t_layout_for_decode()` holds.
//!
//! Owner: model-layers (MoE).
//! Invariants: none beyond the types.

use anyhow::Result;

use super::*;

impl MoeLayer {
    #[allow(clippy::too_many_arguments)]
    pub(super) fn dispatch_unified_t_decode(
        &self,
        ctx: &ForwardContext,
        expert_input: DevicePtr,
        expert_gate_out: DevicePtr,
        expert_up_out: DevicePtr,
        expert_down_out: DevicePtr,
        shared_gate_scratch: DevicePtr,
        shared_up_scratch: DevicePtr,
        shared_out: DevicePtr,
        indices_dev: DevicePtr,
        h: u32,
        inter: u32,
        top_k: u32,
        single_seq_decode: bool,
        stream: u64,
    ) -> Result<()> {
        // 2026-09-25: `use_t_layout_for_decode()` guarantees the three transposed
        // tables (unified layout on, hybrid off, no lazy down scratch); the
        // `expect`s panic if a caller breaks that.
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
        // 2026-09-25: The `_e8m0` fused kernels read the shared expert as NVFP4
        // (GROUP_SIZE 16); `expect` panics on any other shared format.
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
                "decode gate_up_shared_t (unified_t)",
            ),
            expert_input,
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
        // 2026-09-25: Fold routed-expert gate/up LoRA deltas onto
        // `expert_gate_out`/`expert_up_out` before silu+down reads them;
        // single-sequence decode only. A no-op without gate/up deltas. For
        // multi-sequence decode, `forward` calls `reject_decode_lora` first.
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
        ops::moe_expert_silu_down_shared_t(
            ctx.gpu,
            self.e8m0_or(
                self.moe_expert_silu_down_shared_t_k,
                self.moe_expert_silu_down_shared_t_e8m0_k,
                "decode silu_down_shared_t (unified_t)",
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
        Ok(())
    }
}
