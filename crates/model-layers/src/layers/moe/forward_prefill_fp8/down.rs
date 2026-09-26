// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-26: The routed experts' activation and grouped down GEMM of `forward_prefill_fp8`,
//! from the sorted gate/up rows into `expert_down_out`.
//!
//! Owner: model-layers (MoE).
//! Invariants: none beyond the types.

use super::*;

impl MoeLayer {
    /// 2026-09-26: W8A8 down, taken when `fp8_blockscaled_prefill` and its kernels
    /// resolved and `max_m_tiles > 0`.
    pub(super) fn fp8_prefill_down_w8a8(
        &self,
        dp: &Fp8ExpertPtrTable,
        expert_gate_out: DevicePtr,
        expert_up_out: DevicePtr,
        expert_down_out: DevicePtr,
        expert_offsets: DevicePtr,
        total_expanded: u32,
        num_experts: u32,
        h: u32,
        inter: u32,
        n: u32,
        te: usize,
        ne: usize,
        max_m_tiles: u32,
        ctx: &ForwardContext,
        stream: u64,
        mt: &mut Option<std::time::Instant>,
    ) -> Result<()> {
        macro_rules! mprof {
            ($label:expr) => {
                mprof_step!(*mt, ctx, stream, n, $label)
            };
        }
        let m: usize = total_expanded as usize;
        let a_fp8_bytes: usize = m * inter as usize;
        let a_scale_bytes: usize = m * (inter as usize / 128) * 4;
        let down_in_fp8 = ctx.gpu.alloc(a_fp8_bytes)?;
        let down_in_scale = ctx.gpu.alloc(a_scale_bytes)?;
        if self.fused_silu_quant_ok(inter) {
            // 2026-09-25: `apply_expert_lora_prefill_down` reads the post-SiLU
            // BF16 `expert_gate_out`, so with a MoE LoRA installed the kernel
            // also writes those rows, in place. Each thread reads its gate
            // element before writing the same element, so the alias is safe.
            let lora_bf16_out = if self.lora.is_some() {
                expert_gate_out
            } else {
                metrale_gpu_runtime::gpu::DevicePtr::NULL
            };
            ops::silu_mul_quant_fp8(
                ctx.gpu,
                self.silu_mul_quant_fp8_k,
                expert_gate_out,
                expert_up_out,
                down_in_fp8,
                down_in_scale,
                lora_bf16_out,
                m as u32,
                inter,
                stream,
            )?;
        } else {
            ops::silu_mul(
                ctx.gpu,
                self.moe_act_mul,
                expert_gate_out,
                expert_up_out,
                expert_gate_out,
                total_expanded * inter,
                stream,
            )?;
            ops::per_token_group_quant_fp8(
                ctx.gpu,
                self.per_token_group_quant_fp8_k,
                expert_gate_out,
                down_in_fp8,
                down_in_scale,
                m as u32,
                inter,
                stream,
            )?;
        }
        mprof!("silu_mul_quant");
        if self.moe_w8a8_grouped_gemm_pm4_k.0 != 0 && self.moe_build_tile_worklist_k.0 != 0 {
            // 2026-09-25: The down GEMM (N = h) needs its own work-list. Its
            // input rows are already sorted, so `sorted_token_ids` is NULL.
            let n_tiles_dn = h.div_ceil(PM4_N_TILE);
            let wl_cap_items = (te.div_ceil(PM4_M_TILE as usize) + ne + 1) * n_tiles_dn as usize;
            let wl_dn = ctx.gpu.alloc(wl_cap_items * 2 * 4)?;
            let tt_dn = ctx.gpu.alloc(4)?;
            ops::moe_build_tile_worklist(
                ctx.gpu,
                self.moe_build_tile_worklist_k,
                expert_offsets,
                dp.weight_ptrs,
                wl_dn,
                tt_dn,
                num_experts,
                n_tiles_dn,
                PM4_M_TILE,
                stream,
            )?;
            mprof!("tile_worklist");
            ops::moe_w8a8_grouped_gemm_pm4(
                ctx.gpu,
                self.moe_w8a8_grouped_gemm_pm4_k,
                down_in_fp8,
                down_in_scale,
                dp.weight_ptrs,
                dp.scale_ptrs,
                expert_down_out,
                expert_offsets,
                metrale_gpu_runtime::gpu::DevicePtr(0),
                num_experts,
                h,
                inter,
                wl_dn,
                tt_dn,
                wl_cap_items as u32,
                stream,
            )?;
            mprof!("grouped_gemm_w8a8");
            ctx.gpu.synchronize(stream)?;
            ctx.gpu.free(wl_dn)?;
            ctx.gpu.free(tt_dn)?;
        } else {
            ops::moe_w8a8_grouped_gemm(
                ctx.gpu,
                self.moe_w8a8_grouped_gemm_k,
                down_in_fp8,
                down_in_scale,
                dp.weight_ptrs,
                dp.scale_ptrs,
                expert_down_out,
                expert_offsets,
                metrale_gpu_runtime::gpu::DevicePtr(0),
                num_experts,
                h,
                inter,
                max_m_tiles,
                stream,
            )?;
            mprof!("grouped_gemm_w8a8");
            ctx.gpu.synchronize(stream)?;
        }
        ctx.gpu.free(down_in_fp8)?;
        ctx.gpu.free(down_in_scale)?;
        Ok(())
    }

    /// 2026-09-26: FP8 (W8A16) down, taken otherwise when `max_m_tiles > 0`.
    pub(super) fn fp8_prefill_down_fp8(
        &self,
        dp: &Fp8ExpertPtrTable,
        expert_gate_out: DevicePtr,
        expert_up_out: DevicePtr,
        expert_down_out: DevicePtr,
        expert_offsets: DevicePtr,
        total_expanded: u32,
        num_experts: u32,
        h: u32,
        inter: u32,
        n: u32,
        te: usize,
        ne: usize,
        ctx: &ForwardContext,
        stream: u64,
        mt: &mut Option<std::time::Instant>,
    ) -> Result<()> {
        macro_rules! mprof {
            ($label:expr) => {
                mprof_step!(*mt, ctx, stream, n, $label)
            };
        }
        ops::silu_mul(
            ctx.gpu,
            self.moe_act_mul,
            expert_gate_out,
            expert_up_out,
            expert_gate_out,
            total_expanded * inter,
            stream,
        )?;
        mprof!("silu_mul");
        // 2026-09-25: The down GEMM (N = h) needs its own work-list, built
        // from `dp.weight_ptrs`. Its input rows are already sorted, so
        // `sorted_token_ids` is NULL.
        let n_tiles_dn = h.div_ceil(PM4_N_TILE);
        let wl_cap_items = (te.div_ceil(PM4_M_TILE as usize) + ne + 1) * n_tiles_dn as usize;
        let wl_dn = ctx.gpu.alloc(wl_cap_items * 2 * 4)?;
        let tt_dn = ctx.gpu.alloc(4)?;
        ops::moe_build_tile_worklist(
            ctx.gpu,
            self.moe_build_tile_worklist_k,
            expert_offsets,
            dp.weight_ptrs,
            wl_dn,
            tt_dn,
            num_experts,
            n_tiles_dn,
            PM4_M_TILE,
            stream,
        )?;
        mprof!("tile_worklist");
        ops::moe_fp8_grouped_gemm(
            ctx.gpu,
            self.moe_fp8_grouped_gemm_k,
            expert_gate_out,
            dp.weight_ptrs,
            dp.scale_ptrs,
            expert_down_out,
            expert_offsets,
            metrale_gpu_runtime::gpu::DevicePtr(0),
            num_experts,
            h,
            inter,
            wl_dn,
            tt_dn,
            wl_cap_items as u32,
            stream,
        )?;
        mprof!("grouped_gemm_fp8");
        ctx.gpu.synchronize(stream)?;
        ctx.gpu.free(wl_dn)?;
        ctx.gpu.free(tt_dn)?;
        Ok(())
    }
}
