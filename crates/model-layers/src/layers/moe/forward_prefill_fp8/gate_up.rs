// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-26: The routed experts' grouped gate and up GEMMs of `forward_prefill_fp8`, into
//! the sorted `expert_gate_out` / `expert_up_out` rows.
//!
//! Owner: model-layers (MoE).
//! Invariants: none beyond the types.

use super::*;

impl MoeLayer {
    /// 2026-09-26: W8A8 gate/up, taken when `fp8_blockscaled_prefill` and its kernels
    /// resolved and `max_m_tiles > 0`.
    pub(super) fn fp8_prefill_gate_up_w8a8(
        &self,
        input: DevicePtr,
        gp: &Fp8ExpertPtrTable,
        up: &Fp8ExpertPtrTable,
        expert_gate_out: DevicePtr,
        expert_up_out: DevicePtr,
        expert_offsets: DevicePtr,
        sorted_token_ids: DevicePtr,
        num_tokens: usize,
        num_experts: u32,
        h: u32,
        inter: u32,
        n: u32,
        te: usize,
        ne: usize,
        max_m_tiles: u32,
        fp8_scratch: &MoeFp8Scratch,
        ctx: &ForwardContext,
        stream: u64,
        mt: &mut Option<std::time::Instant>,
    ) -> Result<()> {
        macro_rules! mprof {
            ($label:expr) => {
                mprof_step!(*mt, ctx, stream, n, $label)
            };
        }
        // 2026-09-25: One quantised input serves both gate and up.
        let m = num_tokens;
        let input_fp8 = fp8_scratch.activation;
        let input_a_scale = fp8_scratch.scales;
        ops::per_token_group_quant_fp8(
            ctx.gpu,
            self.per_token_group_quant_fp8_k,
            input,
            input_fp8,
            input_a_scale,
            m as u32,
            h,
            stream,
        )?;
        if self.try_adaptive_fp8(
            input_fp8,
            input_a_scale,
            &[(gp, expert_gate_out), (up, expert_up_out)],
            expert_offsets,
            sorted_token_ids,
            inter,
            h,
            num_tokens,
            ctx,
            stream,
        )? {
            mprof!("grouped_gemm_w8a8_adaptive");
        } else if self.moe_w8a8_grouped_gemm_pm4_k.0 != 0 && self.moe_build_tile_worklist_k.0 != 0 {
            // 2026-09-25: PM4 W8A8 over the compacted work-list. One work-list
            // serves gate and up (same expert_offsets, weight NULL-ness and
            // N = inter). Builder and GEMMs share `stream`, which orders the
            // work-list writes before the reads.
            let n_tiles_gu = inter.div_ceil(PM4_N_TILE);
            let wl_cap_items = (te.div_ceil(PM4_M_TILE as usize) + ne + 1) * n_tiles_gu as usize;
            let wl_gu = fp8_scratch.worklist;
            let tt_gu = fp8_scratch.total_tiles;
            ops::moe_build_tile_worklist(
                ctx.gpu,
                self.moe_build_tile_worklist_k,
                expert_offsets,
                gp.weight_ptrs,
                wl_gu,
                tt_gu,
                num_experts,
                n_tiles_gu,
                PM4_M_TILE,
                stream,
            )?;
            mprof!("tile_worklist");
            ops::moe_w8a8_grouped_gemm_pm4(
                ctx.gpu,
                self.moe_w8a8_grouped_gemm_pm4_k,
                input_fp8,
                input_a_scale,
                gp.weight_ptrs,
                gp.scale_ptrs,
                expert_gate_out,
                expert_offsets,
                sorted_token_ids,
                num_experts,
                inter,
                h,
                wl_gu,
                tt_gu,
                wl_cap_items as u32,
                stream,
            )?;
            mprof!("grouped_gemm_w8a8");
            ops::moe_w8a8_grouped_gemm_pm4(
                ctx.gpu,
                self.moe_w8a8_grouped_gemm_pm4_k,
                input_fp8,
                input_a_scale,
                up.weight_ptrs,
                up.scale_ptrs,
                expert_up_out,
                expert_offsets,
                sorted_token_ids,
                num_experts,
                inter,
                h,
                wl_gu,
                tt_gu,
                wl_cap_items as u32,
                stream,
            )?;
            mprof!("grouped_gemm_w8a8");
        } else {
            ops::moe_w8a8_grouped_gemm(
                ctx.gpu,
                self.moe_w8a8_grouped_gemm_k,
                input_fp8,
                input_a_scale,
                gp.weight_ptrs,
                gp.scale_ptrs,
                expert_gate_out,
                expert_offsets,
                sorted_token_ids,
                num_experts,
                inter,
                h,
                max_m_tiles,
                stream,
            )?;
            mprof!("grouped_gemm_w8a8");
            ops::moe_w8a8_grouped_gemm(
                ctx.gpu,
                self.moe_w8a8_grouped_gemm_k,
                input_fp8,
                input_a_scale,
                up.weight_ptrs,
                up.scale_ptrs,
                expert_up_out,
                expert_offsets,
                sorted_token_ids,
                num_experts,
                inter,
                h,
                max_m_tiles,
                stream,
            )?;
            mprof!("grouped_gemm_w8a8");
        }
        Ok(())
    }

    /// 2026-09-26: FP8 (W8A16) gate/up, taken otherwise when `max_m_tiles > 0`.
    pub(super) fn fp8_prefill_gate_up_fp8(
        &self,
        input: DevicePtr,
        gp: &Fp8ExpertPtrTable,
        up: &Fp8ExpertPtrTable,
        expert_gate_out: DevicePtr,
        expert_up_out: DevicePtr,
        expert_offsets: DevicePtr,
        sorted_token_ids: DevicePtr,
        num_experts: u32,
        h: u32,
        inter: u32,
        n: u32,
        te: usize,
        ne: usize,
        fp8_scratch: &MoeFp8Scratch,
        ctx: &ForwardContext,
        stream: u64,
        mt: &mut Option<std::time::Instant>,
    ) -> Result<()> {
        macro_rules! mprof {
            ($label:expr) => {
                mprof_step!(*mt, ctx, stream, n, $label)
            };
        }
        // 2026-09-25: FP8 grouped GEMM over the compacted work-list. One
        // work-list serves gate and up (same expert_offsets, weight NULL-ness
        // and N = inter). Builder and GEMMs share `stream`, which orders the
        // work-list writes before the reads.
        let n_tiles_gu = inter.div_ceil(PM4_N_TILE);
        // 2026-09-25: Work-item bound: the sum of ceil(M_e / 128) over experts
        // is at most ceil(te / 128) + ne; times n_tiles items, 2 u32 words
        // per item.
        let wl_cap_items = (te.div_ceil(PM4_M_TILE as usize) + ne + 1) * n_tiles_gu as usize;
        let wl_gu = fp8_scratch.worklist;
        let tt_gu = fp8_scratch.total_tiles;
        ops::moe_build_tile_worklist(
            ctx.gpu,
            self.moe_build_tile_worklist_k,
            expert_offsets,
            gp.weight_ptrs,
            wl_gu,
            tt_gu,
            num_experts,
            n_tiles_gu,
            PM4_M_TILE,
            stream,
        )?;
        mprof!("tile_worklist");
        ops::moe_fp8_grouped_gemm(
            ctx.gpu,
            self.moe_fp8_grouped_gemm_k,
            input,
            gp.weight_ptrs,
            gp.scale_ptrs,
            expert_gate_out,
            expert_offsets,
            sorted_token_ids,
            num_experts,
            inter,
            h,
            wl_gu,
            tt_gu,
            wl_cap_items as u32,
            stream,
        )?;
        mprof!("grouped_gemm_fp8");
        ops::moe_fp8_grouped_gemm(
            ctx.gpu,
            self.moe_fp8_grouped_gemm_k,
            input,
            up.weight_ptrs,
            up.scale_ptrs,
            expert_up_out,
            expert_offsets,
            sorted_token_ids,
            num_experts,
            inter,
            h,
            wl_gu,
            tt_gu,
            wl_cap_items as u32,
            stream,
        )?;
        mprof!("grouped_gemm_fp8");
        Ok(())
    }
}
