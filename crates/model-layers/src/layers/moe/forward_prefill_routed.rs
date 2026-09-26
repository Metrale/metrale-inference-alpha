// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-25: The routed-expert phase of the NVFP4 `MoeLayer::forward_prefill`:
//! M-tile grid sizing, grouped gate+up GEMM, activation, grouped down GEMM.
//!
//! Owner: model-layers (MoE).
//! Invariants: none beyond the types.

use super::*;

impl MoeLayer {
    /// 2026-09-25: Sizes the M-tile grid, runs the grouped gate+up GEMM, the
    /// activation (`moe_act_mul`) and the grouped down GEMM. The routed outputs
    /// land in `ctx.buffers.expert_down_out()` in expert-sorted order. `t0` is
    /// the caller's profile timer.
    #[allow(clippy::too_many_arguments)]
    // 2026-09-25: `cutlass_grouped_host` is checked with `.is_some()` in the `if`
    // that guards its `.expect`.
    #[allow(clippy::unnecessary_unwrap)]
    pub(super) fn run_routed_grouped_gemm(
        &self,
        expert_input: DevicePtr,
        expert_offsets: DevicePtr,
        sorted_token_ids: DevicePtr,
        n: u32,
        h: u32,
        inter: u32,
        num_experts: u32,
        top_k: u32,
        num_tokens: usize,
        ne: usize,
        t0: &mut Option<std::time::Instant>,
        ctx: &ForwardContext,
        stream: u64,
    ) -> Result<()> {
        macro_rules! prof_step {
            ($label:expr) => {
                if let Some(t) = t0.take() {
                    ctx.gpu.synchronize(stream)?;
                    let elapsed = t.elapsed().as_micros();
                    tracing::info!("  MoE prefill [{}] N={}: {}µs", $label, num_tokens, elapsed);
                    *t0 = Some(std::time::Instant::now());
                }
            };
        }

        let avg_per_expert = (num_tokens * top_k as usize).div_ceil(ne);
        // 2026-09-25: The default bound puts all n * top_k routed rows in one
        // expert, so no expert's rows are cut off. METRALE_MOE_PREFILL_MAX_LOAD_FACTOR
        // caps it at factor x the average rows per expert; an expert's rows past
        // that cap are then not computed.
        let worst_case_m_tiles = (num_tokens * top_k as usize).div_ceil(64).max(1) as u32;
        // 2026-09-25: Exact tiles copy the real expert offsets to the host (a D2H
        // copy on `stream`) and size the grid from the largest expert, never above
        // the bound. METRALE_MOE_PREFILL_EXACT_TILES=1/0 forces it; unset, it is on
        // for NVFP4 experts only. Never under graph capture. The unset default
        // depends on the expert format, so it is resolved here; `ModelLevers`
        // carries only the override.
        let exact_tiles = ctx
            .levers
            .moe_prefill_exact_tiles
            .unwrap_or(self.experts_scale_kind == crate::weight_map::WeightQuantFormat::Nvfp4)
            && !ctx.graph_capture;
        let max_m_tiles = if exact_tiles {
            let mut offsets = vec![0u8; (ne + 1) * 4];
            ctx.gpu
                .copy_d2h_on_stream(expert_offsets, &mut offsets, stream)?;
            let mut prev = 0u32;
            let mut max_rows = 0u32;
            for raw in offsets.chunks_exact(4).skip(1) {
                let cur = u32::from_le_bytes([raw[0], raw[1], raw[2], raw[3]]);
                max_rows = max_rows.max(cur.saturating_sub(prev));
                prev = cur;
            }
            max_rows.div_ceil(64).max(1).min(worst_case_m_tiles)
        } else {
            ctx.levers
                .moe_prefill_max_load_factor
                .map(|factor| {
                    let capped_rows = avg_per_expert.saturating_mul(factor);
                    worst_case_m_tiles.min(capped_rows.div_ceil(64).max(1) as u32)
                })
                .unwrap_or(worst_case_m_tiles)
        };
        super::dump::dump_expert_load(
            ctx.gpu,
            stream,
            expert_offsets,
            ne,
            num_tokens,
            avg_per_expert,
            max_m_tiles,
        );
        prof_step!("grid_setup");

        let total_expanded = n * top_k;

        let expert_gate_out = ctx.buffers.expert_gate_out();
        let expert_up_out = ctx.buffers.expert_up_out();
        // 2026-09-25: The intermediates are zeroed only when a comm is attached
        // (EP) or with METRALE_MOE_PREFILL_ZERO=1. Otherwise the grouped kernels
        // must write every sorted row that the unpermute reads.
        let force_zero = ctx.levers.moe_prefill_zero;
        if ctx.comm.is_some() || force_zero {
            let gate_bytes = total_expanded as usize * inter as usize * 2;
            let up_bytes = gate_bytes;
            let down_bytes = total_expanded as usize * h as usize * 2;
            ctx.gpu
                .memset_async(expert_gate_out, 0, gate_bytes, stream)?;
            ctx.gpu.memset_async(expert_up_out, 0, up_bytes, stream)?;
            ctx.gpu
                .memset_async(ctx.buffers.expert_down_out(), 0, down_bytes, stream)?;
        }
        // 2026-09-25: The CUTLASS gate_up copies the expert offsets to the host and
        // returns them; the CUTLASS down call reuses them instead of copying again.
        let mut cutlass_eoff: Option<Vec<i32>> = None;
        if max_m_tiles > 0 {
            // 2026-09-25: The CUTLASS grouped path reads the host snapshot built at
            // load, so it is tried before the `gate_ptrs_t` check and also serves
            // the originals-only layout. It precedes the E8M0 check below.
            if ctx.levers.moe_grouped_cutlass && self.cutlass_grouped_host.is_some() {
                // 2026-09-25: METRALE_HOLO_MOE_GROUPED_CUTLASS=1: the CUTLASS grouped
                // NVFP4 gate/up GEMM takes the token-major `expert_input` plus
                // `sorted_token_ids` and writes the sorted layout.
                cutlass_eoff = Some(ops::moe_grouped_gate_up_cutlass(
                    ctx.gpu,
                    self.cutlass_grouped_host.as_ref().expect("checked above"),
                    expert_input,
                    sorted_token_ids,
                    expert_gate_out,
                    expert_up_out,
                    expert_offsets,
                    inter,
                    h,
                    stream,
                )?);
            } else if let (Some(gp), Some(up)) = (&self.gate_ptrs_t, &self.up_ptrs_t) {
                if self.experts_scale_kind == crate::weight_map::WeightQuantFormat::Mxfp4E8m0 {
                    // 2026-09-25: E8M0 routed experts take the `_e8m0` kernel ahead
                    // of the FP4 and M128 NVFP4 kernels; panics if it is unresolved.
                    assert!(
                        self.moe_fused_gate_up_t_k64_e8m0.0 != 0,
                        "ARM-2: routed experts Mxfp4E8m0 but fused_gate_up_t_k64_e8m0 unresolved"
                    );
                    ops::moe_w4a16_fused_gate_up_k64_n128(
                        ctx.gpu,
                        self.moe_fused_gate_up_t_k64_e8m0,
                        expert_input,
                        gp.packed_ptrs,
                        gp.scale_ptrs,
                        gp.scale2_vals,
                        up.packed_ptrs,
                        up.scale_ptrs,
                        up.scale2_vals,
                        expert_gate_out,
                        expert_up_out,
                        expert_offsets,
                        sorted_token_ids,
                        num_experts,
                        inter,
                        h,
                        max_m_tiles,
                        stream,
                    )?;
                } else if self.gateup_fp4 && self.moe_fused_gate_up_t_k64_fp4.0 != 0 {
                    // 2026-09-25: METRALE_HOLO_MOE_GATEUP_FP4=1 with the kernel
                    // resolved: the block-scaled FP4 gate_up over the same
                    // `gate_ptrs_t`/`up_ptrs_t` tables (no extra weight memory).
                    ops::moe_w4a16_fused_gate_up_k64_n128(
                        ctx.gpu,
                        self.moe_fused_gate_up_t_k64_fp4,
                        expert_input,
                        gp.packed_ptrs,
                        gp.scale_ptrs,
                        gp.scale2_vals,
                        up.packed_ptrs,
                        up.scale_ptrs,
                        up.scale2_vals,
                        expert_gate_out,
                        expert_up_out,
                        expert_offsets,
                        sorted_token_ids,
                        num_experts,
                        inter,
                        h,
                        max_m_tiles,
                        stream,
                    )?;
                } else {
                    // 2026-09-25: The M=128 kernel needs METRALE_NVFP4_GATE_UP_M128=1
                    // and a resolved handle; its tile count is the M=64 count
                    // halved, rounded up.
                    let use_m128 =
                        self.nvfp4_gate_up_m128 && self.moe_fused_gate_up_t_k64_m128.0 != 0;
                    if use_m128 {
                        let max_m_tiles_m128 = max_m_tiles.div_ceil(2).max(1);
                        ops::moe_w4a16_fused_gate_up_k64_m128(
                            ctx.gpu,
                            self.moe_fused_gate_up_t_k64_m128,
                            expert_input,
                            gp.packed_ptrs,
                            gp.scale_ptrs,
                            gp.scale2_vals,
                            up.packed_ptrs,
                            up.scale_ptrs,
                            up.scale2_vals,
                            expert_gate_out,
                            expert_up_out,
                            expert_offsets,
                            sorted_token_ids,
                            num_experts,
                            inter,
                            h,
                            max_m_tiles_m128,
                            stream,
                        )?;
                    } else {
                        ops::moe_w4a16_fused_gate_up_k64_n128(
                            ctx.gpu,
                            self.moe_fused_gate_up_t_k64,
                            expert_input,
                            gp.packed_ptrs,
                            gp.scale_ptrs,
                            gp.scale2_vals,
                            up.packed_ptrs,
                            up.scale_ptrs,
                            up.scale2_vals,
                            expert_gate_out,
                            expert_up_out,
                            expert_offsets,
                            sorted_token_ids,
                            num_experts,
                            inter,
                            h,
                            max_m_tiles,
                            stream,
                        )?;
                    }
                }
            } else {
                // 2026-09-25: The untransposed fallback has no E8M0 variant;
                // `expect` panics for any expert format but NVFP4.
                self.experts_scale_kind.expect(
                    crate::weight_map::WeightQuantFormat::Nvfp4,
                    "prefill non-transposed gate_up fallback (no E8M0 variant wired)",
                );
                let (gp, up) = (&self.gate_ptrs, &self.up_ptrs);
                self.launch_grouped_gemm(
                    ctx.gpu,
                    expert_input,
                    gp.packed_ptrs,
                    gp.scale_ptrs,
                    gp.scale2_vals,
                    expert_gate_out,
                    expert_offsets,
                    sorted_token_ids,
                    num_experts,
                    inter,
                    h,
                    max_m_tiles,
                    stream,
                )?;
                self.launch_grouped_gemm(
                    ctx.gpu,
                    expert_input,
                    up.packed_ptrs,
                    up.scale_ptrs,
                    up.scale2_vals,
                    expert_up_out,
                    expert_offsets,
                    sorted_token_ids,
                    num_experts,
                    inter,
                    h,
                    max_m_tiles,
                    stream,
                )?;
            }
        }
        prof_step!("grouped_gate_up");

        let expert_down_out = ctx.buffers.expert_down_out();
        if max_m_tiles > 0 {
            // 2026-09-25: Fold routed-expert gate/up LoRA deltas onto the sorted
            // `expert_gate_out`/`expert_up_out` before the activation overwrites
            // `expert_gate_out`; this runs after whichever gate_up branch ran. A
            // no-op without gate/up deltas.
            self.apply_expert_lora_prefill_gateup(
                expert_gate_out,
                expert_up_out,
                expert_input,
                expert_offsets,
                sorted_token_ids,
                total_expanded,
                ctx,
                stream,
            )?;
            ops::silu_mul(
                ctx.gpu,
                self.moe_act_mul,
                expert_gate_out,
                expert_up_out,
                expert_gate_out,
                total_expanded * inter,
                stream,
            )?;
            // 2026-09-25: The down projection reads the activated `expert_gate_out`
            // in sorted order. First match: CUTLASS grouped (METRALE_HOLO_MOE_GROUPED_CUTLASS
            // and METRALE_HOLO_MOE_GROUPED_DOWN); over `down_ptrs_t`: E8M0, FP4
            // (METRALE_HOLO_MOE_DOWN_FP4), FP8 activations (METRALE_MOE_PREFILL_FP8_DOWN)
            // or W4A16; else the untransposed W4A16 fallback. All write
            // `expert_down_out` in sorted order.
            if ctx.levers.moe_grouped_cutlass
                && let Some(down_host) = self
                    .cutlass_grouped_host
                    .as_ref()
                    .and_then(|t| t.down.as_ref())
                && ctx.levers.moe_grouped_down
            {
                // 2026-09-25: The input is already expert-sorted, so no
                // `sorted_token_ids` gather.
                ops::moe_grouped_down_cutlass(
                    ctx.gpu,
                    cutlass_eoff.as_deref(),
                    down_host,
                    expert_gate_out,
                    expert_down_out,
                    expert_offsets,
                    h,
                    inter,
                    stream,
                )?;
            } else if let Some(dp) = &self.down_ptrs_t {
                if self.experts_scale_kind == crate::weight_map::WeightQuantFormat::Mxfp4E8m0 {
                    assert!(
                        self.moe_grouped_gemm_t_k64_e8m0.0 != 0,
                        "ARM-2: routed experts Mxfp4E8m0 but grouped_gemm_t_k64_e8m0 unresolved"
                    );
                    ops::moe_w4a16_grouped_gemm_ptrtable_n128(
                        ctx.gpu,
                        self.moe_grouped_gemm_t_k64_e8m0,
                        expert_gate_out,
                        dp.packed_ptrs,
                        dp.scale_ptrs,
                        dp.scale2_vals,
                        expert_down_out,
                        expert_offsets,
                        DevicePtr(0),
                        num_experts,
                        h,
                        inter,
                        max_m_tiles,
                        stream,
                    )?;
                } else if self.down_fp4 && self.moe_down_t_k64_fp4.0 != 0 {
                    ops::moe_w4a16_grouped_gemm_ptrtable_n128(
                        ctx.gpu,
                        self.moe_down_t_k64_fp4,
                        expert_gate_out,
                        dp.packed_ptrs,
                        dp.scale_ptrs,
                        dp.scale2_vals,
                        expert_down_out,
                        expert_offsets,
                        DevicePtr(0),
                        num_experts,
                        h,
                        inter,
                        max_m_tiles,
                        stream,
                    )?;
                } else {
                    let fp8_down = ctx.levers.moe_prefill_fp8_down
                        && self.moe_fp8_grouped_gemm_t.0 != 0
                        && self.bf16_to_fp8_k.0 != 0;
                    if fp8_down {
                        ops::bf16_to_fp8(
                            ctx.gpu,
                            self.bf16_to_fp8_k,
                            expert_gate_out,
                            expert_up_out,
                            total_expanded * inter,
                            stream,
                        )?;
                        ops::moe_fp8_grouped_gemm_ptrtable_n128(
                            ctx.gpu,
                            self.moe_fp8_grouped_gemm_t,
                            expert_up_out,
                            dp.packed_ptrs,
                            dp.scale_ptrs,
                            dp.scale2_vals,
                            expert_down_out,
                            expert_offsets,
                            DevicePtr(0),
                            num_experts,
                            h,
                            inter,
                            max_m_tiles,
                            stream,
                        )?;
                    } else {
                        ops::moe_w4a16_grouped_gemm_ptrtable_n128(
                            ctx.gpu,
                            self.moe_grouped_gemm_t_k64,
                            expert_gate_out,
                            dp.packed_ptrs,
                            dp.scale_ptrs,
                            dp.scale2_vals,
                            expert_down_out,
                            expert_offsets,
                            DevicePtr(0),
                            num_experts,
                            h,
                            inter,
                            max_m_tiles,
                            stream,
                        )?;
                    }
                }
            } else {
                self.experts_scale_kind.expect(
                    crate::weight_map::WeightQuantFormat::Nvfp4,
                    "prefill non-transposed down fallback (no E8M0 variant wired)",
                );
                self.launch_grouped_gemm(
                    ctx.gpu,
                    expert_gate_out,
                    self.down_ptrs.packed_ptrs,
                    self.down_ptrs.scale_ptrs,
                    self.down_ptrs.scale2_vals,
                    expert_down_out,
                    expert_offsets,
                    DevicePtr(0),
                    num_experts,
                    h,
                    inter,
                    max_m_tiles,
                    stream,
                )?;
            }
        }
        prof_step!("grouped_silu_down");

        Ok(())
    }
}
