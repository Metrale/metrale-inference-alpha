// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-25: The lazy down_proj scratch transpose, the layout predicates
//! (`use_t_layout_for_prefill`, `use_t_layout_for_decode`, `grouped_decode_ok`)
//! and the routed grouped-GEMM launcher.
//!
//! Owner: model-layers (MoE).
//! Invariants: none beyond the types.

use super::*;

impl MoeLayer {
    /// 2026-09-25: Wire the shared down_proj scratch and its transposed pointer
    /// tables. The factory (`m2_setup`) calls this for every layer after the
    /// gate+up-only transpose; one scratch serves all MoE layers, refilled by
    /// each layer's prefill. `scale2_vals` is the untransposed `down_ptrs` table.
    pub fn set_down_transpose_scratch(
        &mut self,
        scratch_packed: DevicePtr,
        scratch_scale: DevicePtr,
        packed_ptrs_t: DevicePtr,
        scale_ptrs_t: DevicePtr,
    ) {
        self.down_t_scratch_packed = Some(scratch_packed);
        self.down_t_scratch_scale = Some(scratch_scale);
        self.down_ptrs_t = Some(ExpertPtrTable {
            packed_ptrs: packed_ptrs_t,
            scale_ptrs: scale_ptrs_t,
            scale2_vals: self.down_ptrs.scale2_vals,
        });
    }

    /// 2026-09-25: Fill the down scratch from this layer's untransposed
    /// `down_ptrs` (`[hidden, inter/2]` packed and `[hidden, inter/16]` scales,
    /// transposed per expert). `forward_prefill` calls it before the routed
    /// GEMMs. A no-op unless the scratch is wired.
    pub(crate) fn transpose_down_into_scratch(
        &self,
        ctx: &crate::layer::ForwardContext,
        stream: u64,
    ) -> Result<()> {
        let Some(dpt) = self.down_ptrs_t.as_ref() else {
            return Ok(());
        };
        if self.down_t_scratch_packed.is_none() {
            return Ok(());
        }
        let num_experts = ctx.config.num_experts as u32;
        let h = ctx.config.hidden_size as u32;
        let inter = ctx.config.moe_intermediate_size as u32;
        crate::layers::ops::moe_transpose_u8_batched(
            ctx.gpu,
            self.moe_transpose_u8_batched_k,
            self.down_ptrs.packed_ptrs,
            dpt.packed_ptrs,
            h,
            inter / 2,
            num_experts,
            stream,
        )?;
        crate::layers::ops::moe_transpose_u8_batched(
            ctx.gpu,
            self.moe_transpose_u8_batched_k,
            self.down_ptrs.scale_ptrs,
            dpt.scale_ptrs,
            h,
            inter / 16,
            num_experts,
            stream,
        )?;
        Ok(())
    }

    /// 2026-09-25: Unused. The down scratch transpose on `prefill_stream`, after
    /// `compute_stream` reaches `event_a`, recording `event_b` when done.
    #[allow(dead_code)]
    pub(crate) fn kick_off_lazy_transpose(
        &self,
        ctx: &crate::layer::ForwardContext,
        compute_stream: u64,
    ) -> Result<()> {
        let Some(dpt) = self.down_ptrs_t.as_ref() else {
            return Ok(());
        };
        if self.down_t_scratch_packed.is_none() {
            return Ok(());
        }
        ctx.gpu.record_event(self.event_a, compute_stream)?;
        ctx.gpu
            .stream_wait_event(self.prefill_stream, self.event_a)?;

        let num_experts = ctx.config.num_experts as u32;
        let h = ctx.config.hidden_size as u32;
        let inter = ctx.config.moe_intermediate_size as u32;
        crate::layers::ops::moe_transpose_u8_batched(
            ctx.gpu,
            self.moe_transpose_u8_batched_k,
            self.down_ptrs.packed_ptrs,
            dpt.packed_ptrs,
            h,
            inter / 2,
            num_experts,
            self.prefill_stream,
        )?;
        crate::layers::ops::moe_transpose_u8_batched(
            ctx.gpu,
            self.moe_transpose_u8_batched_k,
            self.down_ptrs.scale_ptrs,
            dpt.scale_ptrs,
            h,
            inter / 16,
            num_experts,
            self.prefill_stream,
        )?;
        ctx.gpu.record_event(self.event_b, self.prefill_stream)?;
        Ok(())
    }

    /// 2026-09-25: Unused. Whether the down scratch is wired.
    #[allow(dead_code)]
    pub(crate) fn has_overlapped_transpose(&self) -> bool {
        self.down_t_scratch_packed.is_some()
    }

    /// 2026-09-25: Unused. The event `kick_off_lazy_transpose` records.
    #[allow(dead_code)]
    pub(crate) fn lazy_transpose_done_event(&self) -> u64 {
        self.event_b
    }

    /// 2026-09-25: True when `forward_batched` uses the transposed `_t` kernels:
    /// unified or hybrid layout (METRALE_UNIFIED_MOE_LAYOUT / METRALE_HYBRID_MOE_LAYOUT,
    /// read at construction), transposed tables for all three projections, and
    /// no lazy down scratch (it holds one layer's down weights and only
    /// `forward_prefill` fills it).
    #[inline]
    pub(crate) fn use_t_layout_for_prefill(&self) -> bool {
        (self.unified_layout || self.hybrid_layout)
            && self.gate_ptrs_t.is_some()
            && self.up_ptrs_t.is_some()
            && self.down_ptrs_t.is_some()
            && self.down_t_scratch_packed.is_none()
    }

    #[inline]
    /// 2026-09-25: Whether decode may take the grouped `forward_prefill` path: no
    /// BF16 or FP8 expert tables and no hash routing (`tid2eid_dev`). A mixed
    /// BF16 shared expert is allowed (`forward_prefill` runs it separately). EP
    /// is not checked here.
    pub(crate) fn grouped_decode_ok(&self) -> bool {
        self.bf16_gate_weight_ptrs.is_none()
            && self.fp8_gate_weight_ptrs.is_none()
            && self.tid2eid_dev.is_none()
    }

    /// 2026-09-25: True when decode (`forward`, `forward_k2`, `forward_k3`) uses the
    /// transposed `_t` kernels: unified layout without hybrid, transposed tables
    /// for all three projections, and no lazy down scratch.
    pub(crate) fn use_t_layout_for_decode(&self) -> bool {
        self.unified_layout
            && !self.hybrid_layout
            && self.gate_ptrs_t.is_some()
            && self.up_ptrs_t.is_some()
            && self.down_ptrs_t.is_some()
            && self.down_t_scratch_packed.is_none()
    }
}

impl super::MoeLayer {
    /// 2026-09-25: The routed grouped-GEMM kernel: the wider-K twin when it
    /// resolved (METRALE_MOE_GROUPED_K32=1 and the target ships it), else the
    /// base kernel. Both launch through `moe_w4a16_grouped_gemm_ptrtable`.
    pub(super) fn grouped_gemm_kernel(&self) -> metrale_gpu_runtime::gpu::KernelHandle {
        if self.moe_grouped_gemm_k32.0 != 0 {
            self.moe_grouped_gemm_k32
        } else {
            self.moe_grouped_gemm
        }
    }
}

impl super::MoeLayer {
    /// 2026-09-25: Launch the routed grouped GEMM. With the M_TILE=256 kernel
    /// (METRALE_MOE_GROUPED_M256=1 and resolved) the M=64 tile count
    /// `max_m_tiles` is divided by 4, rounded up; otherwise `grouped_gemm_kernel()`.
    ///
    /// Both wide kernels are opt-in. Measured 2026-08-27 on GB10 (qwen4_exp, 28K
    /// prefill, back to back): k32 +1.0% and m256 +1.4% over the arm without them,
    /// inside run-to-run noise, and m256 averaged 31.30 ms per call against 26.17 ms for the base kernel.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn launch_grouped_gemm(
        &self,
        gpu: &dyn metrale_gpu_runtime::gpu::GpuBackend,
        a: metrale_gpu_runtime::gpu::DevicePtr,
        packed_ptrs: metrale_gpu_runtime::gpu::DevicePtr,
        scale_ptrs: metrale_gpu_runtime::gpu::DevicePtr,
        scale2_vals: metrale_gpu_runtime::gpu::DevicePtr,
        c: metrale_gpu_runtime::gpu::DevicePtr,
        expert_offsets: metrale_gpu_runtime::gpu::DevicePtr,
        sorted_token_ids: metrale_gpu_runtime::gpu::DevicePtr,
        num_experts: u32,
        n_out: u32,
        k: u32,
        max_m_tiles: u32,
        stream: u64,
    ) -> anyhow::Result<()> {
        if self.moe_grouped_gemm_m256.0 != 0 {
            return crate::layers::ops::moe_w4a16_grouped_gemm_ptrtable_m256(
                gpu,
                self.moe_grouped_gemm_m256,
                a,
                packed_ptrs,
                scale_ptrs,
                scale2_vals,
                c,
                expert_offsets,
                sorted_token_ids,
                num_experts,
                n_out,
                k,
                max_m_tiles.div_ceil(4).max(1),
                stream,
            );
        }
        crate::layers::ops::moe_w4a16_grouped_gemm_ptrtable(
            gpu,
            self.grouped_gemm_kernel(),
            a,
            packed_ptrs,
            scale_ptrs,
            scale2_vals,
            c,
            expert_offsets,
            sorted_token_ids,
            num_experts,
            n_out,
            k,
            max_m_tiles,
            stream,
        )
    }
}
