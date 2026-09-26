// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-25: The fused dense-FFN gate+up decode arm: one W8A8 block-scaled GEMM at `N = 2 * inter`, not two.
//!
//! Owner: model-layers (dense FFN).
//! Invariants:
//! - The arm is taken only when `gateup_fused_selected` holds: lever on for a
//!   SiLU layer, the W8A8 arm selected for both gate and up, the fused weight
//!   installed, the strided SiLU handle loaded, `m` in 5..=`GATEUP_FUSED_MAX_M`,
//!   and an output buffer of at least `fused_out_bytes(m, inter)`.
//!
//! The loader (`qwen35_dense.rs`) builds one `[2 * inter, K]` E4M3 buffer and
//! one FP32 scale grid, and re-points `gate_proj` and `up_proj` at views inside
//! them, so every un-fused rung of `w8_gemm!` reads the same bytes. A fused
//! output row is `[gate | up]`: gate in columns `[0, inter)`, up in
//! `[inter, 2 * inter)`. `ops::silu_mul_strided` reads the two halves at a row
//! stride of `2 * inter` and writes the contiguous `[m, inter]` the down
//! projection reads.
//!
//! Output column `j` of the fused GEMM is the dot product of gate column `j`
//! (`j < inter`) or up column `j - inter`, over the same K with the same
//! scales. `examples/native_fp8_ffn_gateup_fused_microtest.rs` gates byte
//! equality of both halves against the un-fused cuBLASLt GEMMs at M = 5, 8 and
//! 16. The GEMM is `w8a8_gemm`: cuBLASLt when the dispatch's cuBLASLt scope
//! includes `ffn` and its other clauses pass, the in-tree
//! `fp8_gemm_t_blockscaled` otherwise.
//!
//! The lever is `kernels/<hw>/HARDWARE.toml` `[defaults] ffn_gateup_fused`:
//! true on hopper, false on gb10, b200 and b300. `silu_mul_strided.cu` exists
//! only in `kernels/hopper/common`, so on other targets the SiLU handle is 0
//! and the arm declines even when `METRALE_FFN_GATEUP_FUSED` turns the lever
//! on.

use anyhow::Result;
use metrale_gpu_runtime::buffers::GATEUP_FUSED_MAX_M;
use metrale_gpu_runtime::gpu::DevicePtr;

use super::DenseFfnLayer;
use crate::layer::ForwardContext;
use crate::layers::ops;
use crate::weight_map::Fp8Weight;

/// 2026-09-25: Whether the fused gate+up arm is on for this process:
/// `[defaults] ffn_gateup_fused`, overridden by `METRALE_FFN_GATEUP_FUSED`
/// (`0`, `false`, `off` or `no` turns it off; any other value turns it on).
pub fn ffn_gateup_fused() -> bool {
    ops::target_defaults::resolved().ffn_gateup_fused.value
}

/// 2026-09-25: Bytes the fused `[ceil16(m), 2 * inter]` BF16 output occupies:
/// `cublas_fp8_proj_prequant` hands cuBLASLt `ceil16(m)` rows and writes all
/// of them.
pub(crate) fn fused_out_bytes(m: u32, inter: u32) -> usize {
    ops::cublas_fp8_m_pad(m) as usize * 2 * inter as usize * 2
}

/// 2026-09-25: The fused-arm selection rule, as a pure function, so the CPU
/// tests can pin every clause without a `ForwardContext`. `lever` is injected
/// because [`ffn_gateup_fused`] reads a process-global `OnceLock`.
///
/// Clauses:
///
/// * `lever`: the target's declaration with the environment override applied.
/// * `gate_up_w8a8`: the W8A8 arm was selected for both gate and up. The fused
///   GEMM is that arm at twice the N, so a pair the ladder would have put on a
///   W8A16 rung must not be fused.
/// * `5..=GATEUP_FUSED_MAX_M`: the decode band.
/// * `fused_installed`: the loader built the `[2 * inter, K]` weight.
/// * `silu_strided_loaded`: the strided SiLU consumer's handle is loaded.
/// * `out_capacity_bytes`: the buffer holds `fused_out_bytes(m, inter)`, the
///   padded extent. A short buffer declines instead of being written past its
///   end.
#[allow(clippy::too_many_arguments)]
pub(crate) fn gateup_fused_selected(
    m: u32,
    inter: u32,
    lever: bool,
    gate_up_w8a8: bool,
    fused_installed: bool,
    silu_strided_loaded: bool,
    out_capacity_bytes: usize,
) -> bool {
    lever
        && gate_up_w8a8
        && fused_installed
        && silu_strided_loaded
        && (5..=GATEUP_FUSED_MAX_M as u32).contains(&m)
        && fused_out_bytes(m, inter) <= out_capacity_bytes
}

impl DenseFfnLayer {
    /// 2026-09-26: Whether this layer fuses gate+up for `m` rows.
    ///
    /// `gate_up_w8a8` comes from the caller (`prefill_fp8`), which
    /// already resolved the W8A8 rule for both projections; asking again here
    /// would be a second copy of that rule.
    pub(crate) fn gateup_fused_plan(
        &self,
        ctx: &ForwardContext,
        m: u32,
        inter: u32,
        gate_up_w8a8: bool,
    ) -> Option<&Fp8Weight> {
        let fused = self.fp8_gate_up_fused.as_ref();
        // 2026-09-25: SiLU only: the consumer this arm launches is the SiLU-mul,
        // so any other activation keeps the `w8_gemm!` pair and `self.act_mul`.
        let silu = self.activation == super::FfnActivation::SiLU;
        gateup_fused_selected(
            m,
            inter,
            self.gateup_fused && silu,
            gate_up_w8a8,
            fused.is_some(),
            self.silu_mul_strided_k.0 != 0,
            ctx.buffers.ffn_gate_up_fused_bytes(),
        )
        .then_some(fused)
        .flatten()
    }

    /// 2026-09-25: gate+up in one GEMM into the arena's `ffn_gate_up_fused`, then
    /// the strided SiLU-mul into the contiguous `[m, inter]` at `gate_out`.
    /// `the_fused_arm_allocates_nothing_per_call` checks that the in-tree kernel
    /// path allocates nothing.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn w8a8_gate_up_fused(
        &self,
        ctx: &ForwardContext,
        a_fp8: DevicePtr,
        a_scale: DevicePtr,
        fused_w: &Fp8Weight,
        gate_out: DevicePtr,
        m: u32,
        inter: u32,
        h: u32,
        stream: u64,
    ) -> Result<()> {
        self.log_gateup_fused_route(ctx, m);
        let fused_out = ctx.buffers.ffn_gate_up_fused();
        let cap = ctx.buffers.ffn_gate_up_fused_bytes();
        debug_assert!(fused_out_bytes(m, inter) <= cap);
        self.w8a8_gemm(
            ctx,
            a_fp8,
            a_scale,
            fused_w,
            fused_out,
            cap,
            m,
            2 * inter,
            h,
            stream,
        )?;
        // 2026-09-25: `up` starts `inter` BF16 elements into each fused row.
        const BF16: usize = 2;
        ops::silu_mul_strided(
            ctx.gpu,
            self.silu_mul_strided_k,
            fused_out,
            fused_out.offset(inter as usize * BF16),
            gate_out,
            m,
            inter,
            2 * inter,
            inter,
            stream,
        )
    }

    /// 2026-09-25: Logs the route once per model (`log:ffn_gateup_fused`).
    fn log_gateup_fused_route(&self, ctx: &ForwardContext, m: u32) {
        if ctx.stats.once("log:ffn_gateup_fused") {
            tracing::info!(
                "[metrale] dense FFN decode: gate+up FUSED into ONE W8A8 \
                 block-scaled GEMM at N=2*intermediate (m={m}, band 5..={max}) \
                 — same weight bytes, one launch instead of two. Round 13 \
                 priced the un-fused pair at 5 730.5 us/step, 59.4% of HBM, \
                 against `down`'s 71.4% for the same bytes in one launch. \
                 Bit-identical per element; METRALE_FFN_GATEUP_FUSED=0 restores \
                 the two-GEMM arm (#927).",
                max = GATEUP_FUSED_MAX_M,
            );
        }
    }
}

#[cfg(test)]
#[path = "dense_ffn_gateup_fused_tests.rs"]
mod tests;
