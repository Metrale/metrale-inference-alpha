// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-25: The tensor-core `w8a16_gemm_m16` arm of the FP8 dense-FFN ladder, 5..=32 rows.
//!
//! Owner: model-layers (dense FFN).
//! Invariants:
//! - `m16_tc_plan` returns a plan only when the lever is on, the kernel handle
//!   is loaded, `k` is a multiple of 128 and `m` is in 5..=32.
//! - Every launch of a plan covers at most 16 rows: 17..=32 runs as two
//!   launches, the first `m.div_ceil(2)` rows and then the rest.
//!
//! In `dense_ffn.rs`'s `w8_gemm!` ladder this arm sits after
//! `w8a16_gemv_batch4` (m <= 4) and ahead of the `w8a16_gemv_batch16` rungs,
//! the W8A8 prefill arm and the tile GEMMs. An m16n8k16 MMA sums K in a
//! different order from the scalar `w8a16_gemv`, so its outputs are graded by
//! `within_m16_tc_budget` (2 ordinal BF16 ULP, or an absolute error under
//! `m16_tc_acc_floor`) rather than by bit equality.
//!
//! The target declares the arm in `kernels/<hw>/HARDWARE.toml` `[defaults]
//! ffn_m16_tc`; every target leaves it off, and the hopper row carries the H100
//! serve receipt. `METRALE_FFN_M16_TC`, else the `METRALE_M16_TC` umbrella,
//! overrides the declaration: `0`, `false`, `off` or `no` turns it off and any
//! other value turns it on (`ops::target_defaults::resolve`).

use anyhow::Result;
use metrale_gpu_runtime::gpu::DevicePtr;

/// 2026-09-25: The tier's numerics contract, shared by the GPU microtest
/// (`examples/native_fp8_ffn_m16_tc_microtest.rs`) and the host simulation in
/// `dense_ffn_m16_tc_m32_tests.rs`.
#[path = "dense_ffn_m16_tc_oracle.rs"]
pub mod oracle;

pub use oracle::{
    M16_TC_ACC_FLOOR_MARGIN, M16_TC_MAX_ULP, bf16_ord, m16_tc_acc_floor, within_m16_tc_budget,
};

use super::DenseFfnLayer;
use crate::layer::ForwardContext;
use crate::layers::ops;
use crate::weight_map::Fp8Weight;
use metrale_gpu_runtime::gpu::KernelHandle;

/// 2026-09-25: The resolved `w8a16_gemm_m16` levers: which projection families
/// take the tier, and the FFN arm's CTA N width. `DenseFfnLayer::new_with_activation`
/// and the attention layer's init copy them into fields at construction.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct M16TcLevers {
    /// 2026-09-25: The dense-FFN arm (`TargetLevers::ffn_m16_tc`).
    pub ffn: bool,
    /// 2026-09-25: The attention FP8 QKV and o_proj tiers
    /// (`TargetLevers::attn_m16_tc`).
    pub attn: bool,
    /// 2026-09-25: CTA N width for the FFN arm: `ops::W8A16_GEMM_M16_N_TILE` (32),
    /// or `ops::W8A16_GEMM_M16_N_TILE_WIDE` (64) when
    /// `METRALE_FFN_M16_TC_NTILE=64`. The attention tiers always launch with 32.
    pub ffn_n_tile: u32,
}

/// 2026-09-25: The pure resolver. `ffn` and `attn` are the resolved target
/// levers; `n_tile` is the raw `METRALE_FFN_M16_TC_NTILE` value. Only `"64"`
/// selects the wide tile. Any other value, or none, keeps 32 without failing
/// the boot, and the route log prints the tile that ran.
pub(crate) fn resolve_m16_tc_levers(ffn: bool, attn: bool, n_tile: Option<&str>) -> M16TcLevers {
    M16TcLevers {
        ffn,
        attn,
        ffn_n_tile: match n_tile {
            Some("64") => ops::W8A16_GEMM_M16_N_TILE_WIDE,
            _ => ops::W8A16_GEMM_M16_N_TILE,
        },
    }
}

/// 2026-09-25: The resolved levers for this process, computed once
/// (`OnceLock`).
pub fn m16_tc_levers() -> M16TcLevers {
    static ON: std::sync::OnceLock<M16TcLevers> = std::sync::OnceLock::new();
    *ON.get_or_init(|| {
        let n_tile = std::env::var("METRALE_FFN_M16_TC_NTILE").ok();
        resolve_m16_tc_levers(
            // 2026-09-25: The target's `[defaults]` rows, with `METRALE_FFN_M16_TC`,
            // `METRALE_ATTN_M16_TC` and the `METRALE_M16_TC` umbrella already
            // applied by `ops::target_defaults::resolve`. When both a narrow
            // variable and the umbrella are set, the narrow one wins.
            ops::target_defaults::resolved().ffn_m16_tc.value,
            ops::target_defaults::resolved().attn_m16_tc.value,
            n_tile.as_deref(),
        )
    })
}

/// 2026-09-25: How the tier splits `m` rows into launches; `m16_tc_plan`
/// returns `None` for a width it does not claim. Same shape as `Batch16Plan`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum M16TcPlan {
    /// 2026-09-25: One launch over rows `0..m`, `m <= 16`.
    Single,
    /// 2026-09-25: Two launches on contiguous row halves: rows `0..first`, then
    /// `first..m`. `first` is `ceil(m/2)`, so both halves are <= 16 for every
    /// m <= 32 and the first half is the wider one (m=17 -> 9 + 8).
    Halves { first: u32 },
}

/// 2026-09-25: The selection rule, as a pure function of the row count, the
/// reduction depth, the handle's presence and the lever.
///
/// A `k` that is not a multiple of 128 is declined here, not refused at
/// launch: the kernel reads one FP32 scale per 128-wide K block
/// (`kernels/hopper/common/w8a16_gemm_m16.cu`), and `ops::w8a16_gemm_m16`
/// returns an error for such a `k`.
pub(crate) fn m16_tc_plan(m: u32, k: u32, loaded: bool, enabled: bool) -> Option<M16TcPlan> {
    if !enabled || !loaded || !k.is_multiple_of(128) {
        return None;
    }
    match m {
        5..=16 => Some(M16TcPlan::Single),
        // 2026-09-25: Both halves must be <= 16, the kernel's M tile. `div_ceil`
        // puts the odd row in the first half.
        17..=32 => Some(M16TcPlan::Halves {
            first: m.div_ceil(2),
        }),
        _ => None,
    }
}

/// 2026-09-25: Which contiguous instantiation the FFN arm launches: the `n64`
/// kernel when the wide tile is asked for and its handle is loaded, otherwise
/// the 32-wide kernel. The fallback does not warn; the route log prints the
/// tile that ran.
pub(crate) fn m16_tc_kernel(
    n_tile: u32,
    narrow: KernelHandle,
    wide: KernelHandle,
) -> (ops::ContiguousM16Gemm, KernelHandle, u32) {
    if n_tile == ops::W8A16_GEMM_M16_N_TILE_WIDE && wide.0 != 0 {
        (
            ops::w8a16_gemm_m16_n64,
            wide,
            ops::W8A16_GEMM_M16_N_TILE_WIDE,
        )
    } else {
        (ops::w8a16_gemm_m16, narrow, ops::W8A16_GEMM_M16_N_TILE)
    }
}

impl DenseFfnLayer {
    /// 2026-09-25: The plan for `m` rows at reduction depth `k` on this layer:
    /// handle presence plus the lever.
    pub(crate) fn ffn_m16_tc_plan(&self, m: u32, k: u32) -> Option<M16TcPlan> {
        m16_tc_plan(m, k, self.w8a16_gemm_m16_k.0 != 0, self.m16_tc)
    }

    /// 2026-09-25: Run one dense-FFN projection through the kernel
    /// `m16_tc_kernel` picks.
    ///
    /// `input` is `[m, k]` and `out` is `[m, n]`, both contiguous BF16, so the
    /// second launch of a `Halves` plan starts `first` rows into each.
    /// (`ops::w8a16_gemm_m16_strided` takes row pitches instead; the attention
    /// QKV tier uses it.)
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn w8a16_m16_tc_proj(
        &self,
        ctx: &ForwardContext,
        plan: M16TcPlan,
        w: &Fp8Weight,
        input: DevicePtr,
        out: DevicePtr,
        m: u32,
        n: u32,
        k: u32,
        stream: u64,
    ) -> Result<()> {
        let (gemm, kernel, n_tile) = m16_tc_kernel(
            self.m16_tc_n_tile,
            self.w8a16_gemm_m16_k,
            self.w8a16_gemm_m16_n64_k,
        );
        self.log_m16_tc_route(ctx, plan, n_tile);
        const BF16: usize = 2;
        let launch = |rows: u32, first: u32| {
            gemm(
                ctx.gpu,
                kernel,
                input.offset(first as usize * k as usize * BF16),
                w.weight,
                w.row_scale,
                out.offset(first as usize * n as usize * BF16),
                rows,
                n,
                k,
                stream,
            )
        };
        match plan {
            M16TcPlan::Single => launch(m, 0),
            M16TcPlan::Halves { first } => {
                launch(first, 0)?;
                launch(m - first, first)
            }
        }
    }

    /// 2026-09-25: Logs the route once per model (`log:ffn_m16_tc_decode`), with
    /// the tile and the launch split, so a report at 5..=32 rows can tell this
    /// tier from the batch16 GEMV.
    fn log_m16_tc_route(&self, ctx: &ForwardContext, plan: M16TcPlan, n_tile: u32) {
        if ctx.stats.once("log:ffn_m16_tc_decode") {
            let how = match plan {
                M16TcPlan::Single => "one launch",
                M16TcPlan::Halves { .. } => "two launches on contiguous row halves",
            };
            let asked = self.m16_tc_n_tile;
            tracing::info!(
                "[metrale] dense FFN decode: METRALE_FFN_M16_TC — tensor-core w8a16_gemm_m16 \
                 N_TILE={n_tile} (asked {asked}) ({how}) for 5..=32 rows, ahead of \
                 w8a16_gemv_batch16. One weight pass, m16n8k16 MMA, so outputs are \
                 REASSOCIATED vs the scalar w8a16_gemv (<= 2 BF16 ULP), unlike the batch16 \
                 tier. This lever no longer reaches the attention tiers — that is \
                 METRALE_ATTN_M16_TC, and METRALE_M16_TC is both. Unset it to restore the \
                 bit-exact tier (#927)."
            );
        }
    }
}

#[cfg(test)]
#[path = "dense_ffn_m16_tc_lever_tests.rs"]
mod lever_tests;

#[cfg(test)]
#[path = "dense_ffn_m16_tc_tests.rs"]
mod tests;

/// 2026-09-25: Host simulation of the two-halves geometry at M=32, graded by
/// the oracle's comparison.
#[cfg(test)]
#[path = "dense_ffn_m16_tc_m32_tests.rs"]
mod m32_tests;
