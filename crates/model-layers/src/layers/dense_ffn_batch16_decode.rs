// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-26: The opt-in 5..=32-row native-FP8 dense-FFN tier, `w8a16_gemv_batch16`:
//! one weight pass for up to 16 rows, and two launches on row halves for 17..=32.
//!
//! Owner: model-layers (dense FFN).
//! Invariants:
//! - `batch16_plan` returns `None` unless the tier is armed and the handle resolved, and
//!   for every `m` outside 5..=32.
//! - Both launches of a `Halves` plan cover at most 16 rows, the kernel's `MAX_M`.
//!
//! The `w8_gemm!` ladder in `DenseFfnLayer::prefill_fp8` (reached from `forward_prefill_inner`) tries this tier after
//! `w8a16_gemv_batch4` (m <= 4) and the tensor-core `m16_tc` tier, and before the W8A8
//! and tile GEMMs. It selects by row count, not by phase, so a prefill chunk of 5..=32
//! tokens takes it too. `w8a16_gemv_batch16` and `w8a16_gemv_batch4` are two `MAX_M`
//! instantiations of one template (`w8a16_gemv_batch4.cu`), and
//! `examples/native_fp8_ffn_batch16_microtest.rs` requires every row to equal the scalar
//! `w8a16_gemv` byte for byte.
//!
//! Off by default. Measured 2026-09-11 on 1xH100, Qwen/Qwen3.8-27B-FP8, same binary,
//! 1024x256 at C=16, tier on against off: aggregate 121.4 against 128.0 tok/s, TPOT p50
//! 107.4 against 102.0 ms, and TTFT of a 28-token prompt 150 against 101 ms.

use anyhow::Result;
use metrale_gpu_runtime::gpu::DevicePtr;

use super::DenseFfnLayer;
use crate::layer::ForwardContext;
use crate::layers::ops;
use crate::weight_map::Fp8Weight;

/// 2026-09-25: Whether `METRALE_FFN_BATCH16` is exactly `1`, which arms the tier; any
/// other value, or none, leaves it off. Read once per process; each `DenseFfnLayer`
/// copies it into `batch16_enabled` at construction, so the route cannot change between
/// CUDA-graph replays.
pub fn ffn_batch16_enabled() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var("METRALE_FFN_BATCH16").as_deref() == Ok("1"))
}

/// 2026-09-25: How the batch16 tier serves `m` rows.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Batch16Plan {
    /// 2026-09-25: One launch covering rows `0..m` (m <= 16).
    Single,
    /// 2026-09-25: Two launches on contiguous row halves: rows `0..first`, then
    /// `first..m`. `first` is `ceil(m/2)`, so both halves are <= 16 for every m <= 32 and
    /// the first half is the wider one (m=17 -> 9 + 8).
    Halves { first: u32 },
}

/// 2026-09-25: The batch16 selection rule, pure over the row count, the handle's
/// presence and the opt-in, so the CPU tests can cover it without a `ForwardContext`.
/// `enabled` is a parameter because the process-wide `OnceLock` behind
/// `ffn_batch16_enabled` cannot be toggled per test.
pub(crate) fn batch16_plan(m: u32, batch16_loaded: bool, enabled: bool) -> Option<Batch16Plan> {
    if !enabled || !batch16_loaded {
        return None;
    }
    match m {
        5..=16 => Some(Batch16Plan::Single),
        // 2026-09-25: Both halves must be <= 16, the kernel's `MAX_M`; `div_ceil` puts
        // the odd row in the first half.
        17..=32 => Some(Batch16Plan::Halves {
            first: m.div_ceil(2),
        }),
        _ => None,
    }
}

impl DenseFfnLayer {
    /// 2026-09-25: The plan for `m` rows on this layer: its handle and the opt-in it
    /// copied at construction.
    pub(crate) fn ffn_batch16_plan(&self, m: u32) -> Option<Batch16Plan> {
        batch16_plan(m, self.w8a16_gemv_batch16_k.0 != 0, self.batch16_enabled)
    }

    /// 2026-09-25: Run one dense-FFN projection through `w8a16_gemv_batch16`.
    ///
    /// `input` is `[m, k]` BF16 and `out` is `[m, n]` BF16, both contiguous, so the
    /// `Halves` plan is two launches at byte offsets. Rows that are not contiguous need
    /// `ops::w8a16_gemv_batch16_strided`, which the attention QKV path uses.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn w8a16_batch16_proj(
        &self,
        ctx: &ForwardContext,
        plan: Batch16Plan,
        w: &Fp8Weight,
        input: DevicePtr,
        out: DevicePtr,
        m: u32,
        n: u32,
        k: u32,
        stream: u64,
    ) -> Result<()> {
        self.log_batch16_decode_route(ctx, plan);
        const BF16: usize = 2;
        let launch = |rows: u32, first: u32| {
            ops::w8a16_gemv_batch16(
                ctx.gpu,
                self.w8a16_gemv_batch16_k,
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
            Batch16Plan::Single => launch(m, 0),
            Batch16Plan::Halves { first } => {
                launch(first, 0)?;
                launch(m - first, first)
            }
        }
    }

    /// 2026-09-25: Logs the first batch16 launch once per `ModelStats`
    /// (`log:ffn_batch16_decode`), so a serve's log shows that the opt-in tier ran.
    fn log_batch16_decode_route(&self, ctx: &ForwardContext, plan: Batch16Plan) {
        if ctx.stats.once("log:ffn_batch16_decode") {
            let how = match plan {
                Batch16Plan::Single => "one launch",
                Batch16Plan::Halves { .. } => "two launches on contiguous row halves",
            };
            tracing::info!(
                "[metrale] dense FFN decode: native FP8 w8a16_gemv_batch16 ({how}) \
                 for 5..=32 rows — one weight pass, bit-identical per row to the \
                 M=1 w8a16_gemv. ARMED BY METRALE_FFN_BATCH16=1, off by default: it \
                 measured -5.4% aggregate and +50 ms TTFT on H100, and has never \
                 been measured on GB10 (#927)."
            );
        }
    }
}

#[cfg(test)]
#[path = "dense_ffn_batch16_decode_tests.rs"]
mod tests;
