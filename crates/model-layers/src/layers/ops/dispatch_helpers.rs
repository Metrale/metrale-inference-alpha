// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-25: Batched-prefill admission switches (codispatch, varlen) and GEMM
//! route and shape logging.
//!
//! Owner: model-layers ops.
//! Invariants:
//! - The codispatch and varlen decisions are each resolved at most once per
//!   process, by whichever comes first: the command line or the first read
//!   from the environment. Every later read returns that value.

#![allow(unused_imports)]

use super::*;

use metrale_gpu_runtime::gpu::GpuBackend;

// 2026-09-25: The codispatch and varlen switches gate batched prefill
// admission, not a GEMM path, and have no `GemmDispatch` field.
// `METRALE_Q12_BATCHED_FIRST_CHUNK` has no command-line flag.

/// 2026-09-25: The resolved codispatch decision, set once so it cannot change
/// mid-serve.
static PREFILL_CODISPATCH: std::sync::OnceLock<bool> = std::sync::OnceLock::new();

/// 2026-09-25: Publish the command line's `--prefill-codispatch` decision.
/// Returns the value in force, which differs from `enabled` when the cell was
/// already resolved; the command line then did not take effect. The serve
/// calls this only when the flag is given, so without it the
/// `METRALE_PREFILL_CODISPATCH` fallback applies.
pub fn set_prefill_codispatch_from_cli(enabled: bool) -> bool {
    let _ = PREFILL_CODISPATCH.set(enabled);
    *PREFILL_CODISPATCH.get().expect("just set")
}

/// 2026-09-25: Whether cross-request co-dispatch of fresh prompts is on:
/// `--prefill-codispatch`, else `METRALE_PREFILL_CODISPATCH` (`1` or `true`,
/// any case); off when neither is set. The scheduler's levers and
/// [`prefill_batched_first_chunk_enabled`] both read it.
pub fn prefill_codispatch_enabled() -> bool {
    *PREFILL_CODISPATCH.get_or_init(|| {
        bool_value_enabled(std::env::var("METRALE_PREFILL_CODISPATCH").ok().as_deref())
    })
}

/// 2026-09-25: Whether chunk-zero streams may use the paged batched-prefill
/// path: codispatch on, or `METRALE_Q12_BATCHED_FIRST_CHUNK` set to `1` or
/// `true`. The scheduler reads codispatch alone.
pub fn prefill_batched_first_chunk_enabled() -> bool {
    prefill_batched_first_chunk_from_parts(
        prefill_codispatch_enabled(),
        std::env::var("METRALE_Q12_BATCHED_FIRST_CHUNK")
            .ok()
            .as_deref(),
    )
}

/// 2026-09-25: The pure OR, testable without the process-wide cell or the
/// environment. `codispatch` is already resolved; `q12` is the variable's raw
/// value.
fn prefill_batched_first_chunk_from_parts(codispatch: bool, q12: Option<&str>) -> bool {
    codispatch || bool_value_enabled(q12)
}

/// 2026-09-25: The resolved varlen decision, set once so it cannot change
/// mid-serve.
static PREFILL_VARLEN: std::sync::OnceLock<bool> = std::sync::OnceLock::new();

/// 2026-09-25: Publish the command line's `--prefill-varlen-batch` decision.
/// Returns the value in force, which differs from `enabled` when the cell was
/// already resolved; the command line then did not take effect. The serve
/// calls this only when the flag is given, so without it the
/// `METRALE_PREFILL_VARLEN` fallback applies.
pub fn set_prefill_varlen_from_cli(enabled: bool) -> bool {
    let _ = PREFILL_VARLEN.set(enabled);
    *PREFILL_VARLEN.get().expect("just set")
}

/// 2026-09-25: Whether varlen (ragged) batched prefill is on:
/// `--prefill-varlen-batch`, else `METRALE_PREFILL_VARLEN` (`1` or `true`, any
/// case); off when neither is set. The admission predicate
/// (`check_kernel_batched_eligible`), the batched attention layer and the
/// scheduler's levers all read this one value.
pub fn prefill_varlen_enabled() -> bool {
    *PREFILL_VARLEN
        .get_or_init(|| bool_value_enabled(std::env::var("METRALE_PREFILL_VARLEN").ok().as_deref()))
}

fn bool_value_enabled(value: Option<&str>) -> bool {
    matches!(value, Some("1")) || value.is_some_and(|value| value.eq_ignore_ascii_case("true"))
}

pub fn log_cutlass_nvfp4_route(gpu: &dyn GpuBackend, name: &str, m: u32, n: u32, k: u32) {
    // 2026-09-25: Debug level, because the dedup key includes M and every
    // prefill length is a new M. The dedup probe is skipped when no subscriber
    // takes debug events.
    if !tracing::enabled!(tracing::Level::DEBUG) {
        return;
    }
    // 2026-09-25: Deduplicated per backend by `OpCache::first_shape`.
    if gpu.op_cache().first_shape(name, m, n, k) {
        tracing::debug!("CUTLASS_NVFP4_ROUTE {name} M={m} N={n} K={k}");
    }
}

/// 2026-09-25: With `METRALE_GEMM_SHAPE_LOG=1`, log each (kernel, M, N, K) GEMM
/// shape once per backend at warn level, with its FLOP count.
pub fn log_gemm_shape(gpu: &dyn GpuBackend, name: &str, m: u32, n: u32, k: u32) {
    if std::env::var("METRALE_GEMM_SHAPE_LOG").ok().as_deref() != Some("1") {
        return;
    }
    if gpu.op_cache().first_shape(name, m, n, k) {
        let flop = 2.0 * m as f64 * n as f64 * k as f64;
        tracing::warn!("GEMM_SHAPE {name} M={m} N={n} K={k} FLOP={flop:.3e}");
    }
}

#[cfg(test)]
mod tests {
    use super::{bool_value_enabled, prefill_batched_first_chunk_from_parts};

    #[test]
    fn accepts_boolean_environment_spellings() {
        assert!(bool_value_enabled(Some("1")));
        assert!(bool_value_enabled(Some("true")));
        assert!(bool_value_enabled(Some("TRUE")));
        assert!(!bool_value_enabled(Some("0")));
        assert!(!bool_value_enabled(Some("false")));
        assert!(!bool_value_enabled(None));
    }

    /// 2026-09-25: Either spelling enables the path, and codispatch off does
    /// not veto `METRALE_Q12_BATCHED_FIRST_CHUNK`.
    #[test]
    fn either_chunk_zero_spelling_enables_admission() {
        assert!(prefill_batched_first_chunk_from_parts(true, None));
        assert!(prefill_batched_first_chunk_from_parts(false, Some("true")));
        assert!(prefill_batched_first_chunk_from_parts(false, Some("1")));
        assert!(!prefill_batched_first_chunk_from_parts(false, None));
        assert!(!prefill_batched_first_chunk_from_parts(false, Some("0")));
        assert!(!prefill_batched_first_chunk_from_parts(
            false,
            Some("false")
        ));
        assert!(prefill_batched_first_chunk_from_parts(false, Some("1")));
    }
}
