// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-25: The free-memory baseline: free device memory after the backend is
//! built and before weights load. It lives in
//! `metrale_telemetry::run_metrics::RunMetrics::baseline_free_bytes`, which
//! `reset_for_new_run` clears when a CUDA backend is built.
//!
//! Owner: gpu-runtime.
//! Invariants: none beyond the types.

use super::*;

/// 2026-09-25: Record the free-memory baseline; the last call wins.
pub fn set_baseline_free_bytes(bytes: usize) {
    metrale_telemetry::run_metrics::metrics()
        .baseline_free_bytes
        .store(bytes, Ordering::Relaxed);
}

/// 2026-09-25: The recorded baseline, or `None` while it is unset (stored as 0).
pub fn baseline_free_bytes() -> Option<usize> {
    match metrale_telemetry::run_metrics::metrics()
        .baseline_free_bytes
        .load(Ordering::Relaxed)
    {
        0 => None,
        v => Some(v),
    }
}
