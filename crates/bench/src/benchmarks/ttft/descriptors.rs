// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-26: The descriptors and metadata of the two TTFT gates: warm (cached
//! prefix) and cold (uncached prefill). Both compare a run with a baseline the
//! gate stores for the same host and model.
//!
//! Owner: bench, ttft.
//! Invariants: none beyond the types.

use super::{Mode, TtftGate};
use crate::benchmark::BenchmarkDescriptor;
use crate::hardware::Sensitivity;
use crate::metadata::PluginMetadata;

const WARM_SUMMARY: &str = "Cached-prefix TTFT vs the stored same-box baseline";
const COLD_SUMMARY: &str = "Uncached prefill TTFT vs the stored same-box baseline";
pub const WARM_METADATA: PluginMetadata = PluginMetadata::metrale(WARM_SUMMARY);
pub const COLD_METADATA: PluginMetadata = PluginMetadata::metrale(COLD_SUMMARY);

pub const WARM_DESCRIPTOR: BenchmarkDescriptor = BenchmarkDescriptor {
    id: "ttft-warm-gate",
    name: "Warm TTFT Regression Gate",
    summary: WARM_SUMMARY,
    detail: "Measures time-to-first-token on the WARM path: each sample repeats a bit-identical \
             prompt so the prefix cache hits. Gates at median ≤3% and p90 ≤5% against a baseline \
             recorded on this box — the guard that catches an optimization silently falling back \
             to a slow path while the correctness gates stay green.",
    duration_hint: "~3–6 min",
    expected_secs: 160,
    updated: "2026-07-31",
    needs_confirmation: false,
    // 2026-09-26: The gate compares with a baseline it stores itself, keyed by
    // model and checked for the same host, so it applies to any checkpoint.
    intended_for: None,
    threshold_params: &[],
    // 2026-09-26: Speed: the verdict compares TTFT latencies, which a thermal
    // throttle during the run moves.
    sensitivity: Sensitivity::Speed,
    ctor: || Box::new(TtftGate::new(Mode::Warm)),
};

pub const COLD_DESCRIPTOR: BenchmarkDescriptor = BenchmarkDescriptor {
    id: "ttft-cold-gate",
    name: "Cold TTFT Regression Gate",
    summary: COLD_SUMMARY,
    detail: "Measures time-to-first-token with the prefix cache guaranteed to MISS: every sample \
             carries a unique prefix_tag, so each request pays a full prefill. This is the prefill path \
             on its own, with the cache's contribution removed — the warm gate cannot see a \
             prefill regression that caching is hiding.",
    duration_hint: "~3–6 min",
    expected_secs: 120,
    updated: "2026-07-31",
    needs_confirmation: false,
    // 2026-09-26: The gate compares with a baseline it stores itself, keyed by
    // model and checked for the same host, so it applies to any checkpoint.
    intended_for: None,
    threshold_params: &[],
    // 2026-09-26: Speed, for the same reason as the warm gate.
    sensitivity: Sensitivity::Speed,
    ctor: || Box::new(TtftGate::new(Mode::Cold)),
};
