// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-26: The concurrency sweep's registry entries: the three gate
//! descriptors that run the `ConcurrencySweep` driver, and the rung table
//! their floor params and metric keys are derived from.
//!
//! Owner: bench (concurrency).
//! Invariants: every `GATE_THRESHOLD_PARAMS` pair comes from `RUNGS` or
//! `PEAK_FLOOR`.

use super::{BenchmarkDescriptor, ConcurrencySweep, PluginMetadata, Sensitivity};

const SUMMARY: &str = "Latency/throughput curve across concurrency 1 → 128";
pub const METADATA: PluginMetadata = PluginMetadata::metrale(SUMMARY);

pub const DESCRIPTOR: BenchmarkDescriptor = BenchmarkDescriptor {
    id: "concurrency-sweep",
    name: "Concurrency Sweep",
    summary: SUMMARY,
    detail: "Fires N concurrent streaming requests per (input-length × concurrency) cell and \
             reports client TTFT, TPOT and end-to-end latency as p50/p90/p99, plus the batch's \
             aggregate output throughput. This is the curve the GB10 concurrency campaign is \
             measured on — C=1 is where Metrale Engine leads, C=32 is where time-to-answer starts \
             inverting in Metrale Engine's favour, and C=128 is the widest rung the published ladder \
             quotes. Requests pin temperature 0.0 / \
             seed 0 and send reasoning_effort \"none\" so the ladder measures decode, not \
             thinking. A cell that delivers under 80% of its total output budget, or whose \
             median request delivers under 80% of its own, is flagged vacuous and its tok/s \
             marked non-comparable (`concurrency_vacuity.rs`). REQUIRED gate since \
             2026-08-15: under --pull-request-gate the run serves the calibrated instrument \
             (C=1..128, isl 512, osl 320 via the variant's param_overrides) and \
             self-verdicts against gate-filled per-rung floors; a sweep with any vacuous \
             cell or request error never passes, whatever the floors say.",
    duration_hint: "~25–90 min",
    expected_secs: 1560,
    updated: "2026-08-29",
    needs_confirmation: false,
    // 2026-09-26: Any served model is a valid subject; no threshold here is
    // tied to a checkpoint.
    intended_for: None,
    // 2026-09-26: Under --pull-request-gate each floor is filled from the
    // variant's BENCH.toml `min` bound minus its noise (the server's
    // `apply_threshold_params`). The gated instrument (rungs, ISL, OSL,
    // fixture) comes from the entry's `param_overrides`, not from these schema
    // defaults.
    threshold_params: GATE_THRESHOLD_PARAMS,
    // 2026-09-26: A thermal event mid-sweep slows the later rungs only, which
    // reads as a change in the curve's shape.
    sensitivity: Sensitivity::Speed,
    ctor: || Box::new(ConcurrencySweep::default()),
};

/// 2026-09-26: The concurrency gate served with the DFlash2 drafter armed. A
/// separate gate id rather than a BENCH.toml variant: a required gate counts
/// only records of its declared default checkpoint
/// (`gate::check::record_is_required_subject`). It shares this driver and its
/// floor params with `concurrency-sweep`; its BENCH.toml entry names a
/// DFlash2 recipe and its own instrument and floors.
pub const DFLASH2_DESCRIPTOR: BenchmarkDescriptor = BenchmarkDescriptor {
    id: "concurrency-sweep-dflash2",
    name: "Concurrency Sweep (DFlash2)",
    summary: DFLASH2_SUMMARY,
    detail: "The concurrency ladder with the DFlash2 block-diffusion drafter armed \
             (`--dflash --draft-model incoai/Qwen3.8-27B-DFlash2 --dflash-gamma 8`), pinned by \
             the variant's serve_overrides. Same fixture, same rungs and same vacuity rule as \
             `concurrency-sweep`; the only difference is that the served engine speculates, \
             which is exactly the path no other required gate exercises. Expect the two curves \
             to converge at the wide rungs: DFlash2's verify batches are bounded, so above the \
             point where speculation self-limits this measures the base engine and says so \
             rather than pretending otherwise.",
    duration_hint: "~25–90 min",
    expected_secs: 300,
    updated: "2026-08-29",
    needs_confirmation: false,
    intended_for: Some(crate::benchmark::ModelExpectation {
        families: &["qwen3.8-27b"],
        note: "DFlash2 drafters are trained against one target's hidden states \
               (incoai/Qwen3.8-27B-DFlash2 consumes target layers [5,19,33,47,61] of \
               Qwen3.8-27B). Pointing this gate at another checkpoint measures a mismatched \
               drafter, which is slow rather than wrong and therefore easy to misread.",
    }),
    threshold_params: GATE_THRESHOLD_PARAMS,
    sensitivity: Sensitivity::Speed,
    ctor: || Box::new(ConcurrencySweep::default()),
};

const DFLASH2_SUMMARY: &str = "Latency/throughput curve across concurrency 1 → 128, DFlash2 armed";

/// 2026-09-26: The concurrency gate on the Qwen3.6-35B-A3B MoE. A separate gate
/// id, for the reason the DFlash2 one is: a required gate scores only its
/// declared default checkpoint per box class
/// (`gate::check::record_is_required_subject`), and `gate::bench::baseline_for`
/// refuses two defaults on one box class. Its BENCH.toml entry pins the
/// published ladder's instrument (ISL 128, OSL 1024, the `essay` fixture,
/// C=1..16), the one the repository's vLLM baseline for this checkpoint
/// (`bench/baselines/qwen36-35b-a3b/published.json`) was measured on.
pub const MOE_DESCRIPTOR: BenchmarkDescriptor = BenchmarkDescriptor {
    id: "concurrency-sweep-moe",
    name: "Concurrency Sweep (MoE)",
    summary: MOE_SUMMARY,
    detail: "The concurrency ladder on the Qwen3.6-35B-A3B MoE flagship, pinned by the \
             variant's param_overrides to the PUBLISHED instrument the vLLM one-shot for \
             this checkpoint was measured on: ISL 128 / OSL 1024, the ladder38 essay \
             request byte for byte, C=1..16. Same driver, same rungs-and-floors shape and \
             same vacuity rule as `concurrency-sweep`; the MoE decode path takes the \
             grouped-GEMM expert arm above the width gate that the dense ladder never \
             reaches, which is why a dense record cannot speak for it. Its numbers are \
             NOT comparable to the dense gates' (a different checkpoint on a different \
             instrument, ~4x apart) and each is read against its own history only.",
    duration_hint: "~5–15 min",
    expected_secs: 600,
    updated: "2026-09-20",
    needs_confirmation: false,
    intended_for: Some(crate::benchmark::ModelExpectation {
        families: &["qwen3.6-35b-a3b"],
        note: "The MoE ladder is defined on the Qwen3.6-35B-A3B family (the FP8 flagship \
               is its declared subject). Pointing it at the dense 27B measures the dense \
               FFN path under the MoE's instrument — a number with no history and no floor.",
    }),
    threshold_params: GATE_THRESHOLD_PARAMS,
    sensitivity: Sensitivity::Speed,
    ctor: || Box::new(ConcurrencySweep::default()),
};

const MOE_SUMMARY: &str =
    "Latency/throughput curve across concurrency 1 → 16 on the 35B MoE, published instrument";

/// 2026-09-26: The gated rungs, `(C, floor param, metric key, label)`. The
/// descriptors' `threshold_params`, the floor `ParamSpec`s and `configure`'s
/// `Floors::per_c` are all derived from this table. A rung with no bound in
/// the variant's BENCH.toml keeps its 0.0 default and gates nothing.
pub(super) const RUNGS: [(usize, &str, &str, &str); 8] = [
    (1, "min_c1", "c1_aggregate_tok_s", "C=1 aggregate floor"),
    (2, "min_c2", "c2_aggregate_tok_s", "C=2 aggregate floor"),
    (4, "min_c4", "c4_aggregate_tok_s", "C=4 aggregate floor"),
    (8, "min_c8", "c8_aggregate_tok_s", "C=8 aggregate floor"),
    (16, "min_c16", "c16_aggregate_tok_s", "C=16 aggregate floor"),
    (32, "min_c32", "c32_aggregate_tok_s", "C=32 aggregate floor"),
    (64, "min_c64", "c64_aggregate_tok_s", "C=64 aggregate floor"),
    (
        128,
        "min_c128",
        "c128_aggregate_tok_s",
        "C=128 aggregate floor",
    ),
];

/// 2026-09-26: Not a rung: it bounds `peak_aggregate_tok_s`, whichever C
/// produced it.
pub(super) const PEAK_FLOOR: (&str, &str, &str) =
    ("min_peak", "peak_aggregate_tok_s", "Peak aggregate floor");

/// 2026-09-26: Each floor param paired with the metric its BENCH.toml bound is
/// written on, built from [`RUNGS`] and [`PEAK_FLOOR`] so every pair has a
/// matching `ParamSpec`. The server's `apply_threshold_params` fails on a pair
/// without one once the baseline bounds that metric.
const GATE_THRESHOLD_PARAMS: &[(&str, &str)] = &gate_threshold_params();

const fn gate_threshold_params() -> [(&'static str, &'static str); RUNGS.len() + 1] {
    let mut out = [("", ""); RUNGS.len() + 1];
    let mut i = 0;
    while i < RUNGS.len() {
        out[i] = (RUNGS[i].1, RUNGS[i].2);
        i += 1;
    }
    out[RUNGS.len()] = (PEAK_FLOOR.0, PEAK_FLOOR.1);
    out
}
