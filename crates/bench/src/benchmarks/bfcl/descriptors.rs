// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-26: The registered BFCL benchmark descriptors: `bfcl-subset` (the
//! golden n=995 draw) and `bfcl-subset-echolp` (the n=1004 echolp draw), both
//! gates, and `bfcl-full` (every scored sample), which is not. The two gated
//! draws mix the categories differently, so their normalized scores are not
//! interchangeable.
//!
//! Owner: bench, BFCL benchmark.
//! Invariants: none beyond the types.

use super::{Bfcl, Variant};
use crate::benchmark::BenchmarkDescriptor;
use crate::hardware::Sensitivity;
use crate::metadata::PluginMetadata;

const SUBSET_SUMMARY: &str = "The golden n=995 MLPerf-edge draw, AST-scored";
const FULL_SUMMARY: &str = "Every single-turn sample in the three scored categories";
const ECHOLP_SUMMARY: &str = "The echolp n=1004 draw, AST-scored";
pub const SUBSET_METADATA: PluginMetadata = PluginMetadata::metrale(SUBSET_SUMMARY);
pub const FULL_METADATA: PluginMetadata = PluginMetadata::metrale(FULL_SUMMARY);
pub const ECHOLP_METADATA: PluginMetadata = PluginMetadata::metrale(ECHOLP_SUMMARY);

/// 2026-09-26: The verdict params both gated draws couple to metrics. For a
/// self-started variant, `bench_resolve::apply_threshold_params` (server crate)
/// fills each from the variant's BENCH.toml `min` bound less its noise, so a
/// non-MLPerf checkpoint that clears its committed bars gets the PASS verdict
/// the gate requires (`GateRecord::verdict_passes`). The bars differ per
/// variant, so these are params rather than constants.
const GATE_THRESHOLD_PARAMS: &[(&str, &str)] = &[
    ("min_overall", "overall_accuracy"),
    ("min_normalized", "normalized_single_turn_score"),
];

pub const SUBSET_DESCRIPTOR: BenchmarkDescriptor = BenchmarkDescriptor {
    id: "bfcl-subset",
    name: "BFCL (subset)",
    summary: SUBSET_SUMMARY,
    detail: "Berkeley Function Calling Leaderboard v4, single-turn, on the golden MLPerf-edge \
             draw: categories non_live/live/hallucination at 62/10/10 with a 25-sample floor, \
             which is exactly 995 samples. Reports overall_accuracy and \
             normalized_single_turn_score against the MLPerf-edge floor (83.64 / 85.32); \
             the floor VERDICT applies only to the Qwen3.6-27B submission checkpoints — \
             every other checkpoint is judged by its own BENCH.toml thresholds, with the \
             floor kept as table styling for reference. \
             Downloads bfcl-eval into ~/.metrale/artifacts on first run. \
             ★ CERTIFIED BY FOUR SHARDS (a..d) at one commit — a whole-draw record does \
             not satisfy the gate (since 2026-09-13). ★ SCORED OPEN: cross-request SSM \
             snapshot reuse is ON, not --hermetic, so the number is partition- and \
             order-dependent — 12 of 995 samples are known to answer differently between \
             the whole draw and its shards (#936); the run warns on each and reports the \
             count as known_partition_sensitive. Floors are cut from the SHARDED aggregate.",
    duration_hint: "~1.7 h (measured)",
    expected_secs: 6000,
    updated: "2026-09-13",
    needs_confirmation: false,
    intended_for: Some(crate::benchmark::ModelExpectation {
        families: &["qwen3.6-27b", "qwen3.6-35b-a3b", "qwen3.8-27b"],
        note: "The BFCL gates are defined on Qwen3.6-27B (dense — the MLPerf-edge floor \
               83.64/85.32 rides on this checkpoint), Qwen3.6-35B-A3B (MoE, gate B), and \
               Qwen3.8-27B (dense, UNMEASURED — a run there baselines rather than gates, \
               and inherits neither 3.6's floors nor the MLPerf floor). Scores on any \
               other checkpoint have no recorded baseline to beat.",
    }),
    threshold_params: GATE_THRESHOLD_PARAMS,
    // 2026-09-26: An accuracy benchmark: recorded, never gated on box state.
    sensitivity: Sensitivity::Correctness,
    ctor: || Box::new(Bfcl::new(Variant::Subset)),
};

pub const FULL_DESCRIPTOR: BenchmarkDescriptor = BenchmarkDescriptor {
    id: "bfcl-full",
    name: "BFCL (full)",
    summary: FULL_SUMMARY,
    detail: "The same benchmark with no sampling: every single-turn sample in the three scored \
             categories (~3625). Same composition as the subset draw, so the normalized score \
             stays comparable — it just removes the sampling noise, at roughly 3.6× the wall \
             time.",
    duration_hint: "~12 h",
    expected_secs: 43200,
    updated: "2026-08-15",
    needs_confirmation: false,
    intended_for: Some(crate::benchmark::ModelExpectation {
        families: &["qwen3.6-27b", "qwen3.6-35b-a3b", "qwen3.8-27b"],
        note: "The BFCL gates are defined on Qwen3.6-27B (dense — the MLPerf-edge floor \
               83.64/85.32 rides on this checkpoint), Qwen3.6-35B-A3B (MoE, gate B), and \
               Qwen3.8-27B (dense, UNMEASURED — a run there baselines rather than gates, \
               and inherits neither 3.6's floors nor the MLPerf floor). Scores on any \
               other checkpoint have no recorded baseline to beat.",
    }),
    threshold_params: &[],
    // 2026-09-26: Accuracy, as the subsets.
    sensitivity: Sensitivity::Correctness,
    ctor: || Box::new(Bfcl::new(Variant::Full)),
};

pub const SUBSET_ECHOLP_DESCRIPTOR: BenchmarkDescriptor = BenchmarkDescriptor {
    id: "bfcl-subset-echolp",
    name: "BFCL (subset, echolp draw)",
    summary: ECHOLP_SUMMARY,
    detail: "Berkeley Function Calling Leaderboard v4, single-turn, on the echolp draw: \
             categories non_live/live/hallucination at 46/23/12 with a 25-sample floor, which is \
             exactly 1004 samples. This draw weights `live` more than twice as heavily as the \
             golden one, which moves normalized_single_turn_score by ~1.8 points while leaving \
             overall_accuracy in the same place — so its scores are NOT comparable to the golden \
             draw's, and it carries its own baseline. It exists because the 35B's only recorded \
             BFCL history is on this draw. \
             ★ CERTIFIED BY FOUR SHARDS (a..d) at one commit — a whole-draw record does \
             not satisfy the gate (since 2026-09-13). ★ SCORED OPEN: cross-request SSM \
             snapshot reuse is ON, not --hermetic, so the number is partition- and \
             order-dependent — the 12 samples known (on the golden draw) to answer differently between \
             a whole draw and its shards (#936) share its mechanism; the run warns on each and reports the \
             count as known_partition_sensitive. Floors are cut from the SHARDED aggregate.",
    duration_hint: "~2.1 h (measured)",
    expected_secs: 7500,
    updated: "2026-09-13",
    needs_confirmation: false,
    intended_for: Some(crate::benchmark::ModelExpectation {
        families: &["qwen3.6-35b-a3b"],
        note: "The echolp draw is where the 35B MoE's recorded history lives (84.66 / 83.32 \
               high-water). The dense 27B is gated on the golden n=995 draw instead — do not \
               cross the two, the category mix alone moves normalized by ~1.8 points.",
    }),
    // 2026-09-26: The same two gating metrics as the golden draw (its BENCH.toml
    // entry bounds overall_accuracy and normalized_single_turn_score), with
    // different bars.
    threshold_params: GATE_THRESHOLD_PARAMS,
    // 2026-09-26: Accuracy, as the golden draw.
    sensitivity: Sensitivity::Correctness,
    ctor: || Box::new(Bfcl::new(Variant::SubsetEcholp)),
};

// 2026-09-26: A certification runs each gated draw as `--param shard=i/n`
// slices, and `gate::check_group` aggregates their per-subset counts into one
// verdict. A shard is not a benchmark of its own: its record carries the
// group's id, with `-s<i>of<n>` in the file name (`gate::shard_suffix`).
