// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-26: The registered MLPerf Agentic Inference descriptor,
//! `mlperf-agentic-subset`, the only one in the registry.
//!
//! Owner: bench, mlperf_agentic.
//! Invariants: none beyond the types.

use super::MlperfAgentic;
use crate::benchmark::BenchmarkDescriptor;
use crate::hardware::Sensitivity;
use crate::metadata::PluginMetadata;

const SUBSET_SUMMARY: &str =
    "MLPerf Agentic Inference replay, inline-scored — UNRUNNABLE: dataset unpublished";
pub const SUBSET_METADATA: PluginMetadata = PluginMetadata::metrale(SUBSET_SUMMARY);

pub const SUBSET_DESCRIPTOR: BenchmarkDescriptor = BenchmarkDescriptor {
    id: "mlperf-agentic-subset",
    name: "MLPerf agentic (subset)",
    summary: SUBSET_SUMMARY,
    detail: "Teacher-forced multi-turn replay of the MLPerf Agentic Inference dataset \
             (mlcommons/endpoints@7935df4): recorded trajectories are replayed single-stream \
             under the official immutable sampling params (temp 1.0, top_k 20, top_p 0.95, \
             presence 1.5, max 8192, preserve_thinking) plus a pinned seed, and scored \
             in-process by a fixture-verified port of the upstream inline scorer — workflow \
             intent-code match plus coding bash-executable multiset IoU. \
             ★ CANNOT RUN TODAY: the official dataset is unpublished (\"MLCommons storage, \
             link TBD\") and this leg refuses proxies, so it fails loudly at provisioning \
             until the file ships. No baseline exists; the first measured run on main \
             becomes one. Reports inline accuracy + OSL only — the SWE-bench Verified leg \
             of the official three-part gate is a separate live-agent workflow, not a leg.",
    duration_hint: "unrunnable — dataset TBD",
    expected_secs: 0,
    updated: "2026-08-13",
    needs_confirmation: false,
    intended_for: Some(crate::benchmark::ModelExpectation {
        families: &["qwen3.6-35b-a3b"],
        note: "MLCommons specifies Qwen/Qwen3.6-35B-A3B (BF16); Metrale Engine serves the official \
               FP8 sibling, which is rules-legal quantization but NOT the named checkpoint \
               — every number needs that caveat until the three-part accuracy gate is \
               cleared. Kimi K2.6 (1T) does not fit a GB10.",
    }),
    // 2026-09-26: No threshold parameter: the BENCH.toml entry has no bound to
    // fill one from.
    threshold_params: &[],
    // 2026-09-26: Correctness: the scores are intent-code match and
    // bash-executable IoU. `wall_s` and `output_tok_s` are recorded, but no
    // bound reads them.
    sensitivity: Sensitivity::Correctness,
    ctor: || Box::new(MlperfAgentic::new()),
};
