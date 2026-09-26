// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-26: The benchmark suite, in the order the Benchmarks pane lists
//! it, and lookup by id.
//!
//! Owner: bench (registry).
//! Invariants: none beyond the types; the tests below check that ids are
//! unique and filename-safe and that every descriptor's defaults validate.

use crate::benchmark::BenchmarkDescriptor;
use crate::benchmarks::{
    agentic, bfcl, concurrency, contamination, decode_floor, kat_equality, mlperf_agentic,
    quick_speed, scheduler_equivalence, serve_matrix, ssm_poison, ttft, video, vision,
};

/// 2026-09-26: Every benchmark, in list order. A compile-time table of
/// `&'static` descriptors: registration is a compile-time decision.
const ALL: &[&BenchmarkDescriptor] = &[
    // 2026-09-26: A measurement tool with no baseline and no thresholds,
    // excused from the PR gate set in `gate::coverage::NOT_REQUIRED`.
    &quick_speed::DESCRIPTOR,
    // 2026-09-26: The gate counterpart of the probe above, judged against a
    // BENCH.toml floor and listed in `gate::coverage::REQUIRED`.
    &decode_floor::DESCRIPTOR,
    &concurrency::DESCRIPTOR,
    &concurrency::DFLASH2_DESCRIPTOR,
    // 2026-09-26: The MoE flagship's concurrency ladder, listed in
    // `gate::coverage::REQUIRED`; see `concurrency::MOE_DESCRIPTOR`.
    &concurrency::MOE_DESCRIPTOR,
    &ttft::WARM_DESCRIPTOR,
    &ttft::COLD_DESCRIPTOR,
    &contamination::DESCRIPTOR,
    // 2026-09-26: Listed in `gate::coverage::REQUIRED`. Coverage has no
    // per-model dimension: the checkpoints it measures are the BENCH.toml
    // entries that declare `gate = "vision-fidelity"`
    // (`gate::bench::baseline_for`).
    &vision::DESCRIPTOR,
    // 2026-09-26: The same, for the entries that declare
    // `gate = "video-fidelity"`.
    &video::DESCRIPTOR,
    &ssm_poison::DESCRIPTOR,
    // 2026-09-26: Beside the poisoning gate: both check whether one request's
    // state reaches another. That one asks whether a replay changes; this one
    // whether the order matters.
    &kat_equality::DESCRIPTOR,
    // 2026-09-26: The same family: does the router matter? The same draw
    // under the synchronous and the asynchronous device router. Listed in
    // `gate::coverage::PROMOTION_CANDIDATES`.
    &scheduler_equivalence::DESCRIPTOR,
    &agentic::DESCRIPTOR,
    &bfcl::SUBSET_DESCRIPTOR,
    &bfcl::SUBSET_ECHOLP_DESCRIPTOR,
    &bfcl::FULL_DESCRIPTOR,
    // 2026-09-26: The two subset gates above are also benchmark groups
    // (`gate::group::GROUPS`): a shard is the group's benchmark run with
    // `--param shard=i/n`, not a benchmark of its own. The MLPerf entry below
    // is unrunnable (its dataset is not published) and is listed so the pane
    // shows it exists.
    &mlperf_agentic::SUBSET_DESCRIPTOR,
    // 2026-09-26: Last: it replaces the model the box is serving (its
    // `detail`), so it sits furthest from an accidental start.
    &serve_matrix::DESCRIPTOR,
];

pub fn all() -> &'static [&'static BenchmarkDescriptor] {
    ALL
}

/// 2026-09-26: Look one up by its stable id.
pub fn find(id: &str) -> Option<&'static BenchmarkDescriptor> {
    all().iter().copied().find(|d| d.id == id)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ids_are_unique_and_filename_safe() {
        let mut seen = std::collections::BTreeSet::new();
        for d in all() {
            assert!(!d.id.is_empty(), "benchmark ids address history files");
            assert!(seen.insert(d.id), "duplicate benchmark id {}", d.id);
            assert!(
                d.id.chars()
                    .all(|c| c.is_ascii_lowercase() || c.is_ascii_digit() || c == '-'),
                "{} is not filename-safe",
                d.id
            );
        }
    }

    #[test]
    fn find_round_trips_every_descriptor() {
        for d in all() {
            assert_eq!(find(d.id).unwrap().name, d.name);
        }
        assert!(find("nope").is_none());
    }

    /// 2026-09-26: A zero `expected_secs` would make the certification
    /// planner (`bench_certify/plan.rs`) treat a runnable benchmark as free.
    /// Zero is permitted only for benchmarks whose hint says they cannot run.
    #[test]
    fn every_runnable_benchmark_declares_an_expected_duration() {
        for d in all() {
            if d.duration_hint.starts_with("unrunnable") {
                assert_eq!(
                    d.expected_secs, 0,
                    "{}: unrunnable but has a duration",
                    d.id
                );
                continue;
            }
            assert!(d.expected_secs > 0, "{}: expected_secs is 0", d.id);
        }
        // 2026-09-26: Pin the required gates and the groups by name, so a new
        // entry cannot slip in with the field left at zero.
        for id in crate::gate::REQUIRED_GATES {
            assert!(find(id).unwrap().expected_secs > 0, "{id}");
        }
        for g in crate::gate::group::GROUPS {
            assert!(find(g.id).unwrap().expected_secs > 0, "{}", g.id);
        }
    }

    #[test]
    fn every_benchmark_declares_defaults_that_validate() {
        for d in all() {
            let b = d.build();
            let specs = b.parameters();
            let values = crate::params::ParamValues::defaults(&specs);
            values
                .validate_against(&specs)
                .unwrap_or_else(|e| panic!("{}: {e}", d.id));
        }
    }
}
