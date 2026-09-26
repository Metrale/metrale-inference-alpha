// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-26: Tests for promotion-candidate debt (`coverage::promotion_debt`) and
//! for the intent half's boundary entries, `.github/pr-taxonomy.json` and `required.rs`.
//!
//! Owner: bench gate.
//! Invariants: none beyond the types.

use super::coverage;

/// 2026-09-26: A candidate that is not a registered benchmark would owe a debt that no
/// run can discharge.
#[test]
fn every_promotion_candidate_is_a_registered_benchmark() {
    let known: std::collections::BTreeSet<&str> =
        crate::registry::all().iter().map(|d| d.id).collect();
    assert_eq!(
        coverage::PROMOTION_CANDIDATES
            .iter()
            .map(|gate| gate.id)
            .collect::<Vec<_>>(),
        ["cross-contamination", "scheduler-equivalence"],
        "promotion tracking must not pass vacuously or gain an unreviewed candidate"
    );
    for gate in coverage::PROMOTION_CANDIDATES {
        assert!(
            known.contains(gate.id),
            "{} is a promotion candidate but not a registered benchmark",
            gate.id
        );
        assert!(
            !coverage::REQUIRED.iter().any(|r| r.id == gate.id),
            "{} is BOTH required and a promotion candidate — it cannot be owed \
             and excused at once",
            gate.id
        );
    }
}

#[test]
fn a_candidate_accrues_debt_exactly_where_its_coverage_says() {
    let candidate = coverage::GateCoverage {
        id: "synthetic-candidate",
        excludes: &[],
    };
    let owed = |p: &str| coverage::invalidates(&candidate, p);

    assert!(
        owed("crates/server/src/scheduler/mod.rs"),
        "engine code must accrue debt"
    );
    assert!(
        owed("kernels/gb10/common/paged_decode_attn_fp8.cu"),
        "kernel code must accrue debt"
    );
    assert!(
        !owed("docs/adr/0014-pr-intent-taxonomy-and-the-required-union.md"),
        "docs must not"
    );
    assert!(!owed("site/index.html"), "site must not");
}

#[test]
fn the_contamination_candidate_accrues_debt_for_engine_changes() {
    assert!(
        coverage::PROMOTION_CANDIDATES
            .iter()
            .any(|g| g.id == "cross-contamination"),
        "the cross-contamination candidate must be registered"
    );
    let owed = coverage::promotion_debt(["crates/server/src/scheduler/mod.rs"]);
    assert_eq!(
        owed,
        ["cross-contamination", "scheduler-equivalence"],
        "a scheduler change is exactly the kind of edit that can cross-wire \
         concurrent requests, so the contamination candidate is owed — and the \
         router-equivalence candidate with it, since a scheduler edit is what \
         can make the two routers disagree. \
         kat-equality-gate was the second entry here until 2026-09-10 and \
         concurrency-sweep-moe from 2026-09-20 to 2026-09-23; both are REQUIRED \
         now, and a required gate is owed as a gate, never as debt — \
         `every_promotion_candidate_is_a_registered_benchmark` refuses both at \
         once; got {owed:?}"
    );
    assert!(
        coverage::promotion_debt(["docs/adr/README.md", "site/index.html"]).is_empty(),
        "off-boundary paths must owe nothing"
    );
}

#[test]
fn the_candidate_is_owed_for_its_own_driver_and_not_for_other_drivers() {
    let owed = coverage::promotion_debt(["crates/bench/src/benchmarks/contamination/driver.rs"]);
    assert_eq!(owed, ["cross-contamination"]);
    assert!(
        coverage::promotion_debt(["crates/bench/src/benchmarks/ttft/descriptors.rs"]).is_empty(),
        "another benchmark's driver cannot change what this detector measures"
    );
}

#[test]
fn the_promoted_moe_gate_invalidates_where_it_used_to_accrue_debt() {
    assert!(
        coverage::REQUIRED
            .iter()
            .any(|g| g.id == "concurrency-sweep-moe"),
        "the MoE ladder must be REQUIRED after promotion"
    );
    assert!(
        !coverage::PROMOTION_CANDIDATES
            .iter()
            .any(|g| g.id == "concurrency-sweep-moe"),
        "owed and excused at once is a contradiction"
    );
    assert!(
        !coverage::NOT_REQUIRED
            .iter()
            .any(|(n, _)| *n == "concurrency-sweep-moe"),
        "the MoE ladder must not be excused any more"
    );
    for path in [
        "crates/bench/src/benchmarks/concurrency.rs",
        "crates/bench/src/benchmarks/concurrency_verdict.rs",
        "kernels/gb10/qwen3.6-35b-a3b/nvfp4/moe_grouped_gemm.cu",
        "crates/server/src/scheduler/mod.rs",
    ] {
        let hit = coverage::invalidated_by([path]);
        assert!(
            hit.contains(&"concurrency-sweep-moe"),
            "{path} re-opens the plain ladder, so it must re-open the MoE one: {hit:?}"
        );
        assert!(
            !coverage::promotion_debt([path]).contains(&"concurrency-sweep-moe"),
            "{path}: a required gate is owed as a gate, never as debt"
        );
    }
    for path in [
        "crates/bench/src/benchmarks/bfcl/mod.rs",
        "crates/bench/src/benchmarks/ttft/descriptors.rs",
        "crates/bench/src/benchmarks/agentic/mod.rs",
    ] {
        assert!(
            !coverage::invalidated_by([path]).contains(&"concurrency-sweep-moe"),
            "{path} is another benchmark's driver; it cannot change the MoE ladder"
        );
    }
    assert!(
        !coverage::invalidated_by(["kernels/gb10/qwen3.6-35b-a3b/BENCH.toml"])
            .contains(&"concurrency-sweep-moe"),
        "a BENCH.toml edit is campaign-free — declaring the instrument or a floor owes nothing"
    );
}

#[test]
fn the_promoted_gates_invalidate_where_they_used_to_accrue_debt() {
    for id in ["decode-floor", "concurrency-sweep"] {
        assert!(
            coverage::REQUIRED.iter().any(|g| g.id == id),
            "{id} must be REQUIRED after promotion"
        );
        assert!(
            !coverage::PROMOTION_CANDIDATES.iter().any(|g| g.id == id),
            "{id} must have left the candidate list — owed and excused at once is a contradiction"
        );
        assert!(
            !coverage::NOT_REQUIRED.iter().any(|(n, _)| *n == id),
            "{id} must not be excused any more"
        );
    }
    for path in [
        "crates/server/src/scheduler/mod.rs",
        "kernels/gb10/common/paged_decode_attn_fp8.cu",
        "crates/server/src/openai/encode_stream.rs",
        "crates/bench/src/benchmarks/decode_floor/mod.rs",
    ] {
        let hit = coverage::invalidated_by([path]);
        assert!(
            hit.contains(&"decode-floor") && hit.contains(&"concurrency-sweep"),
            "{path} must invalidate both promoted gates: {hit:?}"
        );
    }
    for path in [
        "crates/bench/src/benchmarks/concurrency.rs",
        "crates/bench/src/benchmarks/concurrency_verdict.rs",
        "crates/bench/src/benchmarks/concurrency_descriptors.rs",
        "crates/bench/src/benchmarks/concurrency_prompt.rs",
        "crates/bench/src/benchmarks/concurrency_cache.rs",
        "crates/bench/src/benchmarks/concurrency_cell.rs",
        "crates/bench/src/benchmarks/concurrency_report.rs",
    ] {
        assert_eq!(
            coverage::invalidated_by([path]),
            [
                "concurrency-sweep",
                "concurrency-sweep-dflash2",
                "concurrency-sweep-moe"
            ],
            "the flat concurrency driver belongs to the concurrency instruments only — all \
             three of them, since the DFlash2 and MoE gates run the same driver: {path}"
        );
    }
    let hit = coverage::invalidated_by(["crates/bench/src/benchmarks/bfcl/report.rs"]);
    assert!(
        !hit.contains(&"decode-floor") && !hit.contains(&"concurrency-sweep"),
        "the BFCL driver can change neither the decode rate nor the ladder: {hit:?}"
    );
}

#[test]
fn candidate_exclusions_meet_the_required_gate_bar() {
    let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .and_then(|p| p.parent())
        .expect("repo root is two levels above the crate")
        .to_path_buf();
    assert!(!coverage::PROMOTION_CANDIDATES.is_empty());
    for gate in coverage::PROMOTION_CANDIDATES {
        for ex in gate.excludes {
            assert!(
                ex.rationale.trim().len() > 20,
                "{} excludes {} with no real rationale",
                gate.id,
                ex.prefix
            );
            assert!(
                root.join(ex.prefix).exists(),
                "{} excludes {}, which does not exist",
                gate.id,
                ex.prefix
            );
            assert!(
                coverage::on_boundary(ex.prefix),
                "{} excludes {}, which is off the boundary — the rule does nothing",
                gate.id,
                ex.prefix
            );
        }
    }
}

/// 2026-09-26: `.github/pr-taxonomy.json` is outside `PERF_PATHS`, so it invalidates
/// only because `invalidates` checks `BOUNDARY_FILES` before `on_boundary`.
#[test]
fn the_taxonomy_and_the_union_are_on_the_boundary() {
    for path in [
        ".github/pr-taxonomy.json",
        "crates/bench/src/gate/required.rs",
    ] {
        assert_eq!(
            coverage::invalidated_by([path]),
            super::REQUIRED_GATES,
            "{path} decides what the gate requires; it must re-open EVERY gate"
        );
    }

    assert!(
        !coverage::on_boundary(".github/pr-taxonomy.json"),
        "the taxonomy is off PERF_PATHS — if that changes, the assertion above \
         starts passing for a different reason than the one documented"
    );
    // 2026-09-26: `required.rs` is under `crates` as well; its `BOUNDARY_FILES` entry
    // is what outranks the `GATE_MACHINERY` exclusion of `crates/bench/src/gate`.
    assert!(coverage::on_boundary("crates/bench/src/gate/required.rs"));
}
