// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-26: Tests for what a PR owes: the path and intent halves, their
//! union, category parsing, and reading classifications from the ledger.
//! Every changed path used here is tracked by this repository.
//!
//! Owner: bench gate (intent).
//! Invariants: none beyond the types.

use super::*;

fn real_taxonomy() -> Vec<Node> {
    let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .unwrap()
        .parent()
        .unwrap()
        .to_path_buf();
    super::super::pr_taxonomy::load(&root).expect("the shipped taxonomy loads")
}

fn cat(s: &str) -> Vec<String> {
    parse_category(s)
}

fn set(ids: &[&str]) -> BTreeSet<String> {
    ids.iter().map(|id| (*id).to_string()).collect()
}

/// 2026-09-26: Tracked paths that invalidate no gate get their gates from
/// intent alone.
#[test]
fn intent_adds_where_the_paths_are_silent() {
    let roots = real_taxonomy();
    for changed in [
        "docker/gb10/Dockerfile",
        "scripts/mlperf-edge/kl_coherence_gate.py",
        "bench/bench_isl_osl.py",
        "kernels/gb10/qwen3.6-27b/BENCH.toml",
    ] {
        let got = required_for(&[changed.to_string()], &[cat("performance/decode")], &roots);
        assert!(
            got.by_path.is_empty(),
            "{changed} was expected off the invalidation floor, got {:?}",
            got.by_path
        );
        assert_eq!(
            got.intent_only(),
            [
                "agentic-webserver",
                "bfcl-subset",
                "decode-floor",
                "ttft-warm-gate"
            ]
            .iter()
            .map(|s| s.to_string())
            .collect::<BTreeSet<String>>(),
            "{changed}: intent should supply all four, since paths supply none \
             (decode-floor joined the leaf in the 2026-08-16 fill)"
        );
    }

    // 2026-09-26: A docs diff classified `performance/scheduling` owes
    // `concurrency-sweep` through intent.
    let promoted = required_for(
        &["docs/adr/0011-ep-batched-decode-optimization.md".to_string()],
        &[cat("performance/scheduling")],
        &roots,
    );
    assert!(
        promoted.by_path.is_empty(),
        "a docs diff must stay off the floor, got {:?}",
        promoted.by_path
    );
    assert_eq!(
        promoted.intent_only(),
        set(&["agentic-webserver", "concurrency-sweep", "ttft-warm-gate"])
    );
}

/// 2026-09-26: `union()` includes the intent half. The other tests would pass
/// if it returned `by_path` alone: they assert `intent_only()` or only
/// `by_path ⊆ union()`.
#[test]
fn union_actually_includes_the_intent_half() {
    let roots = real_taxonomy();
    let got = required_for(
        &["docker/gb10/Dockerfile".to_string()],
        &[cat("performance/decode")],
        &roots,
    );
    assert!(got.by_path.is_empty());
    assert_eq!(
        got.by_intent,
        set(&[
            "agentic-webserver",
            "bfcl-subset",
            "decode-floor",
            "ttft-warm-gate",
        ])
    );
    assert_eq!(got.union(), got.by_intent);
}

/// 2026-09-26: With no classification, a path that invalidates nothing owes
/// nothing.
#[test]
fn an_unclassified_change_gets_no_invented_intent() {
    let roots = real_taxonomy();
    let got = required_for(&["docker/gb10/Dockerfile".to_string()], &[], &roots);
    assert_eq!(got, RequiredSet::default());
}

/// 2026-09-26: Scheduler code owes every gate by path, so intent adds nothing;
/// a gate-machinery file outside `BOUNDARY_FILES` owes none by path
/// (`GATE_MACHINERY`), so intent is its only source. If the first half fails,
/// `by_path` has narrowed; confirm that was intended before changing the
/// test.
#[test]
fn crates_paths_split_into_fully_covered_and_not_covered_at_all() {
    let roots = real_taxonomy();

    let engine = required_for(
        &["crates/server/src/scheduler/mod.rs".to_string()],
        &[cat("performance/scheduling")],
        &roots,
    );
    let all = super::super::coverage::REQUIRED
        .iter()
        .map(|gate| gate.id.to_string())
        .collect();
    assert_eq!(engine.by_path, all, "ordinary engine code owes every gate");
    assert!(
        engine.intent_only().is_empty(),
        "intent should be redundant here; it added {:?}",
        engine.intent_only()
    );

    let machinery = required_for(
        &["crates/bench/src/gate/telemetry.rs".to_string()],
        &[cat("performance/scheduling")],
        &roots,
    );
    assert!(machinery.by_path.is_empty());
    assert_eq!(
        machinery.intent_only(),
        set(&["agentic-webserver", "concurrency-sweep", "ttft-warm-gate"])
    );
}

/// 2026-09-26: An empty taxonomy yields no intent, so a caller must not
/// replace a failed `load` with an empty tree: that would read as "implies
/// nothing".
#[test]
fn an_empty_taxonomy_yields_no_intent_and_must_not_be_mistaken_for_an_answer() {
    let got = required_for(
        &["docker/gb10/Dockerfile".to_string()],
        &[cat("performance/decode")],
        &[],
    );
    assert!(got.by_intent.is_empty());
    assert!(got.union().is_empty());
}

/// 2026-09-26: Whatever the classification, `by_path` stays a subset of the
/// union. (`pr_taxonomy::benches_may_only_add` is the separate claim that
/// `benches_for` grows along a path.)
#[test]
fn intent_can_never_remove_a_path_derived_gate() {
    let roots = real_taxonomy();
    let changed = vec!["kernels/gb10/common/paged_decode_attn_fp8.cu".to_string()];
    let floor = required_for(&changed, &[], &roots).by_path;
    assert!(!floor.is_empty(), "a kernels/ change must owe something");

    for category in [
        "documentation/reference",
        "infrastructure/ci",
        "unknown",
        "correctness/kv-cache",
        "a-category-that-was-renamed",
    ] {
        let got = required_for(&changed, &[cat(category)], &roots);
        assert!(
            floor.is_subset(&got.union()),
            "classifying as {category} DROPPED {:?}",
            floor.difference(&got.union()).collect::<Vec<_>>()
        );
    }
}

/// 2026-09-26: Two classifications give the union of their intent halves, in
/// either order.
#[test]
fn disagreeing_classifications_union_rather_than_last_wins() {
    let roots = real_taxonomy();
    let changed = vec!["docker/gb10/Dockerfile".to_string()];

    let a = required_for(&changed, &[cat("correctness/kv-cache")], &roots);
    let b = required_for(&changed, &[cat("performance/decode")], &roots);
    let both = required_for(
        &changed,
        &[cat("correctness/kv-cache"), cat("performance/decode")],
        &roots,
    );

    assert_eq!(
        a.by_intent,
        set(&[
            "bfcl-subset",
            "ssm-state-poisoning-gate",
            "ttft-cold-gate",
            "ttft-warm-gate",
        ])
    );
    assert_eq!(
        b.by_intent,
        set(&[
            "agentic-webserver",
            "bfcl-subset",
            "decode-floor",
            "ttft-warm-gate",
        ])
    );
    assert_eq!(
        both.by_intent,
        a.by_intent.union(&b.by_intent).cloned().collect()
    );
    let reversed = required_for(
        &changed,
        &[cat("performance/decode"), cat("correctness/kv-cache")],
        &roots,
    );
    assert_eq!(both, reversed);
}

/// 2026-09-26: Empty segments are dropped and segments trimmed.
#[test]
fn empty_segments_are_dropped_not_descended_into() {
    assert_eq!(
        parse_category("performance//decode"),
        ["performance", "decode"]
    );
    assert_eq!(
        parse_category("performance/decode/"),
        ["performance", "decode"]
    );
    assert_eq!(
        parse_category(" performance / decode "),
        ["performance", "decode"]
    );
    assert!(parse_category("").is_empty());
    assert!(parse_category("///").is_empty());
}

/// 2026-09-26: An empty segment stops `benches_for` and loses the benchmarks
/// below it.
#[test]
fn a_truncated_path_would_lose_benches() {
    let roots = real_taxonomy();
    let full = super::super::pr_taxonomy::benches_for(&roots, &cat("performance/decode"));
    let truncated =
        super::super::pr_taxonomy::benches_for(&roots, &["performance".to_string(), String::new()]);
    assert_eq!(
        full,
        set(&[
            "agentic-webserver",
            "bfcl-subset",
            "decode-floor",
            "ttft-warm-gate",
        ])
    );
    assert_eq!(truncated, set(&["agentic-webserver"]));
}

fn ledger_dir() -> super::super::tests::tempdir::Dir {
    super::super::tests::tempdir::Dir::new()
}

fn write_events(root: &std::path::Path, pr: u64, rows: &[(&str, &str, &str, &str)]) {
    let path = metrale_governance::ledger::path_for(root, pr);
    std::fs::create_dir_all(path.parent().unwrap()).unwrap();
    for (i, (run, head, value, status)) in rows.iter().enumerate() {
        let e = metrale_governance::event::Event {
            pr,
            head_sha: (*head).into(),
            run_id: (*run).into(),
            attempt: 1,
            at: 1_786_280_000 + i as u64,
            kind: metrale_governance::event::EventKind::Category {
                value: (*value).into(),
                status: (*status).into(),
            },
        };
        metrale_governance::ledger::append(&path, &e).unwrap();
    }
}

#[test]
fn no_pr_is_not_requested_and_no_pr_is_not_a_missing_ledger() {
    let d = ledger_dir();
    assert_eq!(intent_source(d.path(), None), IntentSource::NotRequested);
    assert_eq!(
        intent_source(d.path(), Some(7)),
        IntentSource::NotRecorded {
            ledger: d.path().join("governance/pr-7.jsonl")
        }
    );
}

/// 2026-09-26: Rows for every `head_sha` count; `intent_source` does not
/// filter by head.
#[test]
fn every_recorded_category_counts_regardless_of_head_sha() {
    let d = ledger_dir();
    write_events(
        d.path(),
        7,
        &[
            ("100", "old-head", "performance/decode", "ok"),
            ("101", "new-head", "correctness/kv-cache", "ok"),
        ],
    );
    let IntentSource::Recorded { categories, .. } = intent_source(d.path(), Some(7)) else {
        panic!("expected Recorded");
    };
    assert_eq!(
        categories,
        [cat("performance/decode"), cat("correctness/kv-cache")]
    );
}

/// 2026-09-26: `abstain` and `error` rows are counted in `skipped` and never
/// used as intent; `partial` rows are.
#[test]
fn error_and_abstain_rows_are_counted_but_never_treated_as_intent() {
    let d = ledger_dir();
    write_events(
        d.path(),
        7,
        &[
            ("100", "head", "performance/decode", "ok"),
            ("101", "head", "unknown", "abstain"),
            ("102", "head", "unknown", "error"),
            ("103", "head", "performance", "partial"),
        ],
    );
    let IntentSource::Recorded {
        categories,
        skipped,
    } = intent_source(d.path(), Some(7))
    else {
        panic!("expected Recorded");
    };
    assert_eq!(skipped, 2, "abstain + error");
    assert_eq!(categories, [cat("performance/decode"), cat("performance")]);
}

/// 2026-09-26: A ledger holding only `error` rows reads as `NotRecorded`.
#[test]
fn a_ledger_of_only_abstentions_reads_as_not_recorded() {
    let d = ledger_dir();
    write_events(d.path(), 7, &[("100", "head", "unknown", "error")]);
    assert_eq!(
        intent_source(d.path(), Some(7)),
        IntentSource::NotRecorded {
            ledger: d.path().join("governance/pr-7.jsonl")
        }
    );
}

/// 2026-09-26: A malformed ledger line gives `Degraded`, not an error and not
/// `NotRecorded`.
#[test]
fn a_malformed_ledger_line_degrades_and_is_distinguishable_from_empty() {
    let d = ledger_dir();
    let path = metrale_governance::ledger::path_for(d.path(), 7);
    std::fs::create_dir_all(path.parent().unwrap()).unwrap();
    std::fs::write(&path, "{\"pr\":7,\"head_sha\"\n").unwrap();
    let got = intent_source(d.path(), Some(7));
    assert!(
        matches!(got, IntentSource::Degraded { .. }),
        "got {got:?} — a corrupt ledger must be Degraded, never NotRecorded"
    );
}

/// 2026-09-26: `report` keeps the source, and only a `Recorded` source
/// contributes intent.
#[test]
fn only_a_recorded_source_contributes_intent() {
    let roots = real_taxonomy();
    let changed = vec!["docker/gb10/Dockerfile".to_string()];
    for source in [
        IntentSource::NotRequested,
        IntentSource::NotRecorded { ledger: "x".into() },
        IntentSource::Degraded {
            reason: "boom".into(),
        },
    ] {
        let r = report(&changed, source.clone(), &roots);
        assert!(r.set.by_intent.is_empty(), "{source:?} contributed intent");
        assert_eq!(r.source, source, "provenance must survive");
    }
    let r = report(
        &changed,
        IntentSource::Recorded {
            categories: vec![cat("performance/decode")],
            skipped: 0,
        },
        &roots,
    );
    assert!(r.set.by_path.is_empty());
    assert_eq!(
        r.set.by_intent,
        set(&[
            "agentic-webserver",
            "bfcl-subset",
            "decode-floor",
            "ttft-warm-gate",
        ])
    );
    assert_eq!(
        r.source,
        IntentSource::Recorded {
            categories: vec![cat("performance/decode")],
            skipped: 0,
        }
    );
}
