// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-26: Tests for the trajectory diagnostics `metrics()` records, and
//! that they change no verdict. The tier fixtures come from `agentic_tests.rs`.
//!
//! Owner: bench, agentic.
//! Invariants: none beyond the types.

use super::tests::{tier, with_budgets, with_rows};
use super::*;

/// 2026-09-26: The record says how each iteration's loop ended.
#[test]
fn the_record_says_how_each_trajectory_ended() {
    let mut capped = tier(300.0, 12);
    capped.hit_turn_cap = true;
    capped.truncated_turns = 1;
    let mut degenerate = tier(400.0, 17);
    degenerate.unparsed_call_turns = 3;
    degenerate.truncated_turns = 2;
    let m = with_rows(vec![capped, degenerate, tier(200.0, 9)], 1300.0).metrics();
    assert_eq!(m["sum_hit_turn_cap"], 1.0);
    assert_eq!(m["sum_truncated_turns"], 3.0);
    assert_eq!(m["sum_unparsed_call_turns"], 3.0);
    assert_eq!(m["max_iter_turns"], 17.0);
    // 2026-09-26: A clean tier records zeros, and an empty tier has no maximum.
    let clean = with_rows(vec![tier(100.0, 5)], 1300.0).metrics();
    assert_eq!(clean["sum_hit_turn_cap"], 0.0);
    assert_eq!(clean["max_iter_turns"], 5.0);
    let empty = with_rows(vec![], 1300.0).metrics();
    assert_eq!(empty["sum_hit_turn_cap"], 0.0);
    assert!(!empty.contains_key("max_iter_turns"));
}

/// 2026-09-26: The diagnostic keys `trajectory_diagnostics` writes.
const DIAGNOSTIC_KEYS: [&str; 4] = [
    "sum_hit_turn_cap",
    "sum_truncated_turns",
    "sum_unparsed_call_turns",
    "max_iter_turns",
];

fn agentic_gate_record(rows: Vec<IterationRow>) -> crate::gate::GateRecord {
    let bench = with_budgets(rows, 1800.0, 8.5);
    let frame = crate::result::BenchmarkResult::completed("done", std::time::Duration::ZERO)
        .with_metrics(bench.metrics())
        .with_verdict(bench.verdict());
    let run = crate::history::RunRecord {
        schema: 1,
        run_id: "run-1".to_string(),
        benchmark_id: DESCRIPTOR.id.to_string(),
        benchmark_name: DESCRIPTOR.name.to_string(),
        recorded_at: 1_785_891_382,
        serve_overrides: Default::default(),
        target_url: String::new(),
        target_model: "Qwen/Qwen3.6-35B-A3B-FP8".to_string(),
        params: Default::default(),
        source: crate::RunSource::Cli,
        metrale_version: "test".to_string(),
        frame,
    };
    let hw = crate::hardware::Hardware {
        gpu: "NVIDIA GB10".to_string(),
        ..Default::default()
    };
    crate::gate::GateRecord::from_run(&run, hw, "b72dad1893".into(), Vec::new(), None).unwrap()
}

/// 2026-09-26: The diagnostic keys change no verdict: the tier verdict ignores
/// them, `check_record` against the committed 35B baseline scores a record the
/// same with and without them, and no committed BENCH.toml entry bounds them.
#[test]
fn trajectory_diagnostics_are_recorded_and_never_gated() {
    // 2026-09-26: The tier verdict reads none of the counters. Closures,
    // because `IterationRow` is not `Clone`. The clean tier must pass the
    // committed 35B bounds (sum_wall_s <= 700, s_per_turn <= 8.5) or the
    // `check_record` comparison below proves nothing: 10 x 57.3 s = 573 s,
    // 57.3 / 9 = 6.37 s per turn.
    let clean = || {
        (0..10)
            .map(|_| tier(57.3, 9))
            .collect::<Vec<IterationRow>>()
    };
    let noisy = || {
        (0..10)
            .map(|i| {
                let mut r = tier(57.3, 9);
                r.hit_turn_cap = i % 2 == 0;
                r.truncated_turns = i;
                r.unparsed_call_turns = 10 - i;
                r
            })
            .collect::<Vec<IterationRow>>()
    };
    let (a, b) = (
        with_budgets(clean(), 1800.0, 8.5).verdict(),
        with_budgets(noisy(), 1800.0, 8.5).verdict(),
    );
    assert_eq!((a.kind, &a.reason), (b.kind, &b.reason));
    assert_eq!(a.kind, crate::result::VerdictKind::Pass, "{}", a.reason);

    // 2026-09-26: The committed baseline scores the record the same with and
    // without the keys.
    let root = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .and_then(std::path::Path::parent)
        .expect("workspace layout")
        .to_path_buf();
    let baseline = crate::gate::bench::baseline_for(&root, DESCRIPTOR.id)
        .expect("the committed agentic-webserver baseline must load");
    for rows in [clean(), noisy()] {
        let mut with = agentic_gate_record(rows);
        // 2026-09-26: The record carries the baseline's own serve overrides, so
        // `check_record` judges the metrics, not the pins.
        with.serve_overrides = baseline
            .resolve(&with.hardware.gate_key(), Some(&with.target_model))
            .expect("the committed 35B agentic entry resolves")
            .1
            .serve_overrides
            .clone();
        for key in DIAGNOSTIC_KEYS {
            assert!(with.metrics.contains_key(key), "{key} must be recorded");
        }
        let mut without = with.clone();
        for key in DIAGNOSTIC_KEYS {
            without.metrics.remove(key);
        }
        // 2026-09-26: Compared before the mutation below.
        assert_eq!(
            crate::gate::check_record(&with, &baseline),
            None,
            "a clean 10/10 tier must pass the committed bounds for this comparison to bite"
        );
        assert_eq!(
            crate::gate::check_record(&with, &baseline),
            crate::gate::check_record(&without, &baseline)
        );
        // 2026-09-26: And on a failing record, so equality is not only "both
        // pass".
        with.metrics.insert("followed_directions".into(), 9.0);
        without.metrics.insert("followed_directions".into(), 9.0);
        let failing = crate::gate::check_record(&with, &baseline);
        assert!(failing.is_some());
        assert_eq!(failing, crate::gate::check_record(&without, &baseline));
    }

    // 2026-09-26: No committed bound names a diagnostic key.
    let mut checked = 0;
    for (target, entry) in crate::gate::bench::load_all(&root).expect("BENCH.toml files load") {
        if entry.gate != DESCRIPTOR.id {
            continue;
        }
        checked += 1;
        for key in DIAGNOSTIC_KEYS {
            assert!(
                !entry.metrics.as_ref().is_some_and(|m| m.contains_key(key)),
                "{}/{} bounds `{key}`: a trajectory diagnostic has become a gate, which is \
                 a benchmark-definition change and needs its own stack and a re-measured bar",
                target.hardware,
                target.model
            );
        }
    }
    assert!(
        checked >= 2,
        "expected the 35B and dense agentic entries, saw {checked}"
    );
}
