// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-26: Tests for the serve-matrix bars and the coverage rule: a planned
//! round that failed to boot or left no result fails the tally.
//!
//! Owner: bench, serve matrix.
//! Invariants: none beyond the types.

use super::*;
use crate::benchmarks::serve_matrix::host::{Absence, ServeCandidate};

fn clean() -> Signals {
    Signals {
        identity: Signal::Pass,
        coherence_pass: 2,
        coherence_total: 2,
        codegen: Signal::Pass,
        tool_call: Signal::Pass,
        long_ctx: Signal::Pass,
        tps: Some(100.0),
    }
}

fn result(label: &str, signals: Signals) -> RoundResult {
    RoundResult {
        label: label.into(),
        outcome: Outcome::Probed(Box::new(signals)),
        baseline_tps: None,
    }
}

fn plan_of(labels: &[&str]) -> Plan {
    let roster: Vec<ServeCandidate> = labels
        .iter()
        .map(|l| ServeCandidate::ready(*l, ""))
        .collect();
    Plan::build(&roster, "")
}

#[test]
fn a_clean_round_clears_every_bar() {
    assert!(result("m", clean()).bars().is_empty());
}

#[test]
fn coherence_needs_every_probe_when_there_are_only_two() {
    let mut s = clean();
    s.coherence_pass = 1;
    assert_eq!(result("m", s).bars(), vec!["coherence(1/2)"]);
}

#[test]
fn a_known_gap_tool_parser_is_not_a_failure_but_a_missing_call_is() {
    let mut na = clean();
    na.tool_call = Signal::NotApplicable("no parser for this architecture".into());
    assert!(result("m", na).bars().is_empty());

    let mut missing = clean();
    missing.tool_call = Signal::Fail("no tool call in the reply".into());
    assert_eq!(result("m", missing).bars(), vec!["tool_call"]);
}

#[test]
fn not_applicable_is_reserved_for_the_tool_call_probe() {
    let mut wrong_model = clean();
    wrong_model.identity = Signal::NotApplicable("identity probe unavailable".into());
    assert_eq!(
        result("m", wrong_model).bars(),
        vec!["wrong-model(not-applicable)"]
    );

    let mut codegen = clean();
    codegen.codegen = Signal::NotApplicable("code parser unavailable".into());
    assert_eq!(result("m", codegen).bars(), vec!["codegen(not-applicable)"]);
}

#[test]
fn long_context_is_reported_but_does_not_gate() {
    let mut s = clean();
    s.long_ctx = Signal::Fail("needle not recalled".into());
    assert!(result("m", s).bars().is_empty());
}

#[test]
fn a_dead_server_fails_the_throughput_bar_even_with_no_baseline() {
    let mut s = clean();
    s.tps = Some(0.0);
    assert_eq!(result("m", s).bars(), vec!["tps(0)"]);
}

#[test]
fn nonfinite_throughput_is_invalid_evidence() {
    for tps in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
        let mut s = clean();
        s.tps = Some(tps);
        let r = result("m", s);
        assert_eq!(r.bars(), vec!["tps(non-finite)"], "tps={tps:?}");
        assert_eq!(r.tps_note(), None, "tps={tps:?}");
    }
}

#[test]
fn without_a_baseline_throughput_is_liveness_only_and_says_so() {
    let r = result("m", clean());
    assert!(r.bars().is_empty());
    assert_eq!(r.tps_note(), Some("no baseline — liveness only"));
}

#[test]
fn with_a_baseline_a_ten_percent_drop_is_a_regression() {
    let mut r = result("m", clean());
    r.baseline_tps = Some(120.0);
    match &mut r.outcome {
        Outcome::Probed(signals) => signals.tps = Some(107.9),
        _ => panic!("fixture must be probed"),
    }
    assert_eq!(r.bars(), vec!["tps(107.9<108.0)"]);
    assert_eq!(r.tps_note(), None, "a real bar is not a 'no baseline' note");

    match &mut r.outcome {
        Outcome::Probed(signals) => signals.tps = Some(108.0),
        _ => panic!("fixture must be probed"),
    }
    assert!(r.bars().is_empty());
}

#[test]
fn an_invalid_baseline_is_not_a_regression_floor() {
    for baseline in [0.0, -1.0, f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
        let mut r = result("m", clean());
        r.baseline_tps = Some(baseline);
        assert!(r.bars().is_empty(), "baseline={baseline:?}");
        assert_eq!(
            r.tps_note(),
            Some("no baseline — liveness only"),
            "baseline={baseline:?}"
        );
    }
}

#[test]
fn coherence_pass_count_cannot_exceed_the_probe_count() {
    let mut s = clean();
    s.coherence_pass = 3;
    assert_eq!(result("m", s).bars(), vec!["coherence(3/2)"]);
}

#[test]
fn unmeasurable_throughput_is_not_a_failure() {
    // 2026-09-26: An undefined TPOT is a measurement limit, not a dead server.
    let mut s = clean();
    s.tps = None;
    assert!(result("m", s).bars().is_empty());
}

#[test]
fn a_round_served_by_the_wrong_checkpoint_fails_however_well_it_answers() {
    // 2026-09-26: Every other signal passes; the identity bar alone fails it.
    let mut s = clean();
    s.identity = Signal::Fail("serving org/previous — not org/this-round".into());
    assert_eq!(result("m", s).bars(), vec!["wrong-model"]);
}

#[test]
fn an_unprobed_signal_fails_rather_than_scoring_clean() {
    // 2026-09-26: `Signals::default()` is all `NotRun`, and must not score as
    // verified.
    assert_eq!(
        result("m", Signals::default()).bars(),
        [
            "wrong-model(not-probed)",
            "codegen(not-probed)",
            "tool_call(not-probed)",
            "coherence(0/0)",
        ]
    );
}

#[test]
fn a_checkpoint_that_failed_to_boot_is_a_failure_not_an_absence() {
    let plan = plan_of(&["a", "b"]);
    let results = vec![
        result("a", clean()),
        RoundResult {
            label: "b".into(),
            outcome: Outcome::BootFailed("CUDA out of memory".into()),
            baseline_tps: None,
        },
    ];
    let t = tally(&plan, &results);
    assert_eq!((t.verified, t.planned, t.skipped, t.excluded), (1, 2, 0, 0));
    assert!(!t.passed());
    assert_eq!(
        t.failures,
        [("b".into(), vec!["did-not-boot (CUDA out of memory)".into()])]
    );
    assert_eq!(
        verdict_text(&t, &plan),
        "1/2 planned checkpoints verified — 1 below bar: b: did-not-boot (CUDA out of memory)"
    );
}

#[test]
fn a_planned_round_that_produced_no_result_at_all_fails() {
    // 2026-09-26: Rounds b and c left no result; they stay in the denominator.
    let plan = plan_of(&["a", "b", "c"]);
    let t = tally(&plan, &[result("a", clean())]);
    assert_eq!((t.verified, t.planned, t.skipped, t.excluded), (1, 3, 0, 0));
    assert_eq!(
        t.failures,
        [
            ("b".into(), vec!["no-result".into()]),
            ("c".into(), vec!["no-result".into()]),
        ]
    );
    assert!(!t.passed());
}

#[test]
fn a_checkpoint_the_box_cannot_serve_is_skipped_not_failed() {
    let roster = vec![
        ServeCandidate::ready("a", ""),
        ServeCandidate::absent("b", "", Absence::NoWeights),
    ];
    let plan = Plan::build(&roster, "");
    let t = tally(&plan, &[result("a", clean())]);
    assert!(t.passed(), "an absent checkpoint must not fail the matrix");
    assert_eq!((t.verified, t.planned, t.skipped, t.excluded), (1, 1, 1, 0));
    assert!(t.failures.is_empty());
    assert_eq!(
        verdict_text(&t, &plan),
        "1/1 planned checkpoints verified · 1 not runnable on this box: b (weights not fully downloaded)"
    );
}

#[test]
fn an_empty_matrix_is_not_a_pass() {
    let plan = Plan::build(&[], "");
    let t = tally(&plan, &[]);
    assert!(
        !t.passed(),
        "zero planned models is nothing measured, not everything verified"
    );
}
