// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-26: Tests for the scheduler-equivalence comparison and verdict.
//!
//! Owner: bench, scheduler-equivalence gate.
//! Invariants: none beyond the types.

use super::compare::*;
use super::host::Lane;
use crate::benchmarks::transcript::Transcript;
use crate::http::{FailureKind, RequestFailure};
use crate::result::VerdictKind;

type RequestOutcome = Result<Box<Transcript>, RequestFailure>;

fn ok(text: &str, tokens: usize) -> RequestOutcome {
    Ok(Box::new(Transcript {
        text: text.to_string(),
        completion_tokens: tokens,
        finish_reason: Some("stop".into()),
        ..Default::default()
    }))
}

fn leg(lane: Lane, pass: Pass, c: usize, replies: &[(&str, RequestOutcome)]) -> Leg {
    Leg {
        lane,
        pass,
        concurrency: c,
        replies: replies
            .iter()
            .map(|(id, o)| Reply {
                sample_id: (*id).to_string(),
                outcome: o.clone(),
                elapsed: std::time::Duration::ZERO,
            })
            .collect(),
    }
}

fn same(text: &str) -> Vec<(&'static str, RequestOutcome)> {
    vec![("s1", ok(text, 3)), ("s2", ok("two", 4))]
}

/// 2026-09-26: A full, agreeing run over one lane and two concurrencies.
fn agreeing(lane: Lane) -> Vec<Leg> {
    let mut v = Vec::new();
    for c in [1usize, 4] {
        for pass in [Pass::Sync, Pass::Control, Pass::Async] {
            v.push(leg(lane, pass, c, &same("one")));
        }
    }
    v
}

#[test]
fn identical_replies_under_both_routers_pass_with_the_control_held() {
    let s = score(&agreeing(Lane::SpecOff));
    assert_eq!(s.cells.len(), 2);
    for c in &s.cells {
        assert_eq!(
            c.async_vs_sync,
            Diff {
                equal: 2,
                ..Default::default()
            }
        );
        assert_eq!(c.control_vs_sync.as_ref().unwrap().equal, 2);
    }
    let v = verdict(&s);
    assert_eq!(v.kind, VerdictKind::Pass, "{}", v.reason);
    assert!(v.reason.contains("control held"), "{}", v.reason);
    let m = metrics(&s, &Default::default());
    assert_eq!(m["diverged"], 0.0);
    assert_eq!(m["control_diverged"], 0.0);
    assert_eq!(m["cells"], 2.0);
    assert_eq!(m["diverged_spec-off_c4"], 0.0);
}

#[test]
fn a_router_difference_at_one_concurrency_fails_and_is_named_per_cell() {
    let mut legs = agreeing(Lane::SpecOff);
    let c4 = legs
        .iter_mut()
        .find(|l| l.pass == Pass::Async && l.concurrency == 4)
        .unwrap();
    c4.replies[0].outcome = ok("one but different", 3);
    let s = score(&legs);
    let v = verdict(&s);
    assert_eq!(v.kind, VerdictKind::Fail);
    assert!(v.reason.starts_with("ROUTER-DEPENDENT"), "{}", v.reason);
    assert!(
        v.reason.contains("spec-off C=4: 1 of 2 (s1)"),
        "{}",
        v.reason
    );
    assert!(!v.reason.contains("C=1:"), "{}", v.reason);
    let m = metrics(&s, &Default::default());
    assert_eq!(m["diverged_spec-off_c4"], 1.0);
    assert_eq!(m["diverged_spec-off_c1"], 0.0);
    assert_eq!(m["diverged"], 1.0);
}

#[test]
fn a_token_count_difference_alone_is_a_divergence() {
    let mut legs = agreeing(Lane::MtpForce);
    legs.iter_mut()
        .find(|l| l.pass == Pass::Async && l.concurrency == 1)
        .unwrap()
        .replies[1]
        .outcome = ok("two", 5);
    let s = score(&legs);
    assert_eq!(
        verdict_for("s2", &legs[0], &legs[2]),
        SampleVerdict::Diverged { common_prefix: 9 }
    );
    assert_eq!(verdict(&s).kind, VerdictKind::Fail);
}

#[test]
fn a_diverging_control_makes_the_async_difference_unattributable() {
    let mut legs = agreeing(Lane::SpecOff);
    // 2026-09-26: The async pass agrees with the reference; the control does not.
    legs.iter_mut()
        .find(|l| l.pass == Pass::Control && l.concurrency == 1)
        .unwrap()
        .replies[0]
        .outcome = ok("one, reworded", 3);
    let v = verdict(&score(&legs));
    assert_eq!(v.kind, VerdictKind::Fail);
    assert!(v.reason.starts_with("CONTROL DIVERGED"), "{}", v.reason);
    assert!(v.reason.contains("spec-off C=1"), "{}", v.reason);
}

#[test]
fn the_control_is_checked_before_the_async_column_is_read() {
    // 2026-09-26: Both diverge: `verdict` checks each cell's control before it
    // collects async divergences.
    let mut legs = agreeing(Lane::SpecOff);
    for l in legs.iter_mut() {
        if l.pass != Pass::Sync {
            l.replies[0].outcome = ok("drift", 3);
        }
    }
    let v = verdict(&score(&legs));
    assert!(v.reason.starts_with("CONTROL DIVERGED"), "{}", v.reason);
}

#[test]
fn a_failed_request_is_unmeasured_and_fails_the_gate() {
    let mut legs = agreeing(Lane::SpecOff);
    legs.iter_mut()
        .find(|l| l.pass == Pass::Async && l.concurrency == 4)
        .unwrap()
        .replies[1]
        .outcome = Err(RequestFailure::new(
        FailureKind::Timeout,
        "request exceeded 300s",
    ));
    let s = score(&legs);
    let v = verdict(&s);
    assert_eq!(v.kind, VerdictKind::Fail);
    assert!(v.reason.contains("UNPROVEN"), "{}", v.reason);
    assert!(
        v.reason.contains("unmeasured in the async pass"),
        "{}",
        v.reason
    );
    assert_eq!(metrics(&s, &Default::default())["unmeasured"], 1.0);
}

#[test]
fn a_missing_async_leg_is_every_sample_unmeasured_not_a_pass() {
    let legs: Vec<Leg> = agreeing(Lane::SpecOff)
        .into_iter()
        .filter(|l| l.pass != Pass::Async)
        .collect();
    let s = score(&legs);
    assert_eq!(s.cells[0].async_vs_sync.unmeasured, 2);
    assert_eq!(verdict(&s).kind, VerdictKind::Fail);
}

#[test]
fn an_all_empty_cell_is_vacuous_even_when_it_agrees() {
    // 2026-09-26: No text, no reasoning, no finish reason.
    let nothing = || -> RequestOutcome { Ok(Box::default()) };
    let empty = || vec![("s1", nothing()), ("s2", nothing())];
    let legs = vec![
        leg(Lane::SpecOff, Pass::Sync, 1, &empty()),
        leg(Lane::SpecOff, Pass::Control, 1, &empty()),
        leg(Lane::SpecOff, Pass::Async, 1, &empty()),
    ];
    let v = verdict(&score(&legs));
    assert_eq!(v.kind, VerdictKind::Fail);
    assert!(v.reason.starts_with("EQUIVALENCE VACUOUS"), "{}", v.reason);
}

#[test]
fn no_cells_or_no_reference_is_unproven() {
    assert!(verdict(&score(&[])).reason.contains("UNPROVEN"));
    let only_async = vec![leg(Lane::SpecOff, Pass::Async, 1, &same("x"))];
    let v = verdict(&score(&only_async));
    assert!(v.reason.contains("issued no samples"), "{}", v.reason);
}

#[test]
fn without_a_control_the_pass_says_so() {
    let legs: Vec<Leg> = agreeing(Lane::MtpForce)
        .into_iter()
        .filter(|l| l.pass != Pass::Control)
        .collect();
    let v = verdict(&score(&legs));
    assert_eq!(v.kind, VerdictKind::Pass);
    assert!(v.reason.contains("no control leg"), "{}", v.reason);
    assert!(v.reason.contains("mtp-force"), "{}", v.reason);
}

#[test]
fn diagnostics_ride_along_as_metrics_outside_the_verdict() {
    let s = score(&agreeing(Lane::SpecOff));
    let diag = [("spec-off_rollbacks".to_string(), 7.0)]
        .into_iter()
        .collect();
    let m = metrics(&s, &diag);
    assert_eq!(m["async_spec-off_rollbacks"], 7.0);
    assert_eq!(verdict(&s).kind, VerdictKind::Pass);
}
