// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-26: Tests for the replay score and verdict.
//!
//! Owner: bench, SSM poisoning gate.
//! Invariants: none beyond the types.

use super::super::compare::{RoundVerdict, TurnDelta};
use super::{RoundRecord, score, verdict};
use crate::result::VerdictKind;

/// 2026-09-26: A nonzero turn-1 cache count; the verdict only asks for nonzero.
const WARM: Option<usize> = Some(992);

fn record(round: usize, verdict: RoundVerdict, turn1_cached: Option<usize>) -> RoundRecord {
    RoundRecord {
        round,
        verdict,
        turn1_cached,
    }
}

fn inv(n: usize) -> RoundRecord {
    record(n, RoundVerdict::Invariant, WARM)
}
fn jit(n: usize) -> RoundRecord {
    record(
        n,
        RoundVerdict::Jittered {
            turns: vec![TurnDelta {
                turn: 2,
                ref_tokens: 200,
                replay_tokens: 206,
                ref_finish: Some("stop".into()),
                replay_finish: Some("stop".into()),
            }],
        },
        WARM,
    )
}
fn col(n: usize) -> RoundRecord {
    record(
        n,
        RoundVerdict::Collapsed {
            turns: vec![TurnDelta {
                turn: 2,
                ref_tokens: 200,
                replay_tokens: 3,
                ref_finish: Some("stop".into()),
                replay_finish: Some("stop".into()),
            }],
        },
        WARM,
    )
}
fn unm(n: usize) -> RoundRecord {
    record(
        n,
        RoundVerdict::Unmeasured {
            reason: "reset".into(),
        },
        None,
    )
}

#[test]
fn all_invariant_is_pass() {
    let replays: Vec<_> = (1..=12).map(inv).collect();
    let v = verdict(&score(&replays), 12);
    assert_eq!(v.kind, VerdictKind::Pass);
    assert_eq!(v.reason, "12 of 12 replays byte-identical to the reference");
}

#[test]
fn jitter_is_recorded_but_passes() {
    // 2026-09-26: Twelve jittered replays pass.
    let replays: Vec<_> = (1..=12).map(jit).collect();
    let s = score(&replays);
    assert_eq!(s.rounds, 12);
    assert_eq!(s.invariant, 0);
    assert_eq!(s.jittered, 12);
    assert_eq!(s.collapsed, 0);
    assert_eq!(s.unmeasured, 0);
    assert_eq!(s.jittered_rounds.len(), 12);
    assert!(s.collapsed_rounds.is_empty());
    assert!(s.unmeasured_rounds.is_empty());
    assert!(s.vacuous_rounds.is_empty());
    assert_eq!(s.min_turn1_cached, WARM);
    let v = verdict(&s, 12);
    assert_eq!(v.kind, VerdictKind::Pass);
    assert_eq!(
        v.reason,
        "0 of 12 replays byte-identical, 12 jittered within bounds (restore anchor selection varies between rounds on a healthy engine), 0 collapsed"
    );
}

#[test]
fn a_single_collapse_is_fail_and_names_the_round() {
    // 2026-09-26: One collapsed round among passing ones.
    let mut replays: Vec<_> = (1..=7).map(inv).collect();
    replays.push(col(8));
    replays.push(inv(9));
    replays.push(jit(10));
    let s = score(&replays);
    assert_eq!(
        (s.rounds, s.invariant, s.jittered, s.collapsed, s.unmeasured),
        (10, 8, 1, 1, 0)
    );
    assert_eq!(
        s.collapsed_rounds,
        [(
            8,
            vec![TurnDelta {
                turn: 2,
                ref_tokens: 200,
                replay_tokens: 3,
                ref_finish: Some("stop".into()),
                replay_finish: Some("stop".into()),
            }],
        )]
    );
    let v = verdict(&s, 10);
    assert_eq!(v.kind, VerdictKind::Fail);
    assert_eq!(
        v.reason,
        "1 of 10 replays COLLAPSED against the reference: round 8: turn 2 (200 -> 3 tokens, finish Some(\"stop\") -> Some(\"stop\")) — a restored prefix produced degenerate output (early-EOS or runaway), the SSM state poisoning signature"
    );
}

#[test]
fn an_unmeasured_round_fails_the_gate() {
    let replays = vec![
        inv(1),
        unm(2),
        inv(3),
        record(
            4,
            RoundVerdict::Unmeasured {
                reason: "timeout".into(),
            },
            None,
        ),
    ];
    let v = verdict(&score(&replays), 4);
    assert_eq!(v.kind, VerdictKind::Fail);
    assert_eq!(
        v.reason,
        "2 of 4 replays were unmeasurable (transport errors): round 2: reset | round 4: timeout — the replay invariant is unproven for those rounds"
    );
}

#[test]
fn a_short_run_cannot_pass_by_running_fewer_replays() {
    let replays = vec![inv(1), inv(2)];
    let v = verdict(&score(&replays), 12);
    assert_eq!(v.kind, VerdictKind::Fail);
    assert_eq!(v.reason, "2 of 12 replay rounds completed");
}

#[test]
fn collapse_wins_over_jitter_in_the_reason() {
    // 2026-09-26: A collapse fails the run even beside tolerated jitter.
    let replays = vec![jit(1), col(2)];
    let v = verdict(&score(&replays), 2);
    assert_eq!(v.kind, VerdictKind::Fail);
    assert_eq!(
        v.reason,
        "1 of 2 replays COLLAPSED against the reference: round 2: turn 2 (200 -> 3 tokens, finish Some(\"stop\") -> Some(\"stop\")) — a restored prefix produced degenerate output (early-EOS or runaway), the SSM state poisoning signature"
    );
}

#[test]
fn a_zero_cache_replay_fails_even_when_every_transcript_matched() {
    // 2026-09-26: Every transcript matches, but round 3 restored nothing.
    let mut replays: Vec<_> = (1..=2).map(inv).collect();
    replays.push(record(3, RoundVerdict::Invariant, Some(0)));
    let s = score(&replays);
    assert_eq!(s.vacuous_rounds, vec![3]);
    assert_eq!(s.min_turn1_cached, Some(0));
    let v = verdict(&s, 3);
    assert_eq!(v.kind, VerdictKind::Fail);
    assert_eq!(
        v.reason,
        "replay round(s) [3] attested 0 cached prompt tokens on turn 1 — the prefix restore path this gate polices was never exercised, so their transcripts prove nothing about the poisoning class (is prefix caching enabled on the served recipe?)"
    );
}

#[test]
fn an_all_cold_run_fails_on_every_round() {
    // 2026-09-26: Every round reports zero cached tokens.
    let replays: Vec<_> = (1..=12)
        .map(|n| record(n, RoundVerdict::Invariant, Some(0)))
        .collect();
    let s = score(&replays);
    assert_eq!(s.vacuous_rounds, (1..=12).collect::<Vec<_>>());
    assert_eq!(s.min_turn1_cached, Some(0));
    let v = verdict(&s, 12);
    assert_eq!(v.kind, VerdictKind::Fail);
    assert_eq!(
        v.reason,
        "replay round(s) [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12] attested 0 cached prompt tokens on turn 1 — the prefix restore path this gate polices was never exercised, so their transcripts prove nothing about the poisoning class (is prefix caching enabled on the served recipe?)"
    );
}

#[test]
fn collapse_outranks_vacuity_in_the_reason() {
    // 2026-09-26: A collapsed, zero-cache round fails on the collapse, which
    // `verdict` checks first.
    let replays = vec![record(
        1,
        RoundVerdict::Collapsed {
            turns: vec![TurnDelta {
                turn: 1,
                ref_tokens: 200,
                replay_tokens: 3,
                ref_finish: Some("stop".into()),
                replay_finish: Some("stop".into()),
            }],
        },
        Some(0),
    )];
    let v = verdict(&score(&replays), 1);
    assert_eq!(v.kind, VerdictKind::Fail);
    assert_eq!(
        v.reason,
        "1 of 1 replays COLLAPSED against the reference: round 1: turn 1 (200 -> 3 tokens, finish Some(\"stop\") -> Some(\"stop\")) — a restored prefix produced degenerate output (early-EOS or runaway), the SSM state poisoning signature"
    );
}

#[test]
fn unmeasured_rounds_carry_their_number_and_reason() {
    // 2026-09-26: The report reads each unmeasured round's number from here.
    let replays = vec![inv(1), unm(2), inv(3)];
    let s = score(&replays);
    assert_eq!(s.unmeasured_rounds, [(2, "reset".into())]);
    // 2026-09-26: `None` is not a vacuous round: the round is already
    // unmeasured, and the server reported no figure.
    assert!(s.vacuous_rounds.is_empty());
    assert_eq!(s.min_turn1_cached, Some(992));
}
