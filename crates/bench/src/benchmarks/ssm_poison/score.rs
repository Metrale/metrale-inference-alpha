// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-26: The decision logic: collected round records → a [`Score`] → the
//! replay verdict.
//!
//! Owner: bench, SSM poisoning gate.
//! Invariants:
//! - No I/O.
//! - The verdict fails a run with fewer replays than configured, or any
//!   collapsed, unmeasured or zero-cache replay; jittered replays pass.

use crate::result::Verdict;

use super::compare::{RoundVerdict, TurnDelta};

/// 2026-09-26: One replay round as the driver collected it: the comparison
/// verdict plus the cached prompt tokens the server reported for turn 1.
#[derive(Debug, Clone)]
pub struct RoundRecord {
    /// 2026-09-26: 1-based round number over the replays.
    pub round: usize,
    pub verdict: RoundVerdict,
    /// 2026-09-26: `usage.prompt_tokens_details.cached_tokens` of the round's
    /// first turn. `None` when the replay errored (the round is then
    /// Unmeasured). `Some(0)` means turn 1 restored nothing from the cache.
    pub turn1_cached: Option<usize>,
}

/// 2026-09-26: Everything the report and the gate record read.
#[derive(Debug, Clone)]
pub struct Score {
    pub rounds: usize,
    pub invariant: usize,
    pub jittered: usize,
    pub collapsed: usize,
    pub unmeasured: usize,
    /// 2026-09-26: Which rounds jittered, with their differing turns.
    pub jittered_rounds: Vec<(usize, Vec<TurnDelta>)>,
    /// 2026-09-26: Which rounds collapsed, with their differing turns.
    pub collapsed_rounds: Vec<(usize, Vec<TurnDelta>)>,
    /// 2026-09-26: Which rounds were unmeasured, with the reason, so the report
    /// attributes each to its own round.
    pub unmeasured_rounds: Vec<(usize, String)>,
    /// 2026-09-26: Rounds whose first turn reported zero cached prompt tokens.
    pub vacuous_rounds: Vec<usize>,
    /// 2026-09-26: Minimum turn-1 cached-token count across the replays that
    /// have one. `None` when none has.
    pub min_turn1_cached: Option<usize>,
}

/// 2026-09-26: Reduce the collected replay records to a [`Score`].
pub(super) fn score(replays: &[RoundRecord]) -> Score {
    let count = |f: fn(&RoundVerdict) -> bool| replays.iter().filter(|r| f(&r.verdict)).count();
    let jittered_rounds = replays
        .iter()
        .filter_map(|r| {
            if let RoundVerdict::Jittered { turns } = &r.verdict {
                Some((r.round, turns.clone()))
            } else {
                None
            }
        })
        .collect();
    let collapsed_rounds = replays
        .iter()
        .filter_map(|r| {
            if let RoundVerdict::Collapsed { turns } = &r.verdict {
                Some((r.round, turns.clone()))
            } else {
                None
            }
        })
        .collect();
    let unmeasured_rounds = replays
        .iter()
        .filter_map(|r| {
            if let RoundVerdict::Unmeasured { reason } = &r.verdict {
                Some((r.round, reason.clone()))
            } else {
                None
            }
        })
        .collect();
    let vacuous_rounds = replays
        .iter()
        .filter(|r| r.turn1_cached == Some(0))
        .map(|r| r.round)
        .collect();
    let min_turn1_cached = replays.iter().filter_map(|r| r.turn1_cached).min();
    Score {
        rounds: replays.len(),
        invariant: count(|v| matches!(v, RoundVerdict::Invariant)),
        jittered: count(|v| matches!(v, RoundVerdict::Jittered { .. })),
        collapsed: count(|v| matches!(v, RoundVerdict::Collapsed { .. })),
        unmeasured: count(|v| matches!(v, RoundVerdict::Unmeasured { .. })),
        jittered_rounds,
        collapsed_rounds,
        unmeasured_rounds,
        vacuous_rounds,
        min_turn1_cached,
    }
}

fn turn_summary(turns: &[TurnDelta]) -> String {
    turns
        .iter()
        .map(|t| {
            format!(
                "turn {} ({} -> {} tokens, finish {:?} -> {:?})",
                t.turn, t.ref_tokens, t.replay_tokens, t.ref_finish, t.replay_finish
            )
        })
        .collect::<Vec<_>>()
        .join("; ")
}

/// 2026-09-26: The replay verdict, first failing rule wins:
/// * fewer replays than the configured `rounds` fails;
/// * any collapsed round fails;
/// * any unmeasured round fails;
/// * any replay whose turn 1 reported zero cached prompt tokens fails, since
///   it never exercised the prefix restore;
/// * otherwise it passes, and names the jittered rounds' count if any.
pub(super) fn verdict(s: &Score, rounds: usize) -> Verdict {
    if s.rounds != rounds {
        return Verdict::fail(format!(
            "{} of {} replay rounds completed",
            s.rounds, rounds
        ));
    }
    if s.collapsed > 0 {
        let detail = s
            .collapsed_rounds
            .iter()
            .map(|(n, turns)| format!("round {n}: {}", turn_summary(turns)))
            .collect::<Vec<_>>()
            .join(" | ");
        return Verdict::fail(format!(
            "{} of {} replays COLLAPSED against the reference: {detail} — a restored prefix \
             produced degenerate output (early-EOS or runaway), the SSM state poisoning \
             signature",
            s.collapsed, rounds
        ));
    }
    if s.unmeasured > 0 {
        let detail = s
            .unmeasured_rounds
            .iter()
            .map(|(round, reason)| format!("round {round}: {reason}"))
            .collect::<Vec<_>>()
            .join(" | ");
        return Verdict::fail(format!(
            "{} of {} replays were unmeasurable (transport errors): {detail} — the replay \
             invariant is unproven for those rounds",
            s.unmeasured, rounds,
        ));
    }
    if !s.vacuous_rounds.is_empty() {
        return Verdict::fail(format!(
            "replay round(s) {:?} attested 0 cached prompt tokens on turn 1 — the prefix \
             restore path this gate polices was never exercised, so their transcripts prove \
             nothing about the poisoning class (is prefix caching enabled on the served \
             recipe?)",
            s.vacuous_rounds
        ));
    }
    if s.jittered > 0 {
        return Verdict::pass(format!(
            "{} of {} replays byte-identical, {} jittered within bounds (restore anchor \
             selection varies between rounds on a healthy engine), 0 collapsed",
            s.invariant, s.rounds, s.jittered
        ));
    }
    Verdict::pass(format!(
        "{} of {} replays byte-identical to the reference",
        s.invariant, rounds
    ))
}

#[cfg(test)]
#[path = "score_tests.rs"]
mod score_tests;
