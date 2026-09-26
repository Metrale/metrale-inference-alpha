// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-26: Round comparison: every replay round against the reference round.
//!
//! Owner: bench, SSM poisoning gate.
//! Invariants:
//! - A round is `Invariant` only when every turn's canonical transcript and
//!   token count equal the reference turn's.
//! - A differing turn is collapsed when its finish reason differs or its token
//!   ratio leaves [`COLLAPSE_RATIO_FLOOR`]..=[`COLLAPSE_RATIO_CEIL`], and
//!   jittered otherwise.
//! - A round's verdict is `Collapsed` if any turn collapsed, else `Unmeasured`
//!   if any turn was, else `Jittered` if any turn jittered.

use crate::benchmarks::transcript::Transcript;

/// 2026-09-26: A replay turn shorter than this fraction of the reference turn,
/// in completion tokens, is a collapse.
pub const COLLAPSE_RATIO_FLOOR: f64 = 0.5;
/// 2026-09-26: A replay turn longer than this multiple of the reference turn is a
/// collapse too.
pub const COLLAPSE_RATIO_CEIL: f64 = 2.0;

/// 2026-09-26: One differing turn's shape.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TurnDelta {
    pub turn: usize,
    pub ref_tokens: usize,
    pub replay_tokens: usize,
    pub ref_finish: Option<String>,
    pub replay_finish: Option<String>,
}

impl TurnDelta {
    /// 2026-09-26: A different finish reason, or a token ratio outside the
    /// collapse window.
    pub fn is_collapse(&self) -> bool {
        if self.ref_finish != self.replay_finish {
            return true;
        }
        if self.ref_tokens == 0 {
            // 2026-09-26: A nonzero replay against a zero-token reference is an
            // infinite ratio, so a collapse. The both-zero pair is not;
            // `compare_round` scores that turn Unmeasured before asking.
            return self.replay_tokens != 0;
        }
        let ratio = self.replay_tokens as f64 / self.ref_tokens as f64;
        !(COLLAPSE_RATIO_FLOOR..=COLLAPSE_RATIO_CEIL).contains(&ratio)
    }
}

/// 2026-09-26: The outcome of comparing one replay round to the reference round.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum RoundVerdict {
    /// 2026-09-26: Every turn equal: canonical transcript and token count.
    Invariant,
    /// 2026-09-26: At least one turn differs, and every differing turn has the
    /// same finish reason and a token ratio inside the window. Recorded; the
    /// verdict passes it.
    Jittered { turns: Vec<TurnDelta> },
    /// 2026-09-26: At least one turn collapsed (`TurnDelta::is_collapse`).
    Collapsed { turns: Vec<TurnDelta> },
    /// 2026-09-26: The round proves nothing: the replay errored (set by the
    /// driver), its turn count differs from the reference's, or a turn
    /// returned no tokens in both rounds.
    Unmeasured { reason: String },
}

/// 2026-09-26: Compare a reference turn list against a replay turn list. A replay
/// whose turn count differs from the reference's is Unmeasured, never
/// Invariant.
pub fn compare_round(reference: &[Transcript], replay: &[Transcript]) -> RoundVerdict {
    if replay.len() != reference.len() {
        return RoundVerdict::Unmeasured {
            reason: format!(
                "replay produced {} turn(s), reference has {}",
                replay.len(),
                reference.len()
            ),
        };
    }
    if reference.is_empty() {
        return RoundVerdict::Unmeasured {
            reason: "reference round has no turns".into(),
        };
    }
    let mut jittered = Vec::new();
    let mut collapsed = Vec::new();
    let mut unmeasured: Option<String> = None;
    for (i, (r, p)) in reference.iter().zip(replay).enumerate() {
        if r.completion_tokens == 0 && p.completion_tokens == 0 {
            // 2026-09-26: Two empty replies are equal and prove nothing.
            unmeasured = Some(format!("turn {} returned no tokens", i + 1));
            continue;
        }
        if r.canonical() == p.canonical() && r.completion_tokens == p.completion_tokens {
            continue;
        }
        let delta = TurnDelta {
            turn: i + 1,
            ref_tokens: r.completion_tokens,
            replay_tokens: p.completion_tokens,
            ref_finish: r.finish_reason.clone(),
            replay_finish: p.finish_reason.clone(),
        };
        if delta.is_collapse() {
            collapsed.push(delta);
        } else {
            jittered.push(delta);
        }
    }
    if !collapsed.is_empty() {
        return RoundVerdict::Collapsed { turns: collapsed };
    }
    // 2026-09-26: Unmeasured outranks Jittered, because the verdict fails an
    // unmeasured round and passes a jittered one. Collapsed, checked above,
    // is the more specific failure.
    if let Some(reason) = unmeasured {
        return RoundVerdict::Unmeasured { reason };
    }
    if !jittered.is_empty() {
        return RoundVerdict::Jittered { turns: jittered };
    }
    RoundVerdict::Invariant
}

#[cfg(test)]
#[path = "compare_tests.rs"]
mod compare_tests;
