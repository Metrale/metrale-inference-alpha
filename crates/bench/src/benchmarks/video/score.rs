// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-26: The video benchmark's cells and verdict. A run that asserted
//! nothing is INCONCLUSIVE; a run whose no-video control named the whole
//! color sequence is VACUOUS; otherwise it passes only if every asserted cell
//! passed.
//!
//! Owner: bench, video.
//! Invariants: none beyond the types.

use std::fmt;

/// 2026-09-26: How one ordered-color reading came out.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum OrderCell {
    /// 2026-09-26: The reply named the clip's colors in the right order.
    Match {
        clip: &'static str,
        seen: String,
    },
    /// 2026-09-26: The reply named palette colors, but not this clip's
    /// sequence.
    WrongOrder {
        clip: &'static str,
        want: String,
        got: String,
    },
    /// 2026-09-26: The reply named no palette color.
    NotSeen {
        clip: &'static str,
        reply: String,
    },
    /// 2026-09-26: Skipped: the server cannot decode the clip's container.
    Skipped {
        clip: &'static str,
        why: String,
    },
    Error {
        clip: &'static str,
        msg: String,
    },
}

/// 2026-09-26: How one other leg came out.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CountCell {
    Match { id: &'static str, detail: String },
    Mismatch { id: &'static str, detail: String },
    Skipped { id: &'static str, why: String },
    Error { id: &'static str, msg: String },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Verdict {
    Pass,
    Fail,
    /// 2026-09-26: The no-video control named every palette color, so the
    /// other legs are not evidence.
    Vacuous,
    /// 2026-09-26: Nothing was asserted: every leg was skipped.
    Inconclusive,
}

impl fmt::Display for Verdict {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            Verdict::Pass => "PASS",
            Verdict::Fail => "FAIL",
            Verdict::Vacuous => "VACUOUS",
            Verdict::Inconclusive => "INCONCLUSIVE",
        })
    }
}

/// 2026-09-26: Palette colors found in `reply` as standalone words, each once,
/// in the order of first appearance, so a recap ("the red one was first") does
/// not add a color.
pub fn colors_in_order(reply: &str, palette: &[&str]) -> Vec<String> {
    let lower = reply.to_lowercase();
    let mut hits: Vec<(usize, String)> = Vec::new();
    for c in palette {
        if let Some(at) = crate::benchmarks::first_standalone_term(&lower, c) {
            hits.push((at, (*c).to_string()));
        }
    }
    hits.sort_by_key(|(at, _)| *at);
    hits.into_iter().map(|(_, c)| c).collect()
}

/// 2026-09-26: Did the reply name exactly this clip's colors, in order?
pub fn order_matches(reply: &str, want: &[&str], palette: &[&str]) -> bool {
    let got = colors_in_order(reply, palette);
    got.len() == want.len() && got.iter().zip(want).all(|(g, w)| g == w)
}

/// 2026-09-26: Cells that count toward the verdict: everything except
/// `Skipped`. An `Error` counts as asserted and not passed.
pub fn asserted(order: &[OrderCell], counts: &[CountCell]) -> usize {
    order
        .iter()
        .filter(|c| {
            matches!(
                c,
                OrderCell::Match { .. }
                    | OrderCell::WrongOrder { .. }
                    | OrderCell::NotSeen { .. }
                    | OrderCell::Error { .. }
            )
        })
        .count()
        + counts
            .iter()
            .filter(|c| {
                matches!(
                    c,
                    CountCell::Match { .. } | CountCell::Mismatch { .. } | CountCell::Error { .. }
                )
            })
            .count()
}

pub fn passed(order: &[OrderCell], counts: &[CountCell]) -> usize {
    order
        .iter()
        .filter(|c| matches!(c, OrderCell::Match { .. }))
        .count()
        + counts
            .iter()
            .filter(|c| matches!(c, CountCell::Match { .. }))
            .count()
}

/// 2026-09-26: Inconclusive when nothing was asserted, else Vacuous when
/// `control_held` is false, else Pass only if every asserted cell passed.
pub fn verdict(order: &[OrderCell], counts: &[CountCell], control_held: bool) -> Verdict {
    let asserted_n = asserted(order, counts);
    if asserted_n == 0 {
        return Verdict::Inconclusive;
    }
    if !control_held {
        return Verdict::Vacuous;
    }
    if passed(order, counts) == asserted_n {
        Verdict::Pass
    } else {
        Verdict::Fail
    }
}

#[cfg(test)]
#[path = "score_tests.rs"]
mod score_tests;
