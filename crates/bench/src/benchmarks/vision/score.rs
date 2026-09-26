// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-26: The vision benchmark's cells and verdict. A run whose no-image
//! control did not hold is VACUOUS, whatever the other cells say.
//!
//! Owner: bench, vision.
//! Invariants: none beyond the types.

use std::fmt;

/// 2026-09-26: How one geometry cell came out.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum GeomCell {
    /// 2026-09-26: The measured vision tokens equal the prediction.
    Match {
        fixture: &'static str,
        tokens: usize,
    },
    /// 2026-09-26: The measured vision tokens differ from the prediction.
    Mismatch {
        fixture: &'static str,
        want: usize,
        got: usize,
    },
    /// 2026-09-26: Not asserted: the server refused the image as over its
    /// encoder's capacity.
    Unmeasured { fixture: &'static str, why: String },
    /// 2026-09-26: The request, or the vision-token subtraction, failed.
    Error { fixture: &'static str, msg: String },
}

/// 2026-09-26: How one capability probe came out.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ProbeCell {
    Pass { id: &'static str },
    Fail { id: &'static str, reply: String },
    Error { id: &'static str, msg: String },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Verdict {
    Pass,
    Fail,
    /// 2026-09-26: The no-image control answered as though it saw an image, so
    /// the capability probes are not evidence.
    Vacuous,
}

impl fmt::Display for Verdict {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            Verdict::Pass => "PASS",
            Verdict::Fail => "FAIL",
            Verdict::Vacuous => "VACUOUS",
        })
    }
}

/// 2026-09-26: Does a reply satisfy a probe's expectations?
///
/// Case-insensitive term matching. A numeric term matches where no digit
/// touches it; any other term must stand alone as a word
/// (`first_standalone_term`). Every `want_all` term must match and no
/// `want_none` term may.
pub fn reply_matches(reply: &str, want_all: &[&str], want_none: &[&str]) -> bool {
    let hay = reply.to_lowercase();
    let contains_term = |term: &str| {
        if !term.is_empty() && term.chars().all(|ch| ch.is_ascii_digit()) {
            // 2026-09-26: The HD fixture's label is `07_hd_1280x720  1280x720`
            // (`scripts/gen_test_images.py`), where letters and `_` touch 1280.
            hay.match_indices(term).any(|(at, _)| {
                !hay[..at]
                    .chars()
                    .next_back()
                    .is_some_and(|ch| ch.is_ascii_digit())
                    && !hay[at + term.len()..]
                        .chars()
                        .next()
                        .is_some_and(|ch| ch.is_ascii_digit())
            })
        } else {
            crate::benchmarks::first_standalone_term(&hay, term).is_some()
        }
    };
    want_all.iter().all(|term| contains_term(term))
        && !want_none.iter().any(|term| contains_term(term))
}

/// 2026-09-26: Fold the geometry and probe cells into one verdict: Vacuous
/// when the control did not hold, whatever else happened; otherwise Fail on
/// any geometry `Mismatch` or `Error` or any probe that did not pass; otherwise
/// Pass. `Unmeasured` cells do not fail the run.
pub fn verdict(geom: &[GeomCell], probes: &[ProbeCell], control_held: bool) -> Verdict {
    let geom_bad = geom
        .iter()
        .any(|c| matches!(c, GeomCell::Mismatch { .. }) || matches!(c, GeomCell::Error { .. }));
    let probes_bad = probes.iter().any(|c| !matches!(c, ProbeCell::Pass { .. }));

    if !control_held {
        return Verdict::Vacuous;
    }
    if geom_bad || probes_bad {
        return Verdict::Fail;
    }
    Verdict::Pass
}

/// 2026-09-26: Fold the integrity and concurrency legs into the verdict: a
/// Pass becomes Fail when an integrity leg failed or the concurrency sweep is
/// not clean. Fail and Vacuous stay as they are.
pub fn with_runtime_checks(
    base: Verdict,
    integrity_failed: bool,
    concurrency_clean: bool,
) -> Verdict {
    match base {
        Verdict::Pass if integrity_failed || !concurrency_clean => Verdict::Fail,
        other => other,
    }
}

/// 2026-09-26: How many geometry cells asserted something (`Match` or
/// `Mismatch`). Reported beside the verdict, because a run where every cell was
/// `Unmeasured` passes `verdict`.
pub fn asserted_cells(geom: &[GeomCell]) -> usize {
    geom.iter()
        .filter(|c| matches!(c, GeomCell::Match { .. } | GeomCell::Mismatch { .. }))
        .count()
}

#[cfg(test)]
#[path = "score_tests.rs"]
mod score_tests;
