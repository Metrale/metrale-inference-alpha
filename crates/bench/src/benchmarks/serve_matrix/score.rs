// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-26: The serve matrix's bars and tally, as pure functions of the round
//! outcomes.
//!
//! Owner: bench, serve matrix.
//! Invariants:
//! - No I/O.
//! - An identity, codegen or tool-call signal that is `NotRun` fails its bar,
//!   and a planned round with no result fails as `no-result`.

use super::plan::Plan;

/// 2026-09-26: Coherence probes that must pass. `bars` caps it at the number
/// run, so with fewer probes every one must pass.
pub const COHERENCE_MIN_PASS: usize = 2;

/// 2026-09-26: Fraction below a stored baseline tok/s that counts as a regression.
pub const TPS_TOLERANCE: f64 = 0.10;

/// 2026-09-26: One probe's answer. `NotApplicable` passes only the tool-call bar,
/// where it means the server refused the tools request with a 4xx; on the
/// identity and codegen bars it fails.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub enum Signal {
    Pass,
    Fail(String),
    NotApplicable(String),
    /// 2026-09-26: The probe was not run. The default, so a half-built
    /// `Signals` fails its bars.
    #[default]
    NotRun,
}

impl Signal {
    pub fn is_fail(&self) -> bool {
        matches!(self, Signal::Fail(_))
    }

    pub fn text(&self) -> &str {
        match self {
            Signal::Pass => "PASS",
            Signal::Fail(_) => "FAIL",
            Signal::NotApplicable(_) => "N/A",
            Signal::NotRun => "—",
        }
    }

    /// 2026-09-26: The detail behind a `Fail` or `NotApplicable`.
    pub fn detail(&self) -> Option<&str> {
        match self {
            Signal::Fail(d) | Signal::NotApplicable(d) => Some(d),
            _ => None,
        }
    }
}

/// 2026-09-26: What one booted round measured.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Signals {
    /// 2026-09-26: Whether the endpoint serves this round's checkpoint; see
    /// [`super::probes::Coherence::identity`].
    pub identity: Signal,
    pub coherence_pass: usize,
    pub coherence_total: usize,
    pub codegen: Signal,
    pub tool_call: Signal,
    /// 2026-09-26: Reported, not gated: `bars` does not read it.
    pub long_ctx: Signal,
    /// 2026-09-26: Decode tokens/sec from `probes::tps_probe`: `None` when the
    /// client TPOT was undefined, `Some(0.0)` when the request failed.
    pub tps: Option<f64>,
}

/// 2026-09-26: What became of a planned round.
#[derive(Clone, Debug, PartialEq)]
pub enum Outcome {
    /// 2026-09-26: Came up and was probed.
    Probed(Box<Signals>),
    /// 2026-09-26: Planned and attempted, but `serve` failed or the endpoint
    /// did not answer. Fails as `did-not-boot`, never a skip.
    BootFailed(String),
    /// 2026-09-26: Planned but not attempted: the benchmark had no handle, host
    /// or options (`run_round`). Fails as `no-result`.
    NotReached,
}

/// 2026-09-26: One planned round's result.
#[derive(Clone, Debug, PartialEq)]
pub struct RoundResult {
    pub label: String,
    pub outcome: Outcome,
    /// 2026-09-26: Stored baseline tok/s for this label, if this box has one.
    pub baseline_tps: Option<f64>,
}

impl RoundResult {
    pub fn signals(&self) -> Option<&Signals> {
        match &self.outcome {
            Outcome::Probed(s) => Some(s),
            _ => None,
        }
    }

    /// 2026-09-26: The bars this round failed. Empty means verified.
    pub fn bars(&self) -> Vec<String> {
        let signals = match &self.outcome {
            // 2026-09-26: Worded apart: a boot failure and an unattempted round
            // need different fixes.
            Outcome::BootFailed(why) => return vec![format!("did-not-boot ({why})")],
            Outcome::NotReached => return vec!["no-result".into()],
            Outcome::Probed(s) => s,
        };
        let mut fails = Vec::new();
        // 2026-09-26: A `NotRun` signal fails its bar; `Signals::default()` is
        // all `NotRun`.
        for (name, signal, allows_not_applicable) in [
            ("wrong-model", &signals.identity, false),
            ("codegen", &signals.codegen, false),
            ("tool_call", &signals.tool_call, true),
        ] {
            match signal {
                Signal::Pass => {}
                Signal::NotApplicable(_) if allows_not_applicable => {}
                Signal::NotApplicable(_) => fails.push(format!("{name}(not-applicable)")),
                Signal::Fail(_) => fails.push(name.to_string()),
                Signal::NotRun => fails.push(format!("{name}(not-probed)")),
            }
        }
        let coherence_bar = COHERENCE_MIN_PASS.min(signals.coherence_total);
        if signals.coherence_total == 0
            || signals.coherence_pass < coherence_bar
            || signals.coherence_pass > signals.coherence_total
        {
            fails.push(format!(
                "coherence({}/{})",
                signals.coherence_pass, signals.coherence_total
            ));
        }
        if let Some(tps) = signals.tps {
            if !tps.is_finite() {
                fails.push("tps(non-finite)".into());
            } else if tps <= 0.0 {
                fails.push("tps(0)".into());
            } else if let Some(base) = self
                .baseline_tps
                .filter(|baseline| baseline.is_finite() && *baseline > 0.0)
            {
                let floor = base * (1.0 - TPS_TOLERANCE);
                if tps < floor {
                    fails.push(format!("tps({tps:.1}<{floor:.1})"));
                }
            }
        }
        fails
    }

    /// 2026-09-26: The note for a positive, finite tok/s that had no valid
    /// baseline to compare with.
    pub fn tps_note(&self) -> Option<&'static str> {
        let tps = self.signals()?.tps?;
        if tps.is_finite()
            && tps > 0.0
            && self
                .baseline_tps
                .filter(|baseline| baseline.is_finite() && *baseline > 0.0)
                .is_none()
        {
            return Some("no baseline — liveness only");
        }
        None
    }
}

/// 2026-09-26: The whole matrix, scored.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Tally {
    pub verified: usize,
    pub planned: usize,
    pub skipped: usize,
    pub excluded: usize,
    /// 2026-09-26: `(label, bars)` for every round below bar.
    pub failures: Vec<(String, Vec<String>)>,
}

impl Tally {
    pub fn passed(&self) -> bool {
        self.planned > 0 && self.failures.is_empty()
    }
}

/// 2026-09-26: Score every planned round against its result. Walks the plan, not
/// the results, so a planned round with no result fails as `no-result`.
pub fn tally(plan: &Plan, results: &[RoundResult]) -> Tally {
    let mut out = Tally {
        planned: plan.planned_count(),
        skipped: plan.skipped().count(),
        excluded: plan.excluded_count(),
        ..Tally::default()
    };
    for round in plan.planned() {
        let label = round.label();
        let bars = match results.iter().find(|r| r.label == label) {
            Some(r) => r.bars(),
            None => vec!["no-result".into()],
        };
        if bars.is_empty() {
            out.verified += 1;
        } else {
            out.failures.push((label, bars));
        }
    }
    out
}

/// 2026-09-26: The verdict sentence: coverage, skips (up to three named), the
/// filter count, and each failing round's bars.
pub fn verdict_text(tally: &Tally, plan: &Plan) -> String {
    let coverage = format!(
        "{}/{} planned checkpoints verified",
        tally.verified, tally.planned
    );
    let mut extra = Vec::new();
    if tally.skipped > 0 {
        let names: Vec<String> = plan
            .skipped()
            .take(3)
            .map(|(r, why)| format!("{} ({})", r.model, why.reason()))
            .collect();
        extra.push(format!(
            "{} not runnable on this box: {}{}",
            tally.skipped,
            names.join(", "),
            if tally.skipped > 3 { ", …" } else { "" }
        ));
    }
    if tally.excluded > 0 {
        extra.push(format!("{} outside the filter", tally.excluded));
    }
    let tail = if extra.is_empty() {
        String::new()
    } else {
        format!(" · {}", extra.join(" · "))
    };
    if tally.failures.is_empty() {
        return format!("{coverage}{tail}");
    }
    let detail: Vec<String> = tally
        .failures
        .iter()
        .map(|(label, bars)| format!("{label}: {}", bars.join(", ")))
        .collect();
    format!(
        "{coverage}{tail} — {} below bar: {}",
        tally.failures.len(),
        detail.join(" · ")
    )
}

#[cfg(test)]
#[path = "score_tests.rs"]
mod tests;
