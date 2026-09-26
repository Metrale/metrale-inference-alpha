// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-26: The certify campaign as a state machine: which unit runs next,
//! what a finished unit means for the rest, and when to stop.
//!
//! Owner: server CLI (`met benchmark certify`).
//! Invariants:
//! - No I/O. The local driver takes units from [`Campaign::next_to_start`]; the
//!   remote driver picks one and calls [`Campaign::start`]. Both report each
//!   outcome to [`Campaign::finished`] and each guard answer to [`Campaign::guard`].
//! - A failed unit (verdict, harness or timeout) skips every pending unit unless
//!   `keep_going`. A retryable harness failure is run once more; a timeout is not.
//! - Perf-path drift, a guard error and a cancel abort the campaign.

use super::guard::Drift;
use super::plan::Unit;
use super::runner::RunOutcome;

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Phase {
    Pending,
    Running,
    Passed,
    MemberDone,
    Failed(String),
    Skipped(String),
}

pub struct Campaign {
    pub units: Vec<Unit>,
    pub phase: Vec<Phase>,
    pub keep_going: bool,
    retried: Vec<bool>,
    /// 2026-09-26: Set by an abort: perf-path drift, a guard that could not
    /// answer, or a cancel. A failed unit without `keep_going` skips the rest
    /// but does not set this.
    aborted: Option<String>,
    fail_seen: bool,
}

#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Summary {
    pub passed: Vec<String>,
    pub member_done: Vec<String>,
    pub failed: Vec<(String, String)>,
    pub skipped: Vec<(String, String)>,
    pub aborted: Option<String>,
}

impl Campaign {
    pub fn new(units: Vec<Unit>, keep_going: bool) -> Self {
        let n = units.len();
        Self {
            units,
            phase: vec![Phase::Pending; n],
            keep_going,
            retried: vec![false; n],
            aborted: None,
            fail_seen: false,
        }
    }

    /// 2026-09-26: The first pending unit, marked running; `None` when there is
    /// none or the campaign is stopped.
    pub fn next_to_start(&mut self) -> Option<usize> {
        if self.stopped() {
            return None;
        }
        let i = self.phase.iter().position(|p| *p == Phase::Pending)?;
        self.phase[i] = Phase::Running;
        Some(i)
    }

    /// 2026-09-26: Mark a pending unit the caller chose as running.
    pub fn start(&mut self, i: usize) {
        debug_assert_eq!(self.phase[i], Phase::Pending);
        self.phase[i] = Phase::Running;
    }

    /// 2026-09-26: True when the campaign is aborted or no unit is pending or
    /// running.
    pub fn stopped(&self) -> bool {
        self.aborted.is_some()
            || self
                .phase
                .iter()
                .all(|p| !matches!(p, Phase::Pending | Phase::Running))
    }

    /// 2026-09-26: Record a unit's outcome. Returns `true` when the unit should
    /// run again: the first retryable harness failure puts it back to pending.
    pub fn finished(&mut self, i: usize, outcome: RunOutcome) -> bool {
        match outcome {
            RunOutcome::Passed { .. } => self.phase[i] = Phase::Passed,
            RunOutcome::MemberDone { .. } => self.phase[i] = Phase::MemberDone,
            RunOutcome::VerdictFail { reason, .. } => {
                self.phase[i] = Phase::Failed(reason);
                self.fail_seen = true;
                if !self.keep_going {
                    self.stop_rest(format!(
                        "stopped after {} failed its verdict (pass --keep-going to run on)",
                        self.units[i].label()
                    ));
                }
            }
            RunOutcome::Harness { reason, retryable } => {
                if retryable && !self.retried[i] {
                    self.retried[i] = true;
                    self.phase[i] = Phase::Pending;
                    return true;
                }
                self.phase[i] = Phase::Failed(format!("harness: {reason}"));
                self.fail_seen = true;
                if !self.keep_going {
                    self.stop_rest(format!(
                        "stopped after {} could not be run",
                        self.units[i].label()
                    ));
                }
            }
            RunOutcome::TimedOut => {
                self.phase[i] = Phase::Failed("timed out".into());
                self.fail_seen = true;
                if !self.keep_going {
                    self.stop_rest(format!("stopped after {} timed out", self.units[i].label()));
                }
            }
            RunOutcome::Cancelled => {
                self.phase[i] = Phase::Failed("cancelled".into());
                self.abort("cancelled".into());
            }
        }
        false
    }

    /// 2026-09-26: Apply the drift guard's answer. Perf-path drift and a guard
    /// error both abort and return the abort reason; any other answer is `None`.
    pub fn guard(&mut self, result: Result<Drift, String>) -> Option<&str> {
        match result {
            Ok(Drift::Unmoved) | Ok(Drift::MovedHarmlessly { .. }) => None,
            Ok(Drift::PerfPathMoved { head, paths }) => {
                self.abort(format!(
                    "a perf path moved on the guarded branch (now {}): {} — every record \
                     taken after this names a dead tree",
                    &head[..head.len().min(10)],
                    paths.join(", ")
                ));
                self.aborted.as_deref()
            }
            Err(e) => {
                self.abort(format!(
                    "the drift guard could not answer ({e}); not continuing blind"
                ));
                self.aborted.as_deref()
            }
        }
    }

    pub fn cancel(&mut self) {
        self.abort("cancelled".into());
    }

    fn abort(&mut self, why: String) {
        if self.aborted.is_none() {
            self.stop_rest(why.clone());
            self.aborted = Some(why);
        }
    }

    fn stop_rest(&mut self, why: String) {
        for p in &mut self.phase {
            if *p == Phase::Pending {
                *p = Phase::Skipped(why.clone());
            }
        }
    }

    pub fn summary(&self) -> Summary {
        let mut s = Summary {
            aborted: self.aborted.clone(),
            ..Default::default()
        };
        for (u, p) in self.units.iter().zip(&self.phase) {
            match p {
                Phase::Passed => s.passed.push(u.label()),
                Phase::MemberDone => s.member_done.push(u.label()),
                Phase::Failed(r) => s.failed.push((u.label(), r.clone())),
                Phase::Skipped(r) => s.skipped.push((u.label(), r.clone())),
                Phase::Pending | Phase::Running => {}
            }
        }
        s
    }

    /// 2026-09-26: `3` when aborted; `2` when a unit failed or `certified` (the
    /// final gate check's answer) is false; otherwise `0`.
    pub fn exit_code(&self, certified: bool) -> i32 {
        if self.aborted.is_some() {
            3
        } else if self.fail_seen || !certified {
            2
        } else {
            0
        }
    }
}

#[cfg(test)]
#[path = "state_tests.rs"]
mod state_tests;
