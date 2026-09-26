// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-25: Why a sample is unmeasured: one entry per request that failed or never came back.
//!
//! Owner: bench, scheduler-equivalence gate.
//! Invariants:
//! - Every sample a cell counts as unmeasured has at least one entry here:
//!   the reference's failure, the compared leg's failure, or its absence.
//! - An entry names its leg, lane, concurrency and sample, so a record
//!   explains its failure without a rerun.
//! - Pure: no I/O, and nothing here takes part in the verdict.

use std::collections::BTreeMap;
use std::fmt;
use std::time::Duration;

use super::compare::{Leg, Pass};
use super::host::Lane;
use crate::http::{FailureKind, RequestFailure};

/// 2026-09-25: Every cause class, each a `unmeasured_cause_<class>` metric
/// even at zero.
pub const CLASSES: [&str; 10] = [
    "timeout",
    "http_4xx",
    "http_5xx",
    "http_other",
    "transport",
    "malformed",
    "truncated",
    "server_error",
    "missing",
    "other",
];

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Cause {
    Failed {
        failure: RequestFailure,
        elapsed: Duration,
    },
    /// 2026-09-25: The leg holds no reply for the sample, or the leg never ran.
    Missing,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Unmeasured {
    pub pass: Pass,
    pub lane: Lane,
    pub concurrency: usize,
    pub sample_id: String,
    pub cause: Cause,
}

impl Unmeasured {
    /// 2026-09-25: One of `CLASSES`.
    pub fn class(&self) -> &'static str {
        let Cause::Failed { failure, .. } = &self.cause else {
            return "missing";
        };
        match failure.kind {
            FailureKind::Timeout => "timeout",
            FailureKind::Status(c) if (400..500).contains(&c) => "http_4xx",
            FailureKind::Status(c) if (500..600).contains(&c) => "http_5xx",
            FailureKind::Status(_) => "http_other",
            FailureKind::Transport => "transport",
            FailureKind::Malformed => "malformed",
            FailureKind::Truncated => "truncated",
            FailureKind::ServerError => "server_error",
            FailureKind::Other => "other",
        }
    }
}

/// 2026-09-25: One line, every string quoted, so a body with newlines stays
/// on its line and the fields stay parseable.
impl fmt::Display for Unmeasured {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "unmeasured: leg={} lane={} C={} sample={:?} cause={}",
            self.pass.label(),
            self.lane.label(),
            self.concurrency,
            self.sample_id,
            self.class()
        )?;
        match &self.cause {
            Cause::Missing => write!(f, " error=\"no reply in this leg\""),
            Cause::Failed { failure, elapsed } => {
                let status = match failure.kind {
                    FailureKind::Status(c) => c.to_string(),
                    _ => "-".to_string(),
                };
                write!(
                    f,
                    " status={status} finish_reason={} elapsed={:.3}s error={:?} body={:?}",
                    failure.finish_reason.as_deref().unwrap_or("-"),
                    elapsed.as_secs_f64(),
                    failure.message,
                    failure.body
                )
            }
        }
    }
}

/// 2026-09-25: The entries for one cell, walking the reference's samples as
/// `compare::diff` does. The candidate leg is required, so its absence is an
/// entry per sample; an absent control leg was not run and is none.
pub fn collect(reference: &Leg, candidate: Option<&Leg>, control: Option<&Leg>) -> Vec<Unmeasured> {
    let entry = |pass, sample_id: &str, cause| Unmeasured {
        pass,
        lane: reference.lane,
        concurrency: reference.concurrency,
        sample_id: sample_id.to_string(),
        cause,
    };
    let mut out: Vec<Unmeasured> = reference
        .replies
        .iter()
        .filter_map(|r| {
            let failure = r.outcome.as_ref().err()?;
            Some(entry(Pass::Sync, &r.sample_id, failed(failure, r.elapsed)))
        })
        .collect();
    if candidate.is_none() {
        out.extend(
            reference
                .replies
                .iter()
                .map(|r| entry(Pass::Async, &r.sample_id, Cause::Missing)),
        );
    }
    for other in [candidate, control].into_iter().flatten() {
        for r in &reference.replies {
            let cause = match other.find(&r.sample_id) {
                None => Cause::Missing,
                Some(o) => match &o.outcome {
                    Ok(_) => continue,
                    Err(failure) => failed(failure, o.elapsed),
                },
            };
            out.push(entry(other.pass, &r.sample_id, cause));
        }
    }
    out
}

fn failed(failure: &RequestFailure, elapsed: Duration) -> Cause {
    Cause::Failed {
        failure: failure.clone(),
        elapsed,
    }
}

/// 2026-09-25: `"<leg> <class> <n>"` per (leg, class), comma-separated, or
/// `"none"`. Leg and class order are fixed, so equal inputs read the same.
pub fn breakdown(entries: &[Unmeasured]) -> String {
    let mut tally: BTreeMap<(Pass, &str), usize> = BTreeMap::new();
    for e in entries {
        *tally.entry((e.pass, e.class())).or_default() += 1;
    }
    if tally.is_empty() {
        return "none".to_string();
    }
    tally
        .iter()
        .map(|((pass, class), n)| format!("{} {class} {n}", pass.label()))
        .collect::<Vec<_>>()
        .join(", ")
}

/// 2026-09-25: `unmeasured_cause_<class>` for every class. These count
/// requests, not comparisons, so they need not sum to `unmeasured`.
pub fn metrics<'a>(entries: impl Iterator<Item = &'a Unmeasured>) -> BTreeMap<String, f64> {
    let key = |class: &str| format!("unmeasured_cause_{class}");
    let mut m: BTreeMap<String, f64> = CLASSES.iter().map(|c| (key(c), 0.0)).collect();
    for e in entries {
        *m.entry(key(e.class())).or_default() += 1.0;
    }
    m
}
