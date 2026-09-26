// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-26: Turning the box's roster into the rounds this run intends to
//! cover. A round is planned (the run boots it and it must produce a result),
//! skipped (the box cannot serve it; counted and reported) or excluded (outside
//! the operator's filter).
//!
//! Owner: bench, serve matrix.
//! Invariants:
//! - Every roster entry is one round; skipped and excluded rounds stay in
//!   `Plan::rounds`.
//! - A round is never both skipped and excluded.
//! - Rounds are sorted by model id, then quant.

pub use super::host::Absence;
use super::host::ServeCandidate;

/// 2026-09-26: Why a candidate cannot be served; `None` when it can.
pub type Skip = Option<Absence>;

/// 2026-09-26: One roster entry, planned, skipped or excluded.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Round {
    pub model: String,
    pub quant: String,
    /// 2026-09-26: Set when the box cannot serve the checkpoint; carries the reason.
    pub skipped: Skip,
    /// 2026-09-26: Filtered out by the operator's `include` pattern. Set only on
    /// a servable checkpoint: one that is unservable and outside the filter
    /// counts as skipped alone.
    pub excluded: bool,
}

impl Round {
    pub fn is_planned(&self) -> bool {
        self.skipped.is_none() && !self.excluded
    }

    /// 2026-09-26: Label used in the table and in a failure line.
    pub fn label(&self) -> String {
        match self.quant.trim() {
            "" | "-" => self.model.clone(),
            q => format!("{} · {q}", self.model),
        }
    }
}

/// 2026-09-26: The whole roster, classified; skipped and excluded rounds are kept
/// and counted in the result.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Plan {
    pub rounds: Vec<Round>,
}

impl Plan {
    /// 2026-09-26: Classify `roster` under an operator filter. `include` is a
    /// case-insensitive substring of the model id; empty means everything the
    /// box can serve. Sorted by model id, then quant, so the round order does
    /// not depend on the roster's order.
    pub fn build(roster: &[ServeCandidate], include: &str) -> Self {
        let needle = include.trim().to_lowercase();
        let mut rounds: Vec<Round> = roster
            .iter()
            .map(|c| Round {
                model: c.model.clone(),
                quant: c.quant.clone(),
                skipped: c.absent,
                excluded: c.absent.is_none()
                    && !needle.is_empty()
                    && !c.model.to_lowercase().contains(&needle),
            })
            .collect();
        rounds.sort_by(|a, b| a.model.cmp(&b.model).then(a.quant.cmp(&b.quant)));
        Self { rounds }
    }

    /// 2026-09-26: The rounds that will be booted, in order. The run's cursor and
    /// `score::tally` both walk this iterator.
    pub fn planned(&self) -> impl Iterator<Item = &Round> {
        self.rounds.iter().filter(|r| r.is_planned())
    }

    pub fn planned_count(&self) -> usize {
        self.planned().count()
    }

    /// 2026-09-26: Checkpoints the box cannot serve, each with its reason.
    pub fn skipped(&self) -> impl Iterator<Item = (&Round, Absence)> {
        self.rounds
            .iter()
            .filter_map(|r| r.skipped.map(|why| (r, why)))
    }

    pub fn excluded_count(&self) -> usize {
        self.rounds.iter().filter(|r| r.excluded).count()
    }
}

#[cfg(test)]
#[path = "plan_tests.rs"]
mod tests;
