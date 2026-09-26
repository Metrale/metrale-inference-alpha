// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-26: The vacuity rule: whether a cell's tok/s measures throughput at
//! all. With n = the cell's successful requests, a cell is vacuous unless both
//! hold:
//!
//! 1. Occupancy: `Σ completion_tokens ≥ VACUITY_FLOOR × n × osl`. The
//!    aggregate is `Σ tokens / wall`, so this is the batch's mean fullness over
//!    the budget.
//! 2. Majority: at least half of the requests each delivered
//!    `VACUITY_FLOOR × osl` or more.
//!
//! So a minority of short requests, down to zero tokens, passes while the
//! cell's total stays at or above 80%: at C=128 and OSL 1024, 25 empty
//! requests beside 103 full ones deliver 80.5% and pass. At C=1 both clauses
//! reduce to the one request delivering 80%.
//!
//! Owner: bench (concurrency).
//! Invariants: none beyond the types.

use super::RequestEvidence;

/// 2026-09-26: The fraction of the total budget a cell must deliver, and of the
/// per-request budget its majority must each clear.
pub(super) const VACUITY_FLOOR: f64 = 0.8;

/// 2026-09-26: The counts the rule is decided on, kept so the warning line
/// prints the numbers the verdict used.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) struct Delivery {
    pub(super) requests: usize,
    pub(super) delivered: usize,
    pub(super) budget: usize,
    pub(super) cleared: usize,
    pub(super) min_completion: usize,
}

impl Delivery {
    pub(super) fn of(requests: &[RequestEvidence], osl: usize) -> Self {
        let bar = VACUITY_FLOOR * osl as f64;
        Self {
            requests: requests.len(),
            delivered: requests.iter().map(|r| r.completion_tokens).sum(),
            budget: requests.len() * osl,
            cleared: requests
                .iter()
                .filter(|r| r.completion_tokens as f64 >= bar)
                .count(),
            min_completion: requests
                .iter()
                .map(|r| r.completion_tokens)
                .min()
                .unwrap_or(0),
        }
    }

    pub(super) fn delivered_pct(&self) -> f64 {
        if self.budget == 0 {
            0.0
        } else {
            self.delivered as f64 / self.budget as f64 * 100.0
        }
    }

    /// 2026-09-26: Both clauses of the module header. A cell with no
    /// successful request is not vacuous: its errors already fail the verdict.
    pub(super) fn is_vacuous(&self) -> bool {
        if self.requests == 0 {
            return false;
        }
        let occupancy_short = (self.delivered as f64) < VACUITY_FLOOR * self.budget as f64;
        let majority_short = self.cleared * 2 < self.requests;
        occupancy_short || majority_short
    }

    pub(super) fn describe(&self, osl: usize) -> String {
        format!(
            "delivered {:.1}% of its {}×{osl}-token budget and {}/{} request(s) cleared \
             {:.0}% of it (min {} tok)",
            self.delivered_pct(),
            self.requests,
            self.cleared,
            self.requests,
            VACUITY_FLOOR * 100.0,
            self.min_completion,
        )
    }
}
