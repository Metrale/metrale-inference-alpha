// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-26: The event types a journey file holds, one JSON object per line.
//!
//! Owner: metrale-governance.
//! Invariants: none beyond the types.

use serde::{Deserialize, Serialize};

/// 2026-09-26: The verdict a gate reached.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Verdict {
    Pass,
    Fail,
    /// 2026-09-26: No record covered the commit, as opposed to `Fail`, a run
    /// whose numbers were out of bounds.
    Missing,
}

/// 2026-09-26: What happened. Serialised with a `kind` tag, flattened into
/// [`Event`].
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum EventKind {
    /// 2026-09-26: A lifecycle transition. `to` is meant to name a state of the
    /// agent loop table in CONTRIBUTING.md (`branch` … `merge`); nothing
    /// validates it.
    State { to: String },
    /// 2026-09-26: A gate was evaluated. `invalidated_by` names the paths that
    /// re-opened it; it is omitted from the JSON when empty.
    Gate {
        id: String,
        verdict: Verdict,
        #[serde(default, skip_serializing_if = "Vec::is_empty")]
        invalidated_by: Vec<String>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        detail: Option<String>,
    },
    /// 2026-09-26: The advisory classifier's opinion. `status` is free text;
    /// the gate check's intent report counts only `ok` and `partial`.
    Category { value: String, status: String },
    /// 2026-09-26: A benchmark run's metrics.
    Measurement {
        benchmark: String,
        #[serde(default)]
        metrics: std::collections::BTreeMap<String, f64>,
    },
}

/// 2026-09-26: One line of a journey. [`Event::identity`] is
/// `(head_sha, run_id, attempt, kind)`: `pr` and `at` are left out, so the same
/// event appended twice collapses when read.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Event {
    pub pr: u64,
    pub head_sha: String,
    /// 2026-09-26: The CI run that observed this; empty when absent from the
    /// JSON.
    #[serde(default)]
    pub run_id: String,
    /// 2026-09-26: Which attempt of that run. Part of the identity, so a re-run
    /// is kept as a separate event.
    #[serde(default)]
    pub attempt: u32,
    /// 2026-09-26: Unix seconds. Not part of the identity.
    pub at: u64,
    #[serde(flatten)]
    pub kind: EventKind,
}

impl Event {
    /// 2026-09-26: The deduplication key: head sha, run id, attempt and the
    /// serialised kind.
    pub fn identity(&self) -> String {
        // 2026-09-26: The serialised kind carries the variant tag and all of its
        // fields, so a new variant needs no change here.
        let kind = serde_json::to_string(&self.kind).unwrap_or_default();
        format!(
            "{}|{}|{}|{}",
            self.head_sha, self.run_id, self.attempt, kind
        )
    }

    /// 2026-09-26: The `label` field of this event's node in
    /// [`crate::ledger::materialize`].
    pub fn node_label(&self) -> &'static str {
        match self.kind {
            EventKind::State { .. } => "state",
            EventKind::Gate { .. } => "gate",
            EventKind::Category { .. } => "category",
            EventKind::Measurement { .. } => "measurement",
        }
    }
}
