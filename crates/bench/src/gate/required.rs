// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-26: What a PR owes: the gates its changed paths invalidate
//! (`by_path`) united with the gates its recorded intent classifications
//! imply (`by_intent`).
//!
//! `pr_taxonomy::validate` refuses any `_benches` id outside
//! [`super::coverage::REQUIRED`], so the intent half names only required
//! gates. The intent half matters where paths invalidate nothing: paths off
//! `PERF_PATHS` (`docker/`, `docs/`, `scripts/`, `bench/`), `BENCH.toml`
//! files, and `crates/bench/src/gate` files outside `BOUNDARY_FILES`.
//!
//! [`super::check::check_gates`] checks every gate in `REQUIRED_GATES`
//! whatever this module computes. `met benchmark --pull-request-gate-check`
//! prints the intent half after the verdicts, and it does not change the
//! exit code.
//!
//! Every classification recorded for the PR counts, and the result is their
//! union, so re-running the classifier can only add gates.
//!
//! Owner: bench gate (intent).
//! Invariants:
//! - `by_path` does not depend on the classifications or the taxonomy.
//! - [`required_for`] and [`report`] read no files.

use std::collections::BTreeSet;

use super::pr_taxonomy::{Node, benches_for};

/// 2026-09-26: Both halves, kept apart so a report can say which half
/// required a gate.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct RequiredSet {
    /// 2026-09-26: Gates the changed paths invalidate
    /// (`coverage::invalidated_by`).
    pub by_path: BTreeSet<String>,
    /// 2026-09-26: Gates the classifications imply, unioned over all of them.
    pub by_intent: BTreeSet<String>,
}

impl RequiredSet {
    /// 2026-09-26: `by_path ∪ by_intent`.
    pub fn union(&self) -> BTreeSet<String> {
        self.by_path.union(&self.by_intent).cloned().collect()
    }

    /// 2026-09-26: `by_intent` minus `by_path`: what intent added.
    pub fn intent_only(&self) -> BTreeSet<String> {
        self.by_intent.difference(&self.by_path).cloned().collect()
    }
}

/// 2026-09-26: Compute both halves. `categories` is every recorded
/// classification path; an empty slice (not classified) or an empty `roots`
/// yields an empty intent half.
pub fn required_for(changed: &[String], categories: &[Vec<String>], roots: &[Node]) -> RequiredSet {
    let by_path = super::coverage::invalidated_by(changed.iter().map(String::as_str))
        .into_iter()
        .map(str::to_string)
        .collect();
    let mut by_intent = BTreeSet::new();
    for category in categories {
        by_intent.extend(benches_for(roots, category));
    }
    RequiredSet { by_path, by_intent }
}

/// 2026-09-26: Parse `performance/decode` into `["performance", "decode"]`.
/// Segments are trimmed and empty ones dropped: `benches_for` would stop at an
/// empty segment and lose the benchmarks below it.
pub fn parse_category(value: &str) -> Vec<String> {
    value
        .split('/')
        .map(str::trim)
        .filter(|s| !s.is_empty())
        .map(str::to_string)
        .collect()
}

#[cfg(test)]
#[path = "required_tests.rs"]
mod required_tests;

/// 2026-09-26: Where the intent half came from. It keeps "not classified"
/// apart from "the ledger could not be read", which both give an empty intent
/// half.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum IntentSource {
    /// 2026-09-26: No PR number was given; not evaluated.
    NotRequested,
    /// 2026-09-26: The ledger is missing or holds no non-empty `ok` or
    /// `partial` classification for this PR.
    NotRecorded { ledger: std::path::PathBuf },
    /// 2026-09-26: The ledger could not be read.
    Degraded { reason: String },
    Recorded {
        /// 2026-09-26: Every `ok` or `partial` path recorded for this PR, in
        /// ledger order, without duplicates.
        categories: Vec<Vec<String>>,
        /// 2026-09-26: Rows with any other status (`abstain`, `error`):
        /// counted, never used as intent.
        skipped: usize,
    },
}

/// 2026-09-26: Both halves plus where the intent half came from.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RequiredReport {
    pub set: RequiredSet,
    pub source: IntentSource,
}

/// 2026-09-26: Read the classifications recorded for `pr` from its ledger,
/// deduplicated. Every Category row counts, whatever its `head_sha`: the row
/// for a head is committed after that head, so a head filter would find
/// nothing. Unioning across older heads can only add gates.
pub fn intent_source(root: &std::path::Path, pr: Option<u64>) -> IntentSource {
    let Some(pr) = pr else {
        return IntentSource::NotRequested;
    };
    let ledger = metrale_governance::ledger::path_for(root, pr);
    if !ledger.exists() {
        return IntentSource::NotRecorded { ledger };
    }
    let journey = match metrale_governance::ledger::read_all(&ledger) {
        Ok(j) => j.deduplicated(),
        // 2026-09-26: `read_all` fails on a malformed line; that is reported
        // as `Degraded`, not as an error and not as "no intent".
        Err(e) => {
            return IntentSource::Degraded {
                reason: format!("{}: {e:#}", ledger.display()),
            };
        }
    };

    let (mut categories, mut skipped) = (Vec::new(), 0usize);
    for event in &journey.events {
        let metrale_governance::event::EventKind::Category { value, status } = &event.kind else {
            continue;
        };
        // 2026-09-26: `partial` counts: its matched prefix still implies the
        // ancestors' `_benches`.
        if status != "ok" && status != "partial" {
            skipped += 1;
            continue;
        }
        let segments = parse_category(value);
        if !segments.is_empty() && !categories.contains(&segments) {
            categories.push(segments);
        }
    }
    if categories.is_empty() {
        return IntentSource::NotRecorded { ledger };
    }
    IntentSource::Recorded {
        categories,
        skipped,
    }
}

/// 2026-09-26: Assemble the report. Only a `Recorded` source contributes
/// intent. Reads no files: the I/O is in [`intent_source`] and the caller's
/// taxonomy load.
pub fn report(changed: &[String], source: IntentSource, roots: &[Node]) -> RequiredReport {
    let categories: &[Vec<String>] = match &source {
        IntentSource::Recorded { categories, .. } => categories,
        _ => &[],
    };
    RequiredReport {
        set: required_for(changed, categories, roots),
        source,
    }
}
