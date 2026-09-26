// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-26: The pull-request journey ledger: one append-only JSONL file per
//! pull request, `governance/pr-<n>.jsonl`, recording gate verdicts,
//! lifecycle states, classifier opinions and measurements.
//!
//! Owner: metrale-governance.
//! Invariants:
//! - This crate writes only by appending a line to a journey file.
//! - Readers deduplicate by [`Event::identity`], which leaves out the
//!   timestamp, so a file's content is a grow-only set: `.gitattributes`
//!   declares `governance/*.jsonl merge=union`.
//! - The crate does not depend on the bench gate. The gate check reads a
//!   journey only to print the advisory intent; its exit code is computed from
//!   the gate verdicts alone.
//!
//! [`materialize`] builds an in-memory graph from a journey; the JSONL is the
//! only form that is committed.

pub mod event;
pub mod ledger;

#[cfg(test)]
#[path = "ledger_tests.rs"]
mod ledger_tests;

pub use event::{Event, EventKind, Verdict};
pub use ledger::{Journey, append, materialize, read_all};
