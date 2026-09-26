// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-26: KAT equality: the same sample, in different request orders,
//! byte for byte.
//!
//! A known-answer test is a claim about its input only if the output does not
//! also depend on what ran before it. This gate issues one BFCL draw against
//! one server in two or more orders and requires every `sample_id`'s reply to
//! be byte-identical across them.
//!
//! `ssm-state-poisoning-gate` cannot answer this: it replays one fixed script
//! in the same order every round, and it counts a reworded reply
//! (`RoundVerdict::Jittered`) as a pass. Here a reworded reply is the finding.
//!
//! Owner: bench, kat_equality.
//! Invariants: none beyond the types.

pub mod compare;
pub mod driver;
pub mod report;

pub use compare::{
    Observation, OrderRun, SampleVerdict, Score, permutation, score, verdict, verdict_for,
};

pub use driver::{DESCRIPTOR, METADATA};

#[cfg(test)]
#[path = "compare_tests.rs"]
mod compare_tests;

#[cfg(test)]
#[path = "driver_tests.rs"]
mod driver_tests;
