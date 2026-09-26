// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-26: Cross-contamination benchmark: concurrent requests must not
//! change each other's output. `score` holds the leg design and the
//! classification, `prompts` the probe corpus, and `driver` the state machine
//! that runs the legs.
//!
//! Owner: bench (contamination).
//! Invariants: none beyond the types.

pub mod driver;
pub mod prompts;
pub mod report;
pub mod score;
pub mod transcript;

pub use driver::{DESCRIPTOR, METADATA};
