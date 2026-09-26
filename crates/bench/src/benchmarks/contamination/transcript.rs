// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-26: Re-export of `benchmarks::transcript` for the contamination
//! module's `super::transcript` imports. The `ssm_poison`,
//! `scheduler_equivalence` and `kat_equality` drivers use the same type, so
//! every driver shares one equality contract.
//!
//! Owner: bench (contamination).
//! Invariants: none beyond the types.

pub use crate::benchmarks::transcript::{RequestOutcome, Transcript};
