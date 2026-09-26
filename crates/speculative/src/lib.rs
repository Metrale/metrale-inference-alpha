// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-25: Metrale Engine speculative-decoding policy: the MTP gate, the
//! adaptive and DFlash rungs, the n-gram proposer, draft-capacity clamps,
//! per-run speculation counters and the scheduler snapshot.
//!
//! Owner: speculative.
//! Invariants: none beyond the types.

pub mod adaptive_rung;
pub mod dflash_rung;
pub mod mtp_gate;
pub mod ngram;
pub mod snapshot;
pub mod spec_capacity;
pub mod spec_stats;
