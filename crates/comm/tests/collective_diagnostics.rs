// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-26: Compiles `src/collective_diagnostics.rs` as its own module, so
//! its unit tests also run without the `nccl` feature, where lib.rs leaves the
//! module out.
//!
//! Owner: metrale-comm.
//! Invariants: none beyond the types.
#[path = "../src/collective_diagnostics.rs"]
mod collective_diagnostics;
