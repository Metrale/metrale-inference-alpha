// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-26: Compiles `src/collective_wait.rs` into this test binary, so its
//! unit tests run also when the library is built without the `nccl` feature,
//! the only feature under which `lib.rs` declares the module.
//!
//! Owner: metrale-comm.
//! Invariants: none beyond the types.

#[path = "../src/collective_wait.rs"]
mod collective_wait;
