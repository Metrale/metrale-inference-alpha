// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-26: Compiles `src/nccl_backend/recv_buffer.rs` into this test
//! binary, so its unit tests run also when the library is built without the
//! `nccl` feature, the only feature under which `lib.rs` declares
//! `nccl_backend`.
//!
//! Owner: metrale-comm.
//! Invariants: none beyond the types.

#[path = "../src/nccl_backend/recv_buffer.rs"]
mod recv_buffer;
