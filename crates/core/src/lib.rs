// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-25: Shared foundation types and host-side helpers for the engine crates, one module each (listed below).
//!
//! Owner: metrale-core.
//! Invariants:
//! - The `cuda` feature gates only `device::MetraleDevice` and the
//!   `error::MetraleError::CudaDriver` variant; the crate builds without it.

#![deny(warnings)]
#![deny(clippy::all)]

pub mod arch;
pub mod compute;
pub mod dtype;
pub mod error;
pub mod fault;
pub mod mxfp4_e8m0;
pub mod numeric;
pub mod safetensors;
pub mod scope;
pub mod target;
pub mod tensor;

// 2026-09-25: `device` is not feature-gated: its `sm121` constants compile on
// every backend, and only the `MetraleDevice` wrapper inside it needs `cuda`.
pub mod device;
