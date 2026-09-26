// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-26: The `vision-fidelity` benchmark: does the served model see the
//! image it was sent, at the resolution the serve permits?
//!
//! - Geometry: the exact vision-token count of each ladder fixture
//!   (`provision::FIXTURES`), read from `usage.prompt_tokens` minus a
//!   calibrated template overhead and compared with `geometry.rs`. A
//!   preprocessing change that still leaves a recognisable picture moves this
//!   count, where a capability probe can still pass.
//! - Capability: probes on the fixtures (`probes.rs`), with a no-image control;
//!   if the control answers, the run is VACUOUS.
//! - Integrity and concurrency legs (`driver.rs`).
//!
//! `tests/vision_sweep.py` (a keyword rubric on one photo) and
//! `tests/vit_reference_check.py` (a layer-by-layer diff against an HF
//! reference) are separate harnesses and assert no vision-token counts.
//!
//! Registered in `registry.rs`; model targets gate on it through
//! `gate = "vision-fidelity"` entries in their BENCH.toml.
//!
//! Owner: bench, vision.
//! Invariants: none beyond the types.

pub mod driver;
pub mod geometry;
pub mod probes;
pub mod provision;
pub mod request;
pub mod score;

pub use driver::{DESCRIPTOR, METADATA};
