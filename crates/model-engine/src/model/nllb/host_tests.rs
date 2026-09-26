// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-25: Non-CUDA test harness for NLLB's host-only language and position helpers.
//!
//! Owner: model-engine.
//! Invariants: none beyond the types.

#[path = "lang.rs"]
mod lang;
pub use lang::NllbLang;

#[path = "util.rs"]
mod util;
