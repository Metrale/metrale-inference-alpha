// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-26: SSM state poisoning gate: `driver` runs it, `probe` holds the
//! script, `compare` judges each replay round, `score` decides, `toolcall`
//! checks path-independence of tool calls, and `report` renders.
//!
//! Owner: bench, SSM poisoning gate.
//! Invariants: none beyond the types.

pub mod compare;
pub mod driver;
pub mod probe;
pub mod report;
pub mod score;
pub mod toolcall;

pub use driver::{DESCRIPTOR, METADATA};
