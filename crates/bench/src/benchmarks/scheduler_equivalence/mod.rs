// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-26: Scheduler equivalence: one BFCL draw served under the sync and
//! the async device router (`--scheduler-config`), compared per `sample_id`
//! and per concurrency.
//!
//! Owner: bench, scheduler-equivalence gate.
//! Invariants:
//! - A sample agrees only when its canonical transcript (reasoning, text, tool
//!   calls with raw arguments, finish reason) and its completion token count
//!   are equal under both routers; a failed request is unmeasured, never equal.
//! - Each serve pins its speculation lane (`host::Lane`), because the MTP gate
//!   otherwise picks verify or plain decode by measured throughput.
//! - With the control on, the sync serve is measured twice per lane, and a
//!   control that diverged fails the run before the async column is read.
//! - The async router's counters are recorded as `async_*` metrics and take no
//!   part in the verdict.
//! - Loading fails without an installed [`host::RouterHost`]; the serving
//!   process installs one as it starts.

pub mod compare;
pub mod driver;
pub mod host;
pub mod report;
pub mod unmeasured;

pub use driver::{DESCRIPTOR, METADATA};

#[cfg(test)]
#[path = "compare_tests.rs"]
mod compare_tests;

#[cfg(test)]
#[path = "driver_tests.rs"]
mod driver_tests;

#[cfg(test)]
#[path = "unmeasured_tests.rs"]
mod unmeasured_tests;
