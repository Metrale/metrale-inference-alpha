// SPDX-License-Identifier: MIT OR Apache-2.0
#![deny(warnings)]
#![deny(clippy::all)]

//! 2026-09-25: The scheduler's I/O contract (SBIO).
//!
//! The core (in the server crate) hands device work to a [`DeviceIo`] as
//! [`StepPlan`]s and [`Effect`]s, and can reach the model itself through
//! [`DeviceIo::model`]. The clock is a [`ClockIo`], swap files a [`SpillIo`].
//! [`driver::run`] is the scheduler loop.
//!
//! [`ScriptedDeviceIo`] wraps a router and answers the decode lane from a
//! script, so a test can drive the real core with chosen tokens.
//! [`TracingDeviceIo`] wraps a router and logs each call.
//!
//! Owner: scheduler.
//! Invariants:
//! - The sequence state, the device logits handle and the model are type
//!   parameters, never a concrete model type, anywhere in this crate.

pub mod clock;
pub mod device;
pub mod driver;
pub mod effect;
pub mod plan;
pub mod scripted;
pub mod spill;
pub mod trace_device;
pub mod types;

pub use clock::{ClockIo, SystemClock};
pub use device::DeviceIo;
pub use driver::{Core, LaneVerdict, TickPlan, run};
pub use effect::{Effect, EffectOutcome};
pub use plan::{
    DecodeCtxCommit, DecodeRows, FeedSource, Label, Readback, RowMask, StepOutcome, StepPlan,
    StepResult,
};
pub use scripted::{Fault, ScriptedDeviceIo};
pub use spill::SpillIo;
pub use trace_device::{TraceSink, TracingDeviceIo};
pub use types::{DeviceError, DeviceFacts, Ticket, WaitPolicy};
