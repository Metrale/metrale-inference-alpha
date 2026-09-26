// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-25: The scheduler's I/O routers (SBIO): the loop's business logic
//! reaches the clock, the client channels, the spill files and the metrics
//! through these.
//!
//! The contract — the router traits, the step plans and effects, the
//! driver — is `metrale_scheduler`, which names no model type. This module
//! instantiates it for the server: the sequence state is
//! `SequenceState`, the logits handle `DevicePtr`, and the server-only
//! routers (`RequestIo`, `TelemetryIo`) live beside the instantiation.
//!
//! * [`ClockIo`] — the clock the loop's business logic reads.
//! * [`RequestIo`] — the inbox (request and LoRA-rotation arrivals) and every
//!   client-bound frame.
//! * [`TelemetryIo`] — timing marks, counters, the dashboard snapshot, the
//!   diagnostic dumps.
//! * [`SpillIo`] — the swap files a sequence's device state is spilled to.
//! * [`DeviceIo`] — the model: plans are launched and effects applied
//!   through it.
//!
//! Owner: scheduler.
//! Invariants:
//! - Outside the paths in `io::tests::SEAM_ALLOW`, no scheduler source line
//!   contains a `SEAM_FORBIDDEN` pattern, except the `SEAM_COMPOSITION`
//!   pairs; `no_scheduler_business_file_does_its_own_io` checks it.

pub mod async_device;
pub mod no_device;
pub mod request;
pub mod spill;
pub mod sync_device;
pub mod telemetry;
#[cfg(test)]
mod tests;

use metrale_gpu_runtime::gpu::DevicePtr;
use metrale_model_engine::traits::{Model, SequenceState};

pub use async_device::AsyncDeviceIo;
#[cfg(test)]
pub use metrale_scheduler::TracingDeviceIo;
pub use metrale_scheduler::{
    ClockIo, DecodeCtxCommit, DecodeRows, DeviceError, DeviceIo, EffectOutcome, Label, Readback,
    SpillIo, SystemClock, WaitPolicy,
};
pub use no_device::NoDeviceIo;
pub use request::{Arrivals, FinishFrame, RequestIo, TokioRequestIo};
pub use spill::FileSpill;
pub use sync_device::SyncDeviceIo;
pub use telemetry::{SysTelemetry, TelemetryIo};

use crate::scheduler::{LoraAck, LoraCommand};

/// 2026-09-25: The contract's plan/effect/result types over this server's
/// sequence state and logits handle.
pub type StepPlan<'a> = metrale_scheduler::StepPlan<'a>;
pub type StepOutcome = metrale_scheduler::StepOutcome<DevicePtr>;
pub type StepResult = metrale_scheduler::StepResult<DevicePtr>;
pub type Effect<'a> = metrale_scheduler::Effect<'a, SequenceState, DevicePtr>;
/// 2026-09-25: The device router as the server sees it.
pub type DynDevice = dyn DeviceIo<
        Seq = SequenceState,
        Ptr = DevicePtr,
        Model = dyn Model + 'static,
        LoraCmd = LoraCommand,
        LoraAck = LoraAck,
    >;

impl Label for LoraCommand {
    fn label(&self) -> String {
        match self {
            LoraCommand::Rotate(name) => format!("Rotate({name})"),
            LoraCommand::LoadIntoSlot { name, slot, .. } => format!("LoadIntoSlot({name}, {slot})"),
            LoraCommand::Promote { name, .. } => format!("Promote({name})"),
            LoraCommand::PromoteDisk { name, .. } => format!("PromoteDisk({name})"),
        }
    }
}

/// 2026-09-25: The routers one scheduler run talks to.
pub struct SchedIo {
    pub dev: Box<DynDevice>,
    pub clock: Box<dyn ClockIo>,
    pub tel: std::sync::Arc<dyn TelemetryIo>,
    pub req: std::sync::Arc<dyn RequestIo>,
    /// 2026-09-25: `None` when swap space is off or failed to initialise.
    pub spill: Option<Box<dyn SpillIo>>,
}

impl SchedIo {
    /// 2026-09-25: The serving routers: the system clock, telemetry opened
    /// from the environment, the tokio inbox and, when `spill` is given,
    /// the spill store.
    pub fn serving(
        dev: Box<DynDevice>,
        telemetry: &'static metrale_telemetry::Telemetry,
        snapshot: std::sync::Arc<metrale_speculative::snapshot::SnapshotCell>,
        req: std::sync::Arc<dyn RequestIo>,
        spill: Option<Box<dyn SpillIo>>,
    ) -> Self {
        Self {
            dev,
            clock: Box::new(SystemClock),
            tel: std::sync::Arc::new(SysTelemetry::from_env(telemetry, snapshot)),
            req,
            spill,
        }
    }

    /// 2026-09-25: Routers that read nothing from the environment and
    /// receive nothing: the system clock, quiet telemetry, a closed inbox,
    /// no spill store and no device (`NoDeviceIo`; wire one with
    /// `for_test_with`).
    pub fn for_test() -> Self {
        Self::for_test_with_device(Box::new(NoDeviceIo))
    }

    /// 2026-09-25: `for_test` over the sync router on `model`.
    pub fn for_test_with(model: std::sync::Arc<dyn Model>) -> Self {
        Self::for_test_with_device(Box::new(SyncDeviceIo::new(model)))
    }

    fn for_test_with_device(dev: Box<DynDevice>) -> Self {
        Self {
            dev,
            clock: Box::new(SystemClock),
            tel: std::sync::Arc::new(SysTelemetry::quiet(std::sync::Arc::new(
                metrale_speculative::snapshot::SnapshotCell::default(),
            ))),
            req: std::sync::Arc::new(TokioRequestIo::closed()),
            spill: None,
        }
    }
}
