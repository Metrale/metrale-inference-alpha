// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-25: The device a test wires when it exercises nothing that
//! reaches one.
//!
//! Owner: scheduler.
//! Invariants:
//! - Every method except `max_depth` and `teardown` panics.

use metrale_model_engine::traits::{Model, SequenceState};
use metrale_scheduler::{DeviceError, DeviceFacts, DeviceIo, EffectOutcome, Label, Ticket};

use super::{Effect, StepPlan, StepResult};
use crate::scheduler::{LoraAck, LoraCommand};

pub struct NoDeviceIo;

impl DeviceIo for NoDeviceIo {
    type Seq = SequenceState;
    type Ptr = metrale_gpu_runtime::gpu::DevicePtr;
    type Model = dyn Model + 'static;
    type LoraCmd = LoraCommand;
    type LoraAck = LoraAck;

    fn model(&self) -> &(dyn Model + 'static) {
        panic!("this test wired no device (SchedIo::for_test); use for_test_with(model)")
    }
    fn facts(&self) -> DeviceFacts {
        panic!("this test wired no device: facts")
    }
    fn max_depth(&self) -> usize {
        1
    }
    fn launch(
        &self,
        plan: StepPlan<'_>,
        _rows: &mut [&mut SequenceState],
    ) -> Result<Ticket, DeviceError> {
        panic!("this test wired no device: launch {}", plan.label())
    }
    fn await_result(&self, ticket: Ticket) -> Result<StepResult, DeviceError> {
        panic!("this test wired no device: await {ticket:?}")
    }
    fn apply(&self, effect: Effect<'_>) -> Result<EffectOutcome, DeviceError> {
        panic!("this test wired no device: apply {}", effect.label())
    }
    fn lora(&mut self, _cmd: LoraCommand) -> Result<LoraAck, String> {
        panic!("this test wired no device: lora")
    }
    fn teardown(self: Box<Self>) -> anyhow::Result<()> {
        Ok(())
    }
}
