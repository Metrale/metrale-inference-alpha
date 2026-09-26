// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-25: [`TracingDeviceIo`], a decorator that writes one line per launch,
//! effect, LoRA command and teardown into a sink, then forwards the call to
//! the inner [`DeviceIo`].
//!
//! Owner: scheduler.
//! Invariants: none beyond the types.

use crate::device::DeviceIo;
use crate::effect::{Effect, EffectOutcome};
use crate::plan::{Label, StepPlan, StepResult};
use crate::types::{DeviceError, DeviceFacts, Ticket};

pub type TraceSink = std::sync::Arc<dyn Fn(String) + Send + Sync>;

pub struct TracingDeviceIo<D: DeviceIo> {
    inner: D,
    sink: TraceSink,
}

impl<D: DeviceIo> TracingDeviceIo<D> {
    pub fn new(inner: D, sink: TraceSink) -> Self {
        Self { inner, sink }
    }
}

impl<D: DeviceIo> DeviceIo for TracingDeviceIo<D>
where
    D::LoraCmd: Label,
{
    type Seq = D::Seq;
    type Ptr = D::Ptr;
    type Model = D::Model;
    type LoraCmd = D::LoraCmd;
    type LoraAck = D::LoraAck;

    fn model(&self) -> &D::Model {
        self.inner.model()
    }
    fn facts(&self) -> DeviceFacts {
        self.inner.facts()
    }
    fn max_depth(&self) -> usize {
        self.inner.max_depth()
    }
    fn launch(&self, plan: StepPlan<'_>, rows: &mut [&mut D::Seq]) -> Result<Ticket, DeviceError> {
        (self.sink)(format!("fx launch {}", plan.label()));
        self.inner.launch(plan, rows)
    }
    fn await_result(&self, ticket: Ticket) -> Result<StepResult<D::Ptr>, DeviceError> {
        self.inner.await_result(ticket)
    }
    fn apply(&self, effect: Effect<'_, D::Seq, D::Ptr>) -> Result<EffectOutcome, DeviceError> {
        (self.sink)(format!("fx apply {}", effect.label()));
        self.inner.apply(effect)
    }
    fn lora(&mut self, cmd: D::LoraCmd) -> Result<D::LoraAck, String> {
        (self.sink)(format!("fx lora {}", cmd.label()));
        self.inner.lora(cmd)
    }
    fn teardown(self: Box<Self>) -> anyhow::Result<()> {
        (self.sink)("fx teardown".into());
        Box::new(self.inner).teardown()
    }
}
