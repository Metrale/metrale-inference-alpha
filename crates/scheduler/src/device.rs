// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-25: [`DeviceIo`], the model-facing router: the core launches
//! [`StepPlan`]s and applies [`Effect`]s, and the router decides how they
//! reach the device.
//!
//! Owner: scheduler.
//! Invariants: none beyond the types.

use crate::effect::{Effect, EffectOutcome};
use crate::plan::{StepPlan, StepResult};
use crate::types::{DeviceError, DeviceFacts, Ticket};

pub trait DeviceIo: Send {
    /// 2026-09-25: The per-sequence device bookkeeping the plans and effects mutate.
    type Seq;
    /// 2026-09-25: The device's logits handle.
    type Ptr: Copy + Send;
    /// 2026-09-25: The model behind this router, returned by [`Self::model`] for
    /// calls that go around plans and effects.
    type Model: ?Sized;
    /// 2026-09-25: A LoRA control command and its acknowledgement.
    type LoraCmd;
    type LoraAck;

    fn model(&self) -> &Self::Model;
    /// 2026-09-25: The KV pool and SSM snapshot counters ([`DeviceFacts`]).
    fn facts(&self) -> DeviceFacts;
    /// 2026-09-25: How many steps may be in flight (1 = every step settles before the
    /// next launches).
    fn max_depth(&self) -> usize;
    /// 2026-09-25: Start `plan` over `rows`; settle it with [`Self::await_result`].
    fn launch(
        &self,
        plan: StepPlan<'_>,
        rows: &mut [&mut Self::Seq],
    ) -> Result<Ticket, DeviceError>;
    fn await_result(&self, ticket: Ticket) -> Result<StepResult<Self::Ptr>, DeviceError>;
    fn apply(&self, effect: Effect<'_, Self::Seq, Self::Ptr>)
    -> Result<EffectOutcome, DeviceError>;
    /// 2026-09-25: A LoRA control command, applied at quiescence (the model mutates its
    /// adapter table, hence the exclusive borrow).
    fn lora(&mut self, cmd: Self::LoraCmd) -> Result<Self::LoraAck, String>;
    /// 2026-09-25: Release the model's device memory. Consumes the router.
    fn teardown(self: Box<Self>) -> anyhow::Result<()>;
}
