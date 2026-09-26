// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-25: [`SyncDeviceIo`]: the synchronous device router
//! (`--scheduler-config sync`, the `async` fallback, and the inner router of
//! `AsyncDeviceIo`). Every launch runs the model call and its readback to
//! completion before returning. Depth is 1.
//!
//! Owner: scheduler.
//! Invariants:
//! - At most one settled, unclaimed result is held; a launch replaces it.

use std::cell::{Cell, RefCell};
use std::sync::Arc;

use metrale_gpu_runtime::gpu::DevicePtr;
use metrale_model_engine::traits::{Model, SequenceState};

use metrale_scheduler::{
    DecodeCtxCommit, DecodeRows, DeviceError, DeviceFacts, DeviceIo, EffectOutcome, Readback,
    Ticket,
};

use super::{Effect, StepOutcome, StepPlan, StepResult};
use crate::scheduler::{LoraAck, LoraCommand};

/// 2026-09-25: The `ep_broadcast_cmd_for_seq` word that frees and re-allocates a
/// worker's mirrored sequence in that slot.
pub const EP_FREE_REALLOC: u32 = 0xFFFFFFF1;
/// 2026-09-25: The word that stops the worker (`seq_id` ignored).
pub const EP_SHUTDOWN: u32 = 0xFFFFFFFF;

pub struct SyncDeviceIo {
    model: Arc<dyn Model>,
    next_ticket: Cell<u32>,
    /// 2026-09-25: The one settled-but-unclaimed result (depth 1).
    pending: RefCell<Option<StepResult>>,
}

impl SyncDeviceIo {
    pub fn new(model: Arc<dyn Model>) -> Self {
        Self {
            model,
            next_ticket: Cell::new(0),
            pending: RefCell::new(None),
        }
    }
}

/// 2026-09-25: Copy the `rows`-row logits block at `logits` into `into`
/// (resized to `rows * vocab * elem_bytes`). Returns the element width.
/// Used by [`execute_readback`] and the `ReadLogits` effect.
pub(crate) fn read_logits_to_host(
    model: &dyn Model,
    logits: DevicePtr,
    rows: usize,
    into: &mut Vec<u8>,
) -> anyhow::Result<usize> {
    let vocab_size = model.vocab_size();
    // 2026-09-25: 4 bytes per element when the model's decode lm_head
    // writes FP32 logits (`decode_logits_fp32`), else 2.
    let elem_bytes = if model.decode_logits_fp32() { 4 } else { 2 };
    into.resize(rows * vocab_size * elem_bytes, 0);
    model.copy_logits_to_host(logits, into)?;
    Ok(elem_bytes)
}

/// 2026-09-25: The DFlash context commit a decode step performs between its
/// forward and its readback; both routers call it at that point. Errors are
/// logged, not propagated.
pub(crate) fn commit_decode_ctx(
    m: &dyn Model,
    ctx: DecodeCtxCommit,
    rows: &mut [&mut SequenceState],
) {
    match ctx {
        DecodeCtxCommit::None => {}
        DecodeCtxCommit::Unified => {
            for (i, seq) in rows.iter_mut().enumerate() {
                // 2026-09-25: The serial token sits at position
                // seq_len - 1: the decode already advanced seq_len past it.
                let base_pos = seq.seq_len.saturating_sub(1);
                if let Err(e) = m.commit_ctx(seq, 1, base_pos, i) {
                    tracing::error!("commit_ctx (decode_only, row {i}): {e:#}");
                }
            }
        }
        DecodeCtxCommit::SerialAppend => {
            if let Some(seq) = rows.first_mut()
                && let Err(e) = m.dflash_serial_ctx_append(seq)
            {
                tracing::error!("dflash_serial_ctx_append (decode_only): {e:#}");
            }
        }
    }
}

/// 2026-09-25: The readback of a decode block: the device argmax per row, or the whole
/// block copied to the host.
pub(crate) fn execute_readback(
    model: &dyn Model,
    logits: DevicePtr,
    rows: usize,
    readback: Readback<'_>,
) -> anyhow::Result<DecodeRows> {
    Ok(match readback {
        Readback::Argmax => DecodeRows::Tokens(model.argmax_batch(logits, rows, 0)?),
        Readback::HostLogits { into } => DecodeRows::HostLogits {
            elem_bytes: read_logits_to_host(model, logits, rows, into)?,
        },
        // 2026-09-25: The masked feed argmax exists for a router that runs ahead of the
        // host; the core plans it only when the router's depth allows that.
        Readback::ArgmaxMasked { .. } => {
            anyhow::bail!("the synchronous router has no masked feed argmax")
        }
    })
}

impl DeviceIo for SyncDeviceIo {
    type Seq = SequenceState;
    type Ptr = DevicePtr;
    type Model = dyn Model + 'static;
    type LoraCmd = LoraCommand;
    type LoraAck = LoraAck;

    fn model(&self) -> &(dyn Model + 'static) {
        &*self.model
    }

    fn facts(&self) -> DeviceFacts {
        let m = &*self.model;
        DeviceFacts {
            free_blocks: m.num_free_blocks(),
            total_blocks: m.num_total_blocks(),
            ssm_occupancy: m.ssm_snapshot_occupancy(),
        }
    }

    fn max_depth(&self) -> usize {
        1
    }

    fn launch(
        &self,
        plan: StepPlan<'_>,
        rows: &mut [&mut SequenceState],
    ) -> Result<Ticket, DeviceError> {
        let m = &*self.model;
        let outcome = match plan {
            StepPlan::Decode {
                tokens,
                ctx,
                readback,
            } => {
                let logits = m
                    .decode_batch(&tokens, rows, 0)
                    .map_err(DeviceError::Forward)?;
                commit_decode_ctx(m, ctx, rows);
                let rows = execute_readback(m, logits, tokens.len(), readback)
                    .map_err(DeviceError::Readback)?;
                StepOutcome::Decode { logits, rows }
            }
            StepPlan::DecodeFed { .. } => {
                return Err(DeviceError::Forward(anyhow::anyhow!(
                    "the synchronous router cannot launch a fed decode step"
                )));
            }
        };
        let ticket = Ticket(self.next_ticket.get());
        self.next_ticket.set(ticket.0.wrapping_add(1));
        *self.pending.borrow_mut() = Some(StepResult { ticket, outcome });
        Ok(ticket)
    }

    fn await_result(&self, ticket: Ticket) -> Result<StepResult, DeviceError> {
        match self.pending.borrow_mut().take() {
            Some(r) if r.ticket == ticket => Ok(r),
            Some(r) => Err(DeviceError::Effect(anyhow::anyhow!(
                "await_result({ticket:?}): the settled step is {:?}",
                r.ticket
            ))),
            None => Err(DeviceError::Effect(anyhow::anyhow!(
                "await_result({ticket:?}): no step in flight"
            ))),
        }
    }

    fn apply(&self, effect: Effect<'_>) -> Result<EffectOutcome, DeviceError> {
        let m = &*self.model;
        match effect {
            Effect::ReleaseSeq { seq, cache, what } => {
                if cache {
                    m.cache_sequence(seq);
                }
                let slot = seq.slot_idx as u32;
                if let Err(e) = m.free_sequence(seq) {
                    tracing::error!("{what}: free_sequence: {e:#}");
                }
                if let Err(e) = m.ep_broadcast_cmd_for_seq(slot, EP_FREE_REALLOC) {
                    tracing::error!("{what}: EP broadcast free+realloc: {e:#}");
                }
            }
            Effect::CompactSlot { seq, target } => {
                m.compact_sequence(seq, target)
                    .map_err(DeviceError::Effect)?;
            }
            Effect::DetachSlot { seq } => m.detach_slot_for_reuse(seq),
            Effect::SaveSequenceState { seq, writer } => {
                m.save_sequence_state(seq, writer)
                    .map_err(DeviceError::Effect)?;
            }
            Effect::ReadLogits { logits, rows, into } => {
                let elem_bytes =
                    read_logits_to_host(m, logits, rows, into).map_err(DeviceError::Readback)?;
                return Ok(EffectOutcome::HostLogits { elem_bytes });
            }
            Effect::MarconiCheckpoint { seq } => m.decode_marconi_checkpoint(seq),
            Effect::SsmSnapshotSave { seq, slot } => m
                .save_decode_ssm_snapshot(seq, slot)
                .map_err(DeviceError::Effect)?,
            Effect::SsmSnapshotRestore { seq, slot } => m
                .restore_decode_ssm_snapshot(seq, slot)
                .map_err(DeviceError::Effect)?,
            Effect::SaveHiddenCatchup { row, pos } => m
                .save_hidden_for_catchup(row, pos)
                .map_err(DeviceError::Effect)?,
            Effect::SyncSecondary => m.sync_secondary().map_err(DeviceError::Effect)?,
            Effect::EpShutdown => {
                let _ = m.ep_broadcast_cmd_for_seq(0, EP_SHUTDOWN);
            }
            Effect::Quiesce { stream } => m.synchronize(stream).map_err(DeviceError::Effect)?,
            // 2026-09-25: The block for the next position, allocated ahead of a launch
            // that will run before the host sees the previous step's tokens.
            // A dry pool is an outcome, not a fault: the core simply does not
            // launch ahead.
            Effect::ReserveKv { seq } => {
                return Ok(match m.reserve_decode_block(seq) {
                    Ok(blocks) => EffectOutcome::Reserved { blocks },
                    Err(_) => EffectOutcome::Exhausted,
                });
            }
            // 2026-09-25: Only a router that reserved ahead holds the ledger a rollback
            // reconciles against (`AsyncDeviceIo`).
            Effect::Rollback { .. } => {
                return Err(DeviceError::Effect(anyhow::anyhow!(
                    "the synchronous router keeps no reservation ledger to roll back"
                )));
            }
        }
        Ok(EffectOutcome::Unit)
    }

    fn lora(&mut self, cmd: LoraCommand) -> Result<LoraAck, String> {
        let Some(m) = Arc::get_mut(&mut self.model) else {
            return Err("LoRA command refused: the model is still shared".into());
        };
        match cmd {
            LoraCommand::Rotate(name) => {
                let r = m
                    .set_active_lora(&name)
                    .map(|()| LoraAck::Done)
                    .map_err(|e| format!("{e:#}"));
                if let Err(ref e) = r {
                    tracing::warn!("LoRA rotation to '{name}' failed: {e}");
                }
                r
            }
            LoraCommand::LoadIntoSlot { name, dir, slot } => {
                let r = m
                    .swap_lora_from_disk(&dir, &name, slot)
                    .map(|()| LoraAck::Done)
                    .map_err(|e| format!("{e:#}"));
                if let Err(ref e) = r {
                    tracing::warn!("LoRA disk swap '{name}' -> slot {slot} failed: {e}");
                }
                r
            }
            LoraCommand::Promote {
                peer_addr,
                adapter_id,
                name,
                peft,
            } => {
                let r = m
                    .promote_lora_from_peer(&peer_addr, &adapter_id, &name, peft)
                    .map(|(slot, evicted)| LoraAck::Promoted { slot, evicted })
                    .map_err(|e| format!("{e:#}"));
                if let Err(ref e) = r {
                    tracing::warn!("LoRA promote '{name}' failed: {e}");
                }
                r
            }
            LoraCommand::PromoteDisk { name, dir } => {
                let r = m
                    .promote_lora_from_disk(&dir, &name)
                    .map(|(slot, evicted)| LoraAck::Promoted { slot, evicted })
                    .map_err(|e| format!("{e:#}"));
                if let Err(ref e) = r {
                    tracing::warn!("LoRA disk-promote '{name}' failed: {e}");
                }
                r
            }
        }
    }

    fn teardown(mut self: Box<Self>) -> anyhow::Result<()> {
        match Arc::get_mut(&mut self.model) {
            Some(m) => m.teardown(),
            None => anyhow::bail!("teardown: the model is still shared"),
        }
    }
}
