// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-25: [`ScriptedDeviceIo`], a test router wrapped around an inner router.
//! It answers the decode lane's readback from a script (per sequence, in
//! order), fails a launch on cue, reports a configurable depth, and records
//! every effect and LoRA command, while the inner router still performs the
//! device bookkeeping the rows need.
//!
//! A row is matched to its script line by the key function the test passes
//! (the server's trace-harness tests key by session hash).
//!
//! Depth: a launch whose readback is a device argmax (plain, masked or a
//! fed step) is forwarded and left in flight with the inner router; the
//! script is substituted when the ticket is awaited, so an inner router
//! that pipelines keeps pipelining under the script. A host-logits launch
//! is settled at once, because its answer is written into the plan's own
//! buffer, which only exists for the duration of the call.
//!
//! Owner: scheduler.
//! Invariants: none beyond the types.

use std::cell::{Cell, RefCell};
use std::collections::{HashMap, VecDeque};

use crate::device::DeviceIo;
use crate::effect::{Effect, EffectOutcome};
use crate::plan::{DecodeRows, Label, Readback, StepOutcome, StepPlan, StepResult};
use crate::types::{DeviceError, DeviceFacts, Ticket};

/// 2026-09-25: A scripted launch failure.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Fault {
    /// 2026-09-25: A forward error that reads as KV exhaustion
    /// ([`crate::DeviceError::is_kv_exhausted`]).
    KvExhausted,
    /// 2026-09-25: Any other forward error.
    Fatal,
}

/// 2026-09-25: A BF16 one-hot logits row: 30.0 (`0x41F0`) at `hit`, 0.0 elsewhere.
fn one_hot_bf16(dst: &mut [u8], hit: u32) {
    dst.fill(0);
    let i = hit as usize * 2;
    if i + 1 < dst.len() {
        dst[i] = 0xF0;
        dst[i + 1] = 0x41;
    }
}

/// 2026-09-25: How a sequence row is matched to its script line.
pub type KeyFn<S> = Box<dyn Fn(&S) -> u64 + Send>;

/// 2026-09-25: A forwarded launch whose script substitution waits for `await_result`.
struct InFlight {
    ticket: Ticket,
    keys: Vec<u64>,
    /// 2026-09-25: The inputs the plan carried (a plain step) — the fallback when a row
    /// has no script line and the inner answered host logits.
    fallback: Vec<u32>,
}

pub struct ScriptedDeviceIo<D: DeviceIo> {
    inner: D,
    key: KeyFn<D::Seq>,
    vocab: usize,
    max_depth: usize,
    script: RefCell<HashMap<u64, VecDeque<u32>>>,
    /// 2026-09-25: Faults to raise on the next launches, consumed in order.
    faults: RefCell<VecDeque<Fault>>,
    /// 2026-09-25: The tokens the last settled step answered, per row, for `ReadLogits`.
    last_rows: RefCell<Vec<u32>>,
    launches: Cell<usize>,
    effects: RefCell<Vec<String>>,
    /// 2026-09-25: Host-logits steps, already settled and substituted at launch.
    settled: RefCell<VecDeque<StepResult<D::Ptr>>>,
    /// 2026-09-25: Device-argmax steps still with the inner router, in launch order.
    in_flight: RefCell<VecDeque<InFlight>>,
}

impl<D: DeviceIo> ScriptedDeviceIo<D> {
    /// 2026-09-25: `key` matches a row to its script; `vocab` sizes the host-logits rows;
    /// `max_depth` is what the router reports (never more than the inner
    /// router's own).
    pub fn new(
        inner: D,
        key: impl Fn(&D::Seq) -> u64 + Send + 'static,
        vocab: usize,
        max_depth: usize,
    ) -> Self {
        Self {
            inner,
            key: Box::new(key),
            vocab,
            max_depth,
            script: RefCell::new(HashMap::new()),
            faults: RefCell::new(VecDeque::new()),
            last_rows: RefCell::new(Vec::new()),
            launches: Cell::new(0),
            effects: RefCell::new(Vec::new()),
            settled: RefCell::new(VecDeque::new()),
            in_flight: RefCell::new(VecDeque::new()),
        }
    }

    /// 2026-09-25: The tokens the decode lane will read back for the row keyed `key`,
    /// in order. A row with no line left answers what the inner router
    /// answered.
    pub fn script(&self, key: u64, tokens: impl IntoIterator<Item = u32>) {
        self.script
            .borrow_mut()
            .insert(key, tokens.into_iter().collect());
    }

    /// 2026-09-25: Raise `fault` on the next launch (queued, in order).
    pub fn fail_next(&self, fault: Fault) {
        self.faults.borrow_mut().push_back(fault);
    }

    /// 2026-09-25: Every effect and LoRA command applied so far, by label, in order.
    pub fn effects(&self) -> Vec<String> {
        self.effects.borrow().clone()
    }

    pub fn launches(&self) -> usize {
        self.launches.get()
    }

    pub fn inner(&self) -> &D {
        &self.inner
    }

    fn take_line(&self, key: u64) -> Option<u32> {
        self.script
            .borrow_mut()
            .get_mut(&key)
            .and_then(VecDeque::pop_front)
    }

    fn next_for(&self, key: u64, fallback: u32) -> u32 {
        self.take_line(key).unwrap_or(fallback)
    }

    fn one_hot_rows(&self, tokens: &[u32], into: &mut Vec<u8>) {
        into.resize(tokens.len() * self.vocab * 2, 0);
        for (i, &tok) in tokens.iter().enumerate() {
            one_hot_bf16(&mut into[i * self.vocab * 2..(i + 1) * self.vocab * 2], tok);
        }
    }

    /// 2026-09-25: Substitute the script into the rows the inner router answered.
    fn substitute(&self, keys: &[u64], fallback: &[u32], model_rows: &DecodeRows) -> Vec<u32> {
        let inner_rows: Vec<u32> = match model_rows {
            DecodeRows::Tokens(t) => t.clone(),
            DecodeRows::HostLogits { .. } => fallback.to_vec(),
        };
        let answered: Vec<u32> = keys
            .iter()
            .zip(&inner_rows)
            .map(|(&k, &f)| self.next_for(k, f))
            .collect();
        *self.last_rows.borrow_mut() = answered.clone();
        answered
    }
}

impl<D: DeviceIo> DeviceIo for ScriptedDeviceIo<D>
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
        self.max_depth.min(self.inner.max_depth())
    }

    fn launch(&self, plan: StepPlan<'_>, rows: &mut [&mut D::Seq]) -> Result<Ticket, DeviceError> {
        self.launches.set(self.launches.get() + 1);
        if let Some(fault) = self.faults.borrow_mut().pop_front() {
            return Err(DeviceError::Forward(match fault {
                Fault::KvExhausted => anyhow::anyhow!("KV cache exhausted (scripted)"),
                Fault::Fatal => anyhow::anyhow!("scripted device fault"),
            }));
        }
        let keys: Vec<u64> = rows.iter().map(|r| (self.key)(r)).collect();
        match plan {
            StepPlan::Decode {
                tokens,
                ctx,
                readback: Readback::HostLogits { into },
            } => {
                // 2026-09-25: The inner router does the row bookkeeping and reads its
                // block into `scratch`, which becomes the plan's buffer; rows with a
                // script line are overwritten below.
                let mut scratch = Vec::new();
                let ticket = self.inner.launch(
                    StepPlan::Decode {
                        tokens: tokens.clone(),
                        ctx,
                        readback: Readback::HostLogits { into: &mut scratch },
                    },
                    rows,
                )?;
                let inner = self.inner.await_result(ticket)?;
                let StepOutcome::Decode { logits, .. } = inner.outcome;
                // 2026-09-25: A row with a script line answers one-hot at that token; a
                // row without one keeps the block the inner router read.
                *into = scratch;
                let row_bytes = self.vocab * 2;
                let mut answered = Vec::with_capacity(keys.len());
                for (i, &k) in keys.iter().enumerate() {
                    match self.take_line(k) {
                        Some(tok) => {
                            if into.len() >= (i + 1) * row_bytes {
                                one_hot_bf16(&mut into[i * row_bytes..(i + 1) * row_bytes], tok);
                            }
                            answered.push(tok);
                        }
                        None => answered.push(tokens[i]),
                    }
                }
                *self.last_rows.borrow_mut() = answered;
                self.settled.borrow_mut().push_back(StepResult {
                    ticket,
                    outcome: StepOutcome::Decode {
                        logits,
                        rows: DecodeRows::HostLogits { elem_bytes: 2 },
                    },
                });
                Ok(ticket)
            }
            StepPlan::Decode {
                tokens,
                ctx,
                readback,
            } => {
                let fallback = tokens.clone();
                let ticket = self.inner.launch(
                    StepPlan::Decode {
                        tokens,
                        ctx,
                        readback,
                    },
                    rows,
                )?;
                self.in_flight.borrow_mut().push_back(InFlight {
                    ticket,
                    keys,
                    fallback,
                });
                Ok(ticket)
            }
            StepPlan::DecodeFed {
                sources,
                masks,
                ctx,
            } => {
                let ticket = self.inner.launch(
                    StepPlan::DecodeFed {
                        sources,
                        masks,
                        ctx,
                    },
                    rows,
                )?;
                self.in_flight.borrow_mut().push_back(InFlight {
                    ticket,
                    keys,
                    fallback: Vec::new(),
                });
                Ok(ticket)
            }
        }
    }

    fn await_result(&self, ticket: Ticket) -> Result<StepResult<D::Ptr>, DeviceError> {
        let settled_at = self
            .settled
            .borrow()
            .iter()
            .position(|r| r.ticket == ticket);
        if let Some(pos) = settled_at {
            return Ok(self.settled.borrow_mut().remove(pos).expect("position"));
        }
        let pending = {
            let mut q = self.in_flight.borrow_mut();
            let pos = q.iter().position(|f| f.ticket == ticket).ok_or_else(|| {
                DeviceError::Effect(anyhow::anyhow!(
                    "await_result({ticket:?}): no scripted step in flight"
                ))
            })?;
            q.remove(pos).expect("position")
        };
        let inner = self.inner.await_result(ticket)?;
        let StepOutcome::Decode {
            logits,
            rows: model_rows,
        } = inner.outcome;
        let answered = self.substitute(&pending.keys, &pending.fallback, &model_rows);
        Ok(StepResult {
            ticket,
            outcome: StepOutcome::Decode {
                logits,
                rows: DecodeRows::Tokens(answered),
            },
        })
    }

    fn apply(&self, effect: Effect<'_, D::Seq, D::Ptr>) -> Result<EffectOutcome, DeviceError> {
        self.effects.borrow_mut().push(effect.label());
        match effect {
            Effect::ReadLogits { rows, into, .. } => {
                let answered: Vec<u32> =
                    self.last_rows.borrow().iter().take(rows).copied().collect();
                self.one_hot_rows(&answered, into);
                into.resize(rows * self.vocab * 2, 0);
                Ok(EffectOutcome::HostLogits { elem_bytes: 2 })
            }
            other => self.inner.apply(other),
        }
    }

    fn lora(&mut self, cmd: D::LoraCmd) -> Result<D::LoraAck, String> {
        self.effects
            .borrow_mut()
            .push(format!("lora {}", cmd.label()));
        self.inner.lora(cmd)
    }

    fn teardown(self: Box<Self>) -> anyhow::Result<()> {
        Box::new(self.inner).teardown()
    }
}
