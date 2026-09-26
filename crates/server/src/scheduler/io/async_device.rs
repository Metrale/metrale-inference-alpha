// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-25: [`AsyncDeviceIo`]: the device router that lets one plain decode step run
//! ahead of the host (`--scheduler-config async`). It wraps [`SyncDeviceIo`]
//! and passes it every call except the launches below, their awaits,
//! `ReserveKv` / `Rollback`, and `teardown` (which first frees the ring and
//! its events). What it adds is
//!
//! * an asynchronous readback: the masked feed argmax writes the ids into a
//!   slot of a `RING_SLOTS`-slot page-locked ring with an async D2H and
//!   records that slot's event; [`DeviceIo::await_result`] polls the event
//!   (`POLL_SPINS` times, then a blocking synchronize) and only then reads
//!   the slot;
//! * the fed launch (`StepPlan::DecodeFed`), whose inputs the model resolves
//!   from the previous step's cells; the DFlash context commit sits between
//!   the forward and the readback, as in the synchronous step;
//! * the reservation ledger: blocks `ReserveKv` adds are attributed to the
//!   next launched ticket, and a `Rollback` of a row hands back that row's
//!   most recent reservation.
//!
//! Depth is a cap here (`max_depth() == 2`); whether a step may run ahead is
//! the core's decision, tick by tick.
//!
//! Owner: scheduler.
//! Invariants:
//! - A ring slot is given to a new readback only after `await_result` has
//!   released it; `take_slot` fails when all `RING_SLOTS` are unread.

use std::cell::{Cell, RefCell};
use std::collections::{HashMap, VecDeque};
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};

use metrale_gpu_runtime::gpu::DevicePtr;
use metrale_model_engine::traits::{Model, SequenceState};
use metrale_scheduler::{
    DecodeRows, DeviceError, DeviceFacts, DeviceIo, EffectOutcome, Readback, Ticket,
};

use super::sync_device::SyncDeviceIo;
use super::{Effect, StepOutcome, StepPlan, StepResult};
use crate::scheduler::{LoraAck, LoraCommand};

/// 2026-09-25: Slots in the pinned readback ring: two in flight plus the one being read.
pub const RING_SLOTS: usize = 3;

/// 2026-09-25: Spins on `event_query` before falling back to the blocking wait.
const POLL_SPINS: usize = 2_000;

/// 2026-09-25: Counters the scheduler-equivalence benchmark reports as
/// diagnostics, outside its verdict: the blocks `ReserveKv` holds ahead of
/// a launch and how often the pipeline threw a step away.
#[derive(Debug, Default)]
pub struct AsyncRouterStats {
    /// 2026-09-25: Fed (`DecodeFed`) launches.
    pub ahead_launches: AtomicU64,
    /// 2026-09-25: Blocks reserved ahead of a launch, in total.
    pub reserved_blocks: AtomicU64,
    /// 2026-09-25: The most blocks the ledger held at a launch.
    pub reserved_blocks_max_inflight: AtomicU64,
    /// 2026-09-25: `Rollback` effects applied (one per discarded row).
    pub rollbacks: AtomicU64,
    /// 2026-09-25: Blocks a `Rollback` handed back.
    pub rollback_released_blocks: AtomicU64,
}

impl AsyncRouterStats {
    pub fn snapshot(&self) -> std::collections::BTreeMap<String, f64> {
        [
            ("ahead_launches", &self.ahead_launches),
            ("reserved_blocks", &self.reserved_blocks),
            (
                "reserved_blocks_max_inflight",
                &self.reserved_blocks_max_inflight,
            ),
            ("rollbacks", &self.rollbacks),
            ("rollback_released_blocks", &self.rollback_released_blocks),
        ]
        .into_iter()
        .map(|(k, v)| (k.to_string(), v.load(Ordering::Relaxed) as f64))
        .collect()
    }
}

/// 2026-09-25: The process's counters, read by the bench host
/// (`tui/bench_host.rs`).
pub static STATS: AsyncRouterStats = AsyncRouterStats {
    ahead_launches: AtomicU64::new(0),
    reserved_blocks: AtomicU64::new(0),
    reserved_blocks_max_inflight: AtomicU64::new(0),
    rollbacks: AtomicU64::new(0),
    rollback_released_blocks: AtomicU64::new(0),
};

/// 2026-09-25: The blocks reserved ahead of one ticket: `(slot_idx, blocks)` per row,
/// and whether the ticket has been awaited.
struct LedgerEntry {
    ticket: Ticket,
    awaited: bool,
    rows: Vec<(usize, usize)>,
}

struct InFlight {
    slot: usize,
    rows: usize,
    logits: DevicePtr,
    /// 2026-09-25: A fed launch; the await calls `fed_step_settled`.
    fed: bool,
}

pub struct AsyncDeviceIo {
    inner: SyncDeviceIo,
    max_rows: usize,
    ring: *mut u8,
    ring_bytes: usize,
    events: [u64; RING_SLOTS],
    /// 2026-09-25: Slots whose result has not been read yet, oldest first.
    busy: RefCell<VecDeque<usize>>,
    in_flight: RefCell<HashMap<u32, InFlight>>,
    next_ticket: Cell<u32>,
    /// 2026-09-25: Blocks reserved since the last launch, attributed to the next ticket.
    pending_reserve: RefCell<Vec<(usize, usize)>>,
    /// 2026-09-25: The reservations of each launched ticket, in launch order.
    ledger: RefCell<VecDeque<LedgerEntry>>,
}

// 2026-09-25: SAFETY: the ring is page-locked host memory this router owns
// alone, and the router is not `Sync` (its `RefCell`s), so one thread at a
// time uses it.
unsafe impl Send for AsyncDeviceIo {}

impl AsyncDeviceIo {
    /// 2026-09-25: Fails when the model has no device token feed (serve then
    /// falls back to the synchronous router), or when the ring or an event
    /// cannot be allocated.
    pub fn new(model: Arc<dyn Model>, max_rows: usize) -> anyhow::Result<Self> {
        if !model.supports_device_token_feed() {
            anyhow::bail!("the model has no device token feed");
        }
        let max_rows = max_rows.max(1);
        let ring_bytes = RING_SLOTS * max_rows * 4;
        let ring = model.alloc_host_pinned(ring_bytes)?;
        let mut events = [0u64; RING_SLOTS];
        for e in events.iter_mut() {
            *e = model.create_event()?;
        }
        Ok(Self {
            inner: SyncDeviceIo::new(model),
            max_rows,
            ring,
            ring_bytes,
            events,
            busy: RefCell::new(VecDeque::new()),
            in_flight: RefCell::new(HashMap::new()),
            next_ticket: Cell::new(0),
            pending_reserve: RefCell::new(Vec::new()),
            ledger: RefCell::new(VecDeque::new()),
        })
    }

    fn slot_ptr(&self, slot: usize) -> *mut u32 {
        // 2026-09-25: SAFETY: `slot < RING_SLOTS`, inside the ring allocation.
        unsafe { self.ring.add(slot * self.max_rows * 4) as *mut u32 }
    }

    fn take_slot(&self) -> Result<usize, DeviceError> {
        let busy = self.busy.borrow();
        let slot = (0..RING_SLOTS).find(|s| !busy.contains(s)).ok_or_else(|| {
            DeviceError::Forward(anyhow::anyhow!(
                "readback ring exhausted: {RING_SLOTS} steps unread"
            ))
        })?;
        drop(busy);
        self.busy.borrow_mut().push_back(slot);
        Ok(slot)
    }

    fn issue(
        &self,
        logits: DevicePtr,
        masks: &[metrale_scheduler::RowMask],
        fed: bool,
    ) -> Result<Ticket, DeviceError> {
        let n = masks.len();
        if n > self.max_rows {
            return Err(DeviceError::Readback(anyhow::anyhow!(
                "{n} rows exceed the readback ring width {}",
                self.max_rows
            )));
        }
        let slot = self.take_slot()?;
        self.inner
            .model()
            .argmax_batch_to_feed(logits, masks, self.slot_ptr(slot), self.events[slot], 0)
            .map_err(DeviceError::Readback)?;
        let ticket = Ticket(self.next_ticket.get());
        self.next_ticket.set(ticket.0.wrapping_add(1));
        self.in_flight.borrow_mut().insert(
            ticket.0,
            InFlight {
                slot,
                rows: n,
                logits,
                fed,
            },
        );
        self.attribute_reservations(ticket);
        Ok(ticket)
    }

    /// 2026-09-25: The reservations made since the last launch belong to
    /// `ticket`; the entries of tickets already awaited are dropped.
    fn attribute_reservations(&self, ticket: Ticket) {
        let mut ledger = self.ledger.borrow_mut();
        ledger.retain(|e| !e.awaited);
        let rows = std::mem::take(&mut *self.pending_reserve.borrow_mut());
        ledger.push_back(LedgerEntry {
            ticket,
            awaited: false,
            rows,
        });
        let held: usize = ledger
            .iter()
            .flat_map(|e| e.rows.iter().map(|(_, b)| b))
            .sum();
        STATS
            .reserved_blocks_max_inflight
            .fetch_max(held as u64, Ordering::Relaxed);
    }

    fn wait(&self, event: u64) -> Result<(), DeviceError> {
        let m = self.inner.model();
        for _ in 0..POLL_SPINS {
            if m.event_query(event).map_err(DeviceError::Readback)? {
                return Ok(());
            }
            std::hint::spin_loop();
        }
        m.event_synchronize(event).map_err(DeviceError::Readback)
    }
}

impl DeviceIo for AsyncDeviceIo {
    type Seq = SequenceState;
    type Ptr = DevicePtr;
    type Model = dyn Model + 'static;
    type LoraCmd = LoraCommand;
    type LoraAck = LoraAck;

    fn model(&self) -> &(dyn Model + 'static) {
        self.inner.model()
    }
    fn facts(&self) -> DeviceFacts {
        self.inner.facts()
    }
    fn max_depth(&self) -> usize {
        2
    }

    fn launch(
        &self,
        plan: StepPlan<'_>,
        rows: &mut [&mut SequenceState],
    ) -> Result<Ticket, DeviceError> {
        match plan {
            // 2026-09-25: The pipeline's first step: the synchronous forward
            // with the asynchronous readback in place of the blocking argmax.
            StepPlan::Decode {
                tokens,
                ctx,
                readback: Readback::ArgmaxMasked { masks },
            } => {
                let m = self.inner.model();
                let logits = m
                    .decode_batch(&tokens, rows, 0)
                    .map_err(DeviceError::Forward)?;
                super::sync_device::commit_decode_ctx(m, ctx, rows);
                self.issue(logits, &masks, false)
            }
            StepPlan::DecodeFed {
                sources,
                masks,
                ctx,
            } => {
                let m = self.inner.model();
                let logits = m
                    .decode_batch_fed(&sources, rows, 0)
                    .map_err(DeviceError::Forward)?;
                super::sync_device::commit_decode_ctx(m, ctx, rows);
                STATS.ahead_launches.fetch_add(1, Ordering::Relaxed);
                self.issue(logits, &masks, true)
            }
            // 2026-09-25: A decode with the plain argmax or the host-logits
            // readback is the synchronous step, settled at once.
            other => {
                let ticket = self.inner.launch(other, rows)?;
                // 2026-09-25: One ticket space: a settled step is found by number, and
                // the next issued number stays above every number in use.
                self.next_ticket
                    .set(self.next_ticket.get().max(ticket.0.wrapping_add(1)));
                Ok(ticket)
            }
        }
    }

    fn await_result(&self, ticket: Ticket) -> Result<StepResult, DeviceError> {
        let Some(step) = self.in_flight.borrow_mut().remove(&ticket.0) else {
            return self.inner.await_result(ticket);
        };
        let waited = self.wait(self.events[step.slot]);
        // 2026-09-25: The slot is consumed whatever the wait said, so a fault cannot
        // wedge the ring.
        self.busy.borrow_mut().retain(|s| *s != step.slot);
        if let Some(entry) = self
            .ledger
            .borrow_mut()
            .iter_mut()
            .find(|e| e.ticket == ticket)
        {
            entry.awaited = true;
        }
        if step.fed {
            self.inner.model().fed_step_settled();
        }
        waited?;
        // 2026-09-25: SAFETY: the event says the D2H into this slot landed; `rows` cells
        // were written and nothing rewrites the slot until it is taken again.
        let ids =
            unsafe { std::slice::from_raw_parts(self.slot_ptr(step.slot), step.rows) }.to_vec();
        Ok(StepResult {
            ticket,
            outcome: StepOutcome::Decode {
                logits: step.logits,
                rows: DecodeRows::Tokens(ids),
            },
        })
    }

    fn apply(&self, effect: Effect<'_>) -> Result<EffectOutcome, DeviceError> {
        match effect {
            Effect::ReserveKv { seq } => {
                let slot_idx = seq.slot_idx;
                let out = self.inner.apply(Effect::ReserveKv { seq })?;
                if let EffectOutcome::Reserved { blocks } = out {
                    STATS
                        .reserved_blocks
                        .fetch_add(blocks as u64, Ordering::Relaxed);
                    self.pending_reserve.borrow_mut().push((slot_idx, blocks));
                }
                Ok(out)
            }
            Effect::Rollback { seq } => {
                // 2026-09-25: The most recent reservation for this row is the over-run's.
                let blocks = {
                    let mut ledger = self.ledger.borrow_mut();
                    ledger
                        .iter_mut()
                        .rev()
                        .find_map(|e| {
                            e.rows
                                .iter()
                                .position(|(s, _)| *s == seq.slot_idx)
                                .map(|i| e.rows.remove(i).1)
                        })
                        .unwrap_or(0)
                };
                STATS.rollbacks.fetch_add(1, Ordering::Relaxed);
                STATS
                    .rollback_released_blocks
                    .fetch_add(blocks as u64, Ordering::Relaxed);
                self.inner
                    .model()
                    .release_decode_blocks(seq, blocks)
                    .map_err(DeviceError::Effect)?;
                Ok(EffectOutcome::Unit)
            }
            other => self.inner.apply(other),
        }
    }

    fn lora(&mut self, cmd: LoraCommand) -> Result<LoraAck, String> {
        self.inner.lora(cmd)
    }

    fn teardown(self: Box<Self>) -> anyhow::Result<()> {
        let Self {
            inner,
            ring,
            ring_bytes,
            events,
            ..
        } = *self;
        {
            let m = inner.model();
            for e in events {
                m.destroy_event(e)?;
            }
            m.free_host_pinned(ring, ring_bytes)?;
        }
        Box::new(inner).teardown()
    }
}
