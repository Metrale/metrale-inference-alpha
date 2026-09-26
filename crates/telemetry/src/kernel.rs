// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-26: L1: GPU time from CUDA event pairs: every serve lane on every
//! step, and every eager kernel launch on one step in N.
//!
//! [`SpanRing`] is the bookkeeping, generic over [`EventOps`] so it is tested
//! here with a scripted event source; the CUDA implementation is
//! `metrale-gpu-runtime`'s `timing` module. A span lands when a later
//! [`SpanRing::harvest`] polls it complete; the ring never waits on the GPU.
//! Kernels replayed inside a CUDA graph get no per-kernel span; only the lane
//! span covers a replay.
//!
//! Owner: telemetry.
//! Invariants: a ring slot is handed out again only after its span landed,
//! failed, or could not start.

use std::sync::atomic::{AtomicBool, AtomicU8, AtomicU64, AtomicUsize, Ordering};

use crate::instrument::{Counter, NanosHistogram};

/// 2026-09-26: The serve lanes timed on the GPU each step at level `Kernel`,
/// indexed as the scheduler's `Lane::telemetry_index` numbers them.
pub const LANES: [&str; 3] = ["admit", "prefill", "decode"];
/// 2026-09-26: [`LANES`] as NVTX range names (NUL-terminated), in the same order.
#[cfg(feature = "nvtx")]
const LANES_NVTX: [&std::ffi::CStr; LANES.len()] = [c"admit", c"prefill", c"decode"];

/// 2026-09-26: Open lane `lane`'s NVTX range on the calling thread; an index
/// outside [`LANES`] is ignored.
#[cfg(feature = "nvtx")]
pub fn nvtx_push_lane(lane: usize) {
    if let Some(name) = LANES_NVTX.get(lane) {
        metrale_gpu_sys::nvtx::push(name);
    }
}

/// 2026-09-26: Close the innermost NVTX range on the calling thread.
#[cfg(feature = "nvtx")]
pub fn nvtx_pop() {
    metrale_gpu_sys::nvtx::pop();
}

/// 2026-09-26: Distinct kernels the per-kernel table can hold.
pub const KERNEL_TABLE_SLOTS: usize = 512;

/// 2026-09-26: What a span measured.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SpanKey {
    Lane(usize),
    /// 2026-09-26: A kernel, by its function handle.
    Kernel(u64),
}

const LANE_BIT: u64 = 1 << 63;

impl SpanKey {
    fn encode(self) -> u64 {
        match self {
            Self::Lane(i) => LANE_BIT | i as u64,
            Self::Kernel(f) => f & !LANE_BIT,
        }
    }
    fn decode(v: u64) -> Self {
        if v & LANE_BIT != 0 {
            Self::Lane((v & !LANE_BIT) as usize)
        } else {
            Self::Kernel(v)
        }
    }
}

/// 2026-09-26: One edge of a span.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Edge {
    Start,
    End,
}

/// 2026-09-26: A span's completion state.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SpanPoll {
    /// 2026-09-26: Both events completed; GPU nanoseconds between them.
    Ready(u64),
    /// 2026-09-26: The end event has not completed yet.
    Pending,
    /// 2026-09-26: The events cannot be read.
    Failed,
}

/// 2026-09-26: A pool of event pairs addressed by slot.
pub trait EventOps: Send + Sync {
    /// 2026-09-26: Record `slot`'s `edge` event on `stream`. `false` = the
    /// record failed.
    fn record(&self, slot: usize, edge: Edge, stream: u64) -> bool;
    /// 2026-09-26: Non-blocking completion query for `slot`'s pair.
    fn poll(&self, slot: usize) -> SpanPoll;
}

/// 2026-09-26: Where span durations land.
#[derive(Debug)]
pub struct KernelInstruments {
    pub lane: [NanosHistogram; LANES.len()],
    keys: [AtomicU64; KERNEL_TABLE_SLOTS],
    total_ns: [AtomicU64; KERNEL_TABLE_SLOTS],
    samples: [AtomicU64; KERNEL_TABLE_SLOTS],
    /// 2026-09-26: Kernel spans not placed: the table had no free slot, or
    /// the handle was 0.
    pub table_full: Counter,
    /// 2026-09-26: Spans not started because the next ring slot was still in
    /// use.
    pub spans_dropped: Counter,
    /// 2026-09-26: Spans whose events could not be recorded or read.
    pub span_failures: Counter,
    pub(crate) armed: AtomicBool,
    pub(crate) step: AtomicU64,
}

impl Default for KernelInstruments {
    fn default() -> Self {
        Self::new()
    }
}

/// 2026-09-26: One row of the per-kernel table.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct KernelTime {
    pub func: u64,
    pub total_ns: u64,
    pub samples: u64,
}

impl KernelInstruments {
    pub const fn new() -> Self {
        Self {
            lane: [const { NanosHistogram::new() }; LANES.len()],
            keys: [const { AtomicU64::new(0) }; KERNEL_TABLE_SLOTS],
            total_ns: [const { AtomicU64::new(0) }; KERNEL_TABLE_SLOTS],
            samples: [const { AtomicU64::new(0) }; KERNEL_TABLE_SLOTS],
            table_full: Counter::new(),
            spans_dropped: Counter::new(),
            span_failures: Counter::new(),
            armed: AtomicBool::new(false),
            step: AtomicU64::new(0),
        }
    }

    /// 2026-09-26: Land one completed span; a lane index outside [`LANES`] is
    /// ignored.
    pub fn record(&self, key: SpanKey, ns: u64) {
        match key {
            SpanKey::Lane(i) => {
                if let Some(h) = self.lane.get(i) {
                    h.observe(ns);
                }
            }
            SpanKey::Kernel(func) => self.record_kernel(func, ns),
        }
    }

    fn record_kernel(&self, func: u64, ns: u64) {
        // 2026-09-26: Key 0 marks an empty slot, so handle 0 cannot be placed.
        if func == 0 {
            self.table_full.add(1);
            return;
        }
        let start = (func.wrapping_mul(0x9E37_79B9_7F4A_7C15) >> 55) as usize;
        for probe in 0..KERNEL_TABLE_SLOTS {
            let i = (start + probe) % KERNEL_TABLE_SLOTS;
            let k = self.keys[i].load(Ordering::Acquire);
            let owned = k == func
                || (k == 0
                    && match self.keys[i].compare_exchange(
                        0,
                        func,
                        Ordering::AcqRel,
                        Ordering::Acquire,
                    ) {
                        Ok(_) => true,
                        Err(now) => now == func,
                    });
            if owned {
                self.total_ns[i].fetch_add(ns, Ordering::Relaxed);
                self.samples[i].fetch_add(1, Ordering::Relaxed);
                return;
            }
        }
        self.table_full.add(1);
    }

    /// 2026-09-26: Every kernel with at least one sample.
    pub fn kernels(&self) -> Vec<KernelTime> {
        (0..KERNEL_TABLE_SLOTS)
            .filter_map(|i| {
                let func = self.keys[i].load(Ordering::Acquire);
                (func != 0).then(|| KernelTime {
                    func,
                    total_ns: self.total_ns[i].load(Ordering::Relaxed),
                    samples: self.samples[i].load(Ordering::Relaxed),
                })
            })
            .filter(|k| k.samples > 0)
            .collect()
    }
}

const FREE: u8 = 0;
const OPEN: u8 = 1;
const CLOSED: u8 = 2;

/// 2026-09-26: A span handed out by [`SpanRing::begin`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SpanToken(usize);

impl SpanToken {
    /// 2026-09-26: The ring slot, for a caller that parks an open span in an
    /// atomic.
    pub fn slot(self) -> usize {
        self.0
    }
    /// 2026-09-26: The token for a slot this caller got from [`SpanToken::slot`].
    pub fn from_slot(slot: usize) -> Self {
        Self(slot)
    }
}

/// 2026-09-26: The in-flight spans over a fixed pool of event pairs.
pub struct SpanRing<E: EventOps> {
    ops: E,
    state: Box<[AtomicU8]>,
    key: Box<[AtomicU64]>,
    head: AtomicUsize,
}

impl<E: EventOps> SpanRing<E> {
    /// 2026-09-26: `slots` event pairs, which `ops` must already hold.
    pub fn new(ops: E, slots: usize) -> Self {
        assert!(slots > 0, "a span ring needs at least one slot");
        Self {
            ops,
            state: (0..slots).map(|_| AtomicU8::new(FREE)).collect(),
            key: (0..slots).map(|_| AtomicU64::new(0)).collect(),
            head: AtomicUsize::new(0),
        }
    }

    pub fn slots(&self) -> usize {
        self.state.len()
    }

    /// 2026-09-26: Start a span on `stream`. `None` (and counted) when the next
    /// slot is still in use or its start event could not be recorded.
    pub fn begin(&self, key: SpanKey, stream: u64, sink: &KernelInstruments) -> Option<SpanToken> {
        let i = self.head.fetch_add(1, Ordering::Relaxed) % self.state.len();
        if self.state[i]
            .compare_exchange(FREE, OPEN, Ordering::Acquire, Ordering::Relaxed)
            .is_err()
        {
            sink.spans_dropped.add(1);
            return None;
        }
        self.key[i].store(key.encode(), Ordering::Relaxed);
        if !self.ops.record(i, Edge::Start, stream) {
            self.state[i].store(FREE, Ordering::Release);
            sink.span_failures.add(1);
            return None;
        }
        Some(SpanToken(i))
    }

    /// 2026-09-26: Close a span on `stream`. If the end event cannot be
    /// recorded, the slot is freed and the failure counted.
    pub fn end(&self, tok: SpanToken, stream: u64, sink: &KernelInstruments) {
        let i = tok.0;
        if self.ops.record(i, Edge::End, stream) {
            self.state[i].store(CLOSED, Ordering::Release);
        } else {
            self.state[i].store(FREE, Ordering::Release);
            sink.span_failures.add(1);
        }
    }

    /// 2026-09-26: Land every completed span into `sink`; returns how many
    /// landed. A pending span is left for a later harvest.
    pub fn harvest(&self, sink: &KernelInstruments) -> usize {
        let mut landed = 0;
        for i in 0..self.state.len() {
            if self.state[i].load(Ordering::Acquire) != CLOSED {
                continue;
            }
            match self.ops.poll(i) {
                SpanPoll::Pending => continue,
                SpanPoll::Ready(ns) => {
                    sink.record(SpanKey::decode(self.key[i].load(Ordering::Relaxed)), ns);
                    landed += 1;
                }
                SpanPoll::Failed => sink.span_failures.add(1),
            }
            self.state[i].store(FREE, Ordering::Release);
        }
        landed
    }

    pub fn ops(&self) -> &E {
        &self.ops
    }
}

#[cfg(test)]
#[path = "kernel_tests.rs"]
mod tests;
