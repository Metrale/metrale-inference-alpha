// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-26: [`Telemetry`]: one instrument set behind one level gate.
//!
//! Every hot-path entry point first loads one relaxed atomic (the level, or
//! for `kernel_spans_armed` the armed flag) and at `Off` returns before any
//! clock read, allocation or atomic write. `tests/zero_cost_off.rs` checks it
//! with a counting allocator and a counting clock.
//!
//! The serving process uses [`global`], a `static`: the HTTP handlers for
//! `/metrics` and `/v1/events`, the GPU backend and the device sampler thread
//! all reach it without a handle from the scheduler. Tests build their own
//! instance.
//!
//! Owner: telemetry.
//! Invariants: at `Off`, no hot-path entry point reads the clock, allocates
//! or writes an atomic.

use std::sync::OnceLock;
use std::sync::atomic::{AtomicU8, AtomicU32, Ordering};
use std::time::Instant;

use crate::clock::{Clock, MonotonicClock};
use crate::device::{DeviceReading, SAMPLE_WORDS};
use crate::energy::{
    Advance, CounterTracker, EnergyPoint, EnergyRing, RING_SLOTS, joules_per_token,
    request_millijoules,
};
use crate::instrument::{Counter, Gauge, GaugeF64};
use crate::kernel::{KernelInstruments, LANES};
use crate::level::{Level, TelemetryConfig};
use crate::request::{RequestInstruments, RequestTiming};
use crate::sched::{MAX_PHASES, SchedInstruments, SchedShape, SpecMatrix};
use crate::seqcell::SeqCell;

/// 2026-09-26: Device samples the live J/token spans (1 s at the serving cadence).
pub const LIVE_WINDOW: u64 = 10;

/// 2026-09-26: Whether the device layer is reading.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u8)]
pub enum DeviceState {
    /// 2026-09-26: No sampler has started (telemetry off, or not yet configured).
    NotStarted = 0,
    Available = 1,
    /// 2026-09-26: The device layer cannot read: NVML is missing, has no such
    /// device, or the sampler thread did not start. See
    /// [`Telemetry::device_unavailable_reason`].
    Unavailable = 2,
}

pub struct Telemetry {
    level: AtomicU8,
    clock: &'static dyn Clock,
    kernel_every: AtomicU32,
    phase_names: OnceLock<&'static [&'static str]>,
    pub(crate) device: SeqCell<SAMPLE_WORDS>,
    device_state: AtomicU8,
    device_reason: OnceLock<String>,
    pub device_read_errors: Gauge,
    pub energy_mj: Counter,
    pub energy_counter_raw: Gauge,
    pub energy_resets: Counter,
    pub energy_wraps: Counter,
    pub tokens: Counter,
    /// 2026-09-26: `tokens` at the first energy sample.
    energy_token_base: Gauge,
    pub joules_per_token_live: GaugeF64,
    ring: OnceLock<EnergyRing>,
    pub kernel: KernelInstruments,
    pub sched: SchedInstruments,
    pub spec: SpecMatrix,
    pub requests: RequestInstruments,
}

impl std::fmt::Debug for Telemetry {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Telemetry")
            .field("level", &self.level())
            .finish_non_exhaustive()
    }
}

static GLOBAL: Telemetry = Telemetry::new(&MonotonicClock);

/// 2026-09-26: The serving process's instrument set.
pub fn global() -> &'static Telemetry {
    &GLOBAL
}

impl Telemetry {
    /// 2026-09-26: A set at level `Off` over `clock`. Allocates nothing.
    pub const fn new(clock: &'static dyn Clock) -> Self {
        Self {
            level: AtomicU8::new(Level::Off as u8),
            clock,
            kernel_every: AtomicU32::new(u32::MAX),
            phase_names: OnceLock::new(),
            device: SeqCell::new(),
            device_state: AtomicU8::new(DeviceState::NotStarted as u8),
            device_reason: OnceLock::new(),
            device_read_errors: Gauge::new(),
            energy_mj: Counter::new(),
            energy_counter_raw: Gauge::new(),
            energy_resets: Counter::new(),
            energy_wraps: Counter::new(),
            tokens: Counter::new(),
            energy_token_base: Gauge::new(),
            joules_per_token_live: GaugeF64::new(),
            ring: OnceLock::new(),
            kernel: KernelInstruments::new(),
            sched: SchedInstruments::new(),
            spec: SpecMatrix::new(),
            requests: RequestInstruments::new(),
        }
    }

    /// 2026-09-26: Apply a validated configuration and disarm kernel spans.
    /// The energy ring is allocated here, on the first call at `Basic` or
    /// above, and never again.
    pub fn configure(&self, cfg: &TelemetryConfig) {
        if cfg.level() >= Level::Basic {
            self.ring
                .get_or_init(|| EnergyRing::with_capacity(RING_SLOTS));
        }
        self.kernel_every
            .store(cfg.kernel_span_every(), Ordering::Relaxed);
        self.kernel.armed.store(false, Ordering::Relaxed);
        self.level.store(cfg.level() as u8, Ordering::Release);
    }

    #[inline]
    pub fn level(&self) -> Level {
        Level::from_u8(self.level.load(Ordering::Relaxed))
    }

    #[inline]
    fn on(&self) -> bool {
        self.level.load(Ordering::Relaxed) != Level::Off as u8
    }

    /// 2026-09-26: Name the loop phases `phase` indexes, keeping at most
    /// [`MAX_PHASES`]. The first call wins.
    pub fn set_phase_names(&self, names: &'static [&'static str]) {
        let _ = self.phase_names.set(&names[..names.len().min(MAX_PHASES)]);
    }

    pub fn phase_names(&self) -> &'static [&'static str] {
        self.phase_names.get().copied().unwrap_or(&[])
    }

    /// 2026-09-26: `n` more tokens were produced.
    #[inline]
    pub fn tokens(&self, n: u64) {
        if !self.on() {
            return;
        }
        self.tokens.add(n);
    }

    /// 2026-09-26: Loop phase `idx` ran from `since` until now. An `idx` of
    /// [`MAX_PHASES`] or more lands in the last histogram.
    #[inline]
    pub fn phase(&self, idx: usize, since: Instant) {
        if !self.on() {
            return;
        }
        let ns = self.clock.elapsed_ns(since);
        self.sched.phase[idx.min(MAX_PHASES - 1)].observe(ns);
    }

    /// 2026-09-26: This tick's scheduler shape.
    #[inline]
    pub fn sched_tick(&self, shape: &SchedShape) {
        if !self.on() {
            return;
        }
        self.sched.record(shape);
    }

    #[inline]
    pub fn stream_sync(&self) {
        if self.on() {
            self.sched.stream_syncs.add(1);
        }
    }

    #[inline]
    pub fn blocking_d2h(&self) {
        if self.on() {
            self.sched.blocking_d2h.add(1);
        }
    }

    #[inline]
    pub fn graph_replay(&self) {
        if self.on() {
            self.sched.graph_replays.add(1);
        }
    }

    #[inline]
    pub fn graph_capture(&self) {
        if self.on() {
            self.sched.graph_captures.add(1);
        }
    }

    /// 2026-09-26: A verify step checked `drafts` drafts and accepted `accepted`.
    #[inline]
    pub fn spec_verified(&self, drafts: usize, accepted: usize) {
        if !self.on() {
            return;
        }
        self.spec.record(drafts, accepted);
    }

    /// 2026-09-26: A request finished. Its energy is attributed from the ring
    /// window its lifetime spans, only while the device layer is available.
    pub fn request_finished(&self, r: &RequestTiming) {
        if !self.on() {
            return;
        }
        let end = self.clock.now_ns();
        let start = end.saturating_sub(r.e2e_ns());
        let energy = self
            .ring
            .get()
            .filter(|_| self.device_state() == DeviceState::Available)
            .and_then(|ring| request_millijoules(ring, start, end, r.output_tokens));
        self.requests.record(r, energy);
    }

    /// 2026-09-26: A scheduler step begins. At level `Kernel`, arms per-kernel
    /// spans on one step in `kernel_span_every` and returns whether this step
    /// is armed; below `Kernel`, returns `false` and changes nothing.
    #[inline]
    pub fn step_begin(&self) -> bool {
        if self.level() != Level::Kernel {
            return false;
        }
        let n = self.kernel.step.fetch_add(1, Ordering::Relaxed);
        let every = u64::from(self.kernel_every.load(Ordering::Relaxed).max(1));
        let arm = n.is_multiple_of(every);
        self.kernel.armed.store(arm, Ordering::Relaxed);
        arm
    }

    /// 2026-09-26: Whether the kernel launch path should bracket launches this
    /// step. Set by `step_begin`, cleared by `configure`.
    #[inline]
    pub fn kernel_spans_armed(&self) -> bool {
        self.kernel.armed.load(Ordering::Relaxed)
    }

    /// 2026-09-26: Whether lane spans are recorded (level `Kernel`).
    #[inline]
    pub fn lane_spans(&self) -> bool {
        self.level() == Level::Kernel
    }

    pub fn lane_names(&self) -> &'static [&'static str] {
        &LANES
    }

    pub fn device_state(&self) -> DeviceState {
        match self.device_state.load(Ordering::Acquire) {
            1 => DeviceState::Available,
            2 => DeviceState::Unavailable,
            _ => DeviceState::NotStarted,
        }
    }

    pub fn device_unavailable_reason(&self) -> Option<&str> {
        self.device_reason.get().map(String::as_str)
    }

    /// 2026-09-26: The device layer cannot read. The first `why` is kept for
    /// [`Telemetry::device_unavailable_reason`].
    pub fn mark_device_unavailable(&self, why: String) {
        let _ = self.device_reason.set(why);
        self.device_state
            .store(DeviceState::Unavailable as u8, Ordering::Release);
    }

    /// 2026-09-26: Absorb one device reading: publish it and, when it carries
    /// the energy counter, advance the energy total; once the ring exists, also
    /// push a ring point pairing it with the tokens so far and update the live
    /// J/token.
    pub fn absorb_device(
        &self,
        tracker: &mut CounterTracker,
        reading: &DeviceReading,
        read_errors: u64,
    ) {
        let at_ns = self.clock.now_ns();
        self.device.store(reading.encode(at_ns));
        self.device_read_errors.set(read_errors);
        self.device_state
            .store(DeviceState::Available as u8, Ordering::Release);
        let Some(raw) = reading.energy_counter_mj else {
            return;
        };
        self.energy_counter_raw.set(raw);
        let step = tracker.advance(raw);
        match step {
            Advance::Wrapped(_) => self.energy_wraps.add(1),
            Advance::Reset(_) => self.energy_resets.add(1),
            Advance::First => self.energy_token_base.set(self.tokens.get()),
            Advance::Delta(_) => {}
        }
        self.energy_mj.add(step.millijoules());
        let Some(ring) = self.ring.get() else { return };
        let now = EnergyPoint {
            at_ns,
            energy_mj: self.energy_mj.get(),
            tokens: self.tokens.get(),
        };
        ring.push(now);
        let live = (1..=LIVE_WINDOW)
            .rev()
            .find_map(|k| ring.back(k))
            .and_then(|then| {
                joules_per_token(now.energy_mj - then.energy_mj, now.tokens - then.tokens)
            });
        self.joules_per_token_live.set(live);
    }

    /// 2026-09-26: The latest device reading, its time, and how many readings
    /// have been stored; `None` before the first.
    pub fn device_reading(&self) -> Option<(DeviceReading, u64, u64)> {
        let (stores, words) = self.device.load();
        (stores > 0).then(|| {
            let (r, at) = DeviceReading::decode(&words);
            (r, at, stores)
        })
    }

    /// 2026-09-26: GPU-rail J/token since the first energy sample.
    pub fn joules_per_token_cumulative(&self) -> Option<f64> {
        joules_per_token(self.energy_mj.get(), self.tokens_since_first_sample())
    }

    /// 2026-09-26: Tokens counted since the first energy sample: the
    /// denominator for [`Telemetry::energy_mj`], which also starts there.
    pub fn tokens_since_first_sample(&self) -> u64 {
        self.tokens
            .get()
            .saturating_sub(self.energy_token_base.get())
    }

    pub fn now_ns(&self) -> u64 {
        self.clock.now_ns()
    }
}

#[cfg(test)]
#[path = "hub_tests.rs"]
mod tests;
