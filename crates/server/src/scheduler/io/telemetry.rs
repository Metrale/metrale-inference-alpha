// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-25: [`TelemetryIo`]: timing marks, the speculation counters, the
//! dashboard snapshot and the diagnostic file sinks.
//!
//! The serving implementation also feeds the metal-up instrument set
//! (`metrale_telemetry`): loop phases and the published snapshot's queue,
//! batch, KV and SSM shape (`sched`), acceptance by draft depth
//! (`SpecMatrix`), finished requests (`request`), emitted tokens (the
//! J/token denominator), and, at level `Kernel`, GPU spans per serve lane.
//! At level `Off` each of these feeds returns after one relaxed load of the
//! level.
//!
//! Owner: scheduler.
//! Invariants: none beyond the types.

use std::time::Instant;

use metrale_telemetry::Telemetry;
use metrale_telemetry::request::RequestTiming;
use metrale_telemetry::sched::SchedShape;

use metrale_speculative::snapshot::{SchedulerSnapshot, SnapshotCell};
use metrale_speculative::spec_stats::SpecStats;

use crate::scheduler::dumps::RunDumps;
use crate::scheduler::mtp_timing::{Phase, RunTiming, StepTimer, step_done};

pub trait TelemetryIo: Send + Sync + std::fmt::Debug {
    /// 2026-09-25: Record the time since `since` under `phase`
    /// (`RunTiming::record`).
    fn mark(&self, phase: Phase, since: Instant);
    /// 2026-09-25: The RAII bracket around one verify step.
    fn step_timer(&self, seq_len: usize) -> StepTimer<'_>;
    /// 2026-09-25: The explicit form of the bracket's end.
    fn step_done(&self, step_start: Instant, seq_len: usize);
    /// 2026-09-25: The run's speculation counters and log-once latches.
    fn stats(&self) -> &SpecStats;
    /// 2026-09-25: The run's diagnostic file sinks.
    fn dumps(&self) -> &RunDumps;
    /// 2026-09-25: Publish the dashboard snapshot.
    fn publish(&self, snapshot: SchedulerSnapshot);
    /// 2026-09-25: Bump the `SPEC_DECODE_VERIFY{mode, outcome}` counter.
    fn count_spec_verify(&self, mode: &str, outcome: &str);
    /// 2026-09-25: Whether the instrument set records at all (level above
    /// `Off`). A caller whose feed needs work to compute asks this first.
    fn enabled(&self) -> bool;
    /// 2026-09-25: A verify step checked `drafts` drafts and accepted
    /// `accepted`.
    fn spec_verified(&self, drafts: usize, accepted: usize);
    /// 2026-09-25: `n` tokens were emitted to clients (the J/token
    /// denominator).
    fn tokens(&self, n: u64);
    /// 2026-09-25: A request reached its terminal frame: its latencies and
    /// attributed energy.
    fn request_finished(&self, timing: RequestTiming);
    /// 2026-09-25: Output tokens of every request finished so far (it does
    /// not grow while the level is `Off`).
    fn finished_output_tokens(&self) -> u64;
    /// 2026-09-25: A scheduler tick begins: land completed GPU spans and
    /// arm the per-kernel sample for this step (level `Kernel`).
    fn step_begin(&self);
    /// 2026-09-25: Serve lane `lane` (an index into
    /// `metrale_telemetry::kernel::LANES`) begins / ends on the GPU (level
    /// `Kernel`).
    fn lane_begin(&self, lane: usize);
    fn lane_end(&self, lane: usize);
}

/// 2026-09-25: The serving implementation: this run's timing, counters,
/// dump sinks and snapshot cell, plus an instrument set and the prometheus
/// counters.
pub struct SysTelemetry {
    /// 2026-09-25: The instrument set this run feeds: the process's for
    /// serving, an always-`Off` one for tests.
    tel: &'static Telemetry,
    timing: RunTiming,
    stats: SpecStats,
    dumps: RunDumps,
    snapshot: std::sync::Arc<SnapshotCell>,
}

impl SysTelemetry {
    /// 2026-09-25: Sinks opened from this run's environment, feeding `tel`.
    pub fn from_env(tel: &'static Telemetry, snapshot: std::sync::Arc<SnapshotCell>) -> Self {
        Self {
            tel,
            timing: RunTiming::from_env(),
            stats: SpecStats::new(),
            dumps: RunDumps::from_env(),
            snapshot,
        }
    }

    /// 2026-09-25: Disarmed timing and no dumps, without reading the
    /// environment.
    pub fn quiet(snapshot: std::sync::Arc<SnapshotCell>) -> Self {
        /// 2026-09-25: Never configured, so every feed into it is the
        /// `Off` branch.
        static QUIET: Telemetry = Telemetry::new(&metrale_telemetry::clock::MonotonicClock);
        Self {
            tel: &QUIET,
            timing: RunTiming::default(),
            stats: SpecStats::new(),
            dumps: RunDumps::default(),
            snapshot,
        }
    }
}

impl std::fmt::Debug for SysTelemetry {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("SysTelemetry")
            .field("timing", &self.timing)
            .field("stats", &self.stats)
            .field("dumps", &self.dumps)
            .finish_non_exhaustive()
    }
}

impl TelemetryIo for SysTelemetry {
    fn mark(&self, phase: Phase, since: Instant) {
        self.timing.record(phase, since);
        self.tel.phase(phase as usize, since);
    }
    fn step_timer(&self, seq_len: usize) -> StepTimer<'_> {
        StepTimer::new(&self.timing, seq_len)
    }
    fn step_done(&self, step_start: Instant, seq_len: usize) {
        step_done(&self.timing, step_start, seq_len);
    }
    fn stats(&self) -> &SpecStats {
        &self.stats
    }
    fn dumps(&self) -> &RunDumps {
        &self.dumps
    }
    fn publish(&self, snapshot: SchedulerSnapshot) {
        self.tel.sched_tick(&SchedShape {
            pending: u64::from(snapshot.pending_len),
            active: u64::from(snapshot.active_seqs),
            prefilling: u64::from(snapshot.prefilling_seqs),
            swapped: u64::from(snapshot.swapped_seqs),
            kv_blocks_free: u64::from(snapshot.kv_blocks_free),
            kv_blocks_total: u64::from(snapshot.kv_blocks_total),
            ssm_slots_used: u64::from(snapshot.ssm_slots_used),
            ssm_slots_total: u64::from(snapshot.ssm_slots_total),
            ticks: snapshot.steps_total,
        });
        self.snapshot.publish(snapshot);
    }
    fn count_spec_verify(&self, mode: &str, outcome: &str) {
        crate::metrics::SPEC_DECODE_VERIFY
            .with_label_values(&[mode, outcome])
            .inc();
    }
    fn enabled(&self) -> bool {
        self.tel.level() != metrale_telemetry::Level::Off
    }
    fn spec_verified(&self, drafts: usize, accepted: usize) {
        self.tel.spec_verified(drafts, accepted);
    }
    fn tokens(&self, n: u64) {
        self.tel.tokens(n);
    }
    fn request_finished(&self, timing: RequestTiming) {
        self.tel.request_finished(&timing);
    }
    fn finished_output_tokens(&self) -> u64 {
        self.tel.requests.output_tokens.get()
    }
    fn step_begin(&self) {
        if self.tel.lane_spans() {
            metrale_gpu_runtime::timing::harvest();
            self.tel.step_begin();
        }
    }
    fn lane_begin(&self, lane: usize) {
        if self.tel.lane_spans() {
            #[cfg(feature = "nvtx")]
            metrale_telemetry::kernel::nvtx_push_lane(lane);
            metrale_gpu_runtime::timing::lane_begin(lane);
        }
    }
    fn lane_end(&self, lane: usize) {
        if self.tel.lane_spans() {
            metrale_gpu_runtime::timing::lane_end(lane);
            #[cfg(feature = "nvtx")]
            metrale_telemetry::kernel::nvtx_pop();
        }
    }
}
