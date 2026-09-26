// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-25: Env-gated phase timing for the MTP verify path.
//!
//! Owner: scheduler.
//! Invariants:
//! - Diagnostic only: nothing here feeds back into scheduling.
//! - Disarmed unless `METRALE_MTP_TIMING=1`; disarmed, `record` and
//!   `step_done` return at once.
//!
//! Armed, it accumulates microseconds per [`Phase`] and logs one `info!` line
//! of per-step averages every [`SUMMARY_PERIOD`] verify steps, then resets
//! the accumulators.

use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Instant;

/// 2026-09-25: Verify steps per summary line.
const SUMMARY_PERIOD: u64 = 25;

/// 2026-09-25: Timed phases. `StepTotal` must stay last (it sizes the arrays).
#[derive(Debug, Clone, Copy)]
#[repr(usize)]
pub(crate) enum Phase {
    SyncSecondary = 0,
    EpBroadcast,
    VerifyForward,
    FastGreedy,
    D2h,
    Dequant,
    PipelineProc,
    GrammarFill,
    ForcedTok,
    Penalties,
    Argmax,
    Commit,
    SaveHidden,
    TrimProposer,
    ProposeMask,
    Propose,
    MarconiCkpt,
    /// 2026-09-25: One whole `step_mtp` call.
    StepOuter,
    /// 2026-09-25: Time from one [`StepTimer`]'s drop to the next
    /// [`StepTimer`]'s construction.
    Gap,
    LoopDrain,
    LoopSnapshot,
    LoopAdmit,
    LoopPrefill,
    LoopRetire,
    LoopSwap,
    StepTotal,
}

const NUM_PHASES: usize = Phase::StepTotal as usize + 1;

pub(crate) const NAMES: [&str; NUM_PHASES] = [
    "sync",
    "ep",
    "fwd",
    "fast_greedy",
    "d2h",
    "dequant",
    "pipeline",
    "grammar_fill",
    "forced_tok",
    "penalties",
    "argmax",
    "commit",
    "save_hidden",
    "trim",
    "propose_mask",
    "propose",
    "marconi",
    "step_mtp",
    "GAP",
    "loop_drain",
    "loop_snapshot",
    "loop_admit",
    "loop_prefill",
    "loop_retire",
    "loop_swap",
    "TOTAL",
];

/// 2026-09-25: Per-phase microsecond accumulators. `SysTelemetry`
/// (io/telemetry.rs) owns the one [`RunTiming::from_env`] builds.
/// `GrammarState` (grammar/state.rs) records `GrammarFill` and `ForcedTok`
/// into its own `timing`, which is always disarmed: its builder sets
/// `Arc::default()`, and the only `with_timing` caller passes the same.
#[derive(Debug)]
pub struct RunTiming {
    /// 2026-09-25: True when [`RunTiming::from_env`] saw
    /// `METRALE_MTP_TIMING=1`. When false, `record` returns at once.
    pub armed: bool,
    sum_us: [AtomicU64; NUM_PHASES],
    count: [AtomicU64; NUM_PHASES],
    steps: AtomicU64,
    /// 2026-09-25: Micros since `anchor` at the last [`StepTimer`] drop
    /// (0 = none yet). Written by the drop, read by [`StepTimer::new`] to
    /// record [`Phase::Gap`].
    last_step_end_us: AtomicU64,
    /// 2026-09-25: Origin of the gap clock. `Instant` cannot live in an
    /// atomic, so gap timestamps are micros since this anchor; only their
    /// difference is recorded.
    anchor: Instant,
}

impl RunTiming {
    pub fn from_env() -> Self {
        Self::new(std::env::var("METRALE_MTP_TIMING").ok().as_deref() == Some("1"))
    }

    pub fn new(armed: bool) -> Self {
        Self {
            armed,
            sum_us: [const { AtomicU64::new(0) }; NUM_PHASES],
            count: [const { AtomicU64::new(0) }; NUM_PHASES],
            steps: AtomicU64::new(0),
            last_step_end_us: AtomicU64::new(0),
            anchor: Instant::now(),
        }
    }
    fn anchor_us(&self) -> u64 {
        u64::try_from(self.anchor.elapsed().as_micros()).unwrap_or(u64::MAX)
    }

    /// 2026-09-25: Record the elapsed time since `since` under `phase`.
    /// No-op when disarmed.
    pub(crate) fn record(&self, phase: Phase, since: Instant) {
        if !self.armed {
            return;
        }
        let us = u64::try_from(since.elapsed().as_micros()).unwrap_or(u64::MAX);
        self.sum_us[phase as usize].fetch_add(us, Ordering::Relaxed);
        self.count[phase as usize].fetch_add(1, Ordering::Relaxed);
    }

    pub(crate) fn sum_us(&self, phase: Phase) -> u64 {
        self.sum_us[phase as usize].load(Ordering::Relaxed)
    }

    pub(crate) fn count(&self, phase: Phase) -> u64 {
        self.count[phase as usize].load(Ordering::Relaxed)
    }

    pub fn bump_steps(&self) -> u64 {
        self.steps.fetch_add(1, Ordering::Relaxed) + 1
    }

    pub fn steps(&self) -> u64 {
        self.steps.load(Ordering::Relaxed)
    }
}

impl Default for RunTiming {
    fn default() -> Self {
        Self::new(false)
    }
}

/// 2026-09-25: When armed: record `StepTotal` since `step_start`, count the
/// step, and every [`SUMMARY_PERIOD`] steps log the summary and reset the
/// accumulators. Call once per verify step.
pub(crate) fn step_done(timing: &RunTiming, step_start: Instant, seq_len: usize) {
    if !timing.armed {
        return;
    }
    timing.record(Phase::StepTotal, step_start);
    let steps = timing.bump_steps();
    if !steps.is_multiple_of(SUMMARY_PERIOD) {
        return;
    }
    use std::fmt::Write as _;
    let mut line = String::with_capacity(NUM_PHASES * 32);
    for i in 0..NUM_PHASES {
        let sum = timing.sum_us[i].swap(0, Ordering::Relaxed);
        let cnt = timing.count[i].swap(0, Ordering::Relaxed);
        if cnt == 0 {
            continue;
        }
        // 2026-09-25: Per-step average; a phase can fire more than once per
        // step, and `xN.N` is its average fires per step.
        let per_step_ms = sum as f64 / 1000.0 / SUMMARY_PERIOD as f64;
        let fires = cnt as f64 / SUMMARY_PERIOD as f64;
        let _ = write!(line, " {}={per_step_ms:.2}ms(x{fires:.1})", NAMES[i]);
    }
    tracing::info!("MTP verify timing [{SUMMARY_PERIOD} steps, seq_len={seq_len}]:{line}");
}

/// 2026-09-25: Guard for one verify step (`verify_k4_step`,
/// `verify_k4_batch_step`). Construction records [`Phase::Gap`] since the
/// previous guard's drop; the drop calls [`step_done`], so every exit path
/// of the step is counted, error returns included.
///
/// `seq_len` is captured at construction and only labels the log line.
/// Disarmed, the guard costs one `Instant::now()`.
pub(crate) struct StepTimer<'a> {
    start: Instant,
    seq_len: usize,
    /// 2026-09-25: The accumulators this step records into.
    timing: &'a RunTiming,
}

impl<'a> StepTimer<'a> {
    pub(crate) fn new(timing: &'a RunTiming, seq_len: usize) -> Self {
        if timing.armed {
            let prev = timing.last_step_end_us.load(Ordering::Relaxed);
            if prev != 0 {
                let gap = timing.anchor_us().saturating_sub(prev);
                timing.sum_us[Phase::Gap as usize].fetch_add(gap, Ordering::Relaxed);
                timing.count[Phase::Gap as usize].fetch_add(1, Ordering::Relaxed);
            }
        }
        Self {
            start: Instant::now(),
            seq_len,
            timing,
        }
    }
}

impl Drop for StepTimer<'_> {
    fn drop(&mut self) {
        step_done(self.timing, self.start, self.seq_len);
        if self.timing.armed {
            self.timing
                .last_step_end_us
                .store(self.timing.anchor_us().max(1), Ordering::Relaxed);
        }
    }
}
