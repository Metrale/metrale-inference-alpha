// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-25: [`SchedCtx`], the per-run scheduler context: vocabulary
//! masks, I/O routers, levers, limits, watchdog tunables, speculation
//! controllers and a few per-run counters.
//!
//! Owner: scheduler.
//! Invariants: none beyond the types.

use crate::scheduler::io::SchedIo;
use crate::scheduler::levers::SchedLevers;
use crate::scheduler::limits::SchedLimits;
use crate::scheduler::vocab_masks::VocabMasks;

/// 2026-09-25: Reusable host buffers for the decode path. Callers take the
/// `Vec` out and put it back, so no `RefCell` borrow is held while they
/// call back into the context.
#[derive(Debug, Default)]
pub struct DecodeScratch {
    /// 2026-09-25: FP32 logits for one sequence.
    pub seq_f32: std::cell::RefCell<Vec<f32>>,
    /// 2026-09-25: The decode step's device-to-host logits readback.
    pub host_bytes: std::cell::RefCell<Vec<u8>>,
}

/// 2026-09-25: The context of one scheduler run.
pub struct SchedCtx {
    /// 2026-09-25: Per-token classification masks for this model's
    /// vocabulary.
    pub masks: VocabMasks,
    /// 2026-09-25: The routers this run does its I/O through: device,
    /// clock, telemetry, requests and, when there is swap space, spill.
    pub io: SchedIo,
    /// 2026-09-25: Reusable host-side decode buffers for this run.
    pub scratch: DecodeScratch,
    /// 2026-09-25: Decode, verify and speculation levers for this run.
    /// `Arc` because the dashboard's `/watchdog` command toggles the loop
    /// watchdog on the same levers while the run reads them.
    pub levers: std::sync::Arc<SchedLevers>,
    /// 2026-09-25: Hard stops from this model's tokenizer and
    /// `--max-seq-len`.
    pub limits: SchedLimits,
    /// 2026-09-25: Decode-time watchdog tunables. Serving builds them with
    /// `WatchdogParams::from_behavior` from the model's MODEL.toml
    /// `[behavior]` table.
    pub watchdog: crate::scheduler::helpers::WatchdogParams,
    /// 2026-09-25: An optional trained repetition-onset head. `new` sets it
    /// to `None` and nothing else sets or reads it outside tests; a
    /// MODEL.toml `[behavior].rom_head` only logs a startup warning.
    pub rom_head: Option<std::sync::Arc<dyn crate::scheduler::rollback::RomHead>>,
    /// 2026-09-25: The n=16 MTP rung controller for this run.
    pub rung: metrale_speculative::adaptive_rung::AdaptiveRung,
    /// 2026-09-25: The DFlash gamma resolver for this run, configured at
    /// serve time.
    pub dflash_rung: metrale_speculative::dflash_rung::DflashRung,
    /// 2026-09-25: Width-attributed accept accounting for this run.
    pub accept: crate::scheduler::mtp_accept_debug::AcceptBuckets,
    /// 2026-09-25: D-Cut retained-rows telemetry for this run.
    pub dcut: crate::scheduler::mtp_dcut::DcutTelemetry,
    /// 2026-09-25: Admission's last queued-overflow count, so the queue
    /// line is logged only when it changes.
    pub admit_last_queued: std::cell::Cell<usize>,
    /// 2026-09-25: Decode steps that fell back to the host path because a
    /// device argmax picked `</think>` or `<think>` for a row whose thinking
    /// had ended.
    pub think_mask_fallbacks: std::cell::Cell<u64>,
}

impl SchedCtx {
    pub fn new(
        masks: VocabMasks,
        levers: std::sync::Arc<SchedLevers>,
        io: SchedIo,
        limits: SchedLimits,
        watchdog: crate::scheduler::helpers::WatchdogParams,
        rung: metrale_speculative::adaptive_rung::AdaptiveRung,
        dflash_rung: metrale_speculative::dflash_rung::DflashRung,
    ) -> Self {
        let accept =
            crate::scheduler::mtp_accept_debug::AcceptBuckets::new(levers.mtp_accept_fold_at_16);
        Self {
            masks,
            io,
            scratch: DecodeScratch::default(),
            levers,
            limits,
            watchdog,
            rom_head: None,
            rung,
            dflash_rung,
            accept,
            dcut: crate::scheduler::mtp_dcut::DcutTelemetry::default(),
            admit_last_queued: std::cell::Cell::new(0),
            think_mask_fallbacks: std::cell::Cell::new(0),
        }
    }

    /// 2026-09-25: A test context: default masks, `SchedLevers::defaults()`
    /// (read from no environment variable), no limits, default watchdog
    /// tunables and test I/O routers.
    pub fn for_test() -> Self {
        Self::for_test_io(SchedIo::for_test())
    }

    /// 2026-09-25: `for_test` with the sync device router over `model`.
    pub fn for_test_with(model: std::sync::Arc<dyn metrale_model_engine::traits::Model>) -> Self {
        Self::for_test_io(SchedIo::for_test_with(model))
    }

    fn for_test_io(io: SchedIo) -> Self {
        Self::new(
            VocabMasks::default(),
            std::sync::Arc::new(SchedLevers::defaults()),
            io,
            SchedLimits::NONE,
            crate::scheduler::helpers::WatchdogParams::default(),
            metrale_speculative::adaptive_rung::AdaptiveRung::new(
                metrale_speculative::adaptive_rung::RungParams::DEFAULTS,
            ),
            metrale_speculative::dflash_rung::DflashRung::new(),
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_test_context_needs_no_environment() {
        let c = SchedCtx::for_test();
        assert!(c.masks.numeric.is_none());
        assert!(c.levers.fast_masked, "an opt-out lever, on by default");
        assert!(!c.levers.dflash_adaptive);
    }

    #[test]
    fn two_contexts_are_independent() {
        let a = SchedCtx::for_test();
        let b = SchedCtx::for_test();
        a.levers.set_loop_watchdog(true);
        assert!(a.levers.loop_watchdog() && !b.levers.loop_watchdog());
    }
}
