// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-25: The `MtpGate` methods the scheduler calls: construction, the
//! snapshot accessor, the depth hooks, and the per-step record and decision
//! calls. The window arithmetic and arbitration are in `mtp_gate.rs`.
//!
//! Owner: speculative.
//! Invariants: none beyond the types.

use super::*;

impl MtpGate {
    /// 2026-09-25: For the scheduler snapshot: the mode (`Probing` during a
    /// probe) and the delivered-throughput estimate of the current mode (0.0
    /// until measured).
    pub fn observe(&self) -> (crate::snapshot::MtpModeSnap, f32) {
        use crate::snapshot::MtpModeSnap;
        let mode = if self.probing {
            MtpModeSnap::Probing
        } else {
            match self.mode {
                Mode::Mtp => MtpModeSnap::Mtp,
                Mode::Serial => MtpModeSnap::Serial,
            }
        };
        let stats = match self.mode {
            Mode::Mtp => &self.mtp,
            Mode::Serial => &self.serial,
        };
        (mode, stats.tps.unwrap_or(0.0) as f32)
    }

    /// 2026-09-25: Starts in Mtp mode. `num_drafts` is only logged;
    /// arbitration does not model the draft count.
    pub fn new(num_drafts: usize) -> Self {
        // 2026-09-25: Read once here: `event_interval` reads them on every
        // recorded step.
        let reprobe = reprobe_tokens();
        let refresh = serial_refresh_tokens();
        tracing::info!(
            "MTP gate: throughput-arbitrated (K={num_drafts}); window={WINDOW_STEPS} steps, \
             dwell={SWITCH_DWELL_WINDOWS}, reprobe={reprobe} tok, refresh={refresh} tok",
        );
        Self {
            reprobe,
            refresh,
            mode: Mode::Mtp,
            probing: false,
            probe_windows_left: 0,
            mtp: ModeStats::default(),
            serial: ModeStats::default(),
            win_tokens: 0.0,
            win_wall: 0.0,
            win_steps: 0,
            losing_windows: 0,
            tokens_since_event: 0,
            observed_depth: 0,
            measured_at_depth: 0,
            width_regime: 0,
            fresh: None,
            regime_reprobes: 0,
        }
    }

    pub fn note_depth(&mut self, depth: usize) {
        self.observed_depth = depth;
    }

    /// 2026-09-25: When the context depth has moved by
    /// `REMEASURE_DEPTH_FACTOR` since the last measurement (both floored at
    /// `REMEASURE_DEPTH_FLOOR`), marks both estimates stale, keeps them as
    /// they are, and brings the next probe (one `WINDOW_STEPS` window)
    /// forward. Returns whether it fired.
    pub fn maybe_remeasure(&mut self, current_depth: usize) -> bool {
        let measured = self.measured_at_depth.max(REMEASURE_DEPTH_FLOOR);
        let live = current_depth.max(REMEASURE_DEPTH_FLOOR);
        if live >= measured * REMEASURE_DEPTH_FACTOR || measured >= live * REMEASURE_DEPTH_FACTOR {
            tracing::info!(
                "MTP gate: depth regime changed ({} -> {} tokens); baselines stale, \
                 will re-probe on cadence",
                self.measured_at_depth,
                current_depth,
            );
            self.mtp.stale = true;
            self.serial.stale = true;
            self.measured_at_depth = current_depth;
            // 2026-09-25: Make the next probe due now.
            self.tokens_since_event = self.tokens_since_event.max(self.event_interval());
            self.regime_reprobes = self.regime_reprobes.saturating_add(1);
            true
        } else {
            false
        }
    }

    /// 2026-09-25: The last mode switch, returned once.
    pub fn take_fresh_decision(&mut self) -> Option<GateDecision> {
        self.fresh.take()
    }

    /// 2026-09-25: Which step the scheduler should run next.
    pub fn next_step(&self) -> GateStep {
        let effective = if self.probing {
            Self::other(self.mode)
        } else {
            self.mode
        };
        match effective {
            Mode::Mtp => GateStep::MeasureVerify,
            Mode::Serial => GateStep::MeasureDecode,
        }
    }

    /// 2026-09-25: Records one plain decode step over `width` sequences. Each
    /// emits one token, so the step is charged `width` tokens (at least 1).
    pub fn record_decode(&mut self, wall: Duration, width: usize) {
        self.note_width(width.max(1));
        self.record_step(wall, width.max(1));
    }

    /// 2026-09-25: Records one MTP step over `width` sequences: `emitted` is
    /// the tokens all of them committed (at least 1 is charged). Bootstrap and
    /// propose time count against Mtp mode.
    pub fn record_verify_step(&mut self, wall: Duration, emitted: usize, width: usize) {
        self.note_width(width.max(1));
        self.record_step(wall, emitted.max(1));
    }
}
