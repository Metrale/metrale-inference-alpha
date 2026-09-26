// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-25: The per-lane telemetry: each lane's GPU span and the tokens it
//! produced, both through `TelemetryIo`. The token tally is computed only when
//! `TelemetryIo::enabled()`.
//!
//! Owner: scheduler.
//! Invariants: none beyond the types.

use super::*;

impl Lane {
    /// 2026-09-25: This lane's index in `metrale_telemetry::kernel::LANES`.
    pub(super) fn telemetry_index(&self) -> usize {
        match self {
            Lane::StartPrefills(_) => 0,
            Lane::ContinuePrefills => 1,
            Lane::Decode => 2,
        }
    }
}

impl SchedulerCore {
    /// 2026-09-25: Output tokens produced by every sequence the core still holds.
    ///
    /// A sequence moving between `active`, `preempted` and `swapped` keeps
    /// its tokens, so those moves leave the sum unchanged.
    fn held_output_tokens(&self) -> u64 {
        self.active
            .iter()
            .map(|a| a.output_tokens.len())
            .chain(self.preempted.iter().map(|p| p.a.output_tokens.len()))
            .chain(self.swapped.iter().map(|s| s.output_tokens.len()))
            .sum::<usize>() as u64
    }

    /// 2026-09-25: Tokens accounted so far: those still held plus those of every
    /// finished request.
    ///
    /// `execute_lane_measured` tallies a lane's growth in this sum as the tokens it
    /// produced, and counts zero when the sum shrinks (a rollback that truncates
    /// output).
    fn accounted_tokens(&self) -> u64 {
        self.held_output_tokens() + self.ctx.io.tel.finished_output_tokens()
    }

    /// 2026-09-25: Run one lane inside its GPU span, tallying the tokens it produced.
    pub(super) fn execute_lane_measured(&mut self, lane: Lane) -> LaneVerdict {
        let index = lane.telemetry_index();
        let before = self.ctx.io.tel.enabled().then(|| self.accounted_tokens());
        self.ctx.io.tel.lane_begin(index);
        let verdict = match lane {
            Lane::StartPrefills(new_reqs) => self.start_prefills(new_reqs),
            Lane::ContinuePrefills => self.continue_prefills(),
            Lane::Decode => self.decode_lane(),
        };
        self.ctx.io.tel.lane_end(index);
        if let Some(before) = before {
            let produced = self.accounted_tokens().saturating_sub(before);
            self.ctx.io.tel.tokens(produced);
        }
        verdict
    }
}
