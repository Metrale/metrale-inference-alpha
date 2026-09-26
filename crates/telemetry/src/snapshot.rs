// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-26: [`TelemetrySnapshot`]: the instruments' current values as one
//! plain struct. The `/metrics` telemetry section, the `/v1/events` stream and
//! the OTLP body are rendered from it.
//!
//! Owner: telemetry.
//! Invariants: none beyond the types.

use serde::Serialize;

use crate::device::DeviceReading;
use crate::hub::{DeviceState, Telemetry};
use crate::instrument::HistogramSnapshot;
use crate::sched::SchedShape;

#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct DeviceSnapshot {
    #[serde(flatten)]
    pub reading: DeviceReading,
    /// 2026-09-26: Device readings stored so far.
    pub samples: u64,
    /// 2026-09-26: How old the reading is, on the telemetry clock.
    pub age_ns: u64,
    pub read_errors: u64,
}

#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct EnergySnapshot {
    /// 2026-09-26: GPU-rail millijoules since the first sample, reset- and
    /// wrap-corrected.
    pub gpu_mj_total: u64,
    /// 2026-09-26: The raw NVML counter at the latest sample that carried it.
    pub gpu_counter_mj: u64,
    /// 2026-09-26: Tokens counted since the first energy sample.
    pub tokens_total: u64,
    /// 2026-09-26: J/token over the last [`crate::hub::LIVE_WINDOW`] samples.
    pub joules_per_token_live: Option<f64>,
    pub joules_per_token_cumulative: Option<f64>,
    pub counter_resets: u64,
    pub counter_wraps: u64,
}

#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct SchedSnapshot {
    #[serde(flatten)]
    pub shape: SchedShape,
    pub stream_syncs: u64,
    pub blocking_d2h: u64,
    pub graph_replays: u64,
    pub graph_captures: u64,
}

#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct SpecRow {
    pub drafts: usize,
    pub steps: u64,
    /// 2026-09-26: `[d-1]`: fraction of steps that accepted at least `d` drafts.
    pub acceptance_by_depth: Vec<f64>,
}

#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct RequestSnapshot {
    pub finished: u64,
    pub ttft_mean_ms: Option<f64>,
    pub tpot_mean_ms: Option<f64>,
    pub e2e_mean_ms: Option<f64>,
    pub energy_mean_j: Option<f64>,
    pub energy_unattributed: u64,
}

#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct TelemetrySnapshot {
    pub level: &'static str,
    pub at_ns: u64,
    pub device_state: &'static str,
    pub device: Option<DeviceSnapshot>,
    pub energy: EnergySnapshot,
    pub sched: SchedSnapshot,
    pub spec: Vec<SpecRow>,
    pub requests: RequestSnapshot,
}

fn mean(h: &HistogramSnapshot, scale: f64) -> Option<f64> {
    (h.count > 0).then(|| h.sum as f64 / h.count as f64 / scale)
}

impl DeviceState {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::NotStarted => "not_started",
            Self::Available => "available",
            Self::Unavailable => "unavailable",
        }
    }
}

impl Telemetry {
    /// 2026-09-26: The instruments, read now. Allocates (the spec rows), so it
    /// is not for a hot path.
    pub fn snapshot(&self) -> TelemetrySnapshot {
        let at_ns = self.now_ns();
        let device = self
            .device_reading()
            .map(|(reading, t, samples)| DeviceSnapshot {
                reading,
                samples,
                age_ns: at_ns.saturating_sub(t),
                read_errors: self.device_read_errors.get(),
            });
        let s = &self.sched;
        let r = &self.requests;
        let spec = self
            .spec
            .widths()
            .into_iter()
            .map(|d| SpecRow {
                drafts: d,
                steps: (0..=d).map(|a| self.spec.steps(d, a)).sum(),
                acceptance_by_depth: self.spec.acceptance_by_depth(d),
            })
            .collect();
        TelemetrySnapshot {
            level: self.level().as_str(),
            at_ns,
            device_state: self.device_state().as_str(),
            device,
            energy: EnergySnapshot {
                gpu_mj_total: self.energy_mj.get(),
                gpu_counter_mj: self.energy_counter_raw.get(),
                tokens_total: self.tokens_since_first_sample(),
                joules_per_token_live: self.joules_per_token_live.get(),
                joules_per_token_cumulative: self.joules_per_token_cumulative(),
                counter_resets: self.energy_resets.get(),
                counter_wraps: self.energy_wraps.get(),
            },
            sched: SchedSnapshot {
                shape: s.shape(),
                stream_syncs: s.stream_syncs.get(),
                blocking_d2h: s.blocking_d2h.get(),
                graph_replays: s.graph_replays.get(),
                graph_captures: s.graph_captures.get(),
            },
            spec,
            requests: RequestSnapshot {
                finished: r.finished.get(),
                ttft_mean_ms: mean(&r.ttft.snapshot(), 1e6),
                tpot_mean_ms: mean(&r.tpot.snapshot(), 1e6),
                e2e_mean_ms: mean(&r.e2e.snapshot(), 1e6),
                energy_mean_j: mean(&r.energy.snapshot(), 1e3),
                energy_unattributed: r.energy_unattributed.get(),
            },
        }
    }
}
