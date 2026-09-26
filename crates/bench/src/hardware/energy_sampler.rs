// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-26: The I/O half of in-window energy: one long-lived `nvidia-smi -lms`
//! child whose lines a background task parses, plus the per-run [`EnergyMeter`].
//!
//! Owner: bench hardware.
//! Invariants:
//! - [`EnergyMeter`] samples only a loopback target: the sampler reads the local
//!   GPU, which is the serving GPU only then.
//! - When the sampler cannot start, the meter holds no sampler, baseline or cost,
//!   so the record carries no `gpu_rail_*` keys: absent, never zero.
//!
//! One child rather than a spawn per sample: measured 2026-09-20 on GB10, a
//! single `nvidia-smi` spawn cost 10–20 ms of CPU, while the loop-mode child used
//! 0.02 s of CPU over 10 s at 250 ms. The child's CPU time is read when it is
//! stopped and goes on the record ([`SamplerCost`]).

use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{Duration, Instant};

use anyhow::{Context, Result, bail};
use parking_lot::Mutex;
use tokio::io::{AsyncBufReadExt, BufReader};

use super::energy::{
    EnergyWindow, IDLE_BASELINE_SECS, PowerSample, RAIL_NOTE, SAMPLE_PERIOD_MS, SamplerCost,
    integrate, parse_line,
};
use crate::plugin::TargetEndpoint;
use crate::result::LogLine;

/// 2026-09-26: The fields the child streams, in the column order [`parse_line`] reads.
const QUERY: &str = "--query-gpu=power.draw.average,\
                     clocks_event_reasons.sw_power_cap,\
                     clocks_event_reasons.hw_power_brake_slowdown";

/// 2026-09-26: How long [`EnergySampler::spawn`] waits for the first reading
/// before it declares the rail unreadable.
const FIRST_SAMPLE_TIMEOUT: Duration = Duration::from_secs(3);

pub struct EnergySampler {
    child: tokio::process::Child,
    pid: u32,
    started: Instant,
    samples: Arc<Mutex<Vec<PowerSample>>>,
    rejected: Arc<AtomicU64>,
}

impl EnergySampler {
    /// 2026-09-26: Start the child and wait for its first reading.
    ///
    /// `Err` when `nvidia-smi` cannot be spawned, exits before it is polled, or
    /// produces no parseable reading within `FIRST_SAMPLE_TIMEOUT`.
    /// [`EnergyMeter::start`] logs the error and the run goes on without energy.
    pub async fn spawn() -> Result<Self> {
        let mut child = tokio::process::Command::new("nvidia-smi")
            .arg(QUERY)
            .arg("--format=csv,noheader,nounits")
            .arg("-lms")
            .arg(SAMPLE_PERIOD_MS.to_string())
            .stdin(std::process::Stdio::null())
            .stdout(std::process::Stdio::piped())
            .stderr(std::process::Stdio::null())
            .kill_on_drop(true)
            .spawn()
            .context("spawning nvidia-smi for power sampling")?;
        let pid = child
            .id()
            .context("nvidia-smi exited before it was polled")?;
        let stdout = child.stdout.take().context("nvidia-smi stdout not piped")?;
        let samples = Arc::new(Mutex::new(Vec::new()));
        let rejected = Arc::new(AtomicU64::new(0));
        let (sink, counter) = (samples.clone(), rejected.clone());
        tokio::spawn(async move {
            let mut lines = BufReader::new(stdout).lines();
            while let Ok(Some(line)) = lines.next_line().await {
                let at = Instant::now();
                match parse_line(&line, at) {
                    Some(s) => sink.lock().push(s),
                    None => {
                        counter.fetch_add(1, Ordering::Relaxed);
                    }
                }
            }
        });
        let this = Self {
            child,
            pid,
            started: Instant::now(),
            samples,
            rejected,
        };
        let deadline = Instant::now() + FIRST_SAMPLE_TIMEOUT;
        while this.samples.lock().is_empty() {
            if Instant::now() >= deadline {
                let rejected = this.rejected.load(Ordering::Relaxed);
                bail!(
                    "nvidia-smi produced no power reading within {FIRST_SAMPLE_TIMEOUT:?} \
                     ({rejected} unparseable line(s)) — the GPU rail is not readable here"
                );
            }
            tokio::time::sleep(Duration::from_millis(20)).await;
        }
        Ok(this)
    }

    /// 2026-09-26: The integral over `[start, end]` of the readings so far.
    pub fn window(&self, start: Instant, end: Instant) -> Option<EnergyWindow> {
        integrate(&self.samples.lock(), start, end)
    }

    /// 2026-09-26: Sleep [`IDLE_BASELINE_SECS`] and integrate whatever the box
    /// drew meanwhile. [`EnergyMeter::start`] calls it; its caller must have the
    /// model loaded and no request in flight.
    pub async fn idle_baseline(&self) -> Option<EnergyWindow> {
        let start = Instant::now();
        tokio::time::sleep(Duration::from_secs(IDLE_BASELINE_SECS)).await;
        self.window(start, Instant::now())
    }

    /// 2026-09-26: CPU seconds the child has consumed: the first field of
    /// `/proc/<pid>/schedstat`, read as nanoseconds.
    fn cpu_seconds(&self) -> Option<f64> {
        std::fs::read_to_string(format!("/proc/{}/schedstat", self.pid))
            .ok()?
            .split_whitespace()
            .next()?
            .parse::<u64>()
            .ok()
            .map(|ns| ns as f64 / 1e9)
    }

    /// 2026-09-26: Measure the child's cost, then kill it.
    pub async fn stop(mut self) -> SamplerCost {
        let cost = SamplerCost {
            cpu_s: self.cpu_seconds(),
            wall_s: self.started.elapsed().as_secs_f64(),
            samples: self.samples.lock().len(),
            rejected_lines: self.rejected.load(Ordering::Relaxed),
        };
        if let Err(e) = self.child.kill().await {
            tracing::debug!("energy sampler already exited: {e}");
        }
        cost
    }
}

/// 2026-09-26: The energy lifecycle of one benchmark run: start the sampler
/// (loopback targets only), take the idle baseline, hand out per-window
/// integrals, stop and report the cost. The concurrency sweep and decode-floor
/// drivers hold one as a field and log the lines it returns.
///
/// When the rail is unreadable the run continues: the returned lines say why,
/// and the record carries no `gpu_rail_*` keys.
#[derive(Default)]
pub struct EnergyMeter {
    sampler: Option<EnergySampler>,
    idle: Option<EnergyWindow>,
    cost: Option<SamplerCost>,
}

impl EnergyMeter {
    /// 2026-09-26: Start sampling for `target`, then sample the idle baseline.
    /// Call with the model loaded and nothing in flight, before the first
    /// measured window.
    pub async fn start(&mut self, target: &TargetEndpoint) -> Vec<LogLine> {
        let mut lines = Vec::new();
        if !target.is_loopback() {
            lines.push(LogLine::info(format!(
                "energy: {} is not loopback — the local GPU rail is not the serving GPU, so \
                 no energy is recorded for this run",
                target.base_url
            )));
            return lines;
        }
        match EnergySampler::spawn().await {
            Ok(sampler) => {
                lines.push(LogLine::info(RAIL_NOTE));
                self.idle = sampler.idle_baseline().await;
                match &self.idle {
                    Some(idle) => lines.push(LogLine::info(format!(
                        "energy: idle baseline (model loaded, nothing in flight) — {}",
                        idle.one_line(None)
                    ))),
                    None => lines.push(LogLine::warn(
                        "energy: the idle baseline window produced no samples; totals will be \
                         recorded without an above-idle figure",
                    )),
                }
                self.sampler = Some(sampler);
            }
            Err(e) => lines.push(LogLine::warn(format!("energy: not recorded — {e:#}"))),
        }
        lines
    }

    /// 2026-09-26: The integral over one measured window, when sampling is live.
    pub fn window(&self, start: Instant, end: Instant) -> Option<EnergyWindow> {
        self.sampler.as_ref()?.window(start, end)
    }

    pub fn idle(&self) -> Option<&EnergyWindow> {
        self.idle.as_ref()
    }

    /// 2026-09-26: Stop the sampler; its cost becomes part of this meter's
    /// record keys. Returns the line to log, or `None` when nothing was running.
    pub async fn stop(&mut self) -> Option<LogLine> {
        let cost = self.sampler.take()?.stop().await;
        let line = LogLine::info(cost.one_line());
        self.cost = Some(cost);
        Some(line)
    }

    /// 2026-09-26: The run-level keys: the idle baseline and the sampler's cost.
    /// Per-window keys come from [`EnergyWindow::metrics`].
    pub fn metrics(&self, m: &mut std::collections::BTreeMap<String, f64>) {
        if let Some(idle) = &self.idle {
            m.insert("gpu_rail_idle_power_w".to_string(), idle.mean_power_w);
            m.insert(
                "gpu_rail_idle_power_samples".to_string(),
                idle.samples as f64,
            );
        }
        if let Some(cost) = &self.cost {
            cost.metrics(m);
        }
    }
}
