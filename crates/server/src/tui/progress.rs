// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-26: `ProgressModel`, the Main tab's startup state, fed [`ProgressEvent`]s from the capture layer.
//!
//! Owner: server tui.
//! Invariants:
//! - `phases` has one entry per `PHASE_NAMES` entry, in the same order.
//! - A phase never leaves `Done` until `reset`, and only a `Pending` phase
//!   becomes `Running`.

use std::time::Instant;

use super::capture_layer::ProgressEvent;

/// 2026-09-26: Display names of the startup phases. The index is the first
/// argument of the `metrale_telemetry::progress::phase` calls in
/// `main_modules` (`serve`, `serve_load`, `serve_router`, `model_swap`).
pub const PHASE_NAMES: [&str; 12] = [
    "banner",
    "model resolve",
    "config",
    "gpu init",
    "topology",
    "weight load",
    "kv cache",
    "kernel audit",
    "tokenizer",
    "scheduler",
    "router",
    "listening",
];

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum PhaseState {
    Pending,
    Running,
    Done,
}

#[derive(Clone, Debug)]
pub struct Phase {
    pub name: &'static str,
    pub state: PhaseState,
    pub started: Option<Instant>,
    pub secs: f64,
}

/// 2026-09-26: Startup progress, ready to render.
pub struct ProgressModel {
    pub phases: Vec<Phase>,
    pub started_at: Instant,
    /// 2026-09-26: GB of weights on disk, from the preflight event.
    pub disk_gb: f64,
    pub shard: u64,
    pub shard_total: u64,
    pub shard_name: String,
    pub layer: u64,
    pub layer_total: u64,
    pub gpu_used_gb: f64,
    pub gpu_free_gb: f64,
    /// 2026-09-26: GPU-used GB x10 per finished shard, the newest 64, for the
    /// MEM sparkline.
    pub mem_history: Vec<u64>,
    /// 2026-09-26: Set by the `Ready` event, with the port it reports.
    pub ready: bool,
    pub port: u16,
    pub ready_in_secs: f64,
    /// 2026-09-26: The weight-load window that GB/s is measured over.
    /// `load_started` is stamped by the first `ShardStart`, so startup before
    /// the weights is not in the divisor. `load_secs` is fixed when the last
    /// shard finishes (or at `Ready`), so a finished load's rate stays put.
    load_started: Option<Instant>,
    load_secs: Option<f64>,
    /// 2026-09-26: Displayed bar fractions. `disp_overall` eases in
    /// `ease_tick`; `disp_shard` is set to 0 at a new shard and 1 when it is
    /// done.
    disp_overall: f64,
    disp_shard: f64,
    last_shard_seen: u64,
}

impl Default for ProgressModel {
    fn default() -> Self {
        Self {
            phases: PHASE_NAMES
                .iter()
                .map(|n| Phase {
                    name: n,
                    state: PhaseState::Pending,
                    started: None,
                    secs: 0.0,
                })
                .collect(),
            started_at: Instant::now(),
            disk_gb: 0.0,
            shard: 0,
            shard_total: 0,
            shard_name: String::new(),
            layer: 0,
            layer_total: 0,
            gpu_used_gb: 0.0,
            gpu_free_gb: 0.0,
            mem_history: Vec::new(),
            ready: false,
            port: 0,
            ready_in_secs: 0.0,
            load_started: None,
            load_secs: None,
            disp_overall: 0.0,
            disp_shard: 0.0,
            last_shard_seen: 0,
        }
    }
}

impl ProgressModel {
    /// 2026-09-26: Start over for a new model load, by reassigning
    /// `Self::default()` so no field is missed. Needed because `enter_phase`
    /// never moves a `Done` phase back, `ready` is never cleared, and
    /// `freeze_load_window` keeps the first close.
    pub fn reset(&mut self) {
        *self = Self::default();
    }

    pub fn apply(&mut self, ev: ProgressEvent) {
        match ev {
            ProgressEvent::Phase { phase, .. } => self.enter_phase(phase as usize),
            ProgressEvent::Preflight { disk_gb, free_gb } => {
                self.disk_gb = disk_gb;
                self.gpu_free_gb = free_gb;
            }
            ProgressEvent::ShardStart { shard, total, name } => {
                // 2026-09-26: The first shard opens the load window.
                self.load_started.get_or_insert_with(Instant::now);
                self.shard = shard;
                self.shard_total = total;
                self.shard_name = name;
                if shard != self.last_shard_seen {
                    // 2026-09-26: A new shard resets the shard bar to 0.
                    self.disp_shard = 0.0;
                    self.last_shard_seen = shard;
                }
            }
            ProgressEvent::ShardDone {
                shard,
                total,
                used_gb,
                free_gb,
            } => {
                self.shard = shard;
                self.shard_total = total;
                self.gpu_used_gb = used_gb;
                self.gpu_free_gb = free_gb;
                self.mem_history.push((used_gb * 10.0) as u64);
                if self.mem_history.len() > 64 {
                    self.mem_history.remove(0);
                }
                self.disp_shard = 1.0;
                // 2026-09-26: The shard numbered `total` closes the window.
                if total > 0 && shard >= total {
                    self.freeze_load_window();
                }
            }
            ProgressEvent::Layer { layer, total } => {
                self.layer = layer;
                self.layer_total = total;
            }
            ProgressEvent::Ready { port } => {
                self.ready = true;
                self.port = port;
                self.ready_in_secs = self.started_at.elapsed().as_secs_f64();
                // 2026-09-26: Closes the window if no final `ShardDone` did.
                self.freeze_load_window();
                for p in &mut self.phases {
                    if p.state != PhaseState::Done {
                        Self::finish(p);
                    }
                }
            }
        }
    }

    fn enter_phase(&mut self, idx: usize) {
        for (i, p) in self.phases.iter_mut().enumerate() {
            match i.cmp(&idx) {
                std::cmp::Ordering::Less => {
                    if p.state != PhaseState::Done {
                        Self::finish(p);
                    }
                }
                std::cmp::Ordering::Equal => {
                    if p.state == PhaseState::Pending {
                        p.state = PhaseState::Running;
                        p.started = Some(Instant::now());
                    }
                }
                std::cmp::Ordering::Greater => {}
            }
        }
    }

    fn finish(p: &mut Phase) {
        if let Some(s) = p.started {
            p.secs = s.elapsed().as_secs_f64();
        }
        p.state = PhaseState::Done;
    }

    /// 2026-09-26: Overall weight-load fraction: finished shards over total
    /// shards. With no shard total it is 1 once ready, else 0.
    pub fn overall_target(&self) -> f64 {
        if self.shard_total == 0 {
            return if self.ready { 1.0 } else { 0.0 };
        }
        (self.shard as f64 / self.shard_total as f64).clamp(0.0, 1.0)
    }

    pub fn shard_target(&self) -> f64 {
        // 2026-09-26: No event reports progress within a shard, so the bar is
        // 0 at `ShardStart` and 1 at `ShardDone`.
        self.disp_shard
    }

    /// 2026-09-26: Ease the overall bar one tick toward its target:
    /// d += (t-d)*0.35, snapping when within 0.002.
    pub fn ease_tick(&mut self) {
        let t = self.overall_target();
        self.disp_overall += (t - self.disp_overall) * 0.35;
        if (t - self.disp_overall).abs() < 0.002 {
            self.disp_overall = t;
        }
    }

    pub fn displayed_overall(&self) -> f64 {
        self.disp_overall
    }

    /// 2026-09-26: Stop the load clock, keeping the first close.
    fn freeze_load_window(&mut self) {
        if self.load_secs.is_none()
            && let Some(t) = self.load_started
        {
            self.load_secs = Some(t.elapsed().as_secs_f64());
        }
    }

    /// 2026-09-26: Seconds the weight load took, once it is over; `None` while
    /// still loading.
    pub fn load_secs(&self) -> Option<f64> {
        self.load_secs
    }

    /// 2026-09-26: Load rate (GB/s) and ETA seconds over the weight-load
    /// window. `GB done` is `disk_gb` times the shard fraction. Once the
    /// window is closed the rate is fixed.
    pub fn rate_eta(&self) -> Option<(f64, f64)> {
        if self.disk_gb <= 0.0 || self.shard == 0 || self.shard_total == 0 {
            return None;
        }
        let elapsed = match self.load_secs {
            Some(s) => s,
            None => self.load_started?.elapsed().as_secs_f64(),
        }
        .max(0.1);
        let frac = (self.shard as f64 / self.shard_total as f64).clamp(0.0, 1.0);
        let gb_done = self.disk_gb * frac;
        let rate = gb_done / elapsed;
        if rate <= 0.0 {
            return None;
        }
        Some((rate, (self.disk_gb - gb_done) / rate))
    }

    /// 2026-09-26: For the panel title: (phases done, phases total, seconds
    /// since `started_at`).
    pub fn phase_counts(&self) -> (usize, usize, f64) {
        let done = self
            .phases
            .iter()
            .filter(|p| p.state == PhaseState::Done)
            .count();
        let secs = self.started_at.elapsed().as_secs_f64();
        (done, self.phases.len(), secs)
    }
}

#[cfg(test)]
#[path = "progress_tests.rs"]
mod tests;
