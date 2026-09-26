// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-26: The `vision-fidelity` state machine: one leg per `next()`, in
//! `Phase` order: calibrate (measures the template overhead every geometry
//! cell subtracts), geometry, capability probes, integrity, concurrency, the
//! no-image control, then the score. The control runs after the legs it can
//! invalidate, so their cells are shown beside a VACUOUS verdict.
//!
//! Owner: bench, vision.
//! Invariants: none beyond the types.

use crate::hardware::Sensitivity;
use std::time::{Duration, Instant};

use anyhow::{Context, Result, bail};

use crate::benchmark::{Benchmark, BenchmarkDescriptor};
use crate::benchmarks::one_line;
use crate::http;
use crate::metadata::PluginMetadata;
use crate::params::{ParamKind, ParamSpec, ParamValue, ParamValues};
use crate::plugin::{Plugin, PluginHandle};
use crate::result::{BenchmarkResult, LogLine, RunStatus, Stat, Verdict as RunVerdict};

use super::geometry::expected_vision_tokens_bounded;
use super::probes::{CONTROL, PROBES, Probe, concurrency_probe};
use super::provision::{self, FIXTURES, provision};
use super::request;
use super::score::{
    GeomCell, ProbeCell, Verdict as VisionVerdict, asserted_cells, reply_matches, verdict,
    with_runtime_checks,
};

const SUMMARY: &str = "Vision fidelity: exact vision-token geometry across a resolution ladder, \
                       plus capability probes with a no-image control.";

pub const METADATA: PluginMetadata = PluginMetadata::metrale(SUMMARY);

pub const DESCRIPTOR: BenchmarkDescriptor = BenchmarkDescriptor {
    id: "vision-fidelity",
    name: "Vision Fidelity",
    summary: SUMMARY,
    detail: "Two legs. GEOMETRY sends a ladder of committed fixtures (224² through 1280×720, \
             square, wide and portrait, deliberately mixing grid-exact sizes with ones that \
             must snap) and asserts the EXACT vision-token count from usage.prompt_tokens \
             against patch/merge arithmetic — the observable that moves when preprocessing \
             changes, and the one a capability check cannot see. CAPABILITY asks unambiguous \
             questions about those images. A no-image CONTROL runs last: if it answers as \
             though it saw a picture, the capability leg proved nothing and the run reports \
             VACUOUS rather than PASS. Images the encoder REFUSES report UNMEASURED, never \
             FAIL — that is a deployment setting, not a defect. A serve that caps vision \
             area does not refuse, it downscales: declare its bound with vision_max_pixels \
             and the ladder predicts the downscaled geometry and still asserts every rung.",
    duration_hint: "~1-2 min",
    expected_secs: 120,
    updated: "2026-08-14",
    needs_confirmation: false,
    // 2026-09-26: The predictions assume patch 16 and merge 2 and calibrate the
    // template overhead per run, so the benchmark names no one checkpoint.
    intended_for: None,
    threshold_params: &[],
    // 2026-09-26: Correctness: the legs score replies and token counts, not
    // time.
    sensitivity: Sensitivity::Correctness,
    ctor: || Box::new(VisionFidelity::default()),
};

#[derive(Default, Clone, Copy, PartialEq, Eq, Debug)]
enum Phase {
    #[default]
    Calibrate,
    Geometry,
    Probes,
    Integrity,
    Concurrency,
    Control,
    Score,
    Done,
}

#[derive(Default)]
pub struct VisionFidelity {
    handle: Option<PluginHandle>,
    phase: Phase,
    started: Option<Instant>,
    /// 2026-09-26: Chat-template cost in tokens, measured in `Calibrate` and
    /// subtracted by every geometry cell.
    overhead: Option<usize>,
    geom: Vec<GeomCell>,
    probes: Vec<ProbeCell>,
    control_held: bool,
    cursor: usize,
    conc_results: Vec<crate::benchmarks::video::concurrency::LevelResult>,
    integrity: Vec<crate::benchmarks::media_integrity::Cell>,
    max_tokens: usize,
    request_timeout_s: u64,
    /// 2026-09-26: The target serve's `--vision-max-pixels`, an area in pixels;
    /// 0 means not declared. Every geometry prediction is made under it,
    /// because a serve with a bound downscales a larger image and answers.
    vision_max_pixels: u64,
}

impl VisionFidelity {
    fn handle(&self) -> Result<&PluginHandle> {
        self.handle.as_ref().context("benchmark was not loaded")
    }

    fn elapsed(&self) -> Duration {
        self.started.map(|s| s.elapsed()).unwrap_or_default()
    }

    fn timeout(&self) -> Duration {
        Duration::from_secs(self.request_timeout_s)
    }

    fn fixture(&self, name: &str) -> Result<&'static [u8]> {
        FIXTURES
            .iter()
            .find(|(n, _, _, _)| *n == name)
            .map(|(_, b, _, _)| *b)
            .with_context(|| format!("fixture {name} is not in the provisioned set"))
    }

    fn frame(&self, phase: &str, log: Vec<LogLine>) -> BenchmarkResult {
        let mut r = BenchmarkResult::running(phase, self.elapsed());
        r.progress = Some((
            (self.geom.len() + self.probes.len()) as u64,
            (FIXTURES.len() + PROBES.len()) as u64,
        ));
        r.log = log;
        r
    }

    /// 2026-09-26: Send one probe and score it.
    async fn run_probe(&self, p: &Probe) -> ProbeCell {
        let handle = match self.handle() {
            Ok(h) => h,
            Err(e) => {
                return ProbeCell::Error {
                    id: p.id,
                    msg: one_line(format!("{e:#}")),
                };
            }
        };
        let images: Vec<&[u8]> = match p
            .images
            .iter()
            .map(|n| self.fixture(n))
            .collect::<Result<Vec<_>>>()
        {
            Ok(v) => v,
            Err(e) => {
                return ProbeCell::Error {
                    id: p.id,
                    msg: one_line(format!("{e:#}")),
                };
            }
        };
        let body = request::body(&handle.target().model, &images, p.prompt, self.max_tokens);
        match http::chat_stream(handle.target(), &body, self.timeout()).await {
            Ok(o) => {
                if reply_matches(&o.text, p.want_all, p.want_none) {
                    ProbeCell::Pass { id: p.id }
                } else {
                    ProbeCell::Fail {
                        id: p.id,
                        reply: one_line(&o.text),
                    }
                }
            }
            Err(e) => ProbeCell::Error {
                id: p.id,
                msg: one_line(format!("{e:#}")),
            },
        }
    }
}

impl Plugin for VisionFidelity {
    fn metadata(&self) -> &'static PluginMetadata {
        &METADATA
    }

    async fn load(&mut self, handle: PluginHandle) -> Result<()> {
        // 2026-09-26: Write the fixtures to disk before any leg runs, so a
        // provisioning failure is reported at load rather than mid-run.
        provision(handle.artifacts()).context("provisioning vision fixtures")?;
        self.handle = Some(handle);
        Ok(())
    }
}

impl Benchmark for VisionFidelity {
    fn descriptor(&self) -> &'static BenchmarkDescriptor {
        &DESCRIPTOR
    }

    fn parameters(&self) -> Vec<ParamSpec> {
        vec![
            ParamSpec::new(
                "max_tokens",
                "Max tokens per reply",
                "Probe replies are short by design; the geometry leg needs almost none. \
                 Keep this well above the model's thinking budget if you disable the \
                 thinking-off default, or a reasoning block will consume the whole budget \
                 and return empty content that reads as a vision failure.",
                ParamKind::Int { min: 16, max: 2048 },
                ParamValue::Int(128),
            ),
            ParamSpec::new(
                "request_timeout_s",
                "Per-request timeout (s)",
                "A large fixture at a high area bound can take a while to prefill.",
                ParamKind::Int { min: 30, max: 3600 },
                ParamValue::Int(300),
            ),
            ParamSpec::new(
                "vision_max_pixels",
                "Target serve's --vision-max-pixels (0 = not set)",
                "Set this to the SAME area the target server was started with. It is an \
                 area in pixels, not an edge length, exactly like the flag. Leave it 0 \
                 when the serve does not pass the flag — the checkpoint's own bound is \
                 far above every fixture, so nothing downscales and the ladder asserts \
                 native geometry. \
                 \n\nWhy this exists: a serve that caps vision area does NOT refuse an \
                 oversized image, it silently downscales it and answers normally. The \
                 reply then carries fewer vision tokens than a native-resolution \
                 prediction expects, and the cell fails while the model is perfectly \
                 healthy. A run against `--vision-max-pixels 262144` scored 9/14 for \
                 exactly that reason (2026-08-21) — the five fixtures above the bound. \
                 Declaring the bound here does not SKIP those rungs: the prediction is \
                 recomputed for the downscaled geometry and the rung is still asserted, \
                 so an engine that does not honour its own declared bound still fails.",
                ParamKind::Int {
                    min: 0,
                    // 2026-09-26: 4096², the long-side ceiling squared: no
                    // served image is larger, so a bigger bound could never bind.
                    max: 16_777_216,
                },
                ParamValue::Int(0),
            ),
        ]
    }

    fn configure(&mut self, values: &ParamValues) -> Result<()> {
        self.max_tokens = values.int("max_tokens")? as usize;
        self.request_timeout_s = values.int("request_timeout_s")? as u64;
        self.vision_max_pixels = values.int("vision_max_pixels")? as u64;
        if self.max_tokens == 0 {
            bail!("max_tokens must be positive");
        }
        Ok(())
    }

    async fn next(&mut self) -> Result<BenchmarkResult> {
        let handle = self.handle()?.clone();
        handle.check_cancelled()?;
        if self.started.is_none() {
            self.started = Some(Instant::now());
        }

        match self.phase {
            Phase::Calibrate => self.calibrate_leg(&handle).await,

            Phase::Geometry => self.geometry_leg(&handle).await,

            Phase::Probes => {
                let p = &PROBES[self.cursor];
                let cell = self.run_probe(p).await;
                let line = match &cell {
                    ProbeCell::Pass { id } => LogLine::info(format!("{id}: pass")),
                    ProbeCell::Fail { id, reply } => LogLine::warn(format!("{id}: {reply}")),
                    ProbeCell::Error { id, msg } => LogLine::warn(format!("{id}: {msg}")),
                };
                self.probes.push(cell);
                self.cursor += 1;
                if self.cursor >= PROBES.len() {
                    self.cursor = 0;
                    self.phase = Phase::Integrity;
                }
                Ok(self.frame("probes", vec![line]))
            }

            Phase::Integrity => self.integrity_leg().await,

            Phase::Concurrency => self.concurrency_leg().await,

            Phase::Control => {
                let cell = self.run_probe(&CONTROL).await;
                self.control_held = matches!(cell, ProbeCell::Pass { .. });
                let line = if self.control_held {
                    LogLine::info("control: no image, no answer — capability results stand")
                } else {
                    LogLine::warn(
                        "control: answered as though it saw an image — capability results are \
                         VACUOUS, the server may not be splicing vision embeddings at all",
                    )
                };
                self.phase = Phase::Score;
                Ok(self.frame("control", vec![line]))
            }

            Phase::Score => {
                self.phase = Phase::Done;
                // 2026-09-26: A measured integrity leg that did not pass, or a
                // concurrency sweep that is not clean, turns a Pass into a
                // Fail (`with_runtime_checks`).
                let integ_failed = self.integrity.iter().any(|c| c.measured() && !c.passed());
                let concurrency_clean =
                    crate::benchmarks::video::concurrency::sweep_ok(&self.conc_results);
                let v = with_runtime_checks(
                    verdict(&self.geom, &self.probes, self.control_held),
                    integ_failed,
                    concurrency_clean,
                );
                let asserted = asserted_cells(&self.geom);
                let passed = self
                    .probes
                    .iter()
                    .filter(|c| matches!(c, ProbeCell::Pass { .. }))
                    .count();

                let mut r = BenchmarkResult::running("score", self.elapsed());
                r.status = if v == VisionVerdict::Pass {
                    RunStatus::Completed
                } else {
                    RunStatus::Failed
                };
                r.summary = vec![
                    Stat::new("verdict", v.to_string(), ""),
                    Stat::new(
                        "geometry",
                        format!("{asserted}/{}", self.geom.len()),
                        "asserted",
                    ),
                    Stat::new(
                        "probes",
                        format!("{passed}/{}", self.probes.len()),
                        "passed",
                    ),
                ];
                r.metrics
                    .insert("geometry_asserted".into(), asserted as f64);
                // 2026-09-26: `geometry_asserted` counts Match and Mismatch,
                // so it drops when cells go Unmeasured; `geometry_matched`
                // counts Match only, so it drops when counts are wrong. The
                // BENCH.toml vision entries bound both (for example
                // `kernels/gb10/qwen3.6-27b/BENCH.toml`).
                let matched = self
                    .geom
                    .iter()
                    .filter(|c| matches!(c, GeomCell::Match { .. }))
                    .count();
                r.metrics.insert("geometry_matched".into(), matched as f64);
                r.metrics
                    .insert("geometry_cells".into(), self.geom.len() as f64);
                r.metrics.insert("probes_passed".into(), passed as f64);
                r.metrics
                    .insert("probes_total".into(), self.probes.len() as f64);
                r.metrics
                    .insert("control_held".into(), self.control_held as u8 as f64);
                // 2026-09-26: No BENCH.toml entry bounds these; the sweep
                // enters the verdict through `with_runtime_checks`.
                let baseline_prompt_tokens = self
                    .conc_results
                    .first()
                    .and_then(|baseline| baseline.prompt_tokens);
                let conc_clean = self
                    .conc_results
                    .iter()
                    .filter(|r| {
                        baseline_prompt_tokens.is_some_and(|baseline| r.ok_against(baseline))
                    })
                    .count();
                r.metrics
                    .insert("concurrency_levels_clean".into(), conc_clean as f64);
                let integ_passed = self.integrity.iter().filter(|c| c.passed()).count();
                let integ_measured = self.integrity.iter().filter(|c| c.measured()).count();
                r.metrics
                    .insert("integrity_passed".into(), integ_passed as f64);
                r.metrics
                    .insert("integrity_measured".into(), integ_measured as f64);
                for lr in &self.conc_results {
                    r.metrics
                        .insert(format!("conc_{}_wall_ms", lr.conc), lr.wall_ms as f64);
                }
                r.verdict = Some(match v {
                    VisionVerdict::Pass => RunVerdict::pass(format!(
                        "{asserted} geometry cells matched, {passed}/{} probes, control held",
                        self.probes.len()
                    )),
                    VisionVerdict::Fail => {
                        // 2026-09-26: The reason names up to five geometry
                        // mismatches or errors, not only the counts.
                        let mism: Vec<String> = self
                            .geom
                            .iter()
                            .filter_map(|c| match c {
                                GeomCell::Mismatch { fixture, want, got } => {
                                    Some(format!("{fixture} want {want} got {got}"))
                                }
                                GeomCell::Error { fixture, msg } => {
                                    Some(format!("{fixture} ERROR {msg}"))
                                }
                                _ => None,
                            })
                            .collect();
                        let mut reason = format!(
                            "{matched}/{asserted} geometry matched, {passed}/{} probes",
                            self.probes.len()
                        );
                        if !mism.is_empty() {
                            reason.push_str(": ");
                            reason.push_str(&mism[..mism.len().min(5)].join("; "));
                            if mism.len() > 5 {
                                reason.push_str(&format!(" (+{} more)", mism.len() - 5));
                            }
                        }
                        // 2026-09-26: With no declared bound and every mismatch
                        // below its prediction, the reason suggests setting
                        // `vision_max_pixels`, since a serve with a bound
                        // downscales.
                        let all_low = self.geom.iter().all(|c| match c {
                            GeomCell::Mismatch { want, got, .. } => got < want,
                            _ => true,
                        });
                        if self.vision_max_pixels == 0 && !mism.is_empty() && all_low {
                            reason.push_str(
                                " — every mismatch reads LOW: if the target serve passes \
                                 --vision-max-pixels, set the vision_max_pixels param to the \
                                 same value",
                            );
                        }
                        RunVerdict::fail(reason)
                    }
                    VisionVerdict::Vacuous => RunVerdict::fail(
                        "VACUOUS: the no-image control answered as though it saw one, so the \
                         capability probes are not evidence"
                            .to_string(),
                    ),
                });
                r.log = vec![match v {
                    VisionVerdict::Pass => LogLine::info(format!(
                        "PASS — {asserted} geometry cells asserted, {passed}/{} probes",
                        self.probes.len()
                    )),
                    VisionVerdict::Fail => LogLine::warn("FAIL — see the cells above"),
                    VisionVerdict::Vacuous => LogLine::warn(
                        "VACUOUS — the no-image control answered, so the capability probes are \
                         not evidence. Geometry results above are still valid.",
                    ),
                }];
                Ok(r)
            }

            Phase::Done => {
                let mut r = BenchmarkResult::running("done", self.elapsed());
                r.status = RunStatus::Completed;
                Ok(r)
            }
        }
    }
}

#[path = "driver_ladder.rs"]
mod ladder;

#[path = "driver_integrity.rs"]
mod integrity;
