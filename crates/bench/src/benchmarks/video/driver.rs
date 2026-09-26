// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-26: The `video-fidelity` state machine: one leg per `next()`, in
//! `Phase` order: geometry, order, parity, mixed media, integrity,
//! concurrency, the no-video control, then the score. The control runs after
//! the legs it can invalidate, so their cells are shown beside a VACUOUS
//! verdict.
//!
//! Owner: bench, video.
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

use super::concurrency::{LEVELS, LevelResult, run_level};
use super::geometry::{check_proportional, tokens_per_group};
use super::provision::{CLIPS, Clip, clip, provision};
use super::request;
use super::score::{
    self, CountCell, OrderCell, Verdict as VideoVerdict, asserted, order_matches, passed, verdict,
};

const SUMMARY: &str = "Video fidelity: temporal-order reading of a color sequence, group-count \
                       geometry, MP4/GIF backend parity, and a no-video control.";

pub const METADATA: PluginMetadata = PluginMetadata::metrale(SUMMARY);

pub const DESCRIPTOR: BenchmarkDescriptor = BenchmarkDescriptor {
    id: "video-fidelity",
    name: "Video Fidelity",
    summary: SUMMARY,
    detail: "Seven check groups over clips of solid colors, one per second. ORDER sends the same \
             sequence forwards and REVERSED and requires the answer to reverse with it — the \
             only assertion that separates 'the frames arrived in order' from 'something \
             arrived', and the one that caught the splice defect where video pad tokens \
             received no encoder rows at all and the model calmly described a gray field while \
             every token count was perfect. GEOMETRY asserts that a clip of twice the duration \
             costs twice the temporal groups, stated as a RATIO so it holds at any \
             --video-fps the server was started with. PARITY requires an MP4 and an identical \
             GIF to produce the same geometry, one through ffmpeg and one through the \
             in-process decoder. MIXED sends an image and a video together, exercising the \
             ordering contract between collection, template markers and pad expansion. \
             INTEGRITY varies media order, request history, and opposite clips in flight. \
             CONCURRENCY requires correct replies and the same prompt-token geometry at \
             C=1, C=2, and C=4. A \
             no-video CONTROL runs last: if it describes a clip it never received, the run is \
             VACUOUS rather than PASS. Legs needing a decoder the server lacks are SKIPPED, \
             never failed — that is a deployment choice.",
    duration_hint: "~1-2 min",
    expected_secs: 70,
    updated: "2026-08-24",
    needs_confirmation: false,
    intended_for: None,
    threshold_params: &[],
    // 2026-09-26: Correctness: the legs score replies and token counts, not
    // time.
    sensitivity: Sensitivity::Correctness,
    ctor: || Box::new(VideoFidelity::default()),
};

#[derive(Default, Clone, Copy, PartialEq, Eq, Debug)]
enum Phase {
    #[default]
    Geometry,
    Order,
    Parity,
    Mixed,
    Integrity,
    Concurrency,
    Control,
    Score,
    Done,
}

#[derive(Default)]
pub struct VideoFidelity {
    handle: Option<PluginHandle>,
    phase: Phase,
    started: Option<Instant>,
    order: Vec<OrderCell>,
    counts: Vec<CountCell>,
    control_held: bool,
    /// 2026-09-26: Prompt tokens of the 4 s MP4 from the geometry leg, reused
    /// by the parity leg so it sends only the GIF.
    full_tokens: Option<usize>,
    /// 2026-09-26: Merged tokens per temporal group of a 224x224 clip.
    plane: usize,
    cursor: usize,
    conc_results: Vec<LevelResult>,
    integrity: Vec<crate::benchmarks::media_integrity::Cell>,
    max_tokens: usize,
    request_timeout_s: u64,
}

/// 2026-09-26: Every color any clip shows (`the_palette_covers_every_fixture_color`).
/// The scorer looks only for these words.
const PALETTE: &[&str] = &["red", "green", "blue", "yellow"];

/// 2026-09-26: Reclassify an `Error` cell as `Skipped` when its message is the
/// server refusing a container it cannot decode.
///
/// The two-clip concurrency leg gets its cell from
/// `media_integrity::heterogeneous_concurrency`, which knows nothing of video
/// decoders, so the check is made here. The other legs that send an MP4 check
/// `is_decoder_unavailable` at their own error site.
fn skip_if_decoder_unavailable(
    cell: crate::benchmarks::media_integrity::Cell,
) -> crate::benchmarks::media_integrity::Cell {
    use crate::benchmarks::media_integrity::Cell;
    match cell {
        Cell::Error { id, msg } if request::is_decoder_unavailable(&msg) => {
            Cell::Skipped { id, why: msg }
        }
        other => other,
    }
}

impl VideoFidelity {
    fn handle(&self) -> Result<&PluginHandle> {
        self.handle.as_ref().context("benchmark was not loaded")
    }

    fn elapsed(&self) -> Duration {
        self.started.map(|s| s.elapsed()).unwrap_or_default()
    }

    fn timeout(&self) -> Duration {
        Duration::from_secs(self.request_timeout_s)
    }

    fn frame(&self, phase: &str, log: Vec<LogLine>) -> BenchmarkResult {
        let mut r = BenchmarkResult::running(phase, self.elapsed());
        r.progress = Some((
            (self.order.len() + self.counts.len()) as u64,
            (CLIPS.len() + 2) as u64,
        ));
        r.log = log;
        r
    }

    /// 2026-09-26: Ask about one clip and score the ordered colors it reports.
    async fn read_order(&self, c: &'static Clip) -> OrderCell {
        let h = match self.handle() {
            Ok(h) => h,
            Err(e) => {
                return OrderCell::Error {
                    clip: c.name,
                    msg: one_line(format!("{e:#}")),
                };
            }
        };
        let body = request::video_body(
            &h.target().model,
            c.mime,
            c.bytes,
            request::ORDER_PROMPT,
            self.max_tokens,
        );
        match http::chat_stream(h.target(), &body, self.timeout()).await {
            Ok(out) => {
                let reply = out.text.trim().to_string();
                if order_matches(&reply, c.colors, PALETTE) {
                    OrderCell::Match {
                        clip: c.name,
                        seen: one_line(reply),
                    }
                } else {
                    let got = super::score::colors_in_order(&reply, PALETTE);
                    if got.is_empty() {
                        OrderCell::NotSeen {
                            clip: c.name,
                            reply: one_line(reply),
                        }
                    } else {
                        OrderCell::WrongOrder {
                            clip: c.name,
                            want: c.colors.join(", "),
                            got: got.join(", "),
                        }
                    }
                }
            }
            Err(e) => {
                let msg = format!("{e:#}");
                if request::is_decoder_unavailable(&msg) {
                    OrderCell::Skipped {
                        clip: c.name,
                        why: one_line(msg),
                    }
                } else {
                    OrderCell::Error {
                        clip: c.name,
                        msg: one_line(msg),
                    }
                }
            }
        }
    }

    /// 2026-09-26: Prompt tokens for one clip, or `(skip, reason)` when it could
    /// not be measured; `skip` is true for a decoder-unavailable error.
    async fn tokens_for(&self, c: &'static Clip) -> std::result::Result<usize, (bool, String)> {
        let h = self.handle().map_err(|e| (false, format!("{e:#}")))?;
        let body = request::video_body(&h.target().model, c.mime, c.bytes, "Reply with OK.", 8);
        match http::chat_stream(h.target(), &body, self.timeout()).await {
            Ok(out) => Ok(out.prompt_tokens),
            Err(e) => {
                let msg = format!("{e:#}");
                let skip = request::is_decoder_unavailable(&msg);
                Err((skip, one_line(msg)))
            }
        }
    }
}

impl Plugin for VideoFidelity {
    fn metadata(&self) -> &'static PluginMetadata {
        &METADATA
    }

    async fn load(&mut self, handle: PluginHandle) -> Result<()> {
        // 2026-09-26: Write the clips to disk before any leg runs, so a
        // provisioning failure is reported at load rather than mid-run.
        provision(handle.artifacts()).context("provisioning video fixtures")?;
        self.handle = Some(handle);
        Ok(())
    }
}

impl Benchmark for VideoFidelity {
    fn descriptor(&self) -> &'static BenchmarkDescriptor {
        &DESCRIPTOR
    }

    fn parameters(&self) -> Vec<ParamSpec> {
        vec![
            ParamSpec::new(
                "max_tokens",
                "Max tokens per reply",
                "The color list is short by design, so this only needs to be generous \
                 enough that a reply is not truncated mid-sequence. Keep it well above the \
                 model's thinking budget if you re-enable thinking, or a reasoning block \
                 consumes the whole budget and returns empty content that reads as a video \
                 failure and is not one. The same trap has a thinking-OFF form, which is \
                 why the default is 320 and not the 120 it was: a cell that sends TWO \
                 media items can provoke a preamble the single-item cells never see, and \
                 a budget that truncates the preamble truncates the answer with it.",
                ParamKind::Int { min: 16, max: 2048 },
                // 2026-09-26: Measured 2026-08-15 on qwen3.8-27B-NVFP4: the
                // video-before-image reply needed 153 completion tokens (a
                // preamble, then the colors) and stopped at `finish_reason=length`
                // under a 120-token cap, losing the fourth color; the video-only
                // reply needed 8. A cell that stops at EOS earlier is not slowed
                // by a larger cap.
                ParamValue::Int(320),
            ),
            ParamSpec::new(
                "request_timeout_s",
                "Per-request timeout (s)",
                "Video prefill is heavier than an image's: a clip costs its whole patch \
                 grid once per temporal group, so a long clip at a high --video-fps is \
                 several images' worth of work.",
                ParamKind::Int { min: 30, max: 3600 },
                ParamValue::Int(300),
            ),
        ]
    }

    fn configure(&mut self, values: &ParamValues) -> Result<()> {
        self.max_tokens = values.int("max_tokens")? as usize;
        self.request_timeout_s = values.int("request_timeout_s")? as u64;
        if self.max_tokens == 0 {
            bail!("max_tokens must be positive");
        }
        // 2026-09-26: Every clip is 224x224, so one figure covers them all.
        self.plane = tokens_per_group(224, 224, 16, 2) as usize;
        self.started = Some(Instant::now());
        Ok(())
    }

    async fn next(&mut self) -> Result<BenchmarkResult> {
        match self.phase {
            Phase::Geometry => self.geometry_leg().await,

            Phase::Order => {
                let ordered: Vec<&'static Clip> =
                    CLIPS.iter().filter(|c| c.colors.len() == 4).collect();
                let c = ordered[self.cursor];
                let cell = self.read_order(c).await;
                let line = match &cell {
                    OrderCell::Match { seen, .. } => LogLine::info(format!("{}: {seen}", c.name)),
                    OrderCell::WrongOrder { want, got, .. } => {
                        LogLine::warn(format!("{}: wanted [{want}], got [{got}]", c.name))
                    }
                    OrderCell::NotSeen { reply, .. } => {
                        LogLine::warn(format!("{}: no colors named — {reply}", c.name))
                    }
                    OrderCell::Skipped { why, .. } => {
                        LogLine::info(format!("{}: skipped — {why}", c.name))
                    }
                    OrderCell::Error { msg, .. } => LogLine::warn(format!("{}: {msg}", c.name)),
                };
                self.order.push(cell);
                self.cursor += 1;
                if self.cursor >= ordered.len() {
                    self.cursor = 0;
                    self.phase = Phase::Parity;
                }
                Ok(self.frame("order", vec![line]))
            }

            Phase::Parity => self.parity_leg().await,

            Phase::Mixed => self.mixed_leg().await,

            Phase::Integrity => self.integrity_leg().await,

            Phase::Concurrency => self.concurrency_leg().await,

            Phase::Control => {
                self.phase = Phase::Score;
                let h = self.handle()?;
                let body = request::text_only_body(
                    &h.target().model,
                    request::ORDER_PROMPT,
                    self.max_tokens,
                );
                let line = match http::chat_stream(h.target(), &body, self.timeout()).await {
                    Ok(out) => {
                        let reply = out.text.trim();
                        // 2026-09-26: The control fails only when the reply
                        // names every palette color.
                        let looks_seen =
                            super::score::colors_in_order(reply, PALETTE).len() >= PALETTE.len();
                        self.control_held = !looks_seen;
                        if self.control_held {
                            LogLine::info(format!(
                                "control: no video, no full sequence — readings stand ({})",
                                one_line(reply.chars().take(60).collect::<String>())
                            ))
                        } else {
                            LogLine::warn(format!(
                                "control: named every color with NO video attached — the \
                                 readings are not evidence ({})",
                                one_line(reply.chars().take(60).collect::<String>())
                            ))
                        }
                    }
                    Err(e) => {
                        // 2026-09-26: A control that could not run does not
                        // hold.
                        self.control_held = false;
                        LogLine::warn(format!("control failed: {}", one_line(format!("{e:#}"))))
                    }
                };
                Ok(self.frame("control", vec![line]))
            }

            Phase::Score => {
                self.phase = Phase::Done;
                let v = verdict(&self.order, &self.counts, self.control_held);
                let asserted_n = asserted(&self.order, &self.counts);
                let passed_n = passed(&self.order, &self.counts);
                let skipped = self
                    .order
                    .iter()
                    .filter(|c| matches!(c, OrderCell::Skipped { .. }))
                    .count()
                    + self
                        .counts
                        .iter()
                        .filter(|c| matches!(c, CountCell::Skipped { .. }))
                        .count();

                let mut r = BenchmarkResult::running("score", self.elapsed());
                r.status = if v == VideoVerdict::Pass {
                    RunStatus::Completed
                } else {
                    RunStatus::Failed
                };
                r.summary = vec![
                    Stat::new("verdict", v.to_string(), ""),
                    Stat::new("legs", format!("{passed_n}/{asserted_n}"), "passed"),
                    Stat::new("skipped", skipped.to_string(), "legs"),
                ];
                r.metrics.insert("legs_passed".into(), passed_n as f64);
                r.metrics.insert("legs_asserted".into(), asserted_n as f64);
                r.metrics.insert("legs_skipped".into(), skipped as f64);
                r.metrics
                    .insert("control_held".into(), self.control_held as u8 as f64);
                for lr in &self.conc_results {
                    r.metrics
                        .insert(format!("conc_{}_wall_ms", lr.conc), lr.wall_ms as f64);
                    r.metrics
                        .insert(format!("conc_{}_correct", lr.conc), lr.correct as f64);
                }
                r.verdict = Some(match v {
                    VideoVerdict::Pass => RunVerdict::pass(format!(
                        "{passed_n}/{asserted_n} legs passed, control held, {skipped} skipped"
                    )),
                    VideoVerdict::Fail => RunVerdict::fail(format!(
                        "{passed_n}/{asserted_n} legs passed, {skipped} skipped"
                    )),
                    VideoVerdict::Vacuous => RunVerdict::fail(
                        "VACUOUS: the no-video control named the whole color sequence, so the \
                         readings are not evidence"
                            .to_string(),
                    ),
                    VideoVerdict::Inconclusive => RunVerdict::fail(
                        "INCONCLUSIVE: every leg was skipped — no video decoder is available, \
                         so nothing was measured"
                            .to_string(),
                    ),
                });
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

#[path = "driver_counts.rs"]
mod counts;

#[path = "driver_media.rs"]
mod media;

#[cfg(test)]
#[path = "driver_tests.rs"]
mod driver_tests;
