// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-26: The SSM state poisoning gate's driver: replay the probe script
//! against one server, compare every replay with round 0, then run the
//! tool-call path probe.
//!
//! Owner: bench, SSM poisoning gate.
//! Invariants:
//! - A replay round that errors is scored
//!   [`super::compare::RoundVerdict::Unmeasured`]; the run goes on.
//! - The run fails on a reference round that misses the script anchors, on a
//!   short run, on any collapsed, unmeasured or zero-cache replay, and on a
//!   tool call that differs between predecessor paths.

use crate::hardware::Sensitivity;
use std::future::Future;
use std::time::{Duration, Instant};

use anyhow::{Context, Result, bail};
use serde_json::{Value, json};

use crate::benchmark::{Benchmark, BenchmarkDescriptor};
use crate::benchmarks::{one_line, transcript::Transcript};
use crate::http;
use crate::metadata::PluginMetadata;
use crate::params::{ParamKind, ParamSpec, ParamValue, ParamValues};
use crate::plugin::{Plugin, PluginHandle};
use crate::result::{BenchmarkResult, LogLine, RunStatus, Verdict};

use super::compare::{self, RoundVerdict};
use super::probe;
use super::score::RoundRecord;
use super::toolcall::{self, Path, PathResult};

const SUMMARY: &str = "Replayed conversations must come back byte-identical";
pub const METADATA: PluginMetadata = PluginMetadata::metrale(SUMMARY);

/// 2026-09-26: The default replay count. The gate's BENCH.toml entry bounds the
/// recorded `rounds` metric at exactly 12.
pub const DEFAULT_ROUNDS: usize = 12;

pub const DESCRIPTOR: BenchmarkDescriptor = BenchmarkDescriptor {
    id: "ssm-state-poisoning-gate",
    name: "SSM State Poisoning Gate",
    summary: SUMMARY,
    detail: "Replays a fixed 4-turn conversation script 12 times against one prefix-cached \
             server at temperature 0, comparing every turn against the first round. It splits \
             two failure classes: restore JITTER (same finish reason, length within bounds) is \
             a healthy engine property — Marconi restores the same token from alternating \
             anchors between rounds, so accumulation differs and turns 2-4 come back reworded — \
             and is recorded but passed; restore POISONING collapses the output (early-EOS \
             stubs or runaway, the exact signature the batch4 stack shipped 2026-08-11) and \
             FAILS the gate. Any collapsed or unmeasured round fails; jitter only records. \
             Two self-honesty checks keep the gate from passing vacuously: the reference \
             round must satisfy the script's semantic anchors (poisoning deterministic from \
             round 0 would otherwise certify as Invariant), and every replay's first turn \
             must attest a nonzero prefix-cache restore (with caching off, the gate would \
             green-light a path it never exercised). Serves with --serve-override \
             ssm_cache_slots=256 so the snapshot pool cannot evict mid-run (churn is noise).",
    duration_hint: "~8–10 min",
    expected_secs: 150,
    updated: "2026-08-13",
    needs_confirmation: false,
    // 2026-09-26: Not pinned to a checkpoint: replay stability is asked of any
    // served model.
    intended_for: None,
    threshold_params: &[],
    // 2026-09-26: The verdict reads transcripts and cache counts, never timing.
    sensitivity: Sensitivity::Correctness,
    ctor: || Box::new(SsmPoison::default()),
};

#[derive(Default, Clone, Copy, PartialEq, Eq, Debug)]
enum Phase {
    #[default]
    Baseline,
    Replay,
    /// 2026-09-26: Path-independence of tool calls: one target turn, reached by
    /// each `toolcall::Path`.
    ToolPath,
    Score,
    Done,
}

#[derive(Default)]
pub struct SsmPoison {
    handle: Option<PluginHandle>,
    phase: Phase,
    rounds: usize,
    /// 2026-09-26: Divergences found by the tool-call path probe.
    tool_divergences: Vec<toolcall::Divergence>,
    /// 2026-09-26: Whether the direct path called a tool on the target. Logged;
    /// not part of the verdict.
    tool_reference_called: bool,
    max_tokens: usize,
    timeout: Duration,
    started: Option<Instant>,
    probed: bool,
    /// 2026-09-26: Round 0's transcripts, one per turn: the reference.
    reference: Vec<Transcript>,
    /// 2026-09-26: One record per replay round: verdict plus the turn-1 cache
    /// count the vacuity check reads.
    replays: Vec<RoundRecord>,
}

impl SsmPoison {
    fn handle(&self) -> Result<&PluginHandle> {
        self.handle.as_ref().context("benchmark was not loaded")
    }

    fn elapsed(&self) -> Duration {
        self.started.map(|s| s.elapsed()).unwrap_or_default()
    }

    /// 2026-09-26: probe + reference + N replays + tool paths + score.
    fn total_steps(&self) -> u64 {
        self.rounds as u64 + 4
    }

    fn steps_done(&self) -> u64 {
        match self.phase {
            Phase::Baseline => 1,
            Phase::Replay => 2 + self.replays.len() as u64,
            Phase::ToolPath => 2 + self.rounds as u64,
            Phase::Score => 3 + self.rounds as u64,
            Phase::Done => self.total_steps(),
        }
    }

    fn frame(&self, phase: &str, line: Option<LogLine>) -> BenchmarkResult {
        let mut f = BenchmarkResult::running(phase, self.elapsed())
            .with_progress(self.steps_done(), self.total_steps());
        if let Some(line) = line {
            f = f.log_line(line);
        }
        f
    }

    /// 2026-09-26: One chat turn against the endpoint. The error is returned,
    /// so the caller decides between a failed round and a failed run.
    async fn turn(&self, messages: &[Value]) -> Result<Transcript> {
        let handle = self.handle()?;
        let target = handle.target();
        let body = probe::request_body(&target.model, messages, self.max_tokens);
        let outcome = http::chat_stream(target, &body, self.timeout)
            .await
            .context("chat request failed")?;
        Ok(Transcript::from(&outcome))
    }

    /// 2026-09-26: Replay the whole script from scratch. Returns the per-turn
    /// transcripts, or the first error that stopped the replay.
    async fn replay_script(&self, label: &str) -> Result<Vec<Transcript>> {
        let handle = self.handle()?.clone();
        let mut messages: Vec<Value> = Vec::with_capacity(probe::TURNS.len() * 2);
        let mut transcripts = Vec::with_capacity(probe::TURNS.len());
        for (i, turn) in probe::TURNS.iter().enumerate() {
            handle.check_cancelled()?;
            let content = if i == 0 {
                probe::first_turn()
            } else {
                turn.to_string()
            };
            messages.push(json!({"role": "user", "content": content}));
            handle.status(format!(
                "{label} · turn {}/{turns}",
                i + 1,
                turns = probe::TURNS.len()
            ));
            let t = self.turn(&messages).await?;
            messages.push(json!({"role": "assistant", "content": t.text.clone()}));
            transcripts.push(t);
        }
        Ok(transcripts)
    }

    /// 2026-09-26: Issue the target turn along each `Path`. The user turns of
    /// the paths differ only in the interposed one.
    async fn run_tool_paths(&self) -> Result<Vec<PathResult>> {
        let handle = self.handle()?.clone();
        let mut out = Vec::with_capacity(Path::ALL.len());
        for path in Path::ALL {
            handle.check_cancelled()?;
            handle.status(format!("tool path · {}", path.label()));
            let mut messages: Vec<Value> =
                vec![json!({"role": "user", "content": probe::first_turn()})];
            let ack = self.tool_turn(&messages).await?;
            messages.push(json!({"role": "assistant", "content": ack.text}));
            if let Some(interposed) = path.interposed() {
                messages.push(json!({"role": "user", "content": interposed}));
                let t = self.tool_turn(&messages).await?;
                // 2026-09-26: The interposed turn's calls are not compared, and
                // only its text is appended to the history.
                messages.push(json!({"role": "assistant", "content": t.text}));
            }
            messages.push(json!({"role": "user", "content": toolcall::TARGET}));
            let target_reply = self.tool_turn(&messages).await?;
            out.push(PathResult {
                path,
                target: target_reply,
            });
        }
        Ok(out)
    }

    /// 2026-09-26: One turn of the tool-path probe: same transport as the replay
    /// probe, but the body offers tools and leaves the choice to the model.
    async fn tool_turn(&self, messages: &[Value]) -> Result<Transcript> {
        let handle = self.handle()?;
        let target = handle.target();
        let body = toolcall::request_body(&target.model, messages, self.max_tokens);
        let outcome = http::chat_stream(target, &body, self.timeout)
            .await
            .context("tool-path chat request failed")?;
        Ok(Transcript::from(&outcome))
    }

    /// 2026-09-26: The replay score and verdict (`score::score`,
    /// `score::verdict`); `next` adds the tool-path finding.
    pub(super) fn scored(&self) -> (super::score::Score, Verdict) {
        let s = super::score::score(&self.replays);
        let v = super::score::verdict(&s, self.rounds);
        (s, v)
    }
}

impl Plugin for SsmPoison {
    fn metadata(&self) -> &'static PluginMetadata {
        &METADATA
    }

    fn load(&mut self, handle: PluginHandle) -> impl Future<Output = Result<()>> + Send {
        self.handle = Some(handle);
        self.started = Some(Instant::now());
        async { Ok(()) }
    }
}

impl Benchmark for SsmPoison {
    fn descriptor(&self) -> &'static BenchmarkDescriptor {
        &DESCRIPTOR
    }

    fn parameters(&self) -> Vec<ParamSpec> {
        vec![
            ParamSpec::new(
                "rounds",
                "Replay rounds",
                "How many times the script is replayed after the reference round. The incident \
                 poisoned rounds 8-9 of 10; BENCH.toml pins the gate at 12.",
                ParamKind::Int { min: 3, max: 30 },
                ParamValue::Int(DEFAULT_ROUNDS as i64),
            ),
            ParamSpec::new(
                "max_tokens",
                "Max tokens per turn",
                "Output budget per turn. Sized so the COLLAPSE_RATIO_CEIL (2.0) stays \
                 reachable: the script's longest answer is a short paragraph (~a few hundred \
                 tokens), and a runaway replay must be able to reach twice the reference \
                 length before the budget clamps it. The old 256 clamped runaways below the \
                 ceiling on any turn past 128 tokens. A reference turn that hits this budget \
                 fails the reference anchors, so a budget still too small is loud, not silent.",
                ParamKind::Int { min: 32, max: 4096 },
                ParamValue::Int(1024),
            ),
            ParamSpec::new(
                "request_timeout_s",
                "Request timeout",
                "Seconds before a single turn is abandoned; the round scores Unmeasured.",
                ParamKind::Int { min: 10, max: 3600 },
                ParamValue::Int(300),
            ),
        ]
    }

    fn configure(&mut self, values: &ParamValues) -> Result<()> {
        let specs = self.parameters();
        values.validate_against(&specs)?;
        self.rounds = values.usize("rounds")?;
        self.max_tokens = values.usize("max_tokens")?;
        self.timeout = Duration::from_secs(values.usize("request_timeout_s")? as u64);
        self.probed = false;
        self.phase = Phase::Baseline;
        self.reference.clear();
        self.replays.clear();
        Ok(())
    }

    async fn next(&mut self) -> Result<BenchmarkResult> {
        let handle = self.handle()?.clone();
        handle.check_cancelled()?;

        if !self.probed {
            self.probed = true;
            http::probe(handle.target(), Duration::from_secs(10))
                .await
                .context("endpoint probe failed — check the target URL and port")?;
            if self.rounds < 3 {
                bail!("need at least 3 replay rounds");
            }
            return Ok(self.frame(
                "probe",
                Some(LogLine::info(format!(
                    "{} · model {} · {} replay rounds, prefix caching under test",
                    handle.target().base_url,
                    handle.target().model,
                    self.rounds
                ))),
            ));
        }

        match self.phase {
            Phase::Baseline => {
                self.reference = self.replay_script("reference").await?;
                // 2026-09-26: The replays are held only to round 0, so round 0
                // is held to the script: output that is wrong the same way in
                // every round would otherwise score Invariant.
                let violations = probe::validate_reference(&self.reference);
                if !violations.is_empty() {
                    self.phase = Phase::Done;
                    let reason = format!(
                        "reference round failed the script anchors ({}) — replays compared \
                         against it would certify nothing",
                        violations.join("; ")
                    );
                    let (s, _) = self.scored();
                    let mut metrics = super::report::metrics(&s);
                    metrics.insert("sum_wall_s".to_string(), self.elapsed().as_secs_f64());
                    return Ok(BenchmarkResult {
                        status: RunStatus::Completed,
                        ..BenchmarkResult::running("reference", self.elapsed())
                    }
                    .with_metrics(metrics)
                    .with_verdict(Verdict::fail(reason.clone()))
                    .log_line(LogLine::error(one_line(reason))));
                }
                self.phase = Phase::Replay;
                Ok(self.frame(
                    "reference",
                    Some(LogLine::info(format!(
                        "reference round captured: {} turns, script anchors hold",
                        self.reference.len()
                    ))),
                ))
            }
            Phase::Replay => {
                let n = self.replays.len() + 1;
                // 2026-09-26: Turn 1's cached-token count is recorded;
                // `score::verdict` fails a replay whose turn 1 reports 0.
                let (v, turn1_cached) = match self.replay_script(&format!("replay {n}")).await {
                    Ok(replay) => {
                        let cached = replay.first().map(|t| t.cached_prompt_tokens);
                        (compare::compare_round(&self.reference, &replay), cached)
                    }
                    Err(e) => (
                        RoundVerdict::Unmeasured {
                            reason: one_line(format!("{e:#}")),
                        },
                        None,
                    ),
                };
                let line = match &v {
                    RoundVerdict::Collapsed { turns } => Some(LogLine::error(format!(
                        "replay {n} COLLAPSED — restored state produced degenerate output on \
                         turns {:?} (the SSM state poisoning signature)",
                        turns.iter().map(|t| t.turn).collect::<Vec<_>>()
                    ))),
                    RoundVerdict::Jittered { turns } => Some(LogLine::info(format!(
                        "replay {n} jittered (healthy) on turns {:?} — restore anchor \
                         selection varies between rounds",
                        turns.iter().map(|t| t.turn).collect::<Vec<_>>()
                    ))),
                    _ => None,
                };
                // 2026-09-26: A collapse or jitter line keeps priority; the
                // vacuity line is logged only when there is no other line.
                let line = if line.is_none() && turn1_cached == Some(0) {
                    Some(LogLine::error(format!(
                        "replay {n} restored 0 cached prompt tokens on turn 1 — the round \
                         never exercised the prefix restore path"
                    )))
                } else {
                    line
                };
                self.replays.push(RoundRecord {
                    round: n,
                    verdict: v,
                    turn1_cached,
                });
                if self.replays.len() >= self.rounds {
                    self.phase = Phase::ToolPath;
                }
                Ok(self.frame(&format!("replay {n}"), line))
            }
            Phase::ToolPath => {
                self.phase = Phase::Score;
                let results = self.run_tool_paths().await?;
                self.tool_divergences = toolcall::divergences(&results);
                self.tool_reference_called = toolcall::reference_called(&results);
                let line = if let Some(d) = self.tool_divergences.first() {
                    LogLine::error(one_line(format!(
                        "TOOL CALL DEPENDS ON HISTORY — {} ({} of {} perturbed paths)",
                        d.describe(),
                        self.tool_divergences.len(),
                        Path::ALL.len() - 1
                    )))
                } else if self.tool_reference_called {
                    LogLine::error(
                        "the direct path itself called a tool on the irrelevance target — the                          reference every other path is compared against is already wrong"
                            .to_string(),
                    )
                } else {
                    LogLine::info(format!(
                        "tool calls identical across all {} predecessor paths",
                        Path::ALL.len()
                    ))
                };
                Ok(self.frame("tool paths", Some(line)))
            }
            Phase::Score => {
                let (s, v) = self.scored();
                // 2026-09-26: Replay and tool-path findings are independent;
                // either alone fails the run.
                let v = if let Some(d) = self.tool_divergences.first() {
                    Verdict::fail(format!(
                        concat!(
                            "TOOL CALLS DEPEND ON HISTORY: {} — the same request ",
                            "answered differently depending only on what preceded ",
                            "it, so a known-answer score is not reproducible under ",
                            "reordering"
                        ),
                        d.describe()
                    ))
                } else {
                    v
                };
                self.phase = Phase::Done;
                let line = LogLine::info(one_line(format!(
                    "{} replays: {} invariant · {} jittered · {} collapsed · {} unmeasured",
                    s.rounds, s.invariant, s.jittered, s.collapsed, s.unmeasured
                )));
                // 2026-09-26: `sum_wall_s` is wall-clock state, so it is added
                // here rather than in the pure `report::metrics`.
                let mut metrics = super::report::metrics(&s);
                metrics.insert("sum_wall_s".to_string(), self.elapsed().as_secs_f64());
                metrics.insert(
                    "tool_path_divergences".to_string(),
                    self.tool_divergences.len() as f64,
                );
                metrics.insert("tool_paths".to_string(), toolcall::Path::ALL.len() as f64);
                Ok(BenchmarkResult {
                    status: RunStatus::Completed,
                    ..BenchmarkResult::running("done", self.elapsed())
                }
                .with_progress(self.total_steps(), self.total_steps())
                .with_summary(super::report::summary(&s))
                .with_table(super::report::table(&s))
                .with_metrics(metrics)
                .with_verdict(v)
                .log_line(line))
            }
            Phase::Done => bail!("next() was called after the run finished"),
        }
    }
}

#[cfg(test)]
#[path = "driver_tests.rs"]
mod driver_tests;
