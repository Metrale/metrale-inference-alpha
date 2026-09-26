// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-26: MLPerf Agentic Inference: teacher-forced replay of recorded
//! trajectories with inline scoring. The prompt for each client turn is the
//! recorded history plus the current client message ([`dataset`]), so the
//! model's output is scored but never fed forward. Scoring ports the upstream
//! inline scorer's two rules, workflow intent-code match and coding
//! bash-executable multiset IoU ([`scoring`]).
//!
//! The dataset is not in the repository and nothing downloads it:
//! `provision::ensure` fails when `dataset.jsonl` is absent, naming the path,
//! and when it does not match a set `expected_sha256`. The BENCH.toml entry
//! (`gate = "mlperf-agentic-subset"`) is `status = "unmeasured"` with no
//! metric bounds, `gate::coverage::NOT_REQUIRED` lists the gate with its
//! reason, and the verdict is always info (`report.rs`).
//!
//! Sampling is not a parameter: every request carries the fixed set below plus
//! a seed. A temperature-0 run would give numbers that look comparable to the
//! MLPerf thresholds and are not.
//!
//! Owner: bench, mlperf_agentic.
//! Invariants: a completed run's verdict is info, and a run with no scorable
//! turn fails instead of reporting 0.0.

pub mod dataset;
pub mod provision;
pub mod scoring;

mod descriptors;
mod report;

pub use descriptors::{SUBSET_DESCRIPTOR, SUBSET_METADATA};

use std::future::Future;
use std::time::{Duration, Instant};

use anyhow::{Context, Result, bail};
use serde_json::{Value, json};

use crate::benchmark::{Benchmark, BenchmarkDescriptor};
use crate::benchmarks::one_line;
use crate::http;
use crate::metadata::PluginMetadata;
use crate::params::{ParamKind, ParamSpec, ParamValue, ParamValues};
use crate::plugin::{Plugin, PluginHandle};
use crate::result::{BenchmarkResult, LogLine, RunStatus};

use dataset::{Conversation, DrawSpec};
use scoring::Domain;

/// 2026-09-26: The sampling set every request carries (`request_body`). The
/// BENCH.toml entry's note records it as the MLPerf set, which submitters may
/// not change.
const TEMPERATURE: f64 = 1.0;
const TOP_K: i64 = 20;
const TOP_P: f64 = 0.95;
const REPETITION_PENALTY: f64 = 1.0;
const PRESENCE_PENALTY: f64 = 1.5;
const MAX_NEW_TOKENS: i64 = 8192;

/// 2026-09-26: The `expected_sha256` value that means "no pin". A named
/// sentinel rather than an empty string, because `ParamKind::Text` refuses
/// empty values.
const UNPINNED: &str = "unpinned";

/// 2026-09-26: One replayed client turn, kept for aggregation and the
/// responses artifact.
pub(crate) struct TurnRecord {
    conversation_id: String,
    turn: i64,
    domain: Domain,
    /// 2026-09-26: `None` when the turn has no scorable ground truth (excluded
    /// from the denominator). A failed request is scored against an empty
    /// assistant turn, which gives `Some(0.0)` for a scorable turn (in the
    /// denominator).
    score: Option<f64>,
    missing: bool,
    completion_tokens: usize,
    /// 2026-09-26: The model turn as the scorer saw it, for rescoring.
    model: Value,
}

enum Phase {
    Provision,
    Replay,
    Score,
    Done,
}

pub struct MlperfAgentic {
    handle: Option<PluginHandle>,
    phase: Phase,
    artifacts: Option<provision::Artifacts>,
    conversations: Vec<Conversation>,
    /// 2026-09-26: Flattened (conversation, client-turn) replay order, one
    /// request at a time.
    schedule: Vec<(usize, usize)>,
    cursor: usize,
    turns: Vec<TurnRecord>,
    scores: Option<report::Scores>,
    /// 2026-09-26: `file-sha256:…;draw-sha256:…`, recorded on the terminal
    /// frame and carried into the gate record.
    fingerprint: Option<String>,
    spec: DrawSpec,
    seed: i64,
    expected_sha256: String,
    request_timeout: Duration,
    started: Option<Instant>,
    replay_started: Option<Instant>,
    replay_wall: Option<Duration>,
}

impl Default for MlperfAgentic {
    fn default() -> Self {
        Self::new()
    }
}

impl MlperfAgentic {
    pub fn new() -> Self {
        Self {
            handle: None,
            phase: Phase::Provision,
            artifacts: None,
            conversations: Vec::new(),
            schedule: Vec::new(),
            cursor: 0,
            turns: Vec::new(),
            scores: None,
            fingerprint: None,
            spec: DrawSpec::all(),
            seed: 42,
            expected_sha256: String::new(),
            request_timeout: Duration::from_secs(600),
            started: None,
            replay_started: None,
            replay_wall: None,
        }
    }

    fn handle(&self) -> Result<&PluginHandle> {
        self.handle.as_ref().context("benchmark was not loaded")
    }

    fn elapsed(&self) -> Duration {
        self.started.map(|s| s.elapsed()).unwrap_or_default()
    }

    async fn replay_one(&mut self) -> Result<()> {
        let handle = self.handle()?.clone();
        let (conv_idx, turn_idx) = self.schedule[self.cursor];
        let conv = &self.conversations[conv_idx];
        let client_turn = &conv.client_turns[turn_idx];
        let body = request_body(
            &handle.target().model,
            &client_turn.messages,
            conv.tools.as_ref(),
            self.seed,
        );
        let outcome = http::chat_stream(handle.target(), &body, self.request_timeout).await;
        let (model, missing, completion_tokens) = match outcome {
            Ok(o) => {
                let tokens = o.completion_tokens;
                (model_turn(&o), false, tokens)
            }
            Err(e) => {
                // 2026-09-26: A failed request stays in the denominator at 0
                // and is logged, so a run degraded by transport errors is
                // visible.
                handle.warn(one_line(format!(
                    "{} turn {}: {e:#}",
                    conv.id, client_turn.turn
                )));
                (json!({"role": "assistant"}), true, 0)
            }
        };
        let (score, domain) = match &client_turn.ground_truth {
            Some(gt) if scoring::has_ground_truth(conv.domain, gt) => (
                Some(scoring::score_turn(conv.domain, gt, &model)),
                conv.domain,
            ),
            _ => (None, conv.domain),
        };
        self.turns.push(TurnRecord {
            conversation_id: conv.id.clone(),
            turn: client_turn.turn,
            domain,
            score,
            missing,
            completion_tokens,
            model,
        });
        self.cursor += 1;
        Ok(())
    }

    /// 2026-09-26: Write the per-turn artifact beside the dataset so a scorer
    /// change can re-score a finished run without re-generating it.
    ///
    /// The filename is fixed, which is safe only while this benchmark has no
    /// group members; the test
    /// `sharding_this_benchmark_would_need_a_per_leg_responses_file` checks
    /// that. `bfcl::exec::responses_file` keys its file by benchmark id.
    fn write_responses(&self) -> Result<()> {
        let artifacts = self.artifacts.as_ref().context("not provisioned")?;
        let path = artifacts.dir.join("responses.jsonl");
        let mut text = String::new();
        for t in &self.turns {
            text.push_str(&serde_json::to_string(&json!({
                "conversation_id": t.conversation_id,
                "turn": t.turn,
                "domain": match t.domain { Domain::Coding => "coding", Domain::Workflow => "workflow" },
                "score": t.score,
                "missing": t.missing,
                "completion_tokens": t.completion_tokens,
                "model": t.model,
            }))?);
            text.push('\n');
        }
        std::fs::write(&path, text).with_context(|| format!("writing {}", path.display()))
    }
}

/// 2026-09-26: The request for one client turn: the fixed sampling set, the
/// prebuilt teacher-forced messages, the conversation's tools if any, and the
/// seed. Nothing else; a test pins the exact key set.
fn request_body(model: &str, messages: &[Value], tools: Option<&Value>, seed: i64) -> Value {
    let mut body = json!({
        "model": model,
        "stream": true,
        "messages": messages,
        "temperature": TEMPERATURE,
        "top_k": TOP_K,
        "top_p": TOP_P,
        "repetition_penalty": REPETITION_PENALTY,
        "presence_penalty": PRESENCE_PENALTY,
        "max_tokens": MAX_NEW_TOKENS,
        "seed": seed,
        "chat_template_kwargs": {"preserve_thinking": true},
    });
    if let Some(tools) = tools {
        body["tools"] = tools.clone();
    }
    body
}

/// 2026-09-26: The assistant turn as the scorer reads it: `tool_calls` is
/// `null` (not `[]`) when the model made none, and arguments stay the raw
/// streamed string.
fn model_turn(outcome: &http::ChatOutcome) -> Value {
    let tool_calls: Vec<Value> = outcome
        .tool_calls
        .iter()
        .map(|c| json!({"function": {"name": c.name, "arguments": c.arguments}}))
        .collect();
    json!({
        "role": "assistant",
        "content": outcome.text,
        "reasoning_content": outcome.reasoning,
        "tool_calls": if tool_calls.is_empty() { Value::Null } else { Value::Array(tool_calls) },
    })
}

impl Plugin for MlperfAgentic {
    fn metadata(&self) -> &'static PluginMetadata {
        &SUBSET_METADATA
    }

    fn load(&mut self, handle: PluginHandle) -> impl Future<Output = Result<()>> + Send {
        self.started = Some(Instant::now());
        self.handle = Some(handle.clone());
        async move {
            // 2026-09-26: An absent or mismatched dataset stops the run here
            // (`provision::ensure`).
            let artifacts = provision::ensure(handle.artifacts(), &handle, &self.expected_sha256)?;
            self.artifacts = Some(artifacts);
            Ok(())
        }
    }
}

impl Benchmark for MlperfAgentic {
    fn descriptor(&self) -> &'static BenchmarkDescriptor {
        &SUBSET_DESCRIPTOR
    }

    fn parameters(&self) -> Vec<ParamSpec> {
        vec![
            ParamSpec::new(
                "coding_trajectories",
                "Coding trajectories",
                "First-K coding conversations in file order; 0 takes all. The pinned draw is \
                 a calibration output — until one exists, there is nothing to default to.",
                ParamKind::Int {
                    min: 0,
                    max: 100_000,
                },
                ParamValue::Int(0),
            ),
            ParamSpec::new(
                "workflow_trajectories",
                "Workflow trajectories",
                "First-K workflow (sim_*) conversations in file order; 0 takes all.",
                ParamKind::Int {
                    min: 0,
                    max: 100_000,
                },
                ParamValue::Int(0),
            ),
            ParamSpec::new(
                "seed",
                "Seed",
                "Pinned sampling seed. MLPerf specifies none (temp-1.0 is nondeterministic \
                 by design); a seed narrows internal run-to-run variance without touching \
                 the immutable sampling set.",
                ParamKind::Int {
                    min: 0,
                    max: i64::MAX,
                },
                ParamValue::Int(42),
            ),
            ParamSpec::new(
                "expected_sha256",
                "Dataset SHA256 pin",
                "Refuse to run unless the dataset hashes to this. The literal \"unpinned\" \
                 means record-only — the only honest default while the official file is \
                 unpublished; pin it the day the file ships.",
                ParamKind::Text,
                ParamValue::Text(String::from(UNPINNED)),
            ),
            ParamSpec::new(
                "request_timeout_s",
                "Request timeout",
                "Seconds before one turn is abandoned and scored 0. The official client \
                 allows 14400 per turn; a merge gate wants a bound that fails fast.",
                ParamKind::Int {
                    min: 10,
                    max: 14_400,
                },
                ParamValue::Int(600),
            ),
        ]
    }

    fn configure(&mut self, values: &ParamValues) -> Result<()> {
        let specs = self.parameters();
        values.validate_against(&specs)?;
        self.spec = DrawSpec {
            coding: values.usize("coding_trajectories")?,
            workflow: values.usize("workflow_trajectories")?,
        };
        self.seed = values.int("seed")?;
        let pin = values.text("expected_sha256")?.trim();
        self.expected_sha256 = if pin.eq_ignore_ascii_case(UNPINNED) {
            String::new()
        } else {
            pin.to_string()
        };
        self.request_timeout = Duration::from_secs(values.usize("request_timeout_s")? as u64);
        self.phase = Phase::Provision;
        self.cursor = 0;
        self.conversations.clear();
        self.schedule.clear();
        self.turns.clear();
        self.scores = None;
        self.fingerprint = None;
        self.replay_started = None;
        self.replay_wall = None;
        Ok(())
    }

    async fn next(&mut self) -> Result<BenchmarkResult> {
        let handle = self.handle()?.clone();
        handle.check_cancelled()?;
        match self.phase {
            Phase::Provision => {
                http::probe(handle.target(), Duration::from_secs(10))
                    .await
                    .context("endpoint probe failed — check the target URL and port")?;
                let artifacts = self.artifacts.clone().context("not provisioned")?;
                self.conversations = dataset::load(&artifacts.dataset, &self.spec)?;
                self.schedule = self
                    .conversations
                    .iter()
                    .enumerate()
                    .flat_map(|(c, conv)| (0..conv.client_turns.len()).map(move |t| (c, t)))
                    .collect();
                let draw = dataset::draw_fingerprint(&self.conversations);
                self.fingerprint = Some(format!(
                    "file-sha256:{};draw-sha256:{draw}",
                    artifacts.file_sha256
                ));
                self.phase = Phase::Replay;
                self.replay_started = Some(Instant::now());
                Ok(BenchmarkResult::running("draw", self.elapsed())
                    .with_progress(0, self.schedule.len() as u64)
                    .log_line(LogLine::info(format!(
                        "drew {} trajectories ({} client turns) · draw fingerprint {}",
                        self.conversations.len(),
                        self.schedule.len(),
                        &draw[..12],
                    ))))
            }
            Phase::Replay => {
                let total = self.schedule.len() as u64;
                if self.cursor >= self.schedule.len() {
                    self.replay_wall = self.replay_started.map(|s| s.elapsed());
                    self.phase = Phase::Score;
                    handle.status("scoring");
                    return Ok(BenchmarkResult::running("scoring", self.elapsed())
                        .with_progress(total, total)
                        .with_summary(self.summary()));
                }
                let (conv_idx, _) = self.schedule[self.cursor];
                let conv_id = self.conversations[conv_idx].id.clone();
                self.replay_one().await?;
                let done = self.cursor as u64;
                handle.progress(done, total);
                handle.status(format!("{conv_id} · {done}/{total}"));
                Ok(BenchmarkResult::running(conv_id, self.elapsed())
                    .with_progress(done, total)
                    .with_summary(self.summary()))
            }
            Phase::Score => {
                let Some(scores) = report::aggregate(&self.turns) else {
                    // 2026-09-26: Refusing, not reporting 0.0: a denominator
                    // of zero means the draw held no scorable ground truth.
                    bail!(
                        "no turn in the draw carried scorable ground truth — refusing to \
                         report a score from an empty denominator"
                    );
                };
                self.write_responses()?;
                self.scores = Some(scores);
                self.phase = Phase::Done;
                let total = self.schedule.len() as u64;
                let mut frame = BenchmarkResult {
                    status: RunStatus::Completed,
                    ..BenchmarkResult::running("done", self.elapsed())
                }
                .with_progress(total, total)
                .with_summary(self.summary())
                .with_metrics(self.metrics())
                .with_verdict(self.verdict());
                frame.dataset_fingerprint = self.fingerprint.clone();
                if let Some(t) = self.table() {
                    frame = frame.with_table(t);
                }
                Ok(frame)
            }
            Phase::Done => bail!("next() was called after the run finished"),
        }
    }
}

#[cfg(test)]
#[path = "mlperf_tests.rs"]
mod tests;
