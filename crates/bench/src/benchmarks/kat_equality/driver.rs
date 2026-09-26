// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-26: The part that talks to a server: issue the draw once per order,
//! keep every reply, and hand the lot to `compare`.
//!
//! Owner: bench, kat_equality.
//! Invariants: every order issues the same samples, a permutation of the
//! loaded draw.

use std::future::Future;
use std::time::{Duration, Instant};

use anyhow::{Context, Result};
use serde_json::json;

use crate::benchmark::{Benchmark, BenchmarkDescriptor};
use crate::benchmarks::bfcl::{MAX_NEW_TOKENS, dataset, draw::DrawSpec, provision};
use crate::benchmarks::transcript::{RequestOutcome, Transcript};
use crate::hardware::Sensitivity;
use crate::http;
use crate::metadata::PluginMetadata;
use crate::params::{ParamKind, ParamSpec, ParamValue, ParamValues};
use crate::plugin::{Plugin, PluginHandle};
use crate::result::{BenchmarkResult, LogLine, RunStatus};

use super::compare::{Observation, OrderRun, permutation, score, verdict};

const SUMMARY: &str = "The same sample must answer the same, whatever ran before it";

/// 2026-09-26: Two orders is the minimum that proves anything, and the second
/// is the reverse, which changes what preceded every sample.
const DEFAULT_ORDERS: usize = 2;

/// 2026-09-26: `0` means the whole draw. A cap exists because the whole golden
/// draw (995 samples) in two orders is 1990 generations; the default is not
/// smaller because a capped run would report a pass with a fraction of the
/// samples.
const DEFAULT_SAMPLE_CAP: usize = 0;

pub const METADATA: PluginMetadata = PluginMetadata::metrale(SUMMARY);

pub const DESCRIPTOR: BenchmarkDescriptor = BenchmarkDescriptor {
    id: "kat-equality-gate",
    name: "KAT Equality Gate",
    summary: SUMMARY,
    detail: "Issues one BFCL draw against ONE server in two or more request ORDERS \
             (canonical, then reversed, then rotations) at temperature 0, and requires \
             every sample_id's reply to be byte-identical across all of them — text, \
             reasoning, tool calls with RAW arguments, finish reason, and completion \
             token count. Any difference FAILS: at temperature 0 a reply must be a \
             function of its own request, and a benchmark whose samples can see each \
             other cannot be sharded, nor can a score taken under one order describe \
             another. A failed request is UNMEASURED, never agreement, and a run whose \
             replies were all empty fails as VACUOUS rather than certifying a dead \
             endpoint as order-independent. Serve it with --hermetic to measure the \
             regime a known-answer test actually requires.",
    duration_hint: "~2x one BFCL leg",
    expected_secs: 4200,
    updated: "2026-09-09",
    needs_confirmation: false,
    // 2026-09-26: The invariant is a property of the engine, so it is not
    // pinned to a checkpoint (`ssm_poison` sets `intended_for: None` too).
    intended_for: None,
    threshold_params: &[],
    // 2026-09-26: A busy box can make a run slower; it cannot make one
    // request's output depend on another's.
    sensitivity: Sensitivity::Correctness,
    ctor: || Box::new(KatEquality::default()),
};

enum Phase {
    Provision,
    Generate,
    Compare,
    Done,
}

pub struct KatEquality {
    handle: Option<PluginHandle>,
    phase: Phase,
    artifacts: Option<provision::Artifacts>,
    /// 2026-09-26: The draw in canonical order. Every order is an index
    /// permutation of this, so no order can measure a different sample set.
    samples: Vec<dataset::Sample>,
    order_index: usize,
    cursor: usize,
    runs: Vec<OrderRun>,
    current: Vec<Observation>,
    orders: usize,
    sample_cap: usize,
    max_new_tokens: usize,
    request_timeout: Duration,
    started: Option<Instant>,
}

impl Default for KatEquality {
    fn default() -> Self {
        Self {
            handle: None,
            phase: Phase::Provision,
            artifacts: None,
            samples: Vec::new(),
            order_index: 0,
            cursor: 0,
            runs: Vec::new(),
            current: Vec::new(),
            orders: DEFAULT_ORDERS,
            sample_cap: DEFAULT_SAMPLE_CAP,
            max_new_tokens: MAX_NEW_TOKENS,
            request_timeout: Duration::from_secs(300),
            started: None,
        }
    }
}

impl KatEquality {
    fn handle(&self) -> Result<&PluginHandle> {
        self.handle.as_ref().context("plugin was not loaded")
    }

    fn elapsed(&self) -> Duration {
        self.started.map(|s| s.elapsed()).unwrap_or_default()
    }

    fn order_label(&self, index: usize) -> String {
        match index {
            0 => "canonical".to_string(),
            1 => "reversed".to_string(),
            k => format!("rotation-{k}"),
        }
    }

    fn total_steps(&self) -> u64 {
        (self.samples.len() * self.orders) as u64
    }

    fn done_steps(&self) -> u64 {
        (self.order_index * self.samples.len() + self.cursor) as u64
    }

    /// 2026-09-26: Issue one sample and keep the whole reply.
    ///
    /// The body is BFCL's (`bfcl/exec.rs`: no `seed`) at temperature 0, plus
    /// `stream_options`: this gate asks whether BFCL's own conditions are
    /// order-independent. `include_usage` asks for the server's
    /// `completion_tokens`; without `usage`, `http::chat_stream` counts
    /// stream deltas instead.
    async fn issue_one(&mut self, sample_index: usize) -> Result<()> {
        let handle = self.handle()?.clone();
        let sample = self.samples[sample_index].clone();
        let target = handle.target();
        let body = json!({
            "model": target.model,
            "stream": true,
            "stream_options": {"include_usage": true},
            "temperature": 0.0,
            "max_tokens": self.max_new_tokens,
            "messages": sample.messages,
            "tools": sample.tools,
            "tool_choice": sample.tool_choice,
        });
        let outcome = match http::chat_stream(target, &body, self.request_timeout).await {
            // 2026-09-26: A transport failure is its own outcome. BFCL scores
            // one as "no call", which here would make two failures look like
            // agreement.
            Err(e) => RequestOutcome::Error(format!("{e:#}")),
            Ok(o) => RequestOutcome::Ok(Box::new(Transcript::from(&o))),
        };
        self.current.push(Observation {
            sample_id: sample.sample_id.clone(),
            outcome,
        });
        Ok(())
    }
}

impl Plugin for KatEquality {
    fn metadata(&self) -> &'static PluginMetadata {
        &METADATA
    }

    fn load(&mut self, handle: PluginHandle) -> impl Future<Output = Result<()>> + Send {
        self.started = Some(Instant::now());
        self.handle = Some(handle.clone());
        async move {
            let artifacts = provision::ensure(handle.artifacts(), &handle).await?;
            self.artifacts = Some(artifacts);
            Ok(())
        }
    }
}

impl Benchmark for KatEquality {
    fn descriptor(&self) -> &'static BenchmarkDescriptor {
        &DESCRIPTOR
    }

    fn parameters(&self) -> Vec<ParamSpec> {
        vec![
            ParamSpec::new(
                "orders",
                "Request orders",
                "How many different orders to issue the draw in. 2 is canonical plus \
                 reversed, which is the strongest pair; further orders are rotations.",
                ParamKind::Int { min: 2, max: 6 },
                ParamValue::Int(DEFAULT_ORDERS as i64),
            ),
            ParamSpec::new(
                "sample_cap",
                "Sample cap",
                "Truncate the draw to this many samples (0 = the whole draw). This is \
                 a PREFIX, not a sample: the draw concatenates subsets in sorted name \
                 order, so a cap decides which subsets are compared at all. Pick it from \
                 where the effect lives, and re-run the negative control AT the cap — one \
                 taken on the whole draw does not describe a capped run, because \
                 truncating changes what ran before every surviving sample.",
                ParamKind::Int { min: 0, max: 5000 },
                ParamValue::Int(DEFAULT_SAMPLE_CAP as i64),
            ),
            ParamSpec::new(
                "max_new_tokens",
                "Max new tokens",
                "Generation cap per sample. Defaults to BFCL's own budget: this \
                 gate asks whether BFCL's conditions are order-independent, so a \
                 different budget would answer about a regime nobody measures.",
                ParamKind::Int { min: 32, max: 4096 },
                ParamValue::Int(MAX_NEW_TOKENS as i64),
            ),
            ParamSpec::new(
                "request_timeout_s",
                "Request timeout (s)",
                "Per-request timeout. A timeout is UNMEASURED, which fails the gate — \
                 it is not evidence that the orders agreed.",
                ParamKind::Int { min: 10, max: 3600 },
                ParamValue::Int(300),
            ),
        ]
    }

    fn configure(&mut self, values: &ParamValues) -> Result<()> {
        let specs = self.parameters();
        values.validate_against(&specs)?;
        self.orders = values.usize("orders")?;
        self.sample_cap = values.usize("sample_cap")?;
        self.max_new_tokens = values.usize("max_new_tokens")?;
        self.request_timeout = Duration::from_secs(values.usize("request_timeout_s")? as u64);
        // 2026-09-26: A re-`configure` must not leave a previous run's replies
        // behind: they would be compared as if this run had produced them.
        self.phase = Phase::Provision;
        self.order_index = 0;
        self.cursor = 0;
        self.runs.clear();
        self.current.clear();
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
                let artifacts = self
                    .artifacts
                    .clone()
                    .context("artifacts were not provisioned")?;
                self.samples = dataset::load_shard(&artifacts.dataset, &DrawSpec::golden(), None)?;
                if self.sample_cap > 0 {
                    self.samples.truncate(self.sample_cap);
                }
                self.phase = Phase::Generate;
                let n = self.samples.len();
                let mut frame = BenchmarkResult::running("draw", self.elapsed())
                    .with_progress(0, self.total_steps())
                    .log_line(LogLine::info(format!(
                        "{n} samples x {} orders = {} generations against one server",
                        self.orders,
                        n * self.orders
                    )));
                if self.sample_cap > 0 {
                    frame = frame.log_line(LogLine::warn(format!(
                        "capped at {n} of the golden 995 — the cap TRUNCATES, and the draw \
                         concatenates in sorted subset order, so this selects which subsets \
                         are compared rather than a random subsample of them; say the cap \
                         when you report a green"
                    )));
                }
                Ok(frame)
            }
            Phase::Generate => {
                let n = self.samples.len();
                if self.cursor >= n {
                    // 2026-09-26: One order finished; bank it and start the
                    // next.
                    let label = self.order_label(self.order_index);
                    self.runs.push(OrderRun {
                        label: label.clone(),
                        observations: std::mem::take(&mut self.current),
                    });
                    self.order_index += 1;
                    self.cursor = 0;
                    if self.order_index >= self.orders {
                        self.phase = Phase::Compare;
                    }
                    return Ok(BenchmarkResult::running("order done", self.elapsed())
                        .with_progress(self.done_steps(), self.total_steps())
                        .log_line(LogLine::info(format!("order `{label}` complete"))));
                }
                let index = permutation(self.order_index, n)[self.cursor];
                self.issue_one(index).await?;
                self.cursor += 1;
                let label = self.order_label(self.order_index);
                handle.progress(self.done_steps(), self.total_steps());
                handle.status(format!("{label} · {}/{n}", self.cursor));
                Ok(BenchmarkResult::running(label, self.elapsed())
                    .with_progress(self.done_steps(), self.total_steps()))
            }
            Phase::Compare => {
                let s = score(&self.runs);
                let v = verdict(&s);
                self.phase = Phase::Done;
                let total = self.total_steps();
                Ok(BenchmarkResult {
                    status: RunStatus::Completed,
                    ..BenchmarkResult::running("done", self.elapsed())
                }
                .with_progress(total, total)
                .with_summary(super::report::summary(&s))
                .with_table(super::report::table(&s, &self.runs))
                .with_metrics(super::report::metrics(&s))
                .with_verdict(v))
            }
            Phase::Done => Ok(BenchmarkResult {
                status: RunStatus::Completed,
                ..BenchmarkResult::running("done", self.elapsed())
            }),
        }
    }
}
