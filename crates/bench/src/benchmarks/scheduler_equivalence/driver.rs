// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-26: The scheduler-equivalence driver: serve each router under each
//! lane, issue the draw at each concurrency, and hand the legs to `compare`.
//!
//! Owner: bench, scheduler-equivalence gate.
//! Invariants:
//! - `plan` serves each (router, lane) variant once, and a lane's sync legs
//!   (reference, then control) run before its async serve.
//! - A failed request becomes its reply's outcome (`request`); it does not
//!   end the pass.

use std::collections::BTreeMap;
use std::future::Future;
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::{Duration, Instant};

use anyhow::{Context, Result};
use futures::StreamExt;
use serde_json::json;

use crate::benchmark::{Benchmark, BenchmarkDescriptor};
use crate::benchmarks::bfcl::{MAX_NEW_TOKENS, dataset, draw::DrawSpec, provision};
use crate::benchmarks::transcript::Transcript;
use crate::hardware::Sensitivity;
use crate::http;
use crate::metadata::PluginMetadata;
use crate::params::{ParamKind, ParamSpec, ParamValue, ParamValues};
use crate::plugin::{Plugin, PluginHandle, TargetEndpoint};
use crate::result::{BenchmarkResult, LogLine, RunStatus};

use super::compare::{Leg, Pass, Reply};
use super::host::{Lane, Router, RouterHost, ServeVariant};

const SUMMARY: &str = "The async router must answer every sample exactly as the sync router does";

/// 2026-09-26: A prefix of the golden draw: `next` truncates the loaded samples
/// to it and logs a warning that names the cap.
const DEFAULT_SAMPLE_CAP: usize = 64;
const DEFAULT_CONCURRENCIES: [i64; 3] = [1, 4, 16];

pub const METADATA: PluginMetadata = PluginMetadata::metrale(SUMMARY);

pub const DESCRIPTOR: BenchmarkDescriptor = BenchmarkDescriptor {
    id: "scheduler-equivalence",
    name: "Scheduler Equivalence Gate",
    summary: SUMMARY,
    detail: "Serves the checkpoint under --scheduler-config sync and async, once per \
             speculation lane (speculation off, then --mtp-gate force), issues one BFCL draw \
             against each at concurrency 1, 4 and 16, and requires every sample_id's reply to \
             be byte-identical between the routers — text, reasoning, tool calls with raw \
             arguments, finish reason and completion token count — per concurrency. The sync \
             serve is measured twice; a control that diverges makes an async difference \
             unattributable and the run UNPROVEN rather than green. A failed request is \
             UNMEASURED, never agreement, and an all-empty cell fails as VACUOUS. The async \
             router's KV-reservation drift is recorded as a diagnostic outside equality.",
    duration_hint: "~9 draw passes + 4 model loads",
    expected_secs: 5400,
    updated: "2026-09-25",
    needs_confirmation: false,
    // 2026-09-26: Not pinned to a checkpoint: the host re-serves whichever
    // checkpoint is being served.
    intended_for: None,
    threshold_params: &[],
    // 2026-09-26: The verdict reads byte equality and token counts, never
    // timing.
    sensitivity: Sensitivity::Correctness,
    ctor: || Box::new(SchedulerEquivalence::default()),
};

/// 2026-09-26: One unit of the run; `plan` orders them.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Step {
    Serve(ServeVariant),
    /// 2026-09-26: Read the async counters before / after a lane's async legs.
    Diagnostics {
        lane: Lane,
        before: bool,
    },
    Generate {
        lane: Lane,
        pass: Pass,
        concurrency: usize,
    },
}

/// 2026-09-26: The run, as a sequence: per lane, the sync serve (reference legs,
/// then the control legs on the same serve), then the async serve with its
/// legs between two diagnostics reads. Each variant is served once.
pub fn plan(lanes: &[Lane], concurrencies: &[usize], control: bool) -> Vec<Step> {
    let mut steps = Vec::new();
    for &lane in lanes {
        steps.push(Step::Serve(ServeVariant {
            router: Router::Sync,
            lane,
        }));
        for &c in concurrencies {
            steps.push(Step::Generate {
                lane,
                pass: Pass::Sync,
                concurrency: c,
            });
        }
        if control {
            for &c in concurrencies {
                steps.push(Step::Generate {
                    lane,
                    pass: Pass::Control,
                    concurrency: c,
                });
            }
        }
        steps.push(Step::Serve(ServeVariant {
            router: Router::Async,
            lane,
        }));
        steps.push(Step::Diagnostics { lane, before: true });
        for &c in concurrencies {
            steps.push(Step::Generate {
                lane,
                pass: Pass::Async,
                concurrency: c,
            });
        }
        steps.push(Step::Diagnostics {
            lane,
            before: false,
        });
    }
    steps
}

pub fn lanes_for(choice: &str) -> Result<Vec<Lane>> {
    Ok(match choice {
        "both" => vec![Lane::SpecOff, Lane::MtpForce],
        "spec-off" => vec![Lane::SpecOff],
        "mtp-force" => vec![Lane::MtpForce],
        other => anyhow::bail!("lanes: {other:?} is not one of both, spec-off, mtp-force"),
    })
}

pub struct SchedulerEquivalence {
    handle: Option<PluginHandle>,
    host: Option<Arc<dyn RouterHost>>,
    artifacts: Option<provision::Artifacts>,
    samples: Vec<dataset::Sample>,
    steps: Vec<Step>,
    cursor: usize,
    target: Option<TargetEndpoint>,
    legs: Vec<Leg>,
    diag_before: BTreeMap<String, f64>,
    diagnostics: BTreeMap<String, f64>,
    done: bool,
    sample_cap: usize,
    max_new_tokens: usize,
    concurrencies: Vec<usize>,
    lanes: Vec<Lane>,
    control: bool,
    request_timeout: Duration,
    started: Option<Instant>,
}

impl Default for SchedulerEquivalence {
    fn default() -> Self {
        Self {
            handle: None,
            host: None,
            artifacts: None,
            samples: Vec::new(),
            steps: Vec::new(),
            cursor: 0,
            target: None,
            legs: Vec::new(),
            diag_before: BTreeMap::new(),
            diagnostics: BTreeMap::new(),
            done: false,
            sample_cap: DEFAULT_SAMPLE_CAP,
            max_new_tokens: MAX_NEW_TOKENS,
            concurrencies: DEFAULT_CONCURRENCIES.iter().map(|c| *c as usize).collect(),
            lanes: vec![Lane::SpecOff, Lane::MtpForce],
            control: true,
            request_timeout: Duration::from_secs(300),
            started: None,
        }
    }
}

impl SchedulerEquivalence {
    fn handle(&self) -> Result<&PluginHandle> {
        self.handle.as_ref().context("plugin was not loaded")
    }

    fn host(&self) -> Result<Arc<dyn RouterHost>> {
        self.host.clone().context(super::host::NO_HOST)
    }

    fn elapsed(&self) -> Duration {
        self.started.map(|s| s.elapsed()).unwrap_or_default()
    }

    fn total_steps(&self) -> u64 {
        self.steps.len() as u64
    }

    /// 2026-09-26: The same body the KAT equality gate sends
    /// (`kat_equality/driver.rs`): streamed with usage, temperature 0, and the
    /// sample's messages, tools and tool_choice.
    fn body(&self, sample: &dataset::Sample, model: &str) -> serde_json::Value {
        json!({
            "model": model,
            "stream": true,
            "stream_options": {"include_usage": true},
            "temperature": 0.0,
            "max_tokens": self.max_new_tokens,
            "messages": sample.messages,
            "tools": sample.tools,
            "tool_choice": sample.tool_choice,
        })
    }

    /// 2026-09-26: Issue the whole draw with `concurrency` requests in flight.
    async fn generate(&self, target: &TargetEndpoint, concurrency: usize) -> Result<Vec<Reply>> {
        let handle = self.handle()?.clone();
        let bodies: Vec<(String, serde_json::Value)> = self
            .samples
            .iter()
            .map(|s| (s.sample_id.clone(), self.body(s, &target.model)))
            .collect();
        let total = bodies.len();
        let finished = Arc::new(AtomicUsize::new(0));
        let timeout = self.request_timeout;
        let mut replies: Vec<Reply> = futures::stream::iter(bodies)
            .map(|(sample_id, body)| {
                let handle = handle.clone();
                let finished = Arc::clone(&finished);
                async move {
                    let reply = request(target, sample_id, &body, timeout).await;
                    let n = finished.fetch_add(1, Ordering::Relaxed) + 1;
                    handle.status(format!("C={concurrency} · {n}/{total}"));
                    reply
                }
            })
            .buffer_unordered(concurrency.max(1))
            .collect()
            .await;
        // 2026-09-26: Completion order depends on the concurrency; sorted by
        // id so a leg's order does not.
        replies.sort_by(|a, b| a.sample_id.cmp(&b.sample_id));
        Ok(replies)
    }

    fn compare(&mut self) -> BenchmarkResult {
        self.done = true;
        let total = self.total_steps();
        super::report::terminal(&self.legs, &self.diagnostics, self.elapsed())
            .with_progress(total, total)
    }
}

/// 2026-09-26: One sample against one serve. Never fails: an error is the
/// reply's outcome, classified, with the time it took to arrive.
pub async fn request(
    target: &TargetEndpoint,
    sample_id: String,
    body: &serde_json::Value,
    timeout: Duration,
) -> Reply {
    let started = Instant::now();
    let outcome = http::chat_stream(target, body, timeout)
        .await
        .map(|o| Box::new(Transcript::from(&o)))
        .map_err(|e| http::classify(&e));
    Reply {
        sample_id,
        outcome,
        elapsed: started.elapsed(),
    }
}

impl Plugin for SchedulerEquivalence {
    fn metadata(&self) -> &'static PluginMetadata {
        &METADATA
    }

    fn load(&mut self, handle: PluginHandle) -> impl Future<Output = Result<()>> + Send {
        self.started = Some(Instant::now());
        self.handle = Some(handle.clone());
        // 2026-09-26: The host check precedes `provision::ensure`, so a process
        // with no host fails before any download.
        let host = super::host::installed();
        async move {
            let host = host.context(super::host::NO_HOST)?;
            self.host = Some(host);
            let artifacts = provision::ensure(handle.artifacts(), &handle).await?;
            self.artifacts = Some(artifacts);
            Ok(())
        }
    }
}

impl Benchmark for SchedulerEquivalence {
    fn descriptor(&self) -> &'static BenchmarkDescriptor {
        &DESCRIPTOR
    }

    fn parameters(&self) -> Vec<ParamSpec> {
        vec![
            ParamSpec::new(
                "sample_cap",
                "Sample cap",
                "Truncate the draw to this many samples (0 = the whole draw). A PREFIX of the \
                 sorted-subset concatenation, not a random sample; say the cap when you \
                 report a green.",
                ParamKind::Int { min: 0, max: 5000 },
                ParamValue::Int(DEFAULT_SAMPLE_CAP as i64),
            ),
            ParamSpec::new(
                "max_new_tokens",
                "Max new tokens",
                "Generation cap per sample. BFCL's own budget, so the gate answers about the \
                 regime the accuracy gates measure.",
                ParamKind::Int { min: 32, max: 4096 },
                ParamValue::Int(MAX_NEW_TOKENS as i64),
            ),
            ParamSpec::new(
                "concurrencies",
                "Concurrencies",
                "Requests in flight per pass. A difference at C >= 4 that is absent at C = 1 \
                 is a batch-shape difference, not a feed one — the report keeps them apart.",
                ParamKind::IntList { min: 1, max: 128 },
                ParamValue::IntList(DEFAULT_CONCURRENCIES.to_vec()),
            ),
            ParamSpec::new(
                "lanes",
                "Speculation lanes",
                "Which lane pins to run: both (speculation off, then --mtp-gate force), or one.",
                ParamKind::Choice(&["both", "spec-off", "mtp-force"]),
                ParamValue::Text("both".to_string()),
            ),
            ParamSpec::new(
                "control",
                "Sync-vs-sync control",
                "Measure the sync serve a second time per lane. Off only for a quick look: \
                 without it an async difference cannot be attributed.",
                ParamKind::Bool,
                ParamValue::Bool(true),
            ),
            ParamSpec::new(
                "request_timeout_s",
                "Request timeout (s)",
                "Per-request timeout. A timeout is UNMEASURED, which fails the gate.",
                ParamKind::Int { min: 10, max: 3600 },
                ParamValue::Int(300),
            ),
        ]
    }

    fn configure(&mut self, values: &ParamValues) -> Result<()> {
        let specs = self.parameters();
        values.validate_against(&specs)?;
        self.sample_cap = values.usize("sample_cap")?;
        self.max_new_tokens = values.usize("max_new_tokens")?;
        self.concurrencies = values
            .int_list("concurrencies")?
            .iter()
            .map(|c| *c as usize)
            .collect();
        self.lanes = lanes_for(values.text("lanes")?)?;
        self.control = values.bool("control")?;
        self.request_timeout = Duration::from_secs(values.usize("request_timeout_s")? as u64);
        // 2026-09-26: A re-`configure` starts over: no leg of an earlier run is
        // compared as this run's.
        self.steps.clear();
        self.cursor = 0;
        self.legs.clear();
        self.diagnostics.clear();
        self.target = None;
        self.done = false;
        Ok(())
    }

    async fn next(&mut self) -> Result<BenchmarkResult> {
        let handle = self.handle()?.clone();
        handle.check_cancelled()?;
        if self.done {
            return Ok(BenchmarkResult {
                status: RunStatus::Completed,
                ..BenchmarkResult::running("done", self.elapsed())
            });
        }
        if self.steps.is_empty() {
            let artifacts = self
                .artifacts
                .clone()
                .context("artifacts were not provisioned")?;
            self.samples = dataset::load_shard(&artifacts.dataset, &DrawSpec::golden(), None)?;
            if self.sample_cap > 0 {
                self.samples.truncate(self.sample_cap);
            }
            self.steps = plan(&self.lanes, &self.concurrencies, self.control);
            let n = self.samples.len();
            let passes = self
                .steps
                .iter()
                .filter(|s| matches!(s, Step::Generate { .. }))
                .count();
            let mut frame = BenchmarkResult::running("draw", self.elapsed())
                .with_progress(0, self.total_steps())
                .log_line(LogLine::info(format!(
                    "{n} samples x {passes} passes ({} lanes x {} concurrencies x {} routers) \
                     = {} generations, {} model loads",
                    self.lanes.len(),
                    self.concurrencies.len(),
                    if self.control { 3 } else { 2 },
                    n * passes,
                    2 * self.lanes.len()
                )));
            if self.sample_cap > 0 {
                frame = frame.log_line(LogLine::warn(format!(
                    "capped at {n} samples of the golden draw — a prefix, not a sample; \
                     say the cap when you report a green"
                )));
            }
            return Ok(frame);
        }
        let Some(step) = self.steps.get(self.cursor).copied() else {
            return Ok(self.compare());
        };
        let label = match step {
            Step::Serve(v) => {
                handle.status(format!("serving {}", v.label()));
                let target = self.host()?.serve(v).await.with_context(|| {
                    format!("could not serve the checkpoint under {}", v.label())
                })?;
                http::probe(&target, Duration::from_secs(10))
                    .await
                    .context("the re-served endpoint does not answer")?;
                self.target = Some(target);
                format!("serve {}", v.label())
            }
            Step::Diagnostics { lane, before } => {
                let now = self.host()?.diagnostics()?;
                if before {
                    self.diag_before = now;
                } else {
                    for (k, v) in now {
                        let base = self.diag_before.get(&k).copied().unwrap_or(0.0);
                        self.diagnostics
                            .insert(format!("{}_{k}", lane.label()), v - base);
                    }
                }
                format!("diagnostics {}", lane.label())
            }
            Step::Generate {
                lane,
                pass,
                concurrency,
            } => {
                let target = self.target.clone().context("no serve preceded this pass")?;
                let replies = self.generate(&target, concurrency).await?;
                self.legs.push(Leg {
                    lane,
                    pass,
                    concurrency,
                    replies,
                });
                format!("{} {} C={concurrency}", lane.label(), pass.label())
            }
        };
        self.cursor += 1;
        handle.progress(self.cursor as u64, self.total_steps());
        Ok(BenchmarkResult::running(label.clone(), self.elapsed())
            .with_progress(self.cursor as u64, self.total_steps())
            .log_line(LogLine::info(format!("{label} complete"))))
    }

    async fn cleanup(&mut self) -> Result<()> {
        match self.host.as_ref() {
            Some(host) => host.restore().await,
            None => Ok(()),
        }
    }
}
