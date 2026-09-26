// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-26: The `agentic-webserver` benchmark: N iterations of one task
//! (write a Rust Axum ping/pong project, test it, run it, curl it, tear it
//! down), each in a fresh sandbox. After each iteration the scorer builds and
//! runs what the agent left and asks `/ping` for `pong`, and checks the agent's
//! commands against the prompt's directions.
//!
//! The prompt is verbatim from `bench/fp8_dgx2_drift/harness/run_tier.sh`; the
//! agent loop is our own, not the opencode client that script drives. It runs
//! model-authored shell; see [`agent`] for the containment.
//!
//! Owner: bench, agentic.
//! Invariants: none beyond the types.

pub mod agent;
mod params;
pub mod preflight;
mod render;
pub mod score;
mod verdict;
pub mod warm;

use std::future::Future;
use std::path::PathBuf;
use std::time::{Duration, Instant};

use anyhow::{Context, Result, bail};

use crate::benchmark::{Benchmark, BenchmarkDescriptor};
use crate::benchmarks::one_line;
use crate::http;
use crate::metadata::PluginMetadata;
use crate::params::{ParamSpec, ParamValues};
use crate::plugin::{Plugin, PluginHandle};
use crate::result::{BenchmarkResult, LogLine, RunStatus, Verdict};

/// 2026-09-26: The task, verbatim from `run_tier.sh`'s `PROMPT`. Changing a word
/// changes the benchmark.
pub const PROMPT: &str = "Please create a pure rust Axum project here in the current working \
directory. Just have a ping/pong endpoint. The server MUST bind to the port from the \
METRALE_HARNESS_PORT env var (default 3001) — use `let port: u16 = \
std::env::var(\"METRALE_HARNESS_PORT\").unwrap_or_else(|_| \"3001\".to_string()).parse().unwrap();` \
then bind to `0.0.0.0:port`. Add tests, run them and prove all tests pass, then run the server and \
use curl to prove it works. Whenever you run the server or any long-lived process in the \
background, always start it detached with its output redirected to a file (for example `setsid \
cargo run > /tmp/server.log 2>&1 &`) so your shell never blocks waiting on it, and wrap any \
command that might hang, such as curl checks or process kills, in a short `timeout 15`. Finally, \
tear down the server by killing whatever is listening on its port rather than guessing the process \
name, always wrapped in a short timeout so it can never stall your shell, for example `timeout 5 \
fuser -k ${METRALE_HARNESS_PORT:-3001}/tcp 2>/dev/null || true`.";

mod descriptors;
pub use descriptors::{DESCRIPTOR, METADATA};

#[derive(Default)]
struct IterationRow {
    index: usize,
    /// 2026-09-26: Iteration wall including the scorer's build and probe;
    /// `sum_wall_s` sums it.
    wall_s: f64,
    /// 2026-09-26: The agent's own wall, scorer excluded. `s_per_turn` divides
    /// its sum by turns; the scorer is a per-iteration cost, so it is left out
    /// of a per-turn ratio.
    agent_wall_s: f64,
    webserver_ok: bool,
    directions: score::Directions,
    turns: usize,
    tool_calls: usize,
    completion_tokens: usize,
    /// 2026-09-26: The loop ended at `max_turns`, not because the agent stopped
    /// calling tools.
    hit_turn_cap: bool,
    /// 2026-09-26: See `Transcript::truncated_turns`.
    truncated_turns: usize,
    /// 2026-09-26: See `Transcript::unparsed_call_turns`.
    unparsed_call_turns: usize,
    note: String,
}

#[derive(Default)]
pub struct AgenticWebserver {
    handle: Option<PluginHandle>,
    iterations: usize,
    max_turns: usize,
    command_timeout: Duration,
    request_timeout: Duration,
    build_timeout: Duration,
    serve_timeout: Duration,
    max_tokens: usize,
    wall_budget_s: f64,
    s_per_turn_budget: f64,
    cursor: usize,
    rows: Vec<IterationRow>,
    sandbox_root: Option<PathBuf>,
    cargo_target_dir: Option<PathBuf>,
    started: Option<Instant>,
    probed: bool,
}

impl AgenticWebserver {
    fn handle(&self) -> Result<&PluginHandle> {
        self.handle.as_ref().context("benchmark was not loaded")
    }

    fn elapsed(&self) -> Duration {
        self.started.map(|s| s.elapsed()).unwrap_or_default()
    }

    fn total_wall(&self) -> f64 {
        self.rows.iter().map(|r| r.wall_s).sum()
    }

    /// 2026-09-26: Sum of `agent_wall_s`, the numerator of `s_per_turn`.
    fn total_agent_wall(&self) -> f64 {
        self.rows.iter().map(|r| r.agent_wall_s).sum()
    }

    fn total_turns(&self) -> usize {
        verdict::total_turns(&self.rows)
    }

    fn total_tool_calls(&self) -> usize {
        self.rows.iter().map(|r| r.tool_calls).sum()
    }

    fn total_completion_tokens(&self) -> usize {
        self.rows.iter().map(|r| r.completion_tokens).sum()
    }

    /// 2026-09-26: `None` when the tier took no turns ([`verdict::seconds_per_turn`]).
    fn seconds_per_turn(&self) -> Option<f64> {
        verdict::seconds_per_turn(self.total_agent_wall(), self.total_turns())
    }

    async fn run_iteration(&mut self, index: usize) -> Result<IterationRow> {
        let handle = self.handle()?.clone();
        let root = self
            .sandbox_root
            .clone()
            .context("sandbox root was not prepared")?;
        let sandbox = root.join(format!("run-{index:02}"));
        // 2026-09-26: A fresh directory per iteration, so an agent is never
        // scored on code a previous run left.
        let _ = std::fs::remove_dir_all(&sandbox);
        std::fs::create_dir_all(&sandbox)
            .with_context(|| format!("creating sandbox {}", sandbox.display()))?;

        let cfg = agent::AgentConfig {
            sandbox: sandbox.clone(),
            max_turns: self.max_turns,
            command_timeout: self.command_timeout,
            request_timeout: self.request_timeout,
            max_tokens: self.max_tokens,
            cargo_target_dir: self.cargo_target_dir.clone(),
        };

        let started = Instant::now();
        let transcript = agent::run_task(&handle, &cfg, PROMPT).await?;
        // 2026-09-26: Taken before the scorer runs.
        let agent_wall_s = started.elapsed().as_secs_f64();
        handle.status(format!("run {index}: scoring"));
        let web = score::webserver_test(
            &sandbox,
            self.cargo_target_dir.as_deref(),
            self.build_timeout,
            self.serve_timeout,
        )
        .await;
        // 2026-09-26: Taken after the scorer, so `wall_s` includes it.
        let wall_s = started.elapsed().as_secs_f64();
        let directions = score::followed_directions(&transcript.commands, &sandbox);

        let mut note = web.error.clone();
        if transcript.hit_turn_cap {
            note = format!("turn cap ({}) reached; {note}", self.max_turns);
        }
        // 2026-09-26: Name the directions the commands did not show, not only
        // their count.
        let missing = directions.missing();
        if !missing.is_empty() {
            note = format!("missing: {}; {note}", missing.join(", "));
        }
        Ok(IterationRow {
            index,
            wall_s,
            agent_wall_s,
            webserver_ok: web.webserver_ok,
            directions,
            turns: transcript.turns,
            tool_calls: transcript.tool_calls,
            completion_tokens: transcript.completion_tokens,
            hit_turn_cap: transcript.hit_turn_cap,
            truncated_turns: transcript.truncated_turns,
            unparsed_call_turns: transcript.unparsed_call_turns,
            note: one_line(note),
        })
    }

    /// 2026-09-26: The record's metrics, computed from the same rows as the
    /// summary and the verdict.
    fn metrics(&self) -> std::collections::BTreeMap<String, f64> {
        let n = self.rows.len();
        let mut m = std::collections::BTreeMap::new();
        m.insert("iterations".to_string(), n as f64);
        m.insert(
            "webserver_ok".to_string(),
            self.rows.iter().filter(|r| r.webserver_ok).count() as f64,
        );
        m.insert(
            "followed_directions".to_string(),
            self.rows.iter().filter(|r| r.directions.overall()).count() as f64,
        );
        m.insert("sum_wall_s".to_string(), self.total_wall());
        m.insert("sum_agent_wall_s".to_string(), self.total_agent_wall());
        m.insert("sum_turns".to_string(), self.total_turns() as f64);
        m.insert("sum_tool_calls".to_string(), self.total_tool_calls() as f64);
        m.insert(
            "sum_completion_tokens".to_string(),
            self.total_completion_tokens() as f64,
        );
        // 2026-09-26: Absent, not 0.0, for a zero-turn tier: a 0.0 would read as
        // the best speed ever recorded.
        if let Some(spt) = self.seconds_per_turn() {
            m.insert("s_per_turn".to_string(), spt);
        }
        // 2026-09-26: Completion tokens per agent-wall second. No BENCH.toml
        // entry bounds it.
        if self.total_agent_wall() > 0.0 {
            m.insert(
                "decode_tps".to_string(),
                self.total_completion_tokens() as f64 / self.total_agent_wall(),
            );
        }

        m.extend(score::per_step_tallies(
            &self.rows.iter().map(|r| &r.directions).collect::<Vec<_>>(),
        ));
        m.extend(trajectory_diagnostics(&self.rows));
        m
    }

    fn verdict(&self) -> Verdict {
        verdict::verdict(
            &self.rows,
            self.total_wall(),
            self.total_agent_wall(),
            self.wall_budget_s,
            self.s_per_turn_budget,
        )
    }
}

/// 2026-09-26: How the iterations' loops ended: counts of turn-cap endings,
/// truncated turns and unparsed-call turns, and the most turns any iteration
/// took. The sums are 0 over an empty tier; the maximum is absent, because "no
/// iteration ran" and "an iteration took zero turns" are different facts.
fn trajectory_diagnostics(rows: &[IterationRow]) -> std::collections::BTreeMap<String, f64> {
    let mut m = std::collections::BTreeMap::new();
    m.insert(
        "sum_hit_turn_cap".to_string(),
        rows.iter().filter(|r| r.hit_turn_cap).count() as f64,
    );
    m.insert(
        "sum_truncated_turns".to_string(),
        rows.iter().map(|r| r.truncated_turns).sum::<usize>() as f64,
    );
    m.insert(
        "sum_unparsed_call_turns".to_string(),
        rows.iter().map(|r| r.unparsed_call_turns).sum::<usize>() as f64,
    );
    if let Some(max) = rows.iter().map(|r| r.turns).max() {
        m.insert("max_iter_turns".to_string(), max as f64);
    }
    m
}

impl Plugin for AgenticWebserver {
    fn metadata(&self) -> &'static PluginMetadata {
        &METADATA
    }

    fn load(&mut self, handle: PluginHandle) -> impl Future<Output = Result<()>> + Send {
        self.started = Some(Instant::now());
        let store = handle.artifacts().clone();
        self.handle = Some(handle.clone());
        async move {
            // 2026-09-26: Fail at load if `cargo` is missing, since nothing could
            // be scored.
            crate::python::run(std::path::Path::new("cargo"), &["--version"], None)
                .await
                .context(
                    "cargo is not on PATH — this benchmark builds the code the model writes",
                )?;
            let root = store.runs_dir(DESCRIPTOR.id)?.join("sandbox");
            std::fs::create_dir_all(&root)?;
            self.sandbox_root = Some(root);
            // 2026-09-26: Build the dependencies into the shared target dir the
            // agent and the scorer use (see [`warm`]).
            self.cargo_target_dir = Some(warm::prepare(&handle).await?);
            Ok(())
        }
    }
}

impl Benchmark for AgenticWebserver {
    fn descriptor(&self) -> &'static BenchmarkDescriptor {
        &DESCRIPTOR
    }

    fn parameters(&self) -> Vec<ParamSpec> {
        params::parameters()
    }

    fn configure(&mut self, values: &ParamValues) -> Result<()> {
        let specs = self.parameters();
        values.validate_against(&specs)?;
        self.iterations = values.usize("iterations")?;
        self.wall_budget_s = values.float("wall_budget_s")?;
        self.s_per_turn_budget = values.float("s_per_turn_budget")?;
        self.max_turns = values.usize("max_turns")?;
        self.command_timeout = Duration::from_secs(values.usize("command_timeout_s")? as u64);
        self.build_timeout = Duration::from_secs(values.usize("build_timeout_s")? as u64);
        self.serve_timeout = Duration::from_secs(values.usize("serve_timeout_s")? as u64);
        self.max_tokens = values.usize("max_tokens")?;
        self.request_timeout = Duration::from_secs(values.usize("request_timeout_s")? as u64);
        self.cursor = 0;
        self.rows.clear();
        Ok(())
    }

    async fn next(&mut self) -> Result<BenchmarkResult> {
        let handle = self.handle()?.clone();
        handle.check_cancelled()?;
        let total = self.iterations as u64;

        if !self.probed {
            self.probed = true;
            http::probe(handle.target(), Duration::from_secs(10))
                .await
                .context("endpoint probe failed — check the target URL and port")?;
            // 2026-09-26: The probe shows a server is listening, not that it
            // decodes; `sanity_check` asks it 2+2 before the first iteration.
            preflight::sanity_check(&handle, Duration::from_secs(60)).await?;
            let root = self.sandbox_root.clone().context("no sandbox root")?;
            return Ok(BenchmarkResult::running("probe", self.elapsed())
                .with_progress(0, total)
                .log_line(LogLine::info(format!(
                    "{total} iteration(s) · sandbox {}",
                    root.display()
                )))
                .log_line(LogLine::warn(
                    "this benchmark executes model-authored shell inside the sandbox",
                )));
        }

        if self.cursor >= self.iterations {
            if self.rows.is_empty() {
                bail!("no iterations ran");
            }
            return Ok(BenchmarkResult {
                status: RunStatus::Completed,
                ..BenchmarkResult::running("done", self.elapsed())
            }
            .with_progress(total, total)
            .with_summary(self.summary())
            .with_table(self.table())
            .with_metrics(self.metrics())
            .with_verdict(self.verdict()));
        }

        let index = self.cursor;
        handle.status(format!("run {index}/{}", self.iterations));
        let row = self.run_iteration(index).await?;
        let line = LogLine::info(format!(
            "run {index}: {} · {}/6 steps · {:.1}s · {} turns{}",
            if row.webserver_ok {
                "webserver_ok"
            } else {
                "FAILED"
            },
            row.directions.met(),
            row.wall_s,
            row.turns,
            if row.note.is_empty() {
                String::new()
            } else {
                format!(" · {}", row.note)
            }
        ));
        self.rows.push(row);
        self.cursor += 1;
        handle.progress(self.cursor as u64, total);
        Ok(
            BenchmarkResult::running(format!("run {index}"), self.elapsed())
                .with_progress(self.cursor as u64, total)
                .with_summary(self.summary())
                .with_table(self.table())
                .log_line(line),
        )
    }

    /// 2026-09-26: Sandboxes are kept: after a failed iteration the code the
    /// model wrote is the evidence. Each is wiped when its index runs again.
    async fn cleanup(&mut self) -> Result<()> {
        Ok(())
    }
}

#[cfg(test)]
#[path = "agentic_diagnostics_tests.rs"]
mod diagnostics_tests;
#[cfg(test)]
#[path = "agentic_tests.rs"]
mod tests;
