// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-26: The concurrency sweep: for every (ISL × concurrency) cell, send
//! `conc` streaming requests at once and report client TTFT, TPOT and E2E
//! percentiles plus the batch's aggregate output tok/s. Each `next()` runs one
//! cell. Every request's delivered evidence is kept, and a cell that is
//! vacuous (`concurrency_vacuity.rs`), errored or cache-uncontrolled is left
//! out of the gate metrics.
//!
//! Owner: bench (concurrency).
//! Invariants: the per-rung and peak throughput keys in [`ConcurrencySweep`]'s
//! metrics come only from cells for which `CellRow::comparable` holds.

use crate::hardware::Sensitivity;
use crate::hardware::energy::EnergyWindow;
use crate::hardware::energy_sampler::EnergyMeter;
use crate::http::{GapSample, GapStats};
use std::collections::BTreeMap;
use std::future::Future;
use std::time::{Duration, Instant};

use anyhow::{Context, Result, bail};
use serde_json::json;

use crate::benchmark::{Benchmark, BenchmarkDescriptor};
use crate::benchmarks::stats::{self, Percentiles, PromptMode};
use crate::http;
use crate::metadata::PluginMetadata;
use crate::params::{ParamKind, ParamSpec, ParamValue, ParamValues};
use crate::plugin::{Plugin, PluginHandle};
use crate::result::{
    BenchmarkResult, Cell, CellStyle, Column, LogLine, ResultTable, RunStatus, Stat,
};
use cache::{
    SSM_CACHE_SLOTS_KEY, WARM_CACHE_FLOOR, cache_is_uncontrolled, slots_needed, warm_cache_capable,
};
pub use descriptors::{DESCRIPTOR, DFLASH2_DESCRIPTOR, METADATA, MOE_DESCRIPTOR};
use descriptors::{PEAK_FLOOR, RUNGS};
use prompt::{Fixture, prompt_plan};
use report::evidence_line;
use vacuity::{Delivery, VACUITY_FLOOR};

/// 2026-09-26: The natural fixture's ask, appended after the ISL filler.
const CODE_TASK: &str = "Ignore the reference text above. Task: write a complete, \
    production-quality MinHeap class in Python with insert, peek_min, extract_min, \
    decrease_key and heapify methods — full docstrings, input validation and a worked \
    usage example — followed by a unit-test class covering every method, including \
    empty-heap, duplicate-key and single-element edge cases. Write every method and \
    every test out in full; do not summarize or elide any code.";

/// 2026-09-26: The published ladder's ask: `SUFFIX_ESSAY` in
/// `bench/ladder38/harness_w55_conc_ladder.py`, byte for byte.
/// `concurrency_essay_tests.rs` pins whole prompts against SHA-256 digests of
/// that harness's output.
///
/// Kept here rather than as a `stats::PromptMode` variant: `gate::coverage`
/// excludes this file from every non-concurrency gate's invalidation set but
/// does not exclude `stats.rs`, so a prompt added there would invalidate every
/// gate.
const ESSAY_TASK: &str = " Using the text above only as a starting point, write a long, \
    richly detailed essay that keeps introducing new specifics, examples and vocabulary. \
    Never repeat a sentence or paraphrase one you have already written. \
    Do not summarise and do not stop early.";

/// 2026-09-26: The harness's `NONCE_WIDTH`. A fixed-width nonce keeps every
/// request's prompt the same length.
const ESSAY_NONCE_WIDTH: usize = 6;

/// 2026-09-26: `req NNNNNN` for request `i`, taken modulo
/// `10^ESSAY_NONCE_WIDTH` so it never widens, as the harness's `make_prompt`
/// renders its nonce.
fn essay_nonce_tag(i: usize) -> String {
    let modulus = 10usize.pow(ESSAY_NONCE_WIDTH as u32);
    format!("req {:0width$}", i % modulus, width = ESSAY_NONCE_WIDTH)
}

/// 2026-09-26: What one completed request delivered, as parsed by
/// `http::ChatOutcome` from the stream.
#[derive(Clone, Debug, Default)]
struct RequestEvidence {
    completion_tokens: usize,
    prompt_tokens: usize,
    cached_prompt_tokens: usize,
    finish_reason: Option<String>,
    server_ttft_ms: Option<f64>,
    server_tps: Option<f64>,
    /// 2026-09-26: `usage.completion_tokens_details.accepted_prediction_tokens`.
    accepted_prediction_tokens: Option<usize>,
}

#[derive(Default)]
struct CellRow {
    isl: usize,
    conc: usize,
    ttft: Percentiles,
    tpot: Percentiles,
    /// 2026-09-26: Server-clock ITL; empty when the server does not report
    /// `usage.decode_time_ms`.
    server_tpot: Percentiles,
    e2e_p50: Option<f64>,
    throughput: f64,
    /// 2026-09-26: `throughput`'s numerator and the J/token denominator.
    tokens: usize,
    errors: usize,
    requests: Vec<RequestEvidence>,
    vacuous: bool,
    cache_uncontrolled: bool,
    gaps: Option<GapStats>,
    energy: Option<EnergyWindow>,
}

#[derive(Default)]
pub struct ConcurrencySweep {
    handle: Option<PluginHandle>,
    cells: Vec<(usize, usize)>,
    cursor: usize,
    osl: usize,
    warmup: usize,
    fixture: Fixture,
    timeout: Duration,
    rows: Vec<CellRow>,
    started: Option<Instant>,
    probed: bool,
    /// 2026-09-26: Verdict floors in tok/s; with all at 0.0 an error-free
    /// sweep gets an info verdict.
    floors: verdict::Floors,
    /// 2026-09-26: Started after the endpoint probe, with the idle baseline
    /// taken before the first cell; one window per measured batch.
    energy: EnergyMeter,
}

impl ConcurrencySweep {
    fn handle(&self) -> Result<&PluginHandle> {
        self.handle.as_ref().context("benchmark was not loaded")
    }

    /// 2026-09-26: One verdict-floor spec. Floors decide only the verdict and
    /// cannot change a measured number.
    fn floor_spec(key: &'static str, label: &'static str) -> ParamSpec {
        ParamSpec::new(
            key,
            label,
            "Run-verdict floor on this rung's aggregate tok/s (comparable cells only). 0 \
             disables (a standalone run reports an info verdict); under --pull-request-gate \
             it is auto-filled from the variant's BENCH.toml `min` bound. Vacuous or errored \
             sweeps never PASS regardless.",
            ParamKind::Float {
                min: 0.0,
                max: 100_000.0,
            },
            // 2026-09-26: 0.0 is the documented off state, not an implicit bar.
            ParamValue::Float(0.0),
        )
    }

    fn elapsed(&self) -> Duration {
        self.started.map(|s| s.elapsed()).unwrap_or_default()
    }

    /// 2026-09-26: `Err` for a transport failure and for a stream the server
    /// failed (an in-band error frame, or no terminal frame; see
    /// `http::chat_stream`). A completed request with zero tokens is `Ok`.
    async fn one(&self, isl: usize, prefix_tag: String) -> Result<http::ChatOutcome> {
        let handle = self.handle()?;
        let target = handle.target();
        let body = self.request_body(&target.model, isl, &prefix_tag);
        http::chat_stream(target, &body, self.timeout).await
    }
}

impl Plugin for ConcurrencySweep {
    fn metadata(&self) -> &'static PluginMetadata {
        &METADATA
    }

    fn load(&mut self, handle: PluginHandle) -> impl Future<Output = Result<()>> + Send {
        self.handle = Some(handle);
        self.started = Some(Instant::now());
        async { Ok(()) }
    }
}

impl Benchmark for ConcurrencySweep {
    fn descriptor(&self) -> &'static BenchmarkDescriptor {
        &DESCRIPTOR
    }

    fn parameters(&self) -> Vec<ParamSpec> {
        let mut specs = vec![
            ParamSpec::new(
                "concurrencies",
                "Concurrency levels",
                "How many requests are in flight at once, one sweep column each.",
                ParamKind::IntList { min: 1, max: 256 },
                ParamValue::IntList(vec![1, 2, 4, 8, 16, 32]),
            ),
            ParamSpec::new(
                "isls",
                "Input lengths",
                "Prompt sizes in tokens. Must fit inside the server's --max-seq-len with the output.",
                ParamKind::IntList {
                    min: 16,
                    max: 131_072,
                },
                ParamValue::IntList(vec![128, 512, 1024, 2048]),
            ),
            ParamSpec::new(
                "osl",
                "Output tokens",
                "Max tokens per request.",
                ParamKind::Int { min: 1, max: 8192 },
                ParamValue::Int(128),
            ),
            ParamSpec::new(
                "warmup",
                "Warm-up rounds",
                "Unmeasured rounds per cell. Each round runs every exact prompt in the measured \
                 batch so prefix-cache state is controlled before timing.",
                ParamKind::Int { min: 0, max: 8 },
                ParamValue::Int(1),
            ),
            ParamSpec::new(
                "prompt_mode",
                "Prompt mode",
                // 2026-09-26: No mode forces the output budget (the server does
                // not accept `ignore_eos`); short completions are caught by the
                // vacuity rule.
                "natural (default) poses a code-generation task that reliably fills the output \
                 budget; count appends a counting instruction the model may still stop early on; \
                 essay sends the published ladder38 request (bench/ladder38/harness_w55_conc_ladder.py \
                 essay mode: the long-essay ask, a fixed-width request nonce, presence/frequency \
                 penalty pinned to 0.0) byte for byte, so a cell can be compared with the published \
                 ISL 128 / OSL 1024 ladder. None forces the budget — under-budget cells are \
                 flagged vacuous.",
                ParamKind::Choice(&["natural", "count", "essay"]),
                ParamValue::Text("natural".into()),
            ),
            ParamSpec::new(
                "request_timeout_s",
                "Request timeout",
                "Seconds before a single request is abandoned and counted as an error.",
                ParamKind::Int { min: 10, max: 3600 },
                ParamValue::Int(600),
            ),
        ];
        // 2026-09-26: One floor per entry of `RUNGS`, plus the peak.
        specs.extend(
            RUNGS
                .iter()
                .map(|(_, key, _, label)| Self::floor_spec(key, label)),
        );
        specs.push(Self::floor_spec(PEAK_FLOOR.0, PEAK_FLOOR.2));
        specs
    }

    fn configure(&mut self, values: &ParamValues) -> Result<()> {
        let specs = self.parameters();
        values.validate_against(&specs)?;
        let concurrencies = values.int_list("concurrencies")?.to_vec();
        let isls = values.int_list("isls")?.to_vec();
        // 2026-09-26: ISL-major: a full concurrency curve at one prompt size
        // before the next size.
        self.cells = isls
            .iter()
            .flat_map(|isl| {
                concurrencies
                    .iter()
                    .map(move |c| (*isl as usize, *c as usize))
            })
            .collect();
        self.osl = values.usize("osl")?;
        self.warmup = values.usize("warmup")?;
        self.fixture = Fixture::parse(values.text("prompt_mode")?)
            .context("prompt_mode must be natural, count or essay")?;
        self.timeout = Duration::from_secs(values.usize("request_timeout_s")? as u64);
        let mut per_c = Vec::with_capacity(RUNGS.len());
        for (c, key, _, _) in RUNGS {
            per_c.push((c, values.float(key)?));
        }
        self.floors = verdict::Floors {
            per_c,
            peak: values.float(PEAK_FLOOR.0)?,
        };
        self.cursor = 0;
        self.rows.clear();
        // 2026-09-26: A fresh meter per configuration; dropping a live sampler
        // kills its child process (`kill_on_drop`).
        self.energy = EnergyMeter::default();
        Ok(())
    }

    async fn next(&mut self) -> Result<BenchmarkResult> {
        let handle = self.handle()?.clone();
        handle.check_cancelled()?;

        // 2026-09-26: Probe first, so a wrong port fails at once rather than
        // as a sweep of transport errors.
        if !self.probed {
            self.probed = true;
            http::probe(handle.target(), Duration::from_secs(10))
                .await
                .context("endpoint probe failed — check the target URL and port")?;
            let total = self.cells.len() as u64;
            if total == 0 {
                bail!("no cells to run — check the concurrency and input-length lists");
            }
            // 2026-09-26: Nothing is in flight yet, so the idle baseline
            // precedes the first measured window.
            for line in self.energy.start(handle.target()).await {
                handle.log(line.level, line.text);
            }
            return Ok(BenchmarkResult::running("probe", self.elapsed())
                .with_progress(0, total)
                .log_line(LogLine::info(format!(
                    "{} · model {} · {total} cells",
                    handle.target().base_url,
                    handle.target().model
                ))));
        }

        if self.cursor >= self.cells.len() {
            let errors: usize = self.rows.iter().map(|r| r.errors).sum();
            let vacuous = self.rows.iter().filter(|r| r.vacuous).count();
            let cache_uncontrolled = self.rows.iter().filter(|r| r.cache_uncontrolled).count();
            let non_mtp_arm = self.rows.iter().filter(|r| r.arm_is_not_mtp()).count();
            // 2026-09-26: Stopped before `metrics()`, which records its cost.
            let sampler_cost_line = self.energy.stop().await;
            // 2026-09-26: The verdict reads the same metrics map the record
            // carries.
            let metrics = self.metrics();
            let verdict = verdict::sweep_verdict(
                &metrics,
                self.rows.len(),
                errors,
                verdict::Exclusions {
                    vacuous,
                    cache_uncontrolled,
                    non_mtp_arm,
                },
                VACUITY_FLOOR * 100.0,
                &self.floors,
            );
            let mut frame = BenchmarkResult {
                status: RunStatus::Completed,
                ..BenchmarkResult::running("done", self.elapsed())
            }
            .with_progress(self.cells.len() as u64, self.cells.len() as u64)
            .with_summary(self.summary())
            .with_table(self.table())
            .with_metrics(metrics)
            .with_verdict(verdict);
            if let Some(line) = sampler_cost_line {
                frame = frame.log_line(line);
            }
            let unmeasured = self.rows.iter().filter(|r| r.tpot.p50.is_none()).count();
            if unmeasured > 0 {
                frame = frame.log_line(LogLine::warn(format!(
                    "TPOT unmeasured in {unmeasured} cell(s): the endpoint sent the whole reply \
                     in one SSE delta, so there is no inter-token interval to time. Raise the \
                     output-token budget to measure decode."
                )));
            }
            return Ok(frame);
        }

        let (isl, conc) = self.cells[self.cursor];
        let row = self.run_cell(isl, conc).await?;
        let line = LogLine::info(format!(
            "isl {isl} conc {conc}: ttft p50 {} ms · tpot p50 {} ms (server clock {} ms) · \
             {:.1} tok/s{}",
            stats::fmt_ms(row.ttft.p50),
            stats::fmt_ms(row.tpot.p50),
            stats::fmt_ms(row.server_tpot.p50),
            row.throughput,
            if row.vacuous { " (vacuous)" } else { "" }
        ));
        self.rows.push(row);
        self.cursor += 1;
        handle.progress(self.cursor as u64, self.cells.len() as u64);
        Ok(
            BenchmarkResult::running(format!("isl {isl} · conc {conc}"), self.elapsed())
                .with_progress(self.cursor as u64, self.cells.len() as u64)
                .with_summary(self.summary())
                .with_table(self.table())
                .log_line(line),
        )
    }
}

#[path = "concurrency_instruments.rs"]
mod instruments;

#[path = "concurrency_verdict.rs"]
mod verdict;

#[path = "concurrency_vacuity.rs"]
mod vacuity;

#[path = "concurrency_descriptors.rs"]
mod descriptors;

#[path = "concurrency_prompt.rs"]
mod prompt;

#[path = "concurrency_cache.rs"]
mod cache;

#[path = "concurrency_cell.rs"]
mod cell;

#[path = "concurrency_report.rs"]
mod report;

#[cfg(test)]
#[path = "concurrency_tests.rs"]
mod concurrency_tests;

#[cfg(test)]
#[path = "concurrency_verdict_tests.rs"]
mod concurrency_verdict_tests;

#[cfg(test)]
#[path = "concurrency_vacuity_tests.rs"]
mod concurrency_vacuity_tests;

#[cfg(test)]
#[path = "concurrency_moe_tests.rs"]
mod concurrency_moe_tests;
