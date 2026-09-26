// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-26: The pure half of the decode-floor gate: the pins (constants),
//! the reduced per-run observation, and the evaluation and verdict functions.
//! Nothing here talks to an endpoint, so every verdict path is unit-tested.
//!
//! Owner: bench, decode_floor.
//! Invariants:
//! - `evaluate` returns `Measured` only for exactly `RUNS` observations that
//!   all pass the vacuity pins.
//! - `verdict_for` never passes an `Inconclusive` evaluation.

use std::collections::BTreeMap;

use crate::benchmarks::stats;
use crate::hardware::energy::EnergyWindow;
use crate::http::{self, GapSample};
use crate::result::Verdict;

/// 2026-09-26: Timed runs. The median over this many runs is the metric's
/// definition; a different run count is a different benchmark.
pub(crate) const RUNS: usize = 3;
/// 2026-09-26: Output budget per run (`max_tokens`).
pub(crate) const MAX_TOKENS: usize = 1500;
/// 2026-09-26: Vacuity floor on every run's `completion_tokens`: a shorter run
/// is too short a decode to measure a floor on. It must sit under the
/// model's natural stop on `MINHEAP_PROMPT`, or the gate's own subject is
/// inconclusive every time. Measured 2026-09-24: the three committed records
/// in `.benchmarks/decode-floor/` report `output_tokens` 817.
pub(crate) const MIN_OUTPUT_TOKENS: usize = 750;
/// 2026-09-26: Vacuity floor on the mean derived tokens per decode step.
pub(crate) const MIN_ACCEPT_LEN: f64 = 1.5;

/// 2026-09-26: The committed code prompt, a pin of the benchmark.
pub(crate) const MINHEAP_PROMPT: &str = "Implement a complete, production-quality MinHeap class in Python. Include the methods \
     insert, extract_min, peek, heapify (bottom-up from an arbitrary list), decrease_key, \
     delete_at_index, merge (with another MinHeap), __len__ and __iter__. Every method needs a \
     full docstring with time-complexity analysis. Then write a comprehensive pytest test \
     suite covering the empty heap, a single element, duplicate keys, and long interleaved \
     insert/extract sequences. Finish with a line-by-line explanation of the sift_up and \
     sift_down invariants. Be exhaustive and do not stop early.";

/// 2026-09-26: One timed run, reduced to what the pins and the metrics need.
#[derive(Clone, Debug, Default, PartialEq)]
pub(crate) struct RunObs {
    pub completion_tokens: usize,
    pub server_tps: Option<f64>,
    /// 2026-09-26: `None`: the server reported no details object. `Some(0)`:
    /// reported, nothing accepted. Both are inconclusive, with different
    /// messages.
    pub accepted_prediction_tokens: Option<usize>,
    pub e2e_ms: f64,
    /// 2026-09-26: Client-clock ITL (`http::ChatOutcome::tpot_ms`), recorded
    /// beside the server rate so the transport overhead is on the record.
    pub client_tpot_ms: Option<f64>,
    /// 2026-09-26: Server-clock ITL from the raw window
    /// (`usage.decode_time_ms`). Not a metric key: `server_decode_tok_s`
    /// already carries the server clock. Used by the run log only.
    pub server_tpot_ms: Option<f64>,
    /// 2026-09-26: The run's arrival gaps (jitter); `instrument_metrics`
    /// pools them across the runs.
    pub arrival_gaps: GapSample,
    /// 2026-09-26: GPU-rail energy over this run's request window; set by the
    /// driver from its sampler, `None` when the rail was not sampled.
    pub energy: Option<EnergyWindow>,
}

impl RunObs {
    pub(crate) fn from_outcome(o: &http::ChatOutcome) -> Self {
        Self {
            completion_tokens: o.completion_tokens,
            server_tps: o.server_tps,
            accepted_prediction_tokens: o.accepted_prediction_tokens,
            e2e_ms: o.e2e_ms,
            client_tpot_ms: o.tpot_ms,
            server_tpot_ms: o.server_tpot_ms(),
            arrival_gaps: o.arrival_gaps.clone(),
            energy: None,
        }
    }

    /// 2026-09-26: Emitted tokens per decode step,
    /// `completion / (completion - accepted)`. `None` when it cannot be
    /// derived: no accept field, or `accepted >= completion`.
    pub(crate) fn accept_len(&self) -> Option<f64> {
        // 2026-09-26: SSOT: `stats::accept_len`, which `concurrency` also uses.
        crate::benchmarks::stats::accept_len(
            self.completion_tokens,
            self.accepted_prediction_tokens,
        )
    }
}

/// 2026-09-26: What the runs add up to.
#[derive(Clone, Debug, PartialEq)]
pub(crate) enum Evaluation {
    /// 2026-09-26: A vacuity pin failed; the message names which one and why.
    Inconclusive(String),
    Measured {
        /// 2026-09-26: Median server decode tok/s across the runs: the metric.
        median_decode_tok_s: f64,
        /// 2026-09-26: Minimum `completion_tokens` across runs, so a BENCH.toml
        /// `output_tokens` floor means "every run", not "on average".
        min_output_tokens: usize,
        /// 2026-09-26: Mean of the per-run derived accept lengths.
        accept_len_mean: f64,
    },
}

/// 2026-09-26: The run verdict for an evaluation.
///
/// An inconclusive evaluation is a failing verdict whatever `min_tok_s` is. A
/// Measured run is judged against `min_tok_s` when it is above 0, and gets an
/// info verdict otherwise. The comparison is the raw `median >= min_tok_s`;
/// under the gate `min_tok_s` arrives as the bound's min minus its noise
/// (`bench_resolve::apply_threshold_params`), which matches `gate::scoring`'s
/// `value + noise >= min`.
pub(crate) fn verdict_for(eval: &Evaluation, min_tok_s: f64) -> Verdict {
    match eval {
        Evaluation::Inconclusive(why) => Verdict::fail(format!("INCONCLUSIVE: {why}")),
        Evaluation::Measured {
            median_decode_tok_s,
            accept_len_mean,
            ..
        } => {
            let basis = format!(
                "median decode {median_decode_tok_s:.1} tok/s over {RUNS} pinned runs \
                 (accept_len_mean {accept_len_mean:.2})"
            );
            if min_tok_s <= 0.0 {
                Verdict::info(format!(
                    "{basis} — judged against the BENCH.toml floor under --pull-request-gate"
                ))
            } else if *median_decode_tok_s >= min_tok_s {
                Verdict::pass(format!("{basis} — clears the {min_tok_s:.1} tok/s floor"))
            } else {
                Verdict::fail(format!(
                    "BELOW THE DECODE FLOOR — {basis} vs the {min_tok_s:.1} tok/s floor"
                ))
            }
        }
    }
}

pub(crate) fn evaluate(samples: &[RunObs]) -> Evaluation {
    if samples.len() != RUNS {
        return Evaluation::Inconclusive(format!(
            "{} run(s) completed, the pinned count is {RUNS}",
            samples.len()
        ));
    }
    for (i, s) in samples.iter().enumerate() {
        if s.completion_tokens < MIN_OUTPUT_TOKENS {
            return Evaluation::Inconclusive(format!(
                "run {} emitted {} tokens, below the {MIN_OUTPUT_TOKENS}-token vacuity floor \
                 (of the {MAX_TOKENS} budget) — too short a decode to measure a floor on",
                i + 1,
                s.completion_tokens
            ));
        }
        match s.server_tps {
            None => {
                return Evaluation::Inconclusive(format!(
                    "run {} reported no server decode rate (usage.\"response_token/s\") — without \
                     the server's own clock there is no defensible per-token number",
                    i + 1
                ));
            }
            Some(rate) if !rate.is_finite() || rate <= 0.0 => {
                return Evaluation::Inconclusive(format!(
                    "run {} reported server decode rate {rate}, which is not a finite positive \
                     per-token measurement",
                    i + 1
                ));
            }
            Some(_) => {}
        }
        match s.accepted_prediction_tokens {
            None => {
                return Evaluation::Inconclusive(format!(
                    "run {} reported no usage.completion_tokens_details.\
                     accepted_prediction_tokens — this gate depends on the accept-stats \
                     instrumentation (the commit wiring real MTP accept counts into usage); \
                     serve a binary that has it",
                    i + 1
                ));
            }
            Some(0) => {
                return Evaluation::Inconclusive(format!(
                    "run {} accepted 0 draft tokens — either the serve is not speculating or \
                     the accept-stats instrumentation is not live; a serial-floor number must \
                     not be recorded as the decode floor",
                    i + 1
                ));
            }
            Some(_) => {}
        }
    }
    let mut accept_lens = Vec::with_capacity(samples.len());
    for (i, s) in samples.iter().enumerate() {
        match s.accept_len() {
            Some(l) => accept_lens.push(l),
            None => {
                return Evaluation::Inconclusive(format!(
                    "run {}: accepted ({}) >= completion_tokens ({}) — corrupt accounting, \
                     nothing derivable",
                    i + 1,
                    s.accepted_prediction_tokens.unwrap_or(0),
                    s.completion_tokens
                ));
            }
        }
    }
    let accept_len_mean = accept_lens.iter().sum::<f64>() / accept_lens.len() as f64;
    if accept_len_mean < MIN_ACCEPT_LEN {
        return Evaluation::Inconclusive(format!(
            "accept_len_mean {accept_len_mean:.2} < {MIN_ACCEPT_LEN} — speculation is not \
             engaged at gate depth, so this run measures the serial floor, not the engine"
        ));
    }
    let tps: Vec<f64> = samples.iter().filter_map(|s| s.server_tps).collect();
    // 2026-09-26: stats::median, not stats::percentile(_, 50): the
    // nearest-rank p50 of three samples is the maximum, and the floor must not
    // ride the best run.
    let median = stats::median(&tps).unwrap_or(0.0);
    Evaluation::Measured {
        median_decode_tok_s: median,
        min_output_tokens: samples
            .iter()
            .map(|s| s.completion_tokens)
            .min()
            .unwrap_or(0),
        accept_len_mean,
    }
}

/// 2026-09-26: The instrument keys beside the verdict metrics: the median
/// client-clock ITL, the pooled arrival-gap (jitter) distribution, and the
/// joules over the sampled windows with the tokens decoded inside them. The
/// driver supplies only the idle baseline. The server-clock ITL gets no key:
/// `server_decode_tok_s` already carries it.
pub(crate) fn instrument_metrics(
    samples: &[RunObs],
    idle: Option<&EnergyWindow>,
    m: &mut BTreeMap<String, f64>,
) {
    let client: Vec<f64> = samples.iter().filter_map(|s| s.client_tpot_ms).collect();
    if let Some(v) = stats::median(&client) {
        m.insert("client_tpot_ms".to_string(), v);
    }
    let mut gaps = GapSample::default();
    for s in samples {
        gaps.merge(&s.arrival_gaps);
    }
    if let Some(g) = gaps.stats() {
        g.metrics("", m);
    }
    let windows: Vec<EnergyWindow> = samples.iter().filter_map(|s| s.energy).collect();
    if let Some(total) = EnergyWindow::sum(&windows) {
        // 2026-09-26: Tokens decoded inside the sampled windows: the J/token
        // denominator must cover the same interval as the joules.
        let tokens = samples
            .iter()
            .filter(|s| s.energy.is_some())
            .map(|s| s.completion_tokens)
            .sum();
        total.metrics("", tokens, idle, m);
    }
}
