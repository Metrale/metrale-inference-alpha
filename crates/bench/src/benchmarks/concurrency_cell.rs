// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-26: One concurrency cell: its warm-up rounds, the measured batch
//! fired at once, and the `CellRow` built from what each request delivered.
//!
//! Owner: bench (concurrency).
//! Invariants: a row is comparable only when it has no errors, is not
//! vacuous and is cache-controlled.

use super::{
    CellRow, ConcurrencySweep, Context, Delivery, GapSample, Instant, Percentiles, RequestEvidence,
    Result, SSM_CACHE_SLOTS_KEY, WARM_CACHE_FLOOR, cache_is_uncontrolled, evidence_line,
    prompt_plan, slots_needed, stats, warm_cache_capable,
};

impl CellRow {
    pub(super) fn min_completion(&self) -> Option<usize> {
        self.requests.iter().map(|r| r.completion_tokens).min()
    }
    pub(super) fn min_cached_prompt(&self) -> Option<usize> {
        self.requests.iter().map(|r| r.cached_prompt_tokens).min()
    }
    pub(super) fn min_cached_prompt_pct(&self) -> Option<f64> {
        self.requests
            .iter()
            .map(|request| {
                if request.prompt_tokens == 0 {
                    0.0
                } else {
                    request.cached_prompt_tokens as f64 / request.prompt_tokens as f64 * 100.0
                }
            })
            .reduce(f64::min)
    }
    /// 2026-09-26: The minimum accept depth (`stats::accept_len`) across the
    /// cell's requests, or `None` when any request did not report it. The
    /// minimum, because one request on the serial arm makes the cell a
    /// mixture.
    pub(super) fn accept_len(&self) -> Option<f64> {
        self.requests
            .iter()
            .map(|r| super::stats::accept_len(r.completion_tokens, r.accepted_prediction_tokens))
            .try_fold(f64::INFINITY, |acc, v| v.map(|v| acc.min(v)))
            .filter(|v| v.is_finite())
    }

    /// 2026-09-26: True when the minimum accept depth is below 1.5; a request
    /// on the serial arm reports exactly 1.0. Counted as `non_mtp_arm_cells`
    /// and never gated on: the benchmark cannot know which arm should have
    /// run. A missing accept field is not treated as serial.
    pub(super) fn arm_is_not_mtp(&self) -> bool {
        self.accept_len().is_some_and(|a| a < 1.5)
    }

    /// 2026-09-26: No errors, not vacuous, and cache-controlled: the only rows
    /// the gate metrics may quote. The speculation arm is not a condition.
    pub(super) fn comparable(&self) -> bool {
        self.errors == 0 && !self.vacuous && !self.cache_uncontrolled
    }
}

impl ConcurrencySweep {
    pub(super) async fn run_cell(&mut self, isl: usize, conc: usize) -> Result<CellRow> {
        let handle = self.handle()?.clone();
        let plan = prompt_plan(conc, self.warmup, self.fixture);
        for (w, tags) in plan.warmup_rounds.iter().enumerate() {
            handle.check_cancelled()?;
            handle.status(format!(
                "isl {isl} · conc {conc} · warmup round {}/{}",
                w + 1,
                self.warmup
            ));
            // 2026-09-26: Warm every exact prompt the measured batch will send,
            // concurrently: a round's tags are pairwise distinct, so there is
            // no duplicate insert to race. A failed warm-up request fails the
            // cell.
            let warmed =
                futures::future::join_all(tags.iter().map(|tag| self.one(isl, tag.clone()))).await;
            for (tag, outcome) in tags.iter().zip(warmed) {
                outcome.with_context(|| {
                    format!("isl {isl} conc {conc}: warm-up prompt {tag} failed")
                })?;
            }
        }
        handle.check_cancelled()?;
        handle.status(format!("isl {isl} · conc {conc} · {conc} in flight"));

        let batch_start = Instant::now();
        let futures: Vec<_> = plan
            .measured
            .into_iter()
            .map(|tag| self.one(isl, tag))
            .collect();
        let outcomes = futures::future::join_all(futures).await;
        let batch_end = Instant::now();
        let wall = batch_end
            .duration_since(batch_start)
            .as_secs_f64()
            .max(1e-6);

        let mut ttft = Vec::new();
        let mut tpot = Vec::new();
        let mut server_tpot = Vec::new();
        let mut gaps = GapSample::default();
        let mut e2e = Vec::new();
        let mut requests = Vec::new();
        let mut tokens = 0usize;
        let mut errors = 0usize;
        for outcome in outcomes {
            match outcome {
                Ok(o) => {
                    if let Some(v) = o.ttft_ms {
                        ttft.push(v);
                    }
                    if let Some(v) = o.tpot_ms {
                        tpot.push(v);
                    }
                    if let Some(v) = o.server_tpot_ms() {
                        server_tpot.push(v);
                    }
                    gaps.merge(&o.arrival_gaps);
                    e2e.push(o.e2e_ms);
                    tokens += o.completion_tokens;
                    requests.push(RequestEvidence {
                        completion_tokens: o.completion_tokens,
                        prompt_tokens: o.prompt_tokens,
                        cached_prompt_tokens: o.cached_prompt_tokens,
                        finish_reason: o.finish_reason.clone(),
                        server_ttft_ms: o.server_ttft_ms,
                        server_tps: o.server_tps,
                        accepted_prediction_tokens: o.accepted_prediction_tokens,
                    });
                }
                Err(e) => {
                    errors += 1;
                    handle.warn(format!("isl {isl} conc {conc}: {e:#}"));
                }
            }
        }
        let delivery = Delivery::of(&requests, self.osl);
        let vacuous = delivery.is_vacuous();
        // 2026-09-26: A pool too small for this cell's warmed prompts measures
        // cold by construction and is not held to the warm rule. The pool size
        // is what the server was started with (its serve overrides).
        let slots = handle.target().serve_override_usize(SSM_CACHE_SLOTS_KEY);
        let earlier: Vec<usize> = self.cells[..self.cursor]
            .iter()
            .filter(|(i, _)| *i == isl)
            .map(|(_, c)| *c)
            .collect();
        let warm_capable = warm_cache_capable(conc, slots, &earlier);
        let cache_uncontrolled = warm_capable && cache_is_uncontrolled(&requests, self.warmup);
        handle.info(evidence_line(isl, conc, &requests));
        let energy = self.energy.window(batch_start, batch_end);
        if let Some(e) = &energy {
            handle.info(format!(
                "isl {isl} conc {conc}: {} · {tokens} tok in the window",
                e.one_line(self.energy.idle())
            ));
        }
        if !warm_capable {
            handle.info(format!(
                "isl {isl} conc {conc}: cache cold by construction — the server's {} \
                 snapshot slot(s) cannot hold {} warm request(s) after the {:?} cell(s) \
                 ({} needed); the warm rule is not applied to this cell",
                slots.unwrap_or(0),
                conc,
                earlier,
                slots_needed(conc, &earlier),
            ));
        }
        if vacuous {
            handle.warn(format!(
                "isl {isl} conc {conc}: this cell {} — its tok/s is NOT comparable",
                delivery.describe(self.osl),
            ));
        }
        if cache_uncontrolled {
            handle.warn(format!(
                "isl {isl} conc {conc}: warm-up was requested but at least one measured request \
                 reported less than {:.0}% of its prompt as cached — this cell is NOT comparable",
                WARM_CACHE_FLOOR * 100.0,
            ));
        }
        Ok(CellRow {
            isl,
            conc,
            ttft: Percentiles::of(&ttft),
            tpot: Percentiles::of(&tpot),
            server_tpot: Percentiles::of(&server_tpot),
            e2e_p50: stats::percentile(&e2e, 50),
            throughput: tokens as f64 / wall,
            tokens,
            errors,
            requests,
            vacuous,
            cache_uncontrolled,
            gaps: gaps.stats(),
            energy,
        })
    }
}
