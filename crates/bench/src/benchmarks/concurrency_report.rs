// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-26: The concurrency sweep's outputs: the result table, the summary
//! stats, the gate metrics map and the per-cell evidence log line.
//!
//! Owner: bench (concurrency).
//! Invariants: the per-rung throughput and TTFT keys come only from cells
//! for which `CellRow::comparable` holds.

use super::{
    BTreeMap, Cell, CellRow, CellStyle, Column, ConcurrencySweep, RequestEvidence, ResultTable,
    Stat, stats,
};

impl ConcurrencySweep {
    pub(super) fn table(&self) -> ResultTable {
        let mut t = ResultTable::new(
            "LATENCY / THROUGHPUT",
            vec![
                Column::right("ISL", 6),
                Column::right("Conc", 5),
                Column::right("TTFT p50", 9),
                Column::right("p90", 8),
                Column::right("p99", 8),
                Column::right("TPOT p50", 9),
                Column::right("p90", 8),
                Column::right("E2E p50", 9),
                Column::right("tok/s", 8),
                Column::right("min tok", 7),
                Column::right("min cache%", 10),
                Column::right("err", 4),
            ],
        );
        for r in &self.rows {
            t.push(vec![
                Cell::new(r.isl.to_string()),
                Cell::new(r.conc.to_string()),
                Cell::styled(stats::fmt_ms(r.ttft.p50), CellStyle::Accent),
                Cell::new(stats::fmt_ms(r.ttft.p90)),
                Cell::new(stats::fmt_ms(r.ttft.p99)),
                Cell::styled(stats::fmt_ms(r.tpot.p50), CellStyle::Accent),
                Cell::new(stats::fmt_ms(r.tpot.p90)),
                Cell::new(stats::fmt_ms(r.e2e_p50)),
                // 2026-09-26: A non-comparable cell's tok/s is still shown, with
                // a `*` and the Bad style.
                if r.vacuous || r.cache_uncontrolled {
                    Cell::styled(format!("{:.1}*", r.throughput), CellStyle::Bad)
                } else {
                    Cell::styled(format!("{:.1}", r.throughput), CellStyle::Good)
                },
                Cell::new(
                    r.min_completion()
                        .map(|v| v.to_string())
                        .unwrap_or_else(|| "—".into()),
                ),
                Cell::new(
                    r.min_cached_prompt_pct()
                        .map(|v| format!("{v:.0}"))
                        .unwrap_or_else(|| "—".into()),
                ),
                Cell::styled(
                    r.errors.to_string(),
                    if r.errors == 0 {
                        CellStyle::Dim
                    } else {
                        CellStyle::Bad
                    },
                ),
            ]);
        }
        t
    }

    pub(super) fn summary(&self) -> Vec<Stat> {
        let peak = self
            .rows
            .iter()
            .filter(|r| r.comparable())
            .max_by(|a, b| a.throughput.total_cmp(&b.throughput));
        let best_ttft = self
            .rows
            .iter()
            .filter_map(|r| r.ttft.p50)
            .fold(f64::INFINITY, f64::min);
        vec![
            Stat::new(
                "Peak throughput",
                peak.map(|r| format!("{:.1}", r.throughput))
                    .unwrap_or_else(|| "—".into()),
                "tok/s",
            )
            .with_style(CellStyle::Good),
            Stat::new(
                "at concurrency",
                peak.map(|r| r.conc.to_string())
                    .unwrap_or_else(|| "—".into()),
                "",
            ),
            Stat::new(
                "Best TTFT p50",
                if best_ttft.is_finite() {
                    format!("{best_ttft:.0}")
                } else {
                    "—".into()
                },
                "ms",
            )
            .with_style(CellStyle::Accent),
            Stat::new(
                "Cells",
                format!("{}/{}", self.rows.len(), self.cells.len()),
                "",
            ),
        ]
    }

    /// 2026-09-26: The gate metrics. Throughput and TTFT keys come only from
    /// comparable cells; `min_completion_tokens` spans every request, because
    /// it is the evidence for the exclusion.
    pub(super) fn metrics(&self) -> BTreeMap<String, f64> {
        let mut m = BTreeMap::new();
        // 2026-09-26: Per C, the best comparable aggregate across ISLs, and
        // the TTFT p50 of that cell.
        let mut per_c: BTreeMap<usize, &CellRow> = BTreeMap::new();
        for r in self.rows.iter().filter(|r| r.comparable()) {
            let slot = per_c.entry(r.conc).or_insert(r);
            if r.throughput > slot.throughput {
                *slot = r;
            }
        }
        for (c, r) in &per_c {
            m.insert(format!("c{c}_aggregate_tok_s"), r.throughput);
            if let Some(t) = r.ttft.p50 {
                m.insert(format!("c{c}_ttft_p50_ms"), t);
            }
            // 2026-09-26: Published so a slow cell can be told from a serial
            // one.
            if let Some(a) = r.accept_len() {
                m.insert(format!("c{c}_accept_len"), a);
            }
            r.instrument_metrics(&format!("c{c}_"), self.energy.idle(), &mut m);
        }
        self.energy.metrics(&mut m);
        if let Some(peak) = per_c.values().map(|r| r.throughput).max_by(f64::total_cmp) {
            m.insert("peak_aggregate_tok_s".to_string(), peak);
        }
        if let Some(min) = self.rows.iter().filter_map(CellRow::min_completion).min() {
            m.insert("min_completion_tokens".to_string(), min as f64);
        }
        if let Some(min) = self
            .rows
            .iter()
            .filter_map(CellRow::min_cached_prompt)
            .min()
        {
            m.insert("min_cached_prompt_tokens".to_string(), min as f64);
        }
        if let Some(min) = self
            .rows
            .iter()
            .filter_map(CellRow::min_cached_prompt_pct)
            .reduce(f64::min)
        {
            m.insert("min_cached_prompt_pct".to_string(), min);
        }
        m.insert(
            "vacuous_cells".to_string(),
            self.rows.iter().filter(|r| r.vacuous).count() as f64,
        );
        m.insert(
            "cache_uncontrolled_cells".to_string(),
            self.rows.iter().filter(|r| r.cache_uncontrolled).count() as f64,
        );
        // 2026-09-26: Cells below the 1.5 accept depth (`arm_is_not_mtp`),
        // counted apart from `vacuous_cells`; they are not excluded.
        m.insert(
            "non_mtp_arm_cells".to_string(),
            self.rows.iter().filter(|r| r.arm_is_not_mtp()).count() as f64,
        );
        m
    }
}

/// 2026-09-26: One per-cell log line: delivered tokens and cached/prompt
/// tokens per request, a finish-reason histogram, and the server's TTFT and
/// decode-rate medians when reported.
pub(super) fn evidence_line(isl: usize, conc: usize, requests: &[RequestEvidence]) -> String {
    let toks: Vec<String> = requests
        .iter()
        .map(|r| r.completion_tokens.to_string())
        .collect();
    let cached: Vec<String> = requests
        .iter()
        .map(|r| format!("{}/{}", r.cached_prompt_tokens, r.prompt_tokens))
        .collect();
    let mut finish: BTreeMap<&str, usize> = BTreeMap::new();
    for r in requests {
        *finish
            .entry(r.finish_reason.as_deref().unwrap_or("?"))
            .or_default() += 1;
    }
    let finish: Vec<String> = finish.iter().map(|(k, n)| format!("{k}×{n}")).collect();
    let sttft: Vec<f64> = requests.iter().filter_map(|r| r.server_ttft_ms).collect();
    let stps: Vec<f64> = requests.iter().filter_map(|r| r.server_tps).collect();
    let mut line = format!(
        "evidence isl {isl} conc {conc}: tok [{}] · cached [{}] · finish [{}]",
        toks.join(","),
        cached.join(","),
        finish.join(",")
    );
    if let Some(v) = stats::percentile(&sttft, 50) {
        line.push_str(&format!(" · server ttft p50 {v:.0} ms"));
    }
    if let Some(v) = stats::percentile(&stps, 50) {
        line.push_str(&format!(" · server decode p50 {v:.1} tok/s"));
    }
    line
}
