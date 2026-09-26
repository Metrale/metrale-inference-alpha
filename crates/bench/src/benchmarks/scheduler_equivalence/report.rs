// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-26: How a scheduler-equivalence run is presented: the terminal frame
//! built from `compare::score`.
//!
//! Owner: bench, scheduler-equivalence gate.
//! Invariants:
//! - No I/O: every function here maps the legs or the score to a value.

use std::collections::BTreeMap;
use std::time::Duration;

use super::compare::{Cell, Leg, Score, metrics, score, verdict};
use super::unmeasured;
use crate::result::{
    BenchmarkResult, Cell as TCell, CellStyle, Column, LogLine, ResultTable, Stat,
};

/// 2026-09-26: The run's terminal frame: summary, table, metrics, verdict, and
/// one warning line per unmeasured request.
pub(super) fn terminal(
    legs: &[Leg],
    diagnostics: &BTreeMap<String, f64>,
    elapsed: Duration,
) -> BenchmarkResult {
    let s = score(legs);
    BenchmarkResult::completed("done", elapsed)
        .with_summary(summary(&s))
        .with_table(table(&s))
        .with_metrics(metrics(&s, diagnostics))
        .with_log(log(&s))
        .with_verdict(verdict(&s))
}

pub(super) fn summary(s: &Score) -> Vec<Stat> {
    let sum = |f: &dyn Fn(&Cell) -> usize| s.cells.iter().map(f).sum::<usize>();
    vec![
        Stat::new("cells", s.cells.len().to_string(), ""),
        Stat::new(
            "samples",
            s.cells.first().map(|c| c.samples).unwrap_or(0).to_string(),
            "",
        ),
        Stat::new(
            "DIVERGED",
            sum(&|c| c.async_vs_sync.diverged).to_string(),
            "",
        ),
        Stat::new(
            "control diverged",
            sum(&|c| c.control_vs_sync.as_ref().map(|k| k.diverged).unwrap_or(0)).to_string(),
            "",
        ),
        Stat::new(
            "unmeasured causes",
            unmeasured::breakdown(&all_unmeasured(s)),
            "",
        ),
    ]
}

fn all_unmeasured(s: &Score) -> Vec<unmeasured::Unmeasured> {
    s.cells.iter().flat_map(|c| c.unmeasured.clone()).collect()
}

/// 2026-09-26: One warning per unmeasured request, in cell order.
pub(super) fn log(s: &Score) -> Vec<LogLine> {
    all_unmeasured(s)
        .iter()
        .map(|u| LogLine::warn(u.to_string()))
        .collect()
}

/// 2026-09-26: One row per cell, agreeing cells included, so the table shows
/// which concurrencies were compared.
pub(super) fn table(s: &Score) -> ResultTable {
    let mut t = ResultTable::new(
        "SYNC VS ASYNC, PER LANE AND CONCURRENCY",
        vec![
            Column::left("Lane", 10),
            Column::left("C", 4),
            Column::left("Samples", 8),
            Column::left("Async diverged", 15),
            Column::left("Control diverged", 17),
            Column::left("Unmeasured", 11),
            Column::left("Causes", 24),
            Column::left("Detail", 30),
        ],
    );
    for c in &s.cells {
        let bad = c.async_vs_sync.diverged > 0
            || c.async_vs_sync.unmeasured > 0
            || c.control_vs_sync
                .as_ref()
                .is_some_and(|k| k.diverged > 0 || k.unmeasured > 0);
        let style = if bad { CellStyle::Bad } else { CellStyle::Good };
        let detail = if c.async_vs_sync.diverged_ids.is_empty() {
            "identical".to_string()
        } else {
            c.async_vs_sync.diverged_ids.join(", ")
        };
        t.push(vec![
            TCell::new(c.lane.label().to_string()),
            TCell::new(c.concurrency.to_string()),
            TCell::new(c.samples.to_string()),
            TCell::styled(c.async_vs_sync.diverged.to_string(), style),
            TCell::new(
                c.control_vs_sync
                    .as_ref()
                    .map(|k| k.diverged.to_string())
                    .unwrap_or_else(|| "-".to_string()),
            ),
            TCell::new(
                (c.async_vs_sync.unmeasured
                    + c.control_vs_sync
                        .as_ref()
                        .map(|k| k.unmeasured)
                        .unwrap_or(0))
                .to_string(),
            ),
            TCell::new(unmeasured::breakdown(&c.unmeasured)),
            TCell::new(detail),
        ]);
    }
    t
}
