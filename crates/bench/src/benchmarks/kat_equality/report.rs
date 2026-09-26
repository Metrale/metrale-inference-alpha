// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-26: How an equality run is presented: functions of the
//! [`super::compare`] results, with no I/O.
//!
//! Owner: bench, kat_equality.
//! Invariants: none beyond the types.

use std::collections::BTreeMap;

use super::compare::{OrderRun, SampleVerdict, Score, verdict_for};
use crate::result::{Cell, CellStyle, Column, ResultTable, Stat};

pub(super) fn summary(s: &Score) -> Vec<Stat> {
    vec![
        Stat::new("orders", s.orders.to_string(), ""),
        Stat::new("samples", s.samples.to_string(), ""),
        Stat::new("identical", s.equal.to_string(), ""),
        Stat::new("DIVERGED", s.diverged.to_string(), ""),
    ]
}

/// 2026-09-26: One row per sample that is not equal, up to 40 and then a row
/// counting the rest; when everything agreed, a single row saying so. Equal
/// samples get no row, so they cannot bury the ones that differ.
pub(super) fn table(s: &Score, runs: &[OrderRun]) -> ResultTable {
    let mut t = ResultTable::new(
        "SAMPLES THAT CHANGED WITH THE ORDER",
        vec![
            Column::left("Sample", 34),
            Column::left("Result", 12),
            Column::left("Detail", 46),
        ],
    );
    let Some(reference) = runs.first() else {
        return t;
    };
    let mut shown = 0usize;
    for obs in &reference.observations {
        let v = verdict_for(&obs.sample_id, reference, &runs[1..]);
        let (what, style, detail) = match v {
            SampleVerdict::Equal => continue,
            SampleVerdict::Diverged {
                other_order,
                common_prefix,
            } => (
                "DIVERGED",
                CellStyle::Bad,
                format!("differs from `{other_order}` after {common_prefix} identical bytes"),
            ),
            SampleVerdict::Unmeasured(reason) => ("unmeasured", CellStyle::Bad, reason),
        };
        t.push(vec![
            Cell::new(obs.sample_id.clone()),
            Cell::styled(what.to_string(), style),
            Cell::new(detail),
        ]);
        shown += 1;
        if shown >= 40 {
            let left = s.diverged + s.unmeasured - shown;
            if left > 0 {
                t.push(vec![
                    Cell::new(format!("… {left} more")),
                    Cell::new(String::new()),
                    Cell::new("see the per-sample metrics".to_string()),
                ]);
            }
            break;
        }
    }
    if shown == 0 {
        t.push(vec![
            Cell::styled("all samples".to_string(), CellStyle::Good),
            Cell::styled("identical".to_string(), CellStyle::Good),
            Cell::new(format!("byte-for-byte across {} request orders", s.orders)),
        ]);
    }
    t
}

/// 2026-09-26: Raw gate numbers. Every class is a key even at zero: a missing
/// key and a zero must stay distinguishable to whatever compares records.
pub(super) fn metrics(s: &Score) -> BTreeMap<String, f64> {
    [
        ("orders", s.orders),
        ("samples", s.samples),
        ("identical", s.equal),
        ("diverged", s.diverged),
        ("unmeasured", s.unmeasured),
        // 2026-09-26: The vacuity count, which a BENCH.toml
        // `[benchmarks.metrics.empty_replies]` bound can refuse: an empty reply
        // agrees perfectly and has measured nothing.
        ("empty_replies", s.empty_replies),
    ]
    .into_iter()
    .map(|(k, v)| (k.to_string(), v as f64))
    .collect()
}

#[cfg(test)]
#[path = "report_tests.rs"]
mod report_tests;
