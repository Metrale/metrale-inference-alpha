// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-26: How a poisoning run is presented: pure functions of the
//! [`super::score::Score`].
//!
//! Owner: bench, SSM poisoning gate.
//! Invariants:
//! - No I/O.

use std::collections::BTreeMap;

use super::compare::TurnDelta;
use super::score::Score;
use crate::result::{Cell, CellStyle, Column, ResultTable, Stat};

fn delta_detail(turns: &[TurnDelta]) -> String {
    turns
        .iter()
        .map(|t| {
            format!(
                "t{}: {}->{}tok fin {:?}->{:?}",
                t.turn, t.ref_tokens, t.replay_tokens, t.ref_finish, t.replay_finish
            )
        })
        .collect::<Vec<_>>()
        .join(", ")
}

/// 2026-09-26: One row per replay round, in round order.
pub(super) fn table(s: &Score) -> ResultTable {
    let mut t = ResultTable::new(
        "REPLAY ROUNDS",
        vec![
            Column::left("Round", 6),
            Column::left("Result", 14),
            Column::left("Detail", 56),
        ],
    );
    let jitter_map: BTreeMap<usize, &Vec<TurnDelta>> =
        s.jittered_rounds.iter().map(|(n, d)| (*n, d)).collect();
    let collapse_map: BTreeMap<usize, &Vec<TurnDelta>> =
        s.collapsed_rounds.iter().map(|(n, d)| (*n, d)).collect();
    // 2026-09-26: Unmeasured rows come from the Score's per-round records.
    let unmeasured_map: BTreeMap<usize, &str> = s
        .unmeasured_rounds
        .iter()
        .map(|(n, r)| (*n, r.as_str()))
        .collect();
    for round in 1..=s.rounds {
        let (what, style, detail) = if let Some(turns) = collapse_map.get(&round) {
            ("COLLAPSED".to_string(), CellStyle::Bad, delta_detail(turns))
        } else if let Some(turns) = jitter_map.get(&round) {
            ("jittered".to_string(), CellStyle::Warn, delta_detail(turns))
        } else if let Some(reason) = unmeasured_map.get(&round) {
            (
                "unmeasured".into(),
                CellStyle::Bad,
                format!("invariant not proven this round — {reason}"),
            )
        } else {
            ("invariant".into(), CellStyle::Good, String::new())
        };
        t.push(vec![
            Cell::new(format!("r{round}")),
            Cell::styled(what, style),
            Cell::new(detail),
        ]);
    }
    t
}

/// 2026-09-26: The headline tiles. `Collapsed`, `Unmeasured` and the turn-1 cache
/// tile are Bad when they would fail the verdict; `Jittered` is Warn when
/// present, because the verdict passes it.
pub(super) fn summary(s: &Score) -> Vec<Stat> {
    vec![
        Stat::new("Invariant", format!("{}/{}", s.invariant, s.rounds), "").with_style(
            if s.rounds > 0 && s.invariant == s.rounds {
                CellStyle::Good
            } else {
                CellStyle::Neutral
            },
        ),
        Stat::new("Jittered", s.jittered.to_string(), "").with_style(if s.jittered == 0 {
            CellStyle::Good
        } else {
            CellStyle::Warn
        }),
        Stat::new("Collapsed", s.collapsed.to_string(), "").with_style(if s.collapsed == 0 {
            CellStyle::Good
        } else {
            CellStyle::Bad
        }),
        Stat::new("Unmeasured", s.unmeasured.to_string(), "").with_style(if s.unmeasured == 0 {
            CellStyle::Good
        } else {
            CellStyle::Bad
        }),
        // 2026-09-26: The vacuity tile: zero, or no figure, is Bad.
        Stat::new(
            "Min t1 cache",
            s.min_turn1_cached.unwrap_or(0).to_string(),
            "tok",
        )
        .with_style(if s.min_turn1_cached.unwrap_or(0) > 0 {
            CellStyle::Good
        } else {
            CellStyle::Bad
        }),
    ]
}

/// 2026-09-26: Raw gate numbers for the record. Every verdict class is a key even
/// when zero.
pub(super) fn metrics(s: &Score) -> BTreeMap<String, f64> {
    let mut m: BTreeMap<String, f64> = [
        ("rounds", s.rounds),
        ("invariant", s.invariant),
        ("jittered", s.jittered),
        ("collapsed", s.collapsed),
        ("unmeasured", s.unmeasured),
    ]
    .into_iter()
    .map(|(k, v)| (k.to_string(), v as f64))
    .collect();
    // 2026-09-26: The vacuity metric; the gate's BENCH.toml entry floors it at
    // `min = 900.0`. It is written only when a replay reported a figure: with
    // no figure the key is absent, and the gate reports it missing from the
    // record rather than as a zero. `concurrency.rs` omits the same key the
    // same way.
    if let Some(min) = s.min_turn1_cached {
        m.insert("min_cached_prompt_tokens".to_string(), min as f64);
    }
    m
}

#[cfg(test)]
#[path = "report_tests.rs"]
mod report_tests;
