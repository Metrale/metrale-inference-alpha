// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-26: Presentation of a finished `agentic-webserver` tier: the
//! per-iteration table and the summary tiles. Nothing here decides the verdict.
//!
//! Owner: bench, agentic.
//! Invariants: the s/turn and Σ wall tiles colour the same aggregates, against
//! the same budgets, that `verdict::verdict` tests.

use super::AgenticWebserver;
use crate::result::{Cell, CellStyle, Column, ResultTable, Stat};

impl AgenticWebserver {
    pub(super) fn table(&self) -> ResultTable {
        let mut t = ResultTable::new(
            "ITERATIONS",
            vec![
                Column::right("Run", 4),
                Column::right("wall s", 8),
                Column::left("ws_ok", 6),
                Column::right("steps", 6),
                Column::right("turns", 6),
                Column::right("tools", 6),
                Column::left("note", 40),
            ],
        );
        for r in &self.rows {
            t.push(vec![
                Cell::new(r.index.to_string()),
                Cell::new(format!("{:.1}", r.wall_s)),
                Cell::styled(
                    if r.webserver_ok { "pass" } else { "FAIL" },
                    if r.webserver_ok {
                        CellStyle::Good
                    } else {
                        CellStyle::Bad
                    },
                ),
                Cell::styled(
                    format!("{}/6", r.directions.met()),
                    if r.directions.overall() {
                        CellStyle::Good
                    } else {
                        CellStyle::Warn
                    },
                ),
                Cell::new(r.turns.to_string()),
                Cell::new(r.tool_calls.to_string()),
                Cell::styled(r.note.clone(), CellStyle::Dim),
            ]);
        }
        t
    }

    pub(super) fn summary(&self) -> Vec<Stat> {
        let ok = self.rows.iter().filter(|r| r.webserver_ok).count();
        let fd = self.rows.iter().filter(|r| r.directions.overall()).count();
        let n = self.rows.len();
        vec![
            Stat::new("webserver_ok", format!("{ok}/{n}"), "").with_style(if n > 0 && ok == n {
                CellStyle::Good
            } else {
                CellStyle::Warn
            }),
            Stat::new("followed_directions", format!("{fd}/{n}"), "").with_style(
                if n > 0 && fd == n {
                    CellStyle::Good
                } else {
                    CellStyle::Warn
                },
            ),
            Stat::new(
                "s/turn",
                self.seconds_per_turn()
                    .map_or_else(|| "n/a".to_string(), |s| format!("{s:.3}")),
                "s",
            )
            .with_style(match self.seconds_per_turn() {
                // 2026-09-26: No turns means the agent never ran, so there is
                // no speed to show as good.
                None => CellStyle::Warn,
                // 2026-09-26: A budget of 0.0 is non-gating, so the tile is
                // neutral.
                Some(_) if self.s_per_turn_budget <= 0.0 => CellStyle::Neutral,
                Some(s) if s <= self.s_per_turn_budget => CellStyle::Good,
                Some(_) => CellStyle::Warn,
            }),
            Stat::new("Σ wall", format!("{:.0}", self.total_wall()), "s").with_style(
                if self.total_wall() <= self.wall_budget_s {
                    CellStyle::Good
                } else {
                    CellStyle::Warn
                },
            ),
        ]
    }
}
