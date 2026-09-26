// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-26: Aggregation and presentation for the MLPerf agentic leg.
//!
//! Unlike `bfcl::report`, which holds MLPerf floors and verdicts pass or fail
//! against them, this leg has no floors: no record of it is committed, and a
//! floor written without one would be invented. The verdict is always info,
//! and the MLPerf full-dataset thresholds appear only in its text.
//!
//! Owner: bench, mlperf_agentic.
//! Invariants: `aggregate` returns `None` when no turn was scored, never a
//! zero score.

use std::collections::BTreeMap;

use crate::result::{Cell, CellStyle, Column, ResultTable, Stat, Verdict};

use super::scoring::Domain;
use super::{MlperfAgentic, TurnRecord};

/// 2026-09-26: Aggregated inline scores: mean over scored turns, per-domain
/// sub-scores, failed requests at 0 in the denominator.
#[derive(Clone, Debug, Default)]
pub(super) struct Scores {
    /// 2026-09-26: Overall mean over scored turns, between 0 and 1.
    pub inline: f64,
    /// 2026-09-26: (mean, scored-turn count) per domain, present only if any
    /// were scored.
    pub coding: Option<(f64, usize)>,
    pub workflow: Option<(f64, usize)>,
    pub turns_scored: usize,
    /// 2026-09-26: Issued turns with no scorable ground truth, excluded from
    /// the denominator.
    pub turns_excluded: usize,
    /// 2026-09-26: Turns whose request failed. A scorable one is scored 0 and
    /// stays in the denominator; an unscorable one is also in
    /// `turns_excluded`.
    pub turns_missing: usize,
    pub output_tokens: usize,
    /// 2026-09-26: Mean output tokens per turn whose request did not fail.
    pub osl_per_turn_mean: f64,
}

/// 2026-09-26: Fold per-turn records into the run's scores. `None` when
/// nothing was scorable; the caller fails the run rather than report 0.0.
pub(super) fn aggregate(turns: &[TurnRecord]) -> Option<Scores> {
    let mut s = Scores::default();
    let (mut total, mut by_domain) = (0.0f64, BTreeMap::<&str, (f64, usize)>::new());
    let mut turns_with_output = 0usize;
    for t in turns {
        if t.missing {
            s.turns_missing += 1;
        } else {
            turns_with_output += 1;
            s.output_tokens += t.completion_tokens;
        }
        let Some(score) = t.score else {
            s.turns_excluded += 1;
            continue;
        };
        s.turns_scored += 1;
        total += score;
        let key = match t.domain {
            Domain::Coding => "coding",
            Domain::Workflow => "workflow",
        };
        let e = by_domain.entry(key).or_default();
        e.0 += score;
        e.1 += 1;
    }
    if s.turns_scored == 0 {
        return None;
    }
    s.inline = total / s.turns_scored as f64;
    s.coding = by_domain
        .get("coding")
        .map(|(sum, n)| (sum / *n as f64, *n));
    s.workflow = by_domain
        .get("workflow")
        .map(|(sum, n)| (sum / *n as f64, *n));
    s.osl_per_turn_mean = if turns_with_output > 0 {
        s.output_tokens as f64 / turns_with_output as f64
    } else {
        0.0
    };
    Some(s)
}

impl MlperfAgentic {
    pub(super) fn table(&self) -> Option<ResultTable> {
        let s = self.scores.as_ref()?;
        let mut t = ResultTable::new(
            "INLINE ACCURACY BY DOMAIN",
            vec![
                Column::left("Domain", 12),
                Column::right("turns", 8),
                Column::right("score %", 9),
            ],
        );
        for (name, entry) in [("coding", &s.coding), ("workflow", &s.workflow)] {
            if let Some((mean, n)) = entry {
                t.push(vec![
                    Cell::new(name),
                    Cell::new(n.to_string()),
                    Cell::styled(format!("{:.2}", mean * 100.0), CellStyle::Accent),
                ]);
            }
        }
        Some(t)
    }

    pub(super) fn summary(&self) -> Vec<Stat> {
        match &self.scores {
            Some(s) => vec![
                Stat::new("Inline accuracy", format!("{:.2}", s.inline * 100.0), "%")
                    .with_style(CellStyle::Accent),
                Stat::new("OSL / turn", format!("{:.1}", s.osl_per_turn_mean), "tok"),
                Stat::new("Turns scored", s.turns_scored.to_string(), ""),
                Stat::new("Missing", s.turns_missing.to_string(), "").with_style(
                    if s.turns_missing == 0 {
                        CellStyle::Good
                    } else {
                        CellStyle::Warn
                    },
                ),
            ],
            None => vec![
                Stat::new(
                    "Turns",
                    format!("{}/{}", self.cursor, self.schedule.len()),
                    "",
                ),
                Stat::new("Trajectories", self.conversations.len().to_string(), ""),
            ],
        }
    }

    pub(super) fn metrics(&self) -> BTreeMap<String, f64> {
        let Some(s) = &self.scores else {
            return BTreeMap::new();
        };
        let mut m = BTreeMap::new();
        m.insert("inline_accuracy".into(), s.inline * 100.0);
        if let Some((mean, _)) = s.coding {
            m.insert("coding_iou".into(), mean * 100.0);
        }
        if let Some((mean, _)) = s.workflow {
            m.insert("workflow_intent_acc".into(), mean * 100.0);
        }
        m.insert("osl_per_turn_mean".into(), s.osl_per_turn_mean);
        m.insert("trajectories".into(), self.conversations.len() as f64);
        m.insert("turns_scored".into(), s.turns_scored as f64);
        m.insert("turns_excluded".into(), s.turns_excluded as f64);
        m.insert("turns_missing".into(), s.turns_missing as f64);
        if let Some(wall) = self.replay_wall {
            m.insert("wall_s".into(), wall.as_secs_f64());
            if wall.as_secs_f64() > 0.0 {
                m.insert(
                    "output_tok_s".into(),
                    s.output_tokens as f64 / wall.as_secs_f64(),
                );
            }
        }
        m
    }

    /// 2026-09-26: Always info, never a pass (module doc). `gate/check.rs`
    /// refuses a record whose run verdict is not PASS, so this leg cannot be a
    /// required gate with this verdict.
    pub(super) fn verdict(&self) -> Verdict {
        let Some(s) = &self.scores else {
            return Verdict::info("not scored");
        };
        let fmt = |e: &Option<(f64, usize)>| match e {
            Some((mean, n)) => format!("{:.2}% (n={n})", mean * 100.0),
            None => "—".to_string(),
        };
        Verdict::info(format!(
            "inline {:.2}% · coding {} · workflow {} · OSL/turn {:.1} · UNMEASURED LEG: no \
             committed baseline exists; MLPerf's 55.86% / 355–434 OSL are full-dataset \
             temp-1.0 thresholds and are NOT this draw's floors. The first measured run on \
             main becomes the baseline.",
            s.inline * 100.0,
            fmt(&s.coding),
            fmt(&s.workflow),
            s.osl_per_turn_mean,
        ))
    }
}
