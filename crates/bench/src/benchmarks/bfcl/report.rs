// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-26: How a BFCL run is presented (table, summary tiles, gate
//! metrics) and its run verdict, including the MLPerf-edge floors.
//!
//! Owner: bench, BFCL benchmark.
//! Invariants: none beyond the types.

use std::collections::BTreeMap;

use anyhow::Result;

use super::Bfcl;
use crate::benchmarks::bfcl::draw;
use crate::params::{ParamKind, ParamSpec, ParamValue, ParamValues};
use crate::result::{Cell, CellStyle, Column, ResultTable, Stat, Verdict};

/// 2026-09-26: MLPerf-edge floors for Qwen3.6-27B: the llama.cpp Q4_K_M
/// reference (86.23 / 87.96, recorded in `kernels/gb10/qwen3.6-27b/BENCH.toml`)
/// times 0.97.
pub const MLPERF_FLOOR_OVERALL: f64 = 83.64;
pub const MLPERF_FLOOR_NORMALIZED: f64 = 85.32;

/// 2026-09-26: The only served models whose run verdict is gated on the MLPerf
/// floors (`floor_verdict`). The summary tiles style every model against the
/// floors.
pub const MLPERF_FLOOR_CHECKPOINTS: [&str; 2] = [
    "unsloth/Qwen3.6-27B-NVFP4",
    "centml/Qwen3.6-27B-NVFP4-W4A4-mlpinf",
];

/// 2026-09-26: Compared ignoring ASCII case, so a serve that spells the org
/// differently still matches.
pub fn is_mlperf_submission_checkpoint(model: &str) -> bool {
    MLPERF_FLOOR_CHECKPOINTS
        .iter()
        .any(|c| c.eq_ignore_ascii_case(model))
}

/// 2026-09-26: The run-verdict floors for a model outside
/// [`MLPERF_FLOOR_CHECKPOINTS`]. Under `--pull-request-gate` the server's
/// `apply_threshold_params` fills them from the model's BENCH.toml `min`
/// bounds minus their noise. 0.0 on both means not gating: the run keeps an
/// info verdict.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub(super) struct BaselineMins {
    pub overall: f64,
    pub normalized: f64,
}

impl BaselineMins {
    // 2026-09-26: The 0.0 defaults are the documented off state (see the
    // parameter help), not an implicit bar.
    pub(super) fn specs() -> [ParamSpec; 2] {
        const PCT: ParamKind = ParamKind::Float {
            min: 0.0,
            max: 100.0,
        };
        [
            ParamSpec::new(
                "min_overall",
                "Overall floor",
                "Run-verdict floor on overall_accuracy. 0 disables (a standalone run reports \
                 an info verdict); under --pull-request-gate this is auto-filled from the \
                 selected variant's BENCH.toml `min` bound. Ignored on the MLPerf submission \
                 checkpoints, which keep the MLPerf floor verdict.",
                PCT,
                ParamValue::Float(0.0),
            ),
            ParamSpec::new(
                "min_normalized",
                "Normalized floor",
                "Run-verdict floor on normalized_single_turn_score. 0 disables (a standalone \
                 run reports an info verdict); under --pull-request-gate this is auto-filled \
                 from the selected variant's BENCH.toml `min` bound. Ignored on the MLPerf \
                 submission checkpoints, which keep the MLPerf floor verdict.",
                PCT,
                ParamValue::Float(0.0),
            ),
        ]
    }

    pub(super) fn from_values(values: &ParamValues) -> Result<Self> {
        Ok(Self {
            overall: values.float("min_overall")?,
            normalized: values.float("min_normalized")?,
        })
    }

    fn gating(self) -> bool {
        self.overall > 0.0 || self.normalized > 0.0
    }
}

impl Bfcl {
    pub(super) fn table(&self) -> Option<ResultTable> {
        let scores = self.scores.as_ref()?;
        let mut t = ResultTable::new(
            "PER-SUBSET ACCURACY",
            vec![
                Column::left("Subset", 24),
                Column::left("Category", 14),
                Column::right("accuracy %", 11),
            ],
        );
        for (subset, value) in &scores.subset_scores {
            t.push(vec![
                Cell::new(subset.clone()),
                Cell::styled(
                    draw::category_of(subset).unwrap_or("unscored").to_string(),
                    CellStyle::Dim,
                ),
                Cell::styled(
                    format!("{value:.2}"),
                    match *value {
                        v if v >= 90.0 => CellStyle::Good,
                        v if v >= 60.0 => CellStyle::Neutral,
                        _ => CellStyle::Warn,
                    },
                ),
            ]);
        }
        for (category, value) in &scores.category_scores {
            t.push(vec![
                Cell::styled(format!("▸ {category}"), CellStyle::Accent),
                Cell::styled("category".to_string(), CellStyle::Dim),
                Cell::styled(format!("{value:.2}"), CellStyle::Accent),
            ]);
        }
        Some(t)
    }

    pub(super) fn summary(&self) -> Vec<Stat> {
        match &self.scores {
            Some(s) => vec![
                Stat::new(
                    "Overall accuracy",
                    format!("{:.2}", s.overall_accuracy),
                    "%",
                )
                .with_style(floor_style(s.overall_accuracy, MLPERF_FLOOR_OVERALL)),
                Stat::new(
                    "Normalized single-turn",
                    format!("{:.2}", s.normalized_single_turn_score),
                    "%",
                )
                .with_style(floor_style(
                    s.normalized_single_turn_score,
                    MLPERF_FLOOR_NORMALIZED,
                )),
                Stat::new("Samples", s.total_samples.to_string(), ""),
            ],
            None => vec![
                Stat::new(
                    "Samples",
                    format!("{}/{}", self.cursor, self.samples.len()),
                    "",
                ),
                Stat::new("With tool calls", self.tool_call_samples.to_string(), ""),
            ],
        }
    }

    /// 2026-09-26: The gate metrics, from the same scores the summary tiles
    /// read. Empty until scoring completes.
    pub(super) fn metrics(&self) -> BTreeMap<String, f64> {
        let Some(s) = &self.scores else {
            return BTreeMap::new();
        };
        let mut m = BTreeMap::new();
        m.insert("overall_accuracy".to_string(), s.overall_accuracy);
        m.insert(
            "normalized_single_turn_score".to_string(),
            s.normalized_single_turn_score,
        );
        m.insert("samples".to_string(), s.total_samples as f64);
        // 2026-09-26: Per-subset counts, so a group can aggregate its shards
        // (`aggregate::tallies_from_metrics`). Counts, not scores: a mean of
        // shard scores is not the whole-set value (see `aggregate`).
        // `check_record` looks up only the metrics the baseline entry names.
        for (subset, (hits, n)) in &s.subset_totals {
            m.insert(format!("subset.{subset}.hits"), *hits as f64);
            m.insert(format!("subset.{subset}.n"), *n as f64);
        }
        // 2026-09-26: Always emitted, so 0 is a measurement rather than an
        // absent key.
        m.insert("transport_errors".to_string(), self.transport_errors as f64);
        // 2026-09-26: How many samples from `sensitive::KNOWN_PARTITION_SENSITIVE`
        // this run collected. Always emitted.
        m.insert(
            "known_partition_sensitive".to_string(),
            self.known_sensitive_seen as f64,
        );
        // 2026-09-26: Which shard this run measured, from the run itself.
        // `GateRecord::shard` (gate/record.rs) reads these two keys.
        if let Some(shard) = self.shard {
            m.insert("shard.index".to_string(), shard.index as f64);
            m.insert("shard.count".to_string(), shard.count as f64);
        }
        m
    }

    pub(super) fn verdict(&self) -> Verdict {
        let Some(s) = &self.scores else {
            return Verdict::info("not scored");
        };
        // 2026-09-26: A shard's slice is not judged against floors cut for the
        // whole draw. It reports an info verdict, and `gate::check_group`
        // judges the aggregate of the partition's per-subset counts.
        if let Some(shard) = self.shard {
            return Verdict::info(format!(
                "shard {}/{} — overall {:.2} · normalized {:.2} · n={} on this slice; the \
                 group's verdict is the aggregate over the whole partition",
                shard.index,
                shard.count,
                s.overall_accuracy,
                s.normalized_single_turn_score,
                s.total_samples
            ));
        }
        floor_verdict(self.target_model.as_deref(), s, self.baseline_mins)
    }
}

/// 2026-09-26: The run verdict, scoped by served model. On
/// [`MLPERF_FLOOR_CHECKPOINTS`] the MLPerf floors give PASS or FAIL. Otherwise
/// nonzero `mins` give PASS or FAIL against them, and without them the verdict
/// is info. An unknown model is treated as a non-submission model. Only PASS
/// satisfies `GateRecord::verdict_passes`.
fn floor_verdict(target_model: Option<&str>, s: &super::Scores, mins: BaselineMins) -> Verdict {
    let detail = format!(
        "overall {:.2} (floor {MLPERF_FLOOR_OVERALL}) · normalized {:.2} (floor \
         {MLPERF_FLOOR_NORMALIZED}) · n={}",
        s.overall_accuracy, s.normalized_single_turn_score, s.total_samples
    );
    match target_model {
        Some(m) if is_mlperf_submission_checkpoint(m) => {
            let overall_ok = s.overall_accuracy >= MLPERF_FLOOR_OVERALL;
            let normalized_ok = s.normalized_single_turn_score >= MLPERF_FLOOR_NORMALIZED;
            if overall_ok && normalized_ok {
                Verdict::pass(detail)
            } else {
                Verdict::fail(format!("BELOW THE MLPERF-EDGE FLOOR — {detail}"))
            }
        }
        // 2026-09-26: Raw value against the bar as handed in. When the gate
        // fills the bar, `apply_threshold_params` has already subtracted the
        // bound's noise, so this agrees with `gate::scoring::compare`.
        _ if mins.gating() => {
            let bars = format!(
                "overall {:.2} (baseline min {:.2}) · normalized {:.2} (baseline min {:.2}) \
                 · n={}",
                s.overall_accuracy,
                mins.overall,
                s.normalized_single_turn_score,
                mins.normalized,
                s.total_samples
            );
            if s.overall_accuracy >= mins.overall
                && s.normalized_single_turn_score >= mins.normalized
            {
                Verdict::pass(format!("{bars} — clears this checkpoint's committed bars"))
            } else {
                Verdict::fail(format!("BELOW THE BASELINE THRESHOLDS — {bars}"))
            }
        }
        Some(m) => Verdict::info(format!(
            "{detail} — judged by baseline thresholds: {m} is not an MLPerf submission \
             checkpoint, so the floor does not gate this run (it does not transfer \
             across weights); the floor styling above is a visual reference only"
        )),
        None => Verdict::info(format!(
            "{detail} — judged by baseline thresholds: served model unknown, so the \
             MLPerf floor (defined on the Qwen3.6-27B submission checkpoints) does \
             not gate this run"
        )),
    }
}

fn floor_style(value: f64, floor: f64) -> CellStyle {
    if value >= floor {
        CellStyle::Good
    } else {
        CellStyle::Bad
    }
}
