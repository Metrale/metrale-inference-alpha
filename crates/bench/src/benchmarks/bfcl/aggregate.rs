// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-26: Combining BFCL shard results into the score the gate judges.
//!
//! Owner: bench, BFCL benchmark.
//! Invariants: `aggregate` depends only on each subset's summed `(hits, n)`,
//! so every partition of the same rows gives the same result through `union`.
//!
//! # Why shard scores cannot simply be averaged
//!
//! `score.py` does not compute a flat mean. It aggregates hierarchically, and
//! the weights are the whole difficulty:
//!
//! ```text
//! non_live      = mean[ mean(simple_python, simple_java, simple_javascript),
//!                       multiple, parallel, parallel_multiple ]      <- four terms
//! live          = sample-weighted mean over its subsets
//! hallucination = mean(irrelevance, live_irrelevance)                <- unweighted
//! normalized    = mean(the categories present)
//! overall       = flat mean over every scored sample
//! ```
//!
//! So `simple_javascript` (31 rows in the golden draw) weighs the same as
//! `simple_python` (248 rows), a third of one term each, and `irrelevance` (24
//! rows) counts equally with `live_irrelevance` (88). A mean of the shards'
//! `normalized_single_turn_score` is therefore not the whole-set value, and
//! `overall_accuracy` survives only a sample-count-weighted mean, to the 2
//! decimal places each shard's JSON is rounded to.
//!
//! `score.py` also builds each category from the subsets present, so a subset
//! missing from a shard changes that category's divisor. `live_parallel` is 16
//! rows in the golden draw; a quarter of it is four.
//!
//! # What this does instead
//!
//! Shards report per-subset `(hits, n)` integer counts (`subset_totals`); every
//! sample scores exactly 0.0 or 1.0 in `score.py`, so nothing is lost. Summing
//! them across shards and aggregating once applies the weighting to the union,
//! never to a partial view.
//!
//! The arithmetic below is a second implementation of `score.py`'s. A test in
//! `provision.rs` runs the real `score.py` on a 12-sample fixture and checks
//! `aggregate` against its output.

use std::collections::BTreeMap;

/// 2026-09-26: `score.py`'s `non_live` subsets, in its order.
pub const NON_LIVE: [&str; 6] = [
    "simple_python",
    "simple_java",
    "simple_javascript",
    "multiple",
    "parallel",
    "parallel_multiple",
];
/// 2026-09-26: `score.py`'s `live` subsets.
pub const LIVE: [&str; 4] = [
    "live_simple",
    "live_multiple",
    "live_parallel",
    "live_parallel_multiple",
];
/// 2026-09-26: `score.py`'s `hallucination` subsets.
pub const HALLUCINATION: [&str; 2] = ["irrelevance", "live_irrelevance"];
/// 2026-09-26: The three `simple_*` subsets collapse to one term inside
/// `non_live`.
pub const SIMPLE_AST: [&str; 3] = ["simple_python", "simple_java", "simple_javascript"];

/// 2026-09-26: Hits and sample count for one subset.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Tally {
    /// 2026-09-26: Samples scored 1.0.
    pub hits: u64,
    /// 2026-09-26: Samples scored at all.
    pub n: u64,
}

impl Tally {
    fn mean(self) -> Option<f64> {
        (self.n > 0).then(|| self.hits as f64 / self.n as f64)
    }
}

/// 2026-09-26: The numbers the gate judges.
#[derive(Clone, Debug, PartialEq)]
pub struct Aggregate {
    /// 2026-09-26: Flat mean over every scored sample, x100, rounded to 2dp.
    pub overall_accuracy: f64,
    /// 2026-09-26: Unweighted mean of the categories present, x100, rounded to
    /// 2dp.
    pub normalized_single_turn_score: f64,
    /// 2026-09-26: Per category present, x100, rounded to 2dp.
    pub category_scores: BTreeMap<String, f64>,
    /// 2026-09-26: Total samples scored.
    pub total_samples: u64,
}

fn round2(x: f64) -> f64 {
    (x * 100.0).round() / 100.0
}

fn mean(v: &[f64]) -> f64 {
    if v.is_empty() {
        0.0
    } else {
        v.iter().sum::<f64>() / v.len() as f64
    }
}

/// 2026-09-26: Aggregates per-subset tallies with `score.py`'s weighting. Given
/// the union of every shard's tallies it scores the whole draw; given one
/// run's tallies it scores that run, which is how the `provision.rs` test
/// compares it with `score.py`.
pub fn aggregate(tallies: &BTreeMap<String, Tally>) -> Aggregate {
    let subset_mean: BTreeMap<&str, f64> = tallies
        .iter()
        .filter_map(|(k, t)| t.mean().map(|m| (k.as_str(), m)))
        .collect();

    let mut category_scores_raw: BTreeMap<String, f64> = BTreeMap::new();

    // 2026-09-26: hallucination: unweighted mean over the subsets present.
    let present: Vec<f64> = HALLUCINATION
        .iter()
        .filter_map(|s| subset_mean.get(s).copied())
        .collect();
    if !present.is_empty() {
        category_scores_raw.insert("hallucination".into(), mean(&present));
    }

    // 2026-09-26: live: sample-weighted, i.e. the flat mean over live samples.
    let live: Vec<(&str, f64)> = LIVE
        .iter()
        .filter_map(|s| subset_mean.get(s).map(|m| (*s, *m)))
        .collect();
    if !live.is_empty() {
        let total: u64 = live.iter().map(|(s, _)| tallies[*s].n).sum();
        if total > 0 {
            let num: f64 = live
                .iter()
                .map(|(s, m)| m * tallies[*s].n as f64)
                .sum::<f64>();
            category_scores_raw.insert("live".into(), num / total as f64);
        }
    }

    // 2026-09-26: non_live: the simple_* subsets present collapse to one term,
    // then an unweighted mean over that term plus each other subset present.
    let non_live_present: Vec<&str> = NON_LIVE
        .iter()
        .copied()
        .filter(|s| subset_mean.contains_key(s))
        .collect();
    if !non_live_present.is_empty() {
        let simple: Vec<f64> = SIMPLE_AST
            .iter()
            .filter_map(|s| subset_mean.get(s).copied())
            .collect();
        let mut top: Vec<f64> = Vec::new();
        if !simple.is_empty() {
            top.push(mean(&simple));
        }
        for s in &non_live_present {
            if !SIMPLE_AST.contains(s) {
                top.push(subset_mean[s]);
            }
        }
        if !top.is_empty() {
            category_scores_raw.insert("non_live".into(), mean(&top));
        }
    }

    let normalized = if category_scores_raw.is_empty() {
        0.0
    } else {
        mean(&category_scores_raw.values().copied().collect::<Vec<_>>())
    };

    let hits: u64 = tallies.values().map(|t| t.hits).sum();
    let n: u64 = tallies.values().map(|t| t.n).sum();
    let overall = if n > 0 { hits as f64 / n as f64 } else { 0.0 };

    Aggregate {
        overall_accuracy: round2(overall * 100.0),
        normalized_single_turn_score: round2(normalized * 100.0),
        category_scores: category_scores_raw
            .into_iter()
            .map(|(k, v)| (k, round2(v * 100.0)))
            .collect(),
        total_samples: n,
    }
}

/// 2026-09-26: Sums shard tallies per subset. A missing or duplicated shard is
/// not detected here.
pub fn union(shards: &[BTreeMap<String, Tally>]) -> BTreeMap<String, Tally> {
    let mut out: BTreeMap<String, Tally> = BTreeMap::new();
    for shard in shards {
        for (k, t) in shard {
            let e = out.entry(k.clone()).or_default();
            e.hits += t.hits;
            e.n += t.n;
        }
    }
    out
}

/// 2026-09-26: Recovers per-subset tallies from a gate record's metrics, the
/// `subset.<name>.hits` and `subset.<name>.n` keys that `report.rs` writes.
/// `None` when the record has no tallies, or a subset has hits but no `n`. A
/// caller treats `None` as "cannot be aggregated", never as an empty
/// contribution, which would score the group over fewer samples than the draw.
pub fn tallies_from_metrics(metrics: &BTreeMap<String, f64>) -> Option<BTreeMap<String, Tally>> {
    let mut out: BTreeMap<String, Tally> = BTreeMap::new();
    for (k, v) in metrics {
        let Some(rest) = k.strip_prefix("subset.") else {
            continue;
        };
        if let Some(name) = rest.strip_suffix(".hits") {
            out.entry(name.to_string()).or_default().hits = *v as u64;
        } else if let Some(name) = rest.strip_suffix(".n") {
            out.entry(name.to_string()).or_default().n = *v as u64;
        }
    }
    if out.is_empty() {
        return None;
    }
    // 2026-09-26: Hits with no `n` is malformed; refuse rather than divide by
    // zero.
    if out.values().any(|t| t.n == 0 && t.hits > 0) {
        return None;
    }
    Some(out)
}
