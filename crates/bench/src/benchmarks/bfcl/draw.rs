// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-26: The BFCL sample draw. `normalized_single_turn_score` depends on
//! the category mix, so a score is comparable only with scores on the same
//! draw.
//!
//! Two deterministic rules, with no RNG, produce it:
//!
//! 1. Selection by category. Asking for `[non_live, live, hallucination]`
//!    expands to those categories' subsets; `live_relevance` belongs to none of
//!    them, so it drops out. That exclusion is the difference between n = 1011
//!    and the golden n = 995.
//! 2. Per-subset head(n). `n = total if total <= subset_floor else
//!    max(1, int(total * pct / 100))`, taking the first `n` rows, concatenated
//!    in sorted subset order.
//!
//! Owner: bench, BFCL benchmark.
//! Invariants: `shard_take` over `0..count` sums to `take` for every `count`
//! of at least 1.

use std::collections::BTreeMap;

/// 2026-09-26: Every single-turn subset, in the order `provision.py` writes
/// them.
pub const SINGLE_TURN_SUBSETS: [&str; 13] = [
    "simple_python",
    "simple_java",
    "simple_javascript",
    "multiple",
    "parallel",
    "parallel_multiple",
    "live_simple",
    "live_multiple",
    "live_parallel",
    "live_parallel_multiple",
    "irrelevance",
    "live_irrelevance",
    "live_relevance",
];

/// 2026-09-26: The three scored categories.
pub const CATEGORIES: [&str; 3] = ["non_live", "live", "hallucination"];

/// 2026-09-26: Which category a subset belongs to. `live_relevance` maps to
/// `None`: `score.py` scores it per sample but in no category, and a category
/// selection leaves it out.
pub fn category_of(subset: &str) -> Option<&'static str> {
    match subset {
        "simple_python" | "simple_java" | "simple_javascript" | "multiple" | "parallel"
        | "parallel_multiple" => Some("non_live"),
        "live_simple" | "live_multiple" | "live_parallel" | "live_parallel_multiple" => {
            Some("live")
        }
        "irrelevance" | "live_irrelevance" => Some("hallucination"),
        _ => None,
    }
}

/// 2026-09-26: Sampling configuration.
#[derive(Clone, Debug, PartialEq)]
pub struct DrawSpec {
    /// 2026-09-26: Categories to include. Empty means every single-turn
    /// subset, including the uncategorised `live_relevance`.
    pub categories: Vec<String>,
    /// 2026-09-26: Per-category percentage. A selected subset whose category
    /// has no entry is taken whole.
    pub category_pct: BTreeMap<String, f64>,
    /// 2026-09-26: A subset of at most this many rows is taken in full,
    /// bypassing the percentage, so `live_parallel` (16 rows) is not cut to one
    /// to three samples.
    pub subset_floor: Option<usize>,
}

impl DrawSpec {
    fn categories_or_all(categories: &[&str]) -> Vec<String> {
        categories.iter().map(|s| s.to_string()).collect()
    }

    /// 2026-09-26: The golden MLPerf-edge draw: the three categories at
    /// 62/10/10 with a floor of 25. On `reference_subset_totals` this is 995.
    pub fn golden() -> Self {
        Self {
            categories: Self::categories_or_all(&CATEGORIES),
            category_pct: [
                ("non_live".to_string(), 62.0),
                ("live".to_string(), 10.0),
                ("hallucination".to_string(), 10.0),
            ]
            .into_iter()
            .collect(),
            subset_floor: Some(25),
        }
    }

    /// 2026-09-26: Every sample of the three scored categories, no sampling.
    /// The same categories as `golden`: taking `live_relevance` too would move
    /// `overall_accuracy` on a subset no category scores.
    pub fn full() -> Self {
        Self {
            categories: Self::categories_or_all(&CATEGORIES),
            category_pct: BTreeMap::new(),
            subset_floor: None,
        }
    }

    /// 2026-09-26: The `echolp` draw: the three categories at 46/23/12 with a
    /// floor of 25. On `reference_subset_totals` this is 1004. It weights `live`
    /// more than twice as heavily as `golden` (23 % vs 10 %), so a score from one
    /// draw is not comparable to a threshold from the other; each has its own
    /// baseline.
    pub fn echolp() -> Self {
        Self {
            categories: Self::categories_or_all(&CATEGORIES),
            category_pct: [
                ("non_live".to_string(), 46.0),
                ("live".to_string(), 23.0),
                ("hallucination".to_string(), 12.0),
            ]
            .into_iter()
            .collect(),
            subset_floor: Some(25),
        }
    }

    /// 2026-09-26: Whether this subset is in the selection.
    pub fn includes(&self, subset: &str) -> bool {
        if self.categories.is_empty() {
            return true;
        }
        category_of(subset).is_some_and(|c| self.categories.iter().any(|sel| sel == c))
    }

    /// 2026-09-26: How many rows to keep from a subset holding `total` rows.
    pub fn take_count(&self, subset: &str, total: usize) -> usize {
        if total == 0 || !self.includes(subset) {
            return 0;
        }
        if self.subset_floor.is_some_and(|floor| total <= floor) {
            return total;
        }
        match category_of(subset).and_then(|c| self.category_pct.get(c).copied()) {
            // 2026-09-26: `as usize` truncates a non-negative f64, like the
            // `int()` in the rule above; `max(1)` keeps a subset from vanishing.
            Some(pct) => ((total as f64 * pct / 100.0) as usize).max(1),
            None => total,
        }
    }
}

/// 2026-09-26: Applies the draw to per-subset totals, returning `(subset,
/// take)` sorted by subset name and without subsets that take nothing.
pub fn plan(spec: &DrawSpec, totals: &BTreeMap<String, usize>) -> Vec<(String, usize)> {
    totals
        .iter()
        .map(|(subset, total)| (subset.clone(), spec.take_count(subset, *total)))
        .filter(|(_, take)| *take > 0)
        .collect()
}

/// 2026-09-26: Total sample count for a plan.
pub fn total(plan: &[(String, usize)]) -> usize {
    plan.iter().map(|(_, n)| n).sum()
}

/// 2026-09-26: The BFCL v4 single-turn row counts per subset, the table the
/// draw and shard tests and `Bfcl::expected_samples` use. `provision.rs`
/// saves the counts `provision.py` wrote as `dataset_summary.json`; they are
/// expected to equal these.
pub fn reference_subset_totals() -> BTreeMap<String, usize> {
    [
        ("irrelevance", 240),
        ("live_irrelevance", 884),
        ("live_multiple", 1053),
        ("live_parallel", 16),
        ("live_parallel_multiple", 24),
        ("live_relevance", 16),
        ("live_simple", 258),
        ("multiple", 200),
        ("parallel", 200),
        ("parallel_multiple", 200),
        ("simple_java", 100),
        ("simple_javascript", 50),
        ("simple_python", 400),
    ]
    .into_iter()
    .map(|(k, v)| (k.to_string(), v))
    .collect()
}

/// 2026-09-26: How many of a subset's `take` drawn rows shard `index` of
/// `count` gets; 0 for an index out of range or a count of 0. Rows are striped,
/// row `i` going to shard `i % count` (`shard_owns`), not cut into contiguous
/// blocks, so every shard gets a proportional slice of every subset, and
/// membership depends on position alone.
pub fn shard_take(take: usize, index: usize, count: usize) -> usize {
    if count == 0 || index >= count {
        return 0;
    }
    // 2026-09-26: Rows index..take striding by `count` from `index`:
    // ceil((take - index) / count).
    take.saturating_sub(index).div_ceil(count)
}

/// 2026-09-26: Whether row `row` of a subset (0-based, within the drawn rows)
/// is in shard `index` of `count`.
pub fn shard_owns(row: usize, index: usize, count: usize) -> bool {
    count > 0 && index < count && row % count == index
}

/// 2026-09-26: The per-subset counts one shard runs, without empty subsets.
pub fn shard_plan(plan: &[(String, usize)], index: usize, count: usize) -> Vec<(String, usize)> {
    plan.iter()
        .map(|(s, take)| (s.clone(), shard_take(*take, index, count)))
        .filter(|(_, n)| *n > 0)
        .collect()
}

#[cfg(test)]
#[path = "draw_tests.rs"]
mod tests;
