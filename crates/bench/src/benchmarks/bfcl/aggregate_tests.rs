// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-26: Tests that the aggregator reproduces `score.py`'s weighting and
//! gives the same result however the rows are split across shards; also the
//! shard arithmetic in `draw`, sharded loading in `dataset`, and the tally round
//! trip through a record's metrics.
//!
//! Owner: bench, BFCL benchmark.
//! Invariants: none beyond the types.

use super::aggregate::{Tally, aggregate, union};
use std::collections::BTreeMap;

fn t(hits: u64, n: u64) -> Tally {
    Tally { hits, n }
}

fn map(pairs: &[(&str, u64, u64)]) -> BTreeMap<String, Tally> {
    pairs
        .iter()
        .map(|(k, h, n)| ((*k).to_string(), t(*h, *n)))
        .collect()
}

/// 2026-09-26: The per-subset tallies of the 12-sample fixture that a
/// `provision.rs` test runs through the real `score.py`, which scores it
/// non_live 25.0, live 75.0, hallucination 50.0, normalized 50.0 and overall
/// 66.67. Normalized and overall differ on the same data, so only the right
/// weighting reproduces both.
fn reference() -> BTreeMap<String, Tally> {
    map(&[
        ("simple_python", 2, 2),
        ("simple_java", 0, 1),
        ("multiple", 0, 1),
        ("live_simple", 3, 3),
        ("live_multiple", 0, 1),
        ("irrelevance", 3, 3),
        ("live_irrelevance", 0, 1),
    ])
}

#[test]
fn the_aggregator_reproduces_the_reference_scores() {
    let a = aggregate(&reference());
    assert_eq!(a.total_samples, 12);
    assert_eq!(a.category_scores.get("non_live"), Some(&25.0), "{a:?}");
    assert_eq!(a.category_scores.get("live"), Some(&75.0), "{a:?}");
    assert_eq!(a.category_scores.get("hallucination"), Some(&50.0), "{a:?}");
    assert_eq!(a.normalized_single_turn_score, 50.0, "{a:?}");
    assert_eq!(a.overall_accuracy, 66.67, "{a:?}");
}

/// 2026-09-26: Any partition of the same rows gives the identical aggregate:
/// an even split, a lopsided one and a four-way one.
#[test]
fn any_partition_of_the_same_rows_gives_the_identical_aggregate() {
    let whole = aggregate(&reference());

    let even = union(&[
        map(&[
            ("simple_python", 1, 1),
            ("live_simple", 2, 2),
            ("irrelevance", 2, 2),
        ]),
        map(&[
            ("simple_python", 1, 1),
            ("simple_java", 0, 1),
            ("multiple", 0, 1),
            ("live_simple", 1, 1),
            ("live_multiple", 0, 1),
            ("irrelevance", 1, 1),
            ("live_irrelevance", 0, 1),
        ]),
    ]);
    assert_eq!(aggregate(&even), whole, "even split diverged");

    // 2026-09-26: Lopsided: one shard holds a single sample, and several
    // subsets are absent from it, which reweights a category if scores rather
    // than counts are averaged.
    let lopsided = union(&[
        map(&[("simple_python", 1, 1)]),
        map(&[
            ("simple_python", 1, 1),
            ("simple_java", 0, 1),
            ("multiple", 0, 1),
            ("live_simple", 3, 3),
            ("live_multiple", 0, 1),
            ("irrelevance", 3, 3),
            ("live_irrelevance", 0, 1),
        ]),
    ]);
    assert_eq!(aggregate(&lopsided), whole, "lopsided split diverged");

    let four = union(&[
        map(&[("simple_python", 1, 1), ("live_simple", 1, 1)]),
        map(&[("simple_python", 1, 1), ("live_simple", 1, 1)]),
        map(&[
            ("simple_java", 0, 1),
            ("live_simple", 1, 1),
            ("irrelevance", 2, 2),
        ]),
        map(&[
            ("multiple", 0, 1),
            ("live_multiple", 0, 1),
            ("irrelevance", 1, 1),
            ("live_irrelevance", 0, 1),
        ]),
    ]);
    assert_eq!(aggregate(&four), whole, "four-way split diverged");
}

/// 2026-09-26: Averaging the shard scores rather than the counts gives a
/// different answer on this data, so the partition test above is not passing
/// because the data is insensitive to weighting.
#[test]
fn averaging_shard_scores_would_have_been_wrong() {
    let a = map(&[("simple_python", 1, 1)]);
    let b = map(&[
        ("simple_python", 1, 1),
        ("simple_java", 0, 1),
        ("multiple", 0, 1),
        ("live_simple", 3, 3),
        ("live_multiple", 0, 1),
        ("irrelevance", 3, 3),
        ("live_irrelevance", 0, 1),
    ]);
    let correct = aggregate(&union(&[a.clone(), b.clone()]));
    let naive_normalized = (aggregate(&a).normalized_single_turn_score
        + aggregate(&b).normalized_single_turn_score)
        / 2.0;
    assert_ne!(
        naive_normalized, correct.normalized_single_turn_score,
        "if these matched, this fixture could not detect a wrong aggregation"
    );
}

/// 2026-09-26: A category with no samples drops out of normalized's divisor,
/// as `score.py`'s `if not present: continue` does, rather than scoring zero.
#[test]
fn an_absent_category_drops_out_rather_than_scoring_zero() {
    let no_hallucination = map(&[("simple_python", 1, 2), ("live_simple", 1, 2)]);
    let a = aggregate(&no_hallucination);
    assert!(
        !a.category_scores.contains_key("hallucination"),
        "{:?}",
        a.category_scores
    );
    // 2026-09-26: The mean of the two present categories (50, 50), not of three
    // including a 0.
    assert_eq!(a.normalized_single_turn_score, 50.0, "{a:?}");
}

/// 2026-09-26: The three `simple_*` subsets collapse to one term inside
/// non_live, so a subset with 1 sample weighs as much as one with 100.
#[test]
fn the_three_simple_subsets_collapse_to_a_single_non_live_term() {
    // 2026-09-26: simple_python 0/100, simple_java 1/1, simple_javascript 1/1
    // give a simple term of mean(0, 1, 1) = 0.667, not 2/102.
    let m = map(&[
        ("simple_python", 0, 100),
        ("simple_java", 1, 1),
        ("simple_javascript", 1, 1),
    ]);
    let a = aggregate(&m);
    let nl = a.category_scores["non_live"];
    assert!(
        (nl - 66.67).abs() < 0.01,
        "expected the unweighted collapse (66.67), got {nl}"
    );
}

/// 2026-09-26: live is sample-weighted (the flat mean over live samples);
/// hallucination is not.
#[test]
fn live_is_sample_weighted_and_hallucination_is_not() {
    let live = map(&[("live_simple", 0, 100), ("live_parallel", 1, 1)]);
    let a = aggregate(&live);
    assert!(
        (a.category_scores["live"] - 0.99).abs() < 0.01,
        "sample-weighted: {a:?}"
    );

    let hall = map(&[("irrelevance", 0, 100), ("live_irrelevance", 1, 1)]);
    let b = aggregate(&hall);
    assert_eq!(
        b.category_scores["hallucination"], 50.0,
        "unweighted: {b:?}"
    );
}

#[test]
fn an_empty_tally_set_is_all_zero_rather_than_a_panic() {
    let a = aggregate(&BTreeMap::new());
    assert_eq!(a.total_samples, 0);
    assert_eq!(a.overall_accuracy, 0.0);
    assert_eq!(a.normalized_single_turn_score, 0.0);
}

use super::draw;

/// 2026-09-26: Four shards together take exactly the draw's count for every
/// subset of both pinned draws.
#[test]
fn the_four_shards_partition_every_pinned_draw_exactly() {
    for (name, spec) in [
        ("golden", draw::DrawSpec::golden()),
        ("echolp", draw::DrawSpec::echolp()),
    ] {
        let totals = draw::reference_subset_totals();
        let plan = draw::plan(&spec, &totals);
        let whole: usize = draw::total(&plan);
        let mut summed = 0usize;
        for (subset, take) in &plan {
            let parts: usize = (0..4).map(|k| draw::shard_take(*take, k, 4)).sum();
            assert_eq!(
                parts, *take,
                "{name}/{subset}: shards sum to {parts}, draw says {take}"
            );
            summed += parts;
        }
        assert_eq!(summed, whole, "{name}: total drifted");
    }
}

/// 2026-09-26: Every drawn row belongs to exactly one shard. The count test
/// above would pass even if two shards claimed the same row and a third claimed
/// none.
#[test]
fn every_row_belongs_to_exactly_one_shard() {
    for take in [0usize, 1, 3, 4, 5, 16, 31, 248, 1004] {
        for row in 0..take {
            let owners: Vec<usize> = (0..4).filter(|k| draw::shard_owns(row, *k, 4)).collect();
            assert_eq!(owners.len(), 1, "row {row} of {take} owned by {owners:?}");
        }
    }
}

/// 2026-09-26: A subset smaller than the shard count still lands in some shard.
/// A subset of 3 leaves one of four shards without rows of it, which is fine.
#[test]
fn a_subset_smaller_than_the_shard_count_still_lands_somewhere() {
    for take in 1..=4usize {
        let parts: Vec<usize> = (0..4).map(|k| draw::shard_take(take, k, 4)).collect();
        assert_eq!(parts.iter().sum::<usize>(), take, "take={take} {parts:?}");
        assert!(
            parts.iter().any(|n| *n > 0),
            "take={take} vanished entirely"
        );
    }
}

/// 2026-09-26: Shard sizes differ by at most one row.
#[test]
fn the_shards_are_balanced_to_within_one_row() {
    for take in [1usize, 7, 16, 31, 92, 248, 995, 1004] {
        let parts: Vec<usize> = (0..4).map(|k| draw::shard_take(take, k, 4)).collect();
        let (lo, hi) = (
            *parts.iter().min().expect("4 shards"),
            *parts.iter().max().expect("4 shards"),
        );
        assert!(hi - lo <= 1, "take={take} unbalanced: {parts:?}");
    }
}

#[test]
fn a_shard_index_outside_the_count_takes_nothing() {
    assert_eq!(draw::shard_take(100, 4, 4), 0);
    assert_eq!(draw::shard_take(100, 0, 0), 0);
}

use super::dataset::{self, Shard};

/// 2026-09-26: Local scratch dir, removed on drop; `gate::tests::tempdir` is
/// visible only inside `gate`.
struct ScratchDir(std::path::PathBuf);
impl ScratchDir {
    fn new() -> Self {
        let n = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_nanos())
            .unwrap_or_default();
        let p = std::env::temp_dir().join(format!("metrale-bfcl-shard-{n}"));
        std::fs::create_dir_all(&p).expect("scratch");
        Self(p)
    }
    fn path(&self) -> &std::path::Path {
        &self.0
    }
}
impl Drop for ScratchDir {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

/// 2026-09-26: Builds a dataset file with the given per-subset row counts.
fn fixture(counts: &[(&str, usize)]) -> (ScratchDir, std::path::PathBuf) {
    let d = ScratchDir::new();
    let p = d.path().join("dataset.jsonl");
    let mut s = String::new();
    for (subset, n) in counts {
        for i in 0..*n {
            s.push_str(&format!(
                r#"{{"sample_id":"{subset}-{i}","subset":"{subset}","messages":[],"tools":[],"tool_choice":"auto","ground_truth":"[]","func_description":"[]"}}"#
            ));
            s.push('\n');
        }
    }
    std::fs::write(&p, s).expect("write fixture");
    (d, p)
}

/// 2026-09-26: The four shards of a draw, loaded from the same file, partition
/// the unsharded draw: the same sample_ids, none twice. The arithmetic tests
/// above check counts; this checks the rows the loader selects.
#[test]
fn the_four_shards_of_a_load_partition_the_unsharded_draw() {
    let (_d, path) = fixture(&[
        ("simple_python", 40),
        ("live_simple", 26),
        ("live_parallel", 16),
        ("irrelevance", 24),
    ]);
    let spec = draw::DrawSpec::golden();

    let whole = dataset::load(&path, &spec).expect("whole draw");
    let mut union: Vec<String> = Vec::new();
    for k in 0..4 {
        let part =
            dataset::load_shard(&path, &spec, Some(Shard { index: k, count: 4 })).expect("shard");
        union.extend(part.iter().map(|s| s.sample_id.clone()));
    }

    let mut whole_ids: Vec<String> = whole.iter().map(|s| s.sample_id.clone()).collect();
    whole_ids.sort();
    let mut union_sorted = union.clone();
    union_sorted.sort();

    assert_eq!(
        union_sorted.len(),
        union.len(),
        "a sample_id appeared in two shards"
    );
    union_sorted.dedup();
    assert_eq!(
        union_sorted, whole_ids,
        "the union of the four shards is not the unsharded draw"
    );
}

/// 2026-09-26: One of four shards is about a quarter of the draw, not all of
/// it.
#[test]
fn one_shard_is_not_the_whole_draw() {
    let (_d, path) = fixture(&[("simple_python", 40), ("live_simple", 26)]);
    let spec = draw::DrawSpec::golden();
    let whole = dataset::load(&path, &spec).expect("whole").len();
    let one = dataset::load_shard(&path, &spec, Some(Shard { index: 0, count: 4 }))
        .expect("shard")
        .len();
    assert!(one < whole, "shard {one} vs whole {whole}");
    assert!(
        one * 4 >= whole && one * 4 <= whole + 4,
        "shard {one} is not ~a quarter of {whole}"
    );
}

/// 2026-09-26: `load_shard` with no shard selects the same rows, in the same
/// order, as `load`.
#[test]
fn an_absent_shard_loads_the_whole_draw() {
    let (_d, path) = fixture(&[("simple_python", 40), ("irrelevance", 24)]);
    let spec = draw::DrawSpec::golden();
    let a = dataset::load(&path, &spec).expect("load");
    let b = dataset::load_shard(&path, &spec, None).expect("load_shard(None)");
    assert_eq!(
        a.iter().map(|s| &s.sample_id).collect::<Vec<_>>(),
        b.iter().map(|s| &s.sample_id).collect::<Vec<_>>()
    );
}

use super::aggregate::tallies_from_metrics;

/// 2026-09-26: Tallies written as `report.rs` writes them read back unchanged.
#[test]
fn tallies_round_trip_through_the_metrics_map() {
    let original = reference();
    let mut metrics: BTreeMap<String, f64> = BTreeMap::new();
    for (subset, t) in &original {
        metrics.insert(format!("subset.{subset}.hits"), t.hits as f64);
        metrics.insert(format!("subset.{subset}.n"), t.n as f64);
    }
    metrics.insert("overall_accuracy".into(), 66.67);
    metrics.insert("samples".into(), 12.0);
    assert_eq!(tallies_from_metrics(&metrics), Some(original));
}

/// 2026-09-26: A record with no tallies is `None`, not an empty contribution.
#[test]
fn a_record_without_tallies_is_none_not_empty() {
    let mut m: BTreeMap<String, f64> = BTreeMap::new();
    m.insert("overall_accuracy".into(), 86.55);
    m.insert("samples".into(), 1004.0);
    assert_eq!(tallies_from_metrics(&m), None);
}

/// 2026-09-26: Hits without an `n` is a malformed record, not a 0/0 subset.
#[test]
fn hits_without_n_is_refused() {
    let mut m: BTreeMap<String, f64> = BTreeMap::new();
    m.insert("subset.simple_python.hits".into(), 5.0);
    assert_eq!(tallies_from_metrics(&m), None);
}

/// 2026-09-26: Metrics other than `subset.<name>.hits|n` are not tallies.
#[test]
fn other_metrics_are_ignored() {
    let mut m: BTreeMap<String, f64> = BTreeMap::new();
    m.insert("subset.simple_python.hits".into(), 2.0);
    m.insert("subset.simple_python.n".into(), 2.0);
    m.insert("subsets_considered".into(), 13.0);
    m.insert("subset_scores".into(), 1.0);
    let got = tallies_from_metrics(&m).expect("tallies");
    assert_eq!(got.len(), 1, "{got:?}");
}
