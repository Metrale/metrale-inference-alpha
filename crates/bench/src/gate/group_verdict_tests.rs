// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-26: Tests for `check_one` on a group: a complete partition of shard records
//! produces one verdict, anything less produces none, and each shard meets the
//! per-record rules of a plain gate record.
//!
//! Owner: bench gate.
//! Invariants: none beyond the types.

use super::check::check_one;
use super::tests::{bfcl_baseline, tempdir};
use super::*;
use crate::result::Verdict;
use std::collections::BTreeMap;

const SHA: &str = "1111111111";
const G: &str = "bfcl-subset";

/// 2026-09-26: Plant one shard record of `G` with `subset.<name>.hits` / `.n` tallies,
/// the keys `benchmarks/bfcl/report.rs` writes, then apply `overrides`.
fn plant_shard_with(
    root: &std::path::Path,
    shard: (usize, usize),
    sha: &str,
    secs: u64,
    overrides: &[(&str, f64)],
) {
    let mut metrics = BTreeMap::new();
    metrics.insert("overall_accuracy".to_string(), 90.0);
    metrics.insert("subset.simple_python.hits".to_string(), 95.0);
    metrics.insert("subset.simple_python.n".to_string(), 100.0);
    metrics.insert("shard.index".to_string(), shard.0 as f64);
    metrics.insert("shard.count".to_string(), shard.1 as f64);
    metrics.insert("transport_errors".to_string(), 0.0);
    for (k, v) in overrides {
        metrics.insert((*k).to_string(), *v);
    }
    let record = super::tests::run_record(metrics, Verdict::pass("ok"));
    let mut gate = GateRecord::from_run(
        &record,
        super::tests::hw(),
        sha.to_string(),
        Vec::new(),
        None,
    )
    .unwrap();
    gate.benchmark_id = G.to_string();
    gate.verdict = Some("PASS".to_string());
    gate.recorded_at = secs;
    write_record(root, &gate).unwrap();
}

fn plant_shard(root: &std::path::Path, shard: (usize, usize), sha: &str, secs: u64) {
    plant_shard_with(root, shard, sha, secs, &[]);
}

/// 2026-09-26: A complete `n`-way partition at `sha`, recorded one second apart.
fn partition(root: &std::path::Path, n: usize, sha: &str, secs: u64) {
    for i in 0..n {
        plant_shard(root, (i, n), sha, secs + i as u64);
    }
}

fn scaffold() -> tempdir::Dir {
    let dir = tempdir::Dir::new();
    for id in REQUIRED_GATES {
        std::fs::create_dir_all(gate_dir(dir.path(), id)).unwrap();
        super::fixture_baseline::write_baseline(dir.path(), id, &bfcl_baseline());
    }
    dir
}

/// 2026-09-26: Edit the record of shard `index` in place. Callers plant records dated
/// before `SIGNATURE_REQUIRED_AFTER`, so the unsigned edit is still accepted.
fn rewrite_shard(root: &std::path::Path, index: usize, edit: impl FnOnce(&mut GateRecord)) {
    let path = records_newest_first(root, G)
        .into_iter()
        .find(|p| read_record(p).is_ok_and(|r| r.shard().is_some_and(|(i, _)| i == index)))
        .expect("a record for that shard");
    let mut r = read_record(&path).unwrap();
    edit(&mut r);
    std::fs::write(&path, serde_json::to_string_pretty(&r).unwrap()).unwrap();
}

#[test]
fn a_complete_partition_satisfies_the_group_at_any_count() {
    for n in [1usize, 3, 6] {
        let dir = scaffold();
        let root = dir.path();
        partition(root, n, SHA, 1_785_891_000);
        assert!(
            matches!(check_one(root, G, SHA), GateStatus::Pass),
            "{n}-way: {:?}",
            check_one(root, G, SHA)
        );
    }
}

#[test]
fn an_incomplete_partition_does_not_satisfy_the_group() {
    let dir = scaffold();
    let root = dir.path();
    for i in [0, 1, 2, 4] {
        plant_shard(root, (i, 5), SHA, 1_785_891_000 + i as u64);
    }
    match check_one(root, G, SHA) {
        GateStatus::Missing(why) => {
            assert!(
                why.contains("5-way at 1111111111 holds [0, 1, 2, 4], missing 3"),
                "{why}"
            );
            assert!(why.contains("different measurement"), "{why}");
        }
        other => panic!("four of five must not satisfy the group, got {other:?}"),
    }
}

/// 2026-09-26: A passing whole-draw record under the group's id, with no shards, does
/// not satisfy the group, and the verdict counts the whole-draw records it ignored.
#[test]
fn a_whole_draw_record_no_longer_satisfies_the_group() {
    let dir = scaffold();
    let root = dir.path();
    super::tests::plant(root, G, SHA, 1_785_891_382, "PASS");
    match check_one(root, G, SHA) {
        GateStatus::Missing(why) => {
            assert!(why.contains("no shard record"), "{why}");
            assert!(why.contains("1 whole-draw record(s)"), "{why}");
            assert!(why.contains("no longer satisfy"), "{why}");
            assert!(why.contains("2026-09-13"), "{why}");
        }
        other => panic!("a whole-draw record must not satisfy the group, got {other:?}"),
    }
}

#[test]
fn a_group_with_no_records_says_so() {
    let dir = scaffold();
    match check_one(dir.path(), G, SHA) {
        GateStatus::Missing(why) => {
            assert!(why.contains("no shard record"), "{why}");
            assert!(!why.contains("whole-draw"), "{why}");
        }
        other => panic!("expected Missing, got {other:?}"),
    }
}

/// 2026-09-26: When the newest shard record no longer stands, the verdict names the
/// perf-path files that invalidated it.
#[test]
fn a_partition_invalidated_by_a_perf_path_names_the_file() {
    let dir = scaffold();
    let root = dir.path();
    use super::coverage_tests::scratch_repo;
    scratch_repo::init(root);
    scratch_repo::commit(root, "crates/lib.rs", "v1", "kernel v1");
    let old = scratch_repo::head(root);
    partition(root, 4, &old, 1_785_891_000);
    scratch_repo::commit(root, "crates/lib.rs", "v2", "kernel v2");
    let head = scratch_repo::head(root);
    match check_one(root, G, &head) {
        GateStatus::Missing(why) => {
            assert!(why.contains("invalidated by"), "{why}");
            assert!(why.contains("crates/lib.rs"), "{why}");
        }
        other => panic!("expected Missing naming the file, got {other:?}"),
    }
}

/// 2026-09-26: A partition measured at an older commit still certifies a newer one when
/// the diff between them touches no perf path.
#[test]
fn a_standing_partition_at_an_older_commit_still_certifies() {
    let dir = scaffold();
    let root = dir.path();
    use super::coverage_tests::scratch_repo;
    scratch_repo::init(root);
    scratch_repo::commit(root, "crates/lib.rs", "v1", "kernel v1");
    let old = scratch_repo::head(root);
    partition(root, 3, &old, 1_785_891_000);
    scratch_repo::commit(root, "docs/notes.md", "hello", "docs only");
    let head = scratch_repo::head(root);
    assert!(
        matches!(check_one(root, G, &head), GateStatus::Pass),
        "{:?}",
        check_one(root, G, &head)
    );
}

/// 2026-09-26: Negative control: shards recorded after `SIGNATURE_REQUIRED_AFTER`
/// with no `.sig` fail the group, and the verdict names the shard.
#[test]
fn an_unsigned_shard_after_the_cutover_fails_the_group() {
    let dir = scaffold();
    let root = dir.path();
    partition(root, 4, SHA, super::signing::SIGNATURE_REQUIRED_AFTER + 10);
    match check_one(root, G, SHA) {
        GateStatus::Fail(why) => {
            let joined = why.join(" ");
            assert!(joined.contains("bfcl-subset[0/4]: "), "{joined}");
            assert!(joined.contains("no signature"), "{joined}");
        }
        other => panic!("an unsigned shard must fail the group, got {other:?}"),
    }
}

/// 2026-09-26: Negative control: a shard measured from a dirty tree fails the group.
#[test]
fn a_dirty_tree_shard_fails_the_group() {
    let dir = scaffold();
    let root = dir.path();
    partition(root, 4, SHA, 1_785_891_000);
    rewrite_shard(root, 2, |r| {
        r.dirty_paths = vec!["crates/model-layers/src/lib.rs".to_string()];
    });
    match check_one(root, G, SHA) {
        GateStatus::Fail(why) => {
            let joined = why.join(" ");
            assert!(
                joined.contains("bfcl-subset[2/4]: measured from a dirty tree"),
                "{joined}"
            );
            assert!(
                joined.contains("crates/model-layers/src/lib.rs"),
                "{joined}"
            );
        }
        other => panic!("a dirty shard must fail the group, got {other:?}"),
    }
}

/// 2026-09-26: Negative control: a shard whose run failed fails the group with the
/// run's own reason.
#[test]
fn a_failed_frame_shard_fails_the_group() {
    let dir = scaffold();
    let root = dir.path();
    partition(root, 4, SHA, 1_785_891_000);
    rewrite_shard(root, 1, |r| {
        r.frame_status = crate::result::RunStatus::Failed;
        r.verdict_reason = "scorer crashed".to_string();
    });
    match check_one(root, G, SHA) {
        GateStatus::Fail(why) => {
            let joined = why.join(" ");
            assert!(
                joined.contains("bfcl-subset[1/4]: the run itself failed"),
                "{joined}"
            );
            assert!(joined.contains("scorer crashed"), "{joined}");
        }
        other => panic!("a failed shard must fail the group, got {other:?}"),
    }
}

/// 2026-09-26: Negative control: a shard of another checkpoint is not the gate's
/// required subject, so it is skipped and the partition is incomplete.
#[test]
fn a_shard_on_another_variant_is_not_the_subject() {
    let dir = scaffold();
    let root = dir.path();
    partition(root, 4, SHA, 1_785_891_000);
    rewrite_shard(root, 3, |r| {
        r.target_model = "some-other/checkpoint".to_string();
    });
    match check_one(root, G, SHA) {
        GateStatus::Missing(why) => assert!(why.contains("missing 3"), "{why}"),
        other => panic!("an off-subject shard must not count, got {other:?}"),
    }
}

/// 2026-09-26: A shard with no per-subset tallies leaves the group unsatisfied instead
/// of counting as an empty contribution.
#[test]
fn a_shard_without_tallies_is_refused_rather_than_counted_as_empty() {
    let dir = scaffold();
    let root = dir.path();
    partition(root, 4, SHA, 1_785_891_000);
    rewrite_shard(root, 3, |r| {
        r.metrics.retain(|k, _| !k.starts_with("subset."));
    });
    match check_one(root, G, SHA) {
        GateStatus::Missing(why) => assert!(why.contains("no per-subset tallies"), "{why}"),
        other => panic!("expected a refusal, got {other:?}"),
    }
}

/// 2026-09-26: Through `check_one`: four records at one commit, two of them for index 2
/// and none for index 3, leave the partition incomplete.
#[test]
fn two_records_of_the_same_shard_do_not_complete_the_partition() {
    let dir = scaffold();
    let root = dir.path();
    for i in 0..3 {
        plant_shard(root, (i, 4), SHA, 1_785_891_000 + i as u64);
    }
    plant_shard_with(root, (3, 4), SHA, 1_785_891_003, &[("shard.index", 2.0)]);
    match check_one(root, G, SHA) {
        GateStatus::Missing(why) => {
            assert!(
                why.contains("4-way at 1111111111 holds [0, 1, 2], missing 3"),
                "{why}"
            );
        }
        other => panic!("a duplicated shard must not complete the group, got {other:?}"),
    }
}

/// 2026-09-26: Through `check_one`: a shard with a nonzero `transport_errors` metric
/// fails the group, named with its failure count.
#[test]
fn a_shard_degraded_by_transport_failures_is_refused() {
    let dir = scaffold();
    let root = dir.path();
    for i in 0..4 {
        let overrides: &[(&str, f64)] = if i == 1 {
            &[("transport_errors", 7.0)]
        } else {
            &[]
        };
        plant_shard_with(root, (i, 4), SHA, 1_785_891_000 + i as u64, overrides);
    }
    match check_one(root, G, SHA) {
        GateStatus::Fail(why) => {
            let joined = why.join(" ");
            assert!(joined.contains("bfcl-subset[1/4]"), "{joined}");
            assert!(joined.contains("7 transport failures"), "{joined}");
        }
        other => panic!("a degraded shard must fail the group, got {other:?}"),
    }
}

/// 2026-09-26: A shard re-run at a newer commit does not complete a partition there.
/// The scaffold is not a git repository, so the older commit's records cannot be
/// diffed against the head and do not stand; the new record alone is incomplete.
#[test]
fn a_partition_is_never_assembled_across_commits() {
    let dir = scaffold();
    let root = dir.path();
    partition(root, 4, SHA, 1_785_891_000);
    plant_shard(root, (2, 4), "2222222222", 1_785_899_000);
    match check_one(root, G, "2222222222") {
        GateStatus::Missing(why) => {
            assert!(
                why.contains("4-way at 2222222222 holds [2], missing 0,1,3"),
                "{why}"
            );
            assert!(why.contains("ONE measurement"), "{why}");
        }
        other => panic!("expected Missing, got {other:?}"),
    }
}

/// 2026-09-26: `shards_owed` owes nothing when a complete partition stands, the
/// missing indices of a partition begun at this commit, and a present shard that
/// fails the per-record rules (here, a failed run).
#[test]
fn shards_owed_names_exactly_the_shards_the_verdict_would_refuse() {
    let group = super::group::find(G).unwrap();
    let dir = scaffold();
    let root = dir.path();
    assert_eq!(
        shards_owed(root, group, SHA, 6),
        vec![(0, 6), (1, 6), (2, 6), (3, 6), (4, 6), (5, 6)]
    );
    // 2026-09-26: A complete 4-way partition owes nothing, whatever count is wanted.
    partition(root, 4, SHA, 1_785_891_000);
    assert!(shards_owed(root, group, SHA, 6).is_empty());
    assert!(matches!(check_one(root, G, SHA), GateStatus::Pass));
    // 2026-09-26: With index 2 removed, only that index is owed, at the begun count.
    let c = records_newest_first(root, G)
        .into_iter()
        .find(|p| read_record(p).unwrap().shard() == Some((2, 4)))
        .unwrap();
    std::fs::remove_file(&c).unwrap();
    assert_eq!(shards_owed(root, group, SHA, 6), vec![(2, 4)]);
    rewrite_shard(root, 1, |r| {
        r.frame_status = crate::result::RunStatus::Failed;
    });
    assert_eq!(shards_owed(root, group, SHA, 6), vec![(1, 4), (2, 4)]);
    assert!(matches!(check_one(root, G, SHA), GateStatus::Missing(_)));
}

/// 2026-09-26: A partition begun at another commit is not finished here: at a new
/// commit `shards_owed` asks for a fresh partition of the wanted count.
#[test]
fn shards_owed_does_not_try_to_finish_another_commits_partition() {
    let group = super::group::find(G).unwrap();
    let dir = scaffold();
    let root = dir.path();
    for i in 0..3 {
        plant_shard(root, (i, 4), SHA, 1_785_891_000 + i as u64);
    }
    assert_eq!(shards_owed(root, group, SHA, 2), vec![(3, 4)]);
    assert_eq!(
        shards_owed(root, group, "2222222222", 2),
        vec![(0, 2), (1, 2)]
    );
}
