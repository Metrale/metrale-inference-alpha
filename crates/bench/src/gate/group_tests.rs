// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-26: Tests for `select_partition`: a group is satisfied only by a complete
//! partition of shard records at one commit, at any shard count.
//!
//! Owner: bench gate.
//! Invariants: none beyond the types.

use super::group::{GroupFault, Partition, ShardRecord, select_partition};

/// 2026-09-26: `(index, count, sha, recorded_at)` rows as shard records; each handle is
/// the row's position, as `check_group` assigns them.
fn shards(rows: &[(usize, usize, &str, u64)]) -> Vec<ShardRecord> {
    rows.iter()
        .enumerate()
        .map(|(handle, (index, count, sha, at))| ShardRecord {
            index: *index,
            count: *count,
            git_sha: (*sha).to_string(),
            recorded_at: *at,
            handle,
        })
        .collect()
}

fn handles(p: &Partition) -> Vec<usize> {
    let mut h = p.handles.clone();
    h.sort_unstable();
    h
}

#[test]
fn a_complete_partition_at_one_commit_is_selected_at_any_count() {
    for n in [1usize, 2, 4, 6, 8, 13] {
        let rows: Vec<_> = (0..n).map(|i| (i, n, "abc", 100 + i as u64)).collect();
        let p = select_partition("g", &shards(&rows)).unwrap_or_else(|f| panic!("{n}-way: {f}"));
        assert_eq!(p.count, n);
        assert_eq!(p.git_sha, "abc");
        assert_eq!(handles(&p), (0..n).collect::<Vec<_>>());
    }
}

#[test]
fn order_of_arrival_does_not_matter() {
    let p = select_partition(
        "g",
        &shards(&[
            (3, 4, "abc", 1),
            (0, 4, "abc", 2),
            (2, 4, "abc", 3),
            (1, 4, "abc", 4),
        ]),
    )
    .unwrap();
    assert_eq!(p.count, 4);
    assert_eq!(handles(&p), vec![0, 1, 2, 3]);
}

/// 2026-09-26: A missing index is a fault whose message names the indices held and
/// missing, and says that a subset is a different measurement.
#[test]
fn a_missing_shard_is_refused_and_named() {
    let fault = select_partition(
        "bfcl-subset",
        &shards(&[(0, 4, "abc", 1), (1, 4, "abc", 1), (3, 4, "abc", 1)]),
    )
    .expect_err("three of four");
    match &fault {
        GroupFault::Missing { group, held } => {
            assert_eq!(*group, "bfcl-subset");
            assert_eq!(held, &vec![(4, "abc".to_string(), vec![0, 1, 3])]);
        }
    }
    let msg = fault.to_string();
    assert!(
        msg.contains("4-way at abc holds [0, 1, 3], missing 2"),
        "{msg}"
    );
    assert!(msg.contains("different measurement"), "{msg}");
}

#[test]
fn no_records_at_all_is_a_missing_fault_not_a_pass() {
    let fault = select_partition("g", &[]).expect_err("empty");
    assert!(matches!(&fault, GroupFault::Missing { held, .. } if held.is_empty()));
    assert!(fault.to_string().contains("no shard record"), "{fault}");
}

/// 2026-09-26: Every index is present, but index 2 was measured at another commit, so
/// no complete partition exists at one commit.
#[test]
fn a_partition_is_never_assembled_across_commits() {
    let fault = select_partition(
        "g",
        &shards(&[
            (0, 4, "abc", 1),
            (1, 4, "abc", 1),
            (2, 4, "def", 9),
            (3, 4, "abc", 1),
        ]),
    )
    .expect_err("index 2 is at another commit");
    let GroupFault::Missing { held, .. } = &fault;
    assert_eq!(
        held,
        &vec![
            (4, "abc".to_string(), vec![0, 1, 3]),
            (4, "def".to_string(), vec![2])
        ]
    );
    assert!(fault.to_string().contains("ONE measurement"), "{fault}");
}

/// 2026-09-26: Two records of index 2 and none of index 3: the newest-per-index rule
/// keeps one record of index 2, so the partition stays incomplete.
#[test]
fn a_duplicated_shard_does_not_stand_in_for_a_missing_one() {
    let fault = select_partition(
        "g",
        &shards(&[
            (0, 4, "abc", 1),
            (1, 4, "abc", 1),
            (2, 4, "abc", 1),
            (2, 4, "abc", 2),
        ]),
    )
    .expect_err("C twice, D never");
    let GroupFault::Missing { held, .. } = &fault;
    assert_eq!(held, &vec![(4, "abc".to_string(), vec![0, 1, 2])]);
}

#[test]
fn the_newest_record_per_index_is_the_one_selected() {
    let p = select_partition(
        "g",
        &shards(&[(0, 2, "abc", 1), (1, 2, "abc", 1), (0, 2, "abc", 5)]),
    )
    .unwrap();
    assert_eq!(handles(&p), vec![1, 2]);
}

/// 2026-09-26: A shard of an 8-way split is keyed under count 8, so it cannot complete
/// a 4-way partition.
#[test]
fn a_shard_of_another_count_never_completes_a_partition() {
    let fault = select_partition(
        "g",
        &shards(&[
            (0, 4, "abc", 1),
            (1, 4, "abc", 1),
            (2, 4, "abc", 1),
            (3, 8, "abc", 1),
        ]),
    )
    .expect_err("an 8-way shard does not finish a 4-way");
    let GroupFault::Missing { held, .. } = &fault;
    assert_eq!(held.len(), 2);
    assert!(
        fault.to_string().contains("8-way at abc holds [3]"),
        "{fault}"
    );
}

#[test]
fn of_two_complete_partitions_the_newer_wins() {
    let rows = [(0, 2, "abc", 10), (1, 2, "abc", 11), (0, 1, "abc", 5)];
    assert_eq!(select_partition("g", &shards(&rows)).unwrap().count, 2);
    let rows = [(0, 2, "abc", 10), (1, 2, "abc", 11), (0, 1, "def", 50)];
    let p = select_partition("g", &shards(&rows)).unwrap();
    assert_eq!((p.count, p.git_sha.as_str()), (1, "def"));
}

/// 2026-09-26: Every group is a registered, required benchmark with a `shard`
/// parameter, of the Correctness class: `check_group` sums shard counts, which
/// recombines an accuracy score but not a timing.
#[test]
fn every_group_is_a_registered_required_correctness_benchmark() {
    use crate::hardware::Sensitivity;
    for g in super::group::GROUPS {
        let d = crate::registry::find(g.id)
            .unwrap_or_else(|| panic!("group {} is not a registered benchmark", g.id));
        assert_eq!(d.sensitivity, Sensitivity::Correctness, "group {}", g.id);
        assert!(
            super::coverage::REQUIRED.iter().any(|r| r.id == g.id),
            "group {} is not in REQUIRED — then nothing asks for it",
            g.id
        );
        assert!(
            d.build().parameters().iter().any(|p| p.key == "shard"),
            "group {} has no `shard` parameter: nothing can run a slice of it",
            g.id
        );
    }
}

/// 2026-09-26: A shard runs under its group's id, so these per-shard ids are neither
/// registered benchmarks nor groups.
#[test]
fn legacy_shard_ids_are_not_registered() {
    for id in ["bfcl-subset-a", "bfcl-subset-d", "bfcl-subset-echolp-a"] {
        assert!(
            crate::registry::find(id).is_none(),
            "{id} is still registered"
        );
        assert!(super::group::find(id).is_none(), "{id} is a group?");
    }
}
