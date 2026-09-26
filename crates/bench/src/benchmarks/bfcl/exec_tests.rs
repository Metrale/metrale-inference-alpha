// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-26: Tests that each BFCL run writes its per-sample output to its own
//! file.
//!
//! Owner: bench, BFCL benchmark.
//! Invariants: none beyond the types.

use super::responses_file;
use crate::benchmarks::bfcl::dataset::Shard;

fn shard(index: usize) -> Option<Shard> {
    Some(Shard { index, count: 4 })
}

/// 2026-09-26: Every shard of `bfcl-subset` reports the id `bfcl-subset`
/// (`Bfcl::descriptor` returns the variant's descriptor), so the test uses one
/// id and relies on the shard alone to separate the whole draw and four shards.
#[test]
fn every_leg_of_a_group_writes_its_own_responses_file() {
    let names: Vec<String> = std::iter::once(responses_file("bfcl-subset", None))
        .chain((0..4).map(|i| responses_file("bfcl-subset", shard(i))))
        .collect();
    let unique: std::collections::BTreeSet<&String> = names.iter().collect();
    assert_eq!(
        unique.len(),
        5,
        "all five legs share the id `bfcl-subset`; the shard must distinguish them: {names:?}"
    );
}

/// 2026-09-26: The whole draw and a shard of it do not share a file.
#[test]
fn the_group_and_its_shard_are_distinct_files() {
    assert_ne!(
        responses_file("bfcl-subset", None),
        responses_file("bfcl-subset", shard(0))
    );
}

/// 2026-09-26: The name says which run wrote it.
#[test]
fn the_file_name_identifies_the_leg_that_wrote_it() {
    assert_eq!(
        responses_file("bfcl-subset", None),
        "responses-bfcl-subset.jsonl"
    );
    assert_eq!(
        responses_file("bfcl-subset", shard(2)),
        "responses-bfcl-subset-2of4.jsonl"
    );
}

/// 2026-09-26: Two draws with the same shard stay distinct.
#[test]
fn two_groups_do_not_collide() {
    assert_ne!(
        responses_file("bfcl-subset", shard(0)),
        responses_file("bfcl-subset-echolp", shard(0))
    );
}
