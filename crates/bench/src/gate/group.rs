// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-26: Benchmark groups: one gate satisfied by a complete partition of shard runs.
//!
//! Owner: bench gate.
//! Invariants:
//! - [`select_partition`] returns only a partition holding one record per distinct
//!   index, `count` of them, all measured at one commit. It relies on each index being
//!   below its count, which `GateRecord::shard` ensures for the records `check_group`
//!   passes.
//!
//! A group id is a gate id: `coverage::REQUIRED`, BENCH.toml and
//! `.github/pr-taxonomy.json` name it, and every shard is that benchmark run with
//! `--param shard=i/n`, filed under the group. `check_one` hands a group id to
//! `check_group`, which skips whole-draw records, judges each shard of the chosen
//! partition by the per-record rules of a plain gate record (required subject, still
//! standing, run status, clean tree, signature), refuses a shard with transport
//! failures or without per-subset tallies, and sums per-subset counts
//! ([`crate::benchmarks::bfcl::aggregate`]) rather than averaging shard scores.
//!
//! The aggregate is exact over its shards' counts, but shards need not answer as the
//! whole draw run in one order would: the samples measured to differ are listed in
//! `crate::benchmarks::bfcl::sensitive`, and a run records how many it scored as
//! `known_partition_sensitive`.

/// 2026-09-26: One shard record, as the partition rule sees it: which slice it measured,
/// at which commit, when, and the caller's handle to the record.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ShardRecord {
    pub index: usize,
    pub count: usize,
    pub git_sha: String,
    pub recorded_at: u64,
    /// 2026-09-26: The caller's handle; `check_group` uses the position in its list.
    pub handle: usize,
}

/// 2026-09-26: The chosen partition: its shard count, the one commit it was measured
/// at, and the caller's handles in index order.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Partition {
    pub count: usize,
    pub git_sha: String,
    pub handles: Vec<usize>,
}

/// 2026-09-26: The newest complete partition among `shards`, or why there is none.
///
/// Records are grouped by `(count, commit)`, keeping the newest record per index, so a
/// re-run shard replaces its earlier record instead of being counted twice. A group is
/// complete when it holds `count` distinct indices. Among complete groups the one whose
/// newest record is newest wins; a tie goes to the larger count. Records at two commits
/// are never combined.
///
/// # Errors
/// [`GroupFault::Missing`] when `shards` is empty or no group is complete; `held` then
/// lists, per `(count, commit)`, the indices present.
pub fn select_partition(
    group: &'static str,
    shards: &[ShardRecord],
) -> Result<Partition, GroupFault> {
    if shards.is_empty() {
        return Err(GroupFault::Missing {
            group,
            held: Vec::new(),
        });
    }
    let held = held_by(shards);
    // 2026-09-26: Every complete partition, ranked by the age of its newest record.
    let mut complete: Vec<(u64, &PartitionKey, &PerIndex<'_>)> = held
        .iter()
        .filter(|(key, per_index)| per_index.len() == key.0)
        .map(|(key, per_index)| {
            let newest = per_index.values().map(|s| s.recorded_at).max().unwrap_or(0);
            (newest, key, per_index)
        })
        .collect();
    complete.sort_by(|a, b| b.0.cmp(&a.0).then_with(|| b.1.0.cmp(&a.1.0)));
    match complete.into_iter().next() {
        Some((_, (count, git_sha), per_index)) => Ok(Partition {
            count: *count,
            git_sha: git_sha.clone(),
            handles: per_index.values().map(|s| s.handle).collect(),
        }),
        None => Err(GroupFault::Missing {
            group,
            held: held
                .iter()
                .map(|((count, sha), per_index)| {
                    (*count, sha.clone(), per_index.keys().copied().collect())
                })
                .collect(),
        }),
    }
}

/// 2026-09-26: `(count, commit)`: what a partition is keyed by.
pub type PartitionKey = (usize, String);
/// 2026-09-26: The newest record per index within one partition.
pub type PerIndex<'a> = std::collections::BTreeMap<usize, &'a ShardRecord>;

/// 2026-09-26: Per `(count, commit)`, the newest record per index. On equal
/// `recorded_at` the record that comes first in `shards` is kept.
pub fn held_by(shards: &[ShardRecord]) -> std::collections::BTreeMap<PartitionKey, PerIndex<'_>> {
    let mut held: std::collections::BTreeMap<PartitionKey, PerIndex<'_>> =
        std::collections::BTreeMap::new();
    for s in shards {
        let slot = held
            .entry((s.count, s.git_sha.clone()))
            .or_default()
            .entry(s.index)
            .or_insert(s);
        if s.recorded_at > slot.recorded_at {
            *slot = s;
        }
    }
    held
}

/// 2026-09-26: A gate whose measurement is produced by a partition of shard runs.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct BenchmarkGroup {
    /// 2026-09-26: The gate id, which is also the benchmark every shard runs.
    pub id: &'static str,
}

/// 2026-09-26: Every group.
pub const GROUPS: &[BenchmarkGroup] = &[
    BenchmarkGroup { id: "bfcl-subset" },
    BenchmarkGroup {
        id: "bfcl-subset-echolp",
    },
];

/// 2026-09-26: The group a benchmark id names, if any.
pub fn find(id: &str) -> Option<&'static BenchmarkGroup> {
    GROUPS.iter().find(|g| g.id == id)
}

/// 2026-09-26: Why a group is not satisfied.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum GroupFault {
    /// 2026-09-26: No complete partition at one commit.
    Missing {
        group: &'static str,
        /// 2026-09-26: `(count, commit, indices present)` for every `(count, commit)`
        /// among the records.
        held: Vec<(usize, String, Vec<usize>)>,
    },
}

impl std::fmt::Display for GroupFault {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Missing { group, held } if held.is_empty() => write!(
                f,
                "{group} is a benchmark group and has no shard record at this commit. \
                 A group is satisfied only by a COMPLETE partition of its draw \
                 (`--param shard=i/n` for every i in 0..n at one commit): an \
                 aggregate over a subset is computed on a sample set the thresholds \
                 were never drawn against, which is a different measurement, not a \
                 partial one."
            ),
            Self::Missing { group, held } => write!(
                f,
                "{group} is a benchmark group and no shard partition is complete at one \
                 commit: {}. A group is satisfied only by a COMPLETE partition of its \
                 draw — every index 0..n once, all at one commit — since an aggregate \
                 over a subset is computed on a sample set the thresholds were never \
                 drawn against, which is a different measurement, not a partial one; \
                 and a group is ONE measurement.",
                held.iter()
                    .map(|(n, sha, idx)| {
                        let missing: Vec<String> = (0..*n)
                            .filter(|i| !idx.contains(i))
                            .map(|i| i.to_string())
                            .collect();
                        format!(
                            "{n}-way at {} holds {:?}, missing {}",
                            sha.chars().take(10).collect::<String>(),
                            idx,
                            missing.join(",")
                        )
                    })
                    .collect::<Vec<_>>()
                    .join("; ")
            ),
        }
    }
}

/// 2026-09-26: Which id's BENCH.toml entry describes how to run this benchmark:
/// `benchmark_id` itself, since a shard runs under its group's id.
pub fn serve_baseline_id(benchmark_id: &str) -> &str {
    benchmark_id
}
