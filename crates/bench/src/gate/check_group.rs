// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-26: `check_one`'s path for a benchmark group: one verdict from its shard records.
//!
//! Owner: bench gate.
//! Invariants:
//! - A group passes only on a complete partition of shard records at one commit
//!   ([`super::group::select_partition`]) whose every shard is for the group and its subject,
//!   still stands, and passes `shard_problems`, with no transport errors, and whose
//!   aggregate passes `check_record`.
//! - A whole-draw record (no shard identity) never counts.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use super::check::{
    GateStatus, record_is_for, record_is_required_subject, record_still_stands,
    records_newest_first,
};
use super::group::ShardRecord;
use super::record::{GateBaseline, GateRecord, read_baseline, read_record};

/// 2026-09-26: A group's records that carry a shard identity, are for this benchmark and its
/// required subject, and still stand at `sha`, newest first. Whole-draw records are skipped
/// here and mentioned by [`whole_draw_note`].
fn standing_shards(
    root: &Path,
    baseline: &GateBaseline,
    group: &'static str,
    sha: &str,
    gate: &super::coverage::GateCoverage,
) -> Vec<(GateRecord, PathBuf)> {
    records_newest_first(root, group)
        .into_iter()
        .filter_map(|path| {
            let r = read_record(&path).ok()?;
            (r.shard().is_some()
                && record_is_for(&r, group, &path)
                && record_is_required_subject(baseline, &r, group, &path)
                && record_still_stands(root, sha, &r, gate))
            .then_some((r, path))
        })
        .collect()
}

/// 2026-09-26: The per-record checks `check_one` also applies to a plain gate's record, asked of
/// one shard: the run did not fail, the tree was clean, the signature verifies. Empty means
/// the shard counts. The shard's own verdict and thresholds are not checked; the aggregate is.
fn shard_problems(root: &Path, label: &str, record: &GateRecord, path: &Path) -> Vec<String> {
    let mut problems = Vec::new();
    if record.frame_status_failed() {
        problems.push(format!(
            "{label}: the run itself failed: {}",
            record.verdict_reason
        ));
    }
    if !record.dirty_paths.is_empty() {
        problems.push(format!(
            "{label}: measured from a dirty tree — {} uncommitted invalidation-set \
             file(s) when the run started ({}), so the binary was not {}",
            record.dirty_paths.len(),
            record.dirty_paths.join(", "),
            record.git_sha
        ));
    }
    if let Err(why) = super::signing::verify_record(root, path, &record.git_sha, record.recorded_at)
    {
        problems.push(format!("{label}: {why}"));
    }
    problems
}

fn label(group: &str, shard: (usize, usize)) -> String {
    format!("{group}[{}/{}]", shard.0, shard.1)
}

/// 2026-09-26: A group's verdict: the newest complete partition of shard records at `sha`,
/// every shard clean, tallies aggregated over counts and judged against the group's
/// thresholds.
pub(super) fn check_group(
    root: &Path,
    group: &'static super::group::BenchmarkGroup,
    sha: &str,
) -> GateStatus {
    use crate::benchmarks::bfcl::aggregate;

    let baseline = match read_baseline(root, group.id) {
        Ok(b) => b,
        Err(e) => return GateStatus::Missing(format!("baseline unreadable: {e:#}")),
    };
    let Some(gate) = super::coverage::find(group.id) else {
        return GateStatus::Missing(format!("{} has no coverage entry", group.id));
    };

    let candidates = standing_shards(root, &baseline, group.id, sha, gate);
    let shards: Vec<ShardRecord> = candidates
        .iter()
        .enumerate()
        .filter_map(|(handle, (r, _))| {
            let (index, count) = r.shard()?;
            Some(ShardRecord {
                index,
                count,
                git_sha: r.git_sha.clone(),
                recorded_at: r.recorded_at,
                handle,
            })
        })
        .collect();
    let partition = match super::group::select_partition(group.id, &shards) {
        Ok(p) => p,
        Err(fault) => {
            let mut why = vec![fault.to_string()];
            if let Some(note) = whole_draw_note(root, group) {
                why.push(note);
            }
            if let Some(note) = why_stale(root, group.id, sha, gate) {
                why.push(note);
            }
            return GateStatus::Missing(why.join(" "));
        }
    };

    let count = partition.count;
    let mut tallies: Vec<BTreeMap<String, aggregate::Tally>> = Vec::new();
    let mut problems: Vec<String> = Vec::new();
    let mut newest: Option<GateRecord> = None;
    for handle in partition.handles {
        let (record, path) = &candidates[handle];
        let lbl = label(group.id, record.shard().unwrap_or((0, count)));
        problems.extend(shard_problems(root, &lbl, record, path));
        // 2026-09-26: A shard without per-subset tallies cannot be folded in; counting it as
        // empty would score the group over fewer samples than the draw.
        let Some(t) = aggregate::tallies_from_metrics(&record.metrics) else {
            return GateStatus::Missing(format!(
                "{lbl} has a covering record but no per-subset tallies — it was \
                 measured by a binary older than the shard split, so the group \
                 cannot be aggregated. Re-run it at this commit."
            ));
        };
        let errs = record
            .metrics
            .get("transport_errors")
            .copied()
            .unwrap_or(0.0);
        if errs > 0.0 {
            return GateStatus::Fail(vec![format!(
                "{lbl} recorded {errs:.0} transport failures. Each was scored as \
                 \"made no call\", which is the CORRECT answer on the irrelevance \
                 subsets, so a degraded shard can raise the aggregate while \
                 measuring less of the draw. Re-run it."
            )]);
        }
        tallies.push(t);
        if newest
            .as_ref()
            .is_none_or(|n| record.recorded_at > n.recorded_at)
        {
            newest = Some(record.clone());
        }
    }
    if !problems.is_empty() {
        return GateStatus::Fail(problems);
    }

    let agg = aggregate::aggregate(&aggregate::union(&tallies));
    let Some(mut record) = newest else {
        return GateStatus::Missing(format!("{} has no shards", group.id));
    };
    // 2026-09-26: Judge the aggregate in the newest shard's record, which supplies the
    // checkpoint, serve overrides and hardware; every shard is at the partition's commit.
    record.benchmark_id = group.id.to_string();
    record
        .metrics
        .insert("overall_accuracy".into(), agg.overall_accuracy);
    record.metrics.insert(
        "normalized_single_turn_score".into(),
        agg.normalized_single_turn_score,
    );
    record
        .metrics
        .insert("samples".into(), agg.total_samples as f64);
    record.metrics.insert("shard.count".into(), count as f64);

    match super::scoring::check_record(&record, &baseline) {
        None => GateStatus::Pass,
        Some(breaches) => GateStatus::Fail(breaches),
    }
}

/// 2026-09-26: The `(index, count)` shards of `group` a certification at `sha` still owes, for a
/// planner that wants `wanted` shards:
/// - nothing, when the clean standing shards already form a complete partition;
/// - else the indices missing from the most complete partition measured at `sha`, whatever
///   its count (a partition is never assembled across commits);
/// - else `0..wanted`.
///
/// A shard that fails `shard_problems` is owed again. A group with no readable baseline or
/// coverage entry owes `0..wanted`.
pub fn shards_owed(
    root: &Path,
    group: &'static super::group::BenchmarkGroup,
    sha: &str,
    wanted: usize,
) -> Vec<(usize, usize)> {
    let fresh = |n: usize| (0..n).map(|i| (i, n)).collect::<Vec<_>>();
    let (Ok(baseline), Some(gate)) = (
        read_baseline(root, group.id),
        super::coverage::find(group.id),
    ) else {
        return fresh(wanted);
    };
    let candidates = standing_shards(root, &baseline, group.id, sha, gate);
    let clean: Vec<ShardRecord> = candidates
        .iter()
        .enumerate()
        .filter_map(|(handle, (record, path))| {
            let (index, count) = record.shard()?;
            shard_problems(root, &label(group.id, (index, count)), record, path)
                .is_empty()
                .then(|| ShardRecord {
                    index,
                    count,
                    git_sha: record.git_sha.clone(),
                    recorded_at: record.recorded_at,
                    handle,
                })
        })
        .collect();
    if super::group::select_partition(group.id, &clean).is_ok() {
        return Vec::new();
    }
    // 2026-09-26: Finish the partition at this commit that is closest to done.
    let held = super::group::held_by(&clean);
    match held
        .iter()
        .filter(|((_, commit), _)| commit.starts_with(sha) || sha.starts_with(commit.as_str()))
        .max_by_key(|((count, _), idx)| (idx.len() * 1000 / *count, *count))
    {
        Some(((count, _), idx)) => (0..*count)
            .filter(|i| !idx.contains_key(i))
            .map(|i| (i, *count))
            .collect(),
        None => fresh(wanted),
    }
}

/// 2026-09-26: Why the group's newest shard record does not stand: the paths that invalidated
/// it, or a build-input mismatch. `None` when there is no shard record, it stands, or git
/// cannot diff.
fn why_stale(
    root: &Path,
    group: &str,
    sha: &str,
    gate: &super::coverage::GateCoverage,
) -> Option<String> {
    let newest = records_newest_first(root, group)
        .into_iter()
        .find_map(|p| {
            read_record(&p)
                .ok()
                .filter(|r| r.shard().is_some())
                .map(|r| (r, p))
        })?;
    let (record, path) = newest;
    let name = path
        .file_name()
        .unwrap_or_default()
        .to_string_lossy()
        .into_owned();
    if record_still_stands(root, sha, &record, gate) {
        return None;
    }
    let why = super::check_paths::invalidating_paths(root, sha, &record.git_sha, gate)?;
    Some(if why.is_empty() {
        format!(
            "Newest shard record is for {} ({name}) — its recorded build inputs do not \
             match this commit.",
            record.git_sha
        )
    } else {
        format!(
            "Newest shard record is for {} ({name}) — invalidated by {}.",
            record.git_sha,
            super::check_fmt::summarize_paths(&why)
        )
    })
}

/// 2026-09-26: The note appended when a group directory holds whole-draw records but no
/// complete partition: those records do not satisfy the gate.
fn whole_draw_note(root: &Path, group: &super::group::BenchmarkGroup) -> Option<String> {
    let whole: Vec<PathBuf> = records_newest_first(root, group.id)
        .into_iter()
        .filter(|p| read_record(p).is_ok_and(|r| r.shard().is_none()))
        .collect();
    let newest = whole.first()?;
    Some(format!(
        "The {} whole-draw record(s) under {} (newest {}) no longer satisfy the gate: since \
         2026-09-13 a group is certified only by a complete partition of shards — see \
         gate::group.",
        whole.len(),
        group.id,
        newest.file_name().unwrap_or_default().to_string_lossy(),
    ))
}
