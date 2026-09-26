// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-26: `--pull-request-gate-check`: does this commit have a passing record for every
//! required gate? Reads `.benchmarks/`, the kernel tree and `git diff`; no endpoint, no GPU.
//!
//! Owner: bench gate.
//! Invariants:
//! - [`exit_code`] depends only on the verdicts it is given.
//! - A gate is [`GateStatus::Pass`] only when a readable record (for a group, a complete
//!   shard partition) stands at the commit and passes every check; otherwise it is
//!   [`GateStatus::Missing`] or [`GateStatus::Fail`].

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use super::check_paths::invalidating_paths;
use super::record::{GateBaseline, GateRecord, read_baseline, read_record};
use super::{REQUIRED_GATES, gate_dir};

pub use super::scoring::{Comparison, check_record, compare};

/// 2026-09-26: One required gate's standing in the committed tree.
#[derive(Debug)]
pub enum GateStatus {
    /// 2026-09-26: The newest covering record passes every check.
    Pass,
    /// 2026-09-26: The newest covering record fails: one entry per problem (run failed, dirty
    /// tree, verdict not PASS, baseline breach, signature).
    Fail(Vec<String>),
    /// 2026-09-26: No record covers this commit; the text says why.
    Missing(String),
}

/// 2026-09-26: The `.json` record files in one benchmark's directory, newest first by each
/// record's own `recorded_at`; `BASELINE.json` is not a record.
///
/// The file name is not a clock: two records from one UTC day would sort by their sha. An
/// unreadable record sorts last but stays in the list, so a directory of corrupt records
/// reports "unreadable" rather than "no records committed".
pub fn records_newest_first(root: &Path, benchmark_id: &str) -> Vec<PathBuf> {
    let dir = gate_dir(root, benchmark_id);
    let mut candidates: Vec<PathBuf> = std::fs::read_dir(&dir)
        .map(|entries| entries.flatten().map(|e| e.path()).collect())
        .unwrap_or_default();
    candidates.retain(|p| {
        p.extension().is_some_and(|e| e == "json")
            && p.file_name()
                .is_some_and(|n| n.to_string_lossy() != "BASELINE.json")
    });
    let mut keyed: Vec<(u64, PathBuf)> = candidates
        .into_iter()
        .map(|p| (read_record(&p).map(|r| r.recorded_at).unwrap_or(0), p))
        .collect();
    // 2026-09-26: The file name breaks a `recorded_at` tie, so the order does not depend on
    // readdir.
    keyed.sort_by(|a, b| b.0.cmp(&a.0).then_with(|| b.1.cmp(&a.1)));
    keyed.into_iter().map(|(_, p)| p).collect()
}

/// 2026-09-26: Whether a record measured at `record_sha` still stands for `head` on the path
/// boundary alone: true when `git diff` between the two commits touches no path that
/// invalidates `gate`, false when it does or git cannot diff them. Ancestry is not required,
/// and the closure hash is not consulted (see [`record_standing`]).
pub fn record_covers(
    root: &Path,
    head: &str,
    record_sha: &str,
    gate: &super::coverage::GateCoverage,
) -> bool {
    invalidating_paths(root, head, record_sha, gate).is_some_and(|p| p.is_empty())
}

/// 2026-09-26: Whether `record` still stands for `sha` ([`record_standing`] is `Stands`).
pub(super) fn record_still_stands(
    root: &Path,
    sha: &str,
    record: &GateRecord,
    gate: &super::coverage::GateCoverage,
) -> bool {
    matches!(record_standing(root, sha, record, gate), Standing::Stands)
}

/// 2026-09-26: Where a record measured at its own commit stands relative to `sha`. The gate
/// verdict (`record_still_stands`) and record agreement (`agreement::standing_at`) both read
/// this, so they cannot disagree about a record.
///
/// A diff whose invalidating paths are all excused by the record's closure attestation
/// ([`super::closure::excuses`]) still stands; the closure can only narrow the path boundary.
pub fn record_standing(
    root: &Path,
    sha: &str,
    record: &GateRecord,
    gate: &super::coverage::GateCoverage,
) -> Standing {
    match invalidating_paths(root, sha, &record.git_sha, gate) {
        None => Standing::Unknown,
        Some(paths) if paths.is_empty() => Standing::Stands,
        Some(paths) => {
            if super::closure::excuses(root, &paths, &record.closure) {
                Standing::Stands
            } else {
                Standing::Invalidated(paths)
            }
        }
    }
}

/// 2026-09-26: A record's standing at a head, from its own commit.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Standing {
    /// 2026-09-26: The diff from the record's commit to the head touches nothing that
    /// invalidates this gate, or the record's closure excuses what it touches.
    Stands,
    /// 2026-09-26: git could not diff the record's commit against the head: a commit this
    /// clone does not have, or git failed.
    Unknown,
    /// 2026-09-26: The diff touches these invalidating paths for the record's gate.
    Invalidated(Vec<String>),
}

/// 2026-09-26: The verdict for `sha` of every gate in `REQUIRED_GATES`, keyed by gate id.
pub fn check_gates(root: &Path, sha: &str) -> BTreeMap<String, GateStatus> {
    let mut out = BTreeMap::new();
    for id in REQUIRED_GATES {
        out.insert((*id).to_string(), check_one(root, id, sha));
    }
    out
}

/// 2026-09-26: Whether a record's own `benchmark_id` is the gate whose directory it sits in.
/// `ttft-warm-gate` and `ttft-cold-gate` share metric names (`median_ms`, `p90_ms`), so a
/// record copied between their directories would otherwise satisfy the wrong gate.
///
/// A mismatch is logged and skipped, not failed, so a stray file leaves the gate reading
/// "no covering record" rather than failing on another gate's run.
pub(super) fn record_is_for(record: &GateRecord, benchmark_id: &str, path: &Path) -> bool {
    if record.benchmark_id == benchmark_id {
        return true;
    }
    tracing::warn!(
        "ignoring {}: it is a `{}` record sitting in the `{benchmark_id}` directory",
        path.display(),
        record.benchmark_id,
    );
    false
}

/// 2026-09-26: Whether this record measured the gate's required subject: the checkpoint the
/// baseline marks `default = true` for the record's box class. A record of another variant
/// in the same directory is logged and skipped, so its pass is not evidence for the gate.
///
/// A record from a box class the baseline does not know returns true; `check_record` then
/// fails it as having no baseline for that hardware.
pub(super) fn record_is_required_subject(
    baseline: &GateBaseline,
    record: &GateRecord,
    benchmark_id: &str,
    path: &Path,
) -> bool {
    let hardware = record.hardware.gate_key();
    let Some(hw) = baseline.hardware.get(&hardware) else {
        return true;
    };
    if hw.default == record.target_model {
        return true;
    }
    tracing::warn!(
        "ignoring {} for the required {benchmark_id} gate: it measured the {} variant, \
         and the gate's declared subject on {hardware} is {}",
        path.display(),
        record.target_model,
        hw.default,
    );
    false
}

pub(super) fn check_one(root: &Path, benchmark_id: &str, sha: &str) -> GateStatus {
    // 2026-09-26: A group is judged only by its shard records (`check_group`); a whole-draw
    // record under the group's id does not satisfy it.
    if let Some(group) = super::group::find(benchmark_id) {
        return super::check_group::check_group(root, group, sha);
    }
    let Some(gate) = super::coverage::find(benchmark_id) else {
        // 2026-09-26: Unreachable through `check_gates`, whose ids come from
        // `coverage::REQUIRED`. Refused rather than judged with no exclusions.
        return GateStatus::Missing(format!("{benchmark_id} has no coverage entry"));
    };
    let paths = records_newest_first(root, benchmark_id);
    if paths.is_empty() {
        return GateStatus::Missing("no gate records committed".into());
    }
    let baseline = match read_baseline(root, benchmark_id) {
        Ok(b) => b,
        Err(e) => return GateStatus::Missing(format!("baseline unreadable: {e:#}")),
    };
    // 2026-09-26: The newest record that is for this gate and subject and still stands for
    // `sha`; others are skipped, not failed. Its path is kept because the signature sidecar
    // is found beside the file.
    let mut covered: Option<(GateRecord, std::path::PathBuf)> = None;
    for path in &paths {
        if let Ok(record) = read_record(path)
            && record_is_for(&record, benchmark_id, path)
            && record_is_required_subject(&baseline, &record, benchmark_id, path)
            && record_still_stands(root, sha, &record, gate)
        {
            covered = Some((record, path.clone()));
            break;
        }
    }
    let Some((record, record_path)) = covered else {
        let newest = read_record(&paths[0]).ok();
        return GateStatus::Missing(match newest {
            // 2026-09-26: Say why the newest record does not count: another gate's record,
            // another variant, a commit git cannot diff, or the paths that invalidated it.
            Some(newest_record) => {
                if newest_record.benchmark_id != benchmark_id {
                    return GateStatus::Missing(format!(
                        "latest record belongs to {}, not {benchmark_id} ({})",
                        newest_record.benchmark_id,
                        paths[0].file_name().unwrap_or_default().to_string_lossy()
                    ));
                }
                if !record_is_required_subject(&baseline, &newest_record, benchmark_id, &paths[0]) {
                    let hardware = newest_record.hardware.gate_key();
                    let subject = baseline
                        .hardware
                        .get(&hardware)
                        .map(|hw| hw.default.clone())
                        .unwrap_or_default();
                    return GateStatus::Missing(format!(
                        "latest record measured the {} variant; the required subject on \
                         {hardware} is {subject}, which has no covering record",
                        newest_record.target_model
                    ));
                }
                let newest = newest_record.git_sha.clone();
                let Some(why) = invalidating_paths(root, sha, &newest, gate) else {
                    return GateStatus::Missing(format!(
                        "latest record is for {newest} ({}) — git cannot diff that commit \
                         against this one; is it in this clone? (the gate job needs \
                         `fetch-depth: 0`)",
                        paths[0].file_name().unwrap_or_default().to_string_lossy()
                    ));
                };
                let because = if why.is_empty() {
                    // 2026-09-26: No invalidating path, yet the record did not stand.
                    "its recorded build inputs do not match this commit".to_string()
                } else {
                    // 2026-09-26: Also name the kernel targets whose device code changed.
                    let targets =
                        super::closure::changed_targets(root, &why, &newest_record.closure);
                    match targets.len() {
                        0 => format!("invalidated by {}", super::check_fmt::summarize_paths(&why)),
                        n => format!(
                            "invalidated by {} — device code changed for {n} target(s): {}",
                            super::check_fmt::summarize_paths(&why),
                            super::check_fmt::summarize_paths(&targets)
                        ),
                    }
                };
                format!(
                    "latest record is for {newest} ({}) — {because}",
                    paths[0].file_name().unwrap_or_default().to_string_lossy()
                )
            }
            None => "latest record is unreadable".to_string(),
        });
    };
    if record.frame_status_failed() {
        return GateStatus::Fail(vec![format!(
            "the run itself failed: {}",
            record.verdict_reason
        )]);
    }
    let mut problems = Vec::new();
    // 2026-09-26: A record measured with uncommitted invalidation-set changes describes no
    // commit, so it fails; the diff check above cannot see uncommitted edits. A record
    // without the field deserializes it as empty.
    if !record.dirty_paths.is_empty() {
        problems.push(format!(
            "measured from a dirty tree — {} uncommitted invalidation-set \
             file(s) when the run started ({}), so the binary was not {}",
            record.dirty_paths.len(),
            record.dirty_paths.join(", "),
            record.git_sha
        ));
    }
    if !record.verdict_passes() {
        problems.push(format!(
            "run verdict is not PASS: {}",
            record.verdict_reason
        ));
    }
    if let Some(breaches) = check_record(&record, &baseline) {
        problems.extend(breaches);
    }
    // 2026-09-26: A signature `verify_record` rejects fails the gate.
    if let Err(why) =
        super::signing::verify_record(root, &record_path, &record.git_sha, record.recorded_at)
    {
        problems.push(format!("{why}"));
    }
    if problems.is_empty() {
        GateStatus::Pass
    } else {
        GateStatus::Fail(problems)
    }
}

/// 2026-09-26: The exit code: 0 when every verdict is `Pass`, else 1.
///
/// It takes only the verdicts, so advisory data reported beside them
/// ([`super::required::RequiredReport`], the governance ledger) cannot change it without a
/// visible change to this signature.
pub fn exit_code(statuses: &BTreeMap<String, GateStatus>) -> i32 {
    let open = statuses
        .values()
        .filter(|s| !matches!(s, GateStatus::Pass))
        .count();
    i32::from(open > 0)
}

#[cfg(test)]
#[path = "check_tests.rs"]
mod check_tests;
