// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-26: File names for gate records under `.benchmarks/<id>/`: the
//! default variant, model variants, shards and same-day re-runs.
//!
//! Owner: bench gate (records).
//! Invariants: none beyond the types.

use std::path::{Path, PathBuf};

use sha2::{Digest, Sha256};

use super::gate_dir;
use super::record::{GateBaseline, GateRecord};

/// 2026-09-26: `YYYY-MM-DD` (UTC) from unix seconds.
pub fn date_of(unix_secs: u64) -> String {
    let days = (unix_secs / 86_400) as i64 + 719_468;
    let era = days.div_euclid(146_097);
    let doe = days.rem_euclid(146_097) as u64;
    let yoe = (doe - doe / 1460 + doe / 36_524 - doe / 146_096) / 365;
    let y = yoe as i64 + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = doy - (153 * mp + 2) / 5 + 1;
    let m = if mp < 10 { mp + 3 } else { mp - 9 };
    let y = if m <= 2 { y + 1 } else { y };
    format!("{y:04}-{m:02}-{d:02}")
}

/// 2026-09-26: `HHMMSS` (UTC) from unix seconds.
pub fn time_of_day(unix_secs: u64) -> String {
    let secs = unix_secs % 86_400;
    format!("{:02}{:02}{:02}", secs / 3_600, (secs / 60) % 60, secs % 60)
}

/// 2026-09-26: `.benchmarks/<id>/YYYY-MM-DD-<sha>.json` under `root`.
pub fn record_path(root: &Path, benchmark_id: &str, unix_secs: u64, sha: &str) -> PathBuf {
    gate_dir(root, benchmark_id).join(format!("{}-{sha}.json", date_of(unix_secs)))
}

/// 2026-09-26: The re-run name beside `canonical`: the day segment gains the
/// run's UTC time of day, `YYYY-MM-DDTHHMMSSZ-<sha>[-<variant>][-s<i>of<n>].json`,
/// and the rest of the name is kept. A lexical sort orders it after
/// `canonical` (`-` sorts before `T`).
///
/// Panics when the file name does not start with the UTC day of `unix_secs`.
pub fn rerun_path(canonical: &Path, unix_secs: u64) -> PathBuf {
    let name = canonical
        .file_name()
        .map(|n| n.to_string_lossy().into_owned())
        .unwrap_or_default();
    let day = date_of(unix_secs);
    let tail = name
        .strip_prefix(day.as_str())
        .unwrap_or_else(|| panic!("record name {name:?} does not start with its day {day}"));
    canonical.with_file_name(format!("{day}T{}Z{tail}", time_of_day(unix_secs)))
}

/// 2026-09-26: The file-name tail for one shard's record,
/// `-s<index>of<count>`, 0-based like the `--param shard=i/n` that ran it.
/// Empty for a whole-draw run.
pub fn shard_suffix(shard: Option<(usize, usize)>) -> String {
    shard.map_or(String::new(), |(i, n)| format!("-s{i}of{n}"))
}

/// 2026-09-26: A file-name-safe slug of a checkpoint id: ASCII alphanumerics
/// and `.` kept and lowercased, each run of other characters one `-`, edge
/// `-` trimmed. Two ids can share a slug.
pub fn variant_slug(model: &str) -> String {
    let mut out = String::with_capacity(model.len());
    for c in model.chars() {
        let mapped = if c.is_ascii_alphanumeric() || c == '.' {
            c.to_ascii_lowercase()
        } else {
            '-'
        };
        if mapped == '-' && out.ends_with('-') {
            continue;
        }
        out.push(mapped);
    }
    out.trim_matches('-').to_string()
}

/// 2026-09-26: The variant's slug, with the first 16 hex digits of the id's
/// SHA-256 appended when another checkpoint in `baseline` has the same slug.
fn variant_file_slug(baseline: &GateBaseline, model: &str) -> String {
    let slug = variant_slug(model);
    let collides = baseline
        .hardware
        .values()
        .flat_map(|hardware| hardware.models.keys())
        .any(|other| other != model && variant_slug(other) == slug);
    if !collides {
        return slug;
    }
    let digest = format!("{:x}", Sha256::digest(model.as_bytes()));
    format!("{slug}-{}", &digest[..16])
}

/// 2026-09-26: Where `record` is written: `<date>-<sha><shard>.json` for its
/// hardware's default model, or when no baseline with hardware entries
/// resolves; otherwise `<date>-<sha>-<variant slug><shard>.json`. A record
/// whose hardware has no entry counts as default when it is any hardware's
/// default.
pub fn record_path_for(root: &Path, record: &GateRecord) -> PathBuf {
    let shard = shard_suffix(record.shard());
    let legacy = gate_dir(root, &record.benchmark_id).join(format!(
        "{}-{}{shard}.json",
        date_of(record.recorded_at),
        record.git_sha
    ));
    let Ok(baseline) = super::bench::baseline_for(root, &record.benchmark_id) else {
        return legacy;
    };
    if baseline.hardware.is_empty() {
        return legacy;
    }
    let hardware = record.hardware.gate_key();
    let is_default = match baseline.hardware.get(&hardware) {
        Some(hw) => hw.default == record.target_model,
        None => baseline
            .hardware
            .values()
            .any(|hw| hw.default == record.target_model),
    };
    if is_default {
        legacy
    } else {
        gate_dir(root, &record.benchmark_id).join(format!(
            "{}-{}-{}{shard}.json",
            date_of(record.recorded_at),
            record.git_sha,
            variant_file_slug(&baseline, &record.target_model)
        ))
    }
}
