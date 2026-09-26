// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-26: Writing one gate record without overwriting a failing one.
//!
//! Record names start `<date>-<sha>`, so a same-day re-run at the same commit
//! maps to the same name. When that name holds a FAIL, the new record is
//! written beside it under the re-run name
//! ([`super::record_path::rerun_path`]). `records_newest_first` orders by
//! `recorded_at`, so the newest record still decides the gate.
//!
//! Owner: bench gate (records).
//! Invariants:
//! - `write_record` does not replace a file that holds a FAIL record or that
//!   does not parse as a record. The check runs just before the write and
//!   takes no lock.

use std::path::{Path, PathBuf};

use anyhow::{Context, Result, bail};

use super::gate_dir;
use super::record::{GateRecord, read_record, record_path_for};
use super::record_path::rerun_path;

/// 2026-09-26: Write one gate record and return its path. Creates the parent
/// directory; commits nothing. The path is `record_path_for` unless that file
/// holds a FAIL, in which case it is the re-run name. Errors when the re-run
/// name holds a FAIL too, or when either file exists but is not a readable
/// record.
pub fn write_record(root: &Path, record: &GateRecord) -> Result<PathBuf> {
    let path = preserving_path(&record_path_for(root, record), record)?;
    std::fs::create_dir_all(path.parent().expect("record path has a parent")).with_context(
        || {
            format!(
                "creating {}",
                gate_dir(root, &record.benchmark_id).display()
            )
        },
    )?;
    let json = serde_json::to_string_pretty(record).context("serializing the gate record")?;
    std::fs::write(&path, json + "\n").with_context(|| format!("writing {}", path.display()))?;
    Ok(path)
}

/// 2026-09-26: `canonical` when it is free or holds a record that is not a
/// FAIL; the re-run name when it holds a FAIL; an error when the re-run name
/// holds a FAIL too.
fn preserving_path(canonical: &Path, record: &GateRecord) -> Result<PathBuf> {
    if !holds_failure(canonical)? {
        return Ok(canonical.to_path_buf());
    }
    let rerun = rerun_path(canonical, record.recorded_at);
    if holds_failure(&rerun)? {
        bail!(
            "{} and {} both hold FAILING records for this commit; neither is overwritten. \
             A failing record is evidence — remove one with `git rm` if it must go.",
            canonical.display(),
            rerun.display()
        );
    }
    Ok(rerun)
}

/// 2026-09-26: Whether `path` holds a record whose verdict is FAIL. A missing
/// file is `false`; a file that exists but does not parse is an error, so it
/// is not overwritten.
fn holds_failure(path: &Path) -> Result<bool> {
    if !path.exists() {
        return Ok(false);
    }
    let existing = read_record(path).with_context(|| {
        format!(
            "{} exists but is not a readable gate record, so it is not overwritten",
            path.display()
        )
    })?;
    Ok(existing.verdict.as_deref() == Some("FAIL"))
}

#[cfg(test)]
#[path = "record_write_tests.rs"]
mod record_write_tests;
