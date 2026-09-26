// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-26: Content-pinned amnesty: a table of paths whose exact blob at the
//! head excuses them from invalidating gate records.
//!
//! `check_paths::invalidating_paths` drops a changed path only when
//! `excused` says the path is listed in `ONE_TIME_AMNESTY` and its blob at
//! the head is the pinned 40-hex OID; each drop is logged with
//! `tracing::warn!`. Any later edit changes the OID, so the path invalidates
//! again. The production table is empty, so today nothing is excused.
//!
//! This file is in `coverage::BOUNDARY_FILES`: editing it re-opens every gate.
//!
//! Owner: bench gate.
//! Invariants:
//! - `excused_by` returns true only when the path is listed and git reports
//!   exactly the pinned 40-hex OID for `<head>:<path>`; every git failure is
//!   `false`.

use std::path::Path;

/// 2026-09-26: One excused path: the file, the exact blob its grant covers,
/// and why.
#[derive(Debug, Clone, Copy)]
pub struct AmnestyEntry {
    pub path: &'static str,
    /// 2026-09-26: The 40-hex blob OID the grant covers, as printed by
    /// `git rev-parse <head>:<path>`. Any value that is not a 40-hex OID
    /// matches no blob.
    pub head_blob_oid: &'static str,
    pub grant: &'static str,
}

/// 2026-09-26: 2026-08-30T00:00:00Z. A record counts as fresh for the expiry
/// test only when its `recorded_at` is later than this.
pub const AMNESTY_EPOCH: u64 = 1_788_048_000;

/// 2026-09-26: The production amnesty table. It is empty.
///
/// `amnesty_expires_once_every_gate_has_a_fresh_record` depends on it: with
/// entries, it fails once every required gate that has a record has one newer
/// than [`AMNESTY_EPOCH`]; empty, it fails if any such gate's newest record is
/// not newer. `the_table_is_exactly_the_pr_816_grant` allows only an empty
/// table or the single path `crates/bench/src/gate/coverage.rs`.
pub const ONE_TIME_AMNESTY: [AmnestyEntry; 0] = [];

/// 2026-09-26: Whether [`ONE_TIME_AMNESTY`] excuses `path` at `head`.
pub fn excused(root: &Path, head: &str, path: &str) -> bool {
    excused_by(root, head, path, &ONE_TIME_AMNESTY)
}

/// 2026-09-26: [`excused`] against an explicit table, so tests can pin real
/// OIDs.
///
/// True iff `path` is in `table` and the blob at `<head>:<path>` is exactly
/// the pinned 40-hex OID. Anything else, including any git failure, is
/// `false`.
pub fn excused_by(root: &Path, head: &str, path: &str, table: &[AmnestyEntry]) -> bool {
    let Some(entry) = table.iter().find(|e| e.path == path) else {
        return false;
    };
    let Ok(out) = std::process::Command::new("git")
        .arg("-C")
        .arg(root)
        .args(["rev-parse", &format!("{head}:{path}")])
        .stdin(std::process::Stdio::null())
        .output()
    else {
        return false;
    };
    if !out.status.success() {
        return false;
    }
    let oid = String::from_utf8_lossy(&out.stdout).trim().to_string();
    oid.len() == 40 && oid.chars().all(|c| c.is_ascii_hexdigit()) && oid == entry.head_blob_oid
}
