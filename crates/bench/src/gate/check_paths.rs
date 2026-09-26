// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-26: Which changed paths invalidate a gate record.
//!
//! Owner: bench gate.
//! Invariants:
//! - When git cannot diff the two commits, [`invalidating_paths`] returns `None`, never an
//!   empty list.

use std::path::Path;

/// 2026-09-26: The changed paths that invalidate `gate` between two commits; empty when the
/// commits are equal.
///
/// `None` means git failed or one commit is not in this clone; callers treat it as not
/// covered.
///
/// The question is whether the trees differ on paths the gate cares about, so this runs
/// `git diff --name-only` and does not require `record_sha` to be an ancestor of `head`: a
/// record measured on a branch still covers the squash-merged commit when those paths agree.
///
/// The diff has no pathspec; [`super::coverage::invalidates`] filters it in Rust, so the
/// per-gate policy stays in `coverage.rs`.
pub fn invalidating_paths(
    root: &Path,
    head: &str,
    record_sha: &str,
    gate: &super::coverage::GateCoverage,
) -> Option<Vec<String>> {
    invalidating_paths_with(root, head, record_sha, gate, |path| {
        super::amnesty::excused(root, head, path)
    })
}

fn invalidating_paths_with(
    root: &Path,
    head: &str,
    record_sha: &str,
    gate: &super::coverage::GateCoverage,
    mut is_excused: impl FnMut(&str) -> bool,
) -> Option<Vec<String>> {
    if head == record_sha {
        return Some(Vec::new());
    }
    let out = std::process::Command::new("git")
        .arg("-C")
        .arg(root)
        .args(["diff", "--name-only", record_sha, head])
        .stdin(std::process::Stdio::null())
        .output()
        .ok()?;
    if !out.status.success() {
        return None;
    }
    Some(
        String::from_utf8_lossy(&out.stdout)
            .lines()
            .map(str::trim)
            .filter(|p| !p.is_empty())
            .filter(|p| super::coverage::invalidates(gate, p))
            // 2026-09-26: A path whose blob at `head` is the OID an amnesty entry pins is
            // excused, with a warning. `ONE_TIME_AMNESTY` is empty, so in production nothing
            // is.
            .filter(|p| {
                if is_excused(p) {
                    tracing::warn!(
                        "amnesty: {p} would re-open {} but its content at {head} is the \
                         pinned one-time grant; excused (see gate/amnesty.rs)",
                        gate.id
                    );
                    return false;
                }
                true
            })
            .map(str::to_string)
            .collect(),
    )
}

#[cfg(test)]
pub(crate) fn invalidating_paths_with_amnesty(
    root: &Path,
    head: &str,
    record_sha: &str,
    gate: &super::coverage::GateCoverage,
    table: &[super::amnesty::AmnestyEntry],
) -> Option<Vec<String>> {
    invalidating_paths_with(root, head, record_sha, gate, |path| {
        super::amnesty::excused_by(root, head, path, table)
    })
}
