// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-26: Is the local copy of a model current: does `refs/main` on disk equal the revision the Hub reports now?
//!
//! `refs/main` is also what `model_resolver` reads to pick the snapshot.
//!
//! Owner: server (model download).
//! Invariants: none beyond the types.

use std::path::Path;

use super::{DownloadError, hf};

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Freshness {
    /// 2026-09-26: Not checked, or the check failed. The TUI library list draws
    /// no badge for it (`tui/render/library/list.rs`).
    Unknown,
    Current,
    Stale {
        local: String,
        remote: String,
    },
    /// 2026-09-26: No `refs/main` on disk.
    Missing,
}

impl Freshness {
    pub fn is_stale(&self) -> bool {
        matches!(self, Self::Stale { .. })
    }
}

/// 2026-09-26: Compare the local `refs/main` with the Hub's revision: one
/// request, or none when nothing is on disk. The TUI runs one check at a time,
/// on request (`tui/download_state.rs`).
pub fn check(repo: &str, cache_root: &Path) -> Result<Freshness, DownloadError> {
    let Some(local) = hf::local_revision(cache_root, repo) else {
        return Ok(Freshness::Missing);
    };
    let (remote, _) = hf::repo_info(repo, hf::token().as_deref())?;
    Ok(if local == remote {
        Freshness::Current
    } else {
        Freshness::Stale { local, remote }
    })
}

/// 2026-09-26: The first 7 characters of a revision, for display.
pub fn short(revision: &str) -> String {
    revision.chars().take(7).collect()
}

#[cfg(test)]
#[path = "stale_tests.rs"]
mod tests;
