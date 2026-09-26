// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-26: A startup warning when the filesystem that holds the Hugging
//! Face cache is nearly full.
//!
//! Owner: server startup.
//! Invariants: it only logs. `warn_if_nearly_full` returns nothing, so startup
//! continues whatever the reading.

use std::path::Path;

/// 2026-09-26: Warn at or above this fraction of the filesystem in use.
const WARN_AT: f64 = 0.97;

/// 2026-09-26: One reading, in bytes: `free` is what an unprivileged writer
/// may still use (`f_bavail`), `total` the filesystem size.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Usage {
    pub free: u64,
    pub total: u64,
}

impl Usage {
    pub fn used_fraction(&self) -> f64 {
        if self.total == 0 {
            return 0.0;
        }
        1.0 - (self.free as f64 / self.total as f64)
    }

    /// 2026-09-26: The warning text when `used_fraction()` is at least
    /// `WARN_AT`, else `None`. A zero `total` reads as 0% used. It does no
    /// I/O; [`usage`] takes the reading.
    pub fn warning(&self, path: &Path) -> Option<String> {
        if self.used_fraction() < WARN_AT {
            return None;
        }
        Some(format!(
            "Disk {:.0}% full at {} — only {:.1} GB free of {:.1} GB. \
             Model downloads will fail and page-cache thrashing can make \
             benchmarks look like regressions. Free space before starting a \
             long run.",
            self.used_fraction() * 100.0,
            path.display(),
            self.free as f64 / 1e9,
            self.total as f64 / 1e9,
        ))
    }
}

/// 2026-09-26: Read the filesystem holding `path`, or its nearest ancestor
/// that `statvfs` accepts. `None` off unix, or when no ancestor gives a
/// non-zero size.
pub fn usage(path: &Path) -> Option<Usage> {
    crate::model_download::hf::disk_usage(path).map(|(free, total)| Usage { free, total })
}

/// 2026-09-26: Log the warning for the Hugging Face cache root that
/// `resolve_cache_root` picks, where model downloads are written. Silent when
/// the disk is under the threshold, the root cannot be resolved, or there is
/// no reading.
pub fn warn_if_nearly_full(cache_dir: Option<&Path>) {
    let Ok(root) = crate::model_resolver::resolve_cache_root(cache_dir) else {
        return;
    };
    let Some(u) = usage(&root) else { return };
    if let Some(msg) = u.warning(&root) {
        tracing::warn!("{msg}");
    }
}

#[cfg(test)]
#[path = "disk_guard_tests.rs"]
mod tests;
