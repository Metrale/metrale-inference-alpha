// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-26: The pull-request gate: benchmark records committed beside the code they
//! measured.
//!
//! Owner: bench gate.
//! Invariants: none beyond the types.
//!
//! Each run writes one record file under `.benchmarks/<id>/`; a default-variant record
//! is named `YYYY-MM-DD-<sha>.json` ([`record_path()`]). The thresholds a record must meet
//! are assembled from every `kernels/<hw>/<model>/BENCH.toml` ([`read_baseline`]).
//! [`check_gates`] judges every id in [`REQUIRED_GATES`] against the committed files.

pub mod amnesty;
pub mod bench;
pub mod card;
pub mod check;
mod check_fmt;
mod check_group;
pub use check_group::shards_owed;
pub mod check_paths;
pub mod closure;
pub mod codeowners;
pub mod coverage;
pub mod hermetic;
pub mod record;
mod record_env;
mod record_path;
pub use record_path::shard_suffix;
pub mod record_serve;
mod record_summary;
mod record_write;
pub mod scoring;
pub mod signing;
pub mod taxon;

pub mod pr_taxonomy;
pub mod required;
pub mod telemetry;

use std::path::{Path, PathBuf};
use std::time::Duration;

use anyhow::{Context, Result, bail};

pub use check::{
    Comparison, GateStatus, Standing, check_gates, check_record, compare, record_covers,
    record_standing, records_newest_first,
};
pub use record::{
    Bound, GateBaseline, GateRecord, HardwareBaseline, ModelBaseline, date_of,
    merge_serve_overrides, now_secs, read_baseline, read_record, record_path, record_path_for,
    variant_slug, write_record,
};

/// 2026-09-26: The ids of `coverage::REQUIRED`, in its order: the gates [`check_gates`]
/// judges. A test in `registry.rs` looks up each id as a registered benchmark.
pub const REQUIRED_GATES: [&str; 13] = [
    coverage::REQUIRED[0].id,
    coverage::REQUIRED[1].id,
    coverage::REQUIRED[2].id,
    coverage::REQUIRED[3].id,
    coverage::REQUIRED[4].id,
    coverage::REQUIRED[5].id,
    coverage::REQUIRED[6].id,
    coverage::REQUIRED[7].id,
    coverage::REQUIRED[8].id,
    coverage::REQUIRED[9].id,
    coverage::REQUIRED[10].id,
    coverage::REQUIRED[11].id,
    coverage::REQUIRED[12].id,
];

/// 2026-09-26: The timeout for the endpoint's `/hardware` fetch when a gate record is
/// written (`bench_record.rs` in `metrale-server`).
pub const HARDWARE_TIMEOUT: Duration = Duration::from_secs(10);

/// 2026-09-26: The paths a diff must touch to invalidate a gate record, before the
/// per-gate exclusions in `coverage` subtract from them ([`check::record_covers`]).
pub use coverage::PERF_PATHS;

/// 2026-09-26: `.benchmarks/<benchmark_id>` under `root`.
pub fn gate_dir(root: &Path, benchmark_id: &str) -> PathBuf {
    root.join(".benchmarks").join(benchmark_id)
}

/// 2026-09-26: The full 40-hex commit id `rev` resolves to in this working tree.
///
/// # Errors
/// If git cannot run, `rev` does not name a commit, or git prints something other
/// than 40 hex digits.
pub fn git_rev_parse(root: &Path, rev: &str) -> Result<String> {
    let out = std::process::Command::new("git")
        .arg("-C")
        .arg(root)
        .args(["rev-parse", "--verify", &format!("{rev}^{{commit}}")])
        .stdin(std::process::Stdio::null())
        .output()
        .context("running git rev-parse")?;
    if !out.status.success() {
        bail!(
            "{rev:?} does not name a commit in {}: {}",
            root.display(),
            String::from_utf8_lossy(&out.stderr).trim()
        );
    }
    let sha = String::from_utf8_lossy(&out.stdout).trim().to_string();
    if sha.len() != 40 || !sha.chars().all(|c| c.is_ascii_hexdigit()) {
        bail!("git rev-parse returned {sha:?}, not a 40-hex commit");
    }
    Ok(sha)
}

/// 2026-09-26: The repository root that contains `dir`, from git itself.
///
/// # Errors
/// If git cannot run, or `dir` is not inside a git working tree.
pub fn git_rev_parse_toplevel(dir: &Path) -> Result<PathBuf> {
    let out = std::process::Command::new("git")
        .arg("-C")
        .arg(dir)
        .args(["rev-parse", "--show-toplevel"])
        .stdin(std::process::Stdio::null())
        .output()
        .context("running git rev-parse --show-toplevel")?;
    if !out.status.success() {
        bail!(
            "{} is not inside a git working tree: {}",
            dir.display(),
            String::from_utf8_lossy(&out.stderr).trim()
        );
    }
    Ok(PathBuf::from(String::from_utf8_lossy(&out.stdout).trim()))
}

/// 2026-09-26: The short commit id of `HEAD` (`--short=10`), or the trimmed value of
/// `METRALE_GATE_SHA` when that is set, for a checkout without git metadata.
///
/// # Errors
/// If `METRALE_GATE_SHA` is set but empty, or git fails or prints nothing.
pub fn git_sha(root: &Path) -> Result<String> {
    if let Some(explicit) = std::env::var_os("METRALE_GATE_SHA") {
        let sha = explicit.to_string_lossy().trim().to_string();
        if sha.is_empty() {
            bail!("METRALE_GATE_SHA is set but empty");
        }
        return Ok(sha);
    }
    let out = std::process::Command::new("git")
        .arg("-C")
        .arg(root)
        .args(["rev-parse", "--short=10", "HEAD"])
        .stdin(std::process::Stdio::null())
        .output()
        .context("running git rev-parse")?;
    if !out.status.success() {
        bail!(
            "git rev-parse failed — {} is not a git checkout (or git is \
             missing); set METRALE_GATE_SHA to record a gate run",
            root.display()
        );
    }
    let sha = String::from_utf8_lossy(&out.stdout).trim().to_string();
    if sha.is_empty() {
        bail!("git rev-parse printed nothing");
    }
    Ok(sha)
}

/// 2026-09-26: The files under [`PERF_PATHS`] that `git status` reports as modified,
/// staged or untracked in this working tree, sorted and deduplicated. The gate run
/// records this list, since a dirty perf path means the binary was not built from the
/// commit the record names.
///
/// Untracked files are listed one by one (`--untracked-files=all`), not collapsed to
/// their directory. `.benchmarks` is not a perf path, so a record file written by an
/// earlier gate run is not reported.
///
/// # Errors
/// If git cannot run or `git status` fails, rather than reporting a clean tree.
pub fn dirty_perf_paths(root: &Path) -> Result<Vec<String>> {
    let out = std::process::Command::new("git")
        .arg("-C")
        .arg(root)
        .args(["status", "--porcelain", "--untracked-files=all", "--"])
        .args(PERF_PATHS)
        .stdin(std::process::Stdio::null())
        .output()
        .context("running git status")?;
    if !out.status.success() {
        bail!(
            "git status failed in {} — cannot tell whether the measured binary \
             matches the commit being stamped",
            root.display()
        );
    }
    let mut paths: Vec<String> = String::from_utf8_lossy(&out.stdout)
        .lines()
        .filter_map(|line| line.get(3..))
        // 2026-09-26: A rename reads `R  old -> new`; keep the destination, the path
        // that exists in the tree now.
        .map(|entry| match entry.split_once(" -> ") {
            Some((_, dest)) => dest.trim().to_string(),
            None => entry.trim().to_string(),
        })
        .filter(|p| !p.is_empty())
        .collect();
    paths.sort();
    paths.dedup();
    Ok(paths)
}

#[cfg(test)]
#[path = "tests.rs"]
mod tests;

#[cfg(test)]
#[path = "record_contract_tests.rs"]
mod record_contract_tests;

#[cfg(test)]
#[path = "energy_ceiling_tests.rs"]
mod energy_ceiling_tests;

#[cfg(test)]
#[path = "variant_tests.rs"]
mod variant_tests;

#[cfg(test)]
#[path = "fixture_baseline.rs"]
mod fixture_baseline;

#[cfg(test)]
#[path = "coverage_map_tests.rs"]
mod coverage_map_tests;

#[cfg(test)]
#[path = "test_only_coverage_tests.rs"]
mod test_only_coverage_tests;

#[cfg(test)]
#[path = "coverage_promotion_tests.rs"]
mod coverage_promotion_tests;
#[cfg(test)]
#[path = "coverage_tests.rs"]
mod coverage_tests;

#[cfg(test)]
#[path = "coverage_squash_tests.rs"]
mod coverage_squash_tests;

#[cfg(test)]
#[path = "hardening_tests.rs"]
mod hardening_tests;

#[cfg(test)]
#[path = "card_tests.rs"]
mod card_tests;

pub mod agreement;
pub mod group;

#[cfg(test)]
#[path = "group_tests.rs"]
mod group_tests;

#[cfg(test)]
#[path = "group_verdict_tests.rs"]
mod group_verdict_tests;

#[cfg(test)]
#[path = "agreement_tests.rs"]
mod agreement_tests;

#[cfg(test)]
#[path = "signer_registry_tests.rs"]
mod signer_registry_tests;

#[cfg(test)]
#[path = "signing_tests.rs"]
mod signing_tests;

#[cfg(test)]
#[path = "amnesty_tests.rs"]
mod amnesty_tests;
#[cfg(test)]
#[path = "standing_tests.rs"]
mod standing_tests;

#[cfg(test)]
#[path = "dirty_tests.rs"]
mod dirty_tests;

#[cfg(test)]
#[path = "override_tests.rs"]
mod override_tests;
