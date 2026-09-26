// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-26: The campaign's last word: the gate table, then the agreement check over the records a commit would add.
//!
//! Owner: server CLI (`met benchmark certify`).
//! The table comes from `gate::check_gates` and is printed by the same
//! `print_statuses` as `--pull-request-gate-check`.
//! Invariants: `Final::certified` holds only when no required gate is open and
//! `agreement::check` found no disagreement.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use anyhow::{Context, Result};
use metrale_bench::gate::agreement::{self, AddedRecord, Disagreement};
use metrale_bench::gate::{self, GateStatus};

/// 2026-09-26: Every untracked, not-ignored `.benchmarks/**/*.json`: the records a commit
/// would add, whichever campaign wrote them.
pub fn untracked_records(root: &Path) -> Result<Vec<PathBuf>> {
    let out = std::process::Command::new("git")
        .arg("-C")
        .arg(root)
        .args([
            "ls-files",
            "--others",
            "--exclude-standard",
            "--",
            ".benchmarks",
        ])
        .stdin(std::process::Stdio::null())
        .output()
        .context("listing untracked records")?;
    if !out.status.success() {
        anyhow::bail!(
            "git ls-files failed: {}",
            String::from_utf8_lossy(&out.stderr).trim()
        );
    }
    Ok(String::from_utf8_lossy(&out.stdout)
        .lines()
        .filter(|l| l.ends_with(".json"))
        .map(|l| root.join(l))
        .collect())
}

/// 2026-09-26: The `key` a record's `.sig` sidecar names, or `NO_SIDECAR` when the
/// sidecar is missing or unreadable.
pub fn signer_of(record: &Path) -> String {
    std::fs::read_to_string(gate::signing::sig_path(record))
        .ok()
        .and_then(|t| serde_json::from_str::<serde_json::Value>(&t).ok())
        .and_then(|v| v.get("key")?.as_str().map(str::to_owned))
        .unwrap_or_else(|| "NO_SIDECAR".into())
}

/// 2026-09-26: Read the added records into the shape the agreement rule judges, each
/// with its standing at `anchor`. An unreadable record is skipped.
pub fn added_records(root: &Path, anchor: &str, paths: &[PathBuf]) -> Vec<AddedRecord> {
    paths
        .iter()
        .filter_map(|p| {
            let r = gate::read_record(p).ok()?;
            Some(AddedRecord {
                path: p.display().to_string(),
                hardware: Some(
                    metrale_bench::hardware::equivalence::HardwareFingerprint::from_record(&r),
                ),
                hardware_class: r.hardware.gate_key(),
                standing: agreement::standing_at(root, anchor, &r),
                benchmark_id: r.benchmark_id,
                git_sha: r.git_sha,
                signer: signer_of(p),
            })
        })
        .collect()
}

pub struct Final {
    pub statuses: BTreeMap<String, GateStatus>,
    pub open: Vec<&'static str>,
    pub added: Vec<AddedRecord>,
    pub disagreements: Vec<Disagreement>,
}

impl Final {
    pub fn certified(&self) -> bool {
        self.open.is_empty() && self.disagreements.is_empty()
    }
}

/// 2026-09-26: Evaluate without printing; the driver prints in its own format.
pub fn evaluate(root: &Path, anchor: &str) -> Result<Final> {
    let statuses = gate::check_gates(root, anchor);
    let open: Vec<&'static str> = gate::REQUIRED_GATES
        .iter()
        .copied()
        .filter(|id| !matches!(statuses.get(*id), Some(GateStatus::Pass)))
        .collect();
    let added = added_records(root, anchor, &untracked_records(root)?);
    let disagreements = agreement::check(root, &added);
    Ok(Final {
        statuses,
        open,
        added,
        disagreements,
    })
}

/// 2026-09-26: The human rendering of the verdict.
pub fn print(f: &Final, anchor: &str, root: &Path) {
    println!();
    println!("gate check for {anchor} ({})", root.display());
    let open = super::super::bench_gate_check::print_statuses(&f.statuses);
    debug_assert_eq!(open, f.open);
    println!();
    println!("{} record(s) would be added by a commit:", f.added.len());
    for a in &f.added {
        println!(
            "  {:<60} gate={} sha={} signer={}",
            a.path, a.benchmark_id, a.git_sha, a.signer
        );
    }
    for d in &f.disagreements {
        println!("  DISAGREE  {d}");
    }
    println!();
    if f.certified() {
        println!(
            "CERTIFIED at {anchor}: all {} required gates pass and the added records agree.",
            gate::REQUIRED_GATES.len()
        );
    } else {
        println!(
            "NOT CERTIFIED at {anchor}: {} gate(s) open{}",
            f.open.len(),
            if f.disagreements.is_empty() {
                String::new()
            } else {
                format!(", {} disagreement(s)", f.disagreements.len())
            }
        );
    }
}
