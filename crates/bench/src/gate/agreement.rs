// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-26: Do the records a PR adds agree with each other and with the
//! head being certified?
//!
//! Each record is bound to its commit by its signature. `check` adds two
//! rules over the whole added set:
//!
//! - Every record must stand at the head (`Standing::Stands`, from
//!   `check::record_standing`): the diff from the record's commit to the head
//!   touches nothing its gate measures, or the record's closure excuses what
//!   it touches. The rule is content, not ancestry, so records measured at
//!   different commits may be added together.
//! - Speed-class records (`Sensitivity::Speed`, read from the registry) from
//!   more than one signer are accepted only when every cross-signer pair is one
//!   box by `hardware::equivalence::equivalent`, judged from the
//!   records' own `hardware` and `hardware_state` captures under the class's
//!   `kernels/<hw>/HARDWARE.toml` limits. Correctness-class records may come
//!   from any number of signers.
//!
//! The one-box rule for Speed rests on a measurement taken 2026-09-06 on one
//! gate at one commit: dgx2 22.78 tok/s (sigma 0.063, n=10) against dgx3 23.44
//! (sigma 0.070, n=10), a gap ten times either box's sigma.
//!
//! Whether a signer is committed in `.github/record-signers/` is checked by
//! `gate::signing`, not here.
//!
//! Owner: bench gate.
//! Invariants:
//! - A benchmark id the registry does not know is reported as
//!   `Disagreement::UnknownBenchmark`, never given a class.
//! - `check` returns every disagreement it finds, not only the first.

use super::check::Standing;
use super::coverage;
use crate::hardware::equivalence::{EquivalencePolicy, HardwareFingerprint, equivalent};
use crate::hardware::policy::Sensitivity;
use crate::registry;

/// 2026-09-26: One record as the agreement rule sees it.
#[derive(Debug, Clone, PartialEq)]
pub struct AddedRecord {
    /// 2026-09-26: Path, for messages only; never parsed.
    pub path: String,
    pub benchmark_id: String,
    pub git_sha: String,
    /// 2026-09-26: The signing key fingerprint from the `.sig` sidecar.
    pub signer: String,
    /// 2026-09-26: What the record says about the box it was measured on.
    /// `None` makes the record equivalent to nothing in [`check`].
    pub hardware: Option<HardwareFingerprint>,
    /// 2026-09-26: The box class the record names (`Hardware::gate_key`). In
    /// a cross-signer pair, the first record's class selects the equivalence
    /// policy.
    pub hardware_class: String,
    /// 2026-09-26: Where the record stands at the head being certified,
    /// computed by the caller, e.g. with [`standing_at`].
    pub standing: Standing,
}

/// 2026-09-26: Why a set of added records does not hang together.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Disagreement {
    /// 2026-09-26: A record whose standing is not `Stands`: its commit cannot
    /// be diffed against the head, or the diff touches what its gate
    /// measures. Reported for every class.
    Straggler {
        path: String,
        git_sha: String,
        why: String,
    },
    /// 2026-09-26: Speed-class records signed by more than one identity, on
    /// boxes the records themselves do not show to be equivalent.
    SpeedSigners {
        /// 2026-09-26: The gates in a non-equivalent pair, sorted and
        /// deduplicated.
        gates: Vec<String>,
        signers: Vec<String>,
        /// 2026-09-26: Why the boxes are not one box, one entry per offending
        /// pair.
        mismatches: Vec<String>,
    },
    /// 2026-09-26: A record naming a benchmark the registry does not have.
    UnknownBenchmark(String),
}

impl std::fmt::Display for Disagreement {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Straggler { path, git_sha, why } => write!(
                f,
                "{path} was measured at {git_sha} and does not stand at the head: {why}. \
                 Re-measure it at the head you intend to merge."
            ),
            Self::SpeedSigners {
                gates,
                signers,
                mismatches,
            } => write!(
                f,
                "{} speed-class gate(s) ({}) carry {} different signing keys ({}) \
                 and the records do not show the boxes to be equivalent: {}. \
                 Throughput and latency are box-dependent — measured 0.66 tok/s \
                 between two boxes against a within-box sigma of 0.07 — so these \
                 must come from ONE box, or from boxes whose captures agree on \
                 GPU, driver line, clock ceiling, memory and thermal state. \
                 Correctness-class gates may span boxes freely.",
                gates.len(),
                gates.join(", "),
                signers.len(),
                signers.join(", "),
                mismatches.join("; ")
            ),
            Self::UnknownBenchmark(id) => write!(
                f,
                "record names benchmark {id:?}, which is not in the registry — \
                 refusing to classify it. An unrecognised gate must not inherit \
                 the permissive rule."
            ),
        }
    }
}

/// 2026-09-26: A record's [`Standing`] at `head`, judged against the coverage
/// entry of its `benchmark_id`. An id with no coverage entry is
/// [`Standing::Unknown`].
pub fn standing_at(root: &std::path::Path, head: &str, record: &super::GateRecord) -> Standing {
    match coverage::find(&record.benchmark_id) {
        Some(gate) => super::check::record_standing(root, head, record, gate),
        None => Standing::Unknown,
    }
}

fn policy_for(root: &std::path::Path, class: &str) -> anyhow::Result<Option<EquivalencePolicy>> {
    EquivalencePolicy::speed_for(root, class)
}

/// 2026-09-26: The class a gate's records belong to, read from the registry
/// and never from the record, so a record cannot choose its own class.
pub fn sensitivity_of(benchmark_id: &str) -> Option<Sensitivity> {
    registry::find(benchmark_id).map(|d| d.sensitivity)
}

/// 2026-09-26: Check that the records a PR adds agree with one another.
///
/// Returns every disagreement found. `root` is where the hardware class's
/// equivalence policy is read from (`kernels/<hw>/HARDWARE.toml`); a class
/// with no `[benchmarks.limits]` makes every cross-signer Speed pair a
/// mismatch.
pub fn check(root: &std::path::Path, added: &[AddedRecord]) -> Vec<Disagreement> {
    let mut out = Vec::new();
    if added.is_empty() {
        return out;
    }

    for r in added {
        let why = match &r.standing {
            Standing::Stands => continue,
            Standing::Unknown => {
                "its commit cannot be diffed against the head (unknown to this repository, \
                 or git failed)"
                    .to_string()
            }
            Standing::Invalidated(paths) => format!(
                "commits since it touched what its gate measures ({})",
                paths.join(", ")
            ),
        };
        out.push(Disagreement::Straggler {
            path: r.path.clone(),
            git_sha: r.git_sha.clone(),
            why,
        });
    }

    let speed: Vec<&AddedRecord> = added
        .iter()
        .filter(|r| match sensitivity_of(&r.benchmark_id) {
            None => {
                out.push(Disagreement::UnknownBenchmark(r.benchmark_id.clone()));
                false
            }
            Some(Sensitivity::Speed) => true,
            Some(Sensitivity::Correctness) => false,
        })
        .collect();
    let mut speed_signers: Vec<String> = speed.iter().map(|r| r.signer.clone()).collect();
    speed_signers.sort();
    speed_signers.dedup();
    if speed_signers.len() > 1 {
        // 2026-09-26: Every cross-signer pair must be one box by the records'
        // own captures.
        let mut mismatches = Vec::new();
        let mut gates = Vec::new();
        for (i, a) in speed.iter().enumerate() {
            for b in &speed[i + 1..] {
                if a.signer == b.signer {
                    continue;
                }
                let why = match (&a.hardware, &b.hardware) {
                    (Some(x), Some(y)) => match policy_for(root, &a.hardware_class) {
                        Ok(Some(policy)) => match equivalent(x, y, &policy) {
                            Ok(()) => continue,
                            Err(m) => m
                                .iter()
                                .map(ToString::to_string)
                                .collect::<Vec<_>>()
                                .join(", "),
                        },
                        Ok(None) => format!(
                            "kernels/{}/HARDWARE.toml declares no [benchmarks.limits.thermal] \
                             envelope, so two boxes of that class are never one box",
                            a.hardware_class
                        ),
                        Err(e) => format!("{e:#}"),
                    },
                    _ => "a record carries no hardware capture".to_owned(),
                };
                mismatches.push(format!(
                    "{} ({}) vs {} ({}): {why}",
                    a.benchmark_id,
                    short(&a.signer),
                    b.benchmark_id,
                    short(&b.signer)
                ));
                gates.push(a.benchmark_id.clone());
                gates.push(b.benchmark_id.clone());
            }
        }
        if !mismatches.is_empty() {
            gates.sort();
            gates.dedup();
            out.push(Disagreement::SpeedSigners {
                gates,
                signers: speed_signers,
                mismatches,
            });
        }
    }
    out
}

fn short(signer: &str) -> &str {
    &signer[..signer.len().min(12)]
}

/// 2026-09-26: Every required gate, split into (Speed, Correctness) by the
/// class that decides its signer rule. A gate the registry does not know is in
/// neither list.
pub fn required_by_class() -> (Vec<&'static str>, Vec<&'static str>) {
    let mut speed = Vec::new();
    let mut correctness = Vec::new();
    for g in coverage::REQUIRED.iter() {
        match sensitivity_of(g.id) {
            Some(Sensitivity::Speed) => speed.push(g.id),
            Some(Sensitivity::Correctness) => correctness.push(g.id),
            None => {}
        }
    }
    (speed, correctness)
}
