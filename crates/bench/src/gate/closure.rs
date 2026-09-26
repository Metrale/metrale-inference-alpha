// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-26: Excuse a kernel-only diff when every affected target still hashes to the closure
//! its record attests.
//!
//! A record stores, per target, the closure hash and the arch, compiler and nvcc flags it was
//! computed under. The check recomputes with the record's own stored arch, compiler and
//! flags, so only a change to the target's sources, the headers they include or its config
//! files (`taxon::configs`) moves the hash, not the toolchain of the machine running CI.
//!
//! Owner: bench gate.
//! Invariants:
//! - [`excuses`] only narrows the path boundary in [`super::coverage`]: it returns `false`
//!   unless every path is under `kernels/`.
//! - It returns `false` for an empty path list or attestation, a tree the resolver cannot
//!   enumerate, paths that reach no target, and any target that is not attested, whose
//!   sources do not resolve ([`super::taxon::sources`] is `None`), whose hash cannot be
//!   computed, or whose hash changed.

use std::collections::BTreeMap;
use std::path::Path;

use serde::{Deserialize, Serialize};

use super::taxon::{self, Target};

/// 2026-09-26: What one target compiled to, and under what. The flags are stored per target
/// because a model's `KERNEL.toml` can add `extra_nvcc_flags`.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct TargetClosure {
    /// 2026-09-26: Hex sha256 from [`metrale_closure::hash`].
    pub hash: String,
    pub arch: String,
    pub compiler: String,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub flags: Vec<String>,
}

impl TargetClosure {
    fn inputs(&self, root: &Path, target: &Target) -> Option<metrale_closure::ClosureInputs> {
        Some(metrale_closure::ClosureInputs {
            sources: taxon::sources(root, target)?,
            configs: taxon::configs(root, target),
            flags: self.flags.clone(),
            arch: self.arch.clone(),
            compiler: self.compiler.clone(),
        })
    }
}

/// 2026-09-26: Per-target attestations carried by a record, keyed by `hw/model/quant`.
pub type Attestation = BTreeMap<String, TargetClosure>;

/// 2026-09-26: Compute an attestation for every target from the working tree.
///
/// A record carries instead the attestation baked into the measuring binary
/// (`metrale_kernels::TARGET_CLOSURES`, attached by `GateRecord::with_closure`), because the
/// tree and the binary can differ. This tree-side version builds test fixtures and is what
/// `crates/server/tests/closure_attestation.rs` compares the baked values against. A target
/// whose sources do not resolve, or whose hash fails, is omitted.
pub fn attest(
    root: &Path,
    arch: &str,
    compiler: &str,
    flags: &BTreeMap<String, Vec<String>>,
) -> Attestation {
    let mut out = BTreeMap::new();
    for target in taxon::walk(root) {
        let key = target.to_string();
        let entry = TargetClosure {
            hash: String::new(),
            arch: arch.to_string(),
            compiler: compiler.to_string(),
            flags: flags.get(&key).cloned().unwrap_or_default(),
        };
        let Some(inputs) = entry.inputs(root, &target) else {
            continue;
        };
        let Ok(hash) = metrale_closure::hash(root, &inputs) else {
            continue;
        };
        out.insert(key, TargetClosure { hash, ..entry });
    }
    out
}

/// 2026-09-26: Whether an unchanged closure excuses every one of `paths`, the paths that
/// survived the path boundary. `true` only when all are under `kernels/` and every target
/// they affect still hashes to what the record attests.
pub fn excuses(root: &Path, paths: &[String], attestation: &Attestation) -> bool {
    if paths.is_empty() || attestation.is_empty() {
        return false;
    }
    // 2026-09-26: A device-code hash says nothing about a path outside `kernels/`.
    if !paths.iter().all(|p| taxon::hardware_of(p).is_some()) {
        return false;
    }
    if !taxon::resolves(root) {
        return false;
    }
    let affected = taxon::affected(root, paths);
    if affected.is_empty() {
        return false;
    }
    affected.iter().all(|target| {
        attestation
            .get(&target.to_string())
            .and_then(|recorded| {
                let inputs = recorded.inputs(root, target)?;
                let current = metrale_closure::hash(root, &inputs).ok()?;
                Some(current == recorded.hash)
            })
            .unwrap_or(false)
    })
}

/// 2026-09-26: The targets `paths` affects whose closure changed, for reporting. A target that
/// is not attested, or whose hash cannot be recomputed, is listed as changed.
pub fn changed_targets(root: &Path, paths: &[String], attestation: &Attestation) -> Vec<String> {
    taxon::affected(root, paths)
        .into_iter()
        .filter(|target| {
            !attestation
                .get(&target.to_string())
                .and_then(|recorded| {
                    let inputs = recorded.inputs(root, target)?;
                    let current = metrale_closure::hash(root, &inputs).ok()?;
                    Some(current == recorded.hash)
                })
                .unwrap_or(false)
        })
        .map(|t| t.to_string())
        .collect()
}

#[cfg(test)]
#[path = "closure_tests.rs"]
mod closure_tests;
