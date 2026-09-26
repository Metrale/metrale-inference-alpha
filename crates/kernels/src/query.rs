// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-25: Whole-registry lookups over the compiled targets: enumerate
//! them, or find one by a substring of its model name. [`super::resolve`]
//! picks the target for a checkpoint.
//!
//! Owner: kernels crate.
//! Invariants: none beyond the types.

use super::{TargetPtxSet, all_ptx_sets};

/// 2026-09-25: Every compiled kernel target, one entry per target
/// (`all_ptx_sets()`); empty under the `METRALE_SKIP_BUILD` stub.
pub fn available_targets() -> Vec<TargetPtxSet> {
    all_ptx_sets()
}

/// 2026-09-25: The first compiled target whose model name contains
/// `needle` (case-sensitive), or `None`.
pub fn ptx_for_model(needle: &str) -> Option<TargetPtxSet> {
    all_ptx_sets()
        .into_iter()
        .find(|t| t.target.model.contains(needle))
}
