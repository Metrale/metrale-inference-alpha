// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-25: Unit tests for `RadixTree`, and the helper that arms the sub-block tail arms on one tree.
//!
//! Owner: cache.
//! Invariants: none beyond the types.

mod adapter;
mod basic;
mod partial_tail;
mod snapshot;
mod snapshot_reap;

/// 2026-09-25: Turn on the sub-block tail arms of `walk` on this tree only.
///
/// `RadixTree::new` copies `METRALE_PREFIX_SUBBLOCK` into
/// `partial_tail_sharing` (off when unset), and the env read is cached for the
/// process, so a test sets the field on its own tree instead. `walk` reads the
/// same field in production.
pub(super) fn arm_legacy_partial_tail(tree: &super::RadixTree) {
    tree.inner.lock().partial_tail_sharing = true;
}
