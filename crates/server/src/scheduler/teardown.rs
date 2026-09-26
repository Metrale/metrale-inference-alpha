// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-25: The wait a model release has to do first, in one testable place.
//!
//! Owner: scheduler.
//! Invariants:
//! - `quiesce_streams` attempts a synchronise on every stream it is given, even after one fails, and returns the failures by name instead of raising them.
//!
//! `Model::teardown`'s contract requires the scheduler to have drained and
//! the stream to be synchronised. The sync arrives as a closure so the tests
//! can run without a real `Model`. The order, quiesce then `teardown`, is
//! kept at the one call site, `core/finish.rs`.

use anyhow::Result;

/// 2026-09-25: Block until every stream has finished, returning the ones that would not.
///
/// Failures are returned, not raised, so the caller still releases the
/// model: refusing to free would leak all of it.
pub(super) fn quiesce_streams(
    streams: &[(&'static str, u64)],
    mut sync: impl FnMut(u64) -> Result<()>,
) -> Vec<&'static str> {
    streams
        .iter()
        .filter(|(_, stream)| sync(*stream).is_err())
        .map(|(name, _)| *name)
        .collect()
}

#[cfg(test)]
#[path = "teardown_tests.rs"]
mod tests;
