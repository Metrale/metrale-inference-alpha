// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-25: The snapshot lookups' session gate, read by both
//! `snapshot::lookup` and `snapshot_tier::lookup_tiered`.
//!
//! Owner: cache.
//! Invariants: none beyond the types.

use super::snapshot::SnapshotEntry;

/// 2026-09-25: Whether the session gate rejects `entry` for a lookup by
/// `session_hash`. It rejects a tail entry (every entry, when `hermetic`)
/// unless the lookup has a nonzero session equal to the entry's.
///
/// Non-tail entries pass without a session check. Gating them would miss
/// warm turns: `session_hash` hashes the first `min(len, 1024)` prompt tokens
/// (the server's `compute_session_hash`), so it changes between turns while a
/// conversation is shorter than that. The cost is that a request can restore
/// state computed by another request.
///
/// `--hermetic` gates every entry, so a known-answer test never restores
/// another request's state. It is a parameter rather than a read of
/// `hermetic_enabled()`, which is cached in a `OnceLock`, so a test can reach
/// both answers in one process.
pub(super) fn session_gate_blocks(
    entry: &SnapshotEntry,
    session_hash: u64,
    hermetic: bool,
) -> bool {
    if !entry.is_tail && !hermetic {
        return false;
    }
    session_hash == 0 || entry.session_hash != session_hash
}

#[cfg(test)]
#[path = "tests/snapshot_session_gate.rs"]
mod snapshot_session_gate;
