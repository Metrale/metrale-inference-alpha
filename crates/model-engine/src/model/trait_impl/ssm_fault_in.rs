// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-25: SSM-snapshot tier fault-in for the prefix-cache restore: when the match has
//! no resident snapshot but carries a tier key, read the spilled snapshot back into a
//! resident Marconi slot. `prefill_a`, `prefill_b/prefix_lookup` and `prefill_c` call it.
//!
//! Owner: model-engine.
//! Invariants: a resident snapshot in the match is never replaced by a fault-in.

#![allow(dead_code)]

use metrale_telemetry::prefix_cache::PrefixMatch;

use super::super::types::TransformerModel;

/// 2026-09-25: Tier-snapshot depth in tokens below which the fault-in is skipped and the
/// prefix recomputed, because the blob read costs a fixed amount while the saved SSM
/// recompute grows with depth. `METRALE_SSM_FAULT_MIN_TOKENS` overrides it; `0` disables
/// the gate.
pub(in crate::model) const DEFAULT_FAULT_MIN_TOKENS: usize = 256;

/// 2026-09-25: The fault-in depth gate in effect. `model::ssm_spill_gate` clamps the spill
/// gate to at least this value, so nothing is spilled that this gate would not read back.
pub(in crate::model) fn fault_in_min_tokens() -> usize {
    parse_fault_min_tokens(std::env::var("METRALE_SSM_FAULT_MIN_TOKENS").ok())
}

/// 2026-09-25: Parse `METRALE_SSM_FAULT_MIN_TOKENS`: an unset or unparseable value gives
/// [`DEFAULT_FAULT_MIN_TOKENS`].
fn parse_fault_min_tokens(raw: Option<String>) -> usize {
    raw.and_then(|v| v.parse::<usize>().ok())
        .unwrap_or(DEFAULT_FAULT_MIN_TOKENS)
}

/// 2026-09-25: Whether a tier snapshot of `depth` tokens is below the gate. `min_depth == 0`
/// never skips.
fn should_skip_fault_for_depth(depth: usize, min_depth: usize) -> bool {
    depth < min_depth
}

impl TransformerModel {
    /// 2026-09-25: The effective SSM snapshot for a prefix match, as `(slot, depth)`:
    /// - `slot`: the resident snapshot, else the one
    ///   [`try_fault_in_ssm_snapshot`](Self::try_fault_in_ssm_snapshot) faulted in,
    ///   else `None` (recompute the prefix).
    /// - `depth`: the restored state's token depth. For a faulted-in slot it is
    ///   `ssm_snapshot_tier_tokens`, not `ssm_snapshot_tokens`; callers use it as the
    ///   skip point.
    pub(in crate::model) fn eff_ssm_snapshot(
        &self,
        prefix_match: &PrefixMatch,
        session_hash: u64,
        stream: u64,
    ) -> (Option<usize>, usize) {
        let faulted_snap = self.try_fault_in_ssm_snapshot(prefix_match, session_hash, stream);
        let eff_snapshot = prefix_match.ssm_snapshot.or(faulted_snap);
        let eff_snapshot_tokens = if faulted_snap.is_some() {
            prefix_match.ssm_snapshot_tier_tokens
        } else {
            prefix_match.ssm_snapshot_tokens
        };
        (eff_snapshot, eff_snapshot_tokens)
    }

    /// 2026-09-25: Fault the spilled SSM snapshot for `prefix_match` back into a resident
    /// Marconi slot tagged with `session_hash`. Returns that slot, or `None` when there is
    /// nothing to fault (a resident snapshot is present, no tier store, no tier key), the
    /// depth gate skips it, no slot can be acquired, or the read misses or fails.
    /// The slot's depth is `prefix_match.ssm_snapshot_tier_tokens`.
    pub(in crate::model) fn try_fault_in_ssm_snapshot(
        &self,
        prefix_match: &PrefixMatch,
        session_hash: u64,
        stream: u64,
    ) -> Option<usize> {
        if prefix_match.ssm_snapshot.is_some() {
            return None;
        }
        let store = self.ssm_tier_store.as_deref()?;
        let key = prefix_match.ssm_snapshot_tier_key?;

        let depth = prefix_match.ssm_snapshot_tier_tokens;
        let min_depth = fault_in_min_tokens();
        if should_skip_fault_for_depth(depth, min_depth) {
            tracing::info!(
                "SSM tier fault-in SKIPPED (cost gate): tier snapshot depth {depth} < \
                 METRALE_SSM_FAULT_MIN_TOKENS={min_depth} — recomputing the shallow prefix \
                 is cheaper than a ~28ms blob fault + replay"
            );
            return None;
        }

        // 2026-09-25: The acquire, fault and promote-or-miss cycle is a pool method so
        // CPU-only tests can drive it without building a `TransformerModel`.
        self.ssm_snapshots.fault_in_for_key(
            self.prefix_cache.as_ref(),
            store,
            self.gpu.as_ref(),
            key,
            session_hash,
            depth,
            stream,
        )
    }
}

#[cfg(test)]
mod tests {
    use super::{DEFAULT_FAULT_MIN_TOKENS, parse_fault_min_tokens, should_skip_fault_for_depth};

    #[test]
    fn min_tokens_defaults_when_unset_or_garbage() {
        assert_eq!(parse_fault_min_tokens(None), DEFAULT_FAULT_MIN_TOKENS);
        assert_eq!(
            parse_fault_min_tokens(Some("not-a-number".into())),
            DEFAULT_FAULT_MIN_TOKENS
        );
        assert_eq!(
            parse_fault_min_tokens(Some("".into())),
            DEFAULT_FAULT_MIN_TOKENS
        );
    }

    #[test]
    fn min_tokens_parses_explicit_values() {
        assert_eq!(parse_fault_min_tokens(Some("0".into())), 0);
        assert_eq!(parse_fault_min_tokens(Some("1024".into())), 1024);
    }

    #[test]
    fn gate_skips_shallow_faults_only() {
        let min = DEFAULT_FAULT_MIN_TOKENS;
        assert!(should_skip_fault_for_depth(0, min));
        assert!(should_skip_fault_for_depth(min - 1, min));
        assert!(!should_skip_fault_for_depth(min, min));
        assert!(!should_skip_fault_for_depth(min + 1, min));
        assert!(!should_skip_fault_for_depth(15_000, min));
    }

    #[test]
    fn gate_disabled_never_skips() {
        assert!(!should_skip_fault_for_depth(0, 0));
        assert!(!should_skip_fault_for_depth(1, 0));
    }
}
