// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-26: [`TierEvict`]: what an SSM spill-tier eviction did with its
//! victim.
//!
//! Owner: telemetry.
//! Invariants: none beyond the types.

/// 2026-09-26: Outcome of
/// [`crate::prefix_cache::PrefixCache::evict_snapshot_to_tier`].
///
/// The depth gate (`min_tokens`, which the model engine derives from
/// `METRALE_SSM_SPILL_MIN_TOKENS`) is applied when the victim is chosen, so a
/// shallow victim's entry is removed rather than left findable with no stored
/// bytes. Either arm returns the victim's slot for the caller to free.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TierEvict {
    /// 2026-09-26: The entry is marked spilled and stays findable; the caller
    /// moves the bytes to the tier.
    Spill {
        /// 2026-09-26: The victim's snapshot slot: spill from it, then free it.
        slot: usize,
        /// 2026-09-26: Tier key: the entry's prefix hash.
        key: u64,
        /// 2026-09-26: Victim depth in tokens.
        depth: usize,
    },
    /// 2026-09-26: Shallower than `min_tokens`: the entry was removed, and
    /// `tier_spills` was not incremented.
    Drop {
        /// 2026-09-26: The victim's snapshot slot, for the caller to free.
        slot: usize,
        /// 2026-09-26: Victim depth in tokens.
        depth: usize,
    },
}

impl TierEvict {
    /// 2026-09-26: The victim's slot, whichever arm was taken.
    pub fn slot(&self) -> usize {
        match *self {
            TierEvict::Spill { slot, .. } | TierEvict::Drop { slot, .. } => slot,
        }
    }
}
