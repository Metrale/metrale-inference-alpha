// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-02: The prompt-lookup copy tier of the SSM verify pools
//! (`--prompt-lookup-decoding`): the first `slots` verify slots hold at least
//! `drafts` h intermediates, so a copy longer than the MTP chain can be
//! verified there. The serve publishes it once, before the preflight reserve
//! and the pool allocation, which both read it through [`pool_counts`]
//! (`super::pool_counts`); neither can then size the pool differently.
//!
//! Owner: model-layers (SSM reserve).
//! Invariants:
//! - A published tier has `slots >= 1` and `drafts >= 1`; a second publication
//!   of a different tier is refused.
//! - [`tier_h`] never lowers a slot's count, and raises only slots below
//!   `slots`.

/// 2026-10-02: Copy-tier geometry.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CopyTier {
    /// 2026-10-02: Verify slots `0..slots` get the copy depth.
    pub slots: usize,
    /// 2026-10-02: Drafts those slots can verify (h intermediates).
    pub drafts: usize,
}

static PUBLISHED: std::sync::OnceLock<CopyTier> = std::sync::OnceLock::new();

/// 2026-10-02: Publish the copy tier, before the preflight reserve.
pub fn set_copy_tier(tier: CopyTier) -> anyhow::Result<()> {
    anyhow::ensure!(
        tier.slots >= 1 && tier.drafts >= 1,
        "copy tier needs at least one slot and one draft, got {tier:?}"
    );
    let got = *PUBLISHED.get_or_init(|| tier);
    anyhow::ensure!(
        got == tier,
        "copy tier already published as {got:?}, refusing {tier:?}"
    );
    Ok(())
}

/// 2026-10-02: The published tier; `None` without prompt lookup.
pub fn copy_tier() -> Option<CopyTier> {
    PUBLISHED.get().copied()
}

/// 2026-10-02: Slot `s`'s h intermediates under `tier`: `base` raised to the
/// copy depth on a copy slot.
pub fn tier_h(base: usize, s: usize, tier: Option<CopyTier>) -> usize {
    match tier {
        Some(t) if s < t.slots => base.max(t.drafts),
        _ => base,
    }
}

/// 2026-10-02: Rows of the widest verify the pools must hold: the MTP `k`
/// (drafts + 1), widened to a copy's.
pub fn tier_rows(k: usize, tier: Option<CopyTier>) -> usize {
    tier.map_or(k, |t| k.max(t.drafts + 1))
}

#[cfg(test)]
mod tests {
    use super::*;

    const T: CopyTier = CopyTier {
        slots: 8,
        drafts: 8,
    };

    #[test]
    fn only_copy_slots_are_raised_and_never_lowered() {
        assert_eq!(tier_h(3, 0, Some(T)), 8);
        assert_eq!(tier_h(3, 7, Some(T)), 8);
        assert_eq!(tier_h(3, 8, Some(T)), 3, "slot 8 is outside the tier");
        assert_eq!(
            tier_h(12, 0, Some(T)),
            12,
            "a deeper ladder slot keeps its depth"
        );
        assert_eq!(tier_h(3, 0, None), 3);
    }

    #[test]
    fn rows_widen_to_the_copy() {
        assert_eq!(tier_rows(4, Some(T)), 9);
        assert_eq!(tier_rows(17, Some(T)), 17);
        assert_eq!(tier_rows(2, None), 2);
    }

    #[test]
    fn an_empty_tier_is_refused() {
        assert!(
            set_copy_tier(CopyTier {
                slots: 0,
                drafts: 4
            })
            .is_err()
        );
        assert!(
            set_copy_tier(CopyTier {
                slots: 4,
                drafts: 0
            })
            .is_err()
        );
    }
}
