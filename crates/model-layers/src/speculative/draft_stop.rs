// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-02: Per-position confidence stop for MTP draft chains
//! (`--draft-confidence-stop <tau>`): the drafter keeps extending a chain only
//! while its last draft's top-1 probability is at least `tau`. The serve
//! publishes `tau` once; the drafter (single and batched propose) and the
//! scheduler's depth planner read it here and apply the one rule
//! [`chain_depth`].
//!
//! Owner: model-layers (speculative).
//! Invariants:
//! - A published `tau` is in `(0, 1)`; a second publication of a different
//!   value is refused.
//! - [`chain_depth`] never returns more than the drafts it is given, and
//!   returns at least 1 for a non-empty chain: the first draft is always
//!   verified, as without the stop.

static PUBLISHED_TAU: std::sync::OnceLock<f32> = std::sync::OnceLock::new();

/// 2026-10-02: Publish the serve's stop threshold, before the model builds and
/// the scheduler starts.
pub fn set_draft_confidence_stop(tau: f32) -> anyhow::Result<()> {
    anyhow::ensure!(
        tau > 0.0 && tau < 1.0,
        "--draft-confidence-stop must be in (0, 1), got {tau}"
    );
    let got = *PUBLISHED_TAU.get_or_init(|| tau);
    anyhow::ensure!(
        got.to_bits() == tau.to_bits(),
        "draft confidence stop already published as {got}, refusing {tau}"
    );
    Ok(())
}

/// 2026-10-02: The published threshold as a log-probability, `ln tau`; `None`
/// when the serve did not ask for the stop (and in unit tests and tools).
pub fn draft_stop_logprob() -> Option<f32> {
    PUBLISHED_TAU.get().map(|t| t.ln())
}

/// 2026-10-02: How many leading drafts of a chain to verify, from each draft's
/// top-1 log-probability `lps[j]` (in draft order): drafting continues past
/// draft `j` only while `lps[j] >= ln_tau`, so the chain ends with the first
/// draft below the threshold, which is still verified. An empty chain is 0.
pub fn chain_depth(lps: &[f32], ln_tau: f32) -> usize {
    lps.iter()
        .position(|&lp| lp < ln_tau)
        .map_or(lps.len(), |j| j + 1)
}

/// 2026-10-02: Whether a chain whose last draft scored `lp` continues.
pub fn chain_continues(lp: f32, ln_tau: f32) -> bool {
    lp >= ln_tau
}

#[cfg(test)]
mod tests {
    use super::*;

    const LN_HALF: f32 = -std::f32::consts::LN_2;

    #[test]
    fn a_confident_chain_keeps_every_draft() {
        assert_eq!(chain_depth(&[-0.1, -0.2, -0.3], LN_HALF), 3);
    }

    #[test]
    fn the_chain_ends_with_its_first_unconfident_draft() {
        assert_eq!(chain_depth(&[-0.1, -2.0, -0.1], LN_HALF), 2);
        assert_eq!(chain_depth(&[-2.0, -0.1, -0.1], LN_HALF), 1);
    }

    #[test]
    fn the_threshold_itself_continues() {
        assert!(chain_continues(LN_HALF, LN_HALF));
        assert!(!chain_continues(LN_HALF - 1e-6, LN_HALF));
        assert_eq!(chain_depth(&[LN_HALF, LN_HALF], LN_HALF), 2);
    }

    #[test]
    fn an_empty_chain_is_zero_and_a_lone_draft_is_one() {
        assert_eq!(chain_depth(&[], LN_HALF), 0);
        assert_eq!(chain_depth(&[-9.0], LN_HALF), 1);
    }

    #[test]
    fn depth_agrees_with_continuation_at_every_position() {
        // 2026-10-02: The single-sequence drafter stops with `chain_continues`,
        // the batched planner truncates with `chain_depth`: the same rule.
        let lps = [-0.05, -0.4, -0.3, -1.5, -0.01];
        let ln_tau = (0.6f32).ln();
        let mut drafted = 0;
        for &lp in &lps {
            drafted += 1;
            if !chain_continues(lp, ln_tau) {
                break;
            }
        }
        assert_eq!(drafted, chain_depth(&lps, ln_tau));
    }

    #[test]
    fn publication_refuses_out_of_range_values() {
        for bad in [0.0, 1.0, -0.5, 1.5, f32::NAN] {
            assert!(set_draft_confidence_stop(bad).is_err(), "{bad} accepted");
        }
    }
}
