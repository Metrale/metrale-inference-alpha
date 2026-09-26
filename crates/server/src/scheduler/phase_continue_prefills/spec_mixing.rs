// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-25: Whether a speculative step blocks fusing a prefill chunk
//! into decode.
//!
//! Owner: scheduler.
//! Invariants:
//! - `mixing_blocked_by_spec` blocks fusing only at `n_active == 1`,
//!   whatever `mtp_max_seqs` is.
//!
//! Callers: `phase_continue_prefills` (`single_active_with_spec`, for the
//! always-mixed and batched-mixed gates) and `run_standard`
//! (`spec_step_this_tick`).
//!
//! The n-gram and self-speculative lanes need `active.len() == 1`, but the
//! MTP lane can run whenever `active.len() <= mtp_max_seqs`
//! (`spec_width_ok`, core/lane_decode.rs; default 32,
//! `speculative/ladder.rs`). So for `2..=mtp_max_seqs` active sequences this
//! predicate allows fusing on a tick where an MTP step could have run. The
//! fused step in `run_standard` then clears every sequence's
//! `pending_drafts` and `pending_draft_conf`, emits one plain token per
//! sequence through `mixed_forward`, and sets `did_mixed_step`, which makes
//! the scheduler skip its decode lane, `step_mtp` included, for that tick.
//!
//! Widening the predicate to the MTP cap would instead run those prefill
//! chunks as separate forwards from the decode. The test below pins the
//! widths where the two gates disagree.

/// 2026-09-25: True when speculation is on and exactly one sequence is
/// active; callers then do not fuse a prefill chunk into this tick's
/// decode. See the module doc for the widths where this disagrees with the
/// MTP dispatch width gate.
pub(super) fn mixing_blocked_by_spec(n_active: usize, any_spec: bool) -> bool {
    any_spec && n_active == 1
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn without_speculation_mixing_is_never_blocked() {
        for n in 0..40usize {
            assert!(!mixing_blocked_by_spec(n, false), "n={n}");
        }
    }

    #[test]
    fn the_c1_rule_blocks_exactly_one_active_sequence() {
        assert!(!mixing_blocked_by_spec(0, true));
        assert!(mixing_blocked_by_spec(1, true));
        for n in 2..40usize {
            assert!(!mixing_blocked_by_spec(n, true), "n={n}");
        }
    }

    /// 2026-09-25: Pins the widths where this predicate and the MTP dispatch
    /// width gate disagree: `2..=mtp_max_seqs()`.
    #[test]
    fn divergence_from_the_real_dispatch_gate_is_exactly_two_through_the_cap() {
        // 2026-09-25: Either variable changes the cap, so the test checks
        // only the default.
        if std::env::var_os("METRALE_MTP_MAX_SEQS").is_some()
            || std::env::var_os("METRALE_NO_MTP_K_LADDER").is_some()
        {
            return;
        }
        let cap = metrale_model_layers::speculative::mtp_max_seqs();
        // 2026-09-25: `spec_width_ok` (core/lane_decode.rs) allows MTP at
        // `active.len() <= cap`.
        let dispatch_would_run = |n: usize| n >= 1 && n <= cap;
        let diverges: Vec<usize> = (0..cap + 8)
            .filter(|&n| dispatch_would_run(n) != mixing_blocked_by_spec(n, true))
            .collect();
        let expected: Vec<usize> = (2..=cap).collect();
        assert_eq!(
            diverges, expected,
            "the widths where mixing is allowed but a spec step would have run \
             are no longer 2..={cap}; re-read this module's doc before changing it"
        );
        // 2026-09-25: With the default cap the range is non-empty, which is
        // what the module doc describes.
        assert!(
            cap >= 32,
            "dispatch cap fell to {cap}; divergence range shrank"
        );
    }
}
