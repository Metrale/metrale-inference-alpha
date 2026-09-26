// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-25: Draft-count limits for speculative dispatch: the verify-slot
//! capacity of every active sequence, and the serial path's verify arm.
//!
//! Owner: speculative.
//! Invariants:
//! - `clamp_drafts_to_slot_capacity` returns no more than `drafts` and no
//!   more than any capacity it is given.
//! - `serial_verify_plan` keeps at most `step_drafts` of an MTP sequence's
//!   pending drafts.
//!
//! The verify pools size each slot for the deepest draft count the static
//! ladder can give a sequence in it (`ssm_reserve::verify_slot_drafts`).
//! With the default ladder `4:3,8:3,16:1,32:1` and `--num-drafts 3`, slots
//! 0..8 hold 3 drafts and slots from 8 on hold 1. A step can ask for more:
//! a sequence can sit on a high slot while `n_active` is small, and
//! `adaptive_rung` can raise n in 9..=16 to 2 drafts. A step's draft count is
//! therefore clamped to the smallest capacity among its active sequences,
//! read from the model's pools (`Model::mtp_slot_draft_capacity`).
//! `METRALE_MTP_POOL_FULL_WIDTH` gives every slot `num_drafts`, which makes
//! the clamp a no-op. With the tiered default the adaptive lift never
//! survives the clamp: nine or more active sequences hold nine distinct
//! slots, one of them at index 8 or above, whose capacity is 1.

/// 2026-09-25: Clamps a step's draft count to the smallest verify-slot
/// capacity among the active sequences. `usize::MAX` (the `Model` default,
/// for a model without SSM verify pools) does not clamp; an empty iterator
/// leaves `drafts` unchanged.
pub fn clamp_drafts_to_slot_capacity(
    drafts: usize,
    slot_capacities: impl IntoIterator<Item = usize>,
) -> usize {
    slot_capacities
        .into_iter()
        .fold(drafts, |acc, cap| acc.min(cap))
}

/// 2026-09-25: The verify arm the serial (per-sequence) path runs for one
/// sequence.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SerialArm {
    DFlash,
    K4,
    K3,
    K2,
}

/// 2026-09-25: How many of a sequence's `pending` drafts the serial path
/// keeps, and which arm verifies them. An MTP sequence can carry more
/// pending drafts than this step's width `step_drafts` (the ladder count
/// after [`clamp_drafts_to_slot_capacity`]); the surplus is dropped so the
/// verify never exceeds the slot. With `dflash` every pending draft is kept.
/// Four or more kept drafts run the DFlash arm; fewer run K4, K3 or K2 by the
/// kept count, each also bounded by `step_drafts`.
pub fn serial_verify_plan(pending: usize, step_drafts: usize, dflash: bool) -> (usize, SerialArm) {
    let keep = if dflash {
        pending
    } else {
        pending.min(step_drafts)
    };
    let arm = if keep >= 4 {
        SerialArm::DFlash
    } else if step_drafts >= 3 && keep >= 3 {
        SerialArm::K4
    } else if step_drafts >= 2 && keep >= 2 {
        SerialArm::K3
    } else {
        SerialArm::K2
    };
    (keep, arm)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_low_capacity_slot_bounds_the_whole_step() {
        // 2026-09-25: One capacity-1 slot limits the whole step to one draft.
        assert_eq!(clamp_drafts_to_slot_capacity(3, [3, 3, 1]), 1);
        // 2026-09-25: One sequence on a high slot at small n.
        assert_eq!(clamp_drafts_to_slot_capacity(3, [1]), 1);
        // 2026-09-25: The adaptive-rung lift (2 drafts at n in 9..=16) is
        // clamped by a capacity-1 slot in the batch.
        assert_eq!(clamp_drafts_to_slot_capacity(2, [3, 1, 3]), 1);
    }

    #[test]
    fn full_capacity_slots_do_not_clamp() {
        assert_eq!(clamp_drafts_to_slot_capacity(3, [3, 3, 3]), 3);
        // 2026-09-25: `usize::MAX` (no SSM verify pools) does not clamp.
        assert_eq!(clamp_drafts_to_slot_capacity(3, [usize::MAX; 4]), 3);
        // 2026-09-25: An empty iterator leaves the count unchanged.
        assert_eq!(clamp_drafts_to_slot_capacity(2, std::iter::empty()), 2);
    }

    #[test]
    fn a_ladder_below_num_drafts_never_reaches_the_k4_verify_at_n1() {
        // 2026-09-25: A one-draft ladder (`METRALE_MTP_K_LADDER=1:1`) under
        // `--num-drafts 3`: slot 0 holds one draft, so the step width is 1 and
        // a sequence carrying three or two pending drafts keeps one and runs
        // K2.
        let step = clamp_drafts_to_slot_capacity(1, [1]);
        assert_eq!(serial_verify_plan(3, step, false), (1, SerialArm::K2));
        assert_eq!(serial_verify_plan(2, step, false), (1, SerialArm::K2));
        // 2026-09-25: A two-draft step keeps two of three and runs K3.
        assert_eq!(serial_verify_plan(3, 2, false), (2, SerialArm::K3));
    }

    #[test]
    fn a_full_width_step_dispatches_as_before() {
        assert_eq!(serial_verify_plan(3, 3, false), (3, SerialArm::K4));
        assert_eq!(serial_verify_plan(2, 3, false), (2, SerialArm::K3));
        assert_eq!(serial_verify_plan(1, 3, false), (1, SerialArm::K2));
        assert_eq!(serial_verify_plan(0, 3, false), (0, SerialArm::K2));
        // 2026-09-25: DFlash γ-blocks are never truncated to an MTP width.
        assert_eq!(serial_verify_plan(16, 16, true), (16, SerialArm::DFlash));
        assert_eq!(serial_verify_plan(3, 3, true), (3, SerialArm::K4));
    }
}
