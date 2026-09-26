// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-25: Tests for [`super`] (`mtp_step`), included by `mtp_step.rs`
//! with `#[path]`: the DFlash batched-verify dispatch order
//! (`sort_batch_by_slot`).
//!
//! Owner: scheduler.
//! Invariants: none beyond the types.

use super::sort_batch_by_slot;

#[test]
fn batched_verify_dispatch_is_slot_order_not_arrival_order() {
    // 2026-09-25: Three sequences arriving with ssm slots [3, 1, 2]
    // (`batchable_idxs` follows `verify_idxs` order); dispatch must be by
    // slot.
    let slots = [Some(3usize), Some(1), Some(2)];
    let mut by_slot = vec![0usize, 1, 2];
    sort_batch_by_slot(&mut by_slot, |i| slots[i]);
    let dispatched: Vec<usize> = by_slot.iter().map(|&i| slots[i].unwrap()).collect();
    assert_eq!(
        dispatched,
        vec![1, 2, 3],
        "the batch must be dispatched in ascending ssm-slot order — arrival \
         order fails the consecutive-slot pointer check and declines the \
         batched verify"
    );
}

#[test]
fn slotless_sequences_sort_last_and_index_breaks_ties() {
    // 2026-09-25: Slotless sequences sort after every slotted one, and two
    // slotless entries keep their active-index order.
    let slots = [None, Some(5usize), None, Some(0)];
    let mut by_slot = vec![0usize, 1, 2, 3];
    sort_batch_by_slot(&mut by_slot, |i| slots[i]);
    assert_eq!(
        by_slot,
        vec![3, 1, 0, 2],
        "slotted ascending first (0 then 5), then slotless by active index"
    );
}
