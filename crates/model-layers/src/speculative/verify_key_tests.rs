// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-25: Tests for `verify_key`: batch order, depth assignment, graph key
//! and the width gate.
//!
//! Owner: model-layers (speculative).
//! Invariants: none beyond the types.
use super::*;

/// 2026-09-25: The key for a batch of `(slot, k)` pairs: `verify_batch_order`, then
/// `verify_graph_key` with both flags clear.
fn key_for(batch: &[(usize, usize)], canonical: bool) -> Vec<u32> {
    let slots: Vec<usize> = batch.iter().map(|&(s, _)| s).collect();
    let ks: Vec<usize> = batch.iter().map(|&(_, k)| k).collect();
    let (order, depths) = verify_batch_order(&slots, &ks, canonical);
    let pairs: Vec<(u32, u32)> = order
        .iter()
        .zip(&depths)
        .map(|(&i, &k)| (slots[i] as u32, k as u32))
        .collect();
    verify_graph_key(&pairs, false, false)
}

/// 2026-09-25: Every arrangement of one depth multiset over one slot set gives one key
/// under the canonical arm.
#[test]
fn same_multiset_different_arrangement_is_one_key() {
    let arrangements: [[(usize, usize); 4]; 6] = [
        [(0, 4), (1, 4), (2, 3), (3, 3)],
        [(0, 4), (1, 3), (2, 4), (3, 3)],
        [(0, 4), (1, 3), (2, 3), (3, 4)],
        [(0, 3), (1, 4), (2, 4), (3, 3)],
        [(0, 3), (1, 4), (2, 3), (3, 4)],
        [(0, 3), (1, 3), (2, 4), (3, 4)],
    ];
    let keys: std::collections::HashSet<Vec<u32>> =
        arrangements.iter().map(|a| key_for(a, true)).collect();
    assert_eq!(keys.len(), 1, "6 arrangements must collapse to 1 key");
    assert_eq!(
        keys.into_iter().next().unwrap(),
        vec![0, 4, 1, 4, 2, 3, 3, 3, u32::MAX]
    );
}

/// 2026-09-25: Three depth multisets at n=8, each in every arrangement over slots 0..8:
/// 266 distinct keys with the canonical arm off, 3 with it on.
#[test]
fn n8_key_space_collapses_266_to_3() {
    // 2026-09-25: (count of depth 4, count of 3, count of 2) per multiset.
    let multisets = [(5usize, 2usize, 1usize), (4, 4, 0), (6, 0, 2)];
    let mut legacy: std::collections::HashSet<Vec<u32>> = Default::default();
    let mut canon: std::collections::HashSet<Vec<u32>> = Default::default();
    for &(c4, c3, c2) in &multisets {
        let mut depths: Vec<usize> = Vec::new();
        depths.extend(std::iter::repeat_n(4usize, c4));
        depths.extend(std::iter::repeat_n(3usize, c3));
        depths.extend(std::iter::repeat_n(2usize, c2));
        assert_eq!(depths.len(), 8);
        let mut perm: Vec<usize> = (0..8).collect();
        permute(&mut perm, 0, &mut |p| {
            let batch: Vec<(usize, usize)> = (0..8).map(|s| (s, depths[p[s]])).collect();
            legacy.insert(key_for(&batch, false));
            canon.insert(key_for(&batch, true));
        });
    }
    assert_eq!(legacy.len(), 266, "pre-canonical arrangement count");
    assert_eq!(canon.len(), 3, "one key per depth multiset");
}

/// 2026-09-25: Calls `f` with every permutation of `v`, permuting in place and restoring
/// `v` afterwards.
fn permute(v: &mut Vec<usize>, i: usize, f: &mut impl FnMut(&[usize])) {
    if i == v.len() {
        f(v);
        return;
    }
    for j in i..v.len() {
        v.swap(i, j);
        permute(v, i + 1, f);
        v.swap(i, j);
    }
}

/// 2026-09-25: Different depth multisets over one slot set give different keys.
#[test]
fn different_multisets_have_different_keys() {
    let a = key_for(&[(0, 4), (1, 4), (2, 3), (3, 3)], true);
    let b = key_for(&[(0, 4), (1, 3), (2, 3), (3, 3)], true);
    let c = key_for(&[(0, 4), (1, 4), (2, 4), (3, 3)], true);
    assert_ne!(a, b);
    assert_ne!(a, c);
    assert_ne!(b, c);
}

/// 2026-09-25: Different slot sets with one depth multiset give different keys.
#[test]
fn different_slot_sets_have_different_keys() {
    let a = key_for(&[(0, 4), (1, 3)], true);
    let b = key_for(&[(0, 4), (2, 3)], true);
    assert_ne!(a, b);
}

/// 2026-09-25: The WY-tables flag changes the key.
#[test]
fn wy_table_presence_splits_the_key() {
    let pairs = [(0u32, 4u32), (1, 3)];
    assert_ne!(
        verify_graph_key(&pairs, true, false),
        verify_graph_key(&pairs, false, false)
    );
}

/// 2026-09-25: The four combinations of the two flags give four distinct keys, and only
/// the sentinel word differs.
#[test]
fn write_on_accept_splits_the_key() {
    let pairs = [(0u32, 4u32), (1, 4)];
    let keys = [
        verify_graph_key(&pairs, false, false),
        verify_graph_key(&pairs, true, false),
        verify_graph_key(&pairs, false, true),
        verify_graph_key(&pairs, true, true),
    ];
    for i in 0..keys.len() {
        for j in 0..keys.len() {
            assert_eq!(keys[i] == keys[j], i == j, "{i} vs {j}");
        }
    }
    assert_eq!(keys[0][..4], keys[2][..4]);
}

/// 2026-09-25: The canonical arm orders slots ascending and depths non-increasing, and
/// keeps the depth multiset.
#[test]
fn canonical_order_is_slot_ascending_and_depth_descending() {
    // 2026-09-25: Slot order and depth order disagree in the input.
    let slots = [7usize, 2, 5, 0, 3];
    let ks = [2usize, 4, 2, 3, 4];
    let (order, depths) = verify_batch_order(&slots, &ks, true);
    let placed: Vec<usize> = order.iter().map(|&i| slots[i]).collect();
    assert!(
        placed.windows(2).all(|w| w[0] < w[1]),
        "slots must be ascending in batch order, got {placed:?}"
    );
    assert!(
        depths.windows(2).all(|w| w[0] >= w[1]),
        "depths must be non-increasing, got {depths:?}"
    );
    let mut before = ks.to_vec();
    before.sort_unstable();
    let mut after = depths.clone();
    after.sort_unstable();
    assert_eq!(before, after);
    assert_eq!(placed, vec![0, 2, 3, 5, 7]);
    assert_eq!(depths, vec![4, 4, 3, 2, 2]);
}

/// 2026-09-25: With consecutive input slots, every run of equal depths sits on
/// consecutive slots, the layout `trait_decode_batched_conv_gdn_multi.rs`
/// checks before its batched launch.
#[test]
fn each_depth_run_owns_a_consecutive_slot_block() {
    let slots = [0usize, 1, 2, 3, 4, 5, 6, 7];
    let ks = [2usize, 4, 3, 4, 2, 3, 4, 3];
    let (order, depths) = verify_batch_order(&slots, &ks, true);
    let placed: Vec<usize> = order.iter().map(|&i| slots[i]).collect();
    let mut g0 = 0usize;
    while g0 < depths.len() {
        let mut g1 = g0 + 1;
        while g1 < depths.len() && depths[g1] == depths[g0] {
            g1 += 1;
        }
        assert!(
            placed[g0..g1].windows(2).all(|w| w[1] == w[0] + 1),
            "run {g0}..{g1} (k={}) must be consecutive slots, got {:?}",
            depths[g0],
            &placed[g0..g1]
        );
        g0 = g1;
    }
}

/// 2026-09-25: `verify_batch_permutation` never re-pairs depths: each member keeps its
/// own row count, the depth `mtp_dcut::plan` truncated its drafts to.
#[test]
fn permutation_leaves_depths_attached_to_their_member() {
    // 2026-09-25: Depths rise along the slots, which the canonical arm of
    // `verify_batch_order` never emits.
    let slots = [7usize, 2, 5, 0];
    let ks = [4usize, 2, 3, 2];
    for canonical in [true, false] {
        let order = verify_batch_permutation(&slots, &ks, canonical);
        let (_, assigned) = verify_batch_order(&slots, &ks, canonical);
        let carried: Vec<usize> = order.iter().map(|&i| ks[i]).collect();
        if canonical {
            assert_eq!(carried, vec![2, 2, 3, 4]);
            assert_eq!(assigned, vec![4, 3, 2, 2]);
        } else {
            assert_eq!(carried, assigned);
        }
    }
}

/// 2026-09-25: Applying `verify_batch_order` to its own canonical output returns the
/// identity order and the same depths.
#[test]
fn canonical_order_is_idempotent() {
    let slots = [7usize, 2, 5, 0, 3];
    let ks = [2usize, 4, 2, 3, 4];
    let (o1, d1) = verify_batch_order(&slots, &ks, true);
    let s1: Vec<usize> = o1.iter().map(|&i| slots[i]).collect();
    let (o2, d2) = verify_batch_order(&s1, &d1, true);
    assert_eq!(o2, (0..s1.len()).collect::<Vec<_>>());
    assert_eq!(d2, d1);
}

/// 2026-09-25: The non-canonical arm keeps each member's own depth, so two arrangements
/// of one multiset give different keys.
#[test]
fn kill_switch_restores_the_arrangement_keyed_behaviour() {
    let a = key_for(&[(0, 4), (1, 4), (2, 3), (3, 3)], false);
    let b = key_for(&[(0, 3), (1, 3), (2, 4), (3, 4)], false);
    assert_ne!(a, b, "legacy keys must still separate arrangements");
    assert_eq!(b, vec![2, 4, 3, 4, 0, 3, 1, 3, u32::MAX]);
}

/// 2026-09-25: With equal depths, as `mtp_dcut::plan` returns when D-Cut does not run,
/// both arms give the same order, ascending by slot.
#[test]
fn uniform_depths_are_identical_under_both_arms() {
    let slots = [3usize, 1, 2, 0];
    let ks = [3usize; 4];
    let canon = verify_batch_order(&slots, &ks, true);
    let legacy = verify_batch_order(&slots, &ks, false);
    assert_eq!(canon, legacy);
    assert_eq!(
        canon.0.iter().map(|&i| slots[i]).collect::<Vec<_>>(),
        vec![0, 1, 2, 3]
    );
}

/// 2026-09-25: Empty and one-member batches do not panic.
#[test]
fn empty_and_single_batches_are_well_formed() {
    assert_eq!(verify_batch_order(&[], &[], true), (vec![], vec![]));
    assert_eq!(verify_batch_order(&[5], &[3], true), (vec![0], vec![3]));
    assert_eq!(verify_graph_key(&[], false, false), vec![u32::MAX]);
}

#[test]
fn mismatched_batch_vectors_fail_closed() {
    for (slots, ks) in [(&[1, 2][..], &[4][..]), (&[1][..], &[4, 3][..])] {
        let result = std::panic::catch_unwind(|| verify_batch_permutation(slots, ks, true));
        assert!(result.is_err(), "slots={slots:?}, ks={ks:?}");
    }
}

/// 2026-09-25: The key for a batch whose arm `canonical_assignment_at` chooses from the
/// batch width and the given threshold and kill-switch state, as
/// `canonical_assignment` does with the environment's values.
fn key_through_gate(batch: &[(usize, usize)], min_width: usize, kill_clear: bool) -> Vec<u32> {
    key_for(
        batch,
        canonical_assignment_at(batch.len(), min_width, kill_clear),
    )
}

/// 2026-09-25: With the environment unset, widths below `CANONICAL_KEY_MIN_WIDTH` take
/// the non-canonical arm, and widths from it up to 64 the canonical one.
#[test]
fn the_gate_boundary_is_the_constant() {
    for n in 0..CANONICAL_KEY_MIN_WIDTH {
        assert!(!canonical_assignment(n), "n={n} must take the legacy arm");
    }
    for n in CANONICAL_KEY_MIN_WIDTH..=64 {
        assert!(canonical_assignment(n), "n={n} must take the canonical arm");
    }
}

/// 2026-09-25: Below the threshold, the assignment and the key bytes equal a reference
/// built here without the `canonical = false` arm: a stable sort by
/// `(Reverse(k), slot)`, each member keeping its own depth. Covers every
/// row-count combination in 2..=4 for batches of 1, 2, 4 and 7 sequences.
#[test]
fn below_the_threshold_is_byte_identical_to_pre_canonical() {
    // 2026-09-25: Row counts 2..=4 are the non-DFlash batched-verify shapes
    // (`can_batch_verify_dispatch`). The slot sets with more than one member
    // are unsorted and have gaps.
    let slot_sets: [&[usize]; 4] = [&[0], &[3, 1], &[5, 0, 2, 9], &[7, 2, 5, 0, 3, 11, 4]];
    for slots in slot_sets {
        let n = slots.len();
        assert!(n < CANONICAL_KEY_MIN_WIDTH);
        for shape in 0..4usize.pow(n as u32) {
            let ks: Vec<usize> = (0..n).map(|i| 2 + (shape >> (2 * i)) % 3).collect();
            let mut pre: Vec<(usize, usize)> = slots.iter().copied().zip(ks.clone()).collect();
            pre.sort_by_key(|&(slot, k)| (std::cmp::Reverse(k), slot));
            let pre_key: Vec<u32> = pre
                .iter()
                .flat_map(|&(s, k)| [s as u32, k as u32])
                .chain(std::iter::once(u32::MAX))
                .collect();

            let (order, depths) = verify_batch_order(slots, &ks, canonical_assignment(n));
            let got: Vec<(usize, usize)> = order
                .iter()
                .zip(&depths)
                .map(|(&i, &k)| (slots[i], k))
                .collect();
            assert_eq!(got, pre, "assignment drifted at n={n} shape={shape}");

            let batch: Vec<(usize, usize)> = slots.iter().copied().zip(ks).collect();
            assert_eq!(
                key_for(&batch, canonical_assignment(n)),
                pre_key,
                "key bytes drifted at n={n} shape={shape}"
            );
        }
    }
}

/// 2026-09-25: At the threshold width, two arrangements of one multiset give one key
/// through the gate, and two keys under the non-canonical arm.
#[test]
fn at_the_threshold_the_gate_selects_the_canonical_arm() {
    let a: Vec<(usize, usize)> = (0..8).map(|s| (s, if s < 5 { 4 } else { 3 })).collect();
    let b: Vec<(usize, usize)> = (0..8).map(|s| (s, if s < 3 { 3 } else { 4 })).collect();
    assert_eq!(a.len(), CANONICAL_KEY_MIN_WIDTH);
    let ka = key_for(&a, canonical_assignment(a.len()));
    assert_eq!(ka, key_for(&b, canonical_assignment(b.len())));
    assert_eq!(ka, key_for(&a, true));
    assert_ne!(key_for(&a, false), key_for(&b, false));
}

/// 2026-09-25: A value that parses as `usize` after trimming is the threshold; anything
/// else gives `CANONICAL_KEY_MIN_WIDTH`.
#[test]
fn min_width_override_parses() {
    let os = |v: &str| Some(std::ffi::OsString::from(v));
    assert_eq!(min_width_from_env(None), CANONICAL_KEY_MIN_WIDTH);
    assert_eq!(min_width_from_env(os("4")), 4);
    assert_eq!(min_width_from_env(os("0")), 0);
    assert_eq!(min_width_from_env(os(" 16 ")), 16);
    assert_eq!(min_width_from_env(os("")), CANONICAL_KEY_MIN_WIDTH);
    assert_eq!(min_width_from_env(os("-1")), CANONICAL_KEY_MIN_WIDTH);
    assert_eq!(min_width_from_env(os("eight")), CANONICAL_KEY_MIN_WIDTH);
}

/// 2026-09-25: Moving the threshold moves the arm, seen in the key bytes: at 4 an n=4
/// batch is keyed canonically and at 16 an n=8 batch is not; at the default
/// the reverse holds for both.
#[test]
fn min_width_override_moves_the_boundary() {
    let n4 = [(0usize, 4usize), (1, 3), (2, 4), (3, 2)];
    let n8: Vec<(usize, usize)> = (0..8).map(|s| (s, if s < 5 { 4 } else { 3 })).collect();
    let lowered = min_width_from_env(Some(std::ffi::OsString::from("4")));
    let raised = min_width_from_env(Some(std::ffi::OsString::from("16")));

    assert_eq!(key_through_gate(&n4, lowered, true), key_for(&n4, true));
    assert_eq!(
        key_through_gate(&n4, CANONICAL_KEY_MIN_WIDTH, true),
        key_for(&n4, false)
    );
    assert_eq!(key_through_gate(&n8, raised, true), key_for(&n8, false));
    assert_eq!(
        key_through_gate(&n8, CANONICAL_KEY_MIN_WIDTH, true),
        key_for(&n8, true)
    );
    let all = min_width_from_env(Some(std::ffi::OsString::from("0")));
    assert_eq!(key_through_gate(&n4, all, true), key_for(&n4, true));
}

/// 2026-09-25: With the kill switch set, no width and no threshold selects the
/// canonical arm.
#[test]
fn kill_switch_dominates_width_and_override() {
    let n8: Vec<(usize, usize)> = (0..8).map(|s| (s, if s < 5 { 4 } else { 3 })).collect();
    for min_width in [0usize, 1, 4, CANONICAL_KEY_MIN_WIDTH, 64] {
        for n in [0usize, 1, 2, 4, 8, 16, 128] {
            assert!(
                !canonical_assignment_at(n, min_width, false),
                "kill switch must dominate at n={n} min_width={min_width}"
            );
        }
        assert_eq!(key_through_gate(&n8, min_width, false), key_for(&n8, false));
    }
}
