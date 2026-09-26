// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-25: Tests for `h_inter_layout`, the prefix-sum layout of tiered per-slot
//! H intermediates.
//!
//! Owner: model-engine SSM state pool.
//! Invariants: none beyond the types.

use super::h_inter_layout;

#[test]
fn offsets_are_prefix_sums_and_total_is_the_sum() {
    let mut counts = vec![3usize; 8];
    counts.extend(std::iter::repeat_n(1usize, 24));
    counts.push(3);
    let (offsets, total) = h_inter_layout(&counts);
    assert_eq!(offsets.len(), counts.len() + 1);
    assert_eq!(offsets[0], 0);
    assert_eq!(offsets[8], 24);
    assert_eq!(offsets[32], 24 + 24);
    assert_eq!(total, 51);
    for s in 0..counts.len() {
        assert_eq!(offsets[s + 1] - offsets[s], counts[s]);
    }
    let (uni, uni_total) = h_inter_layout(&[5usize; 33]);
    assert_eq!(uni_total, 165);
    for (s, off) in uni.iter().take(33).enumerate() {
        assert_eq!(*off, s * 5);
    }
    assert_eq!(h_inter_layout(&[]), (vec![0], 0));
}
