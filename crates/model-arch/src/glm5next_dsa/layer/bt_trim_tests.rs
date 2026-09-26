// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-26: Host tests of `bt_entries_needed`, the block-table prefix `decode_k`
//! uploads.
//!
//! Owner: model-arch (GLM-5.3 DSA).
//! Invariants: none beyond the types.

use super::decode_k::bt_entries_needed;

/// 2026-09-25: A block table of `262_144 / 16 + 2` entries, more than a 16,384-entry
/// buffer, trims to 3 entries for a 20-token sequence and 3 rows.
#[test]
fn a58_short_sequence_at_262k_declared_context() {
    let pool = 262_144 / 16 + 2;
    assert_eq!(pool, 16_386, "the pre-claimed pool that overran bt_cap");
    assert!(pool > 16_384, "and it is over the persistent buffer");
    assert_eq!(bt_entries_needed(20, 3, 16), 3);
    assert!(bt_entries_needed(20, 3, 16) <= 16_384);
}

/// 2026-09-25: Every block index the gather can read is inside the trim.
#[test]
fn trim_covers_every_indexable_position() {
    for &(seq_len, k, bs) in &[
        (0usize, 1usize, 16usize),
        (1, 1, 16),
        (15, 1, 16),
        (16, 1, 16),
        (17, 4, 16),
        (4095, 4, 16),
        (131_072, 3, 16),
        (262_143, 4, 16),
        (1000, 1, 64),
    ] {
        let n = bt_entries_needed(seq_len, k, bs);
        let highest = (seq_len + k).saturating_sub(1) / bs;
        assert!(
            highest < n,
            "seq_len={seq_len} k={k} bs={bs}: highest index {highest} not < {n}"
        );
    }
}

/// 2026-09-25: At 16,384 tokens and 4 rows the trim fits a 16,384-entry buffer.
#[test]
fn trim_fits_the_persistent_buffer_across_the_dsa_window() {
    assert!(bt_entries_needed(16_384, 4, 16) <= 16_384);
}

/// 2026-09-25: `block_size` 0 does not divide by zero.
#[test]
fn zero_block_size_does_not_panic() {
    assert_eq!(bt_entries_needed(8, 1, 0), 11);
}
