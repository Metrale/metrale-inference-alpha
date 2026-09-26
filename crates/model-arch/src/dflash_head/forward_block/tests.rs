// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-26: Tests of `forward_block`: the rt2 vocab GEMV guard.
//!
//! Owner: model-arch (DFlash drafter).
//! Invariants: none beyond the types.

use super::rt2_16_covers_the_batch;

/// 2026-09-25: The guard takes the total row count: eligible exactly for 1 to 16
/// rows, the range `ops::fp8_gemv_rowscale_batch16_rt2` accepts.
#[test]
fn the_rt2_vocab_kernel_is_only_eligible_for_batches_it_covers() {
    for gamma in 1..=16u32 {
        assert!(rt2_16_covers_the_batch(gamma), "n_seq=1, gamma={gamma}");
    }
    assert!(!rt2_16_covers_the_batch(10 * 2));
    assert!(rt2_16_covers_the_batch(16));
    assert!(!rt2_16_covers_the_batch(17));
    assert!(!rt2_16_covers_the_batch(0));
    assert!(!rt2_16_covers_the_batch(8 * 8));
}
