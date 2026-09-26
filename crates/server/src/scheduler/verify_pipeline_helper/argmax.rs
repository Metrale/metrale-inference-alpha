// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-25: the verify path's first-index-wins argmax.
//!
//! Owner: scheduler.
//! Invariants: none beyond the types.

/// 2026-09-25: index of the first maximum under strict `>`: NaN never wins,
/// and empty or all-NaN input returns 0. -0.0 and +0.0 compare equal, so
/// the first of them wins. Not the sampler's greedy tie-break, which takes
/// the last maximum (`greedy_pick_last_wins`).
///
/// `metrale_sampling::argmax_first_wins_f32` computes it in two passes (the
/// maximum over 8 lanes, then the first index equal to it) so the scan
/// vectorises; the tests below check it against the one-loop form.
pub(super) fn argmax_first_wins(logits: &[f32]) -> u32 {
    metrale_sampling::argmax_first_wins_f32(logits)
}

#[cfg(test)]
mod argmax_tests {
    use super::argmax_first_wins;

    /// 2026-09-25: the one-loop reference form.
    fn reference(logits: &[f32]) -> u32 {
        let mut best_id: u32 = 0;
        let mut best_val: f32 = f32::NEG_INFINITY;
        for (i, &v) in logits.iter().enumerate() {
            if v > best_val {
                best_val = v;
                best_id = i as u32;
            }
        }
        best_id
    }

    fn agree(v: &[f32]) {
        assert_eq!(
            argmax_first_wins(v),
            reference(v),
            "diverged from the original loop on {v:?}"
        );
    }

    #[test]
    fn matches_reference_on_edge_cases() {
        agree(&[]);
        agree(&[1.0]);
        agree(&[1.0, 2.0, 3.0]);
        agree(&[3.0, 2.0, 1.0]);
        agree(&[1.0, 5.0, 5.0, 5.0, 2.0]);
        // 2026-09-25: ties across the 8-lane chunk boundary and in the
        // remainder tail.
        agree(&[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 9.0, 9.0, 9.0]);
        agree(&[9.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 9.0]);
        agree(&[-5.0, -1.0, -3.0]);
        agree(&[-0.0, 0.0]);
        agree(&[0.0, -0.0]);
        agree(&[-1.0, -0.0, 0.0, -1.0]);
        agree(&[f32::NAN, 1.0, 2.0]);
        agree(&[1.0, f32::NAN, 2.0]);
        agree(&[1.0, 2.0, f32::NAN]);
        agree(&[f32::NAN, f32::NAN]);
        agree(&[f32::NEG_INFINITY, -1.0]);
        agree(&[f32::INFINITY, 1.0]);
        agree(&[1.0, f32::INFINITY, f32::INFINITY]);
        agree(&[f32::NEG_INFINITY, f32::NEG_INFINITY]);
    }

    #[test]
    fn matches_reference_on_vocab_sized_input() {
        // 2026-09-25: deterministic pseudo-random vocab-sized input with a
        // duplicated maximum.
        let mut v: Vec<f32> = (0..248_320)
            .map(|i| (((i * 2654435761u64 as usize) % 100_003) as f32) / 1000.0 - 50.0)
            .collect();
        v[123_457] = 999.0;
        v[200_003] = 999.0;
        assert_eq!(argmax_first_wins(&v), reference(&v));
        assert_eq!(argmax_first_wins(&v), 123_457);
    }
}
