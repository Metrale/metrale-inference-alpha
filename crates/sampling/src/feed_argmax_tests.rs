// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-25: Tests of the feed argmax reference against the synchronous
//! host path.
//!
//! Owner: metrale-sampling.
//! Invariants: none beyond the types.
//!
//! * `q` (`masked_last_wins`) against the temperature-0 host sampler over the
//!   masked row: every row of up to 6 tokens over a value set with ties, with
//!   every mask (none, in range, out of range, both ids equal), plus random
//!   wide rows with ties.
//! * `p` (`plain_kernel_argmax`) against `argmax_bf16` when the maximum is
//!   unique, and the kernel's tie order pinned where the two differ.
//! * `feed_argmax` against `sync_emission`, built from those two.

use super::feed_argmax::{
    BLOCK, NO_ID, bf16_to_f32, feed_argmax, feed_argmax_f32, masked_last_wins, masked_row,
    plain_kernel_argmax,
};
use crate::{SamplingParams, argmax_bf16, sample_with_params_history};

/// 2026-09-25: The host re-pick: the two ids at -inf, as `PostCloseThinkMask`
/// sets them, then `sample_with_params_history` with `SamplingParams::greedy`.
fn host_repick(row: &[f32], mask: [u32; 2]) -> u32 {
    let masked = masked_row(row, mask);
    let bytes: Vec<u8> = masked.iter().flat_map(|v| v.to_le_bytes()).collect();
    sample_with_params_history(&bytes, &SamplingParams::greedy(0), &[])
}

/// 2026-09-25: The device argmax, re-picked on the host when it landed on a
/// masked id.
fn sync_emission(row: &[f32], mask: [u32; 2]) -> u32 {
    let p = plain_kernel_argmax(row);
    if p == mask[0] || p == mask[1] {
        host_repick(row, mask)
    } else {
        p
    }
}

fn bf16_bits(v: f32) -> u16 {
    (v.to_bits() >> 16) as u16
}

/// 2026-09-25: Exactly representable in BF16, dense with ties, and including -inf.
const VALUES: [f32; 5] = [f32::NEG_INFINITY, -1.0, 0.0, 1.0, 2.0];

fn masks_for(vocab: usize) -> Vec<[u32; 2]> {
    let ids: Vec<u32> = (0..=vocab as u32).chain([NO_ID]).collect();
    let mut out = Vec::new();
    for &a in &ids {
        for &b in &ids {
            out.push([a, b]);
        }
    }
    out
}

#[test]
fn exhaustive_small_rows_every_mask_the_masked_pick_is_the_host_repick() {
    let mut cases = 0u64;
    for vocab in 1..=6usize {
        let combos = VALUES.len().pow(vocab as u32);
        for c in 0..combos {
            let mut k = c;
            let row: Vec<f32> = (0..vocab)
                .map(|_| {
                    let v = VALUES[k % VALUES.len()];
                    k /= VALUES.len();
                    v
                })
                .collect();
            for mask in masks_for(vocab) {
                assert_eq!(
                    masked_last_wins(&row, mask),
                    host_repick(&row, mask),
                    "row {row:?} mask {mask:?}"
                );
                cases += 1;
            }
        }
    }
    assert!(cases > 1_000_000, "{cases} cases");
}

#[test]
fn exhaustive_small_rows_the_feed_answer_is_the_synchronous_emission() {
    for vocab in 1..=6usize {
        let combos = VALUES.len().pow(vocab as u32);
        for c in 0..combos {
            let mut k = c;
            let bits: Vec<u16> = (0..vocab)
                .map(|_| {
                    let v = VALUES[k % VALUES.len()];
                    k /= VALUES.len();
                    bf16_bits(v)
                })
                .collect();
            let row: Vec<f32> = bits.iter().map(|&b| bf16_to_f32(b)).collect();
            for mask in masks_for(vocab) {
                assert_eq!(
                    feed_argmax(&bits, mask),
                    sync_emission(&row, mask),
                    "row {row:?} mask {mask:?}"
                );
            }
        }
    }
}

#[test]
fn the_plain_pick_is_the_sequential_host_argmax_when_the_maximum_is_unique() {
    // 2026-09-25: With one maximum there is nothing to tie-break, so the plain
    // pick equals the host's `argmax_bf16`.
    let finite = &VALUES[1..];
    let mut checked = 0;
    for vocab in 1..=5usize {
        let combos = finite.len().pow(vocab as u32);
        for c in 0..combos {
            let mut k = c;
            let bits: Vec<u16> = (0..vocab)
                .map(|_| {
                    let v = finite[k % finite.len()];
                    k /= finite.len();
                    bf16_bits(v)
                })
                .collect();
            let row: Vec<f32> = bits.iter().map(|&b| bf16_to_f32(b)).collect();
            let max = row.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            if row.iter().filter(|&&v| v == max).count() != 1 {
                continue;
            }
            let bytes: Vec<u8> = bits.iter().flat_map(|b| b.to_le_bytes()).collect();
            assert_eq!(
                plain_kernel_argmax(&row),
                argmax_bf16(&bytes),
                "row {row:?}"
            );
            checked += 1;
        }
    }
    assert!(checked > 300, "{checked} rows");
}

#[test]
fn the_plain_pick_carries_the_kernels_tree_order_on_ties() {
    // 2026-09-25: The tree reduction merges thread pairs (t, t + s) for
    // s = 512 .. 1. A lower thread holding a smaller value takes the higher
    // thread's maximum, then keeps it against an equal middle thread, so a tie
    // does not resolve to the first index. `seq` is the host's `argmax_bf16`,
    // shown beside it.
    let seq = |row: &[f32]| {
        let bytes: Vec<u8> = row
            .iter()
            .flat_map(|v| bf16_bits(*v).to_le_bytes())
            .collect();
        argmax_bf16(&bytes)
    };
    let row = [-1.0f32, 2.0, 2.0];
    assert_eq!(plain_kernel_argmax(&row), 2);
    assert_eq!(seq(&row), 1);
    let row = [2.0f32, 2.0, -1.0];
    assert_eq!((plain_kernel_argmax(&row), seq(&row)), (0, 0));
    let row = [2.0f32, -1.0, 2.0];
    assert_eq!((plain_kernel_argmax(&row), seq(&row)), (0, 0));
    // 2026-09-25: Index 0 at -inf is below the floor, so thread 0 holds nothing,
    // and thread 2's maximum wins the (0, 2) merge.
    let row = [f32::NEG_INFINITY, -1.0, -1.0];
    assert_eq!((plain_kernel_argmax(&row), seq(&row)), (2, 1));
    let mut wide = vec![0.0f32; BLOCK];
    wide[700] = 5.0;
    wide[701] = 5.0;
    assert_eq!(plain_kernel_argmax(&wide), 700);
}

#[test]
fn the_plain_pick_keeps_the_kernels_residue_rule_past_the_block_width() {
    // 2026-09-25: A tie at 1023 (thread 1023) and 1024 (thread 0): the final
    // merge keeps thread 0, so the answer is 1024 where a sequential scan
    // answers 1023.
    let mut row = vec![0.0f32; 2 * BLOCK];
    row[BLOCK - 1] = 3.0;
    row[BLOCK] = 3.0;
    assert_eq!(plain_kernel_argmax(&row), BLOCK as u32);
    let bytes: Vec<u8> = row
        .iter()
        .flat_map(|v| bf16_bits(*v).to_le_bytes())
        .collect();
    assert_eq!(argmax_bf16(&bytes), BLOCK as u32 - 1);
    // 2026-09-25: Within one thread the first occurrence wins: 5 and 5 + BLOCK.
    let mut row = vec![0.0f32; 2 * BLOCK];
    row[5] = 3.0;
    row[5 + BLOCK] = 3.0;
    assert_eq!(plain_kernel_argmax(&row), 5);
    // 2026-09-25: The masked pick takes the last index, whatever the threads.
    assert_eq!(masked_last_wins(&row, [NO_ID, NO_ID]), 5 + BLOCK as u32);
}

#[test]
fn a_row_below_the_floor_answers_zero_on_the_plain_pick_and_last_on_the_repick() {
    let row = vec![f32::NEG_INFINITY; 7];
    assert_eq!(plain_kernel_argmax(&row), 0);
    assert_eq!(masked_last_wins(&row, [NO_ID, NO_ID]), 6);
    assert_eq!(host_repick(&row, [NO_ID, NO_ID]), 6);
    // 2026-09-25: Index 0 masked: the plain pick is a masked id, so the re-pick
    // answers, and with every value at -inf it takes the last index.
    assert_eq!(feed_argmax_f32(&row, [0, NO_ID]), 6);
    assert_eq!(sync_emission(&row, [0, NO_ID]), 6);
}

struct Lcg(u64);
impl Lcg {
    fn next(&mut self) -> u64 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        self.0 >> 11
    }
    fn below(&mut self, n: u64) -> u64 {
        self.next() % n
    }
}

#[test]
fn random_wide_rows_with_ties_agree_with_the_host_repick_and_the_sync_emission() {
    let mut rng = Lcg(0x5eed_f00d);
    let palette: [f32; 7] = [-3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0];
    for _ in 0..3000 {
        let vocab = 1 + rng.below(3 * BLOCK as u64) as usize;
        let bits: Vec<u16> = (0..vocab)
            .map(|_| bf16_bits(palette[rng.below(palette.len() as u64) as usize]))
            .collect();
        let row: Vec<f32> = bits.iter().map(|&b| bf16_to_f32(b)).collect();
        let pick = |rng: &mut Lcg| match rng.below(4) {
            0 => NO_ID,
            1 => vocab as u32 + rng.below(3) as u32,
            // 2026-09-25: Mask the plain winner often, so the re-pick branch
            // runs in many cases.
            2 => plain_kernel_argmax(&row),
            _ => rng.below(vocab as u64) as u32,
        };
        let mask = [pick(&mut rng), pick(&mut rng)];
        assert_eq!(masked_last_wins(&row, mask), host_repick(&row, mask));
        assert_eq!(feed_argmax(&bits, mask), sync_emission(&row, mask));
    }
}
