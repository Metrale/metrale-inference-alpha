// SPDX-License-Identifier: AGPL-3.0-only

use crate::scheduler::logit_processors::min_tokens_eos::mask_eos_before_min;
use spark_runtime::sampler::{SamplingParams, greedy_pick_last_wins, sample_with_params_history};

fn old_sampler(logits: &[f32]) -> u32 {
    let bytes: Vec<u8> = logits.iter().flat_map(|v| v.to_le_bytes()).collect();
    sample_with_params_history(&bytes, &SamplingParams::greedy(1024), &[])
}

#[test]
fn greedy_reuse_matches_sampler_after_real_floor_masks() {
    for len in [0, 1023, 1024, 1025] {
        let mut logits = [7.0, 7.0, 9.0, 10.0];
        mask_eos_before_min(&mut logits, &[2, 3], len, 1024);
        let picked = greedy_pick_last_wins(&logits);
        assert_eq!(picked, old_sampler(&logits));
        assert_eq!(picked, if len < 1024 { 1 } else { 3 });
    }
    let mut all_masked = [1.0, 2.0];
    mask_eos_before_min(&mut all_masked, &[0, 1], 0, 1024);
    assert_eq!(greedy_pick_last_wins(&all_masked), old_sampler(&all_masked));
    assert_eq!(greedy_pick_last_wins(&all_masked), 1);
}

#[test]
fn greedy_reuse_preserves_sampler_special_float_semantics() {
    let values = [
        f32::NEG_INFINITY,
        -1.0,
        -0.0,
        0.0,
        1.0,
        f32::INFINITY,
        f32::NAN,
    ];
    for x in values {
        for y in values {
            for z in values {
                let logits = [x, y, z];
                assert_eq!(
                    greedy_pick_last_wins(&logits),
                    old_sampler(&logits),
                    "{logits:?}"
                );
            }
        }
    }
    assert_eq!(greedy_pick_last_wins(&[]), old_sampler(&[]));
}
