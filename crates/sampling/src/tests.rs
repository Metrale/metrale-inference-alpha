// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-25: Unit tests for the sampling crate: BF16 decoding, argmax,
//! `Sampler` over the mock GPU backend, and the host sampling pipeline.
//!
//! Owner: sampling.
//! Invariants: none beyond the types.

use super::*;

#[test]
fn test_bf16_to_f32() {
    assert_eq!(bf16_to_f32(0x80, 0x3F), 1.0);
    assert_eq!(bf16_to_f32(0x80, 0xBF), -1.0);
    assert_eq!(bf16_to_f32(0x00, 0x00), 0.0);
}

#[test]
fn test_argmax_bf16() {
    let data: Vec<u8> = vec![
        0x80, 0x3F, // 2026-09-25: BF16 1.0
        0x00, 0x40, // 2026-09-25: BF16 2.0
        0x00, 0x3F, // 2026-09-25: BF16 0.5
    ];
    assert_eq!(argmax_bf16(&data), 1);
}

#[test]
fn test_argmax_negative() {
    let data: Vec<u8> = vec![
        0x80, 0xBF, // 2026-09-25: BF16 -1.0
        0x00, 0xBF, // 2026-09-25: BF16 -0.5
        0x00, 0xC0, // 2026-09-25: BF16 -2.0
    ];
    assert_eq!(argmax_bf16(&data), 1);
}

#[test]
fn test_greedy_params() {
    let params = SamplingParams::greedy(100);
    assert!(params.is_greedy());
    assert_eq!(params.max_tokens, 100);
    assert!(params.stop_token_ids.is_empty());
}

#[test]
fn test_argmax_f32() {
    let data: Vec<u8> = [1.0f32, 2.0f32, 0.5f32]
        .iter()
        .flat_map(|f| f.to_le_bytes())
        .collect();
    assert_eq!(argmax_f32(&data), 1);
}

#[test]
fn test_sampler_with_mock() {
    use metrale_gpu_runtime::gpu::mock::MockGpuBackend;

    let gpu = MockGpuBackend::new();
    let vocab_size = 4;
    let mut sampler = Sampler::new(vocab_size);

    // 2026-09-25: `Sampler::sample` reads the logits from the device as
    // BF16, 2 bytes per element, low byte first.
    let ptr = gpu.alloc(vocab_size * 2).unwrap();
    let logits: Vec<u8> = vec![
        0x00, 0x3F, // 2026-09-25: BF16 0.5
        0x40, 0x40, // 2026-09-25: BF16 3.0
        0x80, 0x3F, // 2026-09-25: BF16 1.0
        0x00, 0x40, // 2026-09-25: BF16 2.0
    ];
    gpu.copy_h2d(&logits, ptr).unwrap();

    let params = SamplingParams::greedy(10);
    let token = sampler.sample(ptr, &params, &gpu).unwrap();
    assert_eq!(token, 1);
}

#[test]
fn test_top_n_sigma_keeps_high_logits() {
    // 2026-09-25: mean 1.2 and sigma 0.4, so the threshold mean - sigma = 0.8
    // keeps all five tokens. A sign error (mean + sigma = 1.6) would keep only
    // token 0, and this test would fail.
    let logits_f32 = [2.0f32, 1.0, 1.0, 1.0, 1.0];
    let logits: Vec<u8> = logits_f32.iter().flat_map(|f| f.to_le_bytes()).collect();
    let params = SamplingParams {
        temperature: 1.0,
        top_k: 0,
        top_p: 1.0,
        top_n_sigma: 1.0,
        min_p: 0.0,
        logit_bias: Vec::new(),
        repetition_penalty: 1.0,
        repetition_penalty_window: 0,
        presence_penalty: 0.0,
        frequency_penalty: 0.0,
        lz_penalty: 0.0,
        dry_multiplier: 0.0,
        dry_base: 1.75,
        dry_allowed_length: 2,
        dry_sequence_breakers: Vec::new(),
        max_tokens: 10,
        stop_token_ids: Vec::new(),
        seed: None,
    };
    // 2026-09-25: P(token 0) = e^2 / (e^2 + 4e) ≈ 0.40 at temperature 1, so
    // 500 draws that all return 0 have probability about 0.40^500.
    let mut saw_non_zero = false;
    for _ in 0..500 {
        let token = sample_with_params(&logits, &params);
        if token != 0 {
            saw_non_zero = true;
            break;
        }
    }
    assert!(
        saw_non_zero,
        "top_n_sigma=1.0 should not filter tokens above mean-sigma"
    );
}

#[test]
fn test_top_n_sigma_disabled_at_zero() {
    let logits_f32 = [1.0f32, 1.0, 1.0, 1.0, 1.5];
    let logits: Vec<u8> = logits_f32.iter().flat_map(|f| f.to_le_bytes()).collect();
    let params = SamplingParams {
        temperature: 1.0,
        top_k: 0,
        top_p: 1.0,
        top_n_sigma: 0.0,
        min_p: 0.0,
        logit_bias: Vec::new(),
        repetition_penalty: 1.0,
        repetition_penalty_window: 0,
        presence_penalty: 0.0,
        frequency_penalty: 0.0,
        lz_penalty: 0.0,
        dry_multiplier: 0.0,
        dry_base: 1.75,
        dry_allowed_length: 2,
        dry_sequence_breakers: Vec::new(),
        max_tokens: 10,
        stop_token_ids: Vec::new(),
        seed: None,
    };
    // 2026-09-25: P(token 4) = e^1.5 / (4e + e^1.5) ≈ 0.29 at temperature 1,
    // so 500 draws that all return 4 have probability about 0.29^500.
    let mut saw_low = false;
    for _ in 0..500 {
        let token = sample_with_params(&logits, &params);
        if token < 4 {
            saw_low = true;
            break;
        }
    }
    assert!(saw_low, "top_n_sigma=0.0 should not filter any tokens");
}

#[test]
fn test_sample_with_params_seeded_temperature_zero_returns_argmax() {
    // 2026-09-25: temperature 0 takes the greedy bypass, which returns before
    // the division by temperature.
    let logits_f32 = [0.5f32, 1.7, 0.3, 1.2];
    let logits: Vec<u8> = logits_f32.iter().flat_map(|f| f.to_le_bytes()).collect();
    let mut params = SamplingParams::greedy(10);
    params.temperature = 0.0;
    for _ in 0..10 {
        assert_eq!(sample_with_params_seeded(&logits, &params, &[], None), 1);
    }
}

#[test]
fn test_greedy_applies_repetition_penalty_before_argmax() {
    // 2026-09-25: token 1 has the largest logit, 1.7. The penalty divides it
    // once per occurrence in the history [1, 1]: 1.7 / 1.5 / 1.5 ≈ 0.76, below
    // token 3's 1.2, so the greedy argmax moves from 1 to 3.
    let logits_f32 = [0.5f32, 1.7, 0.3, 1.2];
    let logits: Vec<u8> = logits_f32.iter().flat_map(|f| f.to_le_bytes()).collect();
    let mut params = SamplingParams::greedy(10);
    params.temperature = 0.0;
    params.repetition_penalty = 1.5;
    let history = vec![1u32, 1u32];
    let token = sample_with_params_seeded(&logits, &params, &history, None);
    assert_eq!(
        token, 3,
        "rep_penalty must shift greedy argmax away from history-repeated token"
    );

    let token_no_hist = sample_with_params_seeded(&logits, &params, &[], None);
    assert_eq!(token_no_hist, 1, "no history → no penalty → raw argmax");

    params.repetition_penalty = 1.0;
    let token_no_pen = sample_with_params_seeded(&logits, &params, &history, None);
    assert_eq!(
        token_no_pen, 1,
        "rep_penalty=1.0 is no-op even with history"
    );
}

#[test]
fn test_greedy_applies_logit_bias_before_argmax() {
    let logits_f32 = [0.5f32, 1.7, 0.3, 1.2];
    let logits: Vec<u8> = logits_f32.iter().flat_map(|f| f.to_le_bytes()).collect();
    let mut params = SamplingParams::greedy(10);
    params.temperature = 0.0;
    params.logit_bias = vec![(0, 5.0)];
    let token = sample_with_params_seeded(&logits, &params, &[], None);
    assert_eq!(token, 0, "logit_bias must shift greedy argmax");
}

#[test]
fn test_sample_with_params_seeded_repetition_penalty_zero_doesnt_div_by_zero() {
    // 2026-09-25: `apply_penalties_and_bias` skips a repetition penalty
    // <= 0.0, whose division would turn positive logits into inf.
    let logits_f32 = [0.5f32, 1.7, 0.3, 1.2];
    let logits: Vec<u8> = logits_f32.iter().flat_map(|f| f.to_le_bytes()).collect();
    let mut params = SamplingParams::greedy(10);
    params.temperature = 1.0;
    params.repetition_penalty = 0.0;
    let history = vec![1u32];
    let token = sample_with_params_seeded(&logits, &params, &history, Some(42));
    assert!(token < 4);
}

#[test]
fn test_top_n_sigma_filters_extreme_outliers() {
    // 2026-09-25: mean -60 and sigma 80, so at n = 0.1 the threshold is -68,
    // which masks tokens 1-4 and leaves only token 0.
    let logits_f32 = [100.0f32, -100.0, -100.0, -100.0, -100.0];
    let logits: Vec<u8> = logits_f32.iter().flat_map(|f| f.to_le_bytes()).collect();
    let params = SamplingParams {
        temperature: 1.0,
        top_k: 0,
        top_p: 1.0,
        top_n_sigma: 0.1,
        min_p: 0.0,
        logit_bias: Vec::new(),
        repetition_penalty: 1.0,
        repetition_penalty_window: 0,
        presence_penalty: 0.0,
        frequency_penalty: 0.0,
        lz_penalty: 0.0,
        dry_multiplier: 0.0,
        dry_base: 1.75,
        dry_allowed_length: 2,
        dry_sequence_breakers: Vec::new(),
        max_tokens: 10,
        stop_token_ids: Vec::new(),
        seed: None,
    };
    for _ in 0..50 {
        let token = sample_with_params(&logits, &params);
        assert_eq!(
            token, 0,
            "extreme low-logit tokens should be filtered at tight sigma"
        );
    }
}
