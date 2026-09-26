// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-26: Tests for the timing fields of the wire usage block (`time_to_first_token_ms`,
//! `decode_time_ms`, `total_time_ms`, `response_token/s`) and their consistency.
//!
//! Owner: server (OpenAI API layer) tests.
//! Invariants: none beyond the types.

use crate::ir;
use crate::openai::Usage;

fn ir_usage(completion_tokens: usize, ttft_ms: f64, decode_ms: f64) -> ir::Usage {
    ir::Usage {
        prompt_tokens: 40,
        completion_tokens,
        cached_prompt_tokens: 8,
        reasoning_tokens: 3,
        accepted_prediction_tokens: 5,
        time_to_first_token_ms: ttft_ms,
        decode_time_ms: decode_ms,
        response_tokens_per_second: ir::Usage::decode_rate_tok_s(completion_tokens, decode_ms),
    }
}

#[test]
fn wire_usage_carries_the_raw_timing_components_and_keeps_the_shipped_keys() {
    let v = serde_json::to_value(Usage::from(&ir_usage(10, 100.0, 900.0))).unwrap();
    assert_eq!(v["decode_time_ms"], 900.0, "{v}");
    assert_eq!(v["total_time_ms"], 1000.0, "{v}");
    assert_eq!(v["time_to_first_token_ms"], 100.0, "{v}");
    assert_eq!(v["response_token/s"], 10.0, "{v}");
    assert_eq!(v["prompt_tokens"], 40);
    assert_eq!(v["completion_tokens"], 10);
    assert_eq!(v["total_tokens"], 50);
    assert_eq!(v["prompt_tokens_details"]["cached_tokens"], 8);
    assert_eq!(v["completion_tokens_details"]["reasoning_tokens"], 3);
    assert_eq!(
        v["completion_tokens_details"]["accepted_prediction_tokens"],
        5
    );
}

/// 2026-09-26: The inter-token latency computed three ways from the published numbers,
/// `(total_time_ms − time_to_first_token_ms) / (n − 1)`, `decode_time_ms / (n − 1)` and
/// `1000 / response_token/s`, is the same value.
#[test]
fn the_wire_components_satisfy_the_aiperf_identity() {
    let u = Usage::from(&ir_usage(10, 100.0, 900.0));
    let itl_from_total = (u.total_time_ms - u.time_to_first_token_ms) / 9.0;
    let itl_from_window = u.decode_time_ms / 9.0;
    let itl_from_rate = 1000.0 / u.response_tokens_per_second;
    assert!((itl_from_total - itl_from_window).abs() < 1e-9);
    assert!((itl_from_window - itl_from_rate).abs() < 1e-9);
    assert!((itl_from_window - 100.0).abs() < 1e-9);
}

/// 2026-09-26: `total_time_ms` is computed from the two stored components, never stored.
#[test]
fn total_time_is_the_sum_of_its_two_primitives() {
    assert_eq!(ir_usage(2, 12.5, 0.0).total_time_ms(), 12.5);
    assert_eq!(ir_usage(2, 12.5, 87.5).total_time_ms(), 100.0);
    assert_eq!(ir_usage(2, 0.0, 87.5).total_time_ms(), 87.5);
}

/// 2026-09-26: `response_token/s` is 0.0 when it is undefined (fewer than two tokens or no
/// decode window); `decode_time_ms` is published either way.
#[test]
fn decode_rate_is_zero_when_undefined_and_n_minus_one_per_second_otherwise() {
    assert_eq!(ir::Usage::decode_rate_tok_s(1, 500.0), 0.0);
    assert_eq!(ir::Usage::decode_rate_tok_s(0, 500.0), 0.0);
    assert_eq!(ir::Usage::decode_rate_tok_s(5, 0.0), 0.0);
    assert_eq!(ir::Usage::decode_rate_tok_s(5, 1000.0), 4.0);
    let v = serde_json::to_value(Usage::from(&ir_usage(1, 100.0, 0.4))).unwrap();
    assert_eq!(v["response_token/s"], 0.0);
    assert_eq!(v["decode_time_ms"], 0.4);
}
