// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-25: Tests for `moe_row_adapter`: the route decision, both row
//! maps, and the decode-batch refusal.
//!
//! Owner: model-layers (lora).
//! Invariants: none beyond the types.

use super::*;
use crate::layer::MoeLoraRoute;

#[test]
fn route_off_when_no_moe_lora() {
    assert_eq!(resolve_moe_lora_route(-1, -1, false), MoeLoraRoute::Fold);
    assert_eq!(resolve_moe_lora_route(3, 0, false), MoeLoraRoute::Fold);
}

#[test]
fn route_base_request_skips() {
    assert_eq!(resolve_moe_lora_route(-1, 0, true), MoeLoraRoute::Skip);
    assert_eq!(resolve_moe_lora_route(-5, 2, true), MoeLoraRoute::Skip);
}

#[test]
fn route_active_adapter_folds() {
    assert_eq!(resolve_moe_lora_route(0, 0, true), MoeLoraRoute::Fold);
    assert_eq!(resolve_moe_lora_route(2, 2, true), MoeLoraRoute::Fold);
}

#[test]
fn route_non_active_adapter_refuses() {
    assert_eq!(resolve_moe_lora_route(1, 0, true), MoeLoraRoute::Refuse);
    assert_eq!(resolve_moe_lora_route(0, 3, true), MoeLoraRoute::Refuse);
}

#[test]
fn row_adapter_uniform_single_stream() {
    let map = build_moe_row_adapter_host(&[0, 4], &[2]).unwrap();
    assert_eq!(map, vec![2, 2, 2, 2]);
}

#[test]
fn row_adapter_varlen_mixed_streams() {
    let map = build_moe_row_adapter_host(&[0, 2, 5, 6], &[-1, 1, 0]).unwrap();
    assert_eq!(map, vec![-1, -1, 1, 1, 1, 0]);
}

#[test]
fn row_adapter_empty_stream_span() {
    let map = build_moe_row_adapter_host(&[0, 2, 2, 5], &[7, 9, -1]).unwrap();
    assert_eq!(map, vec![7, 7, -1, -1, -1]);
}

#[test]
fn row_adapter_rejects_malformed() {
    assert!(build_moe_row_adapter_host(&[], &[]).is_none());
    assert!(build_moe_row_adapter_host(&[0], &[]).is_none());
    assert!(build_moe_row_adapter_host(&[0, 2, 4], &[0]).is_none());
    assert!(build_moe_row_adapter_host(&[1, 3], &[0]).is_none());
    assert!(build_moe_row_adapter_host(&[0, 4, 2], &[0, 1]).is_none());
}

#[test]
fn decode_map_off_when_no_moe_lora() {
    // 2026-09-25: With `has_moe_lora == false` every row resolves to `Fold` and
    // carries `active`, which is -1 here.
    let map = build_moe_row_adapter_decode(&[-1, 0, 2], 4, -1, false);
    assert_eq!(map, vec![-1, -1, -1, -1]);
}

#[test]
fn decode_map_all_base_skips() {
    let map = build_moe_row_adapter_decode(&[-1, -1], 4, 0, true);
    assert_eq!(map, vec![-1, -1, -1, -1]);
}

#[test]
fn decode_map_mixed_base_and_active() {
    let map = build_moe_row_adapter_decode(&[3, -1, 3], 4, 3, true);
    assert_eq!(map, vec![3, -1, 3, -1]);
}

#[test]
fn decode_map_refuse_row_defensively_skips() {
    let map = build_moe_row_adapter_decode(&[0, 1], 2, 0, true);
    assert_eq!(map, vec![0, -1]);
}

#[test]
fn decode_map_padding_widths() {
    for padded_n in [2usize, 4, 8] {
        let map = build_moe_row_adapter_decode(&[0], padded_n, 0, true);
        assert_eq!(map.len(), padded_n);
        assert_eq!(map[0], 0);
        assert!(map[1..].iter().all(|&v| v == -1));
    }
}

#[test]
fn decode_map_mixed_batch_16_exact() {
    let slots = [2, -1, 2, 2, -1, -1, 2, -1, 2, 2, -1, 2, -1];
    let map = build_moe_row_adapter_decode(&slots, 16, 2, true);
    let expect = vec![2, -1, 2, 2, -1, -1, 2, -1, 2, 2, -1, 2, -1, -1, -1, -1];
    assert_eq!(map, expect);
    assert_eq!(map.len(), 16);
}

#[test]
fn decode_map_full_cap_32_all_active() {
    let slots = [0i32; 32];
    let map = build_moe_row_adapter_decode(&slots, 32, 0, true);
    assert_eq!(map, vec![0i32; 32]);
    assert_eq!(map.len(), 32);
}

#[test]
fn decode_route_guard_refuses_only_refuse() {
    use super::ensure_decode_route_servable as guard;
    assert!(guard(MoeLoraRoute::Skip, "decode_batch_compute_main").is_ok());
    assert!(guard(MoeLoraRoute::Fold, "decode_batch_compute_main").is_ok());
    let err = guard(MoeLoraRoute::Refuse, "decode_batch_compute_main")
        .expect_err("Refuse must not be servable");
    assert!(
        err.to_string().contains("non-active adapter"),
        "guard must explain the refusal: {err}"
    );
    assert!(
        err.to_string().contains("decode_batch_compute_main"),
        "guard must name the refusing path: {err}"
    );
}
