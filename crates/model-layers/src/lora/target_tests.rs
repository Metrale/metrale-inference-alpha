// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-25: Tests for the router and expert targets: their dims, the
//! expert/router pool size, and the per-expert pair map.
//!
//! Owner: model-layers (lora).
//! Invariants: none beyond the types.

use crate::lora::test_support::*;
use crate::lora::*;

#[test]
fn expert_and_router_dims_use_moe_intermediate() {
    // 2026-09-25: The factory config has hidden 2048, moe_intermediate_size
    // 512 and 512 experts.
    let cfg = cfg();
    assert_eq!(ExpertProj::Gate.dims(&cfg, 7), (512, 2048));
    assert_eq!(ExpertProj::Up.dims(&cfg, 7), (512, 2048));
    assert_eq!(ExpertProj::Down.dims(&cfg, 7), (2048, 512));
    assert_eq!(router_dims(&cfg), (512, 2048));
    assert_eq!(ExpertProj::Gate.peft_name(), "gate_proj");
    assert_eq!(ExpertProj::Up.peft_name(), "up_proj");
    assert_eq!(ExpertProj::Down.peft_name(), "down_proj");
}

#[test]
fn expert_router_bytes_golden() {
    let cfg = cfg();
    // 2026-09-25: Each gate, down and router entry is 81,920 bytes at stride
    // 16, for example gate (16 * 2048 + 512 * 16) * 2.
    let ek = vec![
        (7usize, ExpertProj::Gate),
        (7usize, ExpertProj::Gate),
        (7usize, ExpertProj::Down),
    ];
    let rl = vec![3usize];
    assert_eq!(expert_router_bytes(&cfg, &ek, &rl, 16), 81_920 * 4);
    // 2026-09-25: A rank cap of 12 is padded to a stride of 16.
    assert_eq!(expert_router_bytes(&cfg, &ek, &rl, 12), 81_920 * 4);
    assert_eq!(expert_router_bytes(&cfg, &[], &[], 16), 0);
}

#[test]
fn expert_layer_adapted_experts_sorted_deduped() {
    let mut el = ExpertLoraLayer::default();
    el.pairs
        .insert((5, ExpertProj::Gate), dummy_pair(1, 2048, 512));
    el.pairs
        .insert((5, ExpertProj::Down), dummy_pair(2, 512, 2048));
    el.pairs
        .insert((2, ExpertProj::Up), dummy_pair(3, 2048, 512));
    assert_eq!(el.adapted_experts(), vec![2, 5]);
    assert_eq!(el.pair(5, ExpertProj::Gate).map(|p| p.a.weight.0), Some(1));
    assert!(el.pair(5, ExpertProj::Up).is_none());
    assert!(!el.is_empty());
    assert!(ExpertLoraLayer::default().is_empty());
}
