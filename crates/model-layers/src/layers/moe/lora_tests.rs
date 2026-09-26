// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-25: Tests for the router route build (`build_router_route`) and the
//! `delta` scratch width (`lora_delta_cols`). `delta` is read only by the router
//! fold and the decode down fold, so its width is the larger of their widths (at
//! least 1), and gate/up's `k_in` does not size it.
//!
//! Owner: model-layers (MoE).
//! Invariants: none beyond the types.

use super::lora_delta_cols;
use crate::layers::moe::MoeLayer;
use crate::layers::ops::lora_delta::LoraPair;
use crate::weight_map::DenseWeight;
use metrale_gpu_runtime::gpu::mock::MockGpuBackend;
use metrale_gpu_runtime::gpu::{DevicePtr, GpuBackend};

const NUM_EXPERTS: u32 = 256;
const MOE_INTER: u32 = 512;
const HIDDEN: u32 = 2048;

fn dummy_router_pair(a_addr: u64, b_addr: u64, scale: f32, rank: u32) -> LoraPair {
    LoraPair {
        a: DenseWeight {
            weight: DevicePtr(a_addr),
        },
        b: DenseWeight {
            weight: DevicePtr(b_addr),
        },
        rank,
        k_in: HIDDEN,
        n_out: NUM_EXPERTS,
        scale,
        max_rank: rank,
    }
}

#[test]
fn router_pair_builds_single_uploaded_route() {
    let gpu = MockGpuBackend::new();
    let rp = dummy_router_pair(0xAAAA, 0xBBBB, 2.0, 8);
    let route = MoeLayer::build_router_route(&rp, &gpu).unwrap();
    let mut a = [0u8; 8];
    let mut b = [0u8; 8];
    let mut scale = [0u8; 4];
    gpu.copy_d2h(route.a_table, &mut a).unwrap();
    gpu.copy_d2h(route.b_table, &mut b).unwrap();
    gpu.copy_d2h(route.scale_table, &mut scale).unwrap();

    assert_eq!(route.n_experts, 1);
    assert_eq!(route.k_in, HIDDEN);
    assert_eq!(route.n_out, NUM_EXPERTS);
    assert_eq!(route.max_rank, 8);
    assert_eq!(u64::from_le_bytes(a), 0xAAAA);
    assert_eq!(u64::from_le_bytes(b), 0xBBBB);
    assert_eq!(f32::from_le_bytes(scale), 2.0);
}

#[test]
fn full_adapter_takes_max_of_router_and_down() {
    assert_eq!(
        lora_delta_cols(Some(NUM_EXPERTS), Some(MOE_INTER)),
        MOE_INTER as usize
    );
}

#[test]
fn router_only_is_num_experts() {
    assert_eq!(
        lora_delta_cols(Some(NUM_EXPERTS), None),
        NUM_EXPERTS as usize
    );
}

#[test]
fn gateup_only_no_router_is_unit() {
    assert_eq!(lora_delta_cols(None, None), 1);
}

#[test]
fn down_only_is_moe_inter() {
    assert_eq!(lora_delta_cols(None, Some(MOE_INTER)), MOE_INTER as usize);
}

#[test]
fn router_wins_when_experts_exceed_inter() {
    assert_eq!(lora_delta_cols(Some(1024), Some(MOE_INTER)), 1024);
}
