// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-25: Tests for `MoeLayer::build_expert_route`: gate/up routes map
//! hidden to moe_inter and down is the transpose; each projection gets its own
//! table sized by its highest adapted expert id. Runs on `MockGpuBackend`.
//!
//! Owner: model-layers (MoE).
//! Invariants: none beyond the types.

use metrale_gpu_runtime::gpu::mock::MockGpuBackend;
use metrale_gpu_runtime::gpu::{DevicePtr, GpuBackend};

use crate::layers::moe::MoeLayer;
use crate::layers::ops::lora_delta::LoraPair;
use crate::lora::{ExpertLoraLayer, ExpertProj};
use crate::weight_map::DenseWeight;

fn dummy_pair(tag: u64, k_in: u32, n_out: u32) -> LoraPair {
    LoraPair {
        a: DenseWeight {
            weight: DevicePtr(tag),
        },
        b: DenseWeight {
            weight: DevicePtr(tag + 1),
        },
        rank: 8,
        k_in,
        n_out,
        scale: 0.5,
        max_rank: 16,
    }
}

fn u64s(gpu: &MockGpuBackend, p: DevicePtr, n: usize) -> Vec<u64> {
    let mut b = vec![0u8; n * 8];
    gpu.copy_d2h(p, &mut b).unwrap();
    b.chunks_exact(8)
        .map(|c| u64::from_le_bytes(c.try_into().unwrap()))
        .collect()
}

fn f32s(gpu: &MockGpuBackend, p: DevicePtr, n: usize) -> Vec<f32> {
    let mut b = vec![0u8; n * 4];
    gpu.copy_d2h(p, &mut b).unwrap();
    b.chunks_exact(4)
        .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
        .collect()
}

const H: u32 = 2048;
const INTER: u32 = 512;

fn gate_pair(tag: u64) -> LoraPair {
    dummy_pair(tag, H, INTER)
}
fn down_pair(tag: u64) -> LoraPair {
    dummy_pair(tag, INTER, H)
}

#[test]
fn gate_route_dims_are_hidden_to_inter() {
    let gpu = MockGpuBackend::new();
    let mut el = ExpertLoraLayer::default();
    el.pairs.insert((5, ExpertProj::Gate), gate_pair(0x100));
    let route = MoeLayer::build_expert_route(&el, ExpertProj::Gate, &gpu)
        .unwrap()
        .expect("gate pair present => Some route");
    assert_eq!(route.k_in, H);
    assert_eq!(route.n_out, INTER);
    assert_eq!(route.max_rank, 16);
    assert_eq!(route.n_experts, 6);
    assert_eq!(u64s(&gpu, route.a_table, 6), vec![0, 0, 0, 0, 0, 0x100]);
    assert_eq!(u64s(&gpu, route.b_table, 6), vec![0, 0, 0, 0, 0, 0x101]);
    assert_eq!(
        f32s(&gpu, route.scale_table, 6),
        vec![0.0, 0.0, 0.0, 0.0, 0.0, 0.5]
    );
}

#[test]
fn mixed_proj_layer_yields_three_independent_tables() {
    let gpu = MockGpuBackend::new();
    let mut el = ExpertLoraLayer::default();
    el.pairs.insert((5, ExpertProj::Gate), gate_pair(0x10));
    el.pairs.insert((3, ExpertProj::Up), gate_pair(0x20));
    el.pairs.insert((7, ExpertProj::Down), down_pair(0x30));

    let gate = MoeLayer::build_expert_route(&el, ExpertProj::Gate, &gpu)
        .unwrap()
        .unwrap();
    let up = MoeLayer::build_expert_route(&el, ExpertProj::Up, &gpu)
        .unwrap()
        .unwrap();
    let down = MoeLayer::build_expert_route(&el, ExpertProj::Down, &gpu)
        .unwrap()
        .unwrap();
    assert_eq!(gate.n_experts, 6);
    assert_eq!(up.n_experts, 4);
    assert_eq!(down.n_experts, 8);
    assert_eq!((gate.k_in, gate.n_out), (H, INTER));
    assert_eq!((up.k_in, up.n_out), (H, INTER));
    assert_eq!((down.k_in, down.n_out), (INTER, H));
    assert_eq!(u64s(&gpu, gate.a_table, 6), vec![0, 0, 0, 0, 0, 0x10]);
    assert_eq!(u64s(&gpu, up.a_table, 4), vec![0, 0, 0, 0x20]);
    assert_eq!(u64s(&gpu, down.a_table, 8), vec![0, 0, 0, 0, 0, 0, 0, 0x30]);
}

#[test]
fn absent_proj_returns_none() {
    let gpu = MockGpuBackend::new();
    let mut el = ExpertLoraLayer::default();
    el.pairs.insert((1, ExpertProj::Down), down_pair(0x1));
    assert!(
        MoeLayer::build_expert_route(&el, ExpertProj::Gate, &gpu)
            .unwrap()
            .is_none()
    );
    assert!(
        MoeLayer::build_expert_route(&el, ExpertProj::Up, &gpu)
            .unwrap()
            .is_none()
    );
    assert!(
        MoeLayer::build_expert_route(&el, ExpertProj::Down, &gpu)
            .unwrap()
            .is_some()
    );
}
