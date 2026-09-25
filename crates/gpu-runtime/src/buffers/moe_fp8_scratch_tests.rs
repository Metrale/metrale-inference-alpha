// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-25: Tests for the grouped FP8 MoE scratch slab: its layout covers every
//! live region, the arena hands out the same addresses without allocating or
//! synchronizing, and the FP8 prefill body no longer allocates its own temporaries.
//!
//! Owner: gpu-runtime (buffer arena).
//! Invariants: none beyond the types.

use super::*;
use crate::gpu::mock::MockGpuBackend;
use metrale_core::scope::ModelResource;

fn config() -> ModelConfig {
    let mut c = ModelConfig::qwen3_next_80b_nvfp4();
    c.hidden_size = 2048;
    c.num_experts = 256;
    c.num_experts_per_tok = 8;
    c.moe_intermediate_size = 512;
    c.shared_expert_intermediate_size = 512;
    c
}

#[test]
fn grouped_fp8_scratch_covers_all_live_regions_and_shapes() {
    let mut c = config();
    for shared in [0, 512, 4096] {
        c.shared_expert_intermediate_size = shared;
        let capacity = Layout::new(&c, 2176).bytes;
        for rows in [1, 32, 64, 65, 128, 2176] {
            let l = Layout::new(&c, rows);
            assert!(l.bytes <= capacity);
            // 2026-09-25: Independent inventories of the four activation consumers:
            // shared gate/up, shared down, routed gate/up, expanded down.
            for (m, k) in [(rows, 2048usize), (rows, shared), (rows * 8, 512)] {
                assert!(m * k <= l.scale);
                assert!(m * k.div_ceil(128) * 4 <= l.worklist - l.scale);
            }
            // 2026-09-25: Worst-case all expert rounding plus gate/up and down N tiles.
            for width in [512usize, 2048] {
                let bytes = ((rows * 8).div_ceil(128) + 256 + 1) * width.div_ceil(64) * 8;
                assert!(bytes <= l.counter - l.worklist);
            }
            assert_eq!(l.bytes - l.small_counter, 4);
            assert!(l.counter + 4 <= l.small_worklist);
            assert!(l.small_worklist + l.small_bytes <= l.small_counter);
            assert_eq!(l.scale % 256, 0);
            assert_eq!(l.worklist % 256, 0);
            assert_eq!(l.counter % 256, 0);
        }
    }
}

#[test]
fn grouped_fp8_scratch_is_reused_without_alloc_or_sync_and_released() {
    let c = config();
    let gpu = MockGpuBackend::new();
    let before = gpu.alloc_count();
    let mut arena = BufferArena::new(&c, 8, 128, 16, 8, &gpu).unwrap();
    let allocated = gpu.alloc_count();
    let syncs = gpu.sync_count();
    let first = arena.moe_fp8_scratch(&c, 8).unwrap();
    assert_ne!(first.small_worklist, DevicePtr::NULL);
    assert_ne!(first.small_total_tiles, DevicePtr::NULL);
    assert_eq!(first.small_worklist_bytes, 65536);
    assert!(first.total_tiles.0 + 4 <= first.small_worklist.0);
    assert!(
        first.small_worklist.0 + first.small_worklist_bytes as u64 <= first.small_total_tiles.0
    );
    for _ in 0..3 {
        let other = arena.moe_fp8_scratch(&c, 8).unwrap();
        assert_eq!(first.activation, other.activation);
        assert_eq!(first.scales, other.scales);
        assert_eq!(first.worklist, other.worklist);
        assert_eq!(first.total_tiles, other.total_tiles);
        assert_eq!(first.small_worklist, other.small_worklist);
        assert_eq!(first.small_total_tiles, other.small_total_tiles);
        assert_eq!(gpu.alloc_count(), allocated);
        assert_eq!(gpu.sync_count(), syncs);
    }
    assert!(arena.moe_fp8_scratch(&c, 9).is_err());
    let mut oversized = c.clone();
    oversized.moe_intermediate_size *= 100;
    assert!(arena.moe_fp8_scratch(&oversized, 8).is_err());
    arena.release(&gpu).unwrap();
    assert_eq!(gpu.alloc_count(), before);
    assert!(arena.moe_fp8_scratch(&c, 8).is_err());
}

#[test]
fn dense_model_has_no_grouped_fp8_scratch_charge() {
    let mut c = config();
    c.num_experts = 0;
    assert_eq!(Layout::new(&c, 2176).bytes, 0);
}

#[test]
fn grouped_fp8_path_cannot_reintroduce_capture_unsafe_temporaries() {
    // 2026-09-25: A structural guard beside the real arena lifecycle test: every
    // variant (shared/routed, W8A16/W8A8) lives in `forward_prefill_fp8` and its three
    // child modules. GPU qualification must still establish replay numerics and
    // changing route counts. The two synchronizes are the profile start and the
    // `mprof_step!` timer, both inert unless `--profile` outside graph capture.
    let source = [
        include_str!("../../../model-layers/src/layers/moe/forward_prefill_fp8.rs"),
        include_str!("../../../model-layers/src/layers/moe/forward_prefill_fp8/shared.rs"),
        include_str!("../../../model-layers/src/layers/moe/forward_prefill_fp8/gate_up.rs"),
        include_str!("../../../model-layers/src/layers/moe/forward_prefill_fp8/down.rs"),
    ]
    .concat();
    assert!(!source.contains("ctx.gpu.alloc("));
    assert!(!source.contains("ctx.gpu.free("));
    assert!(source.contains("let profile = ctx.profile && !ctx.graph_capture;"));
    assert_eq!(source.matches("gpu.synchronize(").count(), 2);
    let kernel = include_str!("../../../../kernels/gb10/common/moe_permute.cu");
    assert!(kernel.contains("total_tiles[0] = (int)w;"));
}

#[test]
fn grouped_fp8_scratch_is_charged_to_the_memory_budget() {
    let c = config();
    let mut sizes = super::super::BufferSizes::from_config(&c, 2176, 2048, 16, 128);
    let charged = sizes.moe_fp8_scratch;
    let total = sizes.total_bytes();
    assert_eq!(charged, Layout::new(&c, 2176).bytes);
    assert!(charged > 0);
    sizes.moe_fp8_scratch = 0;
    assert_eq!(sizes.total_bytes() + charged, total);
}

#[test]
fn dense_arena_omits_moe_scratch_and_releases_cleanly() {
    let mut c = config();
    c.num_experts = 0;
    let gpu = MockGpuBackend::new();
    let before = gpu.alloc_count();
    let mut arena = BufferArena::new(&c, 8, 128, 16, 8, &gpu).unwrap();
    assert_eq!(arena.moe_fp8_scratch, DevicePtr::NULL);
    assert!(arena.moe_fp8_scratch(&c, 8).is_err());
    arena.release(&gpu).unwrap();
    assert_eq!(gpu.alloc_count(), before);
}

#[test]
fn adaptive_small_list_capacity_scales_and_rejects_arithmetic_overflow() {
    assert_eq!(small_list_bytes(256, 2048), Some(65536));
    assert_eq!(small_list_bytes(256, 4096), Some(131072));
    assert_eq!(small_list_bytes(256, 4097), Some(133120));
    assert_eq!(small_list_bytes(usize::MAX, 128), None);
    assert_eq!(small_list_bytes(256, usize::MAX), None);
    let mut c = config();
    for width in [2048, 4096, 4097] {
        c.hidden_size = width;
        let l = Layout::new(&c, 128);
        assert_eq!(l.small_bytes, 256 * width.div_ceil(64) * 8);
        assert!(l.small_worklist >= l.counter + 4);
        assert_eq!(l.bytes, l.small_counter + 4);
    }
    c.num_experts = 128;
    let l = Layout::new(&c, 128);
    assert_eq!(l.small_bytes, 0);
    assert_eq!(l.bytes, l.counter + 4);
}
