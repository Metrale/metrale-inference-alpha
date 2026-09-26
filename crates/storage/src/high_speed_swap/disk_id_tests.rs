// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-25: Tests of the disk-block-id allocator. `HighSpeedSwap::new`
//! allocates device memory, so they need a GPU.
//!
//! Owner: metrale-storage high-speed swap.
//! Invariants: none beyond the types.

use super::*;
use crate::cuda_min::CudaCtx;

fn dims() -> ModelDims {
    ModelDims {
        num_layers: 2,
        max_blocks_per_layer: 8,
        num_q_heads: 32,
        num_kv_heads: 8,
        head_dim: 128,
        block_size: 16,
        model_fp: None,
    }
}

fn cfg(dir: &str) -> HighSpeedSwapConfig {
    HighSpeedSwapConfig {
        dir: std::env::temp_dir().join(format!("metrale-hss-disk-id-{dir}-{}", std::process::id())),
        bytes: 64 * (1 << 20),
        resident_blocks: 4,
        rank: 32,
        qd: 4,
        graph: false,
        projection_seed: 0xCAFE_F00D,
    }
}

#[test]
#[ignore = "requires GPU"]
fn alloc_free_round_trip() {
    let _ctx = CudaCtx::new(0).expect("cuda init");
    let mut hss = HighSpeedSwap::new(&_ctx, cfg("rt"), dims()).unwrap();
    let ids: Vec<u32> = (0..8).map(|_| hss.alloc_disk_block_id().unwrap()).collect();
    assert_eq!(ids, (0..8).collect::<Vec<_>>());
    assert!(hss.alloc_disk_block_id().is_none());
    hss.dec_disk_ref(3);
    let reused = hss.alloc_disk_block_id().unwrap();
    assert_eq!(reused, 3);
}

#[test]
#[ignore = "requires GPU"]
fn ref_counting_holds() {
    let _ctx = CudaCtx::new(0).expect("cuda init");
    let mut hss = HighSpeedSwap::new(&_ctx, cfg("rc"), dims()).unwrap();
    let id = hss.alloc_disk_block_id().unwrap();
    assert_eq!(hss.disk_refcount(id), 1);
    hss.inc_disk_ref(id);
    hss.inc_disk_ref(id);
    assert_eq!(hss.disk_refcount(id), 3);
    assert_eq!(hss.dec_disk_ref(id), 2);
    assert_eq!(hss.dec_disk_ref(id), 1);
    assert_eq!(hss.dec_disk_ref(id), 0);
    let reused = hss.alloc_disk_block_id().unwrap();
    assert_eq!(reused, id, "freed id should be reused");
}

#[test]
#[ignore = "requires GPU"]
fn capacity_exhaustion_then_recovery() {
    let _ctx = CudaCtx::new(0).expect("cuda init");
    let mut hss = HighSpeedSwap::new(&_ctx, cfg("cap"), dims()).unwrap();
    let ids: Vec<u32> = (0..8).map(|_| hss.alloc_disk_block_id().unwrap()).collect();
    assert!(hss.alloc_disk_block_id().is_none());
    for &id in &ids[..3] {
        hss.dec_disk_ref(id);
    }
    assert_eq!(hss.disk_free_count(), 3);
    let r = hss.alloc_disk_block_id().unwrap();
    assert!(ids[..3].contains(&r));
    assert_eq!(hss.disk_free_count(), 2);
}
