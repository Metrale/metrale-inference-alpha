// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-25: Tests for `TokenOverlaySet::from_slots`: the per-slot pointer
//! and count tables, read back from `MockGpuBackend`, the tied and untied
//! lm_head cells, the vocab bound, and the mixed-vocab refusal.
//!
//! Owner: model-layers (lora).
//! Invariants: none beyond the types.

use metrale_gpu_runtime::gpu::mock::MockGpuBackend;
use metrale_gpu_runtime::gpu::{DevicePtr, GpuBackend};

use super::*;
use crate::lora::overlay_build::{EmbedOverlay, LmHeadOverlay};

fn u64s(gpu: &MockGpuBackend, p: DevicePtr, n: usize) -> Vec<u64> {
    let mut b = vec![0u8; n * 8];
    gpu.copy_d2h(p, &mut b).unwrap();
    b.chunks_exact(8)
        .map(|c| u64::from_le_bytes(c.try_into().unwrap()))
        .collect()
}

fn u32s(gpu: &MockGpuBackend, p: DevicePtr, n: usize) -> Vec<u32> {
    let mut b = vec![0u8; n * 4];
    gpu.copy_d2h(p, &mut b).unwrap();
    b.chunks_exact(4)
        .map(|c| u32::from_le_bytes(c.try_into().unwrap()))
        .collect()
}

fn embed(rows: u64, ids: u64, sm: u64, n: u32, lm: Option<LmHeadOverlay>) -> EmbedOverlay {
    EmbedOverlay {
        rows: DevicePtr(rows),
        ids_dev: DevicePtr(ids),
        slot_map: DevicePtr(sm),
        n_override: n,
        vocab: 640,
        lmhead: lm,
    }
}

#[test]
fn tied_reuses_embed_rows_and_ids() {
    let gpu = MockGpuBackend::new();
    let ov = embed(0xA0, 0xB0, 0xC0, 3, None);
    let set = TokenOverlaySet::from_slots(&gpu, vec![Some(ov), None], 2, true).unwrap();
    assert_eq!(u64s(&gpu, set.embed_rows_table, 2), vec![0xA0, 0]);
    assert_eq!(u64s(&gpu, set.embed_slot_map_table, 2), vec![0xC0, 0]);
    assert_eq!(u64s(&gpu, set.lmhead_rows_table, 2), vec![0xA0, 0]);
    assert_eq!(u64s(&gpu, set.lmhead_ids_table, 2), vec![0xB0, 0]);
    assert_eq!(u32s(&gpu, set.n_override_table, 2), vec![3, 0]);
    assert_eq!(set.max_n_override, 3);
    assert!(set.any_active());
}

#[test]
fn untied_without_lmhead_zeroes_lmhead_count() {
    let gpu = MockGpuBackend::new();
    let ov = embed(0xA0, 0xB0, 0xC0, 3, None);
    let set = TokenOverlaySet::from_slots(&gpu, vec![Some(ov)], 1, false).unwrap();
    assert_eq!(u64s(&gpu, set.embed_rows_table, 1), vec![0xA0]);
    assert_eq!(u32s(&gpu, set.n_override_table, 1), vec![0]);
    assert_eq!(u64s(&gpu, set.lmhead_rows_table, 1), vec![0]);
    assert_eq!(set.max_n_override, 0);
}

#[test]
fn untied_with_lmhead_uses_distinct_rows() {
    let gpu = MockGpuBackend::new();
    let lm = LmHeadOverlay {
        rows: DevicePtr(0xD0),
        ids_dev: DevicePtr(0xE0),
        n_override: 2,
    };
    let ov = embed(0xA0, 0xB0, 0xC0, 3, Some(lm));
    let set = TokenOverlaySet::from_slots(&gpu, vec![Some(ov)], 1, false).unwrap();
    assert_eq!(u64s(&gpu, set.lmhead_rows_table, 1), vec![0xD0]);
    assert_eq!(u64s(&gpu, set.lmhead_ids_table, 1), vec![0xE0]);
    assert_eq!(u32s(&gpu, set.n_override_table, 1), vec![2]);
    assert_eq!(set.max_n_override, 2);
    // 2026-09-25: The embed count (3) and the lm_head count (2) differ here, so
    // this catches the two tables being swapped.
    assert_eq!(u32s(&gpu, set.embed_n_table, 1), vec![3]);
}

// 2026-09-25: `embed_n_table` and `vocab` are the embed kernel's two bounds
// (`token_overlay.cu`).
#[test]
fn embed_bound_table_and_vocab_are_published() {
    let gpu = MockGpuBackend::new();
    let ov = embed(0xA0, 0xB0, 0xC0, 3, None);
    let set = TokenOverlaySet::from_slots(&gpu, vec![Some(ov), None], 2, true).unwrap();
    assert_eq!(u32s(&gpu, set.embed_n_table, 2), vec![3, 0]);
    assert_eq!(set.vocab, 640);

    let empty = TokenOverlaySet::from_slots(&gpu, vec![None, None], 2, true).unwrap();
    assert_eq!(empty.vocab, 0);
    assert_eq!(u32s(&gpu, empty.embed_n_table, 2), vec![0, 0]);
}

#[test]
fn mixed_vocab_slots_are_refused() {
    let gpu = MockGpuBackend::new();
    let a = embed(0xA0, 0xB0, 0xC0, 3, None);
    let mut b = embed(0xA1, 0xB1, 0xC1, 1, None);
    b.vocab = 768;
    let err = match TokenOverlaySet::from_slots(&gpu, vec![Some(a), Some(b)], 2, true) {
        Err(e) => e,
        Ok(_) => panic!("mixed-vocab overlay slots must be refused"),
    };
    assert!(err.to_string().contains("mixed-vocab"), "{err}");
}

#[test]
fn empty_pool_is_inert() {
    let gpu = MockGpuBackend::new();
    let set = TokenOverlaySet::from_slots(&gpu, vec![None, None], 2, true).unwrap();
    assert!(!set.any_active());
    assert_eq!(set.max_n_override, 0);
    assert_eq!(u64s(&gpu, set.embed_rows_table, 2), vec![0, 0]);
}
