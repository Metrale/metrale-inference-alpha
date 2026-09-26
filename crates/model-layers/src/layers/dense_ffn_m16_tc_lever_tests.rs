// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-25: Tests of `resolve_m16_tc_levers` and `m16_tc_kernel`, the pure halves of the m16 tier's lever.
//!
//! Owner: model-layers (dense FFN).
//! Invariants: none beyond the types.
//!
//! Which arm a row count takes is `dense_ffn_m16_tc_tests.rs`; how the
//! variables fold into the resolved bools is `ops/target_defaults_m16_tests.rs`.

use super::{m16_tc_kernel, resolve_m16_tc_levers};
use crate::layers::ops::{W8A16_GEMM_M16_N_TILE, W8A16_GEMM_M16_N_TILE_WIDE};
use metrale_gpu_runtime::gpu::KernelHandle;

/// 2026-09-25: Distinct handles, so the returned handle tells the two tiles
/// apart.
const M16TC_K: u64 = 0x167C;
const M16TC_N64_K: u64 = 0x1640;

// 2026-09-25: These drive the pure resolver, never the environment: the
// production accessor is a process-global `OnceLock`, so a test that set the
// variables would leak into every other test in this binary.

#[test]
fn no_lever_set_leaves_every_tier_off() {
    let l = resolve_m16_tc_levers(false, false, None);
    assert!(!l.ffn, "the FFN arm must default OFF");
    assert!(!l.attn, "the attention tiers must default OFF");
    assert_eq!(l.ffn_n_tile, W8A16_GEMM_M16_N_TILE, "default tile is 32");
}

/// 2026-09-25: The FFN input does not reach the attention field.
#[test]
fn the_ffn_lever_reaches_the_ffn_arm_only() {
    let l = resolve_m16_tc_levers(true, false, None);
    assert!(l.ffn);
    assert!(
        !l.attn,
        "METRALE_FFN_M16_TC must leave the attention tiers alone"
    );
}

/// 2026-09-25: The attention input does not reach the FFN field.
#[test]
fn the_attn_lever_reaches_the_attention_tiers_only() {
    let l = resolve_m16_tc_levers(false, true, None);
    assert!(l.attn);
    assert!(!l.ffn, "METRALE_ATTN_M16_TC must leave the dense FFN alone");
}

/// 2026-09-25: `ffn` and `attn` are independent inputs; the `METRALE_M16_TC`
/// umbrella is folded in earlier, by `ops::target_defaults::resolve`.
#[test]
fn the_two_families_do_not_leak_into_each_other() {
    for (ffn, attn) in [(false, false), (true, false), (false, true), (true, true)] {
        let l = resolve_m16_tc_levers(ffn, attn, None);
        assert_eq!((l.ffn, l.attn), (ffn, attn), "ffn={ffn} attn={attn}");
    }
}

#[test]
fn the_n_tile_lever_selects_the_wide_instantiation() {
    assert_eq!(
        resolve_m16_tc_levers(true, false, Some("64")).ffn_n_tile,
        W8A16_GEMM_M16_N_TILE_WIDE
    );
    // 2026-09-25: Any other value keeps 32, including an explicit 32, an
    // empty value and a typo.
    for raw in ["32", "", "128", "yes", "6 4"] {
        assert_eq!(
            resolve_m16_tc_levers(true, false, Some(raw)).ffn_n_tile,
            W8A16_GEMM_M16_N_TILE,
            "METRALE_FFN_M16_TC_NTILE={raw:?} must fall back to 32"
        );
    }
}

/// 2026-09-25: The tile picks the entry point, and without the wide handle it
/// falls back to the 32-wide kernel instead of launching a zero handle.
#[test]
fn the_wide_tile_falls_back_when_its_entry_point_is_absent() {
    let (_, kernel, tile) = m16_tc_kernel(
        W8A16_GEMM_M16_N_TILE_WIDE,
        KernelHandle(M16TC_K),
        KernelHandle(M16TC_N64_K),
    );
    assert_eq!(kernel.0, M16TC_N64_K);
    assert_eq!(tile, W8A16_GEMM_M16_N_TILE_WIDE);

    let (_, kernel, tile) = m16_tc_kernel(
        W8A16_GEMM_M16_N_TILE_WIDE,
        KernelHandle(M16TC_K),
        KernelHandle(0),
    );
    assert_eq!(
        kernel.0, M16TC_K,
        "no n64 entry point => the 32-wide kernel"
    );
    assert_eq!(tile, W8A16_GEMM_M16_N_TILE);

    let (_, kernel, tile) = m16_tc_kernel(
        W8A16_GEMM_M16_N_TILE,
        KernelHandle(M16TC_K),
        KernelHandle(M16TC_N64_K),
    );
    assert_eq!(
        kernel.0, M16TC_K,
        "the default tile never reaches the wide arm"
    );
    assert_eq!(tile, W8A16_GEMM_M16_N_TILE);
}
