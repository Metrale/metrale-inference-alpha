// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-25: Tests for the SSM decode W8A8 arm: which `METRALE_CUBLAS_GEMM` families arm it, and over which row band.
//!
//! Owner: model-layers (qwen3 SSM).
//! Invariants: none beyond the types.
//!
//! The shape and capacity rule itself is tested in
//! `ops::dispatch_proj_decode_tests`; this file covers the lever bit and the
//! 5..=16 band at the SSM's widths.

use super::*;
use crate::layers::ops::{
    self, DecodeW8a8Plan, DecodeW8a8Scratch, decode_w8a8_selected, parse_cublas_scope,
};
use crate::weight_map::WeightQuantFormat;
use metrale_gpu_runtime::gpu::{DevicePtr, KernelHandle};

/// 2026-09-25: Qwen3.8-27B: hidden 5120; the fused `in_proj_qkvz` is 16384 wide
/// (2 x 16 key heads x 128 + 2 x 48 value heads x 128).
const H: u32 = 5120;
const QKVZ_N: u32 = 16384;

fn scratch() -> DecodeW8a8Scratch {
    DecodeW8a8Scratch {
        act_fp8: DevicePtr(0x1000),
        act_fp8_bytes: 1 << 20,
        act_scale: DevicePtr(0x2000),
        act_scale_bytes: 1 << 20,
        act_scale_kmajor: DevicePtr(0x3000),
        act_scale_kmajor_bytes: 1 << 20,
        quant_k: ops::Fp8ActQuant::shared_only(KernelHandle(0xA1)),
        scale_kmajor_k: KernelHandle(0xA2),
    }
}

fn qkvz_selected(raw: &str, rows: usize, disabled: bool) -> bool {
    let scope = parse_cublas_scope(Some(raw)).0;
    decode_w8a8_selected(
        ssm_decode_family_armed(scope),
        disabled,
        &DecodeW8a8Plan::contiguous(rows, QKVZ_N, H, 16 * QKVZ_N as usize * 2),
        WeightQuantFormat::Fp8BlockScaled,
        &scratch(),
    )
}

/// 2026-09-25: `ssm` (alone or in a list) and `all`/`1`/`true` arm it; no other
/// spelling does, `ffn` included.
#[test]
fn only_the_ssm_family_arms_the_ssm_decode_arm() {
    for raw in ["ssm", "ffn,ssm", "ssm,attn", "all", "1", "true"] {
        assert!(qkvz_selected(raw, 16, false), "METRALE_CUBLAS_GEMM={raw:?}");
    }
    for raw in ["ffn", "attn", "head", "ffn,attn", "off", "", "junk"] {
        assert!(
            !qkvz_selected(raw, 16, false),
            "METRALE_CUBLAS_GEMM={raw:?}"
        );
    }
}

/// 2026-09-25: The arm's band is 5..=16 rows; 1..=4 and 17+ decline to the GEMV
/// arms.
#[test]
fn the_ssm_decode_arm_takes_five_to_sixteen_rows_only() {
    for rows in [5, 8, 12, 16] {
        assert!(qkvz_selected("ssm", rows, false), "rows={rows}");
    }
    for rows in [1, 2, 4, 17, 32] {
        assert!(!qkvz_selected("ssm", rows, false), "rows={rows}");
    }
}

/// 2026-09-25: The `disabled` flag (`METRALE_NO_W8A8_DECODE_PROJ`) declines even
/// when the family is armed.
#[test]
fn the_kill_switch_beats_an_armed_ssm_family() {
    for rows in [5, 8, 16] {
        assert!(!qkvz_selected("ssm", rows, true), "rows={rows}");
        assert!(!qkvz_selected("all", rows, true), "rows={rows}");
    }
}

/// 2026-09-25: `out_proj` contracts over `value_dim` and writes `hidden`: a
/// different shape under the same rule, including the clause that `K / 128`
/// be a multiple of 4, which a K that is only a multiple of 128 can fail.
#[test]
fn the_out_proj_shape_takes_the_same_rule() {
    let plan = |k: u32| DecodeW8a8Plan::contiguous(16, H, k, 16 * H as usize * 2);
    let go = |k: u32| {
        decode_w8a8_selected(
            ssm_decode_family_armed(parse_cublas_scope(Some("ssm")).0),
            false,
            &plan(k),
            WeightQuantFormat::Fp8BlockScaled,
            &scratch(),
        )
    };
    assert!(go(4096), "value_dim 4096 (a multiple of 512) is eligible");
    assert!(
        !go(640),
        "K/128 must be a multiple of 4 — cuBLAS's own weight-scale stride rule"
    );
}
