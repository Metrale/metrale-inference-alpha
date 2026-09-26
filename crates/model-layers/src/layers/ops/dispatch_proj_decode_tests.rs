// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-25: CPU tests for the 5..=16-row decode W8A8 route and the
//! strided-output write extent.
//!
//! The selector takes the family switch, the kill switch, the shape and the
//! capacities as arguments; only the scale layout comes from the process
//! (`cublas_scale_layout_kmajor`).
//!
//! Owner: model-layers ops.
//! Invariants: none beyond the types.

use super::*;
use crate::weight_map::WeightQuantFormat;
use metrale_gpu_runtime::gpu::{DevicePtr, KernelHandle};

/// 2026-09-25: Qwen3.8-27B hidden size.
const H: u32 = 5120;
/// 2026-09-25: Qwen3.8-27B GDN `in_proj_qkvz` output width: q and k are
/// 16 heads x 128, v and z 48 heads x 128.
const QKVZ_N: u32 = 16384;
/// 2026-09-25: Attention `q_proj` output width (`[Q|gate]`, 24 heads × 256 × 2).
const Q_N: u32 = 12288;
/// 2026-09-25: One sequence's `[Q|K|V]` slot, in BF16 elements: 12288 + 1024 + 1024.
const PER_SEQ_QKV: u32 = 14336;

fn scratch() -> DecodeW8a8Scratch {
    DecodeW8a8Scratch {
        act_fp8: DevicePtr(0x1000),
        act_fp8_bytes: 1 << 20,
        act_scale: DevicePtr(0x2000),
        act_scale_bytes: 1 << 20,
        act_scale_kmajor: DevicePtr(0x3000),
        act_scale_kmajor_bytes: 1 << 20,
        quant_k: Fp8ActQuant::shared_only(KernelHandle(0xA1)),
        scale_kmajor_k: KernelHandle(0xA2),
    }
}

/// 2026-09-25: A contiguous GDN `in_proj_qkvz` at `rows`, with room for 16 rows.
fn ssm_qkvz(rows: usize) -> DecodeW8a8Plan {
    DecodeW8a8Plan::contiguous(rows, QKVZ_N, H, 16 * QKVZ_N as usize * 2)
}

/// 2026-09-25: The attention `q_proj` writing into a `[16, per_seq_qkv]` QKV buffer.
fn attn_q(rows: usize) -> DecodeW8a8Plan {
    DecodeW8a8Plan::strided(rows, Q_N, H, PER_SEQ_QKV, 16 * PER_SEQ_QKV as usize * 2)
}

fn selected(armed: bool, disabled: bool, plan: &DecodeW8a8Plan) -> bool {
    decode_w8a8_selected(
        armed,
        disabled,
        plan,
        WeightQuantFormat::Fp8BlockScaled,
        &scratch(),
    )
}

#[test]
fn the_w8a8_decode_arm_owns_five_to_sixteen_rows() {
    for rows in [5, 6, 8, 12, 16] {
        assert!(selected(true, false, &ssm_qkvz(rows)), "rows={rows}");
        assert!(selected(true, false, &attn_q(rows)), "rows={rows} strided");
    }
}

#[test]
fn the_w8a8_decode_arm_leaves_four_rows_and_below_alone() {
    for rows in [1, 2, 3, 4] {
        assert!(!selected(true, false, &ssm_qkvz(rows)), "rows={rows}");
        assert!(!selected(true, false, &attn_q(rows)), "rows={rows} strided");
    }
}

#[test]
fn the_w8a8_decode_arm_declines_above_the_band() {
    for rows in [17, 24, 32] {
        assert!(!selected(true, false, &ssm_qkvz(rows)), "rows={rows}");
    }
}

#[test]
fn the_w8a8_decode_arm_needs_its_family_armed() {
    assert!(!selected(false, false, &ssm_qkvz(16)));
    assert!(!selected(false, false, &attn_q(16)));
}

#[test]
fn the_kill_switch_deselects_every_row_count_and_family() {
    for rows in [5, 8, 16] {
        assert!(!selected(true, true, &ssm_qkvz(rows)), "rows={rows}");
        assert!(!selected(true, true, &attn_q(rows)), "rows={rows}");
    }
}

/// 2026-09-25: cuBLASLt is given the weight scales as a `[N/128, K/128]` grid,
/// so per-row and single-scale weights are refused.
#[test]
fn the_w8a8_decode_arm_refuses_non_block_scaled_weights() {
    for format in [
        WeightQuantFormat::Fp8PerRow,
        WeightQuantFormat::Fp8SingleScale,
    ] {
        assert!(
            !decode_w8a8_selected(true, false, &ssm_qkvz(16), format, &scratch()),
            "{format:?}"
        );
    }
}

/// 2026-09-25: `N` and `K` must be multiples of 128 (the scale grids), and `K`
/// of 512 (the weight-scale stride rule `blk128x128_stride_ok` checks).
#[test]
fn the_w8a8_decode_arm_refuses_dims_the_scale_grids_do_not_cover() {
    let cases = [
        DecodeW8a8Plan::contiguous(16, 12280, H, 1 << 24),
        DecodeW8a8Plan::contiguous(16, QKVZ_N, 5000, 1 << 24),
        // 2026-09-25: K/128 = 5, not a multiple of 4.
        DecodeW8a8Plan::contiguous(16, QKVZ_N, 640, 1 << 24),
    ];
    for plan in cases {
        assert!(!selected(true, false, &plan), "{plan:?}");
    }
}

#[test]
fn the_strided_write_extent_is_the_last_row_plus_its_width() {
    assert_eq!(strided_out_extent_elems(16, 5120, 5120), 16 * 5120);
    assert_eq!(
        strided_out_extent_elems(16, PER_SEQ_QKV, Q_N),
        15 * PER_SEQ_QKV as usize + Q_N as usize
    );
    assert_eq!(
        strided_out_extent_elems(1, PER_SEQ_QKV, Q_N),
        Q_N as usize,
        "one row must not be charged for a pitch it never crosses"
    );
}

/// 2026-09-25: cuBLASLt writes all `ceil16(rows)` rows, so a strided buffer
/// with slots for the live rows only is refused, and one with 16 slots is
/// accepted.
#[test]
fn a_strided_output_sized_for_the_live_rows_only_is_refused() {
    for rows in [5, 8, 12] {
        let live_only = rows * PER_SEQ_QKV as usize * 2;
        let plan = DecodeW8a8Plan::strided(rows, Q_N, H, PER_SEQ_QKV, live_only);
        assert!(
            !selected(true, false, &plan),
            "rows={rows}: padded write extent {} B must not fit in {live_only} B",
            plan.write_extent_bytes()
        );
        let full =
            DecodeW8a8Plan::strided(rows, Q_N, H, PER_SEQ_QKV, 16 * PER_SEQ_QKV as usize * 2);
        assert!(selected(true, false, &full), "rows={rows} with 16 slots");
    }
}

/// 2026-09-25: K and V start at byte offsets inside each slot, so each
/// projection's capacity is measured from its own base; V, the last, is the
/// tightest.
#[test]
fn each_projection_is_bounded_from_its_own_base() {
    const KV_N: u32 = 1024;
    let arena = 16 * PER_SEQ_QKV as usize * 2;
    let q_bytes = Q_N as usize * 2;
    let kv_bytes = KV_N as usize * 2;
    let v = DecodeW8a8Plan::strided(
        16,
        KV_N,
        H,
        PER_SEQ_QKV,
        arena - q_bytes - kv_bytes, // 2026-09-25: the buffer from V's base on
    );
    assert!(selected(true, false, &v), "V must fit: {v:?}");
    assert_eq!(
        v.write_extent_bytes(),
        (15 * PER_SEQ_QKV as usize + KV_N as usize) * 2
    );
    let tight = DecodeW8a8Plan::strided(16, KV_N, H, PER_SEQ_QKV, v.write_extent_bytes() - 2);
    assert!(!selected(true, false, &tight));
}

#[test]
fn a_row_pitch_narrower_than_the_output_is_refused() {
    let plan = DecodeW8a8Plan::strided(16, Q_N, H, Q_N - 128, 1 << 30);
    assert!(!selected(true, false, &plan));
}

/// 2026-09-25: The GEMM reads `ceil16(M)` activation rows, so a scratch sized
/// for the live rows only is refused.
#[test]
fn the_activation_scratch_is_checked_at_the_padded_row_count() {
    let mut s = scratch();
    let plan = ssm_qkvz(5);
    s.act_fp8_bytes = 5 * H as usize;
    assert!(!decode_w8a8_selected(
        true,
        false,
        &plan,
        WeightQuantFormat::Fp8BlockScaled,
        &s
    ));
    s.act_fp8_bytes = 16 * H as usize;
    assert!(decode_w8a8_selected(
        true,
        false,
        &plan,
        WeightQuantFormat::Fp8BlockScaled,
        &s
    ));
}

/// 2026-09-25: Each missing piece alone declines the route: the quantizer, the
/// FP8 or scale scratch, and, in the K-major layout, the adapter kernel or
/// enough K-major scratch.
#[test]
fn a_missing_scale_layout_adapter_drops_the_arm() {
    let plan = ssm_qkvz(16);
    let mutate: [(&str, fn(&mut DecodeW8a8Scratch)); 5] = [
        ("no quantizer", |s| s.quant_k = Fp8ActQuant::default()),
        ("no fp8 scratch", |s| s.act_fp8 = DevicePtr(0)),
        ("no scale scratch", |s| s.act_scale = DevicePtr(0)),
        ("no kmajor kernel", |s| s.scale_kmajor_k = KernelHandle(0)),
        ("kmajor too small", |s| s.act_scale_kmajor_bytes = 4),
    ];
    for (what, apply) in mutate {
        let mut s = scratch();
        apply(&mut s);
        // 2026-09-25: The two K-major clauses bind only in the K-major layout,
        // the one used when `METRALE_CUBLAS_SCALE_LAYOUT` is unset.
        if !cublas_scale_layout_kmajor() && what.contains("kmajor") {
            continue;
        }
        assert!(
            !decode_w8a8_selected(true, false, &plan, WeightQuantFormat::Fp8BlockScaled, &s),
            "{what}"
        );
    }
}

#[test]
fn every_row_in_the_band_pads_to_sixteen() {
    for rows in 5..=16usize {
        assert_eq!(ssm_qkvz(rows).m_pad(), 16, "rows={rows}");
    }
}
