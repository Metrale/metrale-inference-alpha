// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-28: The grouped FP8 MoE launch geometries against the `#define`s of the kernels
//! they launch, and the tensor-core shape rule.
//!
//! Owner: model-layers ops.
//! Invariants: none beyond the types.

use super::*;
use crate::layers::ops::{
    FP8_GROUPED_DOWN_TC_W8A8, FP8_GROUPED_GATE_UP_TC_W8A8, Fp8GroupedW8a8Layout,
};

const SCALAR_CU: &str =
    include_str!("../../../../../kernels/gb10/common/moe_shared_expert_fused_fp8_grouped.cu");
const TC_CU: &str = include_str!("../../../../../kernels/gb10/common/moe_fp8_grouped_tc.cu");
const TC_ROWS_CUH: &str =
    include_str!("../../../../../kernels/gb10/common/moe_fp8_grouped_tc_rows.cuh");
const TC_W8A8_CU: &str =
    include_str!("../../../../../kernels/gb10/common/moe_fp8_grouped_tc_w8a8.cu");
const BF16_TC_CU: &str = include_str!("../../../../../kernels/gb10/common/moe_bf16_grouped_tc.cu");
const NVFP4_TC_CU: &str =
    include_str!("../../../../../kernels/gb10/common/moe_nvfp4_grouped_tc.cu");

/// 2026-09-28: The integer value of `#define NAME <int>` in `src`.
fn define(src: &str, name: &str) -> u32 {
    let prefix = format!("#define {name} ");
    let line = src
        .lines()
        .find(|l| l.starts_with(&prefix))
        .unwrap_or_else(|| panic!("no `{prefix}` in the kernel source"));
    line[prefix.len()..]
        .trim()
        .parse()
        .unwrap_or_else(|_| panic!("`{line}` is not an integer define"))
}

/// 2026-09-28: A launch with other columns per CTA, rows per pass or threads than the kernel
/// assumes skips or double-writes output columns and rows, so each geometry is pinned to the
/// defines of its kernel.
#[test]
fn geometries_match_the_kernel_defines() {
    let n_per_block = define(SCALAR_CU, "N_PER_BLOCK");
    assert_eq!(FP8_GROUPED_GATE_UP_SCALAR.cols_per_cta, 2 * n_per_block);
    assert_eq!(
        FP8_GROUPED_GATE_UP_SCALAR.rows_per_pass,
        define(SCALAR_CU, "GU_GROUP_ROWS")
    );
    assert_eq!(
        FP8_GROUPED_GATE_UP_SCALAR.threads,
        define(SCALAR_CU, "BLOCK_SIZE")
    );
    let down_block = define(SCALAR_CU, "DOWN_BLOCK");
    let down_cols = (down_block / define(SCALAR_CU, "WARP_SIZE"))
        * define(SCALAR_CU, "DOWN_COLS_PER_WARP")
        * define(SCALAR_CU, "DOWN_CG");
    assert_eq!(FP8_GROUPED_DOWN_SCALAR.cols_per_cta, down_cols);
    assert_eq!(
        FP8_GROUPED_DOWN_SCALAR.rows_per_pass,
        define(SCALAR_CU, "GROUP_ROWS")
    );
    assert_eq!(FP8_GROUPED_DOWN_SCALAR.threads, down_block);

    let warps = define(TC_CU, "TC_WARPS");
    for (g, mt) in [
        (FP8_GROUPED_GATE_UP_TC, define(TC_CU, "TC_GU_MT")),
        (FP8_GROUPED_DOWN_TC, define(TC_CU, "TC_DOWN_MT")),
    ] {
        assert_eq!(g.cols_per_cta, warps * 16 * mt);
        assert_eq!(g.rows_per_pass, define(TC_ROWS_CUH, "TC_ROWS"));
        assert_eq!(g.threads, warps * 32);
    }
    let warps8 = define(TC_W8A8_CU, "TC8_WARPS");
    for (g, mt) in [
        (FP8_GROUPED_GATE_UP_TC_W8A8, define(TC_W8A8_CU, "TC8_GU_MT")),
        (FP8_GROUPED_DOWN_TC_W8A8, define(TC_W8A8_CU, "TC8_DOWN_MT")),
    ] {
        assert_eq!(g.cols_per_cta, warps8 * 16 * mt);
        assert_eq!(g.rows_per_pass, define(TC_ROWS_CUH, "TC_ROWS"));
        assert_eq!(g.threads, warps8 * 32);
    }
    // 2026-09-28: The W8A8 gate+up writes one activation scale per CTA per row, the scale of a
    // 128-column group of the down projection's K.
    assert_eq!(FP8_GROUPED_GATE_UP_TC_W8A8.cols_per_cta, 128);
}

/// 2026-09-28: The W8A8 activations fit the grouped decode's SiLU buffers (sized for FP32
/// products) at the 35B shape up to 64 rows, without overlap, and a buffer one byte short or a
/// width off the 128 groups is refused.
#[test]
fn w8a8_layout_fits_and_refuses() {
    let (top_k, h, inter) = (8, 2048, 512);
    for m in [1usize, 2, 7, 16, 64] {
        let routed = m * top_k * inter * 4;
        let shared = m * inter * 4;
        let l = Fp8GroupedW8a8Layout::new(m, top_k, h, inter, routed, shared).expect("fits");
        assert!(l.act_s >= m * top_k * inter);
        assert!(l.xq >= l.act_s + m * top_k * (inter / 128) * 4);
        assert!(l.xs >= l.xq + m * h);
        assert!(l.sh_s >= m * inter);
        assert!([l.act_s, l.xq, l.xs, l.sh_s].iter().all(|o| o % 16 == 0));
        let need = l.xs + m * (h / 128) * 4;
        assert!(Fp8GroupedW8a8Layout::new(m, top_k, h, inter, need - 1, shared).is_err());
        let sh_need = l.sh_s + m * (inter / 128) * 4;
        assert!(Fp8GroupedW8a8Layout::new(m, top_k, h, inter, routed, sh_need - 1).is_err());
    }
    assert!(Fp8GroupedW8a8Layout::new(4, top_k, 2000, inter, 1 << 24, 1 << 24).is_err());
}

/// 2026-09-28: The Qwen3.6-35B-A3B projections fit the tensor-core tiles; a projection off
/// the 128 scale blocks or off whole CTAs does not.
#[test]
fn tc_shape_rule() {
    assert!(fp8_grouped_tc_shape_ok(512, 2048, FP8_GROUPED_GATE_UP_TC));
    assert!(fp8_grouped_tc_shape_ok(2048, 512, FP8_GROUPED_DOWN_TC));
    assert!(!fp8_grouped_tc_shape_ok(512, 2000, FP8_GROUPED_GATE_UP_TC));
    assert!(!fp8_grouped_tc_shape_ok(576, 2048, FP8_GROUPED_GATE_UP_TC));
    assert!(!fp8_grouped_tc_shape_ok(320, 512, FP8_GROUPED_DOWN_TC));
    assert!(!fp8_grouped_tc_shape_ok(0, 512, FP8_GROUPED_DOWN_TC));
}

/// 2026-10-02: The NVFP4 point of the tensor-core grouped family: its launch geometry equals the
/// kernel's tile constants, and the 256-K shape check covers both projections' load groups.
#[test]
fn nvfp4_tc_geometry_matches_the_kernel() {
    use crate::layers::ops::{
        NVFP4_GROUPED_DOWN_TC, NVFP4_GROUPED_GATE_UP_TC, nvfp4_grouped_tc_shape_ok,
    };
    let warps = define(NVFP4_TC_CU, "NTC_WARPS");
    for (g, mt) in [
        (NVFP4_GROUPED_GATE_UP_TC, define(NVFP4_TC_CU, "NTC_GU_MT")),
        (NVFP4_GROUPED_DOWN_TC, define(NVFP4_TC_CU, "NTC_DOWN_MT")),
    ] {
        assert_eq!(g.cols_per_cta, warps * 16 * mt);
        assert_eq!(g.rows_per_pass, define(TC_ROWS_CUH, "TC_ROWS"));
        assert_eq!(g.threads, warps * 32);
    }
    for groups in [
        define(NVFP4_TC_CU, "NTC_GU_G"),
        define(NVFP4_TC_CU, "NTC_DOWN_G"),
    ] {
        assert_eq!(256 % (128 * groups), 0);
    }
    assert!(nvfp4_grouped_tc_shape_ok(
        512,
        2048,
        NVFP4_GROUPED_GATE_UP_TC
    ));
    assert!(!nvfp4_grouped_tc_shape_ok(
        512,
        2048 + 128,
        NVFP4_GROUPED_GATE_UP_TC
    ));
}

/// 2026-10-02: The BF16 point of the tensor-core grouped family: launch geometry equals the
/// kernel's tile constants, and the 128-K shape check covers both projections' load groups.
#[test]
fn bf16_tc_geometry_matches_the_kernel() {
    use crate::layers::ops::{
        BF16_GROUPED_DOWN_TC, BF16_GROUPED_GATE_UP_TC, bf16_grouped_tc_shape_ok,
    };
    let warps = define(BF16_TC_CU, "BTC_WARPS");
    for (g, mt) in [
        (BF16_GROUPED_GATE_UP_TC, define(BF16_TC_CU, "BTC_GU_MT")),
        (BF16_GROUPED_DOWN_TC, define(BF16_TC_CU, "BTC_DOWN_MT")),
    ] {
        assert_eq!(g.cols_per_cta, warps * 16 * mt);
        assert_eq!(g.rows_per_pass, define(TC_ROWS_CUH, "TC_ROWS"));
        assert_eq!(g.threads, warps * 32);
    }
    for groups in [
        define(BF16_TC_CU, "BTC_GU_G"),
        define(BF16_TC_CU, "BTC_DOWN_G"),
    ] {
        assert_eq!(128 % (32 * groups), 0);
    }
    assert!(bf16_grouped_tc_shape_ok(512, 2048, BF16_GROUPED_GATE_UP_TC));
    assert!(bf16_grouped_tc_shape_ok(2048, 512, BF16_GROUPED_DOWN_TC));
    assert!(!bf16_grouped_tc_shape_ok(
        2048,
        512 + 64,
        BF16_GROUPED_DOWN_TC
    ));
}
