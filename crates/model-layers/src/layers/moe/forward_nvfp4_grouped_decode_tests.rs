// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-27: Shape admission of the grouped NVFP4 MoE decode (`forward_nvfp4_grouped_decode.rs`).
//!
//! Owner: model-layers (MoE).
//! Invariants: none beyond the types.

use super::*;

/// 2026-09-27: The admitted widths and the shape terms, each refused alone.
#[test]
fn shape_admission() {
    let max = NVFP4_GROUPED_DECODE_MAX_ROWS;
    assert!(nvfp4_grouped_decode_shape_ok(1, max, 2048, 512, 512));
    assert!(nvfp4_grouped_decode_shape_ok(max, max, 2048, 512, 512));
    assert!(!nvfp4_grouped_decode_shape_ok(0, max, 2048, 512, 512));
    assert!(!nvfp4_grouped_decode_shape_ok(max + 1, max, 2048, 512, 512));
    assert!(!nvfp4_grouped_decode_shape_ok(4, max, 2048 + 16, 512, 512));
    assert!(!nvfp4_grouped_decode_shape_ok(
        4,
        max,
        2048,
        512 + 8,
        512 + 8
    ));
    assert!(!nvfp4_grouped_decode_shape_ok(4, max, 2048, 512, 1024));
    // 2026-10-02: The tensor-core kernels' envelope: 256 rows, and the 35B's gate+up
    // (512 x 2048) and down (2048 x 512) fit their tiles; a K that is not a whole load
    // group of 256 does not.
    let tc = NVFP4_GROUPED_DECODE_TC_MAX_ROWS;
    assert_eq!(tc, 256);
    assert!(nvfp4_grouped_decode_shape_ok(tc, tc, 2048, 512, 512));
    assert!(!nvfp4_grouped_decode_shape_ok(tc + 1, tc, 2048, 512, 512));
    assert!(ops::nvfp4_grouped_tc_shape_ok(
        512,
        2048,
        ops::NVFP4_GROUPED_GATE_UP_TC
    ));
    assert!(ops::nvfp4_grouped_tc_shape_ok(
        2048,
        512,
        ops::NVFP4_GROUPED_DOWN_TC
    ));
    assert!(!ops::nvfp4_grouped_tc_shape_ok(
        2048,
        384,
        ops::NVFP4_GROUPED_DOWN_TC
    ));
    assert!(!ops::nvfp4_grouped_tc_shape_ok(
        96,
        2048,
        ops::NVFP4_GROUPED_GATE_UP_TC
    ));
}
