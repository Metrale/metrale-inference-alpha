// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-25: Tests for the K=2 dispatch helpers `batch2_block_width` and `k2_e8m0_needs_per_token`.
//!
//! Owner: model-layers (MoE).
//! Invariants: none beyond the types.

use super::{batch2_block_width, k2_e8m0_needs_per_token};
use crate::weight_map::WeightQuantFormat;

#[test]
fn batch2_block_width_switches_at_3072_for_supported_models() {
    assert_eq!(batch2_block_width(1024), 128);
    assert_eq!(batch2_block_width(2048), 128);
    assert_eq!(batch2_block_width(2816), 128);
    assert_eq!(batch2_block_width(3071), 128);
    assert_eq!(batch2_block_width(3072), 256);
    assert_eq!(batch2_block_width(4096), 256);
    assert_eq!(batch2_block_width(5120), 256);
    assert_eq!(batch2_block_width(7168), 256);
}

#[test]
fn no_e8m0_tensor_reaches_gs16_batch2() {
    // 2026-09-25: Every `WeightQuantFormat` variant except `PackedQ2_0`; of
    // these, only Mxfp4E8m0 is routed away from the batch2_t kernel.
    let all = [
        WeightQuantFormat::Bf16,
        WeightQuantFormat::Fp8PerRow,
        WeightQuantFormat::Fp8BlockScaled,
        WeightQuantFormat::Fp8SingleScale,
        WeightQuantFormat::Nvfp4,
        WeightQuantFormat::Mxfp4E8m0,
    ];
    for f in all {
        assert_eq!(
            k2_e8m0_needs_per_token(f),
            f == WeightQuantFormat::Mxfp4E8m0,
            "only Mxfp4E8m0 diverts to the per-token path"
        );
    }
}
