// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-25: Tests for the K=3 E8M0 dispatch guard `k3_e8m0_needs_per_token`.
//!
//! Owner: model-layers (MoE).
//! Invariants: none beyond the types.

use super::k3_e8m0_needs_per_token;
use crate::weight_map::WeightQuantFormat;

#[test]
fn no_e8m0_tensor_reaches_gs16_batch3() {
    // 2026-09-25: Every `WeightQuantFormat` variant except `PackedQ2_0`; of
    // these, only Mxfp4E8m0 is routed away from the batch3_t kernel.
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
            k3_e8m0_needs_per_token(f),
            f == WeightQuantFormat::Mxfp4E8m0,
            "only Mxfp4E8m0 diverts to the per-token path"
        );
    }
}
