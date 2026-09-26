// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-26: The dense-MLP weight sets a loader hands to `DenseFfnLayer` (NVFP4, BF16,
//! native FP8, packed Q2) and the gated activation. Re-exported from `dense_ffn`.
//!
//! Owner: model-layers (dense FFN).
//! Invariants:
//! - Every field is the loader's device allocation; nothing here frees it.

use crate::weight_map::{DenseWeight, Fp8Weight, PackedQ2Weight, QuantizedWeight};

pub struct DenseFfnWeights {
    pub gate_proj: QuantizedWeight,
    pub up_proj: QuantizedWeight,
    pub down_proj: QuantizedWeight,
    /// 2026-09-25: Transposed (`[K/2, N]`) copies read by the tile prefill GEMMs (`w4a16_gemm_t*`).
    /// `None` (never built, or freed by `finalize_q4k_load` / `finalize_nvfp4_mmq_load`) leaves
    /// that projection's prefill to the MMQ, W4A4 or int8 arms or the base `w4a16_gemm`.
    /// Decode reads the non-transposed weights above.
    pub gate_proj_t: Option<QuantizedWeight>,
    pub up_proj_t: Option<QuantizedWeight>,
    pub down_proj_t: Option<QuantizedWeight>,
}

/// 2026-09-25: BF16 dense MLP weights. Once installed with `set_bf16_weights`, decode runs
/// `dense_gemv_bf16` and prefill runs cuBLASLt, `dense_gemm_tc` or `dense_gemm_bf16`.
pub struct DenseFfnWeightsBf16 {
    pub gate_proj: DenseWeight,
    pub up_proj: DenseWeight,
    pub down_proj: DenseWeight,
}

/// 2026-09-26: Native block-scaled FP8 E4M3 dense MLP weights. Once installed with
/// `set_fp8_weights`, decode runs the arm `fp8_down::fp8_down_arm` picks and prefill runs
/// the `w8_gemm!` ladder in `prefill_fp8`.
pub struct DenseFfnWeightsFp8 {
    pub gate_proj: Fp8Weight,
    pub up_proj: Fp8Weight,
    pub down_proj: Fp8Weight,
}

/// 2026-09-25: Packed Q2_0 dense MLP weights (`block_q2_0`), kept 2-bit on the device. Once
/// installed with `set_q2_weights`, decode runs `q2_0_gemv_vec` per projection and prefill
/// runs the Q2_0 MMQ GEMM or a BF16 dequant (`forward_prefill_inner`).
pub struct DenseFfnWeightsQ2 {
    pub gate_proj: PackedQ2Weight,
    pub up_proj: PackedQ2Weight,
    pub down_proj: PackedQ2Weight,
}

/// 2026-09-25: Activation of the gated FFN: `SiLU` runs `moe_silu_mul`, `GeLU` runs `gelu_mul`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FfnActivation {
    SiLU,
    GeLU,
}
