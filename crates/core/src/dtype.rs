// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-25: Element types for weights and scales, and the quantization layout of a weight tensor.
//!
//! Owner: core.
//! Invariants: none beyond the types.

use serde::{Deserialize, Serialize};

/// 2026-09-25: Element type of a weight or scale tensor; the width is
/// [`DType::element_size_bits`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum DType {
    /// 2026-09-25: 4-bit float, 1 sign + 2 exponent + 1 mantissa bits; the
    /// weight type of [`QuantConfig::nvfp4`].
    E2M1,

    /// 2026-09-25: 8-bit float, 1 sign + 4 exponent + 3 mantissa bits; the
    /// block-scale type of [`QuantConfig::nvfp4`] and the weight type of
    /// [`QuantConfig::fp8`].
    FP8E4M3,

    /// 2026-09-25: 8-bit float, 1 sign + 5 exponent + 2 mantissa bits.
    FP8E5M2,

    BF16,

    FP16,

    FP32,
}

impl DType {
    /// 2026-09-25: Width of one element in bits (4 for `E2M1`).
    pub const fn element_size_bits(&self) -> usize {
        match self {
            DType::E2M1 => 4,
            DType::FP8E4M3 | DType::FP8E5M2 => 8,
            DType::BF16 | DType::FP16 => 16,
            DType::FP32 => 32,
        }
    }

    pub const fn elements_per_u32(&self) -> usize {
        32 / self.element_size_bits()
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QuantConfig {
    pub weight_type: DType,

    pub scale_type: DType,

    /// 2026-09-25: Weights per scale factor; 0 means one scale for the whole
    /// tensor ([`QuantConfig::fp8`]).
    pub group_size: usize,

    pub global_scale_type: DType,
}

impl QuantConfig {
    /// 2026-09-25: NVFP4: E2M1 weights, one FP8E4M3 scale per 16 weights, and
    /// an FP32 tensor scale.
    pub fn nvfp4() -> Self {
        Self {
            weight_type: DType::E2M1,
            scale_type: DType::FP8E4M3,
            group_size: 16,
            global_scale_type: DType::FP32,
        }
    }

    /// 2026-09-25: FP8E4M3 weights with FP32 per-tensor scales.
    pub fn fp8() -> Self {
        Self {
            weight_type: DType::FP8E4M3,
            scale_type: DType::FP32,
            group_size: 0,
            global_scale_type: DType::FP32,
        }
    }
}
