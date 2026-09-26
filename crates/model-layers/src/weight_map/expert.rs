// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-25: Per-layer weight structs (attention, SSM, MoE experts) and
//! `QuantWeight`, the one enum layer dispatch matches over NVFP4, FP8, dense
//! BF16 and packed Q2 weights.
//!
//! Owner: model-layers (weight loading).
//! Invariants: none beyond the types.

#![allow(unused_imports)]

use anyhow::{Context, Result, bail, ensure};
use metrale_gpu_runtime::gpu::{DevicePtr, GpuBackend};
use metrale_model_weights::weights::{WeightDtype, WeightStore};

use super::*;

/// 2026-09-25: FP8 expert weight: gate/up/down projections as `Fp8Weight`s.
#[derive(Debug, Clone, Copy)]
pub struct Fp8ExpertWeight {
    pub gate_proj: Fp8Weight,
    pub up_proj: Fp8Weight,
    pub down_proj: Fp8Weight,
}

/// 2026-09-25: Full-attention layer weights.
#[derive(Debug, Clone, Copy)]
pub struct AttentionWeights {
    pub q_proj: DenseWeight,
    pub k_proj: DenseWeight,
    pub v_proj: DenseWeight,
    pub o_proj: QuantizedWeight,
    /// 2026-09-25: Per-head Q RMSNorm weight, `[head_dim]`.
    pub q_norm: DenseWeight,
    /// 2026-09-25: Per-head K RMSNorm weight, `[head_dim]`.
    pub k_norm: DenseWeight,
    /// 2026-09-25: Full-width Q RMSNorm weight, `[num_heads * head_dim]`: one
    /// norm over the whole Q projection before RoPE. The MiniMax loader sets it;
    /// when `Some`, the attention forward applies it instead of the per-head
    /// `q_norm`.
    pub q_norm_full: Option<DenseWeight>,
    /// 2026-09-25: Full-width K RMSNorm weight, `[num_kv_heads * head_dim]`.
    pub k_norm_full: Option<DenseWeight>,
    /// 2026-09-25: K scale for the FP8 KV cache, used when no FP8 calibration is
    /// loaded.
    pub k_scale: f32,
    /// 2026-09-25: V scale for the FP8 KV cache, as `k_scale`.
    pub v_scale: f32,
}

/// 2026-09-25: Linear-attention (Gated DeltaNet) layer weights.
#[derive(Debug, Clone, Copy)]
pub struct SsmWeights {
    pub in_proj_qkvz: DenseWeight,
    pub in_proj_ba: DenseWeight,
    pub conv1d: DenseWeight,
    pub a_log: DenseWeight,
    pub dt_bias: DenseWeight,
    pub norm: DenseWeight,
    pub out_proj: QuantizedWeight,
}

/// 2026-09-25: One MoE expert's NVFP4 gate/up/down projections.
#[derive(Debug, Clone, Copy)]
pub struct ExpertWeight {
    pub gate_proj: QuantizedWeight,
    pub up_proj: QuantizedWeight,
    pub down_proj: QuantizedWeight,
}

impl ExpertWeight {
    /// 2026-09-25: Null expert, every pointer NULL. Loaders use it for experts
    /// this rank does not own under expert parallelism, and the MoE kernels
    /// skip a NULL expert.
    pub fn null() -> Self {
        Self {
            gate_proj: QuantizedWeight::null(),
            up_proj: QuantizedWeight::null(),
            down_proj: QuantizedWeight::null(),
        }
    }
}

/// 2026-09-25: A weight in any supported format. `ops::quant_gemv` /
/// `ops::quant_gemm` match on it to pick the kernel.
#[derive(Debug, Clone, Copy)]
pub enum QuantWeight {
    /// 2026-09-25: NVFP4 E2M1: packed nibbles + FP8 group scales + f32 global
    /// scale. Kernel: `w4a16_gemv` / `w4a16_gemm`.
    Nvfp4(QuantizedWeight),

    /// 2026-09-25: FP8 E4M3 weights and their scale buffer. Kernel: `w8a16_gemv` /
    /// `w8a16_gemm`.
    Fp8(Fp8Weight),

    /// 2026-09-25: BF16 dense (unquantized). Kernel: `dense_gemv` / `dense_gemm`.
    Dense(DenseWeight),

    /// 2026-09-25: Keep-packed Q2_0: raw `block_q2_0` bytes kept 2-bit on the
    /// device. `quant_gemv` / `quant_gemm` refuse it: decode runs
    /// `q2_0_gemv_vec` and prefill dequantizes to BF16 first.
    PackedQ2(PackedQ2Weight),
}

impl QuantWeight {
    /// 2026-09-25: An NVFP4 weight with NULL pointers.
    pub fn null() -> Self {
        Self::Nvfp4(QuantizedWeight::null())
    }

    /// 2026-09-25: Whether this weight's data pointer is NULL.
    pub fn is_null(&self) -> bool {
        match self {
            Self::Nvfp4(w) => w.is_null(),
            Self::Fp8(w) => w.weight.is_null(),
            Self::Dense(w) => w.weight.is_null(),
            Self::PackedQ2(w) => w.is_null(),
        }
    }

    /// 2026-09-25: The keep-packed Q2_0 weight, if this is that variant.
    pub fn as_packed_q2(&self) -> Option<&PackedQ2Weight> {
        match self {
            Self::PackedQ2(w) => Some(w),
            _ => None,
        }
    }

    /// 2026-09-25: The NVFP4 weight, if this is that variant.
    pub fn as_nvfp4(&self) -> Option<&QuantizedWeight> {
        match self {
            Self::Nvfp4(w) => Some(w),
            _ => None,
        }
    }

    /// 2026-09-25: The FP8 weight, if this is that variant.
    pub fn as_fp8(&self) -> Option<&Fp8Weight> {
        match self {
            Self::Fp8(w) => Some(w),
            _ => None,
        }
    }

    /// 2026-09-25: The dense weight, if this is that variant.
    pub fn as_dense(&self) -> Option<&DenseWeight> {
        match self {
            Self::Dense(w) => Some(w),
            _ => None,
        }
    }
}

impl From<QuantizedWeight> for QuantWeight {
    fn from(w: QuantizedWeight) -> Self {
        Self::Nvfp4(w)
    }
}

impl From<Fp8Weight> for QuantWeight {
    fn from(w: Fp8Weight) -> Self {
        Self::Fp8(w)
    }
}

impl From<DenseWeight> for QuantWeight {
    fn from(w: DenseWeight) -> Self {
        Self::Dense(w)
    }
}

impl From<PackedQ2Weight> for QuantWeight {
    fn from(w: PackedQ2Weight) -> Self {
        Self::PackedQ2(w)
    }
}

/// 2026-09-25: Per-expert weights in any supported format. `ExpertWeight`
/// (NVFP4) and `Fp8ExpertWeight` convert into it.
#[derive(Debug, Clone, Copy)]
pub struct QuantExpertWeight {
    pub gate_proj: QuantWeight,
    pub up_proj: QuantWeight,
    pub down_proj: QuantWeight,
}

impl QuantExpertWeight {
    pub fn null() -> Self {
        Self {
            gate_proj: QuantWeight::null(),
            up_proj: QuantWeight::null(),
            down_proj: QuantWeight::null(),
        }
    }
}

impl From<ExpertWeight> for QuantExpertWeight {
    fn from(w: ExpertWeight) -> Self {
        Self {
            gate_proj: QuantWeight::Nvfp4(w.gate_proj),
            up_proj: QuantWeight::Nvfp4(w.up_proj),
            down_proj: QuantWeight::Nvfp4(w.down_proj),
        }
    }
}

impl From<Fp8ExpertWeight> for QuantExpertWeight {
    fn from(w: Fp8ExpertWeight) -> Self {
        Self {
            gate_proj: QuantWeight::Fp8(w.gate_proj),
            up_proj: QuantWeight::Fp8(w.up_proj),
            down_proj: QuantWeight::Fp8(w.down_proj),
        }
    }
}
