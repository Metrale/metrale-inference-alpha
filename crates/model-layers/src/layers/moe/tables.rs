// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-25: Where expert weights live on the device: the NVFP4 and FP8
//! per-expert pointer tables and the BF16 shared-expert weights.
//!
//! Owner: model-layers (MoE).
//! Invariants:
//! - `Bf16SharedExpert` is built only by `new`, which refuses a null weight.

use anyhow::Result;
use metrale_gpu_runtime::gpu::DevicePtr;

use crate::weight_map::DenseWeight;

/// 2026-09-25: Device-side pointer table for one NVFP4 projection across all
/// experts. The expert kernels index it with the device-side expert ids.
pub(crate) struct ExpertPtrTable {
    /// 2026-09-25: u64 device pointer to each expert's packed weight.
    pub(crate) packed_ptrs: DevicePtr,
    /// 2026-09-25: u64 device pointer to each expert's block scales.
    pub(crate) scale_ptrs: DevicePtr,
    /// 2026-09-25: Each expert's `weight_scale_2`, as f32.
    pub(crate) scale2_vals: DevicePtr,
}

/// 2026-09-25: Device-side pointer table for one FP8 projection across all
/// experts.
pub(crate) struct Fp8ExpertPtrTable {
    /// 2026-09-25: u64 device pointer to each expert's FP8 weight.
    pub(crate) weight_ptrs: DevicePtr,
    /// 2026-09-25: u64 device pointer to each expert's `row_scale` (block scales).
    pub(crate) scale_ptrs: DevicePtr,
}

/// 2026-09-25: BF16 shared-expert weights, installed independently of the routed
/// experts' precision: the Laguna loader installs them next to NVFP4 routed
/// experts when the checkpoint keeps its shared expert in BF16.
#[derive(Debug, Clone, Copy)]
pub(crate) struct Bf16SharedExpert {
    pub(super) gate_proj: DenseWeight,
    pub(super) up_proj: DenseWeight,
    pub(super) down_proj: DenseWeight,
}

impl Bf16SharedExpert {
    pub(super) fn new(
        gate_proj: DenseWeight,
        up_proj: DenseWeight,
        down_proj: DenseWeight,
    ) -> Result<Self> {
        anyhow::ensure!(
            !gate_proj.weight.is_null() && !up_proj.weight.is_null() && !down_proj.weight.is_null(),
            "BF16 shared expert requires non-null gate/up/down weights"
        );
        Ok(Self {
            gate_proj,
            up_proj,
            down_proj,
        })
    }
}

#[allow(dead_code)]
pub(crate) enum ExpertPtrSet {
    Nvfp4 {
        packed_ptrs: DevicePtr,
        scale_ptrs: DevicePtr,
        scale2_vals: DevicePtr,
    },
    Fp8 {
        weight_ptrs: DevicePtr,
        scale_ptrs: DevicePtr,
    },
}
