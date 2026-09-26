// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-25: Builders for the device-side expert pointer tables (NVFP4, BF16,
//! FP8): per projection, one device array of per-expert pointers, plus the
//! per-expert `weight_scale_2` values for NVFP4.
//!
//! Owner: model-layers (MoE).
//! Invariants:
//! - Every table has one entry per input expert, in input order.

use anyhow::Result;
use metrale_gpu_runtime::gpu::{DevicePtr, GpuBackend};

use super::{ExpertPtrTable, Fp8ExpertPtrTable};
use crate::weight_map::{DenseWeight, ExpertWeight, Fp8ExpertWeight, Fp8Weight, QuantizedWeight};

/// 2026-09-25: NVFP4 pointer table from one `QuantizedWeight` per expert; the
/// transpose passes in `helpers_a.rs` build their transposed tables with it.
pub(crate) fn build_ptr_table_from_qw(
    weights: &[QuantizedWeight],
    gpu: &dyn GpuBackend,
) -> Result<ExpertPtrTable> {
    let n = weights.len();
    let packed_bytes: Vec<u8> = weights
        .iter()
        .flat_map(|w| w.weight.0.to_le_bytes())
        .collect();
    let scale_bytes: Vec<u8> = weights
        .iter()
        .flat_map(|w| w.weight_scale.0.to_le_bytes())
        .collect();
    let scale2_bytes: Vec<u8> = weights
        .iter()
        .flat_map(|w| w.weight_scale_2.to_le_bytes())
        .collect();

    let packed_ptrs = gpu.alloc(n * 8)?;
    gpu.copy_h2d(&packed_bytes, packed_ptrs)?;
    let scale_ptrs = gpu.alloc(n * 8)?;
    gpu.copy_h2d(&scale_bytes, scale_ptrs)?;
    let scale2_vals = gpu.alloc(n * 4)?;
    gpu.copy_h2d(&scale2_bytes, scale2_vals)?;

    Ok(ExpertPtrTable {
        packed_ptrs,
        scale_ptrs,
        scale2_vals,
    })
}

/// 2026-09-25: NVFP4 pointer table for the projection `proj` selects from each
/// expert.
pub(crate) fn build_ptr_table(
    experts: &[ExpertWeight],
    proj: impl Fn(&ExpertWeight) -> &crate::weight_map::QuantizedWeight,
    gpu: &dyn GpuBackend,
) -> Result<ExpertPtrTable> {
    let n = experts.len();

    let packed_bytes: Vec<u8> = experts
        .iter()
        .flat_map(|e| proj(e).weight.0.to_le_bytes())
        .collect();
    let scale_bytes: Vec<u8> = experts
        .iter()
        .flat_map(|e| proj(e).weight_scale.0.to_le_bytes())
        .collect();
    let scale2_bytes: Vec<u8> = experts
        .iter()
        .flat_map(|e| proj(e).weight_scale_2.to_le_bytes())
        .collect();

    let packed_ptrs = gpu.alloc(n * 8)?;
    gpu.copy_h2d(&packed_bytes, packed_ptrs)?;

    let scale_ptrs = gpu.alloc(n * 8)?;
    gpu.copy_h2d(&scale_bytes, scale_ptrs)?;

    let scale2_vals = gpu.alloc(n * 4)?;
    gpu.copy_h2d(&scale2_bytes, scale2_vals)?;

    Ok(ExpertPtrTable {
        packed_ptrs,
        scale_ptrs,
        scale2_vals,
    })
}

/// 2026-09-25: BF16 pointer table: one device pointer per expert weight, for
/// `set_bf16_experts`.
pub(crate) fn build_bf16_ptr_table(
    experts: &[DenseWeight],
    gpu: &dyn GpuBackend,
) -> Result<DevicePtr> {
    let n = experts.len();
    let weight_bytes: Vec<u8> = experts
        .iter()
        .flat_map(|e| e.weight.0.to_le_bytes())
        .collect();
    let ptrs = gpu.alloc(n * 8)?;
    gpu.copy_h2d(&weight_bytes, ptrs)?;
    Ok(ptrs)
}

/// 2026-09-25: FP8 pointer tables for the projection `proj` selects from each
/// expert: weight pointers and `row_scale` pointers.
pub(crate) fn build_fp8_ptr_table(
    experts: &[Fp8ExpertWeight],
    proj: impl Fn(&Fp8ExpertWeight) -> &Fp8Weight,
    gpu: &dyn GpuBackend,
) -> Result<Fp8ExpertPtrTable> {
    let n = experts.len();

    let weight_bytes: Vec<u8> = experts
        .iter()
        .flat_map(|e| proj(e).weight.0.to_le_bytes())
        .collect();
    let scale_bytes: Vec<u8> = experts
        .iter()
        .flat_map(|e| proj(e).row_scale.0.to_le_bytes())
        .collect();

    let weight_ptrs = gpu.alloc(n * 8)?;
    gpu.copy_h2d(&weight_bytes, weight_ptrs)?;

    let scale_ptrs = gpu.alloc(n * 8)?;
    gpu.copy_h2d(&scale_bytes, scale_ptrs)?;

    Ok(Fp8ExpertPtrTable {
        weight_ptrs,
        scale_ptrs,
    })
}
