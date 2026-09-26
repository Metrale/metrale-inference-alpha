// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-25: The stacked MTP expert slicer and the `Nvfp4Variant` on-disk format enum.
//!
//! Owner: model-layers (weight loading).
//! Invariants:
//! - The stacked expert pointers alias store tensors; no loader frees them.

#![allow(unused_imports)]

use anyhow::{Context, Result, bail, ensure};
use metrale_gpu_runtime::gpu::{DevicePtr, GpuBackend};
use metrale_model_weights::weights::{WeightDtype, WeightStore};

use super::*;

/// 2026-09-25: Slice a stacked MTP expert layout into per-expert
/// `DenseExpertWeight`s by pointer offset, without copying.
///
/// Expects two BF16 tensors with `E == num_experts`:
///   `{mlp}.experts.gate_up_proj` `[E, 2*I, H]`: the first `I` rows of axis 1
///   are gate, the next `I` rows up;
///   `{mlp}.experts.down_proj` `[E, H, I]`.
///
/// The returned pointers alias the store's tensors and must not be passed
/// to `gpu.free()`.
pub(super) fn load_mtp_experts_stacked(
    store: &WeightStore,
    mlp: &str,
    num_experts: usize,
) -> Result<Vec<DenseExpertWeight>> {
    let gate_up = store.get(&format!("{mlp}.experts.gate_up_proj"))?;
    let down = store.get(&format!("{mlp}.experts.down_proj"))?;

    ensure!(
        gate_up.shape.len() == 3,
        "MTP stacked experts.gate_up_proj: expected 3D [E,2I,H], got {:?}",
        gate_up.shape
    );
    ensure!(
        down.shape.len() == 3,
        "MTP stacked experts.down_proj: expected 3D [E,H,I], got {:?}",
        down.shape
    );
    ensure!(
        gate_up.shape[0] == num_experts,
        "MTP stacked experts.gate_up_proj: expert dim {} != num_experts {num_experts}",
        gate_up.shape[0]
    );
    ensure!(
        down.shape[0] == num_experts,
        "MTP stacked experts.down_proj: expert dim {} != num_experts {num_experts}",
        down.shape[0]
    );

    let two_inter = gate_up.shape[1];
    let hidden = gate_up.shape[2];
    ensure!(
        two_inter % 2 == 0,
        "MTP stacked experts.gate_up_proj: 2nd dim must be even (gate+up fused), got {two_inter}"
    );
    let intermediate = two_inter / 2;

    ensure!(
        down.shape[1] == hidden,
        "MTP stacked: gate_up_proj.hidden ({hidden}) != down_proj.hidden ({})",
        down.shape[1]
    );
    ensure!(
        down.shape[2] == intermediate,
        "MTP stacked: down_proj.intermediate ({}) != gate_up_proj/2 ({intermediate})",
        down.shape[2]
    );

    // 2026-09-25: BF16 only: these tensors are handed out without conversion.
    ensure!(
        matches!(gate_up.dtype, WeightDtype::BF16),
        "MTP stacked experts.gate_up_proj: expected BF16, got {:?}",
        gate_up.dtype
    );
    ensure!(
        matches!(down.dtype, WeightDtype::BF16),
        "MTP stacked experts.down_proj: expected BF16, got {:?}",
        down.dtype
    );

    let elt = WeightDtype::BF16.byte_size();
    let half_bytes = intermediate * hidden * elt;
    let gate_up_stride = two_inter * hidden * elt;
    let down_stride = hidden * intermediate * elt;

    let mut experts = Vec::with_capacity(num_experts);
    for e in 0..num_experts {
        let base_gu = gate_up.ptr.offset(e * gate_up_stride);
        experts.push(DenseExpertWeight {
            gate_proj: DenseWeight { weight: base_gu },
            up_proj: DenseWeight {
                weight: base_gu.offset(half_bytes),
            },
            down_proj: DenseWeight {
                weight: down.ptr.offset(e * down_stride),
            },
        });
    }
    Ok(experts)
}

/// 2026-09-25: On-disk weight format, chosen by `detect_nvfp4_variant`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Nvfp4Variant {
    /// 2026-09-25: ModelOpt NVFP4: `weight`, `weight_scale`, `weight_scale_2`,
    /// optional `input_scale` (read by `quantized`).
    Standard,
    /// 2026-09-25: compressed-tensors NVFP4: `weight_packed`, `weight_scale`,
    /// `weight_global_scale`, optional `input_global_scale` (read by `quantized_v2`).
    CompressedTensors,
    /// 2026-09-25: Block-scaled FP8 E4M3 (`weight` plus its block scale).
    /// `quantized_any` turns such a weight into NVFP4 through
    /// a BF16 dequant (`quantized_from_fp8`); loaders that serve FP8 natively
    /// read it with `load_fp8_block_scaled_as_fp8weight` instead.
    Fp8Dequanted,
    /// 2026-09-25: No usable NVFP4 or FP8 metadata: the BF16 `.weight` is
    /// quantized to NVFP4 at load, and detection logs a warning.
    Bf16Raw,
}
