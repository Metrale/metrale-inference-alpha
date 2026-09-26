// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-26: `LayerWeightSetup`, the `&mut self` hooks that change a layer's weights after
//! construction: the downcast for weight overlays and the MoE expert transposes for
//! prefill.
//!
//! Owner: model-layers.
//! Invariants: none beyond the types.

use anyhow::Result;
use metrale_config::ModelConfig;
use metrale_gpu_runtime::gpu::{DevicePtr, GpuBackend};

/// 2026-09-26: A supertrait of `TransformerLayer`; see the module header.
pub trait LayerWeightSetup {
    /// 2026-09-25: `&mut dyn Any` downcast hook for weight overlays applied after
    /// construction, such as a LoRA install. Default `None`.
    fn as_any_mut(&mut self) -> Option<&mut dyn std::any::Any> {
        None
    }

    /// 2026-09-25: Build persistent transposed gate/up/down MoE expert weights for the
    /// prefill GEMMs. The factory's MoE transpose pass (`factory/m2_setup.rs`) calls it,
    /// outside the hybrid and unified layouts, when free memory covers the full transpose
    /// plus a safety margin. Default: nothing.
    fn transpose_moe_for_prefill(
        &mut self,
        _gpu: &dyn GpuBackend,
        _config: &ModelConfig,
    ) -> Result<()> {
        Ok(())
    }

    /// 2026-09-25: Like `transpose_moe_for_prefill`, for gate and up only. Outside the
    /// hybrid and unified layouts, the transpose pass calls it when the full transpose does
    /// not fit but gate and up do. Default: nothing.
    fn transpose_moe_gate_up_for_prefill(
        &mut self,
        _gpu: &dyn GpuBackend,
        _config: &ModelConfig,
    ) -> Result<()> {
        Ok(())
    }

    /// 2026-09-25: Give this layer's MoE block the per-prefill `down_proj` transpose
    /// scratch that every MoE layer shares, after `transpose_moe_gate_up_for_prefill`.
    /// Default: nothing.
    fn set_moe_down_transpose_scratch(
        &mut self,
        _scratch_packed: DevicePtr,
        _scratch_scale: DevicePtr,
        _packed_ptrs_t: DevicePtr,
        _scale_ptrs_t: DevicePtr,
    ) {
    }

    /// 2026-09-25: Unified layout (`METRALE_UNIFIED_MOE_LAYOUT=1`): build transposed
    /// gate/up/down and free the untransposed copies. Decode reads the transposed copies
    /// when `MoeLayer::use_t_layout_for_decode` holds. Default: nothing.
    fn transpose_moe_for_prefill_unified(
        &mut self,
        _gpu: &dyn GpuBackend,
        _config: &ModelConfig,
    ) -> Result<()> {
        Ok(())
    }

    /// 2026-09-25: Hybrid layout (`METRALE_HYBRID_MOE_LAYOUT=1`): build transposed
    /// gate/up/down beside the originals and free nothing, so decode keeps the originals
    /// and prefill uses the transposed copies. The transpose pass calls it only when twice
    /// the full transpose fits. Default: nothing.
    fn transpose_moe_for_prefill_hybrid(
        &mut self,
        _gpu: &dyn GpuBackend,
        _config: &ModelConfig,
    ) -> Result<()> {
        Ok(())
    }
}
