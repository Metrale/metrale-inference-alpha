// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-25: The `QuantWeights` trait: the matvec operations a per-layer forward
//! needs from a quantised weight, so `super::qwen3_5` never names a concrete
//! weight type.
//!
//! Owner: model-layers.
//! Invariants: none beyond the types.
//!
//! `gemv` is required. The fused operations are optional: `gemv_gate_up_with`
//! defaults to two `gemv` calls, and `gemv_silu_gate` / `gemv_silu_gate_resid`
//! default to an error, so a caller learns that no fused kernel exists instead
//! of getting a silent slow path.

use anyhow::{Result, bail};
use metrale_gpu_runtime::gpu::{DevicePtr, GpuBackend};
use metrale_gpu_runtime::mlx_int8::{self, MlxInt8Weight};

/// 2026-09-25: A quantised `[N, K]` weight that runs matvecs on a `GpuBackend`.
/// The one implementation, for `MlxInt8Weight`, is in this file.
pub trait QuantWeights: Send + Sync {
    /// 2026-09-25: Output dimension `N` of the `[N, K]` weight.
    fn out_features(&self) -> u32;

    /// 2026-09-25: Input dimension `K` of the `[N, K]` weight.
    fn in_features(&self) -> u32;

    /// 2026-09-25: Decode matvec `y = self @ x`.
    ///
    /// `x` is a BF16 buffer of length `in_features()`; `y` must hold at
    /// least `out_features()` BF16 slots.
    fn gemv(&self, gpu: &dyn GpuBackend, x: DevicePtr, y: DevicePtr, stream: u64) -> Result<()>;

    /// 2026-09-25: Dual-output GEMV over one input: `gate_y = self @ x` and
    /// `up_y = other @ x`. The default is two `gemv` calls. The
    /// `MlxInt8Weight` impl overrides it with the single
    /// `mlx_int8_gemv_gate_up` launch.
    ///
    /// `where Self: Sized` keeps this method off the `dyn QuantWeights`
    /// surface; the forward modules call it through a generic parameter.
    fn gemv_gate_up_with(
        &self,
        other: &Self,
        gpu: &dyn GpuBackend,
        x: DevicePtr,
        gate_y: DevicePtr,
        up_y: DevicePtr,
        stream: u64,
    ) -> Result<()>
    where
        Self: Sized,
    {
        debug_assert_eq!(self.out_features(), other.out_features());
        debug_assert_eq!(self.in_features(), other.in_features());
        self.gemv(gpu, x, gate_y, stream)?;
        other.gemv(gpu, x, up_y, stream)?;
        Ok(())
    }

    /// 2026-09-25: Fused FFN tail `y = self @ (silu(gate) ⊙ up)`. The default
    /// returns an error; an implementation overrides it with a fused kernel
    /// or its own composition.
    fn gemv_silu_gate(
        &self,
        _gpu: &dyn GpuBackend,
        _gate: DevicePtr,
        _up: DevicePtr,
        _y: DevicePtr,
        _stream: u64,
    ) -> Result<()> {
        bail!(
            "QuantWeights::gemv_silu_gate is not implemented for this weight type — \
             override with a fused kernel or compose silu+mul+gemv on the caller side"
        )
    }

    /// 2026-09-25: [`Self::gemv_silu_gate`] with the residual add folded in:
    ///   `y[n] = x_resid[n] + sum_k self[n, k] * (silu(gate[k]) ⊙ up[k])`.
    ///
    /// The default returns an error.
    fn gemv_silu_gate_resid(
        &self,
        _gpu: &dyn GpuBackend,
        _gate: DevicePtr,
        _up: DevicePtr,
        _x_resid: DevicePtr,
        _y: DevicePtr,
        _stream: u64,
    ) -> Result<()> {
        bail!(
            "QuantWeights::gemv_silu_gate_resid is not implemented for this weight type — \
             override with a fused kernel or compose silu+mul+gemv+add on the caller side"
        )
    }
}

// 2026-09-25: The impl sits beside the trait because of the orphan rule:
// `MlxInt8Weight` belongs to metrale-gpu-runtime, which cannot see this trait.
// Each method forwards to the weight type's own fused-kernel functions.

impl QuantWeights for MlxInt8Weight {
    fn out_features(&self) -> u32 {
        self.out_features
    }
    fn in_features(&self) -> u32 {
        self.in_features
    }
    fn gemv(&self, gpu: &dyn GpuBackend, x: DevicePtr, y: DevicePtr, stream: u64) -> Result<()> {
        MlxInt8Weight::gemv(self, gpu, x, y, stream)
    }
    fn gemv_gate_up_with(
        &self,
        other: &Self,
        gpu: &dyn GpuBackend,
        x: DevicePtr,
        gate_y: DevicePtr,
        up_y: DevicePtr,
        stream: u64,
    ) -> Result<()> {
        // 2026-09-25: One `mlx_int8_gemv_gate_up` launch computes both outputs.
        mlx_int8::gemv_gate_up(gpu, self, other, x, gate_y, up_y, stream)
    }
    fn gemv_silu_gate(
        &self,
        gpu: &dyn GpuBackend,
        gate: DevicePtr,
        up: DevicePtr,
        y: DevicePtr,
        stream: u64,
    ) -> Result<()> {
        MlxInt8Weight::gemv_silu_gate(self, gpu, gate, up, y, stream)
    }
    fn gemv_silu_gate_resid(
        &self,
        gpu: &dyn GpuBackend,
        gate: DevicePtr,
        up: DevicePtr,
        x_resid: DevicePtr,
        y: DevicePtr,
        stream: u64,
    ) -> Result<()> {
        MlxInt8Weight::gemv_silu_gate_resid(self, gpu, gate, up, x_resid, y, stream)
    }
}
