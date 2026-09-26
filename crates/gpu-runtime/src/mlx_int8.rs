// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-25: Launches for 8-bit affine weights in the MLX layout: load a
//! `(.weight, .scales, .biases)` triplet and run the dequant, GEMV, fused GEMV and
//! GEMM kernels of `kernels/metal/common/mlx_int8_*.metal`.
//!
//! Layout, as [`MlxInt8Weight::load`] checks it:
//! - `{base}.weight`: `U32` `[out_features, in_features / 4]`; each word packs four
//!   unsigned bytes, the low byte holding the lowest column.
//! - `{base}.scales`, `{base}.biases`: `BF16` `[out_features, in_features / G]`,
//!   where `G` is `group_size`.
//!
//! The kernels dequantise `w[r, c] = byte * scales[r, c / G] + biases[r, c / G]`,
//! where `byte` is byte `c % 4` of `packed[r, c / 4]`.
//!
//! Owner: gpu-runtime (metal).
//! Invariants: none beyond the types.

use anyhow::{Context, Result, bail};
use safetensors::SafeTensors;
use serde_json::Value as JsonValue;

use crate::gpu::{DevicePtr, GpuBackend, KernelArg};

/// 2026-09-25: Quantization metadata from the model's `config.json`.
#[derive(Debug, Clone, Copy)]
pub struct MlxQuantConfig {
    pub bits: u32,
    pub group_size: u32,
}

impl MlxQuantConfig {
    /// 2026-09-25: Read `bits` and `group_size` from the top-level `quantization`
    /// block, or from `quantization_config` when `quantization` is absent. `None`
    /// when the block or either key is missing or not an unsigned integer.
    pub fn from_config(config: &JsonValue) -> Option<Self> {
        let q = config
            .get("quantization")
            .or_else(|| config.get("quantization_config"))?;
        let bits = q.get("bits")?.as_u64()? as u32;
        let group_size = q.get("group_size")?.as_u64()? as u32;
        Some(Self { bits, group_size })
    }
}

/// 2026-09-25: One 8-bit affine linear weight resident on the GPU. There is no
/// `Drop`: the owner frees the three buffers with [`Self::release`].
pub struct MlxInt8Weight {
    /// 2026-09-25: `[out_features, in_features / 4]` packed bytes (uint32 words).
    pub packed: DevicePtr,
    /// 2026-09-25: `[out_features, in_features / group_size]` per-group BF16 scales.
    pub scales: DevicePtr,
    /// 2026-09-25: `[out_features, in_features / group_size]` per-group BF16 biases.
    pub biases: DevicePtr,
    pub out_features: u32,
    pub in_features: u32,
    pub group_size: u32,
}

impl MlxInt8Weight {
    /// 2026-09-25: Load the `{base}.weight`, `{base}.scales` and `{base}.biases`
    /// tensors from a parsed safetensors blob and upload them. Fails on a missing
    /// tensor, a weight that is not 2-D `U32`, scales or biases that are not `BF16`
    /// of shape `[out_features, in_features / group_size]`, or an `in_features`
    /// that `group_size` does not divide.
    pub fn load(
        gpu: &dyn GpuBackend,
        st: &SafeTensors,
        base: &str,
        group_size: u32,
    ) -> Result<Self> {
        let weight_name = format!("{base}.weight");
        let scales_name = format!("{base}.scales");
        let biases_name = format!("{base}.biases");

        let weight = st
            .tensor(&weight_name)
            .with_context(|| format!("missing tensor {weight_name}"))?;
        let scales = st
            .tensor(&scales_name)
            .with_context(|| format!("missing tensor {scales_name}"))?;
        let biases = st
            .tensor(&biases_name)
            .with_context(|| format!("missing tensor {biases_name}"))?;

        if weight.dtype() != safetensors::Dtype::U32 {
            bail!(
                "{weight_name}: expected U32 (MLX 8-bit packed), got {:?}",
                weight.dtype()
            );
        }
        if scales.dtype() != safetensors::Dtype::BF16 || biases.dtype() != safetensors::Dtype::BF16
        {
            bail!(
                "{base}.scales/biases: expected BF16, got scales={:?}, biases={:?}",
                scales.dtype(),
                biases.dtype()
            );
        }

        let weight_shape = weight.shape();
        if weight_shape.len() != 2 {
            bail!(
                "{weight_name}: expected 2-D weight tensor, got rank {}",
                weight_shape.len()
            );
        }
        let out_features = weight_shape[0] as u32;
        let packed_cols = weight_shape[1] as u32;
        let in_features = packed_cols * 4;

        let groups_per_row = in_features
            .checked_div(group_size)
            .filter(|&g| g * group_size == in_features)
            .ok_or_else(|| {
                anyhow::anyhow!(
                    "{base}: in_features {in_features} not divisible by group_size {group_size}"
                )
            })?;

        let expected = [out_features as usize, groups_per_row as usize];
        if scales.shape() != expected {
            bail!(
                "{}.scales: expected shape {:?}, got {:?}",
                base,
                expected,
                scales.shape()
            );
        }
        if biases.shape() != expected {
            bail!(
                "{}.biases: expected shape {:?}, got {:?}",
                base,
                expected,
                biases.shape()
            );
        }

        let packed_ptr = gpu.alloc(weight.data().len())?;
        gpu.copy_h2d(weight.data(), packed_ptr)?;
        let scales_ptr = gpu.alloc(scales.data().len())?;
        gpu.copy_h2d(scales.data(), scales_ptr)?;
        let biases_ptr = gpu.alloc(biases.data().len())?;
        gpu.copy_h2d(biases.data(), biases_ptr)?;

        Ok(Self {
            packed: packed_ptr,
            scales: scales_ptr,
            biases: biases_ptr,
            out_features,
            in_features,
            group_size,
        })
    }

    /// 2026-09-25: Write the full dequantised weight as BF16 to `out`, which must
    /// hold at least `out_features * in_features * 2` bytes. Runs `mlx_int8_dequant`.
    pub fn dequantize_to(&self, gpu: &dyn GpuBackend, out: DevicePtr, stream: u64) -> Result<()> {
        let kernel = gpu.kernel("mlx_int8_dequant", "mlx_int8_dequant")?;
        // 2026-09-25: One thread per element; the kernel bounds-checks the ragged
        // last column tile.
        let block_x: u32 = 16;
        let block_y: u32 = 1;
        let grid_x = self.in_features.div_ceil(block_x);
        let grid_y = self.out_features;
        gpu.launch_typed(
            kernel,
            [grid_x, grid_y, 1],
            [block_x, block_y, 1],
            0,
            stream,
            &[
                KernelArg::Bytes(&self.out_features.to_le_bytes()),
                KernelArg::Bytes(&self.in_features.to_le_bytes()),
                KernelArg::Bytes(&self.group_size.to_le_bytes()),
                KernelArg::Buffer(self.packed),
                KernelArg::Buffer(self.scales),
                KernelArg::Buffer(self.biases),
                KernelArg::Buffer(out),
            ],
        )
    }

    /// 2026-09-25: `y = W x`, with `x` BF16 `[in_features]` and `y` a BF16 buffer of
    /// at least `out_features` elements. Runs `mlx_int8_gemv`.
    pub fn gemv(
        &self,
        gpu: &dyn GpuBackend,
        x: DevicePtr,
        y: DevicePtr,
        stream: u64,
    ) -> Result<()> {
        let kernel = gpu.kernel("mlx_int8_gemv", "mlx_int8_gemv")?;
        // 2026-09-25: Four rows per threadgroup, one 32-lane simdgroup per row,
        // reduced with `simd_sum`, so no reduction crosses simdgroups.
        const ROWS_PER_TG: u32 = 4;
        const SIMDGROUP_SIZE: u32 = 32;
        let threads_per_tg: u32 = ROWS_PER_TG * SIMDGROUP_SIZE;
        let row_groups = self.out_features.div_ceil(ROWS_PER_TG);
        gpu.launch_typed(
            kernel,
            [row_groups, 1, 1],
            [threads_per_tg, 1, 1],
            0,
            stream,
            &[
                KernelArg::Bytes(&self.out_features.to_le_bytes()),
                KernelArg::Bytes(&self.in_features.to_le_bytes()),
                KernelArg::Bytes(&self.group_size.to_le_bytes()),
                KernelArg::Buffer(self.packed),
                KernelArg::Buffer(self.scales),
                KernelArg::Buffer(self.biases),
                KernelArg::Buffer(x),
                KernelArg::Buffer(y),
            ],
        )
    }

    /// 2026-09-25: [`Self::gemv_silu_gate`] plus the residual, in one launch:
    ///   `y[n] = x_resid[n] + sum_k W[n, k] * (silu(gate[k]) * up[k])`.
    pub fn gemv_silu_gate_resid(
        &self,
        gpu: &dyn GpuBackend,
        gate: DevicePtr,
        up: DevicePtr,
        x_resid: DevicePtr,
        y: DevicePtr,
        stream: u64,
    ) -> Result<()> {
        let kernel = gpu.kernel("mlx_int8_gemv_silu_gate", "mlx_int8_gemv_silu_gate_resid")?;
        const ROWS_PER_TG: u32 = 4;
        const SIMDGROUP_SIZE: u32 = 32;
        let threads_per_tg: u32 = ROWS_PER_TG * SIMDGROUP_SIZE;
        let row_groups = self.out_features.div_ceil(ROWS_PER_TG);
        gpu.launch_typed(
            kernel,
            [row_groups, 1, 1],
            [threads_per_tg, 1, 1],
            0,
            stream,
            &[
                KernelArg::Bytes(&self.out_features.to_le_bytes()),
                KernelArg::Bytes(&self.in_features.to_le_bytes()),
                KernelArg::Bytes(&self.group_size.to_le_bytes()),
                KernelArg::Buffer(self.packed),
                KernelArg::Buffer(self.scales),
                KernelArg::Buffer(self.biases),
                KernelArg::Buffer(gate),
                KernelArg::Buffer(up),
                KernelArg::Buffer(x_resid),
                KernelArg::Buffer(y),
            ],
        )
    }

    /// 2026-09-25: `y = W (silu(gate) * up)` in one launch of
    /// `mlx_int8_gemv_silu_gate`; `silu(gate) * up` is never written to memory.
    pub fn gemv_silu_gate(
        &self,
        gpu: &dyn GpuBackend,
        gate: DevicePtr,
        up: DevicePtr,
        y: DevicePtr,
        stream: u64,
    ) -> Result<()> {
        let kernel = gpu.kernel("mlx_int8_gemv_silu_gate", "mlx_int8_gemv_silu_gate")?;
        const ROWS_PER_TG: u32 = 4;
        const SIMDGROUP_SIZE: u32 = 32;
        let threads_per_tg: u32 = ROWS_PER_TG * SIMDGROUP_SIZE;
        let row_groups = self.out_features.div_ceil(ROWS_PER_TG);
        gpu.launch_typed(
            kernel,
            [row_groups, 1, 1],
            [threads_per_tg, 1, 1],
            0,
            stream,
            &[
                KernelArg::Bytes(&self.out_features.to_le_bytes()),
                KernelArg::Bytes(&self.in_features.to_le_bytes()),
                KernelArg::Bytes(&self.group_size.to_le_bytes()),
                KernelArg::Buffer(self.packed),
                KernelArg::Buffer(self.scales),
                KernelArg::Buffer(self.biases),
                KernelArg::Buffer(gate),
                KernelArg::Buffer(up),
                KernelArg::Buffer(y),
            ],
        )
    }

    /// 2026-09-25: `Y = X W^T`, with `X` BF16 `[m, in_features]` and `Y` BF16
    /// `[m, out_features]`. Runs `mlx_int8_gemm`, one thread per output element.
    pub fn gemm(
        &self,
        gpu: &dyn GpuBackend,
        x: DevicePtr,
        y: DevicePtr,
        m: u32,
        stream: u64,
    ) -> Result<()> {
        let kernel = gpu.kernel("mlx_int8_gemm", "mlx_int8_gemm")?;
        let block_x: u32 = 16;
        let block_y: u32 = 16;
        let grid_x = self.out_features.div_ceil(block_x);
        let grid_y = m.div_ceil(block_y);
        gpu.launch_typed(
            kernel,
            [grid_x, grid_y, 1],
            [block_x, block_y, 1],
            0,
            stream,
            &[
                KernelArg::Bytes(&m.to_le_bytes()),
                KernelArg::Bytes(&self.out_features.to_le_bytes()),
                KernelArg::Bytes(&self.in_features.to_le_bytes()),
                KernelArg::Bytes(&self.group_size.to_le_bytes()),
                KernelArg::Buffer(x),
                KernelArg::Buffer(self.packed),
                KernelArg::Buffer(self.scales),
                KernelArg::Buffer(self.biases),
                KernelArg::Buffer(y),
            ],
        )
    }

    /// 2026-09-25: Free the three GPU buffers. `DevicePtr` is a plain handle, so
    /// nothing frees them automatically; call this once.
    pub fn release(&self, gpu: &dyn GpuBackend) -> Result<()> {
        gpu.free(self.packed)?;
        gpu.free(self.scales)?;
        gpu.free(self.biases)?;
        Ok(())
    }
}

/// 2026-09-25: `gate_y = gate x` and `up_y = up x` in one launch of
/// `mlx_int8_gemv_gate_up`, which reads `x` once for both. Both weights must share
/// `(out_features, in_features, group_size)`; only `debug_assert_eq!` checks it.
pub fn gemv_gate_up(
    gpu: &dyn GpuBackend,
    gate: &MlxInt8Weight,
    up: &MlxInt8Weight,
    x: DevicePtr,
    gate_y: DevicePtr,
    up_y: DevicePtr,
    stream: u64,
) -> Result<()> {
    debug_assert_eq!(gate.out_features, up.out_features);
    debug_assert_eq!(gate.in_features, up.in_features);
    debug_assert_eq!(gate.group_size, up.group_size);
    let kernel = gpu.kernel("mlx_int8_gemv_gate_up", "mlx_int8_gemv_gate_up")?;
    const ROWS_PER_TG: u32 = 4;
    const SIMDGROUP_SIZE: u32 = 32;
    let threads_per_tg: u32 = ROWS_PER_TG * SIMDGROUP_SIZE;
    let row_groups = gate.out_features.div_ceil(ROWS_PER_TG);
    gpu.launch_typed(
        kernel,
        [row_groups, 1, 1],
        [threads_per_tg, 1, 1],
        0,
        stream,
        &[
            KernelArg::Bytes(&gate.out_features.to_le_bytes()),
            KernelArg::Bytes(&gate.in_features.to_le_bytes()),
            KernelArg::Bytes(&gate.group_size.to_le_bytes()),
            KernelArg::Buffer(gate.packed),
            KernelArg::Buffer(gate.scales),
            KernelArg::Buffer(gate.biases),
            KernelArg::Buffer(up.packed),
            KernelArg::Buffer(up.scales),
            KernelArg::Buffer(up.biases),
            KernelArg::Buffer(x),
            KernelArg::Buffer(gate_y),
            KernelArg::Buffer(up_y),
        ],
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_quant_config_from_mlx_layout() {
        let cfg: JsonValue = serde_json::json!({
            "quantization": { "bits": 8, "group_size": 64, "mode": "affine" },
            "model_type": "qwen3_5",
        });
        let q = MlxQuantConfig::from_config(&cfg).expect("expected quant block");
        assert_eq!(q.bits, 8);
        assert_eq!(q.group_size, 64);
    }

    #[test]
    fn parse_quant_config_falls_back_to_quantization_config() {
        let cfg: JsonValue = serde_json::json!({
            "quantization_config": { "bits": 8, "group_size": 64 },
        });
        let q = MlxQuantConfig::from_config(&cfg).expect("expected quant_config block");
        assert_eq!(q.bits, 8);
        assert_eq!(q.group_size, 64);
    }

    #[test]
    fn parse_quant_config_returns_none_when_absent() {
        let cfg: JsonValue = serde_json::json!({ "model_type": "qwen3_5" });
        assert!(MlxQuantConfig::from_config(&cfg).is_none());
    }
}
