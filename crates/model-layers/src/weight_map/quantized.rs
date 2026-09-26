// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-25: Device-resident weight types (`QuantizedWeight`, `DenseWeight`, `Fp8Weight`, `PackedQ2Weight`) and the `WeightQuantFormat` tag.
//!
//! Owner: model-layers (weight loading).
//! Invariants: none beyond the types.

#![allow(unused_imports)]

use anyhow::{Context, Result, bail, ensure};
use metrale_gpu_runtime::gpu::{DevicePtr, GpuBackend};
use metrale_model_weights::weights::{WeightDtype, WeightStore};

use super::*;
#[path = "quantized/transpose.rs"]
mod transpose;

/// 2026-09-25: The format of a weight buffer in device memory, as opposed to
/// the on-disk format (`Nvfp4Variant`). A kernel call site checks it with
/// [`WeightQuantFormat::expect`], because a kernel cannot tell one layout
/// from another by its pointer.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WeightQuantFormat {
    /// 2026-09-25: BF16, not quantized.
    Bf16,
    /// 2026-09-25: FP8 E4M3 with one F32 scale per row (`[N]`), as
    /// `load_fp8_weight` returns.
    Fp8PerRow,
    /// 2026-09-25: FP8 E4M3 with one F32 scale per 128x128 block, as
    /// `load_fp8_block_scaled_as_fp8weight` and `quantize_to_fp8_blockscaled` return.
    Fp8BlockScaled,
    /// 2026-09-25: FP8 E4M3 with no scale buffer.
    Fp8SingleScale,
    /// 2026-09-25: NVFP4: packed E2M1 nibbles, one FP8 scale per 16 weights, and
    /// an F32 per-tensor scale.
    Nvfp4,
    /// 2026-09-25: Native MXFP4: packed E2M1 nibbles and one E8M0 scale per 32
    /// weights, with no per-tensor scale. The DeepSeek-V4 loader tags its routed
    /// experts with it (`MoeLayer::experts_scale_kind`).
    Mxfp4E8m0,
    /// 2026-09-25: Keep-packed ternary Q2_0 (ggml id 42): raw `block_q2_0`
    /// blocks, each an fp16 scale and 2-bit codes, as [`PackedQ2Weight`] holds them.
    PackedQ2_0,
}

impl WeightQuantFormat {
    /// 2026-09-25: Panic, naming `context`, when `self` is not `expected`.
    #[inline]
    #[track_caller]
    pub fn expect(self, expected: WeightQuantFormat, context: &str) {
        if self != expected {
            panic!(
                "WeightQuantFormat mismatch at {context}: kernel expects {expected:?}, \
                 but the weight buffer is tagged {self:?}. This is a silent quant-leak \
                 that would produce wrong outputs without this assertion."
            );
        }
    }
}

/// 2026-09-25: Keep-packed ternary Q2_0 weight: one buffer of raw `block_q2_0`
/// blocks (`[fp16 d][group/4 bytes of 2-bit codes]`, `value = (code - 1) * d`),
/// row-major over `[n, k]`, with the scale inline and no separate scale tensor.
/// Built from a `WeightDtype::PackedQ2_0` store tensor, which the GGUF loader
/// produces under `METRALE_GGUF_NATIVE_Q2=1`. The store owns the buffer; this
/// struct only holds the pointer.
#[derive(Debug, Clone, Copy)]
pub struct PackedQ2Weight {
    /// 2026-09-25: Raw packed `block_q2_0` bytes, `n * (k/group) * (2 + group/4)` long.
    pub weight: DevicePtr,
    /// 2026-09-25: Output rows (the weight is `[n, k]`).
    pub n: u32,
    /// 2026-09-25: Input columns (the contraction dimension).
    pub k: u32,
    /// 2026-09-25: Elements per block and per inline scale.
    pub group: u16,
}

impl PackedQ2Weight {
    /// 2026-09-25: True when the buffer pointer is NULL.
    pub fn is_null(&self) -> bool {
        self.weight == DevicePtr::NULL
    }
}

/// 2026-09-25: 4-bit quantized weight: packed E2M1 data, per-group scales and a
/// host-side per-tensor scale.
#[derive(Debug, Clone, Copy)]
pub struct QuantizedWeight {
    /// 2026-09-25: Packed E2M1 weights, 2 per byte.
    pub weight: DevicePtr,
    /// 2026-09-25: Per-group scales: FP8 for NVFP4, E8M0 for native MXFP4.
    pub weight_scale: DevicePtr,
    /// 2026-09-25: Per-tensor scale, held on the host.
    pub weight_scale_2: f32,
    /// 2026-09-25: Input activation scale on the device, or NULL.
    pub input_scale: DevicePtr,
    /// 2026-09-25: Per-row scale2 on the device, or NULL. The loaders in this
    /// module always set NULL; the transposes copy it through.
    pub weight_scale_2_vec: DevicePtr,
}

impl QuantizedWeight {
    /// 2026-09-25: All pointers NULL and `weight_scale_2` 0.0.
    pub fn null() -> Self {
        Self {
            weight: DevicePtr::NULL,
            weight_scale: DevicePtr::NULL,
            weight_scale_2: 0.0,
            input_scale: DevicePtr::NULL,
            weight_scale_2_vec: DevicePtr::NULL,
        }
    }

    /// 2026-09-25: True when `weight_scale_2_vec` is not NULL.
    pub fn has_per_row_scale2(&self) -> bool {
        self.weight_scale_2_vec != DevicePtr::NULL
    }

    /// 2026-09-25: True when `weight` is NULL.
    pub fn is_null(&self) -> bool {
        self.weight == DevicePtr::NULL
    }

    /// 2026-09-25: Concatenate two NVFP4 weights by rows into new buffers:
    /// `[N1, K/2]` + `[N2, K/2]` → `[N1+N2, K/2]`, with group size 16.
    ///
    /// Both must have the same `K`, which the caller guarantees. Their
    /// `weight_scale_2` must be bit-identical, or it returns an error, because
    /// the result carries `self`'s scale for every row. The result has NULL
    /// `input_scale` and `weight_scale_2_vec`.
    pub fn concat_rows(
        &self,
        other: &QuantizedWeight,
        n1: usize,
        n2: usize,
        k: usize,
        gpu: &dyn GpuBackend,
    ) -> anyhow::Result<QuantizedWeight> {
        anyhow::ensure!(
            self.weight_scale_2 == other.weight_scale_2,
            "concat_rows: weight_scale_2 mismatch (self={}, other={}) — both NVFP4 \
             tensors must share the same per-tensor scale to be concatenated. \
             This is expected for ModelOpt/Standard NVFP4 checkpoints (single \
             global per-tensor scale2); re-quantize with the ModelOpt/Standard \
             quantizer, or report which checkpoint/quantizer produced independent \
             per-tensor scales for these projections",
            self.weight_scale_2,
            other.weight_scale_2,
        );
        const GROUP_SIZE: usize = 16;
        let half_k = k / 2;
        let num_groups = k / GROUP_SIZE;

        let total_n = n1 + n2;
        let packed_size = total_n * half_k;
        let scale_size = total_n * num_groups;

        let new_weight = gpu.alloc(packed_size)?;
        let new_scale = gpu.alloc(scale_size)?;

        gpu.copy_d2d(self.weight, new_weight, n1 * half_k)?;
        gpu.copy_d2d(other.weight, new_weight.offset(n1 * half_k), n2 * half_k)?;

        gpu.copy_d2d(self.weight_scale, new_scale, n1 * num_groups)?;
        gpu.copy_d2d(
            other.weight_scale,
            new_scale.offset(n1 * num_groups),
            n2 * num_groups,
        )?;

        Ok(QuantizedWeight {
            weight: new_weight,
            weight_scale: new_scale,
            weight_scale_2: self.weight_scale_2,
            input_scale: DevicePtr::NULL,
            weight_scale_2_vec: DevicePtr::NULL,
        })
    }

    /// 2026-09-25: Dequantize this NVFP4 weight `[n, k]` into a new FP8 E4M3
    /// buffer `[n, k]` with `weight_scale_2` applied, for prefill GEMMs that read
    /// FP8. It synchronizes `stream` before returning.
    pub fn predequant_to_fp8(
        &self,
        gpu: &dyn GpuBackend,
        predequant_kernel: metrale_gpu_runtime::gpu::KernelHandle,
        n: usize,
        k: usize,
        stream: u64,
    ) -> Result<DevicePtr> {
        let fp8_buf = gpu.alloc(n * k)?;
        crate::layers::ops::predequant_nvfp4_to_fp8(
            gpu,
            predequant_kernel,
            self.weight,
            self.weight_scale,
            self.weight_scale_2,
            fp8_buf,
            n as u32,
            k as u32,
            stream,
        )?;
        gpu.synchronize(stream)?;
        Ok(fp8_buf)
    }
}

/// 2026-09-25: Device pointer to an unquantized weight: BF16 unless the loader
/// that built it says otherwise (for example `dense_keep_f32`).
#[derive(Debug, Clone, Copy)]
pub struct DenseWeight {
    pub weight: DevicePtr,
}

impl DenseWeight {
    /// 2026-09-25: Quantize a BF16 weight `[N, K]` to FP8 E4M3 `[N, K]` with one
    /// f32 scale per row, in new buffers, with the `quantize_bf16_to_fp8`
    /// kernel. It synchronizes `stream` before returning.
    pub fn quantize_to_fp8(
        &self,
        gpu: &dyn GpuBackend,
        quantize_kernel: metrale_gpu_runtime::gpu::KernelHandle,
        n: usize,
        k: usize,
        stream: u64,
    ) -> Result<Fp8DenseWeight> {
        let fp8_buf = gpu.alloc(n * k)?;
        let row_scale_buf = gpu.alloc(n * std::mem::size_of::<f32>())?;
        crate::layers::ops::quantize_bf16_to_fp8(
            gpu,
            quantize_kernel,
            self.weight,
            fp8_buf,
            row_scale_buf,
            n as u32,
            k as u32,
            stream,
        )?;
        gpu.synchronize(stream)?;
        Ok(Fp8DenseWeight {
            weight: fp8_buf,
            row_scale: row_scale_buf,
        })
    }
}

/// 2026-09-25: FP8 E4M3 weight quantized from BF16 at load, with one f32 scale per row.
#[derive(Debug, Clone, Copy)]
pub struct Fp8DenseWeight {
    /// 2026-09-25: `[N, K]` FP8 E4M3 bytes.
    pub weight: DevicePtr,
    /// 2026-09-25: `[N]` f32 row scales.
    pub row_scale: DevicePtr,
}

/// 2026-09-25: FP8 E4M3 weight with its scale. The layout of `row_scale`
/// depends on `scale_format`:
///   - [`WeightQuantFormat::Fp8PerRow`]: `[N]` f32;
///   - [`WeightQuantFormat::Fp8BlockScaled`]: `[N/128, K/128]` f32, rounded up;
///   - [`WeightQuantFormat::Fp8SingleScale`]: no scale buffer.
///
/// Check `scale_format` before reading `row_scale`.
#[derive(Debug, Clone, Copy)]
pub struct Fp8Weight {
    /// 2026-09-25: `[N, K]` FP8 E4M3 bytes.
    pub weight: DevicePtr,
    /// 2026-09-25: The scale buffer, laid out as `scale_format` says.
    pub row_scale: DevicePtr,
    pub n: u32,
    pub k: u32,
    pub scale_format: WeightQuantFormat,
}

/// 2026-09-25: The transpose of a block-scaled [`Fp8Weight`], built by
/// [`Fp8Weight::transpose_for_gemm`] for the prefill GEMM.
#[derive(Debug, Clone, Copy)]
pub struct Fp8WeightTransposed {
    /// 2026-09-25: `[K, N]` FP8 E4M3 bytes.
    pub weight_t: DevicePtr,
    /// 2026-09-25: `[K/128, N/128]` FP32 block scales, rounded up.
    pub scale_t: DevicePtr,
    pub n: u32,
    pub k: u32,
}

impl Fp8Weight {
    /// 2026-09-25: Transpose this weight into new buffers `weight_t[K, N]` and
    /// `scale_t[K/128, N/128]`, and synchronize `stream`. `row_scale` is read as
    /// FP32 block scales; `scale_format` is not checked.
    pub fn transpose_for_gemm(
        &self,
        gpu: &dyn GpuBackend,
        transpose_k: metrale_gpu_runtime::gpu::KernelHandle,
        transpose_scale_k: metrale_gpu_runtime::gpu::KernelHandle,
        stream: u64,
    ) -> anyhow::Result<Fp8WeightTransposed> {
        let n = self.n as usize;
        let k = self.k as usize;

        let weight_t = gpu.alloc(k * n)?;
        crate::layers::ops::transpose_fp8(
            gpu,
            transpose_k,
            self.weight,
            weight_t,
            self.n,
            self.k,
            stream,
        )?;

        let n_blocks = n.div_ceil(128);
        let k_blocks = k.div_ceil(128);
        let scale_t = gpu.alloc(k_blocks * n_blocks * 4)?;
        crate::layers::ops::transpose_block_scale(
            gpu,
            transpose_scale_k,
            self.row_scale,
            scale_t,
            n_blocks as u32,
            k_blocks as u32,
            stream,
        )?;

        gpu.synchronize(stream)?;

        Ok(Fp8WeightTransposed {
            weight_t,
            scale_t,
            n: self.n,
            k: self.k,
        })
    }
}
