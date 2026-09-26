// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-25: FP8 E4M3 → BF16 dequantization for weights with one per-tensor
//! scale: a GPU path for an FP32 scale, and a host path for an FP32 or BF16
//! scale.
//!
//! Owner: model-layers (weight loading).
//! Invariants: none beyond the types.

#![allow(unused_imports)]

use anyhow::{Context, Result, bail, ensure};
use metrale_gpu_runtime::gpu::{DevicePtr, GpuBackend};
use metrale_model_weights::weights::{WeightDtype, WeightStore};

use super::*;

/// 2026-09-25: Dequantize an FP8 E4M3 weight with an FP32 per-tensor scale into
/// `out` (BF16) on the GPU.
///
/// Launches `dequant_fp8_blockscaled_bf16` with a single block covering the
/// tensor (`block_n = N`, `block_k = K`, `sk = 1`), so every element reads
/// `weight_scale[0]`, which is already on the device. Like the host path
/// (`dequant_fp8_bytes_to_bf16`), it computes the E4M3 value times the scale in
/// f32 and rounds to BF16.
fn gpu_dequant_fp8_pertensor(
    store: &WeightStore,
    prefix: &str,
    gpu: &dyn GpuBackend,
    out: DevicePtr,
) -> Result<()> {
    use metrale_gpu_runtime::kernel_args::{KernelLaunch, div_ceil};

    let w = store.get(&format!("{prefix}.weight"))?;
    ensure!(
        w.dtype == WeightDtype::FP8E4M3,
        "Expected FP8E4M3 for {prefix}.weight, got {:?}",
        w.dtype,
    );
    // 2026-09-25: Any factorization N*K = total works (the scale is constant);
    // a 2D weight uses its own shape, anything else is flattened.
    let total = w.num_elements();
    let (n, k) = if w.shape.len() == 2 {
        (w.shape[0], w.shape[1])
    } else {
        (1usize, total)
    };
    ensure!(
        n * k == total,
        "FP8 shape mismatch for {prefix}: {n}*{k} != {total}"
    );

    let s = store.get(&format!("{prefix}.weight_scale"))?;
    ensure!(
        s.dtype == WeightDtype::FP32 && s.num_elements() == 1,
        "Expected FP32 scalar weight_scale for {prefix}, got {:?} ({} elems)",
        s.dtype,
        s.num_elements(),
    );

    let stream = gpu.default_stream();
    let kernel = gpu.kernel(
        "dequant_fp8_blockscaled_bf16",
        "dequant_fp8_blockscaled_bf16",
    )?;
    KernelLaunch::new(gpu, kernel)
        .grid([div_ceil(k as u32, 64), div_ceil(n as u32, 4), 1])
        .block([64, 4, 1])
        .arg_ptr(w.ptr)
        .arg_ptr(s.ptr)
        .arg_ptr(out)
        .arg_u32(n as u32)
        .arg_u32(k as u32)
        .arg_u32(n as u32)
        .arg_u32(k as u32)
        .arg_u32(1)
        .arg_u32(1)
        .launch(stream)?;
    // 2026-09-25: No sync after the launch: later work on the same stream is
    // ordered after it, and a fault surfaces at the next sync.
    Ok(())
}

/// 2026-09-25: Read a per-tensor `weight_scale` scalar (FP32 or BF16) from the
/// store as `f32`, for the host dequant path.
fn read_scalar_weight_scale(
    store: &WeightStore,
    prefix: &str,
    gpu: &dyn GpuBackend,
) -> Result<f32> {
    let scale_key = format!("{prefix}.weight_scale");
    let s = store.get(&scale_key)?;
    ensure!(
        s.num_elements() == 1,
        "Expected scalar for {scale_key}, got {} elements",
        s.num_elements()
    );
    match s.dtype {
        WeightDtype::FP32 => {
            let mut buf = [0u8; 4];
            gpu.copy_d2h(s.ptr, &mut buf)?;
            Ok(f32::from_le_bytes(buf))
        }
        WeightDtype::BF16 => {
            let mut buf = [0u8; 2];
            gpu.copy_d2h(s.ptr, &mut buf)?;
            Ok(bf16_bytes_to_f32(buf))
        }
        other => bail!("Expected FP32 or BF16 for {scale_key}, got {:?}", other),
    }
}

/// 2026-09-25: Dequantize FP8 E4M3 + per-tensor scale → BF16 into a newly
/// allocated GPU buffer. `dequant_fp8_to_bf16_into` writes into a caller's
/// buffer instead.
pub(crate) fn dequant_fp8_to_bf16(
    store: &WeightStore,
    prefix: &str,
    gpu: &dyn GpuBackend,
) -> Result<DenseWeight> {
    let w = store.get(&format!("{prefix}.weight"))?;
    // 2026-09-25: An FP32 scalar scale takes the GPU path; any other scale (a
    // BF16 scalar) takes the host path, which reads FP32 or BF16.
    let scale_is_fp32_scalar = store
        .get(&format!("{prefix}.weight_scale"))
        .map(|s| s.dtype == WeightDtype::FP32 && s.num_elements() == 1)
        .unwrap_or(false);
    if scale_is_fp32_scalar {
        let total = w.num_elements();
        let out = gpu.alloc(total * 2)?;
        gpu_dequant_fp8_pertensor(store, prefix, gpu, out)?;
        Ok(DenseWeight { weight: out })
    } else {
        let n_bytes = w.num_elements();
        let mut fp8_buf = vec![0u8; n_bytes];
        gpu.copy_d2h(w.ptr, &mut fp8_buf)?;

        let scale = read_scalar_weight_scale(store, prefix, gpu)?;
        let bf16_buf = dequant_fp8_bytes_to_bf16(&fp8_buf, scale);
        let ptr = gpu.alloc(bf16_buf.len())?;
        gpu.copy_h2d(&bf16_buf, ptr)?;
        Ok(DenseWeight { weight: ptr })
    }
}

/// 2026-09-25: Dequantize FP8 E4M3 + per-tensor scale → BF16 into `dest`, a
/// caller-provided buffer of at least `2 × elements` bytes, and return it as a
/// `DenseWeight`. The GPU path does not synchronise, so the caller must order
/// any reuse of `dest` after the launch (e.g. after `quantize_to_nvfp4`, which
/// synchronises its stream).
pub fn dequant_fp8_to_bf16_into(
    store: &WeightStore,
    prefix: &str,
    gpu: &dyn GpuBackend,
    dest: DevicePtr,
) -> Result<DenseWeight> {
    // 2026-09-25: FP32 scalar scale → GPU path; otherwise the host path.
    let scale_is_fp32_scalar = store
        .get(&format!("{prefix}.weight_scale"))
        .map(|s| s.dtype == WeightDtype::FP32 && s.num_elements() == 1)
        .unwrap_or(false);
    if scale_is_fp32_scalar {
        gpu_dequant_fp8_pertensor(store, prefix, gpu, dest)?;
    } else {
        let w = store.get(&format!("{prefix}.weight"))?;
        let n_bytes = w.num_elements();
        let mut fp8_buf = vec![0u8; n_bytes];
        gpu.copy_d2h(w.ptr, &mut fp8_buf)?;

        let scale = read_scalar_weight_scale(store, prefix, gpu)?;
        let bf16_buf = dequant_fp8_bytes_to_bf16(&fp8_buf, scale);
        gpu.copy_h2d(&bf16_buf, dest)?;
    }
    Ok(DenseWeight { weight: dest })
}
