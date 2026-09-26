// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-25: FP8 E4M3 with per-row scales: load-time quantization from BF16 and the checkpoint loader.
//!
//! Owner: model-layers (weight loading).
//! Invariants:
//! - Every returned FP8 weight carries one FP32 scale per row.

#![allow(unused_imports)]

use anyhow::{Context, Result, bail, ensure};
use metrale_gpu_runtime::gpu::{DevicePtr, GpuBackend};
use metrale_model_weights::weights::{WeightDtype, WeightStore};

use super::*;

/// 2026-09-25: Quantize a BF16 dense weight `[n, k]` to FP8 E4M3 (`n*k` bytes)
/// with one FP32 scale per row. It synchronizes `stream`, so it belongs at load
/// time, not in a forward pass.
pub fn quantize_to_fp8(
    bf16_weight: &DenseWeight,
    n: usize,
    k: usize,
    gpu: &dyn GpuBackend,
    quantize_kernel: metrale_gpu_runtime::gpu::KernelHandle,
    stream: u64,
) -> Result<Fp8DenseWeight> {
    use metrale_gpu_runtime::kernel_args::KernelLaunch;

    let fp8_buf = gpu.alloc(n * k)?;
    let scale_buf = gpu.alloc(n * 4)?;

    KernelLaunch::new(gpu, quantize_kernel)
        .grid([n as u32, 1, 1])
        .block([256, 1, 1])
        .arg_ptr(bf16_weight.weight)
        .arg_ptr(fp8_buf)
        .arg_ptr(scale_buf)
        .arg_u32(n as u32)
        .arg_u32(k as u32)
        .launch(stream)?;

    gpu.synchronize(stream)?;

    Ok(Fp8DenseWeight {
        weight: fp8_buf,
        row_scale: scale_buf,
    })
}

/// 2026-09-25: Load an FP8 E4M3 checkpoint weight with per-row scales as an
/// [`Fp8Weight`] tagged `Fp8PerRow`.
///
/// Expects `{name}.weight` FP8E4M3 `[N, K]` and `{name}.weight_scale` `[N]`.
/// The weight is the store's pointer. An FP32 scale is the store's pointer
/// too; a BF16 scale is widened on the host into a new buffer; any other dtype
/// is an error.
pub fn load_fp8_weight(store: &WeightStore, name: &str, gpu: &dyn GpuBackend) -> Result<Fp8Weight> {
    let w = store.get(&format!("{name}.weight"))?;
    ensure!(
        w.dtype == WeightDtype::FP8E4M3,
        "Expected FP8E4M3 for {name}.weight, got {:?}",
        w.dtype,
    );
    ensure!(
        w.shape.len() == 2,
        "Expected 2D weight for {name}, got {:?}",
        w.shape
    );
    let n = w.shape[0];
    let k = w.shape[1];

    let weight_ptr = w.ptr;

    let scale_key = format!("{name}.weight_scale");
    let s = store.get(&scale_key).with_context(|| {
        format!("Missing per-row scale tensor {scale_key} for FP8 weight {name}")
    })?;
    ensure!(
        s.shape.len() == 1 && s.shape[0] == n,
        "Expected [{n}] shape for {scale_key}, got {:?}",
        s.shape,
    );

    let row_scale_ptr = if s.dtype == WeightDtype::FP32 {
        s.ptr
    } else if s.dtype == WeightDtype::BF16 {
        let mut bf16_buf = vec![0u8; n * 2];
        gpu.copy_d2h(s.ptr, &mut bf16_buf)?;
        let mut f32_buf = vec![0u8; n * 4];
        for i in 0..n {
            let bf16_bytes = [bf16_buf[i * 2], bf16_buf[i * 2 + 1]];
            let val = bf16_bytes_to_f32(bf16_bytes);
            let f32_bytes = val.to_le_bytes();
            f32_buf[i * 4..i * 4 + 4].copy_from_slice(&f32_bytes);
        }
        let f32_ptr = gpu.alloc(n * 4)?;
        gpu.copy_h2d(&f32_buf, f32_ptr)?;
        f32_ptr
    } else {
        anyhow::bail!(
            "Unsupported dtype {:?} for {scale_key}, expected FP32 or BF16",
            s.dtype,
        );
    };

    Ok(Fp8Weight {
        weight: weight_ptr,
        row_scale: row_scale_ptr,
        n: n as u32,
        k: k as u32,
        scale_format: WeightQuantFormat::Fp8PerRow,
    })
}
