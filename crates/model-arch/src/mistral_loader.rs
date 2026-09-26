// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-25: Weight loader for Mistral Small 4 (`model_type = "mistral"`): MLA attention with MoE FFNs.
//!
//! Owner: model-arch (Mistral loader).
//! Invariants: none beyond the types.

use anyhow::Result;
use metrale_gpu_runtime::gpu::{DevicePtr, GpuBackend};

use metrale_model_layers::layers::ops;
use metrale_model_layers::weight_map::DenseWeight;

pub struct MistralWeightLoader;

/// 2026-09-25: Allocate device memory, or managed (UVM) memory once a device
/// allocation has failed. The first failure latches
/// `OpCache::note_alloc_fallback` on this backend; every later call on the
/// same backend then goes straight to managed memory.
pub(crate) fn gpu_alloc_or_managed(gpu: &dyn GpuBackend, bytes: usize) -> Result<DevicePtr> {
    if gpu.op_cache().alloc_fell_back() {
        return gpu.alloc_managed(bytes);
    }
    match gpu.alloc(bytes) {
        Ok(p) => Ok(p),
        Err(_) => {
            tracing::warn!(
                "GPU alloc failed ({bytes} bytes) — switching to managed for remaining allocations"
            );
            gpu.op_cache().note_alloc_fallback();
            gpu.alloc_managed(bytes)
        }
    }
}

#[allow(dead_code)]
fn gpu_matmul(
    a: DevicePtr,
    b: DevicePtr,
    m: usize,
    n: usize,
    k: usize,
    gpu: &dyn GpuBackend,
) -> Result<DevicePtr> {
    let bf16 = 2usize;
    let c = gpu_alloc_or_managed(gpu, m * n * bf16)?;
    let stream = gpu.default_stream();
    let gemm_k = gpu.kernel("gemm", "dense_gemm_bf16")?;
    let b_dense = DenseWeight { weight: b };
    ops::dense_gemm(
        gpu, gemm_k, a, &b_dense, c, m as u32, n as u32, k as u32, stream,
    )?;
    gpu.synchronize(stream)?;
    Ok(c)
}

pub(crate) mod loader_impl;
