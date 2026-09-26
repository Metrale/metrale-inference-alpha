// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-26: Bring up the GPU backend of the enabled feature, record the
//! free-memory baseline and log the device.
//!
//! Owner: server startup (`met serve`).
//! Invariants:
//! - On success the free-memory baseline is set to the free memory returned.

use anyhow::{Context, Result};

use crate::cli;

/// 2026-09-26: Initialize the GPU backend and return it with its free memory.
///
/// With `cuda`, the device-arch gate runs first, then `MetraleCudaBackend`
/// loads `ptx_set.modules`. With `metal` and not `cuda`, `MetalGpuBackend`
/// loads `ptx_set.modules`. Both use the resolved target's modules, not
/// `ptx_modules()`/`metallib_modules()`, which name the first target in a
/// multi-target build.
#[cfg(feature = "cuda")]
pub(crate) fn init_gpu_backend(
    args: &cli::ServeArgs,
    ptx_set: &metrale_kernels::TargetPtxSet,
) -> Result<(Box<dyn metrale_gpu_runtime::gpu::GpuBackend>, usize)> {
    super::super::kernel_gate::gate_device_arch(args.check_kernels, ptx_set, args.gpu_ordinal)?;

    let backend = metrale_gpu_runtime::cuda_backend::MetraleCudaBackend::new(
        args.gpu_ordinal,
        &ptx_set.modules,
    )
    .context("Failed to initialize CUDA backend")?;

    let gpu: Box<dyn metrale_gpu_runtime::gpu::GpuBackend> = Box::new(backend);
    let total_mem = gpu.total_memory()?;
    let free_mem = gpu.free_memory()?;
    // 2026-09-26: Free memory after the context and modules, before weights:
    // the baseline `factory::build`'s KV budgeting measures this process's
    // own use against.
    metrale_gpu_runtime::gpu::set_baseline_free_bytes(free_mem);
    tracing::info!(
        "GPU {}: {:.1} GB total, {:.1} GB free",
        args.gpu_ordinal,
        total_mem as f64 / (1024.0 * 1024.0 * 1024.0),
        free_mem as f64 / (1024.0 * 1024.0 * 1024.0),
    );
    Ok((gpu, free_mem))
}

#[cfg(all(feature = "metal", not(feature = "cuda")))]
pub(crate) fn init_gpu_backend(
    args: &cli::ServeArgs,
    ptx_set: &metrale_kernels::TargetPtxSet,
) -> Result<(Box<dyn metrale_gpu_runtime::gpu::GpuBackend>, usize)> {
    let gpu: Box<dyn metrale_gpu_runtime::gpu::GpuBackend> = Box::new(
        metrale_gpu_runtime::metal_backend::MetalGpuBackend::new(
            args.gpu_ordinal,
            &ptx_set.modules,
        )
        .context("Failed to initialize Metal backend")?,
    );
    let total_mem = gpu.total_memory()?;
    let free_mem = gpu.free_memory()?;
    metrale_gpu_runtime::gpu::set_baseline_free_bytes(free_mem);
    tracing::info!(
        "Metal device {}: {:.1} GB total, {:.1} GB free",
        args.gpu_ordinal,
        total_mem as f64 / (1024.0 * 1024.0 * 1024.0),
        free_mem as f64 / (1024.0 * 1024.0 * 1024.0),
    );
    Ok((gpu, free_mem))
}
