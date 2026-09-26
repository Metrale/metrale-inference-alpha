// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-25: Load-time TurboQuant+ rotation of BF16 projection weights with the runtime
//! `wht_bf16_inplace` kernel.
//!
//! Owner: model-arch weight loader (Qwen3.5).
//! Invariants: none beyond the types.

use anyhow::Result;
use metrale_gpu_runtime::gpu::{DevicePtr, GpuBackend};
use metrale_gpu_runtime::kernel_args::KernelLaunch;
use metrale_model_layers::layers::try_kernel;

/// 2026-09-25: Runs `wht_bf16_inplace` over `weight_bf16` in place, one launch with grid
/// `outer * n_heads` and block 32: each contiguous `head_dim` chunk of the BF16 buffer gets
/// signs1, the normalised Walsh-Hadamard transform, then signs2 (`wht_bf16.cu`,
/// `tq_plus_signs.cuh`).
///
/// Errors when `head_dim` is not 128, 256 or 512, or when the kernel handle is missing.
/// Panics when `outer * n_heads` overflows. Does nothing when it is zero.
#[allow(dead_code)]
pub fn apply_canonical_rotation_inplace(
    gpu: &dyn GpuBackend,
    weight_bf16: DevicePtr,
    outer: usize,
    n_heads: usize,
    head_dim: usize,
    stream: u64,
) -> Result<()> {
    let total_heads = outer
        .checked_mul(n_heads)
        .expect("outer * n_heads overflow");
    if total_heads == 0 {
        return Ok(());
    }
    if !(head_dim == 128 || head_dim == 256 || head_dim == 512) {
        anyhow::bail!(
            "apply_canonical_rotation_inplace: unsupported head_dim {head_dim} (need 128, 256, or 512)"
        );
    }

    let wht_kernel = try_kernel(gpu, "wht_bf16", "wht_bf16_inplace");
    if wht_kernel.0 == 0 {
        anyhow::bail!("wht_bf16_inplace kernel handle not available");
    }

    KernelLaunch::new(gpu, wht_kernel)
        .grid([total_heads as u32, 1, 1])
        .block([32, 1, 1])
        .arg_ptr(weight_bf16)
        .arg_u32(head_dim as u32)
        .launch(stream)?;

    Ok(())
}
