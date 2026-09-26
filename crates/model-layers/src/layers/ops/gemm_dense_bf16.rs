// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-25: Launchers for the dense BF16 GEMMs: scalar, tensor-core,
//! scaled-accumulate, split-K, router, pipelined and prefill. The W4A16
//! launchers are in `gemm_dense.rs`, which the arity tests read.
//!
//! Owner: model-layers ops.
//! Invariants: none beyond the types.

use super::*;

/// 2026-09-25: Tensor-core BF16 GEMM, `C = A @ B^T`, with m16n8k16 MMAs
/// (kernel `dense_gemm_tc`).
pub fn dense_gemm_tc(
    gpu: &dyn GpuBackend,
    kernel: KernelHandle,
    input: DevicePtr,
    weight: &DenseWeight,
    output: DevicePtr,
    m: u32,
    n: u32,
    k: u32,
    stream: u64,
) -> Result<()> {
    KernelLaunch::new(gpu, kernel)
        .grid([div_ceil(n, 64), div_ceil(m, 16), 1])
        .block([128, 1, 1])
        .arg_ptr(input)
        .arg_ptr(weight.weight)
        .arg_ptr(output)
        .arg_u32(m)
        .arg_u32(n)
        .arg_u32(k)
        .launch(stream)
}

/// 2026-09-25: `output[m, n] += scale * bf16(input[m, k] @ weight[n, k]^T)` in
/// one pass: [`dense_gemm_tc`] with an accumulating epilogue, used for the
/// LoRA expand+fold (`lora_delta.rs`). It saves the `[m, n]` scratch that a
/// GEMM followed by `bf16_scaled_add` writes and reads back.
///
/// Same result as that pair: the kernel rounds each dot product to BF16
/// before it applies `scale` and adds, as `bf16_scaled_add` does with a BF16
/// scratch.
#[allow(clippy::too_many_arguments)]
pub fn dense_gemm_tc_scaled_acc(
    gpu: &dyn GpuBackend,
    kernel: KernelHandle,
    input: DevicePtr,
    weight: &DenseWeight,
    output: DevicePtr,
    m: u32,
    n: u32,
    k: u32,
    scale: f32,
    stream: u64,
) -> Result<()> {
    KernelLaunch::new(gpu, kernel)
        .grid([div_ceil(n, 64), div_ceil(m, 16), 1])
        .block([128, 1, 1])
        .arg_ptr(input)
        .arg_ptr(weight.weight)
        .arg_ptr(output)
        .arg_u32(m)
        .arg_u32(n)
        .arg_u32(k)
        .arg_f32(scale)
        .launch(stream)
}

/// 2026-09-25: Split-K BF16 GEMM: `partial_kernel` writes FP32 partial
/// products for `k_splits` K chunks into `workspace`, and `reduce_kernel` sums
/// them into BF16 `output`. `workspace` holds `k_splits * m * n` FP32 values.
#[allow(clippy::too_many_arguments)]
pub fn dense_gemm_splitk(
    gpu: &dyn GpuBackend,
    partial_kernel: KernelHandle,
    reduce_kernel: KernelHandle,
    input: DevicePtr,
    weight: &DenseWeight,
    output: DevicePtr,
    workspace: DevicePtr,
    m: u32,
    n: u32,
    k: u32,
    k_splits: u32,
    stream: u64,
) -> Result<()> {
    KernelLaunch::new(gpu, partial_kernel)
        .grid([div_ceil(n, 16), div_ceil(m, 16), k_splits])
        .block([16, 16, 1])
        .arg_ptr(input)
        .arg_ptr(weight.weight)
        .arg_ptr(workspace)
        .arg_u32(m)
        .arg_u32(n)
        .arg_u32(k)
        .arg_u32(k_splits)
        .launch(stream)?;
    KernelLaunch::new(gpu, reduce_kernel)
        .grid([div_ceil(n, 256), m, 1])
        .block([256, 1, 1])
        .arg_ptr(workspace)
        .arg_ptr(output)
        .arg_u32(m)
        .arg_u32(n)
        .arg_u32(k_splits)
        .launch(stream)
}

/// 2026-09-25: Scalar BF16 GEMM, `C = A @ B^T` (kernel
/// `dense_gemm_bf16(A, B, C, M, N, K)`): A `[M, K]` row-major, B `[N, K]`
/// row-major (HuggingFace layout), C `[M, N]` row-major.
pub fn dense_gemm(
    gpu: &dyn GpuBackend,
    kernel: KernelHandle,
    input: DevicePtr,
    weight: &DenseWeight,
    output: DevicePtr,
    m: u32,
    n: u32,
    k: u32,
    stream: u64,
) -> Result<()> {
    KernelLaunch::new(gpu, kernel)
        .grid([div_ceil(n, 16), div_ceil(m, 16), 1])
        .block([16, 16, 1])
        .arg_ptr(input)
        .arg_ptr(weight.weight)
        .arg_ptr(output)
        .arg_u32(m)
        .arg_u32(n)
        .arg_u32(k)
        .launch(stream)
}

/// 2026-09-25: Register-blocked BF16 GEMM (kernel `dense_gemm_bf16_router`)
/// that keeps the scalar `dense_gemm_bf16`'s per-output FP32 accumulation
/// order, strict k = 0..K-1, and with it the scalar kernel's results under the
/// `--fmad=false` build (`kernels/gb10/common/KERNEL.toml`).
/// `router_gate_gemm_dense` uses it for the MoE router gate GEMM.
pub fn dense_gemm_router(
    gpu: &dyn GpuBackend,
    kernel: KernelHandle,
    input: DevicePtr,
    weight: &DenseWeight,
    output: DevicePtr,
    m: u32,
    n: u32,
    k: u32,
    stream: u64,
) -> Result<()> {
    KernelLaunch::new(gpu, kernel)
        .grid([div_ceil(n, 64), div_ceil(m, 16), 1])
        .block([16, 16, 1])
        .arg_ptr(input)
        .arg_ptr(weight.weight)
        .arg_ptr(output)
        .arg_u32(m)
        .arg_u32(n)
        .arg_u32(k)
        .launch(stream)
}

/// 2026-09-25: Tensor-core BF16 GEMM (kernel `dense_gemm_bf16_pipelined`):
/// m16n8k16 MMAs fed by a 2-stage `cp.async` pipeline on a 128x128 tile, with
/// the inputs and output layout of [`dense_gemm`].
#[allow(clippy::too_many_arguments)]
pub fn dense_gemm_bf16_pipelined(
    gpu: &dyn GpuBackend,
    kernel: KernelHandle,
    input: DevicePtr,
    weight: &DenseWeight,
    output: DevicePtr,
    m: u32,
    n: u32,
    k: u32,
    stream: u64,
) -> Result<()> {
    KernelLaunch::new(gpu, kernel)
        .grid([div_ceil(n, 128), div_ceil(m, 128), 1])
        .block([256, 1, 1])
        .arg_ptr(input)
        .arg_ptr(weight.weight)
        .arg_ptr(output)
        .arg_u32(m)
        .arg_u32(n)
        .arg_u32(k)
        .launch(stream)
}

/// 2026-09-25: Dense BF16 prefill GEMM: [`dense_gemm_bf16_pipelined`] when
/// `pipelined_kernel` is loaded, otherwise [`dense_gemm`] with
/// `fallback_kernel`.
#[allow(clippy::too_many_arguments)]
pub fn dense_gemm_prefill(
    gpu: &dyn GpuBackend,
    fallback_kernel: KernelHandle,
    pipelined_kernel: KernelHandle,
    input: DevicePtr,
    weight: &DenseWeight,
    output: DevicePtr,
    m: u32,
    n: u32,
    k: u32,
    stream: u64,
) -> Result<()> {
    if pipelined_kernel.0 != 0 {
        dense_gemm_bf16_pipelined(
            gpu,
            pipelined_kernel,
            input,
            weight,
            output,
            m,
            n,
            k,
            stream,
        )
    } else {
        dense_gemm(gpu, fallback_kernel, input, weight, output, m, n, k, stream)
    }
}
