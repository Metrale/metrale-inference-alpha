// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-25: Launch wrappers for element-wise kernels: activation-gated products (SiLU or
//! GeLU), sigmoid and softplus gates, per-head L2 norm, residual and scaled adds.
//!
//! Owner: model-layers ops.
//! Invariants: none beyond the types.

#![allow(unused_imports)]

use anyhow::Result;
use metrale_gpu_runtime::gpu::{DevicePtr, GpuBackend, KernelHandle};
use metrale_gpu_runtime::kernel_args::{KernelLaunch, div_ceil};

use crate::layers::moe;
use crate::weight_map::{DenseWeight, Fp8DenseWeight, Fp8Weight, QuantizedWeight};

use super::*;

/// 2026-09-25: `output = act(gate) * up` over `num_elements` BF16 values, where `kernel`
/// fixes the activation: callers pass `moe_silu_mul` (SiLU) or, for a GeLU FFN, `gelu_mul`.
/// The `moe_silu_mul.cu` copies in `gb10/deepseek-v4-flash` (also compiled for
/// `longcat-flash-lite`) and `gb10/step3p7-flash` first clamp `gate` to at most 10 and `up`
/// to [-10, 10].
pub fn silu_mul(
    gpu: &dyn GpuBackend,
    kernel: KernelHandle,
    gate: DevicePtr,
    up: DevicePtr,
    output: DevicePtr,
    num_elements: u32,
    stream: u64,
) -> Result<()> {
    KernelLaunch::new(gpu, kernel)
        .grid([div_ceil(num_elements, 256), 1, 1])
        .block([256, 1, 1])
        .arg_ptr(gate)
        .arg_ptr(up)
        .arg_ptr(output)
        .arg_u32(num_elements)
        .launch(stream)
}

/// 2026-09-25: `SiLU(gate) * up` over row-strided BF16 operands (kernel
/// `silu_mul_strided`): row `r` reads `gate` and `up` at `r * in_stride + c` for
/// `c < cols` and writes `output` at `r * out_stride + c`. The fused dense-FFN gate+up
/// projection (`layers/dense_ffn_gateup_fused.rs`) passes the two column halves of one
/// `[rows, 2 * cols]` matrix (`up = gate.offset(cols * 2)`) and a contiguous output.
/// One grid row per matrix row.
#[allow(clippy::too_many_arguments)]
pub fn silu_mul_strided(
    gpu: &dyn GpuBackend,
    kernel: KernelHandle,
    gate: DevicePtr,
    up: DevicePtr,
    output: DevicePtr,
    rows: u32,
    cols: u32,
    in_stride: u32,
    out_stride: u32,
    stream: u64,
) -> Result<()> {
    KernelLaunch::new(gpu, kernel)
        .grid([div_ceil(cols, 256), rows, 1])
        .block([256, 1, 1])
        .arg_ptr(gate)
        .arg_ptr(up)
        .arg_ptr(output)
        .arg_u32(rows)
        .arg_u32(cols)
        .arg_u32(in_stride)
        .arg_u32(out_stride)
        .launch(stream)
}

/// 2026-09-25: `SiLU(gate) * up` quantized to FP8 E4M3 with one scale per 128-element
/// group of each row, in one kernel (`silu_mul_quant_fp8`); the FP8 MoE prefill uses it
/// before the down projection (`moe/forward_prefill_fp8.rs`).
///
/// `out_bf16` may be `DevicePtr::NULL`; when it is not, the kernel also writes the
/// post-SiLU BF16 values there. The caller must ensure `k % 128 == 0` and
/// `k / 128 <= 16` (the kernel's `SILU_QUANT_MAX_GROUPS`).
#[allow(clippy::too_many_arguments)]
pub fn silu_mul_quant_fp8(
    gpu: &dyn GpuBackend,
    kernel: KernelHandle,
    gate: DevicePtr,
    up: DevicePtr,
    out_fp8: DevicePtr,
    a_scale: DevicePtr,
    out_bf16: DevicePtr,
    m: u32,
    k: u32,
    stream: u64,
) -> Result<()> {
    KernelLaunch::new(gpu, kernel)
        .grid([m, 1, 1])
        .block([128, 1, 1])
        .arg_ptr(gate)
        .arg_ptr(up)
        .arg_ptr(out_fp8)
        .arg_ptr(a_scale)
        .arg_ptr(out_bf16)
        .arg_u32(m)
        .arg_u32(k)
        .launch(stream)
}

/// 2026-09-25: In-place L2 normalization of each head, `x / sqrt(sum(x^2) + eps)` (kernel
/// `l2_norm_bf16`). `data` is `[num_tokens, stride]` BF16 with `num_heads * head_dim`
/// used per token; one block per (head, token).
pub fn l2_norm(
    gpu: &dyn GpuBackend,
    kernel: KernelHandle,
    data: DevicePtr,
    num_heads: u32,
    head_dim: u32,
    eps: f32,
    num_tokens: u32,
    stride: u32,
    stream: u64,
) -> Result<()> {
    KernelLaunch::new(gpu, kernel)
        .grid([num_heads, num_tokens, 1])
        .block([head_dim.min(1024), 1, 1])
        .arg_ptr(data)
        .arg_u32(head_dim)
        .arg_f32(eps)
        .arg_u32(stride)
        .launch(stream)
}

/// 2026-09-25: `output[i] = input[i] * sigmoid(gate[i])` (kernel `sigmoid_gate_mul`).
pub fn sigmoid_gate_mul(
    gpu: &dyn GpuBackend,
    kernel: KernelHandle,
    input: DevicePtr,
    gate: DevicePtr,
    output: DevicePtr,
    num_elements: u32,
    stream: u64,
) -> Result<()> {
    KernelLaunch::new(gpu, kernel)
        .grid([div_ceil(num_elements, 256), 1, 1])
        .block([256, 1, 1])
        .arg_ptr(input)
        .arg_ptr(gate)
        .arg_ptr(output)
        .arg_u32(num_elements)
        .launch(stream)
}

/// 2026-09-25: `output[t, h, d] = input[t, h, d] * sigmoid(gate[t, h])`: one BF16 gate
/// value per head, `[num_tokens, nq]`, broadcast over the head's `hd` elements (kernel
/// `sigmoid_gate_mul_head_broadcast`).
pub fn sigmoid_gate_mul_head_broadcast(
    gpu: &dyn GpuBackend,
    kernel: KernelHandle,
    input: DevicePtr,
    gate: DevicePtr,
    output: DevicePtr,
    nq: u32,
    hd: u32,
    num_tokens: u32,
    stream: u64,
) -> Result<()> {
    let total = num_tokens * nq * hd;
    KernelLaunch::new(gpu, kernel)
        .grid([div_ceil(total, 256), 1, 1])
        .block([256, 1, 1])
        .arg_ptr(input)
        .arg_ptr(gate)
        .arg_ptr(output)
        .arg_u32(nq)
        .arg_u32(hd)
        .arg_u32(total)
        .launch(stream)
}

/// 2026-09-25: Like [`sigmoid_gate_mul_head_broadcast`] with softplus in place of sigmoid
/// (kernel `softplus_gate_mul_head_broadcast`).
#[allow(clippy::too_many_arguments)]
pub fn softplus_gate_mul_head_broadcast(
    gpu: &dyn GpuBackend,
    kernel: KernelHandle,
    input: DevicePtr,
    gate: DevicePtr,
    output: DevicePtr,
    nq: u32,
    hd: u32,
    num_tokens: u32,
    stream: u64,
) -> Result<()> {
    let total = num_tokens * nq * hd;
    KernelLaunch::new(gpu, kernel)
        .grid([div_ceil(total, 256), 1, 1])
        .block([256, 1, 1])
        .arg_ptr(input)
        .arg_ptr(gate)
        .arg_ptr(output)
        .arg_u32(nq)
        .arg_u32(hd)
        .arg_u32(total)
        .launch(stream)
}

/// 2026-09-25: In-place BF16 `residual[i] += src[i]` (kernel `bf16_residual_add`).
pub fn residual_add(
    gpu: &dyn GpuBackend,
    kernel: KernelHandle,
    residual: DevicePtr,
    src: DevicePtr,
    num_elements: u32,
    stream: u64,
) -> Result<()> {
    KernelLaunch::new(gpu, kernel)
        .grid([div_ceil(num_elements, 256), 1, 1])
        .block([256, 1, 1])
        .arg_ptr(residual)
        .arg_ptr(src)
        .arg_u32(num_elements)
        .launch(stream)
}

/// 2026-09-25: In-place BF16 `output[i] += scale * src[i]` (kernel `bf16_scaled_add`).
pub fn scaled_add(
    gpu: &dyn GpuBackend,
    kernel: KernelHandle,
    output: DevicePtr,
    src: DevicePtr,
    scale: f32,
    num_elements: u32,
    stream: u64,
) -> Result<()> {
    KernelLaunch::new(gpu, kernel)
        .grid([div_ceil(num_elements, 256), 1, 1])
        .block([256, 1, 1])
        .arg_ptr(output)
        .arg_ptr(src)
        .arg_f32(scale)
        .arg_u32(num_elements)
        .launch(stream)
}

/// 2026-09-25: In-place BF16 `output[i] += sigmoid_gate * src[i]`, where `sigmoid_gate`
/// is a host value the caller has already passed through the sigmoid. The arguments are
/// those of kernel `bf16_sigmoid_blend`; nothing in the tree calls this function.
pub fn sigmoid_blend(
    gpu: &dyn GpuBackend,
    kernel: KernelHandle,
    output: DevicePtr,
    src: DevicePtr,
    sigmoid_gate: f32,
    num_elements: u32,
    stream: u64,
) -> Result<()> {
    KernelLaunch::new(gpu, kernel)
        .grid([div_ceil(num_elements, 256), 1, 1])
        .block([256, 1, 1])
        .arg_ptr(output)
        .arg_ptr(src)
        .arg_f32(sigmoid_gate)
        .arg_u32(num_elements)
        .launch(stream)
}
