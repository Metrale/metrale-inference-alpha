// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-25: Launchers for the PLE kernels in
//! `kernels/gb10/qwen3.8-flash-next/nvfp4/ple.cu`: the gate, the dilated
//! depthwise conv, and the highway add.
//!
//! Owner: model-layers ops.
//! Invariants: none beyond the types.

use anyhow::Result;
use metrale_gpu_runtime::gpu::{DevicePtr, GpuBackend, KernelHandle};
use metrale_gpu_runtime::kernel_args::KernelLaunch;

/// 2026-09-25: Gate the n-gram value by the highway, and write both the gated
/// value and its `norm_conv` output. `hidden` is the FP32 highway `[T, hc*H]`;
/// `key` and `value` are BF16 projection outputs. Both outputs are FP32, because
/// the chain ends on the FP32 highway (the precision note in `ple.cu`). One
/// block per token.
#[allow(clippy::too_many_arguments)]
pub fn ple_gate(
    gpu: &dyn GpuBackend,
    kernel: KernelHandle,
    hidden: DevicePtr,
    key: DevicePtr,
    value: DevicePtr,
    norm_query_w: DevicePtr,
    norm_key_w: DevicePtr,
    norm_conv_w: DevicePtr,
    gated_out: DevicePtr,
    gated_normed: DevicePtr,
    num_tokens: u32,
    hidden_size: u32,
    hc_mult: u32,
    norm_eps: f32,
    stream: u64,
) -> Result<()> {
    KernelLaunch::new(gpu, kernel)
        .grid([num_tokens, 1, 1])
        .block([256, 1, 1])
        .arg_ptr(hidden)
        .arg_ptr(key)
        .arg_ptr(value)
        .arg_ptr(norm_query_w)
        .arg_ptr(norm_key_w)
        .arg_ptr(norm_conv_w)
        .arg_ptr(gated_out)
        .arg_ptr(gated_normed)
        .arg_u32(hidden_size)
        .arg_u32(hc_mult)
        .arg_f32(norm_eps)
        .launch(stream)
}

/// 2026-09-25: Depthwise causal conv with kernel `k_size` and dilation
/// `dilation`, then `out = gated + silu(conv)`. All buffers but `weight` (BF16)
/// are FP32. `state` is `[(k_size - 1) * dilation, channels]` and is rolled in
/// place, so prefill and decode use the same launch.
#[allow(clippy::too_many_arguments)]
pub fn ple_conv(
    gpu: &dyn GpuBackend,
    kernel: KernelHandle,
    x: DevicePtr,
    gated: DevicePtr,
    weight: DevicePtr,
    state: DevicePtr,
    out: DevicePtr,
    num_tokens: u32,
    channels: u32,
    k_size: u32,
    dilation: u32,
    stream: u64,
) -> Result<()> {
    let threads = 256u32;
    KernelLaunch::new(gpu, kernel)
        .grid([channels.div_ceil(threads), 1, 1])
        .block([threads, 1, 1])
        .arg_ptr(x)
        .arg_ptr(gated)
        .arg_ptr(weight)
        .arg_ptr(state)
        .arg_ptr(out)
        .arg_u32(num_tokens)
        .arg_u32(channels)
        .arg_u32(k_size)
        .arg_u32(dilation)
        .launch(stream)
}

/// 2026-09-25: `highway += ple_out`, in FP32. The reference model
/// (`bench/qwen4_exp/ref/modeling_qwen4_exp.py`) adds PLE's output to the
/// hidden state before that layer's attention hyper-connection.
pub fn ple_add_highway(
    gpu: &dyn GpuBackend,
    kernel: KernelHandle,
    ple_out: DevicePtr,
    hidden: DevicePtr,
    n: u32,
    stream: u64,
) -> Result<()> {
    let threads = 256u32;
    KernelLaunch::new(gpu, kernel)
        .grid([n.div_ceil(threads), 1, 1])
        .block([threads, 1, 1])
        .arg_ptr(ple_out)
        .arg_ptr(hidden)
        .arg_u32(n)
        .launch(stream)
}
