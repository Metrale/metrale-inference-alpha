// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-25: Launchers for the DeepSeek-V4 Sinkhorn hyper-connection (mHC) kernels `hc_expand`, `hc_pre`, `hc_post` and `hc_head`.
//!
//! Owner: model-layers ops.
//! Invariants:
//! - The stream state is FP32 `[T, hc_mult, H]`, stream-major per token; the collapsed hidden
//!   state and the sublayer output are BF16 `[T, H]`; the HC parameters are FP32
//!   (kernels/gb10/deepseek-v4-flash/nvfp4/hyper_connection.cu).
//! - Every launch is one 256-thread block per token.
//! - The kernels support `hc_mult <= 4` (`HC_MAX_MULT`); these launchers do not check it.

use anyhow::Result;
use metrale_gpu_runtime::gpu::{DevicePtr, GpuBackend, KernelHandle};
use metrale_gpu_runtime::kernel_args::KernelLaunch;

/// 2026-09-25: Broadcast the BF16 hidden state into `hc_mult` identical FP32 streams:
/// `streams[t, i, d] = hidden[t, d]`.
pub fn hc_expand(
    gpu: &dyn GpuBackend,
    kernel: KernelHandle,
    hidden: DevicePtr,
    streams: DevicePtr,
    num_tokens: u32,
    hidden_size: u32,
    hc_mult: u32,
    stream: u64,
) -> Result<()> {
    KernelLaunch::new(gpu, kernel)
        .grid([num_tokens, 1, 1])
        .block([256, 1, 1])
        .arg_ptr(hidden)
        .arg_ptr(streams)
        .arg_u32(hidden_size)
        .arg_u32(hc_mult)
        .launch(stream)
}

/// 2026-09-25: Collapse the `hc_mult` streams to one BF16 row (an RMS-rescaled mix gives sigmoid
/// `pre` weights for the weighted sum), and write `post_out [T, hc]` and the Sinkhorn-normalized
/// `comb_out [T, hc, hc]` for the matching [`hc_post`].
#[allow(clippy::too_many_arguments)]
pub fn hc_pre(
    gpu: &dyn GpuBackend,
    kernel: KernelHandle,
    streams: DevicePtr,
    hc_fn: DevicePtr,
    hc_scale: DevicePtr,
    hc_base: DevicePtr,
    y_out: DevicePtr,
    post_out: DevicePtr,
    comb_out: DevicePtr,
    num_tokens: u32,
    hidden_size: u32,
    hc_mult: u32,
    sinkhorn_iters: u32,
    norm_eps: f32,
    hc_eps: f32,
    stream: u64,
) -> Result<()> {
    KernelLaunch::new(gpu, kernel)
        .grid([num_tokens, 1, 1])
        .block([256, 1, 1])
        .arg_ptr(streams)
        .arg_ptr(hc_fn)
        .arg_ptr(hc_scale)
        .arg_ptr(hc_base)
        .arg_ptr(y_out)
        .arg_ptr(post_out)
        .arg_ptr(comb_out)
        .arg_u32(hidden_size)
        .arg_u32(hc_mult)
        .arg_u32(sinkhorn_iters)
        .arg_f32(norm_eps)
        .arg_f32(hc_eps)
        .launch(stream)
}

/// 2026-09-25: Expand the sublayer output back into `hc_mult` streams:
/// `out[t,j,d] = post[t,j] * block_out[t,d] + sum_i comb[t,i,j] * residual[t,i,d]`. `out` may
/// alias `residual`: each element's residual values are read before its outputs are written.
#[allow(clippy::too_many_arguments)]
pub fn hc_post(
    gpu: &dyn GpuBackend,
    kernel: KernelHandle,
    block_out: DevicePtr,
    residual: DevicePtr,
    post: DevicePtr,
    comb: DevicePtr,
    out: DevicePtr,
    num_tokens: u32,
    hidden_size: u32,
    hc_mult: u32,
    stream: u64,
) -> Result<()> {
    KernelLaunch::new(gpu, kernel)
        .grid([num_tokens, 1, 1])
        .block([256, 1, 1])
        .arg_ptr(block_out)
        .arg_ptr(residual)
        .arg_ptr(post)
        .arg_ptr(comb)
        .arg_ptr(out)
        .arg_u32(hidden_size)
        .arg_u32(hc_mult)
        .launch(stream)
}

/// 2026-09-25: Final collapse before the LM head: one learned sigmoid-weighted sum over the
/// `hc_mult` streams, written as BF16 `[T, H]`.
#[allow(clippy::too_many_arguments)]
pub fn hc_head(
    gpu: &dyn GpuBackend,
    kernel: KernelHandle,
    streams: DevicePtr,
    head_fn: DevicePtr,
    head_scale: DevicePtr,
    head_base: DevicePtr,
    y_out: DevicePtr,
    num_tokens: u32,
    hidden_size: u32,
    hc_mult: u32,
    norm_eps: f32,
    hc_eps: f32,
    stream: u64,
) -> Result<()> {
    KernelLaunch::new(gpu, kernel)
        .grid([num_tokens, 1, 1])
        .block([256, 1, 1])
        .arg_ptr(streams)
        .arg_ptr(head_fn)
        .arg_ptr(head_scale)
        .arg_ptr(head_base)
        .arg_ptr(y_out)
        .arg_u32(hidden_size)
        .arg_u32(hc_mult)
        .arg_f32(norm_eps)
        .arg_f32(hc_eps)
        .launch(stream)
}
