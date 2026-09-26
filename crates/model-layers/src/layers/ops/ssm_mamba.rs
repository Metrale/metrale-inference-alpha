// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-25: Launchers for the causal conv1d updates (decode, fused L2 norm,
//! strided, prefill) and the Mamba-2 SSM prefill kernels, plus the re-export
//! of the GDN verify conv launchers.
//!
//! Owner: model-layers ops (SSM).
//! Invariants: none beyond the types.

#![allow(unused_imports)]

use anyhow::Result;
use metrale_gpu_runtime::gpu::{DevicePtr, GpuBackend, KernelHandle};
use metrale_gpu_runtime::kernel_args::{KernelLaunch, div_ceil};

use crate::layers::moe;
use crate::weight_map::{DenseWeight, Fp8DenseWeight, Fp8Weight, QuantizedWeight};

use super::*;
#[path = "ssm_mamba_gdn_verify.rs"]
mod gdn_verify;
pub use gdn_verify::{
    gdn_verify_fused_conv_k2, gdn_verify_fused_conv_kn, gdn_verify_fused_conv_kn_batched,
    gdn_verify_fused_norm_k2,
};

/// 2026-09-25: Causal conv1d update + SiLU for one decode token per sequence
/// (`causal_conv1d_update`). `conv_state`, `input` and `output` are contiguous
/// per sequence: `[batch, d_inner, d_conv]`, `[batch, d_inner]`,
/// `[batch, d_inner]`.
pub fn conv1d_update(
    gpu: &dyn GpuBackend,
    kernel: KernelHandle,
    conv_state: DevicePtr,
    input: DevicePtr,
    weight: &DenseWeight,
    output: DevicePtr,
    d_inner: u32,
    d_conv: u32,
    batch_size: u32,
    stream: u64,
) -> Result<()> {
    let bias_ptr = DevicePtr::NULL;
    KernelLaunch::new(gpu, kernel)
        .grid([div_ceil(d_inner, 256), batch_size, 1])
        .block([256, 1, 1])
        .arg_ptr(conv_state)
        .arg_ptr(input)
        .arg_ptr(weight.weight)
        .arg_ptr(bias_ptr)
        .arg_ptr(output)
        .arg_u32(batch_size)
        .arg_u32(d_inner)
        .arg_u32(d_conv)
        .launch(stream)
}

/// 2026-09-25: Conv1d update + SiLU, then an L2 norm per `head_dim` group over
/// the Q and K channels (`0..qk_channels`), in one kernel
/// (`causal_conv1d_update_l2norm`). The V channels get SiLU only.
#[allow(clippy::too_many_arguments)]
pub fn conv1d_update_l2norm(
    gpu: &dyn GpuBackend,
    kernel: KernelHandle,
    conv_state: DevicePtr,
    input: DevicePtr,
    weight: &DenseWeight,
    output: DevicePtr,
    d_inner: u32,
    d_conv: u32,
    batch_size: u32,
    qk_channels: u32,
    head_dim: u32,
    l2_eps: f32,
    stream: u64,
) -> Result<()> {
    let bias_ptr = DevicePtr::NULL;
    KernelLaunch::new(gpu, kernel)
        .grid([div_ceil(d_inner, 256), batch_size, 1])
        .block([256, 1, 1])
        .arg_ptr(conv_state)
        .arg_ptr(input)
        .arg_ptr(weight.weight)
        .arg_ptr(bias_ptr)
        .arg_ptr(output)
        .arg_u32(batch_size)
        .arg_u32(d_inner)
        .arg_u32(d_conv)
        .arg_u32(qk_channels)
        .arg_u32(head_dim)
        .arg_f32(l2_eps)
        .launch(stream)
}

/// 2026-09-25: Conv1d update + SiLU + L2 norm for `batch_size` decode
/// sequences in one launch, with FP32 output
/// (`causal_conv1d_update_l2norm_f32_strided`). The input and output row
/// strides are separate arguments because the concurrent-decode input rows
/// are the QKVZ projection's, `qkvz_size` apart, while the output rows are
/// `d_inner` apart; a kernel that used `d_inner` for both would read sequence
/// `b >= 1` from the wrong row.
///
/// `conv_state` keeps the `(b * d_inner + ch) * d_conv` layout, so the caller
/// must have checked that the sequences' pool slots are contiguous.
#[allow(clippy::too_many_arguments)]
pub fn conv1d_update_l2norm_strided(
    gpu: &dyn GpuBackend,
    kernel: KernelHandle,
    conv_state: DevicePtr,
    input: DevicePtr,
    weight: &DenseWeight,
    output: DevicePtr,
    d_inner: u32,
    d_conv: u32,
    batch_size: u32,
    qk_channels: u32,
    head_dim: u32,
    l2_eps: f32,
    input_stride: u32,
    output_stride: u32,
    stream: u64,
) -> Result<()> {
    let bias_ptr = DevicePtr::NULL;
    KernelLaunch::new(gpu, kernel)
        .grid([div_ceil(d_inner, 256), batch_size, 1])
        .block([256, 1, 1])
        .arg_ptr(conv_state)
        .arg_ptr(input)
        .arg_ptr(weight.weight)
        .arg_ptr(bias_ptr)
        .arg_ptr(output)
        .arg_u32(batch_size)
        .arg_u32(d_inner)
        .arg_u32(d_conv)
        .arg_u32(qk_channels)
        .arg_u32(head_dim)
        .arg_f32(l2_eps)
        .arg_u32(input_stride)
        .arg_u32(output_stride)
        .launch(stream)
}

/// 2026-09-25: Conv1d update + SiLU over `seq_len` prefill tokens, with
/// separate per-token input and output strides. Runs the token-parallel kernel
/// (`conv1d_prefill_tp_k`) when its handle is non-zero and
/// `METRALE_CONV1D_TP` is not `0`, else `causal_conv1d_update_prefill`, one
/// thread per channel walking the tokens.
#[allow(clippy::too_many_arguments)]
pub fn conv1d_update_prefill(
    gpu: &dyn GpuBackend,
    kernel: KernelHandle,
    conv1d_prefill_tp_k: KernelHandle,
    conv_state: DevicePtr,
    input: DevicePtr,
    weight: &DenseWeight,
    bias: DevicePtr,
    output: DevicePtr,
    d_inner: u32,
    d_conv: u32,
    seq_len: u32,
    input_stride: u32,
    output_stride: u32,
    stream: u64,
) -> Result<()> {
    // 2026-09-25: Tokens can run in parallel because output[t] depends only on
    // the inputs x[t-3..=t], never on an earlier output. In the token-parallel
    // kernel a warp spans 32 channels, so the `[t * stride + ch]` loads
    // coalesce, and each thread produces 8 consecutive tokens: grid y is
    // `ceil(seq_len / 64)` for blocks of 8 thread rows.
    let tp = std::env::var("METRALE_CONV1D_TP").ok().as_deref() != Some("0")
        && conv1d_prefill_tp_k.0 != 0;
    let (k, grid, block) = if tp {
        (
            conv1d_prefill_tp_k,
            [div_ceil(d_inner, 32), div_ceil(seq_len, 64), 1],
            [32u32, 8u32, 1u32],
        )
    } else {
        (kernel, [div_ceil(d_inner, 256), 1, 1], [256u32, 1u32, 1u32])
    };
    KernelLaunch::new(gpu, k)
        .grid(grid)
        .block(block)
        .arg_ptr(conv_state)
        .arg_ptr(input)
        .arg_ptr(weight.weight)
        .arg_ptr(bias)
        .arg_ptr(output)
        .arg_u32(d_inner)
        .arg_u32(d_conv)
        .arg_u32(seq_len)
        .arg_u32(input_stride)
        .arg_u32(output_stride)
        .launch(stream)
}

/// 2026-09-25: Mamba-2 SSM prefill (`mamba2_ssm_prefill`): the token-sequential
/// recurrence over `seq_len` tokens in one launch. `x_stride`, `bc_stride`,
/// `dt_stride` and `y_stride` are BF16 elements between consecutive tokens.
#[allow(clippy::too_many_arguments)]
pub fn mamba2_ssm_prefill(
    gpu: &dyn GpuBackend,
    kernel: KernelHandle,
    h_state: DevicePtr,
    x: DevicePtr,
    b_proj: DevicePtr,
    c_proj: DevicePtr,
    dt_raw: DevicePtr,
    a_log: DevicePtr,
    d_param: DevicePtr,
    dt_bias: DevicePtr,
    output: DevicePtr,
    batch_size: u32,
    seq_len: u32,
    num_heads: u32,
    head_dim: u32,
    state_size: u32,
    n_groups: u32,
    dt_min: f32,
    dt_max: f32,
    x_stride: u32,
    bc_stride: u32,
    dt_stride: u32,
    y_stride: u32,
    stream: u64,
) -> Result<()> {
    KernelLaunch::new(gpu, kernel)
        .grid([num_heads, batch_size, 1])
        .block([state_size, 1, 1])
        .arg_ptr(h_state)
        .arg_ptr(x)
        .arg_ptr(b_proj)
        .arg_ptr(c_proj)
        .arg_ptr(dt_raw)
        .arg_ptr(a_log)
        .arg_ptr(d_param)
        .arg_ptr(dt_bias)
        .arg_ptr(output)
        .arg_u32(batch_size)
        .arg_u32(seq_len)
        .arg_u32(num_heads)
        .arg_u32(head_dim)
        .arg_u32(state_size)
        .arg_u32(n_groups)
        .arg_f32(dt_min)
        .arg_f32(dt_max)
        .arg_u32(x_stride)
        .arg_u32(bc_stride)
        .arg_u32(dt_stride)
        .arg_u32(y_stride)
        .launch(stream)
}

/// 2026-09-25: Mamba-2 SSM prefill with each block's H slice in shared memory
/// for the whole token loop (`mamba2_ssm_prefill_persistent`). Same arguments
/// as [`mamba2_ssm_prefill`], but launched with `head_dim * SUB` threads and
/// dynamic shared memory.
#[allow(clippy::too_many_arguments)]
pub fn mamba2_ssm_prefill_persistent(
    gpu: &dyn GpuBackend,
    kernel: KernelHandle,
    h_state: DevicePtr,
    x: DevicePtr,
    b_proj: DevicePtr,
    c_proj: DevicePtr,
    dt_raw: DevicePtr,
    a_log: DevicePtr,
    d_param: DevicePtr,
    dt_bias: DevicePtr,
    output: DevicePtr,
    batch_size: u32,
    seq_len: u32,
    num_heads: u32,
    head_dim: u32,
    state_size: u32,
    n_groups: u32,
    dt_min: f32,
    dt_max: f32,
    x_stride: u32,
    bc_stride: u32,
    dt_stride: u32,
    y_stride: u32,
    stream: u64,
) -> Result<()> {
    // 2026-09-25: Must match the kernel's dynamic shared-memory layout:
    //   sH     : head_dim * (state_size + 1)  (the +1 pad avoids bank conflicts)
    //   smem_x : head_dim
    //   smem_B : state_size   (dt*B for the current token)
    //   smem_C : state_size
    let smem = head_dim * (state_size + 1) * 4 + head_dim * 4 + state_size * 4 + state_size * 4;
    // 2026-09-25: SUB threads cooperate per head_dim row; must equal the
    // kernel's `SUB`.
    const SUB: u32 = 4;
    KernelLaunch::new(gpu, kernel)
        .grid([num_heads, batch_size, 1])
        .block([head_dim * SUB, 1, 1])
        .shared_mem(smem)
        .arg_ptr(h_state)
        .arg_ptr(x)
        .arg_ptr(b_proj)
        .arg_ptr(c_proj)
        .arg_ptr(dt_raw)
        .arg_ptr(a_log)
        .arg_ptr(d_param)
        .arg_ptr(dt_bias)
        .arg_ptr(output)
        .arg_u32(batch_size)
        .arg_u32(seq_len)
        .arg_u32(num_heads)
        .arg_u32(head_dim)
        .arg_u32(state_size)
        .arg_u32(n_groups)
        .arg_f32(dt_min)
        .arg_f32(dt_max)
        .arg_u32(x_stride)
        .arg_u32(bc_stride)
        .arg_u32(dt_stride)
        .arg_u32(y_stride)
        .launch(stream)
}
