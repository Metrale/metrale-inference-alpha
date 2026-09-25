// SPDX-License-Identifier: AGPL-3.0-only

use crate::weight_map::DenseWeight;
use anyhow::Result;
use spark_runtime::gpu::{DevicePtr, GpuBackend, KernelHandle};
use spark_runtime::kernel_args::{KernelLaunch, div_ceil};

/// Multi-token conv1d sliding window update + SiLU for prefill.
///
/// Uses read-only token-parallel compute followed by a same-stream state commit.
/// Falls back to the serial kernel for short chunks or missing paired handles.
/// State, input, output and weights must be disjoint allocations/slices.
/// Input/output may be non-contiguous (different strides between tokens).
///
/// Kernel: `causal_conv1d_update_prefill(conv_state, input, weight, bias,
///          output, dim, d_conv, seq_len, input_stride, output_stride)`
/// Grid: (ceil(dim/256), 1, 1)  Block: (256, 1, 1)
#[allow(clippy::too_many_arguments)]
pub fn conv1d_update_prefill(
    gpu: &dyn GpuBackend,
    kernel: KernelHandle,
    conv1d_prefill_tp_k: KernelHandle,
    conv1d_prefill_commit_k: KernelHandle,
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
    if seq_len == 0 || d_inner == 0 {
        return Ok(());
    }
    // Both the serial and TP kernels implement exactly a four-tap window.
    anyhow::ensure!(d_conv == 4, "conv1d prefill requires d_conv=4");
    anyhow::ensure!(
        input_stride >= d_inner && output_stride >= d_inner,
        "conv1d prefill strides must cover all channels"
    );
    let tp = std::env::var("METRALE_CONV1D_TP").ok().as_deref() != Some("0")
        && conv1d_prefill_tp_k.0 != 0
        && conv1d_prefill_commit_k.0 != 0
        && seq_len >= d_conv;
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
        .launch(stream)?;
    if tp {
        // Stream order protects all incoming-state reads, including first-token
        // warps/CTAs that could previously race the final token's write-back.
        KernelLaunch::new(gpu, conv1d_prefill_commit_k)
            .grid([div_ceil(d_inner, 256), 1, 1])
            .block([256, 1, 1])
            .arg_ptr(conv_state)
            .arg_ptr(input)
            .arg_u32(d_inner)
            .arg_u32(d_conv)
            .arg_u32(seq_len)
            .arg_u32(input_stride)
            .launch(stream)?;
    }
    Ok(())
}

#[cfg(test)]
#[path = "conv_prefill_tests.rs"]
mod tests;
