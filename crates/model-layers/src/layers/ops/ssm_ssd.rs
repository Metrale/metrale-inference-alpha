// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-25: Mamba-2 SSD chunked-scan launchers (cumsum, CB bmm, fused scan)
//! and their tiling and shared-memory constants.
//!
//! Owner: model-layers ops.
//! Invariants: none beyond the types.

#![allow(unused_imports)]

use anyhow::Result;
use metrale_gpu_runtime::gpu::{DevicePtr, GpuBackend, KernelHandle};
use metrale_gpu_runtime::kernel_args::{KernelLaunch, div_ceil};

use super::*;

/// 2026-09-25: SSD chunk length. Must equal `SSD_L` in `mamba2_ssd_chunk.cu`.
pub const SSD_L: u32 = 64;
/// 2026-09-25: head_dim rows per SSD scan block. Must equal `SSD_PT` in
/// `mamba2_ssd_chunk.cu`.
pub const SSD_PT: u32 = 64;

/// 2026-09-25: K1: per-chunk dt (softplus and clamp) and the inclusive cumsum
/// of the log-decay.
#[allow(clippy::too_many_arguments)]
pub fn mamba2_ssd_cumsum(
    gpu: &dyn GpuBackend,
    kernel: KernelHandle,
    dt_raw: DevicePtr,
    a_log: DevicePtr,
    dt_bias: DevicePtr,
    dt_out: DevicePtr,
    da_cs: DevicePtr,
    seq_len: u32,
    num_heads: u32,
    nchunks: u32,
    batch_size: u32,
    dt_stride: u32,
    dt_min: f32,
    dt_max: f32,
    stream: u64,
) -> Result<()> {
    KernelLaunch::new(gpu, kernel)
        .grid([nchunks, num_heads, batch_size])
        .block([SSD_L, 1, 1])
        .arg_ptr(dt_raw)
        .arg_ptr(a_log)
        .arg_ptr(dt_bias)
        .arg_ptr(dt_out)
        .arg_ptr(da_cs)
        .arg_u32(seq_len)
        .arg_u32(num_heads)
        .arg_u32(nchunks)
        .arg_u32(dt_stride)
        .arg_f32(dt_min)
        .arg_f32(dt_max)
        .launch(stream)
}

/// 2026-09-25: K2: `CB[c][g][t][s] = C_t . B_s` (raw, FP32).
#[allow(clippy::too_many_arguments)]
pub fn mamba2_ssd_bmm(
    gpu: &dyn GpuBackend,
    kernel: KernelHandle,
    b_proj: DevicePtr,
    c_proj: DevicePtr,
    cb: DevicePtr,
    seq_len: u32,
    nchunks: u32,
    n_groups: u32,
    state_size: u32,
    batch_size: u32,
    bc_stride: u32,
    stream: u64,
) -> Result<()> {
    // 2026-09-25: smem: sC[L][N] + sB[L][N] BF16.
    let smem = 2 * SSD_L * state_size * 2;
    KernelLaunch::new(gpu, kernel)
        .grid([nchunks, n_groups, batch_size])
        .block([128, 1, 1])
        .shared_mem(smem)
        .arg_ptr(b_proj)
        .arg_ptr(c_proj)
        .arg_ptr(cb)
        .arg_u32(seq_len)
        .arg_u32(nchunks)
        .arg_u32(n_groups)
        .arg_u32(state_size)
        .arg_u32(bc_stride)
        .launch(stream)
}

/// 2026-09-25: Largest dynamic shared memory, in bytes, one block may opt into
/// on the GB10 target (sm_121).
pub const MAX_DYNAMIC_SMEM: u32 = 101_376;

/// 2026-09-25: Dynamic shared memory [`mamba2_ssd_scan`] requests for a given
/// SSM state size. It grows linearly in `state_size`, so a large SSM state can
/// exceed [`MAX_DYNAMIC_SMEM`]. The launch calls this function, so the two
/// cannot drift.
pub fn ssd_scan_smem(state_size: u32) -> u32 {
    SSD_PT * (state_size + 1) * 4
        + 2 * SSD_L * state_size * 2
        + 2 * SSD_L * state_size * 2
        + 2 * SSD_L * SSD_PT * 2
        + 2 * SSD_L * 4
        + 2 * SSD_L * 4
}

/// 2026-09-25: Whether the SSD chunked scan fits in shared memory for this
/// `state_size`.
///
/// Callers check this before selecting the SSD path. Over the limit the
/// launch's `cuFuncSetAttribute(MAX_DYNAMIC_SHARED)` fails and the launch
/// returns that error; nothing falls back. `state_size = 96` needs 91,392 B
/// and fits; `state_size = 128` needs 115,968 B and does not.
pub fn ssd_scan_fits(state_size: u32) -> bool {
    ssd_scan_smem(state_size) <= MAX_DYNAMIC_SMEM
}

/// 2026-09-25: K3: chunk_state, state_passing and chunk_scan fused in one
/// kernel. The running state h0 stays in shared memory, so no per-chunk state
/// tensor is written to global memory.
#[allow(clippy::too_many_arguments)]
pub fn mamba2_ssd_scan(
    gpu: &dyn GpuBackend,
    kernel: KernelHandle,
    h_state: DevicePtr,
    x: DevicePtr,
    b_proj: DevicePtr,
    c_proj: DevicePtr,
    d_param: DevicePtr,
    dt_f32: DevicePtr,
    da_cs: DevicePtr,
    cb: DevicePtr,
    output: DevicePtr,
    seq_len: u32,
    num_heads: u32,
    head_dim: u32,
    state_size: u32,
    n_groups: u32,
    nchunks: u32,
    batch_size: u32,
    x_stride: u32,
    bc_stride: u32,
    y_stride: u32,
    stream: u64,
) -> Result<()> {
    // 2026-09-25: sH[PT][N+1] FP32 | double-buffered streaming tiles: sB[2][L][N] |
    // sCM[2][L][N] | sX[2][L][PT] BF16 | sdA[2][L] + sdt[2][L] FP32.
    let smem = ssd_scan_smem(state_size);
    KernelLaunch::new(gpu, kernel)
        .grid([num_heads, head_dim / SSD_PT, batch_size])
        .block([512, 1, 1]) // 2026-09-25: 16 warps, each owning two n-tiles (see kernel).
        .shared_mem(smem)
        .arg_ptr(h_state)
        .arg_ptr(x)
        .arg_ptr(b_proj)
        .arg_ptr(c_proj)
        .arg_ptr(d_param)
        .arg_ptr(dt_f32)
        .arg_ptr(da_cs)
        .arg_ptr(cb)
        .arg_ptr(output)
        .arg_u32(seq_len)
        .arg_u32(num_heads)
        .arg_u32(head_dim)
        .arg_u32(state_size)
        .arg_u32(n_groups)
        .arg_u32(nchunks)
        .arg_u32(x_stride)
        .arg_u32(bc_stride)
        .arg_u32(y_stride)
        .launch(stream)
}
