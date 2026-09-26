// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-25: The GEMM formulation of the low-rank hyper-connection collapse, on either `dense_gemm_bf16_pipelined` or cuBLASLt.
//!
//! Owner: model-layers ops.
//! Invariants: none beyond the types. `hyper_connection_lowrank.rs` passes `use_cublas = true`
//! for `T <= 64` and `false` for larger `T`.

use anyhow::Result;
use metrale_gpu_runtime::gpu::{DevicePtr, GpuBackend, KernelHandle};

use crate::layers::qwen3_attention::HcLowRank;
use metrale_gpu_runtime::kernel_args::KernelLaunch;

/// 2026-09-25: Stage `normed` in BF16, run the down, up and (when `inject`) injection projections
/// as GEMMs, and finish with the kernels `hc_silu_scale` and `hc_pre_mix`. Tokens
/// are processed in slabs of at most 2048 so the scratch region stays bounded.
#[allow(clippy::too_many_arguments)]
pub(crate) fn hc_pre_gemm(
    gpu: &dyn GpuBackend,
    streams: DevicePtr,
    w: &HcLowRank,
    y_out: DevicePtr,
    inj_out: DevicePtr,
    scratch: DevicePtr,
    num_tokens: u32,
    hidden_size: u32,
    hc_mult: u32,
    norm_eps: f32,
    inject: bool,
    use_cublas: bool,
    stream: u64,
) -> Result<()> {
    const SLAB: u32 = 2048;
    let hc_dim = (hc_mult * hidden_size) as usize;
    let rank = w.rank as u32;
    // 2026-09-25: Scratch layout (BF16): normed [L, hc_dim], up_pre [L, hc_dim], low [L, rank],
    // inj_pre [L, hc], with L = min(T, 2048). gpu-runtime `buffers/sizes.rs` reserves this layout
    // at `m.min(2048)` rows, so it fits whenever T <= m.
    let lay = num_tokens.min(SLAB) as usize;
    let normed = scratch;
    let up_pre = scratch.offset(lay * hc_dim * 2);
    let low = scratch.offset(2 * lay * hc_dim * 2);
    let inj_pre = scratch.offset(2 * lay * hc_dim * 2 + lay * w.rank * 2);

    let k_stage = gpu.kernel("hyper_connection", "hc_pre_stage_bf16")?;
    let k_silu = gpu.kernel("hyper_connection", "hc_silu_scale")?;
    let k_mix = gpu.kernel("hyper_connection", "hc_pre_mix")?;
    let k_gemm = gpu.kernel("gemm", "dense_gemm_bf16_pipelined")?;
    let inv_hc = 1.0f32 / hc_mult as f32;

    let mut t0 = 0u32;
    while t0 < num_tokens {
        let ts = SLAB.min(num_tokens - t0);
        let streams_s = streams.offset(t0 as usize * hc_dim * 4);

        KernelLaunch::new(gpu, k_stage)
            .grid([ts, 1, 1])
            .block([1024, 1, 1])
            .arg_ptr(streams_s)
            .arg_ptr(w.norm_w)
            .arg_ptr(normed)
            .arg_u32(hidden_size)
            .arg_u32(hc_mult)
            .arg_f32(norm_eps)
            .launch(stream)?;

        // 2026-09-25: low = normed x down_w^T   [ts, rank]
        if use_cublas {
            crate::layers::ops::cublas_bf16_proj_dense(
                normed,
                w.down_w,
                low,
                ts,
                rank,
                hc_dim as u32,
                stream,
            )?;
        } else {
            gemm_raw(
                gpu,
                k_gemm,
                normed,
                w.down_w,
                low,
                ts,
                rank,
                hc_dim as u32,
                stream,
            )?;
        }
        let n_low = ts * rank;
        KernelLaunch::new(gpu, k_silu)
            .grid([n_low.div_ceil(256), 1, 1])
            .block([256, 1, 1])
            .arg_ptr(low)
            .arg_u32(n_low)
            .arg_f32(inv_hc)
            .launch(stream)?;

        // 2026-09-25: up_pre = low x up_w^T   [ts, hc_dim]
        if use_cublas {
            crate::layers::ops::cublas_bf16_proj_dense(
                low,
                w.up_w,
                up_pre,
                ts,
                hc_dim as u32,
                rank,
                stream,
            )?;
        } else {
            gemm_raw(
                gpu,
                k_gemm,
                low,
                w.up_w,
                up_pre,
                ts,
                hc_dim as u32,
                rank,
                stream,
            )?;
        }
        if inject {
            // 2026-09-25: inj_pre = normed x inject_w^T   [ts, hc]
            if use_cublas {
                crate::layers::ops::cublas_bf16_proj_dense(
                    normed,
                    w.inject_w,
                    inj_pre,
                    ts,
                    hc_mult,
                    hc_dim as u32,
                    stream,
                )?;
            } else {
                gemm_raw(
                    gpu,
                    k_gemm,
                    normed,
                    w.inject_w,
                    inj_pre,
                    ts,
                    hc_mult,
                    hc_dim as u32,
                    stream,
                )?;
            }
        }

        KernelLaunch::new(gpu, k_mix)
            .grid([ts, 1, 1])
            .block([256, 1, 1])
            .arg_ptr(normed)
            .arg_ptr(up_pre)
            .arg_ptr(if inject { inj_pre } else { DevicePtr::NULL })
            .arg_ptr(y_out.offset(t0 as usize * hidden_size as usize * 2))
            .arg_ptr(inj_out.offset(t0 as usize * hc_mult as usize * 4))
            .arg_u32(hidden_size)
            .arg_u32(hc_mult)
            .arg_f32(inv_hc)
            .launch(stream)?;

        t0 += ts;
    }
    Ok(())
}

fn gemm_raw(
    gpu: &dyn GpuBackend,
    kernel: KernelHandle,
    a: DevicePtr,
    w: DevicePtr,
    out: DevicePtr,
    m: u32,
    n: u32,
    k: u32,
    stream: u64,
) -> Result<()> {
    KernelLaunch::new(gpu, kernel)
        .grid([n.div_ceil(128), m.div_ceil(128), 1])
        .block([256, 1, 1])
        .arg_ptr(a)
        .arg_ptr(w)
        .arg_ptr(out)
        .arg_u32(m)
        .arg_u32(n)
        .arg_u32(k)
        .launch(stream)
}
