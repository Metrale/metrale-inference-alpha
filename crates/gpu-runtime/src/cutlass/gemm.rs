// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-25: Dense CUTLASS GEMM wrappers: BF16, and NVFP4 with BF16 output.
//!
//! Owner: gpu-runtime.
//! Invariants: none beyond the types.

use anyhow::{Result, bail};

#[cfg(metrale_cutlass)]
use std::ffi::c_void;

#[cfg(metrale_cutlass)]
use super::*;

/// 2026-09-25: Row-major `out[M,N] = act[M,K] @ weight[N,K]^T`, all BF16.
#[allow(clippy::too_many_arguments)]
pub fn bf16_gemm_act_weight_t(
    act: u64,
    weight: u64,
    out: u64,
    m: u32,
    n: u32,
    k: u32,
    stream: u64,
) -> Result<()> {
    #[cfg(metrale_cutlass)]
    {
        let ctx = ctx()?;
        let status = unsafe {
            metrale_cutlass_bf16_gemm_act_weight_t(
                act as *const c_void,
                weight as *const c_void,
                out as *mut c_void,
                m as i32,
                n as i32,
                k as i32,
                ctx.workspace as *mut c_void,
                ctx.ws_size,
                stream as *mut c_void,
            )
        };
        if status != 0 {
            bail!("CUTLASS bf16 GEMM failed: status {status} for {m}x{n}x{k}");
        }
        Ok(())
    }
    #[cfg(not(metrale_cutlass))]
    {
        let _ = (act, weight, out, m, n, k, stream);
        bail!("CUTLASS support was not built; set CUTLASS_HOME when building")
    }
}

/// 2026-09-25: NVFP4 dense projection with BF16 output:
/// `out[M,N] = quant_nvfp4(act[M,K]) @ weight[N,K]^T`, scaled by `weight_scale_2`.
///
/// `weight_packed_t` is the CUTLASS `[N,K/2]` K-contiguous byte layout that
/// `pack_bf16_weight_to_nvfp4_t` writes and `transpose_nvfp4_packed_kton`
/// produces from `[K/2,N]`; `weight_scale_t` is E4M3 `[K/16,N]`. The wrapper
/// quantizes the activation and swizzles the weight scales into the shared
/// workspace before the GEMM.
#[allow(clippy::too_many_arguments)]
pub fn nvfp4_gemm_bf16_act_weight_t(
    act: u64,
    weight_packed_t: u64,
    weight_scale_t: u64,
    weight_scale_2: f32,
    out: u64,
    m: u32,
    n: u32,
    k: u32,
    stream: u64,
) -> Result<()> {
    #[cfg(metrale_cutlass)]
    {
        let ctx = ctx()?;
        let status = unsafe {
            metrale_cutlass_nvfp4_gemm_bf16_act_weight_t(
                act as *const c_void,
                weight_packed_t as *const c_void,
                weight_scale_t as *const c_void,
                weight_scale_2,
                out as *mut c_void,
                m as i32,
                n as i32,
                k as i32,
                ctx.workspace as *mut c_void,
                ctx.ws_size,
                stream as *mut c_void,
            )
        };
        if status != 0 {
            bail!("CUTLASS nvfp4 GEMM failed: status {status} for {m}x{n}x{k}");
        }
        Ok(())
    }
    #[cfg(not(metrale_cutlass))]
    {
        let _ = (
            act,
            weight_packed_t,
            weight_scale_t,
            weight_scale_2,
            out,
            m,
            n,
            k,
            stream,
        );
        bail!("CUTLASS support was not built; set CUTLASS_HOME when building")
    }
}
