// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-25: CUTLASS NVFP4 weight pack, scale swizzle and transpose wrappers.
//!
//! Owner: gpu-runtime.
//! Invariants: none beyond the types.

use anyhow::{Result, bail};

#[cfg(metrale_cutlass)]
use std::ffi::c_void;

#[cfg(metrale_cutlass)]
use super::*;

/// 2026-09-25: Swizzle an E4M3 weight scale into the CUTLASS SFB layout
/// (`tile_atom_to_shape_SFB`, ue4m3) that the grouped GEMM reads. The layout
/// depends only on `n` and `k`, so it is built once per expert at load
/// (`MoeLayer::build_cutlass_grouped_sfb`). `scale_out` must hold the whole
/// swizzled region.
///
/// `src_n_major` selects the source layout: `false` reads `[K/16,N]`, `true`
/// reads `[N,K/16]`. The output is the same either way.
pub fn pack_weight_sfb(
    scale_in: u64,
    scale_out: u64,
    n: u32,
    k: u32,
    src_n_major: bool,
    stream: u64,
) -> Result<()> {
    #[cfg(metrale_cutlass)]
    {
        let status = unsafe {
            metrale_cutlass_pack_weight_sfb(
                scale_in as *const c_void,
                scale_out as *mut c_void,
                n as i32,
                k as i32,
                i32::from(src_n_major),
                stream as *mut c_void,
            )
        };
        if status != 0 {
            bail!("CUTLASS weight SFB pack failed: status {status} for {n}x{k}");
        }
        Ok(())
    }
    #[cfg(not(metrale_cutlass))]
    {
        let _ = (scale_in, scale_out, n, k, src_n_major, stream);
        bail!("CUTLASS support was not built; set CUTLASS_HOME when building")
    }
}

/// 2026-09-25: Pack a row-major BF16 weight `[N,K]` into CUTLASS NVFP4: packed
/// `[N,K/2]` (K-contiguous) and E4M3 scales `[K/16,N]`, each scale the group's
/// max magnitude / 6. The scales carry no second-level factor, so pass
/// `weight_scale_2 = 1.0` to `nvfp4_gemm_bf16_act_weight_t`.
pub fn pack_bf16_weight_to_nvfp4_t(
    weight_bf16: u64,
    packed_t: u64,
    scale_t: u64,
    n: u32,
    k: u32,
    stream: u64,
) -> Result<()> {
    #[cfg(metrale_cutlass)]
    {
        let status = unsafe {
            metrale_cutlass_pack_bf16_weight_to_nvfp4_t(
                weight_bf16 as *const c_void,
                packed_t as *mut c_void,
                scale_t as *mut c_void,
                n as i32,
                k as i32,
                stream as *mut c_void,
            )
        };
        if status != 0 {
            bail!("CUTLASS BF16->NVFP4 weight pack failed: status {status} for {n}x{k}");
        }
        Ok(())
    }
    #[cfg(not(metrale_cutlass))]
    {
        let _ = (weight_bf16, packed_t, scale_t, n, k, stream);
        bail!("CUTLASS support was not built; set CUTLASS_HOME when building")
    }
}

/// 2026-09-25: Transpose a packed NVFP4 weight from `[K/2, N]` into the CUTLASS
/// `[N, K/2]` layout that `nvfp4_gemm_bf16_act_weight_t` reads. A byte
/// transpose: the two nibbles of each byte stay together. `dst_packed` must hold
/// `N * K/2` bytes.
pub fn transpose_nvfp4_packed_kton(
    src_packed_t: u64,
    dst_packed: u64,
    n: u32,
    k: u32,
    stream: u64,
) -> Result<()> {
    #[cfg(metrale_cutlass)]
    {
        let status = unsafe {
            metrale_cutlass_transpose_nvfp4_packed_kton(
                src_packed_t as *const c_void,
                dst_packed as *mut c_void,
                n as i32,
                k as i32,
                stream as *mut c_void,
            )
        };
        if status != 0 {
            bail!("CUTLASS NVFP4 weight transpose failed: status {status} for {n}x{k}");
        }
        Ok(())
    }
    #[cfg(not(metrale_cutlass))]
    {
        let _ = (src_packed_t, dst_packed, n, k, stream);
        bail!("CUTLASS support was not built; set CUTLASS_HOME when building")
    }
}
