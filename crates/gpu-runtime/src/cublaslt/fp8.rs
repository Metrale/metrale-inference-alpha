// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-25: FP8 E4M3 cuBLASLt GEMMs with BF16 output: row-wise scaled, and
//! 128-block scaled with a contiguous or pitched output.
//!
//! Owner: gpu-runtime (cuBLASLt wrapper).
//! Invariants: none beyond the types.

use anyhow::{Result, bail};
use std::ffi::c_void;

use super::*;

/// 2026-09-25: FP8 E4M3 `out[M,N] = act[M,K] @ weight[N,K]ᵀ`, BF16 output,
/// with FP32 `OUTER_VEC` scales on both operands: `weight_scale[N]` per
/// weight row (cuBLASLt's A) and `act_scale[M]` per token (B).
#[allow(clippy::too_many_arguments)]
pub fn fp8_gemm_act_weight_t_rowwise(
    act_fp8: u64,
    act_scale: u64,
    weight_fp8: u64,
    weight_scale: u64,
    out: u64,
    m: u32,
    n: u32,
    k: u32,
    stream: u64,
) -> Result<()> {
    let ctx = ctx()?;
    unsafe {
        let mut desc: cublasLtMatmulDesc_t = std::ptr::null_mut();
        chk(
            cublasLtMatmulDescCreate(&mut desc, CUBLAS_COMPUTE_32F, CUDA_R_32F),
            "DescCreate",
        )?;
        let ta = CUBLAS_OP_T;
        let tb = CUBLAS_OP_N;
        let set = |attr: u32, val: *const c_void, sz: usize, what: &str| -> Result<()> {
            chk(cublasLtMatmulDescSetAttribute(desc, attr, val, sz), what)
        };
        set(DESC_TRANSA, &ta as *const i32 as *const c_void, 4, "TRANSA")?;
        set(DESC_TRANSB, &tb as *const i32 as *const c_void, 4, "TRANSB")?;
        let mode = SCALE_MODE_OUTER_VEC_32F;
        set(
            DESC_A_SCALE_MODE,
            &mode as *const i32 as *const c_void,
            4,
            "A_SCALE_MODE",
        )?;
        set(
            DESC_B_SCALE_MODE,
            &mode as *const i32 as *const c_void,
            4,
            "B_SCALE_MODE",
        )?;
        set(
            DESC_A_SCALE_POINTER,
            &weight_scale as *const u64 as *const c_void,
            8,
            "A_SCALE_POINTER",
        )?;
        set(
            DESC_B_SCALE_POINTER,
            &act_scale as *const u64 as *const c_void,
            8,
            "B_SCALE_POINTER",
        )?;

        let mut la: cublasLtMatrixLayout_t = std::ptr::null_mut();
        let mut lb: cublasLtMatrixLayout_t = std::ptr::null_mut();
        let mut ld_: cublasLtMatrixLayout_t = std::ptr::null_mut();
        chk(
            cublasLtMatrixLayoutCreate(&mut la, CUDA_R_8F_E4M3, k as u64, n as u64, k as i64),
            "LayoutA",
        )?;
        chk(
            cublasLtMatrixLayoutCreate(&mut lb, CUDA_R_8F_E4M3, k as u64, m as u64, k as i64),
            "LayoutB",
        )?;
        chk(
            cublasLtMatrixLayoutCreate(&mut ld_, CUDA_R_16BF, n as u64, m as u64, n as i64),
            "LayoutD",
        )?;
        let mut pref: cublasLtMatmulPreference_t = std::ptr::null_mut();
        chk(cublasLtMatmulPreferenceCreate(&mut pref), "PrefCreate")?;
        let ws_size = ctx.ws_size;
        chk(
            cublasLtMatmulPreferenceSetAttribute(
                pref,
                PREF_MAX_WORKSPACE_BYTES,
                &ws_size as *const usize as *const c_void,
                std::mem::size_of::<usize>(),
            ),
            "PrefWorkspace",
        )?;
        let mut result = [0u8; 128];
        let mut returned: i32 = 0;
        chk(
            cublasLtMatmulAlgoGetHeuristic(
                ctx.handle,
                desc,
                la,
                lb,
                ld_,
                ld_,
                pref,
                1,
                result.as_mut_ptr() as *mut c_void,
                &mut returned,
            ),
            "AlgoGetHeuristic",
        )?;
        if returned < 1 {
            bail!("cuBLASLt fp8 rowwise: no algorithm for {m}x{n}x{k}");
        }
        let alpha: f32 = 1.0;
        let beta: f32 = 0.0;
        let status = cublasLtMatmul(
            ctx.handle,
            desc,
            &alpha as *const f32 as *const c_void,
            weight_fp8 as *const c_void,
            la,
            act_fp8 as *const c_void,
            lb,
            &beta as *const f32 as *const c_void,
            out as *const c_void,
            ld_,
            out as *mut c_void,
            ld_,
            result.as_ptr() as *const c_void,
            ctx.workspace as *mut c_void,
            ctx.ws_size,
            stream as *mut c_void,
        );
        cublasLtMatmulPreferenceDestroy(pref);
        cublasLtMatrixLayoutDestroy(la);
        cublasLtMatrixLayoutDestroy(lb);
        cublasLtMatrixLayoutDestroy(ld_);
        cublasLtMatmulDescDestroy(desc);
        chk(status, "Matmul")?;
    }
    Ok(())
}

/// 2026-09-25: FP8 E4M3 `out[M,N] = act[M,K] @ weight[N,K]ᵀ`, BF16 output,
/// with FP32 block scales: `BLK128x128_32F` on the weight (cuBLASLt's A) and
/// `VEC128_32F` on the activation (B). The scale layouts are in
/// [`super::scale_layout`]:
///
/// * `weight_block_scale` is the checkpoint's row-major `[N/128, K/128]`
///   grid, passed unchanged, which requires `blk128x128_stride_ok(k)`; this
///   function does not check it.
/// * `act_scale` is expected as `[K/128, m]`, token index contiguous
///   (`vec128_b_index`): the transpose of the `[M, K/128]` that
///   `per_token_group_quant_fp8` writes. The `fp8_act_scale_to_kmajor`
///   kernel converts one to the other.
///
/// `m` goes to cuBLASLt unchanged; the callers pass a padded count
/// (`m_pad`), and `act_fp8` and `act_scale` must cover it. The output is
/// contiguous `[m, N]`;
/// [`fp8_gemm_act_weight_t_blkscaled_ldc`] takes an output row pitch.
#[allow(clippy::too_many_arguments)]
pub fn fp8_gemm_act_weight_t_blkscaled(
    act_fp8: u64,
    act_scale: u64,
    weight_fp8: u64,
    weight_block_scale: u64,
    out: u64,
    m: u32,
    n: u32,
    k: u32,
    stream: u64,
) -> Result<()> {
    fp8_gemm_act_weight_t_blkscaled_ldc(
        act_fp8,
        act_scale,
        weight_fp8,
        weight_block_scale,
        out,
        m,
        n,
        k,
        n,
        stream,
    )
}

/// 2026-09-25: [`fp8_gemm_act_weight_t_blkscaled`] with output rows `ldc`
/// BF16 elements apart: D is column-major `[N, m]` with leading dimension
/// `ldc`, i.e. row-major `[m, N]` with row pitch `ldc`. The multi-seq decode
/// Q/K/V projections use it to write each row into its sequence's slot of
/// the QKV buffer (`w8a8_decode.rs` `qkv_decode_w8a8_plans`, `ldc` =
/// `per_seq_qkv` in BF16 elements).
///
/// All `m` rows are written, padding rows included, so the output extent is
/// `(m - 1) * ldc + n` elements; callers bound it against their buffer with
/// `metrale_model_layers::layers::ops::strided_out_extent_elems`.
///
/// Fails before calling cuBLASLt when `ldc < n`.
#[allow(clippy::too_many_arguments)]
pub fn fp8_gemm_act_weight_t_blkscaled_ldc(
    act_fp8: u64,
    act_scale: u64,
    weight_fp8: u64,
    weight_block_scale: u64,
    out: u64,
    m: u32,
    n: u32,
    k: u32,
    ldc: u32,
    stream: u64,
) -> Result<()> {
    if ldc < n {
        bail!("cuBLASLt fp8: output row pitch ldc={ldc} is shorter than N={n}");
    }
    let ctx = ctx()?;
    unsafe {
        let mut desc: cublasLtMatmulDesc_t = std::ptr::null_mut();
        chk(
            cublasLtMatmulDescCreate(&mut desc, CUBLAS_COMPUTE_32F, CUDA_R_32F),
            "DescCreate",
        )?;
        let ta = CUBLAS_OP_T;
        let tb = CUBLAS_OP_N;
        let set = |attr: u32, val: *const c_void, sz: usize, what: &str| -> Result<()> {
            chk(cublasLtMatmulDescSetAttribute(desc, attr, val, sz), what)
        };
        set(DESC_TRANSA, &ta as *const i32 as *const c_void, 4, "TRANSA")?;
        set(DESC_TRANSB, &tb as *const i32 as *const c_void, 4, "TRANSB")?;
        // 2026-09-25: Weight: one scale per 128x128 block. Activation: one
        // scale per token per 128 elements of K.
        let a_mode = SCALE_MODE_BLK128X128_32F;
        let b_mode = SCALE_MODE_VEC128_32F;
        set(
            DESC_A_SCALE_MODE,
            &a_mode as *const i32 as *const c_void,
            4,
            "A_SCALE_MODE",
        )?;
        set(
            DESC_B_SCALE_MODE,
            &b_mode as *const i32 as *const c_void,
            4,
            "B_SCALE_MODE",
        )?;
        set(
            DESC_A_SCALE_POINTER,
            &weight_block_scale as *const u64 as *const c_void,
            8,
            "A_SCALE_POINTER",
        )?;
        set(
            DESC_B_SCALE_POINTER,
            &act_scale as *const u64 as *const c_void,
            8,
            "B_SCALE_POINTER",
        )?;

        let mut la: cublasLtMatrixLayout_t = std::ptr::null_mut();
        let mut lb: cublasLtMatrixLayout_t = std::ptr::null_mut();
        let mut ld_: cublasLtMatrixLayout_t = std::ptr::null_mut();
        chk(
            cublasLtMatrixLayoutCreate(&mut la, CUDA_R_8F_E4M3, k as u64, n as u64, k as i64),
            "LayoutA",
        )?;
        chk(
            cublasLtMatrixLayoutCreate(&mut lb, CUDA_R_8F_E4M3, k as u64, m as u64, k as i64),
            "LayoutB",
        )?;
        chk(
            cublasLtMatrixLayoutCreate(&mut ld_, CUDA_R_16BF, n as u64, m as u64, ldc as i64),
            "LayoutD",
        )?;
        let mut pref: cublasLtMatmulPreference_t = std::ptr::null_mut();
        chk(cublasLtMatmulPreferenceCreate(&mut pref), "PrefCreate")?;
        let ws_size = ctx.ws_size;
        chk(
            cublasLtMatmulPreferenceSetAttribute(
                pref,
                PREF_MAX_WORKSPACE_BYTES,
                &ws_size as *const usize as *const c_void,
                std::mem::size_of::<usize>(),
            ),
            "PrefWorkspace",
        )?;
        let mut result = [0u8; 128];
        let mut returned: i32 = 0;
        chk(
            cublasLtMatmulAlgoGetHeuristic(
                ctx.handle,
                desc,
                la,
                lb,
                ld_,
                ld_,
                pref,
                1,
                result.as_mut_ptr() as *mut c_void,
                &mut returned,
            ),
            "AlgoGetHeuristic",
        )?;
        if returned < 1 {
            bail!("cuBLASLt fp8: no algorithm for {m}x{n}x{k}");
        }
        let alpha: f32 = 1.0;
        let beta: f32 = 0.0;
        let status = cublasLtMatmul(
            ctx.handle,
            desc,
            &alpha as *const f32 as *const c_void,
            weight_fp8 as *const c_void,
            la,
            act_fp8 as *const c_void,
            lb,
            &beta as *const f32 as *const c_void,
            out as *const c_void,
            ld_,
            out as *mut c_void,
            ld_,
            result.as_ptr() as *const c_void,
            ctx.workspace as *mut c_void,
            ctx.ws_size,
            stream as *mut c_void,
        );
        cublasLtMatmulPreferenceDestroy(pref);
        cublasLtMatrixLayoutDestroy(la);
        cublasLtMatrixLayoutDestroy(lb);
        cublasLtMatrixLayoutDestroy(ld_);
        cublasLtMatmulDescDestroy(desc);
        chk(status, "Matmul")?;
    }
    Ok(())
}
