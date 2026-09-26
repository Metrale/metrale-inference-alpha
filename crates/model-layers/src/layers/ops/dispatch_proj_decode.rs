// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-25: W8A8 block-scaled cuBLASLt route for the 5..=16-row decode
//! projections (GDN `in_proj_qkvz` and `out_proj`, attention Q/K/V and O), and
//! the strided-output extent those need.
//!
//! On this route a projection runs W8A8 instead of the W8A16 GEMV: an E4M3
//! activation with one FP32 scale per token and 128 of K, against the
//! checkpoint's E4M3 weight and its 128x128 FP32 block scales.
//!
//! Owner: model-layers ops.
//! Invariants:
//! - [`decode_w8a8_selected`] reads the environment only through values cached
//!   once per process ([`w8a8_decode_proj_disabled`],
//!   `cublas_scale_layout_kmajor`), so for the same padded rows, shape, weight
//!   format and buffers it gives the same answer on every step, CUDA-graph
//!   capture included.

use anyhow::Result;
use metrale_gpu_runtime::gpu::{DevicePtr, GpuBackend, KernelHandle};

use super::{
    Fp8ActQuant, cublas_fp8_m_pad, cublas_scale_layout_kmajor, fp8_act_scale_to_kmajor,
    per_token_group_quant_fp8,
};

/// 2026-09-25: Kill switch: `METRALE_NO_W8A8_DECODE_PROJ` set to any value,
/// empty and `0` included, turns this route off. Read once, so it is constant
/// for the process and safe to branch on under CUDA-graph capture.
pub fn w8a8_decode_proj_disabled() -> bool {
    static OFF: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *OFF.get_or_init(|| std::env::var_os("METRALE_NO_W8A8_DECODE_PROJ").is_some())
}

/// 2026-09-25: The padded decode row counts this route takes. Any other count
/// stays on the caller's own kernels.
pub const DECODE_W8A8_ROWS: std::ops::RangeInclusive<usize> = 5..=16;

/// 2026-09-25: Elements from the start of the output to one past the last
/// element a cuBLASLt GEMM writes when its `m_pad` output rows of `n` elements
/// are `ldc` elements apart: `(m_pad - 1) * ldc + n`. Pass the padded row
/// count, because the pad rows are written too. The strided call sites bound
/// their buffers with it.
pub fn strided_out_extent_elems(m_pad: u32, ldc: u32, n: u32) -> usize {
    debug_assert!(m_pad >= 1);
    (m_pad as usize - 1) * ldc as usize + n as usize
}

/// 2026-09-25: The shape and output of one decode projection, as
/// [`decode_w8a8_selected`] judges it. The route is taken only when all of
/// these hold:
///
/// * `family_armed`: the caller's [`super::CublasScope`] family (`ssm` or
///   `attn`) is on.
/// * `disabled` is false ([`w8a8_decode_proj_disabled`]).
/// * `rows` is in [`DECODE_W8A8_ROWS`].
/// * The weight is `Fp8BlockScaled`, since cuBLASLt is given its scales as a
///   `[N/128, K/128]` grid.
/// * `n` and `k` are multiples of 128, and `blk128x128_stride_ok(k)` holds
///   (`k` a multiple of 512), the weight-scale stride rule that
///   `metrale_gpu_runtime::cublaslt::scale_layout` quotes.
/// * `ldc >= n`.
/// * The output holds the padded write extent ([`strided_out_extent_elems`]).
/// * The quantizer resolved, and the scratch holds `m_pad` rows in the active
///   scale layout, adapter kernel and buffer included when it is K-major.
#[derive(Clone, Copy, Debug)]
pub struct DecodeW8a8Plan {
    /// 2026-09-25: Padded decode rows (the forward context's `n`).
    pub rows: usize,
    /// 2026-09-25: Output width of this projection.
    pub n: u32,
    /// 2026-09-25: Contract width of this projection.
    pub k: u32,
    /// 2026-09-25: Output row pitch in BF16 elements (`n` for a contiguous
    /// output).
    pub ldc: u32,
    /// 2026-09-25: Allocated size of the output buffer, in bytes.
    pub out_capacity_bytes: usize,
}

impl DecodeW8a8Plan {
    /// 2026-09-25: A contiguous `[rows, n]` output (GDN `in_proj_qkvz` and
    /// `out_proj`, attention `o_proj`).
    pub fn contiguous(rows: usize, n: u32, k: u32, out_capacity_bytes: usize) -> Self {
        Self {
            rows,
            n,
            k,
            ldc: n,
            out_capacity_bytes,
        }
    }

    /// 2026-09-25: A strided output: rows `ldc` BF16 elements apart (attention
    /// Q/K/V into the `[n, per_seq_qkv]` multi-sequence QKV buffer).
    pub fn strided(rows: usize, n: u32, k: u32, ldc: u32, out_capacity_bytes: usize) -> Self {
        Self {
            rows,
            n,
            k,
            ldc,
            out_capacity_bytes,
        }
    }

    /// 2026-09-25: The padded M cuBLASLt is handed.
    pub fn m_pad(&self) -> u32 {
        cublas_fp8_m_pad(self.rows as u32)
    }

    /// 2026-09-25: Bytes of `out` this projection may write, pad rows included.
    pub fn write_extent_bytes(&self) -> usize {
        strided_out_extent_elems(self.m_pad(), self.ldc, self.n) * 2
    }
}

/// 2026-09-25: The activation-quant scratch buffers with their capacities, and
/// the quantizer and scale-layout adapter kernels.
#[derive(Clone, Copy, Debug)]
pub struct DecodeW8a8Scratch {
    pub act_fp8: DevicePtr,
    pub act_fp8_bytes: usize,
    pub act_scale: DevicePtr,
    pub act_scale_bytes: usize,
    pub act_scale_kmajor: DevicePtr,
    pub act_scale_kmajor_bytes: usize,
    pub quant_k: Fp8ActQuant,
    pub scale_kmajor_k: KernelHandle,
}

impl DecodeW8a8Scratch {
    /// 2026-09-25: Whether the quantizer resolved and the scratch holds `m_pad`
    /// rows of a K-wide activation in the active scale layout.
    fn fits(&self, m_pad: u32, k: u32) -> bool {
        let rows = m_pad as usize;
        let kg = k as usize / 128;
        self.act_fp8.0 != 0
            && self.act_scale.0 != 0
            && self.quant_k.available()
            && self.act_fp8_bytes >= rows * k as usize
            && self.act_scale_bytes >= rows * kg * 4
            && (!cublas_scale_layout_kmajor()
                || (self.scale_kmajor_k.0 != 0
                    && self.act_scale_kmajor.0 != 0
                    && self.act_scale_kmajor_bytes >= rows * kg * 4))
    }
}

/// 2026-09-25: Whether one decode projection takes the W8A8 cuBLASLt route;
/// [`DecodeW8a8Plan`] lists the conditions.
pub fn decode_w8a8_selected(
    family_armed: bool,
    disabled: bool,
    plan: &DecodeW8a8Plan,
    scale_format: crate::weight_map::WeightQuantFormat,
    scratch: &DecodeW8a8Scratch,
) -> bool {
    let m_pad = plan.m_pad();
    family_armed
        && !disabled
        && DECODE_W8A8_ROWS.contains(&plan.rows)
        && scale_format == crate::weight_map::WeightQuantFormat::Fp8BlockScaled
        && plan.n.is_multiple_of(128)
        && plan.k.is_multiple_of(128)
        && metrale_gpu_runtime::cublaslt::scale_layout::blk128x128_stride_ok(plan.k as usize)
        && plan.ldc >= plan.n
        && plan.write_extent_bytes() <= plan.out_capacity_bytes
        && scratch.fits(m_pad, plan.k)
}

/// 2026-09-25: Quantize BF16 `act[rows, k]` into the scratch: FP8 E4M3 bytes
/// and one FP32 scale per token and 128 of K, with the pad rows
/// `rows..ceil16(rows)` zeroed, and the scales transposed for cuBLASLt when
/// the layout is K-major.
///
/// Separate from the GEMM so projections over one activation share one
/// quantization; the attention Q/K/V do.
///
/// The pad rows' FP8 bytes are zeroed as well as their scales: a zero scale
/// alone is not enough, because `NaN * 0.0` is `NaN`.
pub fn decode_w8a8_quant_act(
    gpu: &dyn GpuBackend,
    scratch: &DecodeW8a8Scratch,
    act_bf16: DevicePtr,
    rows: u32,
    k: u32,
    stream: u64,
) -> Result<()> {
    per_token_group_quant_fp8(
        gpu,
        scratch.quant_k,
        act_bf16,
        scratch.act_fp8,
        scratch.act_scale,
        rows,
        k,
        stream,
    )?;
    let m_pad = cublas_fp8_m_pad(rows);
    if m_pad > rows {
        gpu.memset_async(
            scratch.act_fp8.offset(rows as usize * k as usize),
            0,
            (m_pad - rows) as usize * k as usize,
            stream,
        )?;
    }
    if cublas_scale_layout_kmajor() {
        // 2026-09-25: Writes every `[K/128, m_pad]` slot, pad rows included.
        fp8_act_scale_to_kmajor(
            gpu,
            scratch.scale_kmajor_k,
            scratch.act_scale,
            scratch.act_scale_kmajor,
            rows,
            m_pad,
            k,
            stream,
        )?;
    } else {
        // 2026-09-25: In the `rowmajor` layout the pad rows are a contiguous
        // tail, so they are zeroed in place.
        let kg = k as usize / 128;
        if m_pad > rows {
            gpu.memset_async(
                scratch.act_scale.offset(rows as usize * kg * 4),
                0,
                (m_pad - rows) as usize * kg * 4,
                stream,
            )?;
        }
    }
    Ok(())
}

/// 2026-09-25: `out[m_pad, n] = act_fp8[m_pad, k] @ weight[n, k]ᵀ` at row pitch
/// `plan.ldc`, with both sets of scales applied. The activation must already
/// be through [`decode_w8a8_quant_act`].
pub fn decode_w8a8_gemm(
    scratch: &DecodeW8a8Scratch,
    fp8w: &crate::weight_map::Fp8Weight,
    out: DevicePtr,
    plan: &DecodeW8a8Plan,
    stream: u64,
) -> Result<()> {
    let b_scale = if cublas_scale_layout_kmajor() {
        scratch.act_scale_kmajor
    } else {
        scratch.act_scale
    };
    metrale_gpu_runtime::cublaslt::fp8_gemm_act_weight_t_blkscaled_ldc(
        scratch.act_fp8.0,
        b_scale.0,
        fp8w.weight.0,
        fp8w.row_scale.0,
        out.0,
        plan.m_pad(),
        plan.n,
        plan.k,
        plan.ldc,
        stream,
    )
}

#[cfg(test)]
#[path = "dispatch_proj_decode_tests.rs"]
mod tests;
