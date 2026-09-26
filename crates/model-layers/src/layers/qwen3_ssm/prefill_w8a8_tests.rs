// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-25: Tests of `qkvz_cublas_selected`, one clause each: every case
//! starts from inputs that select the cuBLASLt arm and changes one of them.
//!
//! Owner: model-layers (qwen3_ssm tests).
//! Invariants: none beyond the types.

use super::qkvz_cublas_selected;
use crate::layers::ops::cublas_fp8_m_pad;
use crate::weight_map::WeightQuantFormat;
use metrale_gpu_runtime::gpu::{DevicePtr, KernelHandle};

const H: u32 = 5120;
const QKVZ: u32 = 16384;
const M: u32 = 1193;
const KMAJOR_K: KernelHandle = KernelHandle(0xC0DE);
const KMAJOR_BUF: DevicePtr = DevicePtr(0x1000);

fn m_pad() -> u32 {
    cublas_fp8_m_pad(M)
}

fn out_bytes() -> usize {
    m_pad() as usize * QKVZ as usize * 2
}

fn scale_bytes() -> usize {
    m_pad() as usize * (H as usize / 128) * 4
}

#[allow(clippy::too_many_arguments)]
fn selected(
    cublas_ssm: bool,
    fmt: WeightQuantFormat,
    n: u32,
    k: u32,
    out_capacity: usize,
    kmajor_k: KernelHandle,
    kmajor_buf: DevicePtr,
    kmajor_capacity: usize,
) -> bool {
    qkvz_cublas_selected(
        cublas_ssm,
        fmt,
        m_pad(),
        n,
        k,
        out_capacity,
        kmajor_k,
        kmajor_buf,
        kmajor_capacity,
    )
}

fn ready() -> bool {
    selected(
        true,
        WeightQuantFormat::Fp8BlockScaled,
        QKVZ,
        H,
        out_bytes(),
        KMAJOR_K,
        KMAJOR_BUF,
        scale_bytes(),
    )
}

#[test]
fn selected_at_the_real_qkvz_shape_when_the_ssm_family_is_armed() {
    assert!(ready(), "the H100 QKVZ shape must take the cuBLASLt arm");
}

/// 2026-09-25: The arm declines when `METRALE_CUBLAS_GEMM` does not name `ssm`.
#[test]
fn not_selected_when_the_ssm_family_is_not_armed() {
    assert!(!selected(
        false,
        WeightQuantFormat::Fp8BlockScaled,
        QKVZ,
        H,
        out_bytes(),
        KMAJOR_K,
        KMAJOR_BUF,
        scale_bytes(),
    ));
}

#[test]
fn not_selected_for_non_block_scaled_weights() {
    // 2026-09-25: cuBLASLt reads the weight scales as a BLK128x128
    // `[N/128, K/128]` grid; a per-row or single scale read that way is wrong.
    for fmt in [
        WeightQuantFormat::Fp8PerRow,
        WeightQuantFormat::Fp8SingleScale,
    ] {
        assert!(
            !selected(
                true,
                fmt,
                QKVZ,
                H,
                out_bytes(),
                KMAJOR_K,
                KMAJOR_BUF,
                scale_bytes()
            ),
            "{fmt:?} must not take the block-scaled cuBLASLt arm"
        );
    }
}

#[test]
fn not_selected_for_unaligned_shapes() {
    let f = WeightQuantFormat::Fp8BlockScaled;
    let case = |n, k| {
        selected(
            true,
            f,
            n,
            k,
            out_bytes(),
            KMAJOR_K,
            KMAJOR_BUF,
            scale_bytes(),
        )
    };
    // 2026-09-25: N not a multiple of 128: the weight scale grid is
    // `[N/128, K/128]`.
    assert!(!case(QKVZ + 1, H));
    // 2026-09-25: K not a multiple of 128: the activation quantizer writes one
    // FP32 scale per 128-wide K group.
    assert!(!case(QKVZ, H + 1));
    // 2026-09-25: K/128 not a multiple of 4 fails `blk128x128_stride_ok`.
    assert!(!case(QKVZ, 128 * 3));
    assert!(case(QKVZ, 128 * 4));
}

/// 2026-09-25: The cuBLASLt helper writes `ceil16(M)` rows, so an output
/// buffer one byte short of that declines.
#[test]
fn not_selected_when_the_output_buffer_cannot_hold_the_padded_m() {
    let f = WeightQuantFormat::Fp8BlockScaled;
    let short = out_bytes() - 1;
    assert!(!selected(
        true,
        f,
        QKVZ,
        H,
        short,
        KMAJOR_K,
        KMAJOR_BUF,
        scale_bytes()
    ));
    assert!(selected(
        true,
        f,
        QKVZ,
        H,
        out_bytes(),
        KMAJOR_K,
        KMAJOR_BUF,
        scale_bytes()
    ));
}

/// 2026-09-25: Under the k-major layout, a missing adapter kernel, a missing
/// scratch, or a scratch too small for the padded M each make the arm decline.
#[test]
fn not_selected_without_a_usable_kmajor_scale_adapter() {
    let f = WeightQuantFormat::Fp8BlockScaled;
    // 2026-09-25: `METRALE_CUBLAS_SCALE_LAYOUT=rowmajor` needs no adapter, and
    // the layout is cached once per process, so this test checks only the
    // default layout.
    if !crate::layers::ops::cublas_scale_layout_kmajor() {
        return;
    }
    assert!(!selected(
        true,
        f,
        QKVZ,
        H,
        out_bytes(),
        KernelHandle(0),
        KMAJOR_BUF,
        scale_bytes()
    ));
    assert!(!selected(
        true,
        f,
        QKVZ,
        H,
        out_bytes(),
        KMAJOR_K,
        DevicePtr::NULL,
        scale_bytes()
    ));
    assert!(!selected(
        true,
        f,
        QKVZ,
        H,
        out_bytes(),
        KMAJOR_K,
        KMAJOR_BUF,
        scale_bytes() - 1
    ));
}
