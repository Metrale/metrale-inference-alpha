// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-25: Tests of `out_proj_cublas_selected`, one clause each: every case
//! starts from inputs that select the cuBLASLt arm and changes one of them.
//!
//! Owner: model-layers (qwen3_ssm tests).
//! Invariants: none beyond the types.

use super::out_proj_cublas_selected;
use crate::layers::ops::{self, cublas_fp8_m_pad};
use crate::weight_map::WeightQuantFormat;
use metrale_gpu_runtime::gpu::{DevicePtr, KernelHandle};

const N: u32 = 5120;
const K: u32 = 6144;
const M: u32 = 1168;
/// 2026-09-25: A quantizer with only the shared kernel resolved; the selector
/// reads only `Fp8ActQuant::available`.
const QUANT_K: ops::Fp8ActQuant = ops::Fp8ActQuant {
    shared: KernelHandle(0xBEEF),
    hopper: KernelHandle(0),
};
const KMAJOR_K: KernelHandle = KernelHandle(0xC0DE);
const KMAJOR_BUF: DevicePtr = DevicePtr(0x1000);

fn m_pad(m: u32) -> usize {
    cublas_fp8_m_pad(m) as usize
}

fn out_bytes(m: u32) -> usize {
    m_pad(m) * N as usize * 2
}

fn act_bytes(m: u32) -> usize {
    m_pad(m) * K as usize
}

fn scale_bytes(m: u32) -> usize {
    m_pad(m) * (K as usize / 128) * 4
}

#[allow(clippy::too_many_arguments)]
fn selected(
    cublas_ssm: bool,
    blockscaled: bool,
    w8a16_only: bool,
    fmt: WeightQuantFormat,
    m: u32,
    n: u32,
    k: u32,
    out_capacity: usize,
    act_capacity: usize,
    act_scale_capacity: usize,
    quant_k: ops::Fp8ActQuant,
    kmajor_k: KernelHandle,
    kmajor_buf: DevicePtr,
    kmajor_capacity: usize,
) -> bool {
    out_proj_cublas_selected(
        cublas_ssm,
        blockscaled,
        w8a16_only,
        fmt,
        m,
        n,
        k,
        out_capacity,
        act_capacity,
        act_scale_capacity,
        quant_k,
        kmajor_k,
        kmajor_buf,
        kmajor_capacity,
    )
}

fn ready_at(m: u32) -> bool {
    selected(
        true,
        true,
        false,
        WeightQuantFormat::Fp8BlockScaled,
        m,
        N,
        K,
        out_bytes(m),
        act_bytes(m),
        scale_bytes(m),
        QUANT_K,
        KMAJOR_K,
        KMAJOR_BUF,
        scale_bytes(m),
    )
}

#[test]
fn selected_at_prefill_m_with_the_ssm_scope_armed() {
    assert!(ready_at(M), "the round-9 chunk-0 shape must select");
    assert!(ready_at(25), "the 25-token tail chunk must select too");
}

#[test]
fn not_selected_without_the_ssm_scope() {
    // 2026-09-25: `METRALE_CUBLAS_GEMM` without `ssm`, for example `ffn`.
    assert!(!selected(
        false,
        true,
        false,
        WeightQuantFormat::Fp8BlockScaled,
        M,
        N,
        K,
        out_bytes(M),
        act_bytes(M),
        scale_bytes(M),
        QUANT_K,
        KMAJOR_K,
        KMAJOR_BUF,
        scale_bytes(M),
    ));
}

#[test]
fn kill_switch_refuses_even_with_the_scope_armed() {
    assert!(!selected(
        true,
        true,
        true,
        WeightQuantFormat::Fp8BlockScaled,
        M,
        N,
        K,
        out_bytes(M),
        act_bytes(M),
        scale_bytes(M),
        QUANT_K,
        KMAJOR_K,
        KMAJOR_BUF,
        scale_bytes(M),
    ));
}

#[test]
fn single_scale_kill_switch_refuses() {
    // 2026-09-25: `METRALE_FP8_SINGLE_SCALE=1` clears
    // `dispatch.fp8_blockscaled_prefill`.
    assert!(!selected(
        true,
        false,
        false,
        WeightQuantFormat::Fp8BlockScaled,
        M,
        N,
        K,
        out_bytes(M),
        act_bytes(M),
        scale_bytes(M),
        QUANT_K,
        KMAJOR_K,
        KMAJOR_BUF,
        scale_bytes(M),
    ));
}

#[test]
fn m_at_or_below_four_stays_on_the_gemv_tier() {
    for m in [1, 2, 4] {
        assert!(!ready_at(m), "M={m} must stay on the GEMV tier");
    }
    assert!(ready_at(5), "M=5 is the first row count this arm claims");
}

#[test]
fn per_row_scales_are_refused() {
    assert!(!selected(
        true,
        true,
        false,
        WeightQuantFormat::Fp8PerRow,
        M,
        N,
        K,
        out_bytes(M),
        act_bytes(M),
        scale_bytes(M),
        QUANT_K,
        KMAJOR_K,
        KMAJOR_BUF,
        scale_bytes(M),
    ));
}

#[test]
fn shape_clauses() {
    let case = |n: u32, k: u32| {
        selected(
            true,
            true,
            false,
            WeightQuantFormat::Fp8BlockScaled,
            M,
            n,
            k,
            m_pad(M) * n as usize * 2,
            m_pad(M) * k as usize,
            m_pad(M) * (k as usize).div_ceil(128) * 4,
            QUANT_K,
            KMAJOR_K,
            KMAJOR_BUF,
            m_pad(M) * (k as usize).div_ceil(128) * 4,
        )
    };
    assert!(case(N, K));
    assert!(!case(N + 64, K), "N must be a multiple of 128");
    assert!(!case(N, K + 64), "K must be a multiple of 128");
    // 2026-09-25: K/128 must be a multiple of 4 (`blk128x128_stride_ok`).
    assert!(
        !case(N, 5120 + 128),
        "K=5248 has stride 41, not a multiple of 4"
    );
}

#[test]
fn capacity_clauses_each_refuse_on_their_own() {
    let full = (out_bytes(M), act_bytes(M), scale_bytes(M), scale_bytes(M));
    let case = |o: usize, a: usize, s: usize, km: usize| {
        selected(
            true,
            true,
            false,
            WeightQuantFormat::Fp8BlockScaled,
            M,
            N,
            K,
            o,
            a,
            s,
            QUANT_K,
            KMAJOR_K,
            KMAJOR_BUF,
            km,
        )
    };
    assert!(case(full.0, full.1, full.2, full.3));
    assert!(!case(full.0 - 1, full.1, full.2, full.3), "output room");
    assert!(!case(full.0, full.1 - 1, full.2, full.3), "activation room");
    assert!(!case(full.0, full.1, full.2 - 1, full.3), "act-scale room");
    assert!(!case(full.0, full.1, full.2, full.3 - 1), "k-major room");
}

#[test]
fn an_unpadded_output_buffer_is_refused_when_m_is_not_a_multiple_of_16() {
    // 2026-09-25: M=25 pads to 32 rows, so a buffer sized for 25 is too small.
    let m = 25;
    let unpadded = m as usize * N as usize * 2;
    assert!(!selected(
        true,
        true,
        false,
        WeightQuantFormat::Fp8BlockScaled,
        m,
        N,
        K,
        unpadded,
        act_bytes(m),
        scale_bytes(m),
        QUANT_K,
        KMAJOR_K,
        KMAJOR_BUF,
        scale_bytes(m),
    ));
}

#[test]
fn missing_handles_fall_back() {
    let case = |quant: ops::Fp8ActQuant, kmajor: KernelHandle, buf: DevicePtr| {
        selected(
            true,
            true,
            false,
            WeightQuantFormat::Fp8BlockScaled,
            M,
            N,
            K,
            out_bytes(M),
            act_bytes(M),
            scale_bytes(M),
            quant,
            kmajor,
            buf,
            scale_bytes(M),
        )
    };
    assert!(case(QUANT_K, KMAJOR_K, KMAJOR_BUF));
    assert!(
        !case(ops::Fp8ActQuant::default(), KMAJOR_K, KMAJOR_BUF),
        "quantizer"
    );
    // 2026-09-25: The k-major clauses apply only under the default layout, so
    // these asserts are skipped under `METRALE_CUBLAS_SCALE_LAYOUT=rowmajor`.
    if crate::layers::ops::cublas_scale_layout_kmajor() {
        assert!(!case(QUANT_K, KernelHandle(0), KMAJOR_BUF), "kmajor kernel");
        assert!(!case(QUANT_K, KMAJOR_K, DevicePtr(0)), "kmajor scratch");
    }
}
