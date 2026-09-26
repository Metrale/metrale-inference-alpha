// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-25: Clause-by-clause tests of the cache-skip Q/K/V prefill cuBLASLt
//! selector, and of the buffer extents the pad-row argument rests on. Each
//! selection case starts from a selecting input and changes one thing, so a
//! failure names the clause.
//!
//! Owner: model-layers (qwen3 attention).
//! Invariants: none beyond the types.

use super::{cache_skip_qkv_cublas_selected, cache_skip_qkv_extents};
use crate::layers::ops::{self, cublas_fp8_m_pad};
use crate::weight_map::WeightQuantFormat;
use metrale_gpu_runtime::gpu::{DevicePtr, KernelHandle};

/// 2026-09-25: Test shapes: hidden `H`, gated `q_proj` width `Q_N`
/// (`[Q | gate]`), kv width `KV_N`, chunk rows `M`.
const H: u32 = 5120;
const Q_N: u32 = 12288;
const KV_N: u32 = 1024;
const M: u32 = 1168;
/// 2026-09-25: A quantizer with only the shared kernel; the selector reads only
/// `Fp8ActQuant::available()`, which is true when either kernel is present.
const QUANT_K: ops::Fp8ActQuant = ops::Fp8ActQuant {
    shared: KernelHandle(0xBEEF),
    hopper: KernelHandle(0),
};
const KMAJOR_K: KernelHandle = KernelHandle(0xC0DE);
const KMAJOR_BUF: DevicePtr = DevicePtr(0x1000);
const BLK: Option<WeightQuantFormat> = Some(WeightQuantFormat::Fp8BlockScaled);

fn m_pad(m: u32) -> usize {
    cublas_fp8_m_pad(m) as usize
}

fn caps(m: u32) -> (usize, usize, usize, usize) {
    let (q_elems, kv_elems) = cache_skip_qkv_extents(m, Q_N, KV_N);
    (
        q_elems * 2,
        kv_elems * 2,
        m_pad(m) * H as usize,
        m_pad(m) * (H as usize / 128) * 4,
    )
}

#[allow(clippy::too_many_arguments)]
fn selected(
    cublas_attn: bool,
    blockscaled: bool,
    w8a16_only: bool,
    formats: [Option<WeightQuantFormat>; 3],
    m: u32,
    q_n: u32,
    kv_n: u32,
    k: u32,
    qkv_cap: usize,
    qkvz_cap: usize,
    act_cap: usize,
    act_scale_cap: usize,
    quant_k: ops::Fp8ActQuant,
    kmajor_k: KernelHandle,
    kmajor_buf: DevicePtr,
    kmajor_cap: usize,
) -> bool {
    cache_skip_qkv_cublas_selected(
        cublas_attn,
        blockscaled,
        w8a16_only,
        formats,
        m,
        q_n,
        kv_n,
        k,
        qkv_cap,
        qkvz_cap,
        act_cap,
        act_scale_cap,
        quant_k,
        kmajor_k,
        kmajor_buf,
        kmajor_cap,
    )
}

fn ready_at(m: u32) -> bool {
    let (q, kv, a, s) = caps(m);
    selected(
        true,
        true,
        false,
        [BLK, BLK, BLK],
        m,
        Q_N,
        KV_N,
        H,
        q,
        kv,
        a,
        s,
        QUANT_K,
        KMAJOR_K,
        KMAJOR_BUF,
        s,
    )
}

/// 2026-09-25: `v` starts at row `m` and writes `ceil16(m)` rows, so the
/// furthest element is `(m + ceil16(m)) * kv_dim`: what the `ssm_qkvz` clause
/// checks, and what `gpu-runtime` `buffers/sizes.rs` allocates.
#[test]
fn extents_cover_the_furthest_padded_write() {
    for m in [16u32, 25, 1168, 1193, 4593] {
        let (q, kv) = cache_skip_qkv_extents(m, Q_N, KV_N);
        assert_eq!(q, m_pad(m) * Q_N as usize);
        assert_eq!(kv, (m as usize + m_pad(m)) * KV_N as usize);
        // 2026-09-25: Never smaller than `2 * m * kv_dim`.
        assert!(kv >= 2 * m as usize * KV_N as usize);
        // 2026-09-25: k's pad rows (m..m_pad of k's region) are inside v's region.
        assert!(m_pad(m) <= m as usize + m_pad(m));
    }
}

#[test]
fn selected_at_prefill_m_with_the_attn_scope_armed() {
    assert!(ready_at(M), "the round-9 chunk-0 shape must select");
    assert!(ready_at(4593), "the long-prompt shape must select");
}

#[test]
fn not_selected_without_the_attn_scope() {
    let (q, kv, a, s) = caps(M);
    assert!(!selected(
        false,
        true,
        false,
        [BLK, BLK, BLK],
        M,
        Q_N,
        KV_N,
        H,
        q,
        kv,
        a,
        s,
        QUANT_K,
        KMAJOR_K,
        KMAJOR_BUF,
        s,
    ));
}

#[test]
fn kill_switch_refuses_even_with_the_scope_armed() {
    let (q, kv, a, s) = caps(M);
    assert!(!selected(
        true,
        true,
        true,
        [BLK, BLK, BLK],
        M,
        Q_N,
        KV_N,
        H,
        q,
        kv,
        a,
        s,
        QUANT_K,
        KMAJOR_K,
        KMAJOR_BUF,
        s,
    ));
}

#[test]
fn single_scale_kill_switch_refuses() {
    let (q, kv, a, s) = caps(M);
    assert!(!selected(
        true,
        false,
        false,
        [BLK, BLK, BLK],
        M,
        Q_N,
        KV_N,
        H,
        q,
        kv,
        a,
        s,
        QUANT_K,
        KMAJOR_K,
        KMAJOR_BUF,
        s,
    ));
}

/// 2026-09-25: Below 16 rows `ceil16(m) - m` can exceed `m`, so `v`'s real rows
/// would not cover `k`'s pad rows; those `m` fall back.
#[test]
fn fewer_than_sixteen_rows_falls_back() {
    for m in [1u32, 5, 8, 15] {
        assert!(!ready_at(m), "M={m} must fall back to W8A16");
    }
    assert!(ready_at(16));
}

/// 2026-09-25: All three or none: if any weight is not block-scaled FP8, none
/// of the three takes the route.
#[test]
fn every_projection_must_be_block_scaled() {
    let (q, kv, a, s) = caps(M);
    let case = |formats: [Option<WeightQuantFormat>; 3]| {
        selected(
            true, true, false, formats, M, Q_N, KV_N, H, q, kv, a, s, QUANT_K, KMAJOR_K,
            KMAJOR_BUF, s,
        )
    };
    assert!(case([BLK, BLK, BLK]));
    let rowwise = Some(WeightQuantFormat::Fp8PerRow);
    assert!(!case([rowwise, BLK, BLK]), "q per-row");
    assert!(!case([BLK, rowwise, BLK]), "k per-row");
    assert!(!case([BLK, BLK, rowwise]), "v per-row");
    assert!(!case([None, BLK, BLK]), "q not FP8 at all");
}

#[test]
fn shape_clauses() {
    let (q, kv, a, s) = caps(M);
    let case = |q_n: u32, kv_n: u32, k: u32| {
        let (qe, kve) = cache_skip_qkv_extents(M, q_n, kv_n);
        selected(
            true,
            true,
            false,
            [BLK, BLK, BLK],
            M,
            q_n,
            kv_n,
            k,
            qe * 2,
            kve * 2,
            m_pad(M) * k as usize,
            m_pad(M) * (k as usize).div_ceil(128) * 4,
            QUANT_K,
            KMAJOR_K,
            KMAJOR_BUF,
            m_pad(M) * (k as usize).div_ceil(128) * 4,
        )
    };
    let _ = (q, kv, a, s);
    assert!(case(Q_N, KV_N, H));
    assert!(!case(Q_N + 64, KV_N, H), "q N must be a multiple of 128");
    assert!(!case(Q_N, KV_N + 64, H), "kv N must be a multiple of 128");
    assert!(!case(Q_N, KV_N, H + 64), "K must be a multiple of 128");
    assert!(!case(Q_N, KV_N, H + 128), "K/128 must be a multiple of 4");
}

#[test]
fn capacity_clauses_each_refuse_on_their_own() {
    let (q, kv, a, s) = caps(M);
    let case = |q: usize, kv: usize, a: usize, sc: usize, km: usize| {
        selected(
            true,
            true,
            false,
            [BLK, BLK, BLK],
            M,
            Q_N,
            KV_N,
            H,
            q,
            kv,
            a,
            sc,
            QUANT_K,
            KMAJOR_K,
            KMAJOR_BUF,
            km,
        )
    };
    assert!(case(q, kv, a, s, s));
    assert!(!case(q - 1, kv, a, s, s), "qkv_output room");
    assert!(!case(q, kv - 1, a, s, s), "ssm_qkvz room for v's pad");
    assert!(!case(q, kv, a - 1, s, s), "activation room");
    assert!(!case(q, kv, a, s - 1, s), "act-scale room");
    assert!(!case(q, kv, a, s, s - 1), "k-major room");
}

/// 2026-09-25: A `ssm_qkvz` of `2 * m * kv_dim` is `ceil16(m) - m` rows short
/// when M is not a multiple of 16 (7 rows at M = 1193), and is refused.
#[test]
fn the_old_kv_sizing_is_refused_for_a_ragged_m() {
    let m = 1193;
    let (q, _kv, a, s) = caps(m);
    let old = 2 * m as usize * KV_N as usize * 2;
    assert!(!selected(
        true,
        true,
        false,
        [BLK, BLK, BLK],
        m,
        Q_N,
        KV_N,
        H,
        q,
        old,
        a,
        s,
        QUANT_K,
        KMAJOR_K,
        KMAJOR_BUF,
        s,
    ));
}

#[test]
fn missing_handles_fall_back() {
    let (q, kv, a, s) = caps(M);
    let case = |quant: ops::Fp8ActQuant, kmajor: KernelHandle, buf: DevicePtr| {
        selected(
            true,
            true,
            false,
            [BLK, BLK, BLK],
            M,
            Q_N,
            KV_N,
            H,
            q,
            kv,
            a,
            s,
            quant,
            kmajor,
            buf,
            s,
        )
    };
    assert!(case(QUANT_K, KMAJOR_K, KMAJOR_BUF));
    assert!(
        !case(ops::Fp8ActQuant::default(), KMAJOR_K, KMAJOR_BUF),
        "quantizer"
    );
    if crate::layers::ops::cublas_scale_layout_kmajor() {
        assert!(!case(QUANT_K, KernelHandle(0), KMAJOR_BUF), "kmajor kernel");
        assert!(!case(QUANT_K, KMAJOR_K, DevicePtr(0)), "kmajor scratch");
    }
}
