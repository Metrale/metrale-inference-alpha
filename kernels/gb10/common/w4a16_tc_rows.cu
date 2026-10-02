// SPDX-License-Identifier: MIT OR Apache-2.0

// 2026-10-02: NVFP4 W4A16 projection for 1..=64 decode rows on mma.sync m16n8k16 BF16 tiles with
// the rows as the MMA's N columns: the NVFP4 point (Nvfp4G16) of tc_rows.cuh, whose FP8 point is
// w8a16_tc_rows.cu.
//   C[r, n] = s2 * sum_k A[r, k] * BF16(E2M1(W[n, k]) * E4M3(scale[n, k / 16])),  r < M, n < N
//
// Why: a row-invariant W4A16 tile at every row count, so a declared-W4A16 projection (the lm_head
// of nvidia/Qwen3.6-35B-A3B-NVFP4) keeps 16-bit activations and one summation order per row; the
// NVFP4 tile GEMM it replaces there casts activations to E4M3.
//
// Owner: gb10 kernels.
// Invariants:
// - W: packed E2M1 [N, K / 2] (element 2j in the low nibble of byte j), E4M3 scales [N, K / 16],
//   FP32 s2. The host guarantees 1 <= M <= 8 * NT, N > 0 (any: the ragged tail loads zero weight
//   rows and stores nothing past N; a 248070-row vocab is not a multiple of 64), K a positive
//   multiple of 256, lda a multiple of 8 and >= K, ldc >= N. Grid (ceil(N / TR_COLS), 1, 1).
// - A row's output bits do not depend on M, on the other rows, or on the entry point.
// - Each entry point keeps the shared-memory footprint of its FP8 twin: a group spans the same
//   K (G halves, CHUNK_K doubles).
// - Block TR_THREADS, static shared memory only.

#include "tc_rows.cuh"

// 2026-10-02: 1..=16 rows.
extern "C" __global__ void __launch_bounds__(TR_THREADS) w4a16_tc_rows_16(
    const __nv_bfloat16* __restrict__ A, const unsigned char* __restrict__ packed,
    const unsigned char* __restrict__ scale, float s2, __nv_bfloat16* __restrict__ C,
    unsigned int M, unsigned int N, unsigned int K, unsigned int lda, unsigned int ldc
) {
    tr_block<Nvfp4G16, 2, 2, true>(A, {packed, scale, s2}, C, M, N, K, lda, ldc, blockIdx.x);
}

// 2026-10-02: 1..=32 rows.
extern "C" __global__ void __launch_bounds__(TR_THREADS) w4a16_tc_rows_32(
    const __nv_bfloat16* __restrict__ A, const unsigned char* __restrict__ packed,
    const unsigned char* __restrict__ scale, float s2, __nv_bfloat16* __restrict__ C,
    unsigned int M, unsigned int N, unsigned int K, unsigned int lda, unsigned int ldc
) {
    tr_block<Nvfp4G16, 4, 1, true>(A, {packed, scale, s2}, C, M, N, K, lda, ldc, blockIdx.x);
}

// 2026-10-02: 1..=64 rows.
extern "C" __global__ void __launch_bounds__(TR_THREADS) w4a16_tc_rows_64(
    const __nv_bfloat16* __restrict__ A, const unsigned char* __restrict__ packed,
    const unsigned char* __restrict__ scale, float s2, __nv_bfloat16* __restrict__ C,
    unsigned int M, unsigned int N, unsigned int K, unsigned int lda, unsigned int ldc
) {
    tr_block<Nvfp4G16, 8, 1, true>(A, {packed, scale, s2}, C, M, N, K, lda, ldc, blockIdx.x);
}
