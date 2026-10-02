// SPDX-License-Identifier: MIT OR Apache-2.0

// 2026-09-28: Block-scaled W8A16 projection for 1..=64 decode rows on mma.sync m16n8k16 BF16
// tiles with the rows as the MMA's N columns (2026-10-02: the FP8 point of tc_rows.cuh; the
// NVFP4 point is w4a16_tc_rows.cu):
//   C[r, n] = sum_k A[r, k] * E4M3(B[n, k]) * block_scale[n / 128, k / 128],  r < M, n < N
//
// Why: the tile kernels this replaces under the canonical row tiers (w8a16_gemm_pipelined_m32 /
// _m64) decode every weight byte through a shared-memory table into a BF16 shared tile and
// read it back per MMA; on GB10 that draws 50-75 W at 1..32 rows. Here a weight byte goes
// straight from a 16-byte global load to a BF16 fragment register (a byte permute and two
// logic ops) and is applied to every row tile; the activations are staged once per block in
// shared memory. Standalone at Qwen3.6-35B-A3B shapes (dgx2/dgx3): 30-50% fewer GPU-rail
// joules per launch; the same or less time from 16 rows up; at 1..8 rows up to 5% more time
// on the widest projection (12288 x 2048) and less on the others.
//
// Owner: gb10 kernels.
// Invariants:
// - A [M, lda] BF16 (the first K of each row read), B [N, K] E4M3 bytes, block_scale
//   [N / 128, K / 128] FP32, C [M, ldc] BF16 (the first N of each row written). The host
//   guarantees 1 <= M <= 8 * NT (any M >= 1 for `_64c`), N a positive multiple of TR_COLS and of 128, K a positive
//   multiple of 128, lda a multiple of 8 (16-byte rows) and lda >= K, ldc >= N.
// - A weight byte b becomes the BF16 whose bits are sign(b) | (b & 0x7F) << 4, which equals
//   E4M3(b) * 2^-120 exactly; activations are staged times 2^60 (exact in BF16) and each
//   128-K block's partial sum is scaled by block_scale * 2^60, so products and partial sums
//   stay FP32 normals (moe_fp8_grouped_tc.cu uses the same scheme).
// - Inside a 64-wide K chunk, lane t = lane & 3 holds K = 16t .. 16t + 15 of its weight rows
//   and activation rows; MMA j (0..3) takes K = 16t + 4j + {0,1} as fragment slots 2t, 2t+1
//   and K = 16t + 4j + {2,3} as 2t+8, 2t+9. A row's sum order is fixed by K alone and its
//   column of the MMA reads only its own activations, so a row's output bits do not depend
//   on M or on the other rows (the three entry points agree row for row).
// - Grid (N / TR_COLS, 1, 1) (`_64c`: times ceil(M / 64)), block TR_THREADS, static shared
//   memory only.

#include "tc_rows.cuh"

// 2026-09-28: 1..=16 rows, 256-byte weight runs.
extern "C" __global__ void __launch_bounds__(TR_THREADS) w8a16_tc_rows_16(
    const __nv_bfloat16* __restrict__ A, const unsigned char* __restrict__ B,
    const float* __restrict__ block_scale, __nv_bfloat16* __restrict__ C,
    unsigned int M, unsigned int N, unsigned int K, unsigned int lda, unsigned int ldc
) {
    tr_block<Fp8Block128, 2, 4>(A, {B, block_scale}, C, M, N, K, lda, ldc, blockIdx.x);
}

// 2026-09-28: 1..=32 rows (from 17 rows the 256-byte runs of `_16` cost more than they save).
extern "C" __global__ void __launch_bounds__(TR_THREADS) w8a16_tc_rows_32(
    const __nv_bfloat16* __restrict__ A, const unsigned char* __restrict__ B,
    const float* __restrict__ block_scale, __nv_bfloat16* __restrict__ C,
    unsigned int M, unsigned int N, unsigned int K, unsigned int lda, unsigned int ldc
) {
    tr_block<Fp8Block128, 4, 2>(A, {B, block_scale}, C, M, N, K, lda, ldc, blockIdx.x);
}

// 2026-09-28: 33..=64 rows (any 1..=64; a row's bits equal the other entry points').
extern "C" __global__ void __launch_bounds__(TR_THREADS) w8a16_tc_rows_64(
    const __nv_bfloat16* __restrict__ A, const unsigned char* __restrict__ B,
    const float* __restrict__ block_scale, __nv_bfloat16* __restrict__ C,
    unsigned int M, unsigned int N, unsigned int K, unsigned int lda, unsigned int ldc
) {
    tr_block<Fp8Block128, 8, 2>(A, {B, block_scale}, C, M, N, K, lda, ldc, blockIdx.x);
}

// 2026-09-30: Any number of rows, in 64-row chunks: block b computes chunk b % chunks of column
// block b / chunks, so the chunks of one column block run next to each other and share its
// weight tile through L2. Every row runs `tr_block<8, 2>`, whose row bits do not depend on M,
// so a row's output equals the other entry points' at every row count.
extern "C" __global__ void __launch_bounds__(TR_THREADS) w8a16_tc_rows_64c(
    const __nv_bfloat16* __restrict__ A, const unsigned char* __restrict__ B,
    const float* __restrict__ block_scale, __nv_bfloat16* __restrict__ C,
    unsigned int M, unsigned int N, unsigned int K, unsigned int lda, unsigned int ldc
) {
    const unsigned int chunks = (M + 63) / 64;
    const unsigned int chunk = blockIdx.x % chunks;
    const unsigned int rows = min(64u, M - chunk * 64);
    tr_block<Fp8Block128, 8, 2>(A + (unsigned long long)chunk * 64 * lda, {B, block_scale},
                                C + (unsigned long long)chunk * 64 * ldc, rows, N, K, lda, ldc,
                                blockIdx.x / chunks);
}
