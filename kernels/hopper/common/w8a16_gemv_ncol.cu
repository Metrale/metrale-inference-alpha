// SPDX-License-Identifier: MIT OR Apache-2.0

// 2026-09-25: W8A16 batched decode GEMV, N-column blocked: block-scaled FP8 E4M3
// weights, BF16 activations, up to 16 rows.
//
//   C[t, n] = sum_k A[t, k] * E4M3_LUT[B[n, k]] * block_scale[n / 128, k / 128],  t < M
//
// A is [M, a_row_stride] BF16 (first K used), B is [N, K] E4M3 bytes, block_scale is
// [ceil(N/128), ceil(K/128)] FP32 and C is [M, c_row_stride] BF16 (first N used).
// Block (256, 1, 1), grid (ceil(N / (4 * N_COLS)), 1, 1). Rows t >= 16 are not
// computed; `ops::w8a16_gemv_batch16_ncol*` refuses M outside 1..=16 and a K that
// is not a multiple of 16.
//
// The result is `w8a16_gemv_batchm_impl<16>`'s (gb10/common/w8a16_gemv_batch4.cu),
// with each thread owning N_COLS adjacent columns instead of one, so one activation
// load and one BF16 -> FP32 convert serve N_COLS columns. Each accumulator gets the
// same operands in the same order as there: lane l of a 64-thread group walks
// chunks l, l + 64, ...; a chunk's products are added in K order; the reduction is
// the same shfl.down tree and two-warp add per (row, column).
// crates/model-arch/examples/native_fp8_attn_decode_batch_microtest.rs requires
// every route's output to equal the scalar `w8a16_gemv` loop's bits.
//
// Entry points: `w8a16_gemv_batch16_ncol2` / `_ncol4` (contiguous A and C) and their
// `_strided` siblings, which take the A and C row pitches in elements. The decode
// attention projections use them when `attn_ncol_gemv` resolves on
// (crates/model-layers/src/layers/qwen3_attention/attn_ncol_gemv.rs).
//
// Owner: hopper kernels.
// Invariants:
// - Only C[t, n] with t < M and n < N is written, once each.

















#include <cuda_bf16.h>

#include "e4m3_lut.cuh"

#define BLOCK_SIZE 256
// 2026-09-25: 64-thread output groups per block. Each group owns N_COLS adjacent
// columns, so a block covers 4 * N_COLS columns.

#define N_GROUPS_PER_BLOCK 4
#define WARP_SIZE 32
#define FP8_BLOCK 128

template <int MAX_M, int N_COLS>
__device__ __forceinline__ void w8a16_gemv_ncol_impl(
    const __nv_bfloat16* __restrict__ A,
    const unsigned char* __restrict__ B,
    const float* __restrict__ block_scale,
    __nv_bfloat16* __restrict__ C,
    unsigned int M,
    unsigned int N,
    unsigned int K,
    unsigned int a_row_stride,
    unsigned int c_row_stride
) {
    const unsigned int threads_per_out = BLOCK_SIZE / N_GROUPS_PER_BLOCK;
    const unsigned int local_out = threadIdx.x / threads_per_out;
    const unsigned int lane = threadIdx.x % threads_per_out;

// 2026-09-25: The group's first column, and how many of its N_COLS columns are in
// range. A group past N does not return early, so every thread reaches both
// `__syncthreads`.
    const unsigned int n0 = (blockIdx.x * N_GROUPS_PER_BLOCK + local_out) * N_COLS;
    const unsigned int ncols = (n0 < N) ? min((unsigned int)N_COLS, N - n0) : 0u;

    __shared__ float s_lut[256];
    s_lut[threadIdx.x] = E4M3_LUT[threadIdx.x];
    __syncthreads();

    const unsigned int K16 = K / 16;
    const unsigned int k_blocks = (K + FP8_BLOCK - 1) / FP8_BLOCK;

    float acc[MAX_M][N_COLS];
    #pragma unroll
    for (int t = 0; t < MAX_M; t++) {
        #pragma unroll
        for (int c = 0; c < N_COLS; c++) acc[t][c] = 0.0f;
    }

// 2026-09-25: A group with no column in range starts at k16 = K16 and skips the
// loop; the loop holds no `__syncthreads`.


    for (unsigned int k16 = (ncols != 0) ? lane : K16; k16 < K16; k16 += threads_per_out) {
        const unsigned int base_k = k16 * 16;
        const unsigned int k_block = base_k / FP8_BLOCK;

// 2026-09-25: Each column's 16 weight bytes are decoded and scaled once for all M
// rows. A tail group's out-of-range columns are clamped to column N - 1: they
// compute a duplicate into accumulators that are never stored, so the
// multiply-add loop below carries no per-column predicate. No group is a tail
// group when N is a multiple of 4 * N_COLS.






        float wf[N_COLS][16];
        #pragma unroll
        for (int c = 0; c < N_COLS; c++) {
            const unsigned int n = min(n0 + (unsigned int)c, N - 1);
            const float scale = block_scale[(n / FP8_BLOCK) * k_blocks + k_block];
            uint4 b_data = ((const uint4*)(B + (unsigned long long)n * K))[k16];
            const unsigned int b_raw[4] = {b_data.x, b_data.y, b_data.z, b_data.w};
            #pragma unroll
            for (int i = 0; i < 4; i++) {
                unsigned int w32 = b_raw[i];
                wf[c][i * 4 + 0] = s_lut[(w32      ) & 0xFF] * scale;
                wf[c][i * 4 + 1] = s_lut[(w32 >>  8) & 0xFF] * scale;
                wf[c][i * 4 + 2] = s_lut[(w32 >> 16) & 0xFF] * scale;
                wf[c][i * 4 + 3] = s_lut[(w32 >> 24) & 0xFF] * scale;
            }
        }

// 2026-09-25: One activation load and one convert per element, used for all N_COLS
// columns.

        #pragma unroll
        for (int t = 0; t < MAX_M; t++) {
            if ((unsigned int)t >= M) continue;
            const __nv_bfloat16* At = A + (unsigned long long)t * a_row_stride;
            uint4 a_lo = ((const uint4*)At)[k16 * 2];
            uint4 a_hi = ((const uint4*)At)[k16 * 2 + 1];
            const unsigned int ar[8] = {a_lo.x, a_lo.y, a_lo.z, a_lo.w,
                                        a_hi.x, a_hi.y, a_hi.z, a_hi.w};
            #pragma unroll
            for (int j = 0; j < 8; j++) {
                __nv_bfloat16 lo, hi;
                *(unsigned short*)&lo = (unsigned short)(ar[j] & 0xFFFF);
                *(unsigned short*)&hi = (unsigned short)(ar[j] >> 16);
                const float flo = __bfloat162float(lo);
                const float fhi = __bfloat162float(hi);
                #pragma unroll
                for (int c = 0; c < N_COLS; c++) {
// 2026-09-25: Two separate adds, as in the single-column kernel; the pair is never summed first.
                    acc[t][c] += flo * wf[c][j * 2];
                    acc[t][c] += fhi * wf[c][j * 2 + 1];
                }
            }
        }
    }

// 2026-09-25: Per (row, column), the single-column kernel's shfl.down tree, then one
// shared-memory slot per warp.
    __shared__ float smem[MAX_M][N_GROUPS_PER_BLOCK * N_COLS * 2];
    const unsigned int warp_in_out = lane / WARP_SIZE;
    #pragma unroll
    for (int t = 0; t < MAX_M; t++) {
        if ((unsigned int)t >= M) continue;
        #pragma unroll
        for (int c = 0; c < N_COLS; c++) {
// 2026-09-25: A tail group's clamped columns are reduced too but not stored, so every
// lane reaches every `__shfl_down_sync`.
            float a = acc[t][c];
            #pragma unroll
            for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1) {
                a += __shfl_down_sync(0xFFFFFFFF, a, offset);
            }
            if (lane % WARP_SIZE == 0) {
                smem[t][(local_out * N_COLS + c) * 2 + warp_in_out] = a;
            }
        }
    }
    __syncthreads();

    if (lane == 0) {
        #pragma unroll
        for (int t = 0; t < MAX_M; t++) {
            if ((unsigned int)t >= M) continue;
            #pragma unroll
            for (int c = 0; c < N_COLS; c++) {
                if ((unsigned int)c >= ncols) continue;
                const unsigned int slot = (local_out * N_COLS + c) * 2;
                float r = smem[t][slot] + smem[t][slot + 1];
                C[(unsigned long long)t * c_row_stride + (n0 + c)] = __float2bfloat16(r);
            }
        }
    }
}


extern "C" __global__ void w8a16_gemv_batch16_ncol2(
    const __nv_bfloat16* __restrict__ A,
    const unsigned char* __restrict__ B,
    const float* __restrict__ block_scale,
    __nv_bfloat16* __restrict__ C,
    unsigned int M,
    unsigned int N,
    unsigned int K
) {
    w8a16_gemv_ncol_impl<16, 2>(A, B, block_scale, C, M, N, K, K, N);
}


extern "C" __global__ void w8a16_gemv_batch16_ncol4(
    const __nv_bfloat16* __restrict__ A,
    const unsigned char* __restrict__ B,
    const float* __restrict__ block_scale,
    __nv_bfloat16* __restrict__ C,
    unsigned int M,
    unsigned int N,
    unsigned int K
) {
    w8a16_gemv_ncol_impl<16, 4>(A, B, block_scale, C, M, N, K, K, N);
}


extern "C" __global__ void w8a16_gemv_batch16_ncol2_strided(
    const __nv_bfloat16* __restrict__ A,
    const unsigned char* __restrict__ B,
    const float* __restrict__ block_scale,
    __nv_bfloat16* __restrict__ C,
    unsigned int M,
    unsigned int N,
    unsigned int K,
    unsigned int a_row_stride,
    unsigned int c_row_stride
) {
    w8a16_gemv_ncol_impl<16, 2>(A, B, block_scale, C, M, N, K, a_row_stride, c_row_stride);
}

extern "C" __global__ void w8a16_gemv_batch16_ncol4_strided(
    const __nv_bfloat16* __restrict__ A,
    const unsigned char* __restrict__ B,
    const float* __restrict__ block_scale,
    __nv_bfloat16* __restrict__ C,
    unsigned int M,
    unsigned int N,
    unsigned int K,
    unsigned int a_row_stride,
    unsigned int c_row_stride
) {
    w8a16_gemv_ncol_impl<16, 4>(A, B, block_scale, C, M, N, K, a_row_stride, c_row_stride);
}
