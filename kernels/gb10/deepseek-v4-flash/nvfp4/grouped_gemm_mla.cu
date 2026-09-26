// SPDX-License-Identifier: MIT OR Apache-2.0

// 2026-09-25: Grouped GEMM for MLA: G independent GEMMs in one launch,
//   C_g[M, N_g] = A_g[M, K_g] @ B_g[N_g, K_g]^T   for g = 0..G-1,
// with A_g at column g * K_g of each A row, B_g at row g * N_g of B and C_g at column g * N_g of each C
// row: a block-diagonal GEMM without the zero blocks. A_stride and C_stride are the row pitches of A and C
// in elements. qwen3_attention/init.rs looks the kernel up as `grouped_gemm_mla_k`; no forward path
// launches that handle.
// Owner: gb10 kernels (deepseek-v4-flash). Invariants: none beyond the types.















#include <cuda_bf16.h>

// 2026-09-25: A block computes GG_TILE_N outputs of one (token, group) row, 64 threads reducing K_g for each.
// Grid: (M * G, ceil(N_g / GG_TILE_N), 1)
// Block: (256, 1, 1)


#define GG_BLOCK 256
#define GG_TILE_N 4

extern "C" __global__ void grouped_gemm_mla(
    const __nv_bfloat16* __restrict__ A,
    const __nv_bfloat16* __restrict__ B,
    __nv_bfloat16* __restrict__ C,
    unsigned int M,
    unsigned int G,
    unsigned int K_g,
    unsigned int N_g,
    unsigned int A_stride,
    unsigned int C_stride
) {

    unsigned int mg_idx = blockIdx.x;
    unsigned int n_tile = blockIdx.y;
    unsigned int token = mg_idx / G;
    unsigned int group = mg_idx % G;

    if (token >= M) return;

    unsigned int tid = threadIdx.x;



    const unsigned int threads_per_out = GG_BLOCK / GG_TILE_N;
    const unsigned int local_n = tid / threads_per_out;
    const unsigned int k_lane = tid % threads_per_out;

    unsigned int n_idx = n_tile * GG_TILE_N + local_n;
    if (n_idx >= N_g) return;


    const __nv_bfloat16* A_row = A + (unsigned long long)token * A_stride + group * K_g;
    const __nv_bfloat16* B_row = B + (unsigned long long)(group * N_g + n_idx) * K_g;


    float acc = 0.0f;
    for (unsigned int k = k_lane; k < K_g; k += threads_per_out) {
        float a_val = __bfloat162float(A_row[k]);
        float b_val = __bfloat162float(B_row[k]);
        acc += a_val * b_val;
    }

    // 2026-09-25: An output's 64 threads are two warps: each warp shuffle-reduces, then the two partials add.

    for (int offset = 16; offset > 0; offset >>= 1) {
        acc += __shfl_down_sync(0xFFFFFFFF, acc, offset);
    }


    __shared__ float s_partial[GG_TILE_N][2];
    unsigned int warp_in_out = k_lane / 32;
    unsigned int lane_in_warp = k_lane % 32;
    if (lane_in_warp == 0) {
        s_partial[local_n][warp_in_out] = acc;
    }
    __syncthreads();


    if (k_lane == 0) {
        float sum = s_partial[local_n][0] + s_partial[local_n][1];
        C[(unsigned long long)token * C_stride + group * N_g + n_idx] = __float2bfloat16(sum);
    }
}
