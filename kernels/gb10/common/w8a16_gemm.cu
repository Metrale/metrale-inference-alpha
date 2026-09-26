// SPDX-License-Identifier: MIT OR Apache-2.0

// 2026-09-25: W8A16 GEMM for M > 1, and a standalone dequant: FP8 E4M3 weights with 128 x 128
// FP32 block scales, BF16 activations, BF16 tensor-core MMA (`mma.sync.m16n8k16`, FP32
// accumulate).
//   C[m, n] = sum_k A[m, k] * E4M3(B[n, k]) * block_scale[n / 128, k / 128]
//
// Owner: gb10 kernels.
// Invariants:
// - A [M, K] BF16, B [N, K] E4M3 bytes, C [M, N] BF16. Both kernels read block_scale at
//   (n / 128) * (K / 128) + k / 128, a row pitch of K / 128 rounded down, so they assume
//   K % 128 == 0.
// - w8a16_gemm: grid (ceil(N / 64), ceil(M / 64), 1), block 128 (ops::w8a16_gemm). Only C[m, n]
//   with m < M and n < N is written.
// - w8a16_dequant writes B_bf16[i] for i < N * K, one element per thread.


#include <cuda_bf16.h>
// 2026-09-25: E4M3_LUT, the 256-entry E4M3 -> FP32 table, shared with the other gb10 W8A16 GEMMs.



#include "e4m3_lut.cuh"

#define M_TILE 64
#define N_TILE 64
#define K_STEP 16
#define PAD 2
#define FP8_BLOCK 128

// 2026-09-25: One K_STEP of MMAs for a warp: its 16 rows of smem_A times the 64 columns of
// smem_B (eight n8 tiles), accumulated into acc. It writes nothing to global memory.

__device__ __forceinline__ void w8a16_mma_and_store(
    __nv_bfloat16 smem_A[][K_STEP + PAD],
    __nv_bfloat16 smem_B[][N_TILE + PAD],
    float acc[8][4],
    unsigned int warp_m_offset, unsigned int group_id, unsigned int tid
) {
    const unsigned int a_stride = K_STEP + PAD;
    const unsigned int b_stride = N_TILE + PAD;
    const unsigned short* sA = (const unsigned short*)smem_A;
    const unsigned short* sB = (const unsigned short*)smem_B;

    unsigned int frag_r0 = warp_m_offset + group_id;
    unsigned int frag_r1 = warp_m_offset + group_id + 8;
    unsigned int frag_c0 = tid * 2;
    unsigned int frag_c1 = tid * 2 + 8;

    unsigned int a0 = *(const unsigned int*)&sA[frag_r0 * a_stride + frag_c0];
    unsigned int a1 = *(const unsigned int*)&sA[frag_r1 * a_stride + frag_c0];
    unsigned int a2 = *(const unsigned int*)&sA[frag_r0 * a_stride + frag_c1];
    unsigned int a3 = *(const unsigned int*)&sA[frag_r1 * a_stride + frag_c1];

    #pragma unroll
    for (int n_tile = 0; n_tile < 8; n_tile++) {
        unsigned int n_col = n_tile * 8 + group_id;
        unsigned int k0 = tid * 2;
        unsigned int k1 = tid * 2 + 8;

        unsigned int b0 = ((unsigned int)sB[(k0 + 1) * b_stride + n_col] << 16) |
                          (unsigned int)sB[k0 * b_stride + n_col];
        unsigned int b1 = ((unsigned int)sB[(k1 + 1) * b_stride + n_col] << 16) |
                          (unsigned int)sB[k1 * b_stride + n_col];

        asm volatile(
            "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
            "{%0, %1, %2, %3}, "
            "{%4, %5, %6, %7}, "
            "{%8, %9}, "
            "{%10, %11, %12, %13};"
            : "=f"(acc[n_tile][0]), "=f"(acc[n_tile][1]),
              "=f"(acc[n_tile][2]), "=f"(acc[n_tile][3])
            : "r"(a0), "r"(a1), "r"(a2), "r"(a3),
              "r"(b0), "r"(b1),
              "f"(acc[n_tile][0]), "f"(acc[n_tile][1]),
              "f"(acc[n_tile][2]), "f"(acc[n_tile][3])
        );
    }
}





extern "C" __global__ void w8a16_gemm(
    const __nv_bfloat16* __restrict__ A,
    const unsigned char* __restrict__ B,
    const float* __restrict__ block_scale,
    __nv_bfloat16* __restrict__ C,
    unsigned int M,
    unsigned int N,
    unsigned int K
) {
    const unsigned int cta_m = blockIdx.y * M_TILE;
    const unsigned int cta_n = blockIdx.x * N_TILE;
    const unsigned int warp_id = threadIdx.x / 32;
    const unsigned int lane_id = threadIdx.x % 32;
    const unsigned int warp_m_offset = warp_id * 16;
    const unsigned int group_id = lane_id >> 2;
    const unsigned int tid = lane_id & 3;

    __shared__ __nv_bfloat16 smem_A[M_TILE][K_STEP + PAD];
    __shared__ __nv_bfloat16 smem_B[K_STEP][N_TILE + PAD];

    // 2026-09-25: Two-level FP32 accumulation. inner_acc sums the MMA steps of one 128-wide K block
    // over unscaled weights: E4M3 converts to BF16 exactly (3 mantissa bits into 7). At each block
    // boundary inner_acc times the block's FP32 scale is added to outer_acc and inner_acc is reset,
    // so the scale never meets a BF16 rounding.
    //
    // n_block is constant per CTA: N_TILE (64) divides FP8_BLOCK (128) and cta_n is a multiple of
    // N_TILE, so all columns of a CTA share one n-block.




    float inner_acc[8][4];
    float outer_acc[8][4];
    #pragma unroll
    for (int i = 0; i < 8; i++) {
        inner_acc[i][0] = 0.0f; inner_acc[i][1] = 0.0f;
        inner_acc[i][2] = 0.0f; inner_acc[i][3] = 0.0f;
        outer_acc[i][0] = 0.0f; outer_acc[i][1] = 0.0f;
        outer_acc[i][2] = 0.0f; outer_acc[i][3] = 0.0f;
    }

    const unsigned int k_blocks = K / FP8_BLOCK;
    const unsigned int k_steps_per_block = FP8_BLOCK / K_STEP;
    const unsigned int n_block = cta_n / FP8_BLOCK;
    unsigned int k_step_in_block = 0;

    for (unsigned int k_base = 0; k_base < K; k_base += K_STEP) {
        // 2026-09-25: A tile [M_TILE, K_STEP] BF16 to shared memory; out-of-range elements are 0.
        {
            const unsigned int elems_per_thread = (M_TILE * K_STEP) / 128;
            #pragma unroll
            for (unsigned int i = 0; i < elems_per_thread; i++) {
                unsigned int idx = threadIdx.x * elems_per_thread + i;
                unsigned int row = idx / K_STEP;
                unsigned int col = idx % K_STEP;
                unsigned int gr = cta_m + row;
                unsigned int gc = k_base + col;
                smem_A[row][col] = (gr < M && gc < K) ? A[gr * K + gc] : __float2bfloat16(0.0f);
            }
        }

        // 2026-09-25: B tile [K_STEP, N_TILE], 16 x 64 = 1024 elements, 8 per thread: E4M3 to BF16
        // through E4M3_LUT with no scale. The conversion is exact; the scale is applied to the
        // FP32 accumulator below.

        {
            #pragma unroll
            for (unsigned int i = 0; i < 8; i++) {
                unsigned int idx = threadIdx.x * 8 + i;
                unsigned int k = idx / N_TILE;
                unsigned int n = idx % N_TILE;
                unsigned int gk = k_base + k;
                unsigned int gn = cta_n + n;

                if (gk < K && gn < N) {
                    unsigned char weight_byte = B[(unsigned long long)gn * K + gk];
                    smem_B[k][n] = __float2bfloat16(E4M3_LUT[weight_byte]);
                } else {
                    smem_B[k][n] = __float2bfloat16(0.0f);
                }
            }
        }

        __syncthreads();
        w8a16_mma_and_store(smem_A, smem_B, inner_acc, warp_m_offset, group_id, tid);
        __syncthreads();

        // 2026-09-25: End of a 128-wide K block: add the scaled inner sum to outer_acc, reset inner.
        k_step_in_block++;
        if (k_step_in_block == k_steps_per_block) {
            const unsigned int k_block = k_base / FP8_BLOCK;
            const float scale = block_scale[n_block * k_blocks + k_block];
            #pragma unroll
            for (int i = 0; i < 8; i++) {
                outer_acc[i][0] += inner_acc[i][0] * scale;
                outer_acc[i][1] += inner_acc[i][1] * scale;
                outer_acc[i][2] += inner_acc[i][2] * scale;
                outer_acc[i][3] += inner_acc[i][3] * scale;
                inner_acc[i][0] = 0.0f; inner_acc[i][1] = 0.0f;
                inner_acc[i][2] = 0.0f; inner_acc[i][3] = 0.0f;
            }
            k_step_in_block = 0;
        }
    }

    // 2026-09-25: Fold a trailing partial K block (K % 128 != 0). Its scale is read at
    // n_block * (K / 128) + (K - 1) / 128, with the same rounded-down pitch as the full blocks.

    if (k_step_in_block != 0) {
        const unsigned int k_block = (K - 1) / FP8_BLOCK;
        const float scale = block_scale[n_block * k_blocks + k_block];
        #pragma unroll
        for (int i = 0; i < 8; i++) {
            outer_acc[i][0] += inner_acc[i][0] * scale;
            outer_acc[i][1] += inner_acc[i][1] * scale;
            outer_acc[i][2] += inner_acc[i][2] * scale;
            outer_acc[i][3] += inner_acc[i][3] * scale;
        }
    }


    #pragma unroll
    for (int n_tile = 0; n_tile < 8; n_tile++) {
        unsigned int base_n = cta_n + n_tile * 8;
        unsigned int col0 = base_n + (tid * 2);
        unsigned int col1 = col0 + 1;
        unsigned int row0 = cta_m + warp_m_offset + group_id;
        unsigned int row1 = row0 + 8;

        if (row0 < M && col0 < N) C[row0 * N + col0] = __float2bfloat16(outer_acc[n_tile][0]);
        if (row0 < M && col1 < N) C[row0 * N + col1] = __float2bfloat16(outer_acc[n_tile][1]);
        if (row1 < M && col0 < N) C[row1 * N + col0] = __float2bfloat16(outer_acc[n_tile][2]);
        if (row1 < M && col1 < N) C[row1 * N + col1] = __float2bfloat16(outer_acc[n_tile][3]);
    }
}

// 2026-09-25: B_bf16[n, k] = E4M3(B[n, k]) * block_scale[n / 128, k / 128], one element per thread.

extern "C" __global__ void w8a16_dequant(
    const unsigned char* __restrict__ B,
    const float* __restrict__ block_scale,
    __nv_bfloat16* __restrict__ B_bf16,
    unsigned int K,
    unsigned int N
) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int total = N * K;
    if (idx >= total) return;

    unsigned int n = idx / K;
    unsigned int k = idx % K;

    unsigned char weight_byte = B[idx];

    unsigned int k_blocks = K / FP8_BLOCK;
    unsigned int n_block = n / FP8_BLOCK;
    unsigned int k_block = k / FP8_BLOCK;
    float scale = block_scale[n_block * k_blocks + k_block];

    float val = E4M3_LUT[weight_byte] * scale;
    B_bf16[idx] = __float2bfloat16(val);
}
