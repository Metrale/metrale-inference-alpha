// SPDX-License-Identifier: MIT OR Apache-2.0

// 2026-09-25: 32-row M-tile twin of w8a16_gemm_pipelined, with caller-supplied A and C row
// pitches. FP8 E4M3 weights with 128 x 128 FP32 block scales, BF16 activations, BF16
// tensor-core MMA (`mma.sync.m16n8k16`, FP32 accumulate).
//   C[r, n] = sum_k A[r, k] * E4M3(B[n, k]) * block_scale[n / 128, k / 128],  r < M, n < N
//
// Owner: gb10 kernels.
// Invariants:
// - A [M, lda] BF16 (the first K of each row are read), B [N, K] E4M3 bytes, block_scale
//   [N / 128, K / 128] FP32, C [M, ldc] BF16 (the first N of each row are written). The caller
//   guarantees K a positive multiple of 128, lda >= K with lda % 8 == 0, and ldc >= N;
//   ops::w8a16_gemm_pipelined_m32_strided refuses a launch otherwise.
// - Grid (ceil(N / 32), ceil(M / 32), 1), block 256. Only C[r, n] with r < M and n < N is
//   written; A rows at or past M are zero-filled in shared memory.
// - Each output gets the same m16n8k16 sub-MMAs over the same 16-wide K windows in ascending
//   order, folded with the block scale after every 128 K, as in w8a16_gemm_pipelined; the
//   model-arch example native_fp8_gdn_proj_m32_microtest compares the two byte for byte.
//
// At M <= 32 the 128-row tile of w8a16_gemm_pipelined is at least 75% zero rows. Measured
// 2026-09-22 to 2026-09-23 on GB10 at M = 32: w8a16_gemm_pipelined took 296 us for N = 12288,
// K = 2048 (25.2 MB of weight, 85 GB/s), and 166 us for N = 2048, K = 4096, where 64 CTAs on
// 48 SMs each run 128 K-steps. This twin changes the tile, not the math:
//   * PM32_M_TILE = 32: the 8 warps are laid out 2 (M) x 4 (N), each owning a 16 x 8 slab.
//   * PM32_K_STEP = 128, one scale block per K-step: a CTA runs K / 128 steps of three
//     __syncthreads() rather than K / 32, and folds the scale once per step.
//   * lda and ldc let the multi-seq attention Q/K/V tier (qkv_fp8_batch.rs) write each row's
//     Q, K and V straight into the per-sequence decode buffer.
//
// Static shared memory: A 2 x 32 x 136 x 2 B = 17,408; B as BF16 ([n][k], K-contiguous, +8 pad)
// 17,408; raw B 2 x 32 x 128 = 8,192; LUT 1,024; 44,032 B in all, under the 48 KiB static
// limit. Both padded strides are 68 words, 68 mod 32 = 4, so the eight group_id rows of an MMA
// fragment read land on banks 0, 4, ..., 28 (+ tid) without conflicts, as the 20-word A rows of
// w8a16_gemm_pipelined do.




























#include <cuda_bf16.h>
#include "e4m3_lut.cuh"
// 2026-09-25: PM32_PAD (8 BF16) keeps every A and B row 16-byte aligned: 136 BF16 = 272 bytes.
#define PM32_M_TILE 32
#define PM32_N_TILE 32
#define PM32_K_STEP 128
#define PM32_K_SUB 16
#define PM32_K_SUBS (PM32_K_STEP / PM32_K_SUB)
#define PM32_PAD 8
#define PM32_A_STRIDE (PM32_K_STEP + PM32_PAD)
#define PM32_B_STRIDE (PM32_K_STEP + PM32_PAD)
#define PM32_FP8_BLOCK 128
#define PM32_WARPS 8
#define PM32_THREADS (PM32_WARPS * 32)
#define PM32_WARPS_M (PM32_M_TILE / 16)
#define PM32_WARPS_N (PM32_WARPS / PM32_WARPS_M)
#define PM32_STAGES 2

__device__ __forceinline__ void pm32_cp_async_cg_16(void* smem_ptr, const void* gmem_ptr) {
    unsigned int s = (unsigned int)__cvta_generic_to_shared(smem_ptr);
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" ::"r"(s), "l"(gmem_ptr));
}
__device__ __forceinline__ void pm32_cp_async_commit() {
    asm volatile("cp.async.commit_group;\n" ::);
}
template <int N>
__device__ __forceinline__ void pm32_cp_async_wait_group() {
    asm volatile("cp.async.wait_group %0;\n" ::"n"(N));
}
// 2026-09-25: Wait until at most n cp.async groups are in flight; an n of 2 or more waits for at
// most 2, which is never looser than asked.
__device__ __forceinline__ void pm32_cp_async_wait_le(unsigned int n) {
    switch (n) {
        case 0:  pm32_cp_async_wait_group<0>(); break;
        case 1:  pm32_cp_async_wait_group<1>(); break;
        default: pm32_cp_async_wait_group<2>(); break;
    }
}

// 2026-09-25: One K-step: PM32_K_SUBS m16n8k16 sub-MMAs over the warp's 16 x 8 slab, accumulated
// into inner[4]. The fragment addressing is w8a16_gemm_pipelined's pm_mma_kstep with one n8 tile
// per warp and the warp's column offset added to n_col.

__device__ __forceinline__ void pm32_mma_kstep(
    const __nv_bfloat16* smem_A,
    const __nv_bfloat16* smem_B,
    float inner[4],
    unsigned int warp_m_offset, unsigned int warp_n_offset,
    unsigned int group_id, unsigned int tid
) {
    const unsigned short* sA = (const unsigned short*)smem_A;
    const unsigned short* sB = (const unsigned short*)smem_B;
    const unsigned int frag_r0 = warp_m_offset + group_id;
    const unsigned int frag_r1 = frag_r0 + 8;
    const unsigned int n_col = warp_n_offset + group_id;

    #pragma unroll
    for (int s = 0; s < PM32_K_SUBS; s++) {
        const unsigned int k_off = s * PM32_K_SUB;
        const unsigned int c0 = k_off + tid * 2;
        const unsigned int c1 = c0 + 8;

        unsigned int a0 = *(const unsigned int*)&sA[frag_r0 * PM32_A_STRIDE + c0];
        unsigned int a1 = *(const unsigned int*)&sA[frag_r1 * PM32_A_STRIDE + c0];
        unsigned int a2 = *(const unsigned int*)&sA[frag_r0 * PM32_A_STRIDE + c1];
        unsigned int a3 = *(const unsigned int*)&sA[frag_r1 * PM32_A_STRIDE + c1];
        unsigned int b0 = *(const unsigned int*)&sB[n_col * PM32_B_STRIDE + c0];
        unsigned int b1 = *(const unsigned int*)&sB[n_col * PM32_B_STRIDE + c1];

        asm volatile(
            "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
            "{%0, %1, %2, %3}, "
            "{%4, %5, %6, %7}, "
            "{%8, %9}, "
            "{%10, %11, %12, %13};"
            : "=f"(inner[0]), "=f"(inner[1]), "=f"(inner[2]), "=f"(inner[3])
            : "r"(a0), "r"(a1), "r"(a2), "r"(a3),
              "r"(b0), "r"(b1),
              "f"(inner[0]), "f"(inner[1]), "f"(inner[2]), "f"(inner[3])
        );
    }
}

// 2026-09-25: M may be any value: A rows past M in the last M tile are zero-filled and never stored.


extern "C" __global__ void w8a16_gemm_pipelined_m32(
    const __nv_bfloat16* __restrict__ A,
    const unsigned char* __restrict__ B,
    const float* __restrict__ block_scale,
    __nv_bfloat16* __restrict__ C,
    unsigned int M,
    unsigned int N,
    unsigned int K,
    unsigned int lda,
    unsigned int ldc
) {
    const unsigned int cta_m = blockIdx.y * PM32_M_TILE;
    const unsigned int cta_n = blockIdx.x * PM32_N_TILE;
    const unsigned int warp_id = threadIdx.x / 32;
    const unsigned int lane_id = threadIdx.x % 32;
    const unsigned int warp_m_offset = (warp_id % PM32_WARPS_M) * 16;
    const unsigned int warp_n_offset = (warp_id / PM32_WARPS_M) * 8;
    const unsigned int group_id = lane_id >> 2;
    const unsigned int tid = lane_id & 3;

    __shared__ __align__(16) __nv_bfloat16 smem_A[PM32_STAGES][PM32_M_TILE][PM32_A_STRIDE];
    __shared__ __align__(16) __nv_bfloat16 smem_B[PM32_STAGES][PM32_N_TILE][PM32_B_STRIDE];
    __shared__ __align__(16) unsigned char smem_Braw[PM32_STAGES][PM32_N_TILE][PM32_K_STEP];
    __shared__ float smem_lut[256];
    smem_lut[threadIdx.x] = E4M3_LUT[threadIdx.x];
    __syncthreads();

    float inner_acc[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    float outer_acc[4] = {0.0f, 0.0f, 0.0f, 0.0f};

    const unsigned int k_blocks = K / PM32_FP8_BLOCK;
    const unsigned int n_block = cta_n / PM32_FP8_BLOCK;
    const unsigned int n_steps = K / PM32_K_STEP;

    // 2026-09-25: A: 32 rows x 128 K BF16 = 512 16-byte chunks (2 per thread).
    // B: 32 rows x 128 K E4M3 = 256 16-byte chunks (1 per thread).
    const unsigned int a_chunks = (PM32_M_TILE * PM32_K_STEP) / 8;
    const unsigned int b_chunks = (PM32_N_TILE * PM32_K_STEP) / 16;

    auto prefetch = [&](unsigned int step, unsigned int stage) {
        const unsigned int k_base = step * PM32_K_STEP;
        #pragma unroll
        for (unsigned int c = threadIdx.x; c < a_chunks; c += PM32_THREADS) {
            const unsigned int row = (c * 8) / PM32_K_STEP;
            const unsigned int col = (c * 8) % PM32_K_STEP;
            const unsigned int gr = cta_m + row;
            const unsigned int gc = k_base + col;
            __nv_bfloat16* dst = &smem_A[stage][row][col];
            if (gr < M) {
                pm32_cp_async_cg_16(dst, &A[(unsigned long long)gr * lda + gc]);
            } else {
                #pragma unroll
                for (unsigned int e = 0; e < 8; e++) dst[e] = __float2bfloat16(0.0f);
            }
        }
        #pragma unroll
        for (unsigned int c = threadIdx.x; c < b_chunks; c += PM32_THREADS) {
            const unsigned int nrow = (c * 16) / PM32_K_STEP;
            const unsigned int kcol = (c * 16) % PM32_K_STEP;
            const unsigned int gn = cta_n + nrow;
            const unsigned int gk = k_base + kcol;
            unsigned char* dst = &smem_Braw[stage][nrow][kcol];
            if (gn < N) {
                pm32_cp_async_cg_16(dst, &B[(unsigned long long)gn * K + gk]);
            } else {
                #pragma unroll
                for (unsigned int e = 0; e < 16; e++) dst[e] = 0;
            }
        }
        pm32_cp_async_commit();
    };

    // 2026-09-25: Convert the arrived raw step (32 x 128 bytes, 16 per thread) to BF16 in the same
    // layout, with no scale; the scale is applied to the FP32 accumulator.
    auto dequant_B = [&](unsigned int stage) {
        #pragma unroll
        for (unsigned int idx = threadIdx.x; idx < PM32_K_STEP * PM32_N_TILE; idx += PM32_THREADS) {
            const unsigned int n = idx / PM32_K_STEP;
            const unsigned int k = idx % PM32_K_STEP;
            smem_B[stage][n][k] = __float2bfloat16(smem_lut[smem_Braw[stage][n][k]]);
        }
    };

    #pragma unroll
    for (unsigned int p = 0; p < PM32_STAGES - 1; p++) {
        if (p < n_steps) prefetch(p, p % PM32_STAGES);
    }

    for (unsigned int step = 0; step < n_steps; step++) {
        const unsigned int cur = step % PM32_STAGES;
        const unsigned int ahead = step + (PM32_STAGES - 1);
        if (ahead < n_steps) prefetch(ahead, ahead % PM32_STAGES);
        // 2026-09-25: Groups complete in commit order; min(n_steps, STAGES + step) - (step + 1) may
        // stay in flight once `cur`, the oldest, is complete. w8a16_gemm_pipelined uses the same rule.

        const unsigned int committed = min(n_steps, PM32_STAGES + step);
        pm32_cp_async_wait_le(committed - (step + 1));
        __syncthreads();

        dequant_B(cur);
        __syncthreads();

        pm32_mma_kstep(&smem_A[cur][0][0], &smem_B[cur][0][0],
                       inner_acc, warp_m_offset, warp_n_offset, group_id, tid);
        __syncthreads();

        // 2026-09-25: One K-step is one 128-wide scale block, so every step folds.
        const float scale = block_scale[n_block * k_blocks + step];
        #pragma unroll
        for (int i = 0; i < 4; i++) {
            outer_acc[i] += inner_acc[i] * scale;
            inner_acc[i] = 0.0f;
        }
    }

    const unsigned int col0 = cta_n + warp_n_offset + tid * 2;
    const unsigned int col1 = col0 + 1;
    const unsigned int row0 = cta_m + warp_m_offset + group_id;
    const unsigned int row1 = row0 + 8;
    if (row0 < M && col0 < N) C[(unsigned long long)row0 * ldc + col0] = __float2bfloat16(outer_acc[0]);
    if (row0 < M && col1 < N) C[(unsigned long long)row0 * ldc + col1] = __float2bfloat16(outer_acc[1]);
    if (row1 < M && col0 < N) C[(unsigned long long)row1 * ldc + col0] = __float2bfloat16(outer_acc[2]);
    if (row1 < M && col1 < N) C[(unsigned long long)row1 * ldc + col1] = __float2bfloat16(outer_acc[3]);
}
