// SPDX-License-Identifier: MIT OR Apache-2.0

// 2026-09-25: W8A16 GEMMs over a load-time transposed FP8 E4M3 weight B_t [K, N], so weight rows
// are read along N, the contiguous axis, and the two kernels that build B_t and its block scales
// from the checkpoint's [N, K] layout.
//   C[m, n] = sum_k A[m, k] * E4M3(B_t[k, n]) * block_scale_t[k / 128, n / 128]
//
// Owner: gb10 kernels.
// Invariants:
// - A [M, K] BF16, C [M, N] BF16, block_scale_t [ceil(K / 128), ceil(N / 128)] FP32 as
//   transpose_block_scale writes it. Only C[m, n] with m < M and n < N is written.
// - w8a16_gemm_t: grid (ceil(N / 64), ceil(M / 64), 1), block 128. w8a16_gemm_t_pipelined: grid
//   (ceil(N / 32), ceil(M / 128), 1), block 256 (ops::w8a16_gemm_t, ops::w8a16_gemm_t_pipelined).
// - Both apply each scale to the FP32 sum of one 128-wide K block of unscaled weights.



#include <cuda_bf16.h>

#define M_TILE 64
#define N_TILE 128
#define K_STEP 32
#define PAD 2
#define FP8_BLOCK 128

// 2026-09-25: E4M3 -> FP32 table of w8a16_gemm_t, the same 256 values as E4M3_LUT (e4m3_lut.cuh).
__device__ __constant__ float E4M3_LUT_T[256] = {
    0.0f, 0.001953125f, 0.00390625f, 0.005859375f,
    0.0078125f, 0.009765625f, 0.01171875f, 0.013671875f,
    0.015625f, 0.017578125f, 0.01953125f, 0.021484375f,
    0.0234375f, 0.025390625f, 0.02734375f, 0.029296875f,
    0.03125f, 0.03515625f, 0.0390625f, 0.04296875f,
    0.046875f, 0.05078125f, 0.0546875f, 0.05859375f,
    0.0625f, 0.0703125f, 0.078125f, 0.0859375f,
    0.09375f, 0.1015625f, 0.109375f, 0.1171875f,
    0.125f, 0.140625f, 0.15625f, 0.171875f,
    0.1875f, 0.203125f, 0.21875f, 0.234375f,
    0.25f, 0.28125f, 0.3125f, 0.34375f,
    0.375f, 0.40625f, 0.4375f, 0.46875f,
    0.5f, 0.5625f, 0.625f, 0.6875f,
    0.75f, 0.8125f, 0.875f, 0.9375f,
    1.0f, 1.125f, 1.25f, 1.375f,
    1.5f, 1.625f, 1.75f, 1.875f,
    2.0f, 2.25f, 2.5f, 2.75f,
    3.0f, 3.25f, 3.5f, 3.75f,
    4.0f, 4.5f, 5.0f, 5.5f,
    6.0f, 6.5f, 7.0f, 7.5f,
    8.0f, 9.0f, 10.0f, 11.0f,
    12.0f, 13.0f, 14.0f, 15.0f,
    16.0f, 18.0f, 20.0f, 22.0f,
    24.0f, 26.0f, 28.0f, 30.0f,
    32.0f, 36.0f, 40.0f, 44.0f,
    48.0f, 52.0f, 56.0f, 60.0f,
    64.0f, 72.0f, 80.0f, 88.0f,
    96.0f, 104.0f, 112.0f, 120.0f,
    128.0f, 144.0f, 160.0f, 176.0f,
    192.0f, 208.0f, 224.0f, 240.0f,
    256.0f, 288.0f, 320.0f, 352.0f,
    384.0f, 416.0f, 448.0f, 0.0f,
    -0.0f, -0.001953125f, -0.00390625f, -0.005859375f,
    -0.0078125f, -0.009765625f, -0.01171875f, -0.013671875f,
    -0.015625f, -0.017578125f, -0.01953125f, -0.021484375f,
    -0.0234375f, -0.025390625f, -0.02734375f, -0.029296875f,
    -0.03125f, -0.03515625f, -0.0390625f, -0.04296875f,
    -0.046875f, -0.05078125f, -0.0546875f, -0.05859375f,
    -0.0625f, -0.0703125f, -0.078125f, -0.0859375f,
    -0.09375f, -0.1015625f, -0.109375f, -0.1171875f,
    -0.125f, -0.140625f, -0.15625f, -0.171875f,
    -0.1875f, -0.203125f, -0.21875f, -0.234375f,
    -0.25f, -0.28125f, -0.3125f, -0.34375f,
    -0.375f, -0.40625f, -0.4375f, -0.46875f,
    -0.5f, -0.5625f, -0.625f, -0.6875f,
    -0.75f, -0.8125f, -0.875f, -0.9375f,
    -1.0f, -1.125f, -1.25f, -1.375f,
    -1.5f, -1.625f, -1.75f, -1.875f,
    -2.0f, -2.25f, -2.5f, -2.75f,
    -3.0f, -3.25f, -3.5f, -3.75f,
    -4.0f, -4.5f, -5.0f, -5.5f,
    -6.0f, -6.5f, -7.0f, -7.5f,
    -8.0f, -9.0f, -10.0f, -11.0f,
    -12.0f, -13.0f, -14.0f, -15.0f,
    -16.0f, -18.0f, -20.0f, -22.0f,
    -24.0f, -26.0f, -28.0f, -30.0f,
    -32.0f, -36.0f, -40.0f, -44.0f,
    -48.0f, -52.0f, -56.0f, -60.0f,
    -64.0f, -72.0f, -80.0f, -88.0f,
    -96.0f, -104.0f, -112.0f, -120.0f,
    -128.0f, -144.0f, -160.0f, -176.0f,
    -192.0f, -208.0f, -224.0f, -240.0f,
    -256.0f, -288.0f, -320.0f, -352.0f,
    -384.0f, -416.0f, -448.0f, -0.0f,
};

// 2026-09-25: One 16-wide K step of MMAs for a warp: its 16 rows of smem_A [M_TILE][16 + PAD] times
// the 64 columns of smem_B [16][64 + PAD] (eight n8 tiles), accumulated into acc.
__device__ __forceinline__ void w8a16_mma_and_store_t(
    __nv_bfloat16 smem_A[][16 + PAD],
    __nv_bfloat16 smem_B[][64 + PAD],
    float acc[8][4],
    unsigned int warp_m_offset, unsigned int group_id, unsigned int tid
) {
    const unsigned int a_stride = 16 + PAD;
    const unsigned int b_stride = 64 + PAD;
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

// 2026-09-25: Each CTA covers 64 columns (cta_n = blockIdx.x * 64; the N_TILE and K_STEP defines are
// not used here) and 64 rows, with K steps of 16.





extern "C" __global__ void w8a16_gemm_t(
    const __nv_bfloat16* __restrict__ A,
    const unsigned char* __restrict__ B_t,
    const float* __restrict__ block_scale_t,
    __nv_bfloat16* __restrict__ C,
    unsigned int M,
    unsigned int N,
    unsigned int K
) {
    const unsigned int cta_m = blockIdx.y * M_TILE;
    const unsigned int cta_n = blockIdx.x * 64;
    const unsigned int warp_id = threadIdx.x / 32;
    const unsigned int lane_id = threadIdx.x % 32;
    const unsigned int warp_m_offset = warp_id * 16;
    const unsigned int group_id = lane_id >> 2;
    const unsigned int tid = lane_id & 3;

    __shared__ __nv_bfloat16 smem_A[M_TILE][16 + PAD];
    __shared__ __nv_bfloat16 smem_B[16][64 + PAD];

    // 2026-09-25: Two-level FP32 accumulation, as in w8a16_gemm. n_block is constant per CTA: 64
    // columns lie inside one 128-wide scale block because cta_n is a multiple of 64.

    float inner_acc[8][4];
    float outer_acc[8][4];
    #pragma unroll
    for (int i = 0; i < 8; i++) {
        inner_acc[i][0] = 0.0f; inner_acc[i][1] = 0.0f;
        inner_acc[i][2] = 0.0f; inner_acc[i][3] = 0.0f;
        outer_acc[i][0] = 0.0f; outer_acc[i][1] = 0.0f;
        outer_acc[i][2] = 0.0f; outer_acc[i][3] = 0.0f;
    }

    const unsigned int n_scale_blocks = (N + FP8_BLOCK - 1) / FP8_BLOCK;
    const unsigned int k_step_inner = 16;
    const unsigned int k_steps_per_block = FP8_BLOCK / k_step_inner;
    const unsigned int n_block = cta_n / FP8_BLOCK;
    unsigned int k_step_in_block = 0;

    for (unsigned int k_base = 0; k_base < K; k_base += 16) {
        // 2026-09-25: A tile [M_TILE, 16] BF16 to shared memory, 8 per thread; out-of-range elements are 0.
        {

            #pragma unroll
            for (unsigned int i = 0; i < 8; i++) {
                unsigned int idx = threadIdx.x * 8 + i;
                unsigned int row = idx / 16;
                unsigned int col = idx % 16;
                unsigned int gr = cta_m + row;
                unsigned int gc = k_base + col;
                smem_A[row][col] = (gr < M && gc < K) ? A[gr * K + gc] : __float2bfloat16(0.0f);
            }
        }

        // 2026-09-25: B tile [16, 64] from B_t, 8 elements per thread, each thread reading 8 consecutive
        // N bytes of one K row. The E4M3 value is stored as BF16 exactly; the block scale is applied
        // to the FP32 accumulator at each 128-wide K block boundary below.

        {
            #pragma unroll
            for (unsigned int i = 0; i < 8; i++) {
                unsigned int idx = threadIdx.x * 8 + i;
                unsigned int k = idx / 64;
                unsigned int n = idx % 64;
                unsigned int gk = k_base + k;
                unsigned int gn = cta_n + n;

                if (gk < K && gn < N) {

                    unsigned char weight_byte = B_t[(unsigned long long)gk * N + gn];
                    smem_B[k][n] = __float2bfloat16(E4M3_LUT_T[weight_byte]);
                } else {
                    smem_B[k][n] = __float2bfloat16(0.0f);
                }
            }
        }

        __syncthreads();
        w8a16_mma_and_store_t(smem_A, smem_B, inner_acc, warp_m_offset, group_id, tid);
        __syncthreads();

        // 2026-09-25: End of a 128-wide K block: add the scaled inner sum to outer_acc, reset inner.
        k_step_in_block++;
        if (k_step_in_block == k_steps_per_block) {
            const unsigned int k_block = k_base / FP8_BLOCK;

            const float scale = block_scale_t[k_block * n_scale_blocks + n_block];
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

    // 2026-09-25: Fold a trailing partial K block (K % 128 != 0) with scale row (K - 1) / 128.

    if (k_step_in_block != 0) {
        const unsigned int k_block = (K - 1) / FP8_BLOCK;
        const float scale = block_scale_t[k_block * n_scale_blocks + n_block];
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

// 2026-09-25: w8a16_gemm_t_pipelined: the transposed contract of w8a16_gemm_t with the structure
// of w8a16_gemm_pipelined: a 128 x 32 output tile over 8 warps, PT_K_STEP = 32 (two m16n8k16
// sub-MMAs per step), the E4M3 table staged in shared memory, a K-contiguous smem_B, and a
// PT_STAGES-deep cp.async pipeline.
//
// B_t is N-contiguous but each MMA B register holds a K pair. Each K-step is copied in 16-byte
// cp.async chunks along N into smem_Braw [k][n], and the dequant, which touches every element
// anyway, writes the transposed smem_B [n][k], so each (k, k + 1) register is one 32-bit load.
//
// Two-level FP32 accumulation: inner sums the unscaled weights of the 4 K-steps (8 sub-MMAs) of
// one 128-wide K block; at each block boundary outer += inner * block_scale_t.
//
// A 16-byte cp.async needs an aligned source: K % 8 == 0 for A and N % 16 == 0 for B_t (the
// scalar fallback covers only chunks that cross M, N or K). Measured 2026-09-25 (nvcc 13.0.88,
// sm_121f, --fmad=false): 64 registers, no spill, 27,904 B of static shared memory.





















#include "e4m3_lut.cuh"

#define PT_M_TILE 128
#define PT_N_TILE 32
#define PT_K_STEP 32
#define PT_K_SUB 16
#define PT_K_SUBS (PT_K_STEP / PT_K_SUB)
#define PT_PAD 2
// 2026-09-25: A-tile row stride: 32 K columns + 8 pad = 40 BF16 = 80 bytes, a multiple of 16, so
// every 16-byte cp.async destination is aligned; with 20-word rows the 32-bit A fragment reads
// of a warp fall in 32 distinct banks.

#define PT_A_STRIDE 40
#define PT_FP8_BLOCK 128
#define PT_WARPS 8
#define PT_THREADS (PT_WARPS * 32)
#define PT_N_TILES_PER_WARP (PT_N_TILE / 8)
#define PT_STAGES 2

// 2026-09-25: 16-byte cp.async.cg copy, global -> shared (cached in L2 only). Both addresses
// must be 16-byte aligned.
__device__ __forceinline__ void pt_cp_async_cg_16(void* smem_ptr, const void* gmem_ptr) {
    unsigned int s = (unsigned int)__cvta_generic_to_shared(smem_ptr);
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" ::"r"(s), "l"(gmem_ptr));
}
__device__ __forceinline__ void pt_cp_async_commit() {
    asm volatile("cp.async.commit_group;\n" ::);
}
template <int N>
__device__ __forceinline__ void pt_cp_async_wait_group() {
    asm volatile("cp.async.wait_group %0;\n" ::"n"(N));
}
__device__ __forceinline__ void pt_cp_async_wait_le(unsigned int n) {
    switch (n) {
        case 0:  pt_cp_async_wait_group<0>(); break;
        case 1:  pt_cp_async_wait_group<1>(); break;
        case 2:  pt_cp_async_wait_group<2>(); break;
        default: pt_cp_async_wait_group<3>(); break;
    }
}

// 2026-09-25: One K-step: PT_K_SUBS m16n8k16 sub-MMAs per n8 tile over the K-contiguous smem_B
// [n][k], with the fragment addressing of w8a16_gemm_pipelined's pm_mma_kstep.
__device__ __forceinline__ void pt_mma_kstep(
    const __nv_bfloat16* smem_A,
    const __nv_bfloat16* smem_B,
    float inner[PT_N_TILES_PER_WARP][4],
    unsigned int warp_m_offset, unsigned int group_id, unsigned int tid
) {
    const unsigned int a_stride = PT_A_STRIDE;
    const unsigned int b_stride = PT_K_STEP + PT_PAD;
    const unsigned short* sA = (const unsigned short*)smem_A;
    const unsigned short* sB = (const unsigned short*)smem_B;

    unsigned int frag_r0 = warp_m_offset + group_id;
    unsigned int frag_r1 = warp_m_offset + group_id + 8;

    #pragma unroll
    for (int s = 0; s < PT_K_SUBS; s++) {
        const unsigned int k_off = s * PT_K_SUB;
        unsigned int frag_c0 = k_off + tid * 2;
        unsigned int frag_c1 = k_off + tid * 2 + 8;

        unsigned int a0 = *(const unsigned int*)&sA[frag_r0 * a_stride + frag_c0];
        unsigned int a1 = *(const unsigned int*)&sA[frag_r1 * a_stride + frag_c0];
        unsigned int a2 = *(const unsigned int*)&sA[frag_r0 * a_stride + frag_c1];
        unsigned int a3 = *(const unsigned int*)&sA[frag_r1 * a_stride + frag_c1];

        #pragma unroll
        for (int n_tile = 0; n_tile < PT_N_TILES_PER_WARP; n_tile++) {
            unsigned int n_col = n_tile * 8 + group_id;
            unsigned int k0 = k_off + tid * 2;
            unsigned int k1 = k_off + tid * 2 + 8;


            unsigned int b0 = *(const unsigned int*)&sB[n_col * b_stride + k0];
            unsigned int b1 = *(const unsigned int*)&sB[n_col * b_stride + k1];

            asm volatile(
                "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
                "{%0, %1, %2, %3}, "
                "{%4, %5, %6, %7}, "
                "{%8, %9}, "
                "{%10, %11, %12, %13};"
                : "=f"(inner[n_tile][0]), "=f"(inner[n_tile][1]),
                  "=f"(inner[n_tile][2]), "=f"(inner[n_tile][3])
                : "r"(a0), "r"(a1), "r"(a2), "r"(a3),
                  "r"(b0), "r"(b1),
                  "f"(inner[n_tile][0]), "f"(inner[n_tile][1]),
                  "f"(inner[n_tile][2]), "f"(inner[n_tile][3])
            );
        }
    }
}




extern "C" __global__ void w8a16_gemm_t_pipelined(
    const __nv_bfloat16* __restrict__ A,
    const unsigned char* __restrict__ B_t,
    const float* __restrict__ block_scale_t,
    __nv_bfloat16* __restrict__ C,
    unsigned int M,
    unsigned int N,
    unsigned int K
) {
    const unsigned int cta_m = blockIdx.y * PT_M_TILE;
    const unsigned int cta_n = blockIdx.x * PT_N_TILE;
    const unsigned int warp_id = threadIdx.x / 32;
    const unsigned int lane_id = threadIdx.x % 32;
    const unsigned int warp_m_offset = warp_id * 16;
    const unsigned int group_id = lane_id >> 2;
    const unsigned int tid = lane_id & 3;

    // 2026-09-25: smem_A and smem_Braw are cp.async destinations: __align__(16), 16-byte row
    // strides. smem_B is the converted, K-contiguous buffer the MMAs read. Per stage: smem_A
    // 128 * 40 * 2 = 10240 B + smem_B 32 * 34 * 2 = 2176 B + smem_Braw 32 * 32 = 1024 B = 13440 B;
    // 26880 B for 2 stages, plus the 1 KiB table.
    __shared__ __align__(16) __nv_bfloat16 smem_A[PT_STAGES][PT_M_TILE][PT_A_STRIDE];
    __shared__ __nv_bfloat16 smem_B[PT_STAGES][PT_N_TILE][PT_K_STEP + PT_PAD];
    // 2026-09-25: Raw E4M3 bytes [k][n], N-contiguous like B_t.
    __shared__ __align__(16) unsigned char smem_Braw[PT_STAGES][PT_K_STEP][PT_N_TILE];

    // 2026-09-25: Stage E4M3_LUT in shared memory, one entry per thread (PT_THREADS = 256).
    __shared__ float smem_lut[256];
    smem_lut[threadIdx.x] = E4M3_LUT[threadIdx.x];
    __syncthreads();

    // 2026-09-25: Two-level FP32 accumulation, as described above.
    float inner_acc[PT_N_TILES_PER_WARP][4];
    float outer_acc[PT_N_TILES_PER_WARP][4];
    #pragma unroll
    for (int i = 0; i < PT_N_TILES_PER_WARP; i++) {
        inner_acc[i][0] = 0.0f; inner_acc[i][1] = 0.0f;
        inner_acc[i][2] = 0.0f; inner_acc[i][3] = 0.0f;
        outer_acc[i][0] = 0.0f; outer_acc[i][1] = 0.0f;
        outer_acc[i][2] = 0.0f; outer_acc[i][3] = 0.0f;
    }

    const unsigned int n_scale_blocks = (N + PT_FP8_BLOCK - 1) / PT_FP8_BLOCK;
    const unsigned int k_steps_per_block = PT_FP8_BLOCK / PT_K_STEP;
    const unsigned int n_block = cta_n / PT_FP8_BLOCK;
    const unsigned int n_steps = (K + PT_K_STEP - 1) / PT_K_STEP;

    // 2026-09-25: A tile: 128 rows x PT_K_STEP BF16, copied along K in 16-byte chunks.
    const unsigned int a_chunks = (PT_M_TILE * PT_K_STEP) / 8;

    // 2026-09-25: Issue the cp.async copies of K-step `step` into buffer `stage`.
    auto prefetch = [&](unsigned int step, unsigned int stage) {
        unsigned int k_base = step * PT_K_STEP;


        #pragma unroll
        for (unsigned int c = threadIdx.x; c < a_chunks; c += PT_THREADS) {
            unsigned int row = (c * 8) / PT_K_STEP;
            unsigned int col = (c * 8) % PT_K_STEP;
            unsigned int gr = cta_m + row;
            unsigned int gc = k_base + col;
            __nv_bfloat16* dst = &smem_A[stage][row][col];
            if (gr < M && gc + 8 <= K) {
                pt_cp_async_cg_16(dst, &A[(unsigned long long)gr * K + gc]);
            } else {
                #pragma unroll
                for (unsigned int e = 0; e < 8; e++) {
                    unsigned int gcol = gc + e;
                    dst[e] = (gr < M && gcol < K) ? A[(unsigned long long)gr * K + gcol]
                                                  : __float2bfloat16(0.0f);
                }
            }
        }

        // 2026-09-25: B_t: smem_Braw[stage][k][n] mirrors B_t[k_base + k, cta_n + n]; each 32-byte
        // K row of the tile is two 16-byte chunks, PT_K_STEP x PT_N_TILE / 16 chunks in all.


        const unsigned int b_chunks = (PT_K_STEP * PT_N_TILE) / 16;
        #pragma unroll
        for (unsigned int c = threadIdx.x; c < b_chunks; c += PT_THREADS) {
            unsigned int krow = (c * 16) / PT_N_TILE;
            unsigned int ncol = (c * 16) % PT_N_TILE;
            unsigned int gk = k_base + krow;
            unsigned int gn = cta_n + ncol;
            unsigned char* dst = &smem_Braw[stage][krow][ncol];
            if (gk < K && gn + 16 <= N) {
                pt_cp_async_cg_16(dst, &B_t[(unsigned long long)gk * N + gn]);
            } else {
                #pragma unroll
                for (unsigned int e = 0; e < 16; e++) {
                    unsigned int gne = gn + e;
                    dst[e] = (gk < K && gne < N) ? B_t[(unsigned long long)gk * N + gne] : 0;
                }
            }
        }
        pt_cp_async_commit();
    };

    // 2026-09-25: Convert the arrived raw B of `stage` through smem_lut and transpose it: read
    // smem_Braw[k][n], write smem_B[n][k]. No scale; it is applied to the FP32 accumulator.



    auto dequant_B = [&](unsigned int stage) {
        #pragma unroll
        for (unsigned int idx = threadIdx.x; idx < PT_K_STEP * PT_N_TILE; idx += PT_THREADS) {
            unsigned int k = idx / PT_N_TILE;
            unsigned int n = idx % PT_N_TILE;
            unsigned char wb = smem_Braw[stage][k][n];
            smem_B[stage][n][k] = __float2bfloat16(smem_lut[wb]);
        }
    };


    #pragma unroll
    for (unsigned int p = 0; p < PT_STAGES - 1; p++) {
        if (p < n_steps) {
            prefetch(p, p % PT_STAGES);
        }
    }
    unsigned int k_step_in_block = 0;

    for (unsigned int step = 0; step < n_steps; step++) {
        unsigned int cur = step % PT_STAGES;

        unsigned int ahead = step + (PT_STAGES - 1);
        if (ahead < n_steps) {
            prefetch(ahead, ahead % PT_STAGES);
        }
        unsigned int committed = min(n_steps, PT_STAGES + step);
        unsigned int target = committed - (step + 1);
        pt_cp_async_wait_le(target);
        __syncthreads();

        dequant_B(cur);
        __syncthreads();

        pt_mma_kstep(&smem_A[cur][0][0], &smem_B[cur][0][0],
                     inner_acc, warp_m_offset, group_id, tid);
        __syncthreads();

        // 2026-09-25: End of a 128-wide K block: add the scaled inner sum to outer_acc, reset inner.
        k_step_in_block++;
        if (k_step_in_block == k_steps_per_block) {
            const unsigned int k_block = (step * PT_K_STEP) / PT_FP8_BLOCK;

            const float scale = block_scale_t[k_block * n_scale_blocks + n_block];
            #pragma unroll
            for (int i = 0; i < PT_N_TILES_PER_WARP; i++) {
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

    // 2026-09-25: Fold a trailing partial K block (K % 128 != 0) with scale row (K - 1) / 128.
    if (k_step_in_block != 0) {
        const unsigned int k_block = (K - 1) / PT_FP8_BLOCK;
        const float scale = block_scale_t[k_block * n_scale_blocks + n_block];
        #pragma unroll
        for (int i = 0; i < PT_N_TILES_PER_WARP; i++) {
            outer_acc[i][0] += inner_acc[i][0] * scale;
            outer_acc[i][1] += inner_acc[i][1] * scale;
            outer_acc[i][2] += inner_acc[i][2] * scale;
            outer_acc[i][3] += inner_acc[i][3] * scale;
        }
    }


    #pragma unroll
    for (int n_tile = 0; n_tile < PT_N_TILES_PER_WARP; n_tile++) {
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

// 2026-09-25: B_t[k, n] = B[n, k] for the [N, K] E4M3 weight, one element per thread.

extern "C" __global__ void transpose_fp8(
    const unsigned char* __restrict__ B,
    unsigned char* __restrict__ B_t,
    unsigned int N,
    unsigned int K
) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int total = N * K;
    if (idx >= total) return;

    unsigned int n = idx / K;
    unsigned int k = idx % K;
    B_t[(unsigned long long)k * N + n] = B[(unsigned long long)n * K + k];
}

// 2026-09-25: scale_t[kb, nb] = scale[nb, kb] for FP32 block scales, one element per thread; the
// caller passes N_blocks = ceil(N / 128) and K_blocks = ceil(K / 128).
extern "C" __global__ void transpose_block_scale(
    const float* __restrict__ scale,
    float* __restrict__ scale_t,
    unsigned int N_blocks,
    unsigned int K_blocks
) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int total = N_blocks * K_blocks;
    if (idx >= total) return;

    unsigned int nb = idx / K_blocks;
    unsigned int kb = idx % K_blocks;
    scale_t[kb * N_blocks + nb] = scale[nb * K_blocks + kb];
}
