// SPDX-License-Identifier: MIT OR Apache-2.0
// forked-from: kernels/gb10/common/moe_fp8_grouped_gemm.cu (2026-09-24; 652 of 524 lines differ, see kernels/FORKS.md)

// 2026-09-25: Routed-expert FP8 grouped GEMM for gfx1151 on AMD WMMA:
//
//   C[M_expert, N] = A[M_expert, K] (BF16) @ dequant(B_expert[N, K] (E4M3))^T
//
// Tokens are sorted by expert. The grid strides (by gridDim.x) over a
// compacted work-list of (expert, m_tile, n_tile) entries written by
// moe_build_tile_worklist (kernels/gb10/common/moe_permute.cu), packed as
// worklist[2w] = expert, worklist[2w + 1] = (m_tile << 6) | n_tile. Row m of
// an expert reads token sorted_token_ids[m_start + m] (m_start + m when that
// pointer is null) and writes output row m_start + m. An expert whose weight
// pointer is null is skipped.
//
// Weights: B[N, K] E4M3 bytes with block_scale[N/128, K/128] FP32. Bytes are
// decoded through an E4M3 table into shared memory without the scale; each
// 128-K block accumulates into an FP32 `inner`, folded as
// outer += inner * block_scale at the block boundary, and outer is rounded
// to BF16 with __float2bfloat16 at the store.
//
// Tile: 128 x 64 per block of 512 threads, 16 waves in an 8 x 2 grid: each
// wave owns 16 rows and 2 of the 4 16-wide WMMA n-subtiles. K advances 16 at
// a time through two shared-memory buffers: the next tile is loaded into
// registers during the current tile's WMMA, then stored to the other buffer.
// LDS: smem_A 2*128*18*2 + smem_B 2*16*66*2 + lut_s 256*4 = 14464 bytes.
//
// WMMA fragments, lane l: a[i] = smem_A[warp_m_offset + (l & 15)][i],
// b[k] = smem_B[k][nb*16 + (l & 15)]; accumulator element e goes to row
// warp_m_offset + 2e + (l >> 4), column nb*16 + (l & 15).
//
// Owner: strix-hip kernels.
// Invariants: none beyond the types.
















#include <cuda_bf16.h>

typedef __bf16 v16bf __attribute__((ext_vector_type(16)));
typedef float  v8f   __attribute__((ext_vector_type(8)));



















#define FP8_BLOCK 128

#define PM4_M_TILE 128
#define PM4_N_TILE 64
#define PM4_K_STEP 16
#define PM4_PAD 2
#define PM4_WARPS_M (PM4_M_TILE / 16)
#define PM4_WARPS_N 2
#define PM4_WARPS (PM4_WARPS_M * PM4_WARPS_N)
#define PM4_THREADS (PM4_WARPS * 32)
#define PM4_N_SUBTILES (PM4_N_TILE / 16)
#define PM4_SUBTILES_PER_WARP (PM4_N_SUBTILES / PM4_WARPS_N)

__device__ __constant__ float E4M3_LUT_GMOE[256] = {
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



#define PM4_A_EPT ((PM4_M_TILE * PM4_K_STEP) / PM4_THREADS)
#define PM4_B_EPT ((PM4_K_STEP * PM4_N_TILE) / PM4_THREADS)


__device__ __forceinline__ void load_A_regs(
    const __nv_bfloat16* __restrict__ A,
    const int* __restrict__ sorted_token_ids,
    __nv_bfloat16 reg_A[PM4_A_EPT],
    int m_start, unsigned int cta_m_local, unsigned int k_base,
    unsigned int M_expert, unsigned int K
) {
    #pragma unroll
    for (unsigned int i = 0; i < PM4_A_EPT; i++) {
        unsigned int idx = threadIdx.x * PM4_A_EPT + i;
        unsigned int row = idx / PM4_K_STEP;
        unsigned int col = idx % PM4_K_STEP;
        unsigned int m_global = cta_m_local + row;
        unsigned int gc = k_base + col;
        if (m_global < M_expert && gc < K) {
            int sorted_idx = m_start + (int)m_global;
            int token_id = sorted_token_ids ? sorted_token_ids[sorted_idx] : sorted_idx;
            reg_A[i] = A[(unsigned long long)token_id * K + gc];
        } else {
            reg_A[i] = __float2bfloat16(0.0f);
        }
    }
}

__device__ __forceinline__ void store_A_regs(
    __nv_bfloat16 smem_A[][PM4_K_STEP + PM4_PAD], const __nv_bfloat16 reg_A[PM4_A_EPT]
) {
    #pragma unroll
    for (unsigned int i = 0; i < PM4_A_EPT; i++) {
        unsigned int idx = threadIdx.x * PM4_A_EPT + i;
        unsigned int row = idx / PM4_K_STEP;
        unsigned int col = idx % PM4_K_STEP;
        smem_A[row][col] = reg_A[i];
    }
}




__device__ __forceinline__ void load_B_regs(
    const unsigned char* __restrict__ B_exp,
    unsigned char reg_B[PM4_B_EPT],
    unsigned int cta_n, unsigned int k_base,
    unsigned int N, unsigned int K
) {
    #pragma unroll
    for (unsigned int i = 0; i < PM4_B_EPT; i++) {
        unsigned int idx = threadIdx.x * PM4_B_EPT + i;
        unsigned int k = idx / PM4_N_TILE;
        unsigned int n = idx % PM4_N_TILE;
        unsigned int gk = k_base + k;
        unsigned int gn = cta_n + n;
        reg_B[i] = (gk < K && gn < N) ? B_exp[(unsigned long long)gn * K + gk] : 0;
    }
}

__device__ __forceinline__ void store_B_regs(
    __nv_bfloat16 smem_B[][PM4_N_TILE + PM4_PAD],
    const unsigned char reg_B[PM4_B_EPT], const float* lut_s
) {
    #pragma unroll
    for (unsigned int i = 0; i < PM4_B_EPT; i++) {
        unsigned int idx = threadIdx.x * PM4_B_EPT + i;
        unsigned int k = idx / PM4_N_TILE;
        unsigned int n = idx % PM4_N_TILE;
        smem_B[k][n] = __float2bfloat16(lut_s[reg_B[i]]);
    }
}



__device__ __forceinline__ void mma_kstep(
    const __nv_bfloat16 smem_A[][PM4_K_STEP + PM4_PAD],
    const __nv_bfloat16 smem_B[][PM4_N_TILE + PM4_PAD],
    v8f inner[PM4_SUBTILES_PER_WARP],
    unsigned int warp_m_offset, unsigned int n_sub_base, unsigned int lane
) {
    v16bf a;
    #pragma unroll
    for (int i = 0; i < 16; i++) a[i] = (__bf16)(float)smem_A[warp_m_offset + (lane & 15)][i];
    #pragma unroll
    for (int j = 0; j < PM4_SUBTILES_PER_WARP; j++) {
        unsigned int nb = n_sub_base + j;
        v16bf b;
        #pragma unroll
        for (int k = 0; k < 16; k++) b[k] = (__bf16)(float)smem_B[k][nb * 16 + (lane & 15)];
        inner[j] = __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32(a, b, inner[j]);
    }
}

extern "C" __global__ void __launch_bounds__(PM4_THREADS, 2) moe_fp8_grouped_gemm(
    const __nv_bfloat16* __restrict__ A,
    const unsigned long long* __restrict__ B_weight_ptrs,
    const unsigned long long* __restrict__ B_scale_ptrs,
    __nv_bfloat16* __restrict__ C,
    const int* __restrict__ expert_offsets,
    const int* __restrict__ sorted_token_ids,
    unsigned int num_experts,
    unsigned int N,
    unsigned int K,
    const unsigned int* __restrict__ worklist,
    const int* __restrict__ total_tiles
) {
    (void)num_experts;

    __shared__ float lut_s[256];
    #pragma unroll
    for (unsigned int i = threadIdx.x; i < 256; i += PM4_THREADS) {
        lut_s[i] = E4M3_LUT_GMOE[i];
    }



    __shared__ __nv_bfloat16 smem_A[2][PM4_M_TILE][PM4_K_STEP + PM4_PAD];
    __shared__ __nv_bfloat16 smem_B[2][PM4_K_STEP][PM4_N_TILE + PM4_PAD];

    const unsigned int warp_id = threadIdx.x / 32;
    const unsigned int lane_id = threadIdx.x % 32;

    const unsigned int warp_m_offset = (warp_id / PM4_WARPS_N) * 16;
    const unsigned int n_sub_base    = (warp_id % PM4_WARPS_N) * PM4_SUBTILES_PER_WARP;

    const int total = *total_tiles;

    for (int wid = blockIdx.x; wid < total; wid += (int)gridDim.x) {
        __syncthreads();

        unsigned int expert_id = worklist[wid * 2 + 0];
        unsigned int packed    = worklist[wid * 2 + 1];
        unsigned int mt = packed >> 6;
        unsigned int nt = packed & 0x3F;

        const int m_start = expert_offsets[expert_id];
        const unsigned int M_expert = (unsigned int)(expert_offsets[expert_id + 1] - m_start);

        const unsigned char* B_exp = (const unsigned char*)B_weight_ptrs[expert_id];
        const float* S_exp = (const float*)B_scale_ptrs[expert_id];
        if (B_exp == 0) continue;

        const unsigned int cta_m_local = mt * PM4_M_TILE;
        const unsigned int cta_n = nt * PM4_N_TILE;




        v8f inner_acc[PM4_SUBTILES_PER_WARP];
        v8f outer_acc[PM4_SUBTILES_PER_WARP];
        #pragma unroll
        for (int i = 0; i < PM4_SUBTILES_PER_WARP; i++) {
            inner_acc[i] = v8f{0, 0, 0, 0, 0, 0, 0, 0};
            outer_acc[i] = v8f{0, 0, 0, 0, 0, 0, 0, 0};
        }

        const unsigned int k_blocks = (K + FP8_BLOCK - 1) / FP8_BLOCK;
        const unsigned int k_steps_per_block = FP8_BLOCK / PM4_K_STEP;
        const unsigned int n_block = cta_n / FP8_BLOCK;
        const unsigned int n_steps = (K + PM4_K_STEP - 1) / PM4_K_STEP;


        __nv_bfloat16 reg_A[PM4_A_EPT];
        unsigned char reg_B[PM4_B_EPT];
        load_A_regs(A, sorted_token_ids, reg_A, m_start, cta_m_local, 0, M_expert, K);
        load_B_regs(B_exp, reg_B, cta_n, 0, N, K);
        store_A_regs(smem_A[0], reg_A);
        store_B_regs(smem_B[0], reg_B, lut_s);
        __syncthreads();

        unsigned int k_step_in_block = 0;

        // 2026-09-25: One barrier per K step: the barrier after step s's commit
        // orders every wave's read of buffer s & 1 before step s + 1 stores
        // into that buffer again.

        for (unsigned int step = 0; step < n_steps; step++) {
            const unsigned int cur = step & 1;


            if (step + 1 < n_steps) {
                unsigned int k_next = (step + 1) * PM4_K_STEP;
                load_A_regs(A, sorted_token_ids, reg_A, m_start, cta_m_local, k_next, M_expert, K);
                load_B_regs(B_exp, reg_B, cta_n, k_next, N, K);
            }

            mma_kstep(smem_A[cur], smem_B[cur], inner_acc, warp_m_offset, n_sub_base, lane_id);


            k_step_in_block++;
            if (k_step_in_block == k_steps_per_block) {
                const unsigned int k_block = (step * PM4_K_STEP) / FP8_BLOCK;
                const float scale = S_exp[n_block * k_blocks + k_block];
                #pragma unroll
                for (int i = 0; i < PM4_SUBTILES_PER_WARP; i++) {
                    outer_acc[i] += inner_acc[i] * scale;
                    inner_acc[i] = v8f{0, 0, 0, 0, 0, 0, 0, 0};
                }
                k_step_in_block = 0;
            }


            if (step + 1 < n_steps) {
                store_A_regs(smem_A[(step + 1) & 1], reg_A);
                store_B_regs(smem_B[(step + 1) & 1], reg_B, lut_s);
                __syncthreads();
            }
        }

        if (k_step_in_block != 0) {
            const unsigned int k_block = (K - 1) / FP8_BLOCK;
            const float scale = S_exp[n_block * k_blocks + k_block];
            #pragma unroll
            for (int i = 0; i < PM4_SUBTILES_PER_WARP; i++) {
                outer_acc[i] += inner_acc[i] * scale;
            }
        }



        #pragma unroll
        for (int j = 0; j < PM4_SUBTILES_PER_WARP; j++) {
            unsigned int nb = n_sub_base + j;
            #pragma unroll
            for (int e = 0; e < 8; e++) {
                unsigned int row_local = cta_m_local + warp_m_offset + 2 * e + (lane_id >> 4);
                unsigned int col = cta_n + nb * 16 + (lane_id & 15);
                if (row_local < M_expert && col < N) {
                    unsigned int out_row = (unsigned int)m_start + row_local;
                    C[(unsigned long long)out_row * N + col] = __float2bfloat16(outer_acc[j][e]);
                }
            }
        }
    }
}

// 2026-09-25: A no-op with the same arguments minus the work-list; no crate
// looks it up.
extern "C" __global__ void moe_fp8_grouped_gemm_v2(
    const __nv_bfloat16* __restrict__ A,
    const unsigned long long* __restrict__ B_weight_ptrs,
    const unsigned long long* __restrict__ B_scale_ptrs,
    __nv_bfloat16* __restrict__ C,
    const int* __restrict__ expert_offsets,
    const int* __restrict__ sorted_token_ids,
    unsigned int num_experts,
    unsigned int N,
    unsigned int K
) {
    (void)A; (void)B_weight_ptrs; (void)B_scale_ptrs; (void)C;
    (void)expert_offsets; (void)sorted_token_ids; (void)num_experts; (void)N; (void)K;
}
