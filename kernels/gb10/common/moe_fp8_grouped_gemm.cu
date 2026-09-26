// SPDX-License-Identifier: MIT OR Apache-2.0

// 2026-09-25: Routed-expert grouped GEMM for prefill, FP8 E4M3 block-scaled weights. For
// expert e and its rows m in [expert_offsets[e], expert_offsets[e + 1]):
//   C[m, :] = A[token(m), :] @ dequant(B_e)^T, token(m) = sorted_token_ids[m] (m when NULL)
//   dequant(B_e)[n][k] = E4M3(B_e[n][k]) * scale_e[n / 128][k / 128], FP32 scales.
//
// Owner: gb10 kernels.
// Invariants:
// - The grid is 1-D. Block b takes work-items b, b + gridDim.x, ... below *total_tiles, so
//   any grid size covers every item exactly once. Item w is expert worklist[2w] and the
//   tile (m_tile, n_tile) = (worklist[2w + 1] >> 6, worklist[2w + 1] & 63), written by
//   moe_build_tile_worklist (moe_permute.cu).
// - moe_build_tile_worklist and this kernel must be enqueued on the same stream: this
//   kernel reads total_tiles and worklist with no event after the builder writes them.
// - A tile is 128 rows by 64 columns, computed by 256 threads: 8 warps of 16 rows.
// - Accumulation is two-level FP32: the MMAs sum into inner over one 128-wide K block,
//   which is scaled once and added to outer at the block boundary and after a trailing
//   partial block.
// - An expert with a NULL weight pointer writes nothing. Rows past the expert's count and
//   columns past N are not written; A and B load as zero past M, N and K.








#include <cuda_bf16.h>

#define FP8_BLOCK 128

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

// 2026-09-25: Inner GEMM. Each resident K step is PM4_K_STEP = 32 wide and runs PM4_K_SUBS
// m16n8k16 sub-MMAs between barriers, summing into the same inner accumulator in K order.
// smem_B is stored [n][k], so the (k, k + 1) BF16 pair of a B fragment is one aligned 32-bit
// load. An smem_A row is PM4_K_STEP + 8 BF16 (80 bytes), a multiple of the 16 bytes each
// cp.async moves.













































#define PM4_M_TILE 128
#define PM4_N_TILE 64
#define PM4_K_STEP 32
#define PM4_K_SUB 16
#define PM4_K_SUBS (PM4_K_STEP / PM4_K_SUB)
#define PM4_PAD 2
#define PM4_A_STRIDE (PM4_K_STEP + 8)
#define PM4_WARPS 8
#define PM4_THREADS (PM4_WARPS * 32)
#define PM4_N_TILES_PER_WARP (PM4_N_TILE / 8)
#define PM4_STAGES 2

__device__ __forceinline__ void pm4_cp_async_cg_16(void* smem_ptr, const void* gmem_ptr) {
    unsigned int s = (unsigned int)__cvta_generic_to_shared(smem_ptr);
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" ::"r"(s), "l"(gmem_ptr));
}
__device__ __forceinline__ void pm4_cp_async_commit() {
    asm volatile("cp.async.commit_group;\n" ::);
}
template <int N>
__device__ __forceinline__ void pm4_cp_async_wait_group() {
    asm volatile("cp.async.wait_group %0;\n" ::"n"(N));
}
__device__ __forceinline__ void pm4_cp_async_wait_le(unsigned int n) {
    switch (n) {
        case 0:  pm4_cp_async_wait_group<0>(); break;
        case 1:  pm4_cp_async_wait_group<1>(); break;
        case 2:  pm4_cp_async_wait_group<2>(); break;
        default: pm4_cp_async_wait_group<3>(); break;
    }
}






__device__ __forceinline__ void pm4_mma_kstep(
    const __nv_bfloat16* smem_A,
    const __nv_bfloat16* smem_B,
    float inner[PM4_N_TILES_PER_WARP][4],
    unsigned int warp_m_offset, unsigned int group_id, unsigned int tid
) {
    const unsigned int a_stride = PM4_A_STRIDE;
    const unsigned int b_stride = PM4_K_STEP + PM4_PAD;
    const unsigned short* sA = (const unsigned short*)smem_A;
    const unsigned short* sB = (const unsigned short*)smem_B;

    unsigned int frag_r0 = warp_m_offset + group_id;
    unsigned int frag_r1 = warp_m_offset + group_id + 8;

    #pragma unroll
    for (int s = 0; s < PM4_K_SUBS; s++) {
        const unsigned int k_off = s * PM4_K_SUB;
        unsigned int frag_c0 = k_off + tid * 2;
        unsigned int frag_c1 = k_off + tid * 2 + 8;

        unsigned int a0 = *(const unsigned int*)&sA[frag_r0 * a_stride + frag_c0];
        unsigned int a1 = *(const unsigned int*)&sA[frag_r1 * a_stride + frag_c0];
        unsigned int a2 = *(const unsigned int*)&sA[frag_r0 * a_stride + frag_c1];
        unsigned int a3 = *(const unsigned int*)&sA[frag_r1 * a_stride + frag_c1];

        #pragma unroll
        for (int n_tile = 0; n_tile < PM4_N_TILES_PER_WARP; n_tile++) {
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
    // 2026-09-25: The E4M3 table is copied to shared memory because the lookups are
    // data-dependent and divergent, which __constant__ memory serialises.


    __shared__ float lut_s[256];
    #pragma unroll
    for (unsigned int i = threadIdx.x; i < 256; i += PM4_THREADS) {
        lut_s[i] = E4M3_LUT_GMOE[i];
    }




    __shared__ __align__(16) __nv_bfloat16 smem_A[PM4_STAGES][PM4_M_TILE][PM4_A_STRIDE];
    __shared__ __nv_bfloat16 smem_B[PM4_STAGES][PM4_N_TILE][PM4_K_STEP + PM4_PAD];
    __shared__ __align__(16) unsigned char smem_Braw[PM4_STAGES][PM4_N_TILE][PM4_K_STEP];


    const unsigned int warp_id = threadIdx.x / 32;
    const unsigned int lane_id = threadIdx.x % 32;
    const unsigned int warp_m_offset = warp_id * 16;
    const unsigned int group_id = lane_id >> 2;
    const unsigned int tid = lane_id & 3;

    const int total = *total_tiles;










    for (int wid = blockIdx.x; wid < total; wid += (int)gridDim.x) {
        __syncthreads();




        unsigned int expert_id = worklist[wid * 2 + 0];
        unsigned int packed    = worklist[wid * 2 + 1];
        unsigned int mt = packed >> 6;
        unsigned int nt = packed & 0x3F;

        const int m_start = expert_offsets[expert_id];
        const int M_expert = expert_offsets[expert_id + 1] - m_start;

        const unsigned char* B_exp = (const unsigned char*)B_weight_ptrs[expert_id];
        const float* S_exp = (const float*)B_scale_ptrs[expert_id];
        if (B_exp == 0) continue;

        const unsigned int cta_m_local = mt * PM4_M_TILE;
        const unsigned int cta_n = nt * PM4_N_TILE;





        float inner_acc[PM4_N_TILES_PER_WARP][4];
        float outer_acc[PM4_N_TILES_PER_WARP][4];
        #pragma unroll
        for (int i = 0; i < PM4_N_TILES_PER_WARP; i++) {
            inner_acc[i][0] = 0.0f; inner_acc[i][1] = 0.0f;
            inner_acc[i][2] = 0.0f; inner_acc[i][3] = 0.0f;
            outer_acc[i][0] = 0.0f; outer_acc[i][1] = 0.0f;
            outer_acc[i][2] = 0.0f; outer_acc[i][3] = 0.0f;
        }

        const unsigned int k_blocks = (K + FP8_BLOCK - 1) / FP8_BLOCK;
        const unsigned int k_steps_per_block = FP8_BLOCK / PM4_K_STEP;
        const unsigned int n_block = cta_n / FP8_BLOCK;
        const unsigned int n_steps = (K + PM4_K_STEP - 1) / PM4_K_STEP;





        const unsigned int a_chunks = (PM4_M_TILE * PM4_K_STEP) / 8;

        auto prefetch = [&](unsigned int step, unsigned int stage) {
            unsigned int k_base = step * PM4_K_STEP;


            #pragma unroll
            for (unsigned int c = threadIdx.x; c < a_chunks; c += PM4_THREADS) {
                unsigned int row = (c * 8) / PM4_K_STEP;
                unsigned int col = (c * 8) % PM4_K_STEP;
                unsigned int m_global = cta_m_local + row;
                unsigned int gc = k_base + col;
                __nv_bfloat16* dst = &smem_A[stage][row][col];
                if (m_global < (unsigned int)M_expert && gc + 8 <= K) {
                    int sorted_idx = m_start + (int)m_global;
                    int token_id = sorted_token_ids ? sorted_token_ids[sorted_idx] : sorted_idx;
                    pm4_cp_async_cg_16(dst, &A[(unsigned long long)token_id * K + gc]);
                } else {
                    #pragma unroll
                    for (unsigned int e = 0; e < 8; e++) {
                        unsigned int gcol = gc + e;
                        if (m_global < (unsigned int)M_expert && gcol < K) {
                            int sorted_idx = m_start + (int)m_global;
                            int token_id = sorted_token_ids ? sorted_token_ids[sorted_idx] : sorted_idx;
                            dst[e] = A[(unsigned long long)token_id * K + gcol];
                        } else {
                            dst[e] = __float2bfloat16(0.0f);
                        }
                    }
                }
            }




            const unsigned int b_chunks = (PM4_N_TILE * PM4_K_STEP) / 16;
            #pragma unroll
            for (unsigned int c = threadIdx.x; c < b_chunks; c += PM4_THREADS) {
                unsigned int nrow = (c * 16) / PM4_K_STEP;
                unsigned int kcol = (c * 16) % PM4_K_STEP;
                unsigned int gn = cta_n + nrow;
                unsigned int gk = k_base + kcol;
                unsigned char* dst = &smem_Braw[stage][nrow][kcol];
                if (gn < N && gk + 16 <= K) {
                    pm4_cp_async_cg_16(dst, &B_exp[(unsigned long long)gn * K + gk]);
                } else {
                    #pragma unroll
                    for (unsigned int e = 0; e < 16; e++) {
                        unsigned int gke = gk + e;
                        dst[e] = (gn < N && gke < K) ? B_exp[(unsigned long long)gn * K + gke] : 0;
                    }
                }
            }
            pm4_cp_async_commit();
        };






        auto dequant_B = [&](unsigned int stage) {
            #pragma unroll
            for (unsigned int idx = threadIdx.x; idx < PM4_K_STEP * PM4_N_TILE; idx += PM4_THREADS) {
                unsigned int n = idx / PM4_K_STEP;
                unsigned int k = idx % PM4_K_STEP;
                unsigned char wb = smem_Braw[stage][n][k];
                smem_B[stage][n][k] = __float2bfloat16(lut_s[wb]);
            }
        };


        #pragma unroll
        for (unsigned int p = 0; p < PM4_STAGES - 1; p++) {
            if (p < n_steps) {
                prefetch(p, p % PM4_STAGES);
            }
        }
        unsigned int k_step_in_block = 0;

        for (unsigned int step = 0; step < n_steps; step++) {
            unsigned int cur = step % PM4_STAGES;

            unsigned int ahead = step + (PM4_STAGES - 1);
            if (ahead < n_steps) {
                prefetch(ahead, ahead % PM4_STAGES);
            }
            unsigned int committed = min(n_steps, PM4_STAGES + step);
            unsigned int target = committed - (step + 1);
            pm4_cp_async_wait_le(target);
            __syncthreads(); // 2026-09-25: stage `cur` (A and raw B) is resident.

            dequant_B(cur);
            __syncthreads(); // 2026-09-25: smem_B[cur] is written before the MMA reads it.

            pm4_mma_kstep(&smem_A[cur][0][0], &smem_B[cur][0][0],
                          inner_acc, warp_m_offset, group_id, tid);
            __syncthreads(); // 2026-09-25: smem_*[cur] may now be refilled.


            k_step_in_block++;
            if (k_step_in_block == k_steps_per_block) {
                const unsigned int k_block = (step * PM4_K_STEP) / FP8_BLOCK;
                const float scale = S_exp[n_block * k_blocks + k_block];
                #pragma unroll
                for (int i = 0; i < PM4_N_TILES_PER_WARP; i++) {
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


        if (k_step_in_block != 0) {
            const unsigned int k_block = (K - 1) / FP8_BLOCK;
            const float scale = S_exp[n_block * k_blocks + k_block];
            #pragma unroll
            for (int i = 0; i < PM4_N_TILES_PER_WARP; i++) {
                outer_acc[i][0] += inner_acc[i][0] * scale;
                outer_acc[i][1] += inner_acc[i][1] * scale;
                outer_acc[i][2] += inner_acc[i][2] * scale;
                outer_acc[i][3] += inner_acc[i][3] * scale;
            }
        }


        #pragma unroll
        for (int n_tile = 0; n_tile < PM4_N_TILES_PER_WARP; n_tile++) {
            unsigned int base_n = cta_n + n_tile * 8;
            unsigned int col0 = base_n + (tid * 2);
            unsigned int col1 = col0 + 1;
            unsigned int row0 = cta_m_local + warp_m_offset + group_id;
            unsigned int row1 = row0 + 8;

            if (row0 < (unsigned int)M_expert) {
                unsigned int out_row = m_start + row0;
                if (col0 < N) C[(unsigned long long)out_row * N + col0] = __float2bfloat16(outer_acc[n_tile][0]);
                if (col1 < N) C[(unsigned long long)out_row * N + col1] = __float2bfloat16(outer_acc[n_tile][1]);
            }
            if (row1 < (unsigned int)M_expert) {
                unsigned int out_row = m_start + row1;
                if (col0 < N) C[(unsigned long long)out_row * N + col0] = __float2bfloat16(outer_acc[n_tile][2]);
                if (col1 < N) C[(unsigned long long)out_row * N + col1] = __float2bfloat16(outer_acc[n_tile][3]);
            }
        }
    }
}
