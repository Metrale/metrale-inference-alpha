// SPDX-License-Identifier: MIT OR Apache-2.0

// 2026-09-25: Transposed W8A16 GEMM with a 128 x 128 (M x N) CTA tile, for prefill: FP8 E4M3
// B_t [K, N] with transposed FP32 block scales, BF16 activations, BF16 tensor-core MMA
// (`mma.sync.m16n8k16`, FP32 accumulate). The arguments and layouts are w8a16_gemm_t's.
//   C[m, n] = sum_k A[m, k] * E4M3(B_t[k, n]) * block_scale_t[k / 128, n / 128]
//
// Owner: gb10 kernels.
// Invariants:
// - A [M, K] BF16, B_t [K, N] E4M3 bytes, block_scale_t [ceil(K / 128), ceil(N / 128)] FP32,
//   C [M, N] BF16. Grid (ceil(N / 128), ceil(M / 128), 1), block 256 (ops::w8a16_gemm_n128_m128).
// - Only C[m, n] with m < M and n < N is written.
// - Each output gets w8a16_gemm_t's 16-wide K windows in the same order, and each scale is
//   applied to the FP32 sum of one 128-wide K block.
//
// Two 64-row chunks: warps 0-3 own chunk 0 and warps 4-7 chunk 1, 16 rows each, and every warp
// covers all 128 columns (16 n8 tiles). WM128_K_STEP = 32 gives two m16n8k16 sub-MMAs per step.
// B_t is N-contiguous but each MMA B register holds a K pair: each K-step is copied in 16-byte
// cp.async chunks along N into smem_Braw [k][n], and the dequant writes the transposed smem_B
// [n][k], so each (k, k + 1) register is one 32-bit load. The E4M3 table is staged in shared
// memory. Measured 2026-09-25 (nvcc 13.0.88, sm_121f, --fmad=false): 128 registers, the cap
// that __launch_bounds__(256, 2) sets, with 496 bytes of spill stores, and 47,104 B of static
// shared memory.








#include <cuda_bf16.h>

#include "e4m3_lut.cuh"
// 2026-09-25: WM128_M_TILE is one 64-row chunk; a CTA covers 2 * WM128_M_TILE rows.
#define WM128_M_TILE   64
#define WM128_N_TILE   128
#define WM128_K_STEP   32
#define WM128_K_SUB    16
#define WM128_K_SUBS   (WM128_K_STEP / WM128_K_SUB)
#define WM128_PAD      8
#define WM128_BPAD     2
#define WM128_FP8_BLOCK 128
#define WM128_WARPS    8
#define WM128_THREADS  (WM128_WARPS * 32)

// 2026-09-25: 16-byte cp.async.cg copy, global -> shared (cached in L2 only). Both addresses must be 16-byte aligned.
__device__ __forceinline__ void wm128_cp_async_cg_16(void* smem_ptr, const void* gmem_ptr) {
    unsigned int s = (unsigned int)__cvta_generic_to_shared(smem_ptr);
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" ::"r"(s), "l"(gmem_ptr));
}
__device__ __forceinline__ void wm128_cp_async_commit() {
    asm volatile("cp.async.commit_group;\n" ::);
}
__device__ __forceinline__ void wm128_cp_async_wait_all() {
    asm volatile("cp.async.wait_group 0;\n" ::);
}




extern "C" __global__
__launch_bounds__(256, 2)
void w8a16_gemm_t_m128(
    const __nv_bfloat16* __restrict__ A,
    const unsigned char* __restrict__ B_t,
    const float* __restrict__ block_scale_t,
    __nv_bfloat16* __restrict__ C,
    unsigned int M,
    unsigned int N,
    unsigned int K
) {
    const unsigned int cta_n = blockIdx.x * WM128_N_TILE;
    const unsigned int cta_m = blockIdx.y * (2 * WM128_M_TILE);
    if (cta_m >= M) return;

    const unsigned int warp_id = threadIdx.x >> 5;
    const unsigned int lane_id = threadIdx.x & 31;
    const unsigned int chunk   = warp_id >> 2;
    const unsigned int sub     = warp_id & 3;
    const unsigned int warp_m_offset = sub * 16;
    const unsigned int group_id = lane_id >> 2;
    const unsigned int tid = lane_id & 3;

    // 2026-09-25: Two buffers. smem_A and smem_Braw are cp.async destinations with 16-byte-aligned
    // rows; smem_B is the converted, K-contiguous buffer the MMAs read.
    //   Per buffer: smem_A 128 * 40 * 2 = 10240 B + smem_Braw 32 * 128 = 4096 B +
    //               smem_B 128 * 34 * 2 = 8704 B = 23040 B; 46080 B for two buffers,
    //               plus the 1 KiB table.
    __shared__ __align__(16) __nv_bfloat16 smem_A[2][2 * WM128_M_TILE][WM128_K_STEP + WM128_PAD];
    __shared__ __align__(16) unsigned char smem_Braw[2][WM128_K_STEP][WM128_N_TILE];
    __shared__ __nv_bfloat16 smem_B[2][WM128_N_TILE][WM128_K_STEP + WM128_BPAD];

    // 2026-09-25: Stage E4M3_LUT in shared memory, one entry per thread (WM128_THREADS = 256).
    __shared__ float smem_lut[256];
    smem_lut[threadIdx.x] = E4M3_LUT[threadIdx.x];
    __syncthreads();

    // 2026-09-25: Two-level FP32 accumulation. Each warp owns 16 n8 tiles x its 16 rows. inner sums
    // the unscaled weights of the K-steps of one 128-wide K block; at each block boundary
    // outer += inner * block_scale_t.
    float inner_acc[16][4];
    float outer_acc[16][4];
    #pragma unroll
    for (int i = 0; i < 16; i++) {
        inner_acc[i][0] = 0.f; inner_acc[i][1] = 0.f;
        inner_acc[i][2] = 0.f; inner_acc[i][3] = 0.f;
        outer_acc[i][0] = 0.f; outer_acc[i][1] = 0.f;
        outer_acc[i][2] = 0.f; outer_acc[i][3] = 0.f;
    }

    const unsigned int a_stride = WM128_K_STEP + WM128_PAD;
    const unsigned int b_stride = WM128_K_STEP + WM128_BPAD;
    const unsigned int n_scale_blocks = (N + WM128_FP8_BLOCK - 1) / WM128_FP8_BLOCK;
    const unsigned int k_steps_per_block = WM128_FP8_BLOCK / WM128_K_STEP;
    const unsigned int n_block = cta_n / WM128_FP8_BLOCK;
    const unsigned int n_steps = (K + WM128_K_STEP - 1) / WM128_K_STEP;

    // 2026-09-25: A: 128 rows x WM128_K_STEP BF16 along K in 16-byte chunks; 256 threads x 16 B =
    // 4096 B per round, 128 * 32 * 2 = 8192 B per tile, so 2 rounds.
    #define WM128_LOADS(buf, kb) do { \
        { \
            unsigned int a_row_base = threadIdx.x >> 2; \
            unsigned int a_col      = (threadIdx.x & 3) << 3; \
            unsigned int gc = (kb) + a_col; \
            _Pragma("unroll") \
            for (int rnd = 0; rnd < 2; rnd++) { \
                unsigned int row = (unsigned int)(rnd * 64) + a_row_base; \
                unsigned int gr  = cta_m + row; \
                __nv_bfloat16* dst = &smem_A[(buf)][row][a_col]; \
                /* 2026-09-25: A 16-byte cp.async needs an aligned source: A[gr * K + gc] is aligned iff K % 8 == 0 (gc is a multiple of 8). When K % 8 != 0 every chunk takes the scalar path. */ \
                \
                \
                if ((gr < M) && (gc + 7 < K) && ((K & 7) == 0)) { \
                    wm128_cp_async_cg_16(dst, &A[(unsigned long long)gr * K + gc]); \
                } else { \
                    _Pragma("unroll") \
                    for (int e = 0; e < 8; e++) { \
                        unsigned int gcol = gc + e; \
                        dst[e] = (gr < M && gcol < K) \
                            ? A[(unsigned long long)gr * K + gcol] : __float2bfloat16(0.0f); \
                    } \
                } \
            } \
        } \
        { \
            /* 2026-09-25: B_t: 16-byte chunks along N of each K row into smem_Braw[k][n], mirroring B_t[kb + k, cta_n + n]; K_STEP * N_TILE / 16 = 256 chunks, one per thread. */ \
            \
            \
            unsigned int c = threadIdx.x; \
            unsigned int krow = (c * 16) / WM128_N_TILE; \
            unsigned int ncol = (c * 16) % WM128_N_TILE; \
            unsigned int gk = (kb) + krow; \
            unsigned int gn = cta_n + ncol; \
            unsigned char* dst = &smem_Braw[(buf)][krow][ncol]; \
            /* 2026-09-25: B_t[gk * N + gn] is 16-byte aligned iff N % 16 == 0 (gn is a multiple of 16); otherwise every chunk takes the scalar path. */ \
            \
            if (gk < K && gn + 15 < N && ((N & 15) == 0)) { \
                wm128_cp_async_cg_16(dst, &B_t[(unsigned long long)gk * N + gn]); \
            } else { \
                _Pragma("unroll") \
                for (int e = 0; e < 16; e++) { \
                    unsigned int gne = gn + e; \
                    dst[e] = (gk < K && gne < N) ? B_t[(unsigned long long)gk * N + gne] : 0; \
                } \
            } \
        } \
    } while(0)

    // 2026-09-25: Convert buffer buf's raw B and transpose it: read smem_Braw[k][n], write
    // smem_B[n][k]. No scale; it is applied to the FP32 accumulator at the block boundary.
    // K_STEP * N_TILE = 4096 elements, 16 per thread.

    #define WM128_DEQUANT(buf) do { \
        _Pragma("unroll") \
        for (unsigned int idx = threadIdx.x; idx < WM128_K_STEP * WM128_N_TILE; idx += WM128_THREADS) { \
            unsigned int k = idx / WM128_N_TILE; \
            unsigned int n = idx % WM128_N_TILE; \
            unsigned char wb = smem_Braw[(buf)][k][n]; \
            smem_B[(buf)][n][k] = __float2bfloat16(smem_lut[wb]); \
        } \
    } while(0)

    // 2026-09-25: Each warp runs WM128_K_SUBS x 16 MMAs on its own 16 rows; the warps of both
    // chunks run at the same time.
    #define WM128_COMPUTE(buf) do { \
        const unsigned short* sA = (const unsigned short*)smem_A[(buf)]; \
        const unsigned short* sB = (const unsigned short*)smem_B[(buf)]; \
        _Pragma("unroll") \
        for (int s = 0; s < WM128_K_SUBS; s++) { \
            unsigned int k_off = s * WM128_K_SUB; \
            unsigned int fr0 = chunk * WM128_M_TILE + warp_m_offset + group_id; \
            unsigned int fr1 = fr0 + 8; \
            unsigned int fc0 = k_off + tid * 2; \
            unsigned int fc1 = k_off + tid * 2 + 8; \
            unsigned int a0 = *(const unsigned int*)&sA[fr0 * a_stride + fc0]; \
            unsigned int a1 = *(const unsigned int*)&sA[fr1 * a_stride + fc0]; \
            unsigned int a2 = *(const unsigned int*)&sA[fr0 * a_stride + fc1]; \
            unsigned int a3 = *(const unsigned int*)&sA[fr1 * a_stride + fc1]; \
            _Pragma("unroll") \
            for (int nt = 0; nt < 16; nt++) { \
                unsigned int nc = nt * 8 + group_id; \
                unsigned int k0 = k_off + tid * 2; \
                unsigned int k1 = k_off + tid * 2 + 8; \
                unsigned int b0 = *(const unsigned int*)&sB[nc * b_stride + k0]; \
                unsigned int b1 = *(const unsigned int*)&sB[nc * b_stride + k1]; \
                asm volatile( \
                    "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 " \
                    "{%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%10,%11,%12,%13};" \
                    :"=f"(inner_acc[nt][0]),"=f"(inner_acc[nt][1]), \
                     "=f"(inner_acc[nt][2]),"=f"(inner_acc[nt][3]) \
                    :"r"(a0),"r"(a1),"r"(a2),"r"(a3),"r"(b0),"r"(b1), \
                     "f"(inner_acc[nt][0]),"f"(inner_acc[nt][1]), \
                     "f"(inner_acc[nt][2]),"f"(inner_acc[nt][3])); \
            } \
        } \
    } while(0)

    // 2026-09-25: Add the scaled inner accumulator to the outer one and reset inner.
    #define WM128_FOLD(scale_val) do { \
        float _sc = (scale_val); \
        _Pragma("unroll") \
        for (int i = 0; i < 16; i++) { \
            outer_acc[i][0] += inner_acc[i][0] * _sc; \
            outer_acc[i][1] += inner_acc[i][1] * _sc; \
            outer_acc[i][2] += inner_acc[i][2] * _sc; \
            outer_acc[i][3] += inner_acc[i][3] * _sc; \
            inner_acc[i][0] = 0.f; inner_acc[i][1] = 0.f; \
            inner_acc[i][2] = 0.f; inner_acc[i][3] = 0.f; \
        } \
    } while(0)

    // 2026-09-25: Two-buffer pipeline: the loads of step s + 1 are issued before the MMAs of step s and waited for after them.
    WM128_LOADS(0, 0);
    wm128_cp_async_commit();
    wm128_cp_async_wait_all();
    __syncthreads();
    WM128_DEQUANT(0);
    __syncthreads();

    unsigned int k_step_in_block = 0;
    int cur = 0;
    for (unsigned int step = 1; step < n_steps; step++) {
        unsigned int k_base = step * WM128_K_STEP;
        int nxt = 1 - cur;
        WM128_LOADS(nxt, k_base);
        wm128_cp_async_commit();
        WM128_COMPUTE(cur);

        // 2026-09-25: Block boundary check for the step just computed, step - 1.
        k_step_in_block++;
        if (k_step_in_block == k_steps_per_block) {
            const unsigned int k_block = ((step - 1) * WM128_K_STEP) / WM128_FP8_BLOCK;
            WM128_FOLD(block_scale_t[k_block * n_scale_blocks + n_block]);
            k_step_in_block = 0;
        }

        wm128_cp_async_wait_all();
        __syncthreads();
        WM128_DEQUANT(nxt);
        __syncthreads();
        cur = nxt;
    }

    WM128_COMPUTE(cur);
    k_step_in_block++;
    if (k_step_in_block == k_steps_per_block) {
        const unsigned int k_block = ((n_steps - 1) * WM128_K_STEP) / WM128_FP8_BLOCK;
        WM128_FOLD(block_scale_t[k_block * n_scale_blocks + n_block]);
        k_step_in_block = 0;
    } else if (k_step_in_block != 0) {
        // 2026-09-25: Trailing partial K block (K % 128 != 0), scale row (K - 1) / 128.
        const unsigned int k_block = (K - 1) / WM128_FP8_BLOCK;
        WM128_FOLD(block_scale_t[k_block * n_scale_blocks + n_block]);
    }

    #undef WM128_LOADS
    #undef WM128_DEQUANT
    #undef WM128_COMPUTE
    #undef WM128_FOLD

    // 2026-09-25: Each warp writes its own 16 rows x 128 columns.
    const unsigned int row_base = cta_m + chunk * WM128_M_TILE + warp_m_offset;
    #pragma unroll
    for (int nt = 0; nt < 16; nt++) {
        unsigned int c0 = cta_n + nt * 8 + tid * 2;
        unsigned int c1 = c0 + 1;
        unsigned int r0 = row_base + group_id;
        unsigned int r1 = r0 + 8;
        if (r0 < M && c0 < N) C[(unsigned long long)r0 * N + c0] = __float2bfloat16(outer_acc[nt][0]);
        if (r0 < M && c1 < N) C[(unsigned long long)r0 * N + c1] = __float2bfloat16(outer_acc[nt][1]);
        if (r1 < M && c0 < N) C[(unsigned long long)r1 * N + c0] = __float2bfloat16(outer_acc[nt][2]);
        if (r1 < M && c1 < N) C[(unsigned long long)r1 * N + c1] = __float2bfloat16(outer_acc[nt][3]);
    }
}
