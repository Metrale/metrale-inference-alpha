// SPDX-License-Identifier: MIT OR Apache-2.0
// provenance-id: 526f6e616c6420522e205374657369616b

// 2026-09-25: Register-tiled batched FP8 GEMV for the DFlash drafter's propose path:
//   C[M, N] (bf16) = A[M, K] (bf16) @ fp8(B[N, K])^T * row_scale[N]
//
// Owner: gb10 kernels.
// Invariants:
// - B is [N, K] FP8 E4M3 with a per-row f32 row_scale; A and C are row-major BF16.
// - Launch: grid (ceil(N / 8), 1, 1), block (256, 1, 1): four 64-lane groups per block,
//   each computing RT_T = 2 adjacent output rows, so one activation load feeds both.
// - K % 16 == 0 and 1 <= M <= RT_MAXM (8 or 16): there is no K tail, rows past RT_MAXM are
//   never computed, and the host wrappers (ops/fp8_gemv_batch.rs) refuse other values.
// - No bit-order contract: products are fused (fmaf) in a single accumulation and
//   row_scale is applied once at write-out. The E4M3 decode is a 256-entry shared-memory
//   table filled from the __nv_fp8_e4m3 conversion.
//
// METRALE_NO_DFLASH_FP8_RT=1 makes the drafter use the tile GEMMs instead (dflash_head.rs).




















#include <cuda_bf16.h>
#include <cuda_fp8.h>

#define RT_BLOCK 256
#define RT_GROUPS 4
#define RT_T 2

template <int RT_MAXM>
__device__ __forceinline__ void fp8_gemv_rowscale_rt2_impl(
    const __nv_bfloat16* __restrict__ A,
    const unsigned char* __restrict__ B,
    const float* __restrict__ row_scale,
    __nv_bfloat16* __restrict__ C,
    unsigned int M,
    unsigned int N,
    unsigned int K
) {
    const unsigned int tpo = RT_BLOCK / RT_GROUPS;
    const unsigned int local_out = threadIdx.x / tpo;
    const unsigned int lane = threadIdx.x % tpo;
    const unsigned int n0 = (blockIdx.x * RT_GROUPS + local_out) * RT_T;

    __shared__ float s_lut[256];
    {
        __nv_fp8_e4m3 f;
        *(unsigned char*)&f = (unsigned char)threadIdx.x;
        s_lut[threadIdx.x] = (float)f;
    }
    __syncthreads();
    if (n0 >= N) return;

    const unsigned int K16 = K / 16;

    float acc[RT_T][RT_MAXM];
    #pragma unroll
    for (int o = 0; o < RT_T; o++)
        #pragma unroll
        for (int t = 0; t < RT_MAXM; t++) acc[o][t] = 0.0f;

    for (unsigned int kk = lane; kk < K16; kk += tpo) {
        // 2026-09-25: RT_T weight chunks of 16 FP8 bytes, one uint4 load per output row.
        float wl[RT_T][16];
        #pragma unroll
        for (int o = 0; o < RT_T; o++) {
            const unsigned long long n = n0 + o;
            if (n < N) {
                uint4 wb = *(const uint4*)(B + n * K + (unsigned long long)kk * 16u);
                const unsigned int wr[4] = {wb.x, wb.y, wb.z, wb.w};
                #pragma unroll
                for (int w = 0; w < 4; w++)
                    #pragma unroll
                    for (int b = 0; b < 4; b++)
                        wl[o][w * 4 + b] = s_lut[(wr[w] >> (b * 8)) & 0xFF];
            } else {
                #pragma unroll
                for (int i = 0; i < 16; i++) wl[o][i] = 0.0f;
            }
        }

        #pragma unroll
        for (int t = 0; t < RT_MAXM; t++) {
            if ((unsigned int)t >= M) continue;
            const __nv_bfloat16* At = A + (unsigned long long)t * K;
            // 2026-09-25: One activation load per (chunk, row) feeds both output rows.
            uint4 a_lo = ((const uint4*)At)[kk * 2];
            uint4 a_hi = ((const uint4*)At)[kk * 2 + 1];
            const unsigned int ar[8] = {a_lo.x, a_lo.y, a_lo.z, a_lo.w,
                                        a_hi.x, a_hi.y, a_hi.z, a_hi.w};
            #pragma unroll
            for (int o = 0; o < RT_T; o++) {
                float part = 0.0f;
                #pragma unroll
                for (int b = 0; b < 8; b++) {
                    float2 af = __bfloat1622float2(*(const __nv_bfloat162*)&ar[b]);
                    part = fmaf(af.x, wl[o][b * 2], part);
                    part = fmaf(af.y, wl[o][b * 2 + 1], part);
                }
                acc[o][t] += part;
            }
        }
    }

    // 2026-09-25: 64-lane reduce: a 32-lane shuffle tree per warp, then two warps via smem.
    __shared__ float s_red[RT_MAXM][RT_GROUPS * RT_T * 2];
    const unsigned int warp_in_out = lane / 32u;
    #pragma unroll
    for (int o = 0; o < RT_T; o++) {
        #pragma unroll
        for (int t = 0; t < RT_MAXM; t++) {
            if ((unsigned int)t >= M) continue;
            float a = acc[o][t];
            #pragma unroll
            for (int off = 16; off > 0; off >>= 1)
                a += __shfl_down_sync(0xFFFFFFFF, a, off);
            if ((lane & 31u) == 0u)
                s_red[t][(local_out * RT_T + o) * 2 + warp_in_out] = a;
        }
    }
    __syncthreads();

    if (lane == 0) {
        #pragma unroll
        for (int o = 0; o < RT_T; o++) {
            const unsigned int n = n0 + o;
            if (n >= N) continue;
            const float rs = row_scale[n];
            #pragma unroll
            for (int t = 0; t < RT_MAXM; t++) {
                if ((unsigned int)t >= M) continue;
                float r = s_red[t][(local_out * RT_T + o) * 2]
                        + s_red[t][(local_out * RT_T + o) * 2 + 1];
                C[(unsigned long long)t * N + n] = __float2bfloat16(r * rs);
            }
        }
    }
}

extern "C" __global__ void fp8_gemv_rowscale_batch8_rt2(
    const __nv_bfloat16* __restrict__ A,
    const unsigned char* __restrict__ B,
    const float* __restrict__ row_scale,
    __nv_bfloat16* __restrict__ C,
    unsigned int M,
    unsigned int N,
    unsigned int K
) {
    fp8_gemv_rowscale_rt2_impl<8>(A, B, row_scale, C, M, N, K);
}

// 2026-09-25: RT_MAXM = 16 twin: acc grows to [2][16] and s_red to 16 rows (1 KB). Neither
// entry pins __launch_bounds__.



extern "C" __global__ void fp8_gemv_rowscale_batch16_rt2(
    const __nv_bfloat16* __restrict__ A,
    const unsigned char* __restrict__ B,
    const float* __restrict__ row_scale,
    __nv_bfloat16* __restrict__ C,
    unsigned int M,
    unsigned int N,
    unsigned int K
) {
    fp8_gemv_rowscale_rt2_impl<16>(A, B, row_scale, C, M, N, K);
}
