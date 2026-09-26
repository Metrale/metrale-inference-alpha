// SPDX-License-Identifier: MIT OR Apache-2.0
// forked-from: kernels/gb10/common/dense_gemm_tc.cu (2026-09-24; 154 of 207 lines differ, see kernels/FORKS.md)

// 2026-09-25: BF16 GEMM on AMD WMMA (16x16x16, wave32) for gfx1151:
//
//   C[M,N] = A[M,K] @ B[N,K]^T   (BF16 in, BF16 out, FP32 accumulation)
//
// One block of 128 threads (4 waves) per 16 x 64 tile of C, each wave one
// 16 x 16 WMMA tile, K consumed 16 at a time through shared memory.
// Grid: (ceil(N/64), ceil(M/16), 1).
//
// WMMA fragments, lane l: a[i] = smem_A[l & 15][i], b[k] = smem_B[k][n + (l & 15)];
// accumulator element e goes to C[row + 2e + (l >> 4)][col + (l & 15)].
//
// Owner: strix-hip kernels.
// Invariants: none beyond the types.



#include <cuda_bf16.h>

typedef __bf16 v16bf __attribute__((ext_vector_type(16)));
typedef float  v8f   __attribute__((ext_vector_type(8)));

#define TC_TM 16
#define TC_TN 64
#define TC_TK 16
#define TC_PAD 8
#define TC_BLOCK 128

extern "C" __global__ void dense_gemm_tc(
    const __nv_bfloat16* __restrict__ A,
    const __nv_bfloat16* __restrict__ B,
    __nv_bfloat16* __restrict__ C,
    unsigned int M,
    unsigned int N,
    unsigned int K
) {
    const unsigned int m_block = blockIdx.y * TC_TM;
    const unsigned int n_block = blockIdx.x * TC_TN;
    const unsigned int tid = threadIdx.x;
    const unsigned int warp_id = tid / 32;
    const unsigned int lane_id = tid % 32;


    const unsigned int n_warp_base = warp_id * 16;


    __shared__ __nv_bfloat16 smem_A[TC_TM][TC_TK + TC_PAD];
    __shared__ __nv_bfloat16 smem_B[TC_TK][TC_TN + TC_PAD];

    v8f acc = v8f{0, 0, 0, 0, 0, 0, 0, 0};


    for (unsigned int k_base = 0; k_base < K; k_base += TC_TK) {

        {
            unsigned int idx = tid;

            if (idx < TC_TM * TC_TK) {
                unsigned int r = idx / TC_TK;
                unsigned int c = idx % TC_TK;
                unsigned int gr = m_block + r;
                unsigned int gc = k_base + c;
                smem_A[r][c] = (gr < M && gc < K) ? A[gr * K + gc] : __float2bfloat16(0.0f);
            }


            for (unsigned int i = tid; i < TC_TK * TC_TN; i += TC_BLOCK) {
                unsigned int bk = i / TC_TN;
                unsigned int bn = i % TC_TN;
                unsigned int gn = n_block + bn;
                unsigned int gk = k_base + bk;
                smem_B[bk][bn] = (gn < N && gk < K) ? B[(unsigned long long)gn * K + gk] : __float2bfloat16(0.0f);
            }
        }
        __syncthreads();


        v16bf a;
        #pragma unroll
        for (int i = 0; i < 16; i++) a[i] = (__bf16)(float)smem_A[lane_id & 15][i];
        v16bf b;
        #pragma unroll
        for (int k = 0; k < 16; k++) b[k] = (__bf16)(float)smem_B[k][n_warp_base + (lane_id & 15)];
        acc = __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32(a, b, acc);

        __syncthreads();
    }


    #pragma unroll
    for (int e = 0; e < 8; e++) {
        unsigned int r = m_block + 2 * e + (lane_id >> 4);
        unsigned int c = n_block + n_warp_base + (lane_id & 15);
        if (r < M && c < N) C[r * N + c] = __float2bfloat16(acc[e]);
    }
}
