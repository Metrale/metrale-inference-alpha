// SPDX-License-Identifier: MIT OR Apache-2.0

// 2026-09-25: BF16 tensor-core GEMM, C[M, N] = A[M, K] * B[N, K]^T with FP32 accumulation, on m16n8k16 MMA.
//
// `dense_gemm_tc` stores C; `dense_gemm_tc_scaled_acc` folds C += scale * bf16(A * B^T) for the LoRA path
// (crates/model-layers/src/layers/ops/lora_delta.rs). A block computes a 16x64 tile of C with 4 warps, each 16x16 as
// two 16x8 MMAs, and walks K in steps of 16. Grid (ceil(N/64), ceil(M/16)), block 128.
//
// Owner: gb10 kernels.
// Invariants: none beyond the types.




#include <cuda_bf16.h>

#define TC_TM 16
#define TC_TN 64
#define TC_TK 16
#define TC_PAD 8
#define TC_BLOCK 128



template <bool ACC>
__device__ __forceinline__ void store(
    __nv_bfloat16* __restrict__ C, unsigned int idx, float a, float scale
) {
    if (ACC) {
        float d = __bfloat162float(__float2bfloat16(a));
        C[idx] = __float2bfloat16(__bfloat162float(C[idx]) + scale * d);
    } else {
        C[idx] = __float2bfloat16(a);
    }
}

// 2026-09-25: Shared body. ACC=false stores C. ACC=true folds C += scale * bf16(A * B^T), which spares the LoRA path an
// [M, N] scratch write and a separate bf16_scaled_add pass over it.







template <bool ACC>
__device__ __forceinline__ void dense_gemm_tc_body(
    const __nv_bfloat16* __restrict__ A,
    const __nv_bfloat16* __restrict__ B,
    __nv_bfloat16* __restrict__ C,
    unsigned int M,
    unsigned int N,
    unsigned int K,
    float scale
) {
    const unsigned int m_block = blockIdx.y * TC_TM;
    const unsigned int n_block = blockIdx.x * TC_TN;
    const unsigned int tid = threadIdx.x;
    const unsigned int warp_id = tid / 32;
    const unsigned int lane_id = tid % 32;
    const unsigned int group_id = lane_id >> 2;
    const unsigned int tid_in_group = lane_id & 3;



    const unsigned int n_warp_base = warp_id * 16;


    __shared__ __nv_bfloat16 smem_A[TC_TM][TC_TK + TC_PAD];
    __shared__ __nv_bfloat16 smem_B[TC_TK][TC_TN + TC_PAD];


    float acc[2][4];
    #pragma unroll
    for (int t = 0; t < 2; t++) {
        acc[t][0] = 0.0f; acc[t][1] = 0.0f;
        acc[t][2] = 0.0f; acc[t][3] = 0.0f;
    }

    const unsigned short* sA_u16 = (const unsigned short*)smem_A;
    const unsigned short* sB_u16 = (const unsigned short*)smem_B;
    const unsigned int sA_stride = TC_TK + TC_PAD;
    const unsigned int sB_stride = TC_TN + TC_PAD;


    for (unsigned int k_base = 0; k_base < K; k_base += TC_TK) {


        {

            for (unsigned int idx = tid; idx < TC_TM * TC_TK; idx += TC_BLOCK) {
                unsigned int r = idx / TC_TK;
                unsigned int c = idx % TC_TK;
                unsigned int gr = m_block + r;
                unsigned int gc = k_base + c;
                smem_A[r][c] = (gr < M && gc < K) ? A[gr * K + gc] : __float2bfloat16(0.0f);
            }
            // 2026-09-25: The mapping walks k fastest: B is [N, K] row-major, so neighbouring threads read neighbouring
            // elements of one B row and the global loads coalesce.









            for (unsigned int i = tid; i < TC_TK * TC_TN; i += TC_BLOCK) {
                unsigned int bn = i / TC_TK;
                unsigned int bk = i % TC_TK;
                unsigned int gn = n_block + bn;
                unsigned int gk = k_base + bk;
                smem_B[bk][bn] = (gn < N && gk < K) ? B[(unsigned long long)gn * K + gk] : __float2bfloat16(0.0f);
            }
        }
        __syncthreads();





        unsigned int ar0 = group_id;
        unsigned int ar1 = group_id + 8;
        unsigned int ac0 = tid_in_group * 2;
        unsigned int ac1 = tid_in_group * 2 + 8;
        unsigned int a0 = *(const unsigned int*)&sA_u16[ar0 * sA_stride + ac0];
        unsigned int a1 = *(const unsigned int*)&sA_u16[ar1 * sA_stride + ac0];
        unsigned int a2 = *(const unsigned int*)&sA_u16[ar0 * sA_stride + ac1];
        unsigned int a3 = *(const unsigned int*)&sA_u16[ar1 * sA_stride + ac1];


        #pragma unroll
        for (int nt = 0; nt < 2; nt++) {
            unsigned int n_col = n_warp_base + nt * 8 + group_id;
            unsigned int k0 = tid_in_group * 2;
            unsigned int k1 = tid_in_group * 2 + 8;
            unsigned int b0 = ((unsigned int)sB_u16[k0 * sB_stride + n_col] |
                              ((unsigned int)sB_u16[(k0+1) * sB_stride + n_col] << 16));
            unsigned int b1 = ((unsigned int)sB_u16[k1 * sB_stride + n_col] |
                              ((unsigned int)sB_u16[(k1+1) * sB_stride + n_col] << 16));

            asm volatile(
                "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
                "{%0, %1, %2, %3}, "
                "{%4, %5, %6, %7}, "
                "{%8, %9}, "
                "{%10, %11, %12, %13};"
                : "=f"(acc[nt][0]), "=f"(acc[nt][1]),
                  "=f"(acc[nt][2]), "=f"(acc[nt][3])
                : "r"(a0), "r"(a1), "r"(a2), "r"(a3),
                  "r"(b0), "r"(b1),
                  "f"(acc[nt][0]), "f"(acc[nt][1]),
                  "f"(acc[nt][2]), "f"(acc[nt][3])
            );
        }

        __syncthreads();
    }






    #pragma unroll
    for (int nt = 0; nt < 2; nt++) {
        unsigned int r0 = m_block + group_id;
        unsigned int r1 = m_block + group_id + 8;
        unsigned int c0 = n_block + n_warp_base + nt * 8 + tid_in_group * 2;
        unsigned int c1 = c0 + 1;

        // 2026-09-25: The fold rounds the product to BF16 first and then stores bf16(f32(C) + scale * f32(that)): the
        // arithmetic of dense_gemm_tc followed by bf16_scaled_add (residual_add.cu), which lora_delta.rs runs when this
        // kernel is not loaded. Folding the FP32 accumulator directly would give different results.


        if (r0 < M && c0 < N) store<ACC>(C, r0 * N + c0, acc[nt][0], scale);
        if (r0 < M && c1 < N) store<ACC>(C, r0 * N + c1, acc[nt][1], scale);
        if (r1 < M && c0 < N) store<ACC>(C, r1 * N + c0, acc[nt][2], scale);
        if (r1 < M && c1 < N) store<ACC>(C, r1 * N + c1, acc[nt][3], scale);
    }
}

extern "C" __global__ void dense_gemm_tc(
    const __nv_bfloat16* __restrict__ A,
    const __nv_bfloat16* __restrict__ B,
    __nv_bfloat16* __restrict__ C,
    unsigned int M,
    unsigned int N,
    unsigned int K
) {
    dense_gemm_tc_body<false>(A, B, C, M, N, K, 0.0f);
}

// 2026-09-25: C[M, N] += scale * bf16(A[M, K] * B[N, K]^T): the LoRA expand and fold in one pass.
extern "C" __global__ void dense_gemm_tc_scaled_acc(
    const __nv_bfloat16* __restrict__ A,
    const __nv_bfloat16* __restrict__ B,
    __nv_bfloat16* __restrict__ C,
    unsigned int M,
    unsigned int N,
    unsigned int K,
    float scale
) {
    dense_gemm_tc_body<true>(A, B, C, M, N, K, scale);
}
