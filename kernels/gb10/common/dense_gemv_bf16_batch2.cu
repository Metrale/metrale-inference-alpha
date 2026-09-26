// SPDX-License-Identifier: MIT OR Apache-2.0

// 2026-09-25: Dense BF16 GEMV over two activation rows in one pass over the weight:
// C[t, n] = sum_k A[t, k] * B[n, k] for t in {0, 1}.
//
// Owner: gb10 kernels.
// Invariants:
// - A is [2, K] contiguous, B is [N, K] row-major, row t of C starts at C + t * out_stride
//   (BF16 elements).
// - Launch: grid (ceil(N / 4), 1, 1), block (256, 1, 1); 64 threads (2 warps) per output.
// - Assumes K % 8 == 0, as dense_gemv_bf16 does, for aligned uint4 loads of every row.
// - Each row's result is bit-identical to dense_gemv_bf16 on that row: the same K order,
//   the same warp and cross-warp reduction, and the common build passes --fmad=false.
//
// Caller: the GDN layer's two-token in_proj_qkvz when it has no NVFP4 copy
// (qwen3_ssm/trait_decode_batched.rs).








#include <cuda_bf16.h>

#define BLOCK_SIZE 256
#define N_PER_BLOCK 4
#define WARP_SIZE 32
#define VEC_SIZE 8

extern "C" __global__ void dense_gemv_bf16_batch2(
    const __nv_bfloat16* __restrict__ A,
    const __nv_bfloat16* __restrict__ B,
    __nv_bfloat16* __restrict__ C,
    unsigned int N,
    unsigned int K,
    unsigned int out_stride
) {
    const unsigned int threads_per_out = BLOCK_SIZE / N_PER_BLOCK;
    const unsigned int local_out = threadIdx.x / threads_per_out;
    const unsigned int lane = threadIdx.x % threads_per_out;

    const unsigned int n = blockIdx.x * N_PER_BLOCK + local_out;
    if (n >= N) return;

    float acc0 = 0.0f;
    float acc1 = 0.0f;

    const unsigned int K_VEC = K / VEC_SIZE;
    const uint4* A0_vec = (const uint4*)A;
    const uint4* A1_vec = (const uint4*)(A + K);
    const uint4* B_vec = (const uint4*)(B + (unsigned long long)n * K);

    for (unsigned int kv = lane; kv < K_VEC; kv += threads_per_out) {
        uint4 a0_data = A0_vec[kv];
        uint4 a1_data = A1_vec[kv];
        uint4 b_data = B_vec[kv];

        const unsigned int a0_raw[4] = {a0_data.x, a0_data.y, a0_data.z, a0_data.w};
        const unsigned int a1_raw[4] = {a1_data.x, a1_data.y, a1_data.z, a1_data.w};
        const unsigned int b_raw[4] = {b_data.x, b_data.y, b_data.z, b_data.w};

        #pragma unroll
        for (int i = 0; i < 4; i++) {
            __nv_bfloat16 b_lo, b_hi;
            *(unsigned short*)&b_lo = (unsigned short)(b_raw[i] & 0xFFFF);
            *(unsigned short*)&b_hi = (unsigned short)(b_raw[i] >> 16);
            const float bf_lo = __bfloat162float(b_lo);
            const float bf_hi = __bfloat162float(b_hi);

            __nv_bfloat16 a_lo, a_hi;
            *(unsigned short*)&a_lo = (unsigned short)(a0_raw[i] & 0xFFFF);
            *(unsigned short*)&a_hi = (unsigned short)(a0_raw[i] >> 16);
            acc0 += __bfloat162float(a_lo) * bf_lo;
            acc0 += __bfloat162float(a_hi) * bf_hi;

            *(unsigned short*)&a_lo = (unsigned short)(a1_raw[i] & 0xFFFF);
            *(unsigned short*)&a_hi = (unsigned short)(a1_raw[i] >> 16);
            acc1 += __bfloat162float(a_lo) * bf_lo;
            acc1 += __bfloat162float(a_hi) * bf_hi;
        }
    }


    {
        const unsigned int tail_start = K_VEC * VEC_SIZE;
        const __nv_bfloat16* B_row = B + (unsigned long long)n * K;
        for (unsigned int k = tail_start + lane; k < K; k += threads_per_out) {
            const float bf = __bfloat162float(B_row[k]);
            acc0 += __bfloat162float(A[k]) * bf;
            acc1 += __bfloat162float(A[K + k]) * bf;
        }
    }

    const unsigned int warp_lane = threadIdx.x % WARP_SIZE;

    #pragma unroll
    for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1) {
        acc0 += __shfl_down_sync(0xFFFFFFFF, acc0, offset);
        acc1 += __shfl_down_sync(0xFFFFFFFF, acc1, offset);
    }

    // 2026-09-25: Two warps per output: add the warp partials through shared memory, per row.
    __shared__ float smem0[N_PER_BLOCK * 2];
    __shared__ float smem1[N_PER_BLOCK * 2];

    if (warp_lane == 0) {
        unsigned int smem_idx = local_out * 2 + (lane / WARP_SIZE);
        smem0[smem_idx] = acc0;
        smem1[smem_idx] = acc1;
    }
    __syncthreads();

    if (lane == 0) {
        float r0 = smem0[local_out * 2] + smem0[local_out * 2 + 1];
        float r1 = smem1[local_out * 2] + smem1[local_out * 2 + 1];
        C[n] = __float2bfloat16(r0);
        C[(unsigned long long)out_stride + n] = __float2bfloat16(r1);
    }
}
