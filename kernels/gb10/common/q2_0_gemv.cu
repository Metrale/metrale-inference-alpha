// SPDX-License-Identifier: MIT OR Apache-2.0

// 2026-09-25: Q2_0 GEMVs that keep the ternary weight packed:
// C[m, n] = sum_k A[m, k] * (code(n, k) - 1) * d(n, k / group), FP32 accumulation, BF16 out.
//
// A block_q2_0 is [fp16 d][group/4 bytes of 2-bit codes, low bits first]; row n of the [N, K]
// weight is K/group blocks of 2 + group/4 bytes. A is [M, K] and C is [M, N], both row-major
// BF16. Grid (ceil(N/4), 1, 1), block (256, 1, 1): 4 outputs per block and 64 threads (2 warps)
// per output, reduced by warp shuffle and a 2-slot smem combine. q2_0_gemv_batchm reads each
// weight block once for all M rows.
// Only crates/model-arch/examples/q2_0_gemv_microtest.rs launches these two kernels, as the
// baseline for q2_0_gemv_vec.cu. The layout test in crates/model-layers/src/layers/ops/gemv_q2.rs
// reads this source as text.
//
// Owner: gb10 kernels.
// Invariants:
// - K is a multiple of group: the kernels walk K / group blocks and never read a tail.
// - q2_0_gemv_batchm needs M <= MAX_M (8): acc[] and smem are sized by MAX_M and M is not
//   checked here.





#include <cuda_bf16.h>

#define BLOCK_SIZE 256
#define N_PER_BLOCK 4
#define WARP_SIZE 32
#define MAX_M 8


// 2026-09-25: Little-endian fp16 -> f32, the same conversion as dq_rd_f16 in dequant_gguf_bf16.cu.
__device__ __forceinline__ float q2_rd_f16(const unsigned char* p) {
    unsigned short bits = (unsigned short)p[0] | ((unsigned short)p[1] << 8);
    return __half2float(__ushort_as_half(bits));
}








extern "C" __global__ void q2_0_gemv(
    const __nv_bfloat16* __restrict__ A,
    const unsigned char* __restrict__ B,
    __nv_bfloat16* __restrict__ C,
    unsigned int N,
    unsigned int K,
    unsigned int group)
{
    const unsigned int threads_per_out = BLOCK_SIZE / N_PER_BLOCK;
    const unsigned int local_out = threadIdx.x / threads_per_out;
    const unsigned int lane = threadIdx.x % threads_per_out;

    const unsigned int n = blockIdx.x * N_PER_BLOCK + local_out;

    __shared__ float smem[N_PER_BLOCK * 2];
    if (n >= N) {

        return;
    }

    const unsigned int blocks_per_row = K / group;
    const unsigned int block_bytes = 2u + group / 4u;
    const unsigned char* row = B + (unsigned long long)n * blocks_per_row * block_bytes;

    float acc = 0.0f;


    for (unsigned int b = lane; b < blocks_per_row; b += threads_per_out) {
        const unsigned char* blk = row + (unsigned long long)b * block_bytes;
        const float d = q2_rd_f16(blk);
        const unsigned char* qs = blk + 2;
        const unsigned int base_k = b * group;


        #pragma unroll 4
        for (unsigned int cb = 0; cb < group / 4u; ++cb) {
            const unsigned int byte = qs[cb];
            const unsigned int k0 = base_k + cb * 4u;
            #pragma unroll
            for (unsigned int t = 0; t < 4u; ++t) {
                const int code = (int)((byte >> (2u * t)) & 3u);
                const float a = __bfloat162float(A[k0 + t]);
                acc += a * (float)(code - 1) * d;
            }
        }
    }


    #pragma unroll
    for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1) {
        acc += __shfl_down_sync(0xFFFFFFFF, acc, offset);
    }

    const unsigned int warp_in_out = lane / WARP_SIZE;
    if (lane % WARP_SIZE == 0) {
        smem[local_out * 2 + warp_in_out] = acc;
    }
    __syncthreads();

    if (lane == 0) {
        C[n] = __float2bfloat16(smem[local_out * 2] + smem[local_out * 2 + 1]);
    }
}









extern "C" __global__ void q2_0_gemv_batchm(
    const __nv_bfloat16* __restrict__ A,
    const unsigned char* __restrict__ B,
    __nv_bfloat16* __restrict__ C,
    unsigned int N,
    unsigned int K,
    unsigned int group,
    unsigned int M)
{
    const unsigned int threads_per_out = BLOCK_SIZE / N_PER_BLOCK;
    const unsigned int local_out = threadIdx.x / threads_per_out;
    const unsigned int lane = threadIdx.x % threads_per_out;

    const unsigned int n = blockIdx.x * N_PER_BLOCK + local_out;

    __shared__ float smem[N_PER_BLOCK * 2 * MAX_M];
    if (n >= N) return;

    const unsigned int blocks_per_row = K / group;
    const unsigned int block_bytes = 2u + group / 4u;
    const unsigned char* row = B + (unsigned long long)n * blocks_per_row * block_bytes;

    float acc[MAX_M];
    #pragma unroll
    for (unsigned int m = 0; m < MAX_M; ++m) acc[m] = 0.0f;

    for (unsigned int b = lane; b < blocks_per_row; b += threads_per_out) {
        const unsigned char* blk = row + (unsigned long long)b * block_bytes;
        const float d = q2_rd_f16(blk);
        const unsigned char* qs = blk + 2;
        const unsigned int base_k = b * group;

        #pragma unroll 4
        for (unsigned int cb = 0; cb < group / 4u; ++cb) {
            const unsigned int byte = qs[cb];
            const unsigned int k0 = base_k + cb * 4u;
            #pragma unroll
            for (unsigned int t = 0; t < 4u; ++t) {
                const int code = (int)((byte >> (2u * t)) & 3u);
                const float wv = (float)(code - 1) * d;
                const unsigned int k = k0 + t;
                for (unsigned int m = 0; m < M; ++m) {
                    acc[m] += __bfloat162float(A[m * K + k]) * wv;
                }
            }
        }
    }

    const unsigned int warp_in_out = lane / WARP_SIZE;
    for (unsigned int m = 0; m < M; ++m) {
        float v = acc[m];
        #pragma unroll
        for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1) {
            v += __shfl_down_sync(0xFFFFFFFF, v, offset);
        }
        if (lane % WARP_SIZE == 0) {
            smem[(local_out * MAX_M + m) * 2 + warp_in_out] = v;
        }
    }
    __syncthreads();

    if (lane == 0) {
        for (unsigned int m = 0; m < M; ++m) {
            float r = smem[(local_out * MAX_M + m) * 2] + smem[(local_out * MAX_M + m) * 2 + 1];
            C[m * N + n] = __float2bfloat16(r);
        }
    }
}
