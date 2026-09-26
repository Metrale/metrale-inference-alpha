// SPDX-License-Identifier: MIT OR Apache-2.0

// 2026-09-25: Split-K BF16 GEMM, C[M, N] = A[M, K] * B[N, K]^T, as two kernels.
//
// `dense_gemm_splitk_partial` writes the FP32 product over one K chunk of ceil(K / K_splits) columns (the chunk is
// blockIdx.z) to C_partial[K_splits, M, N]. `dense_gemm_splitk_reduce` sums the K_splits partials of each element and
// stores BF16.
//
// Owner: gb10 kernels.
// Invariants: none beyond the types.







#include <cuda_bf16.h>

#define SK_TILE_M 16
#define SK_TILE_N 16
#define SK_TILE_K 16

// 2026-09-25: Grid (ceil(N/16), ceil(M/16), K_splits), block (16, 16): one thread per output element of a 16x16 tile.


extern "C" __global__ void dense_gemm_splitk_partial(
    const __nv_bfloat16* __restrict__ A,
    const __nv_bfloat16* __restrict__ B,
    float* __restrict__ C_partial,
    unsigned int M,
    unsigned int N,
    unsigned int K,
    unsigned int K_splits
) {
    unsigned int split = blockIdx.z;
    unsigned int row = blockIdx.y * SK_TILE_M + threadIdx.y;
    unsigned int col = blockIdx.x * SK_TILE_N + threadIdx.x;


    unsigned int k_chunk = (K + K_splits - 1) / K_splits;
    unsigned int k_start = split * k_chunk;
    unsigned int k_end = min(k_start + k_chunk, K);

    __shared__ __nv_bfloat16 smem_A[SK_TILE_M][SK_TILE_K];
    __shared__ __nv_bfloat16 smem_B[SK_TILE_K][SK_TILE_N];

    float acc = 0.0f;

    for (unsigned int k_base = k_start; k_base < k_end; k_base += SK_TILE_K) {
        unsigned int k_local = k_base + threadIdx.x;
        if (row < M && k_local < k_end) {
            smem_A[threadIdx.y][threadIdx.x] = A[row * K + k_local];
        } else {
            smem_A[threadIdx.y][threadIdx.x] = __float2bfloat16(0.0f);
        }

        unsigned int k_local_y = k_base + threadIdx.y;
        if (k_local_y < k_end && col < N) {
            smem_B[threadIdx.y][threadIdx.x] = B[(unsigned long long)col * K + k_local_y];
        } else {
            smem_B[threadIdx.y][threadIdx.x] = __float2bfloat16(0.0f);
        }

        __syncthreads();

        unsigned int tile_end = min(SK_TILE_K, k_end - k_base);
        for (unsigned int kk = 0; kk < tile_end; kk++) {
            acc += __bfloat162float(smem_A[threadIdx.y][kk])
                 * __bfloat162float(smem_B[kk][threadIdx.x]);
        }

        __syncthreads();
    }

    if (row < M && col < N) {
        C_partial[(unsigned long long)split * M * N + row * N + col] = acc;
    }
}

// 2026-09-25: Grid (ceil(N/256), M), block 256: one thread per output element.


extern "C" __global__ void dense_gemm_splitk_reduce(
    const float* __restrict__ C_partial,
    __nv_bfloat16* __restrict__ C,
    unsigned int M,
    unsigned int N,
    unsigned int K_splits
) {
    unsigned int row = blockIdx.y;
    unsigned int col = blockIdx.x * 256 + threadIdx.x;

    if (row >= M || col >= N) return;

    float sum = 0.0f;
    for (unsigned int s = 0; s < K_splits; s++) {
        sum += C_partial[(unsigned long long)s * M * N + row * N + col];
    }

    C[row * N + col] = __float2bfloat16(sum);
}
