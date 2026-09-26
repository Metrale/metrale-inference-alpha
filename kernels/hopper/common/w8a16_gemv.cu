// SPDX-License-Identifier: MIT OR Apache-2.0

// 2026-09-25: Hopper W8A16 GEMV for M=1 decode: FP8 E4M3 weights with 128x128 block
// scales, BF16 activations and output.
// forked-from: kernels/gb10/common/w8a16_gemv.cu (2026-09-24; 214 of 223 lines differ, see kernels/FORKS.md)
//
//   C[n] = sum_k A[k] * E4M3(B[n, k]) * block_scale[n / 128, k / 128]
//
// A is [1, K] BF16, B is [N, K] E4M3 bytes, block_scale is [ceil(N/128), ceil(K/128)]
// FP32 and C is [1, N] BF16. K is read in 16-value chunks, K / 16 of them; the last
// K mod 16 values are not read.
//
// It replaces gb10/common/w8a16_gemv.cu on hopper targets (the [shadow] entry in
// kernels/hopper/common/KERNEL.toml) with the same entry point, the same six
// arguments and the launch `ops::w8a16_gemv` makes: grid (ceil(N/4), 1, 1), block
// (256, 1, 1), four outputs per block and 64 threads per output. The loop, the
// reduction and how the result compares with the gb10 kernel's are in
// w8a16_gemv_hopper.cuh.
//
// Owner: hopper kernels.
// Invariants:
// - C[n] for each n < N is written once, by the first thread of its 64-thread
//   group; no other global memory is written.








#include "w8a16_gemv_hopper.cuh"

// 2026-09-25: The second `__launch_bounds__` argument, 4 resident blocks per SM,
// caps ptxas at 64 registers per thread (65,536 / (4 x 256)). `hopper_gemv_row`
// keeps `HOPPER_GEMV_UNROLL` weight chunks and their scales live at once.
// Measured 2026-09-25 with nvcc 13.0.88, -arch=sm_90a --fmad=false -Xptxas -v:
//
//   min blocks/SM | registers | spill bytes
//               4 |        64 | 0   <- this kernel
//               5 |        48 | 0
//               6 |        40 | 0
//               8 |        32 | 0










extern "C" __global__ __launch_bounds__(BLOCK_SIZE, 4) void w8a16_gemv(
    const __nv_bfloat16* __restrict__ A,
    const unsigned char* __restrict__ B,
    const float* __restrict__ block_scale,
    __nv_bfloat16* __restrict__ C,
    unsigned int N,
    unsigned int K
) {
    const unsigned int threads_per_out = BLOCK_SIZE / N_PER_BLOCK;
    const unsigned int local_out = threadIdx.x / threads_per_out;
    const unsigned int lane = threadIdx.x % threads_per_out;

    const unsigned int n = blockIdx.x * N_PER_BLOCK + local_out;
    if (n >= N) return;

    const unsigned int K16 = K / K_PER_CHUNK;
    const unsigned int k_blocks = (K + FP8_BLOCK - 1) / FP8_BLOCK;
    const unsigned int n_block = n / FP8_BLOCK;





    __shared__ float smem[N_PER_BLOCK * 2];

    const HopperActRow act{A};
    const float acc = hopper_gemv_row<HOPPER_GEMV_UNROLL>(
        B + (unsigned long long)n * K,
        block_scale + (unsigned long long)n_block * k_blocks,
        act,
        K16,
        lane
    );

    hopper_gemv_reduce_store(acc, smem, local_out, lane, C, n);
}
