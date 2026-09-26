// SPDX-License-Identifier: MIT OR Apache-2.0

// 2026-09-25: Hopper W8A16 dual-projection and SiLU-input GEMVs for M=1 decode: FP8
// E4M3 weights with 128x128 block scales, BF16 activations and outputs.
// forked-from: kernels/gb10/common/w8a16_gemv_fused.cu (2026-09-24; 346 of 389 lines differ, see kernels/FORKS.md)
//
// They replace gb10/common/w8a16_gemv_fused.cu on hopper targets (the [shadow] entry
// in kernels/hopper/common/KERNEL.toml) with the same entry points, arguments and
// launch geometry; `layers::dense_ffn` looks both up in module `w8a16_gemv_fused`.
// The loop, the reduction and how the result compares with the gb10 kernels' are in
// w8a16_gemv_hopper.cuh.
//
// w8a16_gemv_dual: blockIdx.z selects projection 0 (B1, B1_scale, C1) or 1 (B2,
//   B2_scale, C2); both read A[1, K] and share N and K.
//   Grid (ceil(N/4), 1, 2), block (256, 1, 1) (`ops::w8a16_gemv_dual`).
// w8a16_gemv_silu_input: the activation is silu(gate_out[k]) * up_out[k], and B is
//   the down projection. Grid (ceil(N/4), 1, 1), block (256, 1, 1)
//   (`ops::w8a16_gemv_silu_input`).
// Each B is [N, K] E4M3 bytes with a [ceil(N/128), ceil(K/128)] FP32 scale grid.
//
// Owner: hopper kernels.
// Invariants:
// - Each output element with n < N is written once, by the first thread of its
//   64-thread group; no other global memory is written.











#include "w8a16_gemv_hopper.cuh"

/// 2026-09-25: Activation source for `w8a16_gemv_silu_input`: `silu(gate[k]) * up[k]`,
/// computed as the gb10 kernel computes it, `(g / (1 + __expf(-g))) * u` on the FP32
/// widenings of the BF16 inputs, with the same approximate `__expf`.

struct HopperActSilu {
    const __nv_bfloat16* __restrict__ gate;
    const __nv_bfloat16* __restrict__ up;

    __device__ __forceinline__ void half8(const uint4 g4, const uint4 u4, float* out) const {
        float g[8];
        float u[8];
        hopper_unpack_bf16x8(g4, g);
        hopper_unpack_bf16x8(u4, u);
#pragma unroll
        for (int i = 0; i < 8; i++) {
            out[i] = (g[i] / (1.0f + __expf(-g[i]))) * u[i];
        }
    }

    __device__ __forceinline__ void chunk(unsigned int k16, HopperActChunk& out) const {
        const uint4* g4 = (const uint4*)gate;
        const uint4* u4 = (const uint4*)up;
        half8(g4[k16 * 2], u4[k16 * 2], &out.v[0]);
        half8(g4[k16 * 2 + 1], u4[k16 * 2 + 1], &out.v[8]);
    }
};


extern "C" __global__ __launch_bounds__(BLOCK_SIZE, 4) void w8a16_gemv_dual(
    const __nv_bfloat16* __restrict__ A,
    const unsigned char* __restrict__ B1,
    const float* __restrict__ B1_scale,
    __nv_bfloat16* __restrict__ C1,
    const unsigned char* __restrict__ B2,
    const float* __restrict__ B2_scale,
    __nv_bfloat16* __restrict__ C2,
    unsigned int N,
    unsigned int K
) {
    const unsigned int proj = blockIdx.z;
    const unsigned char* B = proj == 0 ? B1 : B2;
    const float* block_scale = proj == 0 ? B1_scale : B2_scale;
    __nv_bfloat16* C = proj == 0 ? C1 : C2;

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

// 2026-09-25: Two chunks in flight, not `HOPPER_GEMV_UNROLL`: the SiLU activation
// decode needs more live registers under the 64-register cap that
// `__launch_bounds__(BLOCK_SIZE, 4)` sets. Measured 2026-09-25 with nvcc 13.0.88,
// -arch=sm_90a --fmad=false -Xptxas -v: 63 registers and no spills at 2; at 4, 64
// registers and 60 bytes of spill stores.


extern "C" __global__ __launch_bounds__(BLOCK_SIZE, 4) void w8a16_gemv_silu_input(
    const __nv_bfloat16* __restrict__ gate_out,
    const __nv_bfloat16* __restrict__ up_out,
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

    const HopperActSilu act{gate_out, up_out};
    const float acc = hopper_gemv_row<2>(
        B + (unsigned long long)n * K,
        block_scale + (unsigned long long)n_block * k_blocks,
        act,
        K16,
        lane
    );

    hopper_gemv_reduce_store(acc, smem, local_out, lane, C, n);
}
