// SPDX-License-Identifier: MIT OR Apache-2.0
// forked-from: kernels/gb10/gemma-4-31b/nvfp4/embed_scale.cu (2026-09-24; 4 of 36 lines differ, see kernels/FORKS.md)

// 2026-09-25: In-place scale kernels: bf16_scale_inplace multiplies N BF16 values by `scale`, f32_scale_inplace
// N FP32 values. The engine launches bf16_scale_inplace on the embedding rows with config.embed_scale
// (sqrt(hidden_size) for gemma4) and on the hidden state with a layer scalar, 256 threads per block.
// Owner: gb10 kernels (gemma-4-26b-a4b).
// Invariants: none beyond the types; a thread with idx >= N returns without touching memory.

#include <cuda_bf16.h>

extern "C" __global__ void bf16_scale_inplace(
    __nv_bfloat16* __restrict__ data,
    unsigned int N,
    float scale
) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N) return;

    float x = __bfloat162float(data[idx]);
    data[idx] = __float2bfloat16(x * scale);
}

// 2026-09-25: Nothing in crates/ loads this kernel.
extern "C" __global__ void f32_scale_inplace(
    float* __restrict__ data,
    unsigned int N,
    float scale
) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N) return;
    data[idx] = data[idx] * scale;
}
