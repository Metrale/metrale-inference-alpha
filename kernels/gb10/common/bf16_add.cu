// SPDX-License-Identifier: MIT OR Apache-2.0

// 2026-09-25: `bf16_add_inplace`: dst[i] += src[i] for every i < n, one thread per element.
// Owner: gb10 kernels. Callers: `NcclBackend::all_reduce_2rank` and the GLM-5-Next MTP layer's residual add.
// Invariants: none beyond the types.
#include <cuda_bf16.h>

extern "C" __global__ void bf16_add_inplace(
    __nv_bfloat16* __restrict__ dst,
    const __nv_bfloat16* __restrict__ src,
    int n
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        dst[i] = __hadd(dst[i], src[i]);
    }
}
