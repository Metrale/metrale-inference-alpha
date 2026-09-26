// SPDX-License-Identifier: MIT OR Apache-2.0

// 2026-09-25: FP32 add, c[i] = a[i] + b[i] for i < n, one thread per element.
// Owner: gb10 kernels. Invariants: none beyond the types.

extern "C" __global__ void vector_add(
    const float* __restrict__ a,
    const float* __restrict__ b,
    float* __restrict__ c,
    unsigned int n
) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        c[idx] = a[idx] + b[idx];
    }
}
