// SPDX-License-Identifier: AGPL-3.0-only

#include <cuda_bf16.h>

// Optional unique, finite maximum certificate for CPU-sampler-compatible
// greedy admission. Legacy kernels above retain their original tie policy.
// UINT_MAX is never a valid index (host validates vocab_size < UINT_MAX).
// Any nonfinite input or tied maximum returns UINT_MAX for exact host fallback.
extern "C" __global__ void argmax_bf16_batch_unique(
    const __nv_bfloat16* __restrict__ logits,
    unsigned int* __restrict__ out,
    unsigned int n,
    unsigned int row_stride
) {
    __shared__ float maxima[1024];
    __shared__ unsigned int indices[1024];
    __shared__ unsigned int counts[1024];
    __shared__ unsigned int invalid[1024];
    const unsigned int tid = threadIdx.x;
    const __nv_bfloat16* row_logits = logits +
        (unsigned long long)blockIdx.x * (unsigned long long)row_stride;
    float maximum = -__int_as_float(0x7f800000);
    unsigned int index = 0;
    unsigned int count = 0;
    unsigned int bad = 0;
    for (unsigned int i = tid; i < n; i += blockDim.x) {
        const float value = __bfloat162float(row_logits[i]);
        // Detect both infinities and every NaN without relying on fast-math.
        if ((__float_as_uint(value) & 0x7f800000u) == 0x7f800000u) {
            bad = 1;
        } else if (value > maximum) {
            maximum = value;
            index = i;
            count = 1;
        } else if (value == maximum) {
            count = 2;
        }
    }
    maxima[tid] = maximum;
    indices[tid] = index;
    counts[tid] = count;
    invalid[tid] = bad;
    __syncthreads();
    for (unsigned int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            invalid[tid] |= invalid[tid + stride];
            if (maxima[tid + stride] > maxima[tid]) {
                maxima[tid] = maxima[tid + stride];
                indices[tid] = indices[tid + stride];
                counts[tid] = counts[tid + stride];
            } else if (maxima[tid + stride] == maxima[tid]) {
                const unsigned int sum = counts[tid] + counts[tid + stride];
                counts[tid] = sum > 1 ? 2 : sum;
            }
        }
        __syncthreads();
    }
    if (tid == 0) {
        out[blockIdx.x] = invalid[0] || counts[0] != 1 ? 0xffffffffu : indices[0];
    }
}
