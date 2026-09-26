// SPDX-License-Identifier: MIT OR Apache-2.0

// 2026-09-25: Grid-stride conversion of SSM h-state elements between FP32 and FP16, for the
// FP16 h storage behind METRALE_SSM_H_FP16 (`--ssm-h-dtype f16`). ssm_h_state_f32_to_f16
// rounds to nearest even (__float2half); ssm_h_state_f16_to_f32 is exact.
//
// Owner: gb10 kernels.
// Invariants: none beyond the types.
//
// Both are out of place: src and dst must not alias. The element sizes differ, so in place
// one element's write would overlap another element's source with nothing ordering the two
// (for f32 -> f16, element 2i writes bytes [4i, 4i + 2), inside element i's source).










#include <cuda_fp16.h>

extern "C" __global__ void ssm_h_state_f32_to_f16(
    const float* __restrict__ src,
    __half* __restrict__ dst,
    unsigned long long n
) {
    unsigned long long stride = (unsigned long long)gridDim.x * blockDim.x;
    for (unsigned long long i = (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;
         i < n; i += stride) {
        dst[i] = __float2half(src[i]);
    }
}

extern "C" __global__ void ssm_h_state_f16_to_f32(
    const __half* __restrict__ src,
    float* __restrict__ dst,
    unsigned long long n
) {
    unsigned long long stride = (unsigned long long)gridDim.x * blockDim.x;
    for (unsigned long long i = (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;
         i < n; i += stride) {
        dst[i] = __half2float(src[i]);
    }
}
