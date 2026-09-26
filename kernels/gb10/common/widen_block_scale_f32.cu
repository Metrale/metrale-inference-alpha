// SPDX-License-Identifier: MIT OR Apache-2.0

// 2026-09-25: Kernel `widen_block_scale_f32`: widen a block-scale tensor to FP32 at weight load, so every FP8
// block-scale kernel can read `const float*`. One element per thread; the launcher uses Grid (ceil(total/256), 1, 1),
// Block (256, 1, 1).
//
// input_dtype 1: src is `const float*`, copied. input_dtype 2: src is F8_E8M0 bytes, widened to a power of two.
// Any other value: src is `const __nv_bfloat16*`, widened (the loader passes 0 for BF16).
//
// Owner: gb10 kernels.
// Invariants: none beyond the types. Threads with i >= total return before any access.







#include <cuda_bf16.h>

extern "C" __global__ void widen_block_scale_f32(
    const void* __restrict__ src,
    float* __restrict__ dst,
    unsigned int total,
    unsigned int input_dtype
) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= total) return;

    if (input_dtype == 1) {
        dst[i] = ((const float*)src)[i];
    } else if (input_dtype == 2) {
        unsigned int exp = ((const unsigned char*)src)[i];
        // 2026-09-25: E8M0 has no zero encoding: exp 0 is 2^-127 (the FP32 subnormal 0x00400000). exp 255,
        // the NaN encoding, is written as 0.0f.

        dst[i] = (exp == 255u)
                     ? 0.0f
                     : (exp == 0u ? __uint_as_float(0x00400000u)
                                  : __uint_as_float(exp << 23));
    } else {
        unsigned short raw = ((const unsigned short*)src)[i];
        dst[i] = __bfloat162float(*(const __nv_bfloat16*)&raw);
    }
}
