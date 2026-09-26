// SPDX-License-Identifier: MIT OR Apache-2.0

// 2026-09-25: `mx_block_scale<E8M0>` and `metrale_dec_e4m3`: the per-block weight dequant scale, defined once for the
// kernels that include this header (common/moe_shared_expert_fused_t.cu and
// deepseek-v4-flash/nvfp4/moe_w4a16_grouped_gemm.cu), so they decode a scale byte to the same float.
//
// Owner: gb10 kernels.
// Invariants: mx_block_scale<true> returns 0 for scale bytes 0 and 255 and the float with bits b << 23,
// 2^(b - 127), otherwise: the same construction as metrale_core::mxfp4_e8m0::fp8_e8m0_to_f32.








#pragma once

#include <cuda_fp8.h>

// 2026-09-25: FP8 E4M3 byte to float. SCALE and HIP builds decode in software, mapping the NaN encoding
// (exponent 15, mantissa 7) to 0; other builds use the cuda_fp8.h conversion.

#if defined(__SCALE__) || defined(__HIP_PLATFORM_AMD__)
__device__ __forceinline__ float metrale_dec_e4m3(unsigned char b) {
    unsigned int s = (b >> 7) & 1u, e = (b >> 3) & 0xFu, m = b & 0x7u; float v;
    if (e == 0u)               v = (float)m * 0.001953125f;
    else if (e == 15u && m == 7u) v = 0.0f;
    else                       v = __uint_as_float(((e + 120u) << 23) | (m << 20));
    return s ? -v : v;
}
#else
__device__ __forceinline__ float metrale_dec_e4m3(unsigned char b) {
    __nv_fp8_e4m3 f; *(unsigned char*)&f = b; return (float)f;
}
#endif

// 2026-09-25: The dequant scale of one weight block.
//   E8M0 = false (NVFP4): the E4M3 scale byte times the per-tensor scale s2.
//   E8M0 = true (MXFP4): 2^(sb - 127) built from the exponent bits, with 0 for sb 0 and 255; s2 is unused.





template<bool E8M0>
__device__ __forceinline__ float mx_block_scale(unsigned char sb, float s2) {
    if (E8M0) {
        if (sb == 0u || sb == 255u) return 0.0f;
        return __uint_as_float((unsigned int)sb << 23);
    } else {
        return metrale_dec_e4m3(sb) * s2;
    }
}
