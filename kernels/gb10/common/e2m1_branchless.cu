// SPDX-License-Identifier: MIT OR Apache-2.0

// 2026-09-25: FP32 -> E2M1 (4-bit) conversion by comparing the magnitude bits against seven
// thresholds, and a kernel that packs eight results per uint32.
//
// Owner: gb10 kernels.
// Invariants:
// - The thresholds are the IEEE-754 patterns of the midpoints 0.25, 0.75, 1.25, 1.75, 2.5,
//   3.5 and 5.0. `>` versus `>=` sends each tie to the neighbour with an even mantissa, so
//   the result is round-to-nearest-even; magnitudes above 5.0 (and NaN) give 6.0.
// - The sign bit becomes bit 3 of the code; code k (low three bits) is the k-th value of
//   {0, 0.5, 1, 1.5, 2, 3, 4, 6}.
// - e2m1_quantize writes only whole groups of 8: a tail of n % 8 inputs is not written.



__device__ __forceinline__ unsigned char branchless_float_to_e2m1(float x) {
    unsigned char sign = (unsigned char)((__float_as_uint(x) >> 28) & 8u);
    unsigned int abits = __float_as_uint(x) & 0x7FFFFFFFu;
    unsigned char mag = (abits >  0x3E800000u)
                      + (abits >= 0x3F400000u)
                      + (abits >  0x3FA00000u)
                      + (abits >= 0x3FE00000u)
                      + (abits >  0x40200000u)
                      + (abits >= 0x40600000u)
                      + (abits >  0x40A00000u);
    return sign | mag;
}

// 2026-09-25: Packs 8 E2M1 codes into one uint32, the first value in the low nibble.
__device__ __forceinline__ unsigned int pack_8xe2m1(
    float f0, float f1, float f2, float f3,
    float f4, float f5, float f6, float f7
) {
    unsigned int val = 0;
    val |= (unsigned int)branchless_float_to_e2m1(f0);
    val |= (unsigned int)branchless_float_to_e2m1(f1) << 4;
    val |= (unsigned int)branchless_float_to_e2m1(f2) << 8;
    val |= (unsigned int)branchless_float_to_e2m1(f3) << 12;
    val |= (unsigned int)branchless_float_to_e2m1(f4) << 16;
    val |= (unsigned int)branchless_float_to_e2m1(f5) << 20;
    val |= (unsigned int)branchless_float_to_e2m1(f6) << 24;
    val |= (unsigned int)branchless_float_to_e2m1(f7) << 28;
    return val;
}


extern "C" __global__ void e2m1_quantize(
    const float* __restrict__ input,
    unsigned int* __restrict__ output,
    unsigned int n
) {
    unsigned int idx = (blockIdx.x * blockDim.x + threadIdx.x) * 8;
    if (idx + 7 < n) {
        output[idx / 8] = pack_8xe2m1(
            input[idx], input[idx+1], input[idx+2], input[idx+3],
            input[idx+4], input[idx+5], input[idx+6], input[idx+7]
        );
    }
}
