// SPDX-License-Identifier: MIT OR Apache-2.0

// 2026-09-25: Load-time BF16 -> NVFP4 weight quantization, plus an FP32 -> BF16 truncation kernel.
//
// nvfp4_global_absmax finds max |w| over the matrix, and the host sets scale2 = max / (6 * 448)
// (quantize_to_nvfp4 in crates/model-layers/src/weight_map/loaders_fp8.rs). quantize_bf16_to_nvfp4
// then writes packed [N, K/2] (two E2M1 codes per byte, the even element in the low nibble) and
// scales [N, K/16] (one E4M3 per 16 elements). Dequant: E2M1 value * E4M3 scale * scale2.
//
// Owner: gb10 kernels.
// Invariants:
// - No scale byte is the E4M3 NaN code: float_to_fp8_e4m3 saturates at 448 (exponent 15,
//   mantissa 6), NaN input included.


#include <cuda_bf16.h>
#include <cuda_fp8.h>

#define WARP_SIZE 32
#define GROUP_SIZE 16



// 2026-09-25: FP32 -> BF16 by keeping the high 16 bits of each value (truncation, which rounds
// toward zero), one element per thread. dense_f32_safe in
// crates/model-layers/src/weight_map/model_a.rs launches it at load for FP32 checkpoint tensors.
extern "C" __global__ void f32_to_bf16_trunc(
    const unsigned int* __restrict__ in,
    unsigned short* __restrict__ out,
    unsigned int n
) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        out[i] = (unsigned short)(in[i] >> 16);
    }
}



// 2026-09-25: Software float -> E4M3. Saturates at +-448, flushes magnitudes below 2^-9 to a
// signed zero, rounds mantissa ties away from zero, and never produces the NaN code.
__device__ unsigned char float_to_fp8_e4m3(float v) {
    unsigned int bits = __float_as_uint(v);
    unsigned int sign = (bits >> 31) & 1;
    int f32_exp = (int)((bits >> 23) & 0xFF) - 127;
    unsigned int f32_man = bits & 0x7FFFFF;


    if ((bits & 0x7FFFFFFF) == 0) return (unsigned char)(sign << 7);









    float absv = fabsf(v);
    if (absv > 448.0f) absv = 448.0f;


    bits = __float_as_uint(absv);
    f32_exp = (int)((bits >> 23) & 0xFF) - 127;
    f32_man = bits & 0x7FFFFF;

    int e4m3_exp;
    unsigned int e4m3_man;

    if (f32_exp < -9) {

        return (unsigned char)(sign << 7);
    } else if (f32_exp < -6) {


        int man = (int)(absv * 512.0f + 0.5f);
        if (man > 7) man = 7;
        if (man < 0) man = 0;
        return (unsigned char)((sign << 7) | man);
    } else {

        e4m3_exp = f32_exp + 7;
        if (e4m3_exp < 1) e4m3_exp = 1;
        if (e4m3_exp > 15) {

            e4m3_exp = 15;
            e4m3_man = 6;
        } else {

            e4m3_man = (f32_man + (1 << 19)) >> 20;
            if (e4m3_man > 7) {
                e4m3_man = 0;
                e4m3_exp++;
                if (e4m3_exp > 15) {
                    e4m3_exp = 15;
                    e4m3_man = 6;
                }
            }
        }
        return (unsigned char)((sign << 7) | (e4m3_exp << 3) | e4m3_man);
    }
}




// 2026-09-25: Nearest E2M1 code for v: the magnitude indexes {0, 0.5, 1, 1.5, 2, 3, 4, 6} and the
// sign is bit 3. A tie goes to the smaller magnitude; anything above 5 saturates to 6.
__device__ unsigned int quantize_e2m1(float v) {
    float absv = fabsf(v);
    unsigned int sign = (v < 0.0f) ? 8u : 0u;
    unsigned int idx;

    if      (absv <= 0.25f) idx = 0;
    else if (absv <= 0.75f) idx = 1;
    else if (absv <= 1.25f) idx = 2;
    else if (absv <= 1.75f) idx = 3;
    else if (absv <= 2.5f)  idx = 4;
    else if (absv <= 3.5f)  idx = 5;
    else if (absv <= 5.0f)  idx = 6;
    else                    idx = 7;

    return sign | idx;
}



// 2026-09-25: Grid-stride max |x| over total_elements, folded into *global_max with an unsigned
// atomicMax. The caller zeroes *global_max first. blockDim.x must be at most 256: smem holds
// one maximum per warp, 8 slots.
extern "C" __global__ void nvfp4_global_absmax(
    const __nv_bfloat16* __restrict__ input,
    float* __restrict__ global_max,
    unsigned int total_elements
) {
    float local_max = 0.0f;

    for (unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
         idx < total_elements;
         idx += gridDim.x * blockDim.x) {
        float v = fabsf(__bfloat162float(input[idx]));
        if (v > local_max) local_max = v;
    }


    #pragma unroll
    for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1) {
        float other = __shfl_down_sync(0xFFFFFFFF, local_max, offset);
        if (other > local_max) local_max = other;
    }


    __shared__ float smem[8];
    unsigned int warp_id = threadIdx.x / WARP_SIZE;
    unsigned int lane = threadIdx.x % WARP_SIZE;
    if (lane == 0) smem[warp_id] = local_max;
    __syncthreads();

    if (threadIdx.x == 0) {
        float block_max = 0.0f;
        for (int w = 0; w < (int)(blockDim.x / WARP_SIZE); w++) {
            if (smem[w] > block_max) block_max = smem[w];
        }
        // 2026-09-25: block_max >= 0, and non-negative floats order like their bits: a float max.
        atomicMax((unsigned int*)global_max, __float_as_uint(block_max));
    }
}




// 2026-09-25: One block per row, grid (N, 1, 1); the threads stride over the row's 16-element
// groups. A group's scale byte is E4M3(group_max / (6 * scale2)). Its elements are divided by
// the decoded scale byte times scale2, the scale the dequant applies, then rounded to E2M1.
// Assumes K is a multiple of 16: a tail is ignored.
extern "C" __global__ void quantize_bf16_to_nvfp4(
    const __nv_bfloat16* __restrict__ input,
    unsigned char* __restrict__ packed_out,
    unsigned char* __restrict__ scale_out,
    float scale2,
    unsigned int N,
    unsigned int K
) {
    unsigned int row = blockIdx.x;
    if (row >= N) return;

    const __nv_bfloat16* row_in = input + (unsigned long long)row * K;
    unsigned char* row_packed = packed_out + (unsigned long long)row * (K / 2);
    unsigned char* row_scale = scale_out + (unsigned long long)row * (K / GROUP_SIZE);

    float inv_scale2 = (scale2 > 0.0f) ? (1.0f / scale2) : 0.0f;
    unsigned int num_groups = K / GROUP_SIZE;

    for (unsigned int g = threadIdx.x; g < num_groups; g += blockDim.x) {
        unsigned int base = g * GROUP_SIZE;


        float group_max = 0.0f;
        #pragma unroll
        for (int i = 0; i < GROUP_SIZE; i++) {
            float v = fabsf(__bfloat162float(row_in[base + i]));
            if (v > group_max) group_max = v;
        }



        float fp8_float = (group_max > 0.0f) ? (group_max * inv_scale2 / 6.0f) : 0.0f;
        unsigned char fp8_byte = float_to_fp8_e4m3(fp8_float);
        row_scale[g] = fp8_byte;



        unsigned int fp8_sign = (fp8_byte >> 7) & 1;
        unsigned int fp8_exp = (fp8_byte >> 3) & 0xF;
        unsigned int fp8_man = fp8_byte & 0x7;
        float fp8_decoded;
        if (fp8_exp == 0) {
            fp8_decoded = (float)fp8_man * 0.001953125f;
        } else if (fp8_exp == 15 && fp8_man == 7) {
            fp8_decoded = 0.0f;
        } else {
            unsigned int f32_bits = ((fp8_exp + 120u) << 23) | (fp8_man << 20);
            fp8_decoded = __uint_as_float(f32_bits);
        }
        if (fp8_sign) fp8_decoded = -fp8_decoded;

        float effective_scale = fp8_decoded * scale2;
        float inv_eff = (effective_scale > 0.0f) ? (1.0f / effective_scale) : 0.0f;


        #pragma unroll
        for (int i = 0; i < GROUP_SIZE; i += 2) {
            float v0 = __bfloat162float(row_in[base + i]) * inv_eff;
            float v1 = __bfloat162float(row_in[base + i + 1]) * inv_eff;

            unsigned int n0 = quantize_e2m1(v0);
            unsigned int n1 = quantize_e2m1(v1);


            row_packed[g * 8 + i / 2] = (unsigned char)((n1 << 4) | (n0 & 0xF));
        }
    }
}
