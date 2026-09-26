// SPDX-License-Identifier: MIT OR Apache-2.0

// 2026-09-25: Per-token, per-128-K-group FP8 E4M3 quantization of BF16 activations.
//
// For token m and K-group g:
//   a_scale[m, g] = max(max_k |A[m, g*128 + k]| / 448, 1e-12)
//   A_fp8[m, k]   = E4M3(clamp(A[m, k] / a_scale[m, g], -448, 448))
// a_scale is row-major [M, K/128], the layout fp8_gemm_t_blockscaled.cu reads.
//
// Owner: gb10 kernels.
// Invariants:
// - blockDim.x is 128: one thread per element of a group, one smem slot per
//   warp. ops::per_token_group_quant_fp8 launches grid (M, K/128, 1); M is on
//   grid X, the one dimension not capped at 65535.

#include <cuda_bf16.h>
#include <cuda_fp8.h>

#define FP8_GROUP_K 128
#define FP8_E4M3_MAX 448.0f

#if defined(__SCALE__) || defined(__HIP_PLATFORM_AMD__)
// 2026-09-25: SCALE and HIP builds encode E4M3 in software: NaN to 0x7F,
// saturation to magnitude 0x7E (448), mantissa rounded half up.


__device__ __forceinline__ unsigned char scl_enc_fp8(float v) {
    if (v != v) return 0x7F;
    unsigned int bb = __float_as_uint(v); unsigned int sign = (bb >> 31) & 1u;
    int e = (int)((bb >> 23) & 0xFF) - 127; unsigned int man = bb & 0x7FFFFFu;
    int ee = e + 7; unsigned int em;
    if (ee < 1) { ee = 0; em = 0; if (e >= -10) { float a = v < 0 ? -v : v; em = (unsigned int)(a / 0.001953125f + 0.5f); if (em > 7u) em = 7u; } }
    else if (ee > 15) { ee = 15; em = 6; }
    else { em = (man + (1u << 19)) >> 20; if (em > 7u) { em = 0; ee++; if (ee > 15) { ee = 15; em = 6; } } }
    return (unsigned char)((sign << 7) | ((unsigned)ee << 3) | em);
}
#endif

extern "C" __global__ void per_token_group_quant_fp8(
    const __nv_bfloat16* __restrict__ A,
    unsigned char* __restrict__ A_fp8,
    float* __restrict__ a_scale,
    unsigned int M,
    unsigned int K
) {
    const unsigned int m = blockIdx.x;
    const unsigned int kg = blockIdx.y;
    if (m >= M) return;

    const unsigned int k_start = kg * FP8_GROUP_K;
    const unsigned int tid = threadIdx.x;


    float my_abs = 0.0f;
    if (tid < FP8_GROUP_K && k_start + tid < K) {
        my_abs = fabsf(__bfloat162float(A[m * K + k_start + tid]));
    }


    float warp_max = my_abs;
    #pragma unroll
    for (int off = 16; off > 0; off >>= 1) {
        warp_max = fmaxf(warp_max, __shfl_down_sync(0xFFFFFFFF, warp_max, off));
    }
    __shared__ float smem_warp_max[4];
    const unsigned int warp_id = tid >> 5;
    const unsigned int lane = tid & 31;
    if (lane == 0) smem_warp_max[warp_id] = warp_max;
    __syncthreads();

    __shared__ float smem_scale;
    if (tid == 0) {
        float global_max = 0.0f;
        #pragma unroll
        for (int i = 0; i < 4; i++) global_max = fmaxf(global_max, smem_warp_max[i]);

        float scale = global_max / FP8_E4M3_MAX;
        if (scale < 1e-12f) scale = 1e-12f;
        a_scale[m * (K / FP8_GROUP_K) + kg] = scale;
        smem_scale = scale;
    }
    __syncthreads();


    if (tid < FP8_GROUP_K && k_start + tid < K) {
        float v = __bfloat162float(A[m * K + k_start + tid]) / smem_scale;

        v = fmaxf(fminf(v, FP8_E4M3_MAX), -FP8_E4M3_MAX);
#if defined(__SCALE__) || defined(__HIP_PLATFORM_AMD__)
        A_fp8[m * K + k_start + tid] = scl_enc_fp8(v);
#else
        __nv_fp8_storage_t fp8_v = __nv_cvt_float_to_fp8(v, __NV_SATFINITE, __NV_E4M3);
        A_fp8[m * K + k_start + tid] = (unsigned char)fp8_v;
#endif
    }
}
