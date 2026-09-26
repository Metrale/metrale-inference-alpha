// SPDX-License-Identifier: MIT OR Apache-2.0

// 2026-09-25: Per-row FP8 (E4M3) quantization of a BF16 [R, K] matrix, for the row-wise
// (OUTER_VEC) cuBLASLt FP8 GEMM in crates/gpu-runtime/src/cublaslt/fp8.rs.
//
// scale[r] = max_k |X[r, k]| / 448, floored at 1e-12; X_fp8[r, k] = E4M3(X[r, k] / scale[r]),
// clamped to +-448. crates/model-layers/src/layers/ops/dispatch_proj_rowwise.rs launches it for
// the weight re-quant ([N, K] -> scale[N]) and the per-token activation quant ([M, K] -> scale[M]).
// Grid (R, 1, 1), block (RW_THREADS, 1, 1): one block per row, the threads stride over K.
//
// Owner: gb10 kernels.
// Invariants:
// - blockDim.x is RW_THREADS: the K stride and the per-warp max array assume it.
// - Every row r < R gets a scale[r] >= 1e-12, so an all-zero row encodes 0 rather than 0/0.


#include <cuda_bf16.h>
#include <cuda_fp8.h>

#define RW_FP8_MAX 448.0f
#define RW_THREADS 256

#if defined(__SCALE__) || defined(__HIP_PLATFORM_AMD__)
// 2026-09-25: Software E4M3 encode for the __SCALE__ and HIP builds, which do not use
// __nv_cvt_float_to_fp8. Mantissa ties round away from zero here; the CUDA path rounds to even.
__device__ __forceinline__ unsigned char rw_enc_fp8(float v) {
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

extern "C" __global__ void quant_rowwise_fp8(
    const __nv_bfloat16* __restrict__ X,
    unsigned char* __restrict__ X_fp8,
    float* __restrict__ scale,
    unsigned int R,
    unsigned int K
) {
    const unsigned int r = blockIdx.x;
    if (r >= R) return;
    const unsigned int tid = threadIdx.x;
    const unsigned long long base = (unsigned long long)r * K;


    float my_max = 0.0f;
    for (unsigned int k = tid; k < K; k += RW_THREADS) {
        my_max = fmaxf(my_max, fabsf(__bfloat162float(X[base + k])));
    }


    #pragma unroll
    for (int off = 16; off > 0; off >>= 1) {
        my_max = fmaxf(my_max, __shfl_down_sync(0xFFFFFFFF, my_max, off));
    }
    __shared__ float smem_warp_max[RW_THREADS / 32];
    const unsigned int warp_id = tid >> 5;
    const unsigned int lane = tid & 31;
    if (lane == 0) smem_warp_max[warp_id] = my_max;
    __syncthreads();

    __shared__ float smem_scale;
    if (tid == 0) {
        float gmax = 0.0f;
        #pragma unroll
        for (int i = 0; i < RW_THREADS / 32; i++) gmax = fmaxf(gmax, smem_warp_max[i]);
        float s = gmax / RW_FP8_MAX;
        if (s < 1e-12f) s = 1e-12f;
        scale[r] = s;
        smem_scale = s;
    }
    __syncthreads();


    const float inv = smem_scale;
    for (unsigned int k = tid; k < K; k += RW_THREADS) {
        float v = __bfloat162float(X[base + k]) / inv;
        v = fmaxf(fminf(v, RW_FP8_MAX), -RW_FP8_MAX);
#if defined(__SCALE__) || defined(__HIP_PLATFORM_AMD__)
        X_fp8[base + k] = rw_enc_fp8(v);
#else
        X_fp8[base + k] = (unsigned char)__nv_cvt_float_to_fp8(v, __NV_SATFINITE, __NV_E4M3);
#endif
    }
}
