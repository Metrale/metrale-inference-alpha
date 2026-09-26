// SPDX-License-Identifier: MIT OR Apache-2.0

// 2026-09-25: BF16 -> FP8 (E4M3) quantization with one FP32 scale per 128x128 tile, the load-time
// quantizer for BF16 expert weights (crates/model-layers/src/weight_map/quantize_fp8_bs.rs, called
// from crates/model-arch/src/weight_loader/longcat/ffn.rs).
//
// Per tile: scale = max|X| / 448, floored at 1e-12; byte = E4M3(X / scale), clamped to +-448.
// X_fp8 is [N, K] E4M3 row-major. block_scale is [ceil(N/128), ceil(K/128)] FP32 row-major,
// indexed n_block * k_blocks + k_block, which is how moe_fp8_grouped_gemm.cu reads it. Edge tiles
// are partial, so N and K need not be multiples of 128. Grid (ceil(K/128), ceil(N/128), 1), block
// (QBS_THREADS, 1, 1): one block per tile, an absmax pass and then an encode pass.
//
// Owner: gb10 kernels.
// Invariants:
// - k_blocks is ceil(K/128), the row stride of the consumer's scale table
//   (`(K + FP8_BLOCK - 1) / FP8_BLOCK` in moe_fp8_grouped_gemm.cu).
// - blockDim.x is QBS_THREADS: the element stride and the per-warp max array assume it.














#include <cuda_bf16.h>
#include <cuda_fp8.h>

#define QBS_FP8_MAX 448.0f
#define QBS_BLOCK 128
#define QBS_THREADS 256

#if defined(__SCALE__) || defined(__HIP_PLATFORM_AMD__)
// 2026-09-25: Software E4M3 encode for the __SCALE__ and HIP builds, the same code as rw_enc_fp8
// in quant_rowwise_fp8.cu. Mantissa ties round away from zero; the CUDA path rounds to even.
__device__ __forceinline__ unsigned char qbs_enc_fp8(float v) {
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

extern "C" __global__ void quantize_bf16_to_fp8_blockscaled(
    const __nv_bfloat16* __restrict__ X,
    unsigned char* __restrict__ X_fp8,
    float* __restrict__ block_scale,
    unsigned int N,
    unsigned int K
) {
    const unsigned int k_block = blockIdx.x;
    const unsigned int n_block = blockIdx.y;
    const unsigned int k_blocks = (K + QBS_BLOCK - 1) / QBS_BLOCK;

    const unsigned int n0 = n_block * QBS_BLOCK;
    const unsigned int k0 = k_block * QBS_BLOCK;
    if (n0 >= N || k0 >= K) return;

    const unsigned int n_len = min((unsigned int)QBS_BLOCK, N - n0);
    const unsigned int k_len = min((unsigned int)QBS_BLOCK, K - k0);
    const unsigned int count = n_len * k_len;

    const unsigned int tid = threadIdx.x;


    float my_max = 0.0f;
    for (unsigned int i = tid; i < count; i += QBS_THREADS) {
        const unsigned int r = i / k_len;
        const unsigned int c = i - r * k_len;
        const unsigned long long off = (unsigned long long)(n0 + r) * K + (k0 + c);
        my_max = fmaxf(my_max, fabsf(__bfloat162float(X[off])));
    }

    #pragma unroll
    for (int o = 16; o > 0; o >>= 1) {
        my_max = fmaxf(my_max, __shfl_down_sync(0xFFFFFFFF, my_max, o));
    }
    __shared__ float smem_warp_max[QBS_THREADS / 32];
    const unsigned int warp_id = tid >> 5;
    const unsigned int lane = tid & 31;
    if (lane == 0) smem_warp_max[warp_id] = my_max;
    __syncthreads();

    __shared__ float smem_scale;
    if (tid == 0) {
        float gmax = 0.0f;
        #pragma unroll
        for (int i = 0; i < QBS_THREADS / 32; i++) gmax = fmaxf(gmax, smem_warp_max[i]);
        float s = gmax / QBS_FP8_MAX;

        // 2026-09-25: Without the floor an all-zero tile gets scale 0, and its encode computes 0 / 0.
        if (s < 1e-12f) s = 1e-12f;
        block_scale[n_block * k_blocks + k_block] = s;
        smem_scale = s;
    }
    __syncthreads();


    const float s = smem_scale;
    for (unsigned int i = tid; i < count; i += QBS_THREADS) {
        const unsigned int r = i / k_len;
        const unsigned int c = i - r * k_len;
        const unsigned long long off = (unsigned long long)(n0 + r) * K + (k0 + c);
        float v = __bfloat162float(X[off]) / s;
        v = fmaxf(fminf(v, QBS_FP8_MAX), -QBS_FP8_MAX);
#if defined(__SCALE__) || defined(__HIP_PLATFORM_AMD__)
        X_fp8[off] = qbs_enc_fp8(v);
#else
        X_fp8[off] = (unsigned char)__nv_cvt_float_to_fp8(v, __NV_SATFINITE, __NV_E4M3);
#endif
    }
}
