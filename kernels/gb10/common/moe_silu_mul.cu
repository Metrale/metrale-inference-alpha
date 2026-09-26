// SPDX-License-Identifier: MIT OR Apache-2.0

// 2026-09-25: MoE SiLU(gate) * up, element-wise (moe_silu_mul) and fused with per-128-group
// FP8 E4M3 quantization (silu_mul_quant_fp8).
//
// Owner: gb10 kernels.
// Invariants:
// - No swiglu clamp is applied. The deepseek-v4-flash and step3p7-flash trees have their
//   own moe_silu_mul.cu, which clamps gate to at most 10 and up to [-10, 10].
// - moe_silu_mul: output[i] = silu(gate[i]) * up[i], silu(x) = x * sigmoid(x), one thread
//   per element; ops::moe_silu_mul launches grid ceil(total_elements / 256), block 256.























#include <cuda_bf16.h>
#include <cuda_fp8.h>

extern "C" __global__ void moe_silu_mul(
    const __nv_bfloat16* __restrict__ gate,
    const __nv_bfloat16* __restrict__ up,
    __nv_bfloat16* __restrict__ output,
    unsigned int total_elements
) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total_elements) return;

    float g = __bfloat162float(gate[idx]);
    float u = __bfloat162float(up[idx]);
    float sigmoid_g = 1.0f / (1.0f + __expf(-g));
    float result = g * sigmoid_g * u;
    output[idx] = __float2bfloat16(result);
}

// 2026-09-25: SiLU(gate) * up fused with per-128-group FP8 E4M3 quantization. It writes the
// same bytes and scales as moe_silu_mul followed by per_token_group_quant_fp8:
// - the product is rounded to BF16 before the group max and the encode, as the unfused
//   pair stores it and reads it back;
// - the max reduction has per_token_group_quant_fp8's structure (warp __shfl_down_sync,
//   four shared slots, sequential fmaxf on thread 0), and the 1e-12 scale floor, the
//   clamp to 448 and the SATFINITE encode are the same.
// out_bf16 may be null; when set it also receives the BF16 product.
// Launch: grid M, block 128 (one block per row). K must be a multiple of 128 with
// K / 128 <= SILU_QUANT_MAX_GROUPS (16); MoeLayer::fused_silu_quant_ok runs the unfused
// pair otherwise. The deepseek-v4-flash and step3p7-flash moe_silu_mul.cu files do not
// define this kernel, so its handle is 0 for those models and the pair runs.




















#define FP8_GROUP_K 128
#define FP8_E4M3_MAX 448.0f
#define SILU_QUANT_MAX_GROUPS 16

#if defined(__SCALE__) || defined(__HIP_PLATFORM_AMD__)
// 2026-09-25: Software SATFINITE E4M3 encode, the same code as scl_enc_fp8 in
// per_token_group_quant_fp8.cu.


__device__ __forceinline__ unsigned char silu_quant_enc_fp8(float v) {
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

extern "C" __global__ void silu_mul_quant_fp8(
    const __nv_bfloat16* __restrict__ gate,
    const __nv_bfloat16* __restrict__ up,
    unsigned char* __restrict__ out_fp8,
    float* __restrict__ a_scale,
    __nv_bfloat16* __restrict__ out_bf16,
    unsigned int M,
    unsigned int K
) {
    const unsigned int m = blockIdx.x;
    if (m >= M) return;
    const unsigned int tid = threadIdx.x;
    const unsigned int ngroups = K / FP8_GROUP_K;
    const __nv_bfloat16* grow = gate + (unsigned long long)m * K;
    const __nv_bfloat16* urow = up + (unsigned long long)m * K;
    unsigned char* orow = out_fp8 + (unsigned long long)m * K;

    __shared__ float smem_warp_max[4];
    __shared__ float smem_scale;
    const unsigned int warp_id = tid >> 5;
    const unsigned int lane = tid & 31;

    float vals[SILU_QUANT_MAX_GROUPS];
    for (unsigned int kg = 0; kg < ngroups; kg++) {
        const unsigned int k = kg * FP8_GROUP_K + tid;
        float g = __bfloat162float(grow[k]);
        float u = __bfloat162float(urow[k]);
        float sigmoid_g = 1.0f / (1.0f + __expf(-g));
        float result = g * sigmoid_g * u;
        // 2026-09-25: Round to BF16 first: the quantizer input must equal the BF16 value the
        // unfused pair stores and reads back.
        __nv_bfloat16 r16 = __float2bfloat16(result);
        if (out_bf16 != nullptr) out_bf16[(unsigned long long)m * K + k] = r16;
        float r = __bfloat162float(r16);
        vals[kg] = r;

        float warp_max = fabsf(r);
        #pragma unroll
        for (int off = 16; off > 0; off >>= 1) {
            warp_max = fmaxf(warp_max, __shfl_down_sync(0xFFFFFFFF, warp_max, off));
        }
        if (lane == 0) smem_warp_max[warp_id] = warp_max;
        __syncthreads();
        if (tid == 0) {
            float global_max = 0.0f;
            #pragma unroll
            for (int i = 0; i < 4; i++) global_max = fmaxf(global_max, smem_warp_max[i]);
            float scale = global_max / FP8_E4M3_MAX;
            if (scale < 1e-12f) scale = 1e-12f;
            a_scale[m * ngroups + kg] = scale;
            smem_scale = scale;
        }
        __syncthreads();
        float v = vals[kg] / smem_scale;
        v = fmaxf(fminf(v, FP8_E4M3_MAX), -FP8_E4M3_MAX);
#if defined(__SCALE__) || defined(__HIP_PLATFORM_AMD__)
        orow[k] = silu_quant_enc_fp8(v);
#else
        orow[k] = (unsigned char)__nv_cvt_float_to_fp8(v, __NV_SATFINITE, __NV_E4M3);
#endif
        __syncthreads();
    }
}
