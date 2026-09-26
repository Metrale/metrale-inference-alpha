// SPDX-License-Identifier: MIT OR Apache-2.0

// 2026-09-25: MLA cache assembly for N tokens: BF16 K and V latents to E4M3 K and V cache rows.
//
// Owner: gb10 kernels (deepseek-v4-flash).
// Invariants: each head's K row is [latent | rope] and its V row is [latent | K's rope], both
// mla_cache_dim bytes; K values are scaled by k_scale and V values by v_scale before a
// saturating (__NV_SATFINITE) E4M3 conversion.



#include <cuda_bf16.h>
#include <cuda_fp8.h>

#define BLOCK_SIZE 256



__device__ __forceinline__ unsigned char bf16_to_fp8(__nv_bfloat16 b) {
    return __nv_cvt_float_to_fp8(__bfloat162float(b), __NV_SATFINITE, __NV_E4M3);
}

// 2026-09-25: Grid (num_tokens), block BLOCK_SIZE; blockIdx.x is the token.
// k_bf16 [N, nkv * mla_cache_dim] with rope applied, v_bf16 [N, nkv * kv_lora], and
// k_cache_fp8 / v_cache_fp8 [N, nkv * mla_cache_dim], where mla_cache_dim is meant to be kv_lora + rope;
// cache bytes at or past kv_lora + rope are not written.





extern "C" __global__ void mla_cache_assemble_fp8_batched(
    const __nv_bfloat16* __restrict__ k_bf16,
    const __nv_bfloat16* __restrict__ v_bf16,
    unsigned char* __restrict__ k_cache_fp8,
    unsigned char* __restrict__ v_cache_fp8,
    unsigned int num_tokens,
    unsigned int nkv,
    unsigned int kv_lora,
    unsigned int rope,
    unsigned int mla_cache_dim,
    float k_scale,
    float v_scale
) {
    unsigned int t = blockIdx.x;
    unsigned int idx = threadIdx.x;

    const unsigned long long k_bf16_offset = (unsigned long long)t * nkv * mla_cache_dim;
    const unsigned long long v_bf16_offset = (unsigned long long)t * nkv * kv_lora;
    const unsigned long long k_cache_offset = (unsigned long long)t * nkv * mla_cache_dim;
    const unsigned long long v_cache_offset = (unsigned long long)t * nkv * mla_cache_dim;


    for (unsigned int d = idx; d < mla_cache_dim; d += BLOCK_SIZE) {
        if (d < kv_lora) {

            for (unsigned int head = 0; head < nkv; head++) {
                unsigned long long k_idx = k_bf16_offset + head * mla_cache_dim + d;
                unsigned long long v_idx = v_bf16_offset + head * kv_lora + d;
                unsigned long long k_cache_idx = k_cache_offset + head * mla_cache_dim + d;
                unsigned long long v_cache_idx = v_cache_offset + head * mla_cache_dim + d;


                float k_val = __bfloat162float(k_bf16[k_idx]);
                float v_val = __bfloat162float(v_bf16[v_idx]);

                k_cache_fp8[k_cache_idx] = __nv_cvt_float_to_fp8(k_val * k_scale, __NV_SATFINITE, __NV_E4M3);
                v_cache_fp8[v_cache_idx] = __nv_cvt_float_to_fp8(v_val * v_scale, __NV_SATFINITE, __NV_E4M3);
            }
        } else if (d < kv_lora + rope) {
            // 2026-09-25: V's rope tail is K's: the checkpoint's inference/model.py passes one kv
            // tensor to sparse_attn as both key and value.




            for (unsigned int head = 0; head < nkv; head++) {
                unsigned long long k_idx = k_bf16_offset + head * mla_cache_dim + d;
                unsigned long long k_cache_idx = k_cache_offset + head * mla_cache_dim + d;
                unsigned long long v_cache_idx = v_cache_offset + head * mla_cache_dim + d;


                float k_val = __bfloat162float(k_bf16[k_idx]);
                k_cache_fp8[k_cache_idx] = __nv_cvt_float_to_fp8(k_val * k_scale, __NV_SATFINITE, __NV_E4M3);
                v_cache_fp8[v_cache_idx] = __nv_cvt_float_to_fp8(k_val * v_scale, __NV_SATFINITE, __NV_E4M3);
            }
        }
    }
}