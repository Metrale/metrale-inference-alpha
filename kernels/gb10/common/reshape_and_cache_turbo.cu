// SPDX-License-Identifier: MIT OR Apache-2.0

// 2026-09-25: TurboQuant KV cache writes: BF16 K and V rows quantized per 16-element group into
// turbo8, turbo4, turbo3 or turbo2, and asymmetric pairs that give K and V different formats.
// - turbo4 / turbo3 / turbo2: 4-, 3- or 2-bit codebook indices plus one E4M3 scale byte per group.
// - turbo8: one E4M3 byte per element plus one BF16 scale per group.
// These kernels do not rotate. The caller applies the Walsh-Hadamard rotation (wht_bf16.cu) to a
// turbo side first, or the weights are pre-rotated (crates/model-layers/src/layers/qwen3_attention/
// decode/write_kv_cache.rs). One block per token, grid (num_tokens, 1, 1).
//
// Owner: gb10 kernels.
// Invariants:
// - A token whose slot is negative writes nothing.
// - A pool block is a data section followed, at its data-section size, by a scale section; a
//   slot's data and scales sit at block_offset times their per-token size in each.

#include <cuda_bf16.h>

#if defined(__SCALE__) || defined(__HIP_PLATFORM_AMD__)

// 2026-09-25: Software E4M3 decode and encode for the __SCALE__ and HIP builds; float_to_fp8
// uses scl_enc_fp8 there, and scl_fp8 has no caller. Mantissa ties round away from zero,
// while the CUDA path (cvt.rn.satfinite) rounds them to even.
__device__ __forceinline__ float scl_fp8(unsigned char b) {
    unsigned int s = (b >> 7) & 1u, e = (b >> 3) & 0xFu, m = b & 0x7u; float v;
    if (e == 0u)               v = (float)m * 0.001953125f;
    else if (e == 15u && m == 7u) v = 0.0f;
    else                       v = __uint_as_float(((e + 120u) << 23) | (m << 20));
    return s ? -v : v;
}
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

#include <cuda_fp8.h>

#define GROUP_SIZE 16


// 2026-09-25: Symmetric codebooks. Each bound is the midpoint of two adjacent levels, so the
// quantize helpers pick the nearest level; no level is 0.
__device__ __constant__ float TURBO4_CODEBOOK[16] = {
    -2.7326f, -2.0690f, -1.6180f, -1.2562f, -0.9423f, -0.6568f, -0.3880f, -0.1284f,
     0.1284f,  0.3880f,  0.6568f,  0.9423f,  1.2562f,  1.6180f,  2.0690f,  2.7326f
};
__device__ __constant__ float TURBO4_BOUNDS[15] = {
    -2.4008f, -1.8435f, -1.4371f, -1.0993f, -0.7996f, -0.5224f, -0.2582f, 0.0f,
     0.2582f,  0.5224f,  0.7996f,  1.0993f,  1.4371f,  1.8435f,  2.4008f
};
#define TURBO4_MAX 2.7326f


__device__ __constant__ float TURBO3_CODEBOOK[8] = {
    -2.1520f, -1.3440f, -0.7560f, -0.2451f, 0.2451f, 0.7560f, 1.3440f, 2.1520f
};
__device__ __constant__ float TURBO3_BOUNDS[7] = {
    -1.748f, -1.050f, -0.501f, 0.0f, 0.501f, 1.050f, 1.748f
};
#define TURBO3_MAX 2.1520f




__device__ __constant__ float TURBO2_CODEBOOK[4] = {
    -1.5104f, -0.4528f, 0.4528f, 1.5104f
};
__device__ __constant__ float TURBO2_BOUNDS[3] = {
    -0.9816f, 0.0f, 0.9816f
};
#define TURBO2_MAX 1.5104f


// 2026-09-25: FP32 -> E4M3: cvt.rn.satfinite on CUDA, scl_enc_fp8 elsewhere. Every caller caps |val| at 448 first.
__device__ __forceinline__ __nv_fp8_storage_t float_to_fp8(float val) {
#if defined(__SCALE__)




    return scl_enc_fp8(val);
#elif defined(__HIP_PLATFORM_AMD__)




    return (__nv_fp8_storage_t)scl_enc_fp8(val);
#else
    unsigned short pair;
    asm volatile("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %1;"
                 : "=h"(pair) : "f"(val));
    return (__nv_fp8_storage_t)(pair & 0xFF);
#endif
}


#define FP8_E4M3_MAX 448.0f

// 2026-09-25: In-place 256-point Walsh-Hadamard transform over one warp, element lane * 8 + i in
// vals[i], scaled by 1/16. No kernel in this file calls it.

__device__ __forceinline__ void wht256_warp(float vals[8], unsigned int lane) {

    #pragma unroll
    for (int stride = 1; stride <= 4; stride <<= 1) {
        #pragma unroll
        for (int i = 0; i < 8; i += stride * 2) {
            for (int j = 0; j < stride; j++) {
                float a = vals[i + j];
                float b = vals[i + j + stride];
                vals[i + j] = a + b;
                vals[i + j + stride] = a - b;
            }
        }
    }

    #pragma unroll
    for (int xor_mask = 1; xor_mask <= 16; xor_mask <<= 1) {
        #pragma unroll
        for (int i = 0; i < 8; i++) {
            float other = __shfl_xor_sync(0xFFFFFFFF, vals[i], xor_mask);

            if (lane & xor_mask)
                vals[i] = other - vals[i];
            else
                vals[i] = vals[i] + other;
        }
    }

    #pragma unroll
    for (int i = 0; i < 8; i++) vals[i] *= 0.0625f;
}


// 2026-09-25: Nearest-level index: the number of bounds at or below x, by binary search.
__device__ __forceinline__ unsigned char turbo4_quantize(float x) {

    unsigned char idx = 0;
    if (x >= TURBO4_BOUNDS[7]) {
        idx = 8;
        if (x >= TURBO4_BOUNDS[11]) { idx = 12; if (x >= TURBO4_BOUNDS[13]) { idx = 14; if (x >= TURBO4_BOUNDS[14]) idx = 15; } else if (x >= TURBO4_BOUNDS[12]) idx = 13; }
        else { if (x >= TURBO4_BOUNDS[9]) { idx = 10; if (x >= TURBO4_BOUNDS[10]) idx = 11; } else if (x >= TURBO4_BOUNDS[8]) idx = 9; }
    } else {
        if (x >= TURBO4_BOUNDS[3]) { idx = 4; if (x >= TURBO4_BOUNDS[5]) { idx = 6; if (x >= TURBO4_BOUNDS[6]) idx = 7; } else if (x >= TURBO4_BOUNDS[4]) idx = 5; }
        else { if (x >= TURBO4_BOUNDS[1]) { idx = 2; if (x >= TURBO4_BOUNDS[2]) idx = 3; } else if (x >= TURBO4_BOUNDS[0]) idx = 1; }
    }
    return idx;
}

__device__ __forceinline__ unsigned char turbo2_quantize(float x) {

    if (x >= TURBO2_BOUNDS[1]) return (x >= TURBO2_BOUNDS[2]) ? 3 : 2;
    else                       return (x >= TURBO2_BOUNDS[0]) ? 1 : 0;
}

__device__ __forceinline__ unsigned char turbo3_quantize(float x) {

    unsigned char idx = 0;
    if (x >= TURBO3_BOUNDS[3]) {
        idx = 4;
        if (x >= TURBO3_BOUNDS[5]) { idx = 6; if (x >= TURBO3_BOUNDS[6]) idx = 7; }
        else if (x >= TURBO3_BOUNDS[4]) idx = 5;
    } else {
        if (x >= TURBO3_BOUNDS[1]) { idx = 2; if (x >= TURBO3_BOUNDS[2]) idx = 3; }
        else if (x >= TURBO3_BOUNDS[0]) idx = 1;
    }
    return idx;
}


// 2026-09-25: turbo4 K and V: 16 codes per group in 8 bytes, the even element in the low nibble.
extern "C" __global__ void reshape_and_cache_flash_turbo4(
    const __nv_bfloat16* __restrict__ key,
    const __nv_bfloat16* __restrict__ value,
    unsigned char* __restrict__ k_cache,
    unsigned char* __restrict__ v_cache,
    const long long* __restrict__ slot_mapping,
    const unsigned int num_kv_heads,
    const unsigned int head_dim,
    const unsigned int block_size,
    const unsigned int key_stride,
    const unsigned int value_stride,
    const unsigned long long block_stride_bytes,
    const unsigned long long data_section_bytes
) {
    const unsigned int token_idx = blockIdx.x;
    const long long slot = slot_mapping[token_idx];
    if (slot < 0) return;

    const unsigned int block_idx = (unsigned int)(slot / block_size);
    const unsigned int block_offset = (unsigned int)(slot % block_size);
    const unsigned int n_elems = num_kv_heads * head_dim;
    const unsigned int num_groups = n_elems / GROUP_SIZE;

    const __nv_bfloat16* key_src = key + (unsigned long long)token_idx * key_stride;
    const __nv_bfloat16* val_src = value + (unsigned long long)token_idx * value_stride;

    unsigned char* block_k = k_cache + (unsigned long long)block_idx * block_stride_bytes;
    unsigned char* block_v = v_cache + (unsigned long long)block_idx * block_stride_bytes;
    unsigned long long data_off = (unsigned long long)block_offset * (n_elems / 2);
    unsigned long long scale_off = data_section_bytes + (unsigned long long)block_offset * num_groups;




    // 2026-09-25: Codes are picked at x * TURBO4_MAX / amax. The group scale is then
    // ||x|| / ||chosen levels||, so the dequantized group keeps the input's L2 norm, capped at 448
    // before its E4M3 encode. No level is 0, so ||chosen levels|| > 0 and the amax fallback is not
    // reached; an all-zero group gets scale 0.
    for (unsigned int g = threadIdx.x; g < num_groups; g += blockDim.x) {
        unsigned int elem_offset = g * GROUP_SIZE;

        float kf[16], vf[16];
        float k_norm_sq = 0.0f, v_norm_sq = 0.0f;
        for (int i = 0; i < 16; i++) {
            kf[i] = __bfloat162float(key_src[elem_offset + i]);
            vf[i] = __bfloat162float(val_src[elem_offset + i]);
            k_norm_sq += kf[i] * kf[i];
            v_norm_sq += vf[i] * vf[i];
        }

        float k_max = 0.0f, v_max = 0.0f;
        for (int i = 0; i < 16; i++) {
            k_max = fmaxf(k_max, fabsf(kf[i]));
            v_max = fmaxf(v_max, fabsf(vf[i]));
        }

        float k_inv = (k_max > 1e-12f) ? (TURBO4_MAX / k_max) : 1.0f;
        float v_inv = (v_max > 1e-12f) ? (TURBO4_MAX / v_max) : 1.0f;


        unsigned char k_idx[16], v_idx[16];
        float k_recon_sq = 0.0f, v_recon_sq = 0.0f;
        for (int i = 0; i < 16; i++) {
            k_idx[i] = turbo4_quantize(kf[i] * k_inv);
            v_idx[i] = turbo4_quantize(vf[i] * v_inv);
            float kc = TURBO4_CODEBOOK[k_idx[i]];
            float vc = TURBO4_CODEBOOK[v_idx[i]];
            k_recon_sq += kc * kc;
            v_recon_sq += vc * vc;
        }
        float k_recon_norm = sqrtf(k_recon_sq);
        float v_recon_norm = sqrtf(v_recon_sq);



        float ks = (k_recon_norm > 1e-10f) ? (sqrtf(k_norm_sq) / k_recon_norm) : (k_max / TURBO4_MAX);
        float vs = (v_recon_norm > 1e-10f) ? (sqrtf(v_norm_sq) / v_recon_norm) : (v_max / TURBO4_MAX);
        if (ks > FP8_E4M3_MAX) ks = FP8_E4M3_MAX;
        if (vs > FP8_E4M3_MAX) vs = FP8_E4M3_MAX;

        ((__nv_fp8_storage_t*)(block_k + scale_off))[g] = float_to_fp8(ks);
        ((__nv_fp8_storage_t*)(block_v + scale_off))[g] = float_to_fp8(vs);

        unsigned char* kd = block_k + data_off + elem_offset / 2;
        unsigned char* vd = block_v + data_off + elem_offset / 2;
        for (int i = 0; i < 16; i += 2) {
            kd[i/2] = k_idx[i] | (k_idx[i+1] << 4);
            vd[i/2] = v_idx[i] | (v_idx[i+1] << 4);
        }
    }
}



// 2026-09-25: turbo8 K and V: E4M3(x / scale) per element, clamped to +-448, with
// scale = amax / 448 floored at 1e-12, stored as BF16 per group.
extern "C" __global__ void reshape_and_cache_flash_turbo8(
    const __nv_bfloat16* __restrict__ key,
    const __nv_bfloat16* __restrict__ value,
    unsigned char* __restrict__ k_cache,
    unsigned char* __restrict__ v_cache,
    const long long* __restrict__ slot_mapping,
    const unsigned int num_kv_heads,
    const unsigned int head_dim,
    const unsigned int block_size,
    const unsigned int key_stride,
    const unsigned int value_stride,
    const unsigned long long block_stride_bytes,
    const unsigned long long data_section_bytes
) {
    const unsigned int token_idx = blockIdx.x;
    const long long slot = slot_mapping[token_idx];
    if (slot < 0) return;

    const unsigned int block_idx = (unsigned int)(slot / block_size);
    const unsigned int block_offset = (unsigned int)(slot % block_size);
    const unsigned int n_elems = num_kv_heads * head_dim;
    const unsigned int num_groups = n_elems / GROUP_SIZE;

    const __nv_bfloat16* key_src = key + (unsigned long long)token_idx * key_stride;
    const __nv_bfloat16* val_src = value + (unsigned long long)token_idx * value_stride;





    // 2026-09-25: data [block_size, n_elems] bytes, scales [block_size, num_groups] BF16, the layout
    // dequant_turbo8_block_to_bf16 in crates/cache/src/kv_dequant.rs reads.
    unsigned char* block_k = k_cache + (unsigned long long)block_idx * block_stride_bytes;
    unsigned char* block_v = v_cache + (unsigned long long)block_idx * block_stride_bytes;
    unsigned long long data_off = (unsigned long long)block_offset * n_elems;

    unsigned long long scale_off =
        data_section_bytes + (unsigned long long)block_offset * num_groups * 2;


    for (unsigned int g = threadIdx.x; g < num_groups; g += blockDim.x) {
        unsigned int elem_offset = g * GROUP_SIZE;

        float kf[16], vf[16];
        for (int i = 0; i < 16; i++) {
            kf[i] = __bfloat162float(key_src[elem_offset + i]);
            vf[i] = __bfloat162float(val_src[elem_offset + i]);
        }


        float k_max = 0.0f, v_max = 0.0f;
        for (int i = 0; i < 16; i++) {
            k_max = fmaxf(k_max, fabsf(kf[i]));
            v_max = fmaxf(v_max, fabsf(vf[i]));
        }

        float k_scale = k_max / FP8_E4M3_MAX;
        float v_scale = v_max / FP8_E4M3_MAX;
        if (k_scale < 1e-12f) k_scale = 1e-12f;
        if (v_scale < 1e-12f) v_scale = 1e-12f;


        ((__nv_bfloat16*)(block_k + scale_off))[g] = __float2bfloat16(k_scale);
        ((__nv_bfloat16*)(block_v + scale_off))[g] = __float2bfloat16(v_scale);


        float k_inv = 1.0f / k_scale;
        float v_inv = 1.0f / v_scale;
        unsigned char* kd = block_k + data_off + elem_offset;
        unsigned char* vd = block_v + data_off + elem_offset;
        for (int i = 0; i < 16; i++) {
            float ks = fminf(fmaxf(kf[i] * k_inv, -FP8_E4M3_MAX), FP8_E4M3_MAX);
            float vs = fminf(fmaxf(vf[i] * v_inv, -FP8_E4M3_MAX), FP8_E4M3_MAX);
            kd[i] = (unsigned char)float_to_fp8(ks);
            vd[i] = (unsigned char)float_to_fp8(vs);
        }
    }
}

// 2026-09-25: turbo3 K and V: 16 codes per group in 6 bytes, each run of 8 codes in 3 bytes with
// the first code in the low bits. Scales as in turbo4.
extern "C" __global__ void reshape_and_cache_flash_turbo3(
    const __nv_bfloat16* __restrict__ key,
    const __nv_bfloat16* __restrict__ value,
    unsigned char* __restrict__ k_cache,
    unsigned char* __restrict__ v_cache,
    const long long* __restrict__ slot_mapping,
    const unsigned int num_kv_heads,
    const unsigned int head_dim,
    const unsigned int block_size,
    const unsigned int key_stride,
    const unsigned int value_stride,
    const unsigned long long block_stride_bytes,
    const unsigned long long data_section_bytes
) {
    const unsigned int token_idx = blockIdx.x;
    const long long slot = slot_mapping[token_idx];
    if (slot < 0) return;

    const unsigned int block_idx = (unsigned int)(slot / block_size);
    const unsigned int block_offset = (unsigned int)(slot % block_size);
    const unsigned int n_elems = num_kv_heads * head_dim;
    const unsigned int num_groups = n_elems / GROUP_SIZE;

    const __nv_bfloat16* key_src = key + (unsigned long long)token_idx * key_stride;
    const __nv_bfloat16* val_src = value + (unsigned long long)token_idx * value_stride;


    unsigned char* block_k = k_cache + (unsigned long long)block_idx * block_stride_bytes;
    unsigned char* block_v = v_cache + (unsigned long long)block_idx * block_stride_bytes;
    unsigned long long data_off = (unsigned long long)block_offset * (n_elems * 3 / 8);
    unsigned long long scale_off = data_section_bytes + (unsigned long long)block_offset * num_groups;






    for (unsigned int g = threadIdx.x; g < num_groups; g += blockDim.x) {
        unsigned int elem_offset = g * GROUP_SIZE;

        float kf[16], vf[16];
        float k_norm_sq = 0.0f, v_norm_sq = 0.0f;
        for (int i = 0; i < 16; i++) {
            kf[i] = __bfloat162float(key_src[elem_offset + i]);
            vf[i] = __bfloat162float(val_src[elem_offset + i]);
            k_norm_sq += kf[i] * kf[i];
            v_norm_sq += vf[i] * vf[i];
        }

        float k_max = 0.0f, v_max = 0.0f;
        for (int i = 0; i < 16; i++) {
            k_max = fmaxf(k_max, fabsf(kf[i]));
            v_max = fmaxf(v_max, fabsf(vf[i]));
        }

        float k_inv = (k_max > 1e-12f) ? (TURBO3_MAX / k_max) : 1.0f;
        float v_inv = (v_max > 1e-12f) ? (TURBO3_MAX / v_max) : 1.0f;

        unsigned char ki[16], vi[16];
        float k_recon_sq = 0.0f, v_recon_sq = 0.0f;
        for (int i = 0; i < 16; i++) {
            ki[i] = turbo3_quantize(kf[i] * k_inv);
            vi[i] = turbo3_quantize(vf[i] * v_inv);
            float kc = TURBO3_CODEBOOK[ki[i]];
            float vc = TURBO3_CODEBOOK[vi[i]];
            k_recon_sq += kc * kc;
            v_recon_sq += vc * vc;
        }
        float k_recon_norm = sqrtf(k_recon_sq);
        float v_recon_norm = sqrtf(v_recon_sq);

        float ks = (k_recon_norm > 1e-10f) ? (sqrtf(k_norm_sq) / k_recon_norm) : (k_max / TURBO3_MAX);
        float vs = (v_recon_norm > 1e-10f) ? (sqrtf(v_norm_sq) / v_recon_norm) : (v_max / TURBO3_MAX);
        if (ks > FP8_E4M3_MAX) ks = FP8_E4M3_MAX;
        if (vs > FP8_E4M3_MAX) vs = FP8_E4M3_MAX;

        ((__nv_fp8_storage_t*)(block_k + scale_off))[g] = float_to_fp8(ks);
        ((__nv_fp8_storage_t*)(block_v + scale_off))[g] = float_to_fp8(vs);


        unsigned int byte_base = elem_offset * 3 / 8;
        unsigned char* kd = block_k + data_off + byte_base;
        unsigned char* vd = block_v + data_off + byte_base;


        kd[0] = (ki[0]) | (ki[1] << 3) | (ki[2] << 6);
        kd[1] = (ki[2] >> 2) | (ki[3] << 1) | (ki[4] << 4) | (ki[5] << 7);
        kd[2] = (ki[5] >> 1) | (ki[6] << 2) | (ki[7] << 5);
        vd[0] = (vi[0]) | (vi[1] << 3) | (vi[2] << 6);
        vd[1] = (vi[2] >> 2) | (vi[3] << 1) | (vi[4] << 4) | (vi[5] << 7);
        vd[2] = (vi[5] >> 1) | (vi[6] << 2) | (vi[7] << 5);


        kd[3] = (ki[8]) | (ki[9] << 3) | (ki[10] << 6);
        kd[4] = (ki[10] >> 2) | (ki[11] << 1) | (ki[12] << 4) | (ki[13] << 7);
        kd[5] = (ki[13] >> 1) | (ki[14] << 2) | (ki[15] << 5);
        vd[3] = (vi[8]) | (vi[9] << 3) | (vi[10] << 6);
        vd[4] = (vi[10] >> 2) | (vi[11] << 1) | (vi[12] << 4) | (vi[13] << 7);
        vd[5] = (vi[13] >> 1) | (vi[14] << 2) | (vi[15] << 5);
    }
}








// 2026-09-25: turbo2 K and V: 16 codes per group in 4 bytes, four per byte with the first code in
// the low bits. Scales as in turbo4.
extern "C" __global__ void reshape_and_cache_flash_turbo2(
    const __nv_bfloat16* __restrict__ key,
    const __nv_bfloat16* __restrict__ value,
    unsigned char* __restrict__ k_cache,
    unsigned char* __restrict__ v_cache,
    const long long* __restrict__ slot_mapping,
    const unsigned int num_kv_heads,
    const unsigned int head_dim,
    const unsigned int block_size,
    const unsigned int key_stride,
    const unsigned int value_stride,
    const unsigned long long block_stride_bytes,
    const unsigned long long data_section_bytes
) {
    const unsigned int token_idx = blockIdx.x;
    const long long slot = slot_mapping[token_idx];
    if (slot < 0) return;

    const unsigned int block_idx = (unsigned int)(slot / block_size);
    const unsigned int block_offset = (unsigned int)(slot % block_size);
    const unsigned int n_elems = num_kv_heads * head_dim;
    const unsigned int num_groups = n_elems / GROUP_SIZE;

    const __nv_bfloat16* key_src = key + (unsigned long long)token_idx * key_stride;
    const __nv_bfloat16* val_src = value + (unsigned long long)token_idx * value_stride;



    unsigned char* block_k = k_cache + (unsigned long long)block_idx * block_stride_bytes;
    unsigned char* block_v = v_cache + (unsigned long long)block_idx * block_stride_bytes;
    unsigned long long data_off = (unsigned long long)block_offset * (n_elems / 4);
    unsigned long long scale_off = data_section_bytes + (unsigned long long)block_offset * num_groups;


    for (unsigned int g = threadIdx.x; g < num_groups; g += blockDim.x) {
        unsigned int elem_offset = g * GROUP_SIZE;

        float kf[16], vf[16];
        float k_norm_sq = 0.0f, v_norm_sq = 0.0f;
        for (int i = 0; i < 16; i++) {
            kf[i] = __bfloat162float(key_src[elem_offset + i]);
            vf[i] = __bfloat162float(val_src[elem_offset + i]);
            k_norm_sq += kf[i] * kf[i];
            v_norm_sq += vf[i] * vf[i];
        }

        float k_max = 0.0f, v_max = 0.0f;
        for (int i = 0; i < 16; i++) {
            k_max = fmaxf(k_max, fabsf(kf[i]));
            v_max = fmaxf(v_max, fabsf(vf[i]));
        }

        float k_inv = (k_max > 1e-12f) ? (TURBO2_MAX / k_max) : 1.0f;
        float v_inv = (v_max > 1e-12f) ? (TURBO2_MAX / v_max) : 1.0f;

        unsigned char ki[16], vi[16];
        float k_recon_sq = 0.0f, v_recon_sq = 0.0f;
        for (int i = 0; i < 16; i++) {
            ki[i] = turbo2_quantize(kf[i] * k_inv);
            vi[i] = turbo2_quantize(vf[i] * v_inv);
            float kc = TURBO2_CODEBOOK[ki[i]];
            float vc = TURBO2_CODEBOOK[vi[i]];
            k_recon_sq += kc * kc;
            v_recon_sq += vc * vc;
        }
        float k_recon_norm = sqrtf(k_recon_sq);
        float v_recon_norm = sqrtf(v_recon_sq);

        float ks = (k_recon_norm > 1e-10f) ? (sqrtf(k_norm_sq) / k_recon_norm) : (k_max / TURBO2_MAX);
        float vs = (v_recon_norm > 1e-10f) ? (sqrtf(v_norm_sq) / v_recon_norm) : (v_max / TURBO2_MAX);
        if (ks > FP8_E4M3_MAX) ks = FP8_E4M3_MAX;
        if (vs > FP8_E4M3_MAX) vs = FP8_E4M3_MAX;

        ((__nv_fp8_storage_t*)(block_k + scale_off))[g] = float_to_fp8(ks);
        ((__nv_fp8_storage_t*)(block_v + scale_off))[g] = float_to_fp8(vs);


        unsigned int byte_base = elem_offset / 4;
        unsigned char* kd = block_k + data_off + byte_base;
        unsigned char* vd = block_v + data_off + byte_base;
        kd[0] = ki[0]  | (ki[1]  << 2) | (ki[2]  << 4) | (ki[3]  << 6);
        kd[1] = ki[4]  | (ki[5]  << 2) | (ki[6]  << 4) | (ki[7]  << 6);
        kd[2] = ki[8]  | (ki[9]  << 2) | (ki[10] << 4) | (ki[11] << 6);
        kd[3] = ki[12] | (ki[13] << 2) | (ki[14] << 4) | (ki[15] << 6);
        vd[0] = vi[0]  | (vi[1]  << 2) | (vi[2]  << 4) | (vi[3]  << 6);
        vd[1] = vi[4]  | (vi[5]  << 2) | (vi[6]  << 4) | (vi[7]  << 6);
        vd[2] = vi[8]  | (vi[9]  << 2) | (vi[10] << 4) | (vi[11] << 6);
        vd[3] = vi[12] | (vi[13] << 2) | (vi[14] << 4) | (vi[15] << 6);
    }
}












// 2026-09-25: Asymmetric BF16 K and turbo3 V. K is copied as BF16, 4 elements per uint2, into a
// [block_size, n_elems] block whose stride is computed from block_size * n_elems;
// k_block_stride_bytes is not read. V is written as in reshape_and_cache_flash_turbo3, at
// v_block_stride_bytes and v_data_section_bytes.
extern "C" __global__ void reshape_and_cache_flash_bf16k_turbo3v(
    const __nv_bfloat16* __restrict__ key,
    const __nv_bfloat16* __restrict__ value,
    __nv_bfloat16* __restrict__ k_cache,
    unsigned char* __restrict__ v_cache,
    const long long* __restrict__ slot_mapping,
    const unsigned int num_kv_heads,
    const unsigned int head_dim,
    const unsigned int block_size,
    const unsigned int key_stride,
    const unsigned int value_stride,
    const unsigned long long k_block_stride_bytes,
    const unsigned long long v_block_stride_bytes,
    const unsigned long long v_data_section_bytes
) {
    const unsigned int token_idx = blockIdx.x;
    const long long slot = slot_mapping[token_idx];
    if (slot < 0) return;

    const unsigned int block_idx = (unsigned int)(slot / block_size);
    const unsigned int block_offset = (unsigned int)(slot % block_size);
    const unsigned int n_elems = num_kv_heads * head_dim;
    const unsigned int num_groups = n_elems / GROUP_SIZE;

    const __nv_bfloat16* key_src = key + (unsigned long long)token_idx * key_stride;
    const __nv_bfloat16* val_src = value + (unsigned long long)token_idx * value_stride;






    {
        const unsigned long long k_block_stride_elems = (unsigned long long)block_size * n_elems;
        __nv_bfloat16* key_dst = k_cache
            + (unsigned long long)block_idx * k_block_stride_elems
            + (unsigned long long)block_offset * n_elems;

        const unsigned int n_vec = n_elems / 4;
        const unsigned int n_rem = n_elems % 4;
        const uint2* key_src_vec = (const uint2*)key_src;
        uint2* key_dst_vec = (uint2*)key_dst;
        for (unsigned int i = threadIdx.x; i < n_vec; i += blockDim.x) {
            key_dst_vec[i] = key_src_vec[i];
        }
        if (n_rem > 0) {
            unsigned int base = n_vec * 4;
            for (unsigned int i = threadIdx.x; i < n_rem; i += blockDim.x) {
                key_dst[base + i] = key_src[base + i];
            }
        }
    }





    unsigned char* block_v = v_cache + (unsigned long long)block_idx * v_block_stride_bytes;
    unsigned long long v_data_off = (unsigned long long)block_offset * (n_elems * 3 / 8);
    unsigned long long v_scale_off = v_data_section_bytes
        + (unsigned long long)block_offset * num_groups;

    for (unsigned int g = threadIdx.x; g < num_groups; g += blockDim.x) {
        unsigned int elem_offset = g * GROUP_SIZE;

        float vf[16];
        float v_norm_sq = 0.0f;
        for (int i = 0; i < 16; i++) {
            vf[i] = __bfloat162float(val_src[elem_offset + i]);
            v_norm_sq += vf[i] * vf[i];
        }
        float v_max = 0.0f;
        for (int i = 0; i < 16; i++) v_max = fmaxf(v_max, fabsf(vf[i]));

        float v_inv = (v_max > 1e-12f) ? (TURBO3_MAX / v_max) : 1.0f;

        unsigned char vi[16];
        float v_recon_sq = 0.0f;
        for (int i = 0; i < 16; i++) {
            vi[i] = turbo3_quantize(vf[i] * v_inv);
            float vc = TURBO3_CODEBOOK[vi[i]];
            v_recon_sq += vc * vc;
        }
        float v_recon_norm = sqrtf(v_recon_sq);
        float vs = (v_recon_norm > 1e-10f)
            ? (sqrtf(v_norm_sq) / v_recon_norm)
            : (v_max / TURBO3_MAX);
        if (vs > FP8_E4M3_MAX) vs = FP8_E4M3_MAX;

        ((__nv_fp8_storage_t*)(block_v + v_scale_off))[g] = float_to_fp8(vs);


        unsigned int byte_base = elem_offset * 3 / 8;
        unsigned char* vd = block_v + v_data_off + byte_base;

        vd[0] = (vi[0]) | (vi[1] << 3) | (vi[2] << 6);
        vd[1] = (vi[2] >> 2) | (vi[3] << 1) | (vi[4] << 4) | (vi[5] << 7);
        vd[2] = (vi[5] >> 1) | (vi[6] << 2) | (vi[7] << 5);

        vd[3] = (vi[8]) | (vi[9] << 3) | (vi[10] << 6);
        vd[4] = (vi[10] >> 2) | (vi[11] << 1) | (vi[12] << 4) | (vi[13] << 7);
        vd[5] = (vi[13] >> 1) | (vi[14] << 2) | (vi[15] << 5);
    }
}








// 2026-09-25: reshape_and_cache_flash_bf16k_turbo3v with a turbo4 V side.
extern "C" __global__ void reshape_and_cache_flash_bf16k_turbo4v(
    const __nv_bfloat16* __restrict__ key,
    const __nv_bfloat16* __restrict__ value,
    __nv_bfloat16* __restrict__ k_cache,
    unsigned char* __restrict__ v_cache,
    const long long* __restrict__ slot_mapping,
    const unsigned int num_kv_heads,
    const unsigned int head_dim,
    const unsigned int block_size,
    const unsigned int key_stride,
    const unsigned int value_stride,
    const unsigned long long k_block_stride_bytes,
    const unsigned long long v_block_stride_bytes,
    const unsigned long long v_data_section_bytes
) {
    const unsigned int token_idx = blockIdx.x;
    const long long slot = slot_mapping[token_idx];
    if (slot < 0) return;

    const unsigned int block_idx = (unsigned int)(slot / block_size);
    const unsigned int block_offset = (unsigned int)(slot % block_size);
    const unsigned int n_elems = num_kv_heads * head_dim;
    const unsigned int num_groups = n_elems / GROUP_SIZE;

    const __nv_bfloat16* key_src = key + (unsigned long long)token_idx * key_stride;
    const __nv_bfloat16* val_src = value + (unsigned long long)token_idx * value_stride;


    {
        const unsigned long long k_block_stride_elems = (unsigned long long)block_size * n_elems;
        __nv_bfloat16* key_dst = k_cache
            + (unsigned long long)block_idx * k_block_stride_elems
            + (unsigned long long)block_offset * n_elems;
        const unsigned int n_vec = n_elems / 4;
        const unsigned int n_rem = n_elems % 4;
        const uint2* key_src_vec = (const uint2*)key_src;
        uint2* key_dst_vec = (uint2*)key_dst;
        for (unsigned int i = threadIdx.x; i < n_vec; i += blockDim.x) {
            key_dst_vec[i] = key_src_vec[i];
        }
        if (n_rem > 0) {
            unsigned int base = n_vec * 4;
            for (unsigned int i = threadIdx.x; i < n_rem; i += blockDim.x) {
                key_dst[base + i] = key_src[base + i];
            }
        }
    }


    unsigned char* block_v = v_cache + (unsigned long long)block_idx * v_block_stride_bytes;
    unsigned long long v_data_off = (unsigned long long)block_offset * (n_elems / 2);
    unsigned long long v_scale_off = v_data_section_bytes
        + (unsigned long long)block_offset * num_groups;

    for (unsigned int g = threadIdx.x; g < num_groups; g += blockDim.x) {
        unsigned int elem_offset = g * GROUP_SIZE;

        float vf[16];
        float v_norm_sq = 0.0f;
        for (int i = 0; i < 16; i++) {
            vf[i] = __bfloat162float(val_src[elem_offset + i]);
            v_norm_sq += vf[i] * vf[i];
        }
        float v_max = 0.0f;
        for (int i = 0; i < 16; i++) v_max = fmaxf(v_max, fabsf(vf[i]));

        float v_inv = (v_max > 1e-12f) ? (TURBO4_MAX / v_max) : 1.0f;

        unsigned char vi[16];
        float v_recon_sq = 0.0f;
        for (int i = 0; i < 16; i++) {
            vi[i] = turbo4_quantize(vf[i] * v_inv);
            float vc = TURBO4_CODEBOOK[vi[i]];
            v_recon_sq += vc * vc;
        }
        float v_recon_norm = sqrtf(v_recon_sq);
        float vs = (v_recon_norm > 1e-10f)
            ? (sqrtf(v_norm_sq) / v_recon_norm)
            : (v_max / TURBO4_MAX);
        if (vs > FP8_E4M3_MAX) vs = FP8_E4M3_MAX;

        ((__nv_fp8_storage_t*)(block_v + v_scale_off))[g] = float_to_fp8(vs);


        unsigned char* vd = block_v + v_data_off + elem_offset / 2;
        for (int i = 0; i < 16; i += 2) {
            vd[i/2] = vi[i] | (vi[i+1] << 4);
        }
    }
}







// 2026-09-25: reshape_and_cache_flash_bf16k_turbo3v with a turbo2 V side.
extern "C" __global__ void reshape_and_cache_flash_bf16k_turbo2v(
    const __nv_bfloat16* __restrict__ key,
    const __nv_bfloat16* __restrict__ value,
    __nv_bfloat16* __restrict__ k_cache,
    unsigned char* __restrict__ v_cache,
    const long long* __restrict__ slot_mapping,
    const unsigned int num_kv_heads,
    const unsigned int head_dim,
    const unsigned int block_size,
    const unsigned int key_stride,
    const unsigned int value_stride,
    const unsigned long long k_block_stride_bytes,
    const unsigned long long v_block_stride_bytes,
    const unsigned long long v_data_section_bytes
) {
    const unsigned int token_idx = blockIdx.x;
    const long long slot = slot_mapping[token_idx];
    if (slot < 0) return;

    const unsigned int block_idx = (unsigned int)(slot / block_size);
    const unsigned int block_offset = (unsigned int)(slot % block_size);
    const unsigned int n_elems = num_kv_heads * head_dim;
    const unsigned int num_groups = n_elems / GROUP_SIZE;

    const __nv_bfloat16* key_src = key + (unsigned long long)token_idx * key_stride;
    const __nv_bfloat16* val_src = value + (unsigned long long)token_idx * value_stride;


    {
        const unsigned long long k_block_stride_elems = (unsigned long long)block_size * n_elems;
        __nv_bfloat16* key_dst = k_cache
            + (unsigned long long)block_idx * k_block_stride_elems
            + (unsigned long long)block_offset * n_elems;
        const unsigned int n_vec = n_elems / 4;
        const unsigned int n_rem = n_elems % 4;
        const uint2* key_src_vec = (const uint2*)key_src;
        uint2* key_dst_vec = (uint2*)key_dst;
        for (unsigned int i = threadIdx.x; i < n_vec; i += blockDim.x) {
            key_dst_vec[i] = key_src_vec[i];
        }
        if (n_rem > 0) {
            unsigned int base = n_vec * 4;
            for (unsigned int i = threadIdx.x; i < n_rem; i += blockDim.x) {
                key_dst[base + i] = key_src[base + i];
            }
        }
    }


    unsigned char* block_v = v_cache + (unsigned long long)block_idx * v_block_stride_bytes;
    unsigned long long v_data_off = (unsigned long long)block_offset * (n_elems / 4);
    unsigned long long v_scale_off = v_data_section_bytes
        + (unsigned long long)block_offset * num_groups;

    for (unsigned int g = threadIdx.x; g < num_groups; g += blockDim.x) {
        unsigned int elem_offset = g * GROUP_SIZE;

        float vf[16];
        float v_norm_sq = 0.0f;
        for (int i = 0; i < 16; i++) {
            vf[i] = __bfloat162float(val_src[elem_offset + i]);
            v_norm_sq += vf[i] * vf[i];
        }
        float v_max = 0.0f;
        for (int i = 0; i < 16; i++) v_max = fmaxf(v_max, fabsf(vf[i]));

        float v_inv = (v_max > 1e-12f) ? (TURBO2_MAX / v_max) : 1.0f;

        unsigned char vi[16];
        float v_recon_sq = 0.0f;
        for (int i = 0; i < 16; i++) {
            vi[i] = turbo2_quantize(vf[i] * v_inv);
            float vc = TURBO2_CODEBOOK[vi[i]];
            v_recon_sq += vc * vc;
        }
        float v_recon_norm = sqrtf(v_recon_sq);
        float vs = (v_recon_norm > 1e-10f)
            ? (sqrtf(v_norm_sq) / v_recon_norm)
            : (v_max / TURBO2_MAX);
        if (vs > FP8_E4M3_MAX) vs = FP8_E4M3_MAX;

        ((__nv_fp8_storage_t*)(block_v + v_scale_off))[g] = float_to_fp8(vs);


        unsigned char* vd = block_v + v_data_off + elem_offset / 4;
        vd[0] = vi[0]  | (vi[1]  << 2) | (vi[2]  << 4) | (vi[3]  << 6);
        vd[1] = vi[4]  | (vi[5]  << 2) | (vi[6]  << 4) | (vi[7]  << 6);
        vd[2] = vi[8]  | (vi[9]  << 2) | (vi[10] << 4) | (vi[11] << 6);
        vd[3] = vi[12] | (vi[13] << 2) | (vi[14] << 4) | (vi[15] << 6);
    }
}







// 2026-09-25: Asymmetric FP8 K and turbo3 V. K is E4M3(x / k_scale), k_scale being the
// per-tensor dequant scale, with the same paired saturating cast as reshape_and_cache_flash_fp8,
// into a [block_size, n_elems] byte block whose stride is computed from block_size * n_elems;
// k_block_stride_bytes is not read. V is written as in reshape_and_cache_flash_turbo3.
extern "C" __global__ void reshape_and_cache_flash_fp8k_turbo3v(
    const __nv_bfloat16* __restrict__ key,
    const __nv_bfloat16* __restrict__ value,
    unsigned char* __restrict__ k_cache,
    unsigned char* __restrict__ v_cache,
    const long long* __restrict__ slot_mapping,
    const unsigned int num_kv_heads,
    const unsigned int head_dim,
    const unsigned int block_size,
    const unsigned int key_stride,
    const unsigned int value_stride,
    const float k_scale,
    const unsigned long long k_block_stride_bytes,
    const unsigned long long v_block_stride_bytes,
    const unsigned long long v_data_section_bytes
) {
    const unsigned int token_idx = blockIdx.x;
    const long long slot = slot_mapping[token_idx];
    if (slot < 0) return;

    const unsigned int block_idx = (unsigned int)(slot / block_size);
    const unsigned int block_offset = (unsigned int)(slot % block_size);
    const unsigned int n_elems = num_kv_heads * head_dim;
    const unsigned int num_groups = n_elems / GROUP_SIZE;

    const __nv_bfloat16* key_src = key + (unsigned long long)token_idx * key_stride;
    const __nv_bfloat16* val_src = value + (unsigned long long)token_idx * value_stride;



    {
        const unsigned long long k_block_stride_elems = (unsigned long long)block_size * n_elems;
        unsigned char* key_dst = k_cache
            + (unsigned long long)block_idx * k_block_stride_elems
            + (unsigned long long)block_offset * n_elems;
        const float inv_k_scale = 1.0f / k_scale;


        const unsigned int n_pairs = n_elems / 2;
        const unsigned int n_rem = n_elems % 2;
        const unsigned int* key_src32 = (const unsigned int*)key_src;
        __nv_fp8x2_storage_t* key_dst16 = (__nv_fp8x2_storage_t*)key_dst;
        for (unsigned int i = threadIdx.x; i < n_pairs; i += blockDim.x) {
            unsigned int pk = key_src32[i];
            float v0 = __bfloat162float(__ushort_as_bfloat16((unsigned short)(pk & 0xFFFF)));
            float v1 = __bfloat162float(__ushort_as_bfloat16((unsigned short)(pk >> 16)));
            float2 scaled = make_float2(v0 * inv_k_scale, v1 * inv_k_scale);
            key_dst16[i] = __nv_cvt_float2_to_fp8x2(scaled, __NV_SATFINITE, __NV_E4M3);
        }
        if (n_rem > 0 && threadIdx.x == 0) {
            unsigned int base = n_pairs * 2;
            float kf = __bfloat162float(key_src[base]) * inv_k_scale;
            ((__nv_fp8_storage_t*)key_dst)[base] = __nv_cvt_float_to_fp8(kf, __NV_SATFINITE, __NV_E4M3);
        }
    }



    unsigned char* block_v = v_cache + (unsigned long long)block_idx * v_block_stride_bytes;
    unsigned long long v_data_off = (unsigned long long)block_offset * (n_elems * 3 / 8);
    unsigned long long v_scale_off = v_data_section_bytes
        + (unsigned long long)block_offset * num_groups;

    for (unsigned int g = threadIdx.x; g < num_groups; g += blockDim.x) {
        unsigned int elem_offset = g * GROUP_SIZE;

        float vf[16];
        float v_norm_sq = 0.0f;
        for (int i = 0; i < 16; i++) {
            vf[i] = __bfloat162float(val_src[elem_offset + i]);
            v_norm_sq += vf[i] * vf[i];
        }
        float v_max = 0.0f;
        for (int i = 0; i < 16; i++) v_max = fmaxf(v_max, fabsf(vf[i]));

        float v_inv = (v_max > 1e-12f) ? (TURBO3_MAX / v_max) : 1.0f;

        unsigned char vi[16];
        float v_recon_sq = 0.0f;
        for (int i = 0; i < 16; i++) {
            vi[i] = turbo3_quantize(vf[i] * v_inv);
            float vc = TURBO3_CODEBOOK[vi[i]];
            v_recon_sq += vc * vc;
        }
        float v_recon_norm = sqrtf(v_recon_sq);
        float vs = (v_recon_norm > 1e-10f)
            ? (sqrtf(v_norm_sq) / v_recon_norm)
            : (v_max / TURBO3_MAX);
        if (vs > FP8_E4M3_MAX) vs = FP8_E4M3_MAX;

        ((__nv_fp8_storage_t*)(block_v + v_scale_off))[g] = float_to_fp8(vs);

        unsigned int byte_base = elem_offset * 3 / 8;
        unsigned char* vd = block_v + v_data_off + byte_base;
        vd[0] = (vi[0]) | (vi[1] << 3) | (vi[2] << 6);
        vd[1] = (vi[2] >> 2) | (vi[3] << 1) | (vi[4] << 4) | (vi[5] << 7);
        vd[2] = (vi[5] >> 1) | (vi[6] << 2) | (vi[7] << 5);
        vd[3] = (vi[8]) | (vi[9] << 3) | (vi[10] << 6);
        vd[4] = (vi[10] >> 2) | (vi[11] << 1) | (vi[12] << 4) | (vi[13] << 7);
        vd[5] = (vi[13] >> 1) | (vi[14] << 2) | (vi[15] << 5);
    }
}







// 2026-09-25: reshape_and_cache_flash_fp8k_turbo3v with a turbo4 V side.
extern "C" __global__ void reshape_and_cache_flash_fp8k_turbo4v(
    const __nv_bfloat16* __restrict__ key,
    const __nv_bfloat16* __restrict__ value,
    unsigned char* __restrict__ k_cache,
    unsigned char* __restrict__ v_cache,
    const long long* __restrict__ slot_mapping,
    const unsigned int num_kv_heads,
    const unsigned int head_dim,
    const unsigned int block_size,
    const unsigned int key_stride,
    const unsigned int value_stride,
    const float k_scale,
    const unsigned long long k_block_stride_bytes,
    const unsigned long long v_block_stride_bytes,
    const unsigned long long v_data_section_bytes
) {
    const unsigned int token_idx = blockIdx.x;
    const long long slot = slot_mapping[token_idx];
    if (slot < 0) return;

    const unsigned int block_idx = (unsigned int)(slot / block_size);
    const unsigned int block_offset = (unsigned int)(slot % block_size);
    const unsigned int n_elems = num_kv_heads * head_dim;
    const unsigned int num_groups = n_elems / GROUP_SIZE;

    const __nv_bfloat16* key_src = key + (unsigned long long)token_idx * key_stride;
    const __nv_bfloat16* val_src = value + (unsigned long long)token_idx * value_stride;


    {
        const unsigned long long k_block_stride_elems = (unsigned long long)block_size * n_elems;
        unsigned char* key_dst = k_cache
            + (unsigned long long)block_idx * k_block_stride_elems
            + (unsigned long long)block_offset * n_elems;
        const float inv_k_scale = 1.0f / k_scale;
        const unsigned int n_pairs = n_elems / 2;
        const unsigned int n_rem = n_elems % 2;
        const unsigned int* key_src32 = (const unsigned int*)key_src;
        __nv_fp8x2_storage_t* key_dst16 = (__nv_fp8x2_storage_t*)key_dst;
        for (unsigned int i = threadIdx.x; i < n_pairs; i += blockDim.x) {
            unsigned int pk = key_src32[i];
            float v0 = __bfloat162float(__ushort_as_bfloat16((unsigned short)(pk & 0xFFFF)));
            float v1 = __bfloat162float(__ushort_as_bfloat16((unsigned short)(pk >> 16)));
            float2 scaled = make_float2(v0 * inv_k_scale, v1 * inv_k_scale);
            key_dst16[i] = __nv_cvt_float2_to_fp8x2(scaled, __NV_SATFINITE, __NV_E4M3);
        }
        if (n_rem > 0 && threadIdx.x == 0) {
            unsigned int base = n_pairs * 2;
            float kf = __bfloat162float(key_src[base]) * inv_k_scale;
            ((__nv_fp8_storage_t*)key_dst)[base] = __nv_cvt_float_to_fp8(kf, __NV_SATFINITE, __NV_E4M3);
        }
    }


    unsigned char* block_v = v_cache + (unsigned long long)block_idx * v_block_stride_bytes;
    unsigned long long v_data_off = (unsigned long long)block_offset * (n_elems / 2);
    unsigned long long v_scale_off = v_data_section_bytes
        + (unsigned long long)block_offset * num_groups;

    for (unsigned int g = threadIdx.x; g < num_groups; g += blockDim.x) {
        unsigned int elem_offset = g * GROUP_SIZE;

        float vf[16];
        float v_norm_sq = 0.0f;
        for (int i = 0; i < 16; i++) {
            vf[i] = __bfloat162float(val_src[elem_offset + i]);
            v_norm_sq += vf[i] * vf[i];
        }
        float v_max = 0.0f;
        for (int i = 0; i < 16; i++) v_max = fmaxf(v_max, fabsf(vf[i]));

        float v_inv = (v_max > 1e-12f) ? (TURBO4_MAX / v_max) : 1.0f;

        unsigned char vi[16];
        float v_recon_sq = 0.0f;
        for (int i = 0; i < 16; i++) {
            vi[i] = turbo4_quantize(vf[i] * v_inv);
            float vc = TURBO4_CODEBOOK[vi[i]];
            v_recon_sq += vc * vc;
        }
        float v_recon_norm = sqrtf(v_recon_sq);
        float vs = (v_recon_norm > 1e-10f)
            ? (sqrtf(v_norm_sq) / v_recon_norm)
            : (v_max / TURBO4_MAX);
        if (vs > FP8_E4M3_MAX) vs = FP8_E4M3_MAX;

        ((__nv_fp8_storage_t*)(block_v + v_scale_off))[g] = float_to_fp8(vs);

        unsigned char* vd = block_v + v_data_off + elem_offset / 2;
        for (int i = 0; i < 16; i += 2) {
            vd[i/2] = vi[i] | (vi[i+1] << 4);
        }
    }
}






// 2026-09-25: reshape_and_cache_flash_fp8k_turbo3v with a turbo2 V side.
extern "C" __global__ void reshape_and_cache_flash_fp8k_turbo2v(
    const __nv_bfloat16* __restrict__ key,
    const __nv_bfloat16* __restrict__ value,
    unsigned char* __restrict__ k_cache,
    unsigned char* __restrict__ v_cache,
    const long long* __restrict__ slot_mapping,
    const unsigned int num_kv_heads,
    const unsigned int head_dim,
    const unsigned int block_size,
    const unsigned int key_stride,
    const unsigned int value_stride,
    const float k_scale,
    const unsigned long long k_block_stride_bytes,
    const unsigned long long v_block_stride_bytes,
    const unsigned long long v_data_section_bytes
) {
    const unsigned int token_idx = blockIdx.x;
    const long long slot = slot_mapping[token_idx];
    if (slot < 0) return;

    const unsigned int block_idx = (unsigned int)(slot / block_size);
    const unsigned int block_offset = (unsigned int)(slot % block_size);
    const unsigned int n_elems = num_kv_heads * head_dim;
    const unsigned int num_groups = n_elems / GROUP_SIZE;

    const __nv_bfloat16* key_src = key + (unsigned long long)token_idx * key_stride;
    const __nv_bfloat16* val_src = value + (unsigned long long)token_idx * value_stride;


    {
        const unsigned long long k_block_stride_elems = (unsigned long long)block_size * n_elems;
        unsigned char* key_dst = k_cache
            + (unsigned long long)block_idx * k_block_stride_elems
            + (unsigned long long)block_offset * n_elems;
        const float inv_k_scale = 1.0f / k_scale;
        const unsigned int n_pairs = n_elems / 2;
        const unsigned int n_rem = n_elems % 2;
        const unsigned int* key_src32 = (const unsigned int*)key_src;
        __nv_fp8x2_storage_t* key_dst16 = (__nv_fp8x2_storage_t*)key_dst;
        for (unsigned int i = threadIdx.x; i < n_pairs; i += blockDim.x) {
            unsigned int pk = key_src32[i];
            float v0 = __bfloat162float(__ushort_as_bfloat16((unsigned short)(pk & 0xFFFF)));
            float v1 = __bfloat162float(__ushort_as_bfloat16((unsigned short)(pk >> 16)));
            float2 scaled = make_float2(v0 * inv_k_scale, v1 * inv_k_scale);
            key_dst16[i] = __nv_cvt_float2_to_fp8x2(scaled, __NV_SATFINITE, __NV_E4M3);
        }
        if (n_rem > 0 && threadIdx.x == 0) {
            unsigned int base = n_pairs * 2;
            float kf = __bfloat162float(key_src[base]) * inv_k_scale;
            ((__nv_fp8_storage_t*)key_dst)[base] = __nv_cvt_float_to_fp8(kf, __NV_SATFINITE, __NV_E4M3);
        }
    }


    unsigned char* block_v = v_cache + (unsigned long long)block_idx * v_block_stride_bytes;
    unsigned long long v_data_off = (unsigned long long)block_offset * (n_elems / 4);
    unsigned long long v_scale_off = v_data_section_bytes
        + (unsigned long long)block_offset * num_groups;

    for (unsigned int g = threadIdx.x; g < num_groups; g += blockDim.x) {
        unsigned int elem_offset = g * GROUP_SIZE;

        float vf[16];
        float v_norm_sq = 0.0f;
        for (int i = 0; i < 16; i++) {
            vf[i] = __bfloat162float(val_src[elem_offset + i]);
            v_norm_sq += vf[i] * vf[i];
        }
        float v_max = 0.0f;
        for (int i = 0; i < 16; i++) v_max = fmaxf(v_max, fabsf(vf[i]));

        float v_inv = (v_max > 1e-12f) ? (TURBO2_MAX / v_max) : 1.0f;

        unsigned char vi[16];
        float v_recon_sq = 0.0f;
        for (int i = 0; i < 16; i++) {
            vi[i] = turbo2_quantize(vf[i] * v_inv);
            float vc = TURBO2_CODEBOOK[vi[i]];
            v_recon_sq += vc * vc;
        }
        float v_recon_norm = sqrtf(v_recon_sq);
        float vs = (v_recon_norm > 1e-10f)
            ? (sqrtf(v_norm_sq) / v_recon_norm)
            : (v_max / TURBO2_MAX);
        if (vs > FP8_E4M3_MAX) vs = FP8_E4M3_MAX;

        ((__nv_fp8_storage_t*)(block_v + v_scale_off))[g] = float_to_fp8(vs);

        unsigned char* vd = block_v + v_data_off + elem_offset / 4;
        vd[0] = vi[0]  | (vi[1]  << 2) | (vi[2]  << 4) | (vi[3]  << 6);
        vd[1] = vi[4]  | (vi[5]  << 2) | (vi[6]  << 4) | (vi[7]  << 6);
        vd[2] = vi[8]  | (vi[9]  << 2) | (vi[10] << 4) | (vi[11] << 6);
        vd[3] = vi[12] | (vi[13] << 2) | (vi[14] << 4) | (vi[15] << 6);
    }
}













// 2026-09-25: Both sides quantized: turbo4 K and turbo3 V, each with its own block stride and
// data-section size.
extern "C" __global__ void reshape_and_cache_flash_turbo4k_turbo3v(
    const __nv_bfloat16* __restrict__ key,
    const __nv_bfloat16* __restrict__ value,
    unsigned char* __restrict__ k_cache,
    unsigned char* __restrict__ v_cache,
    const long long* __restrict__ slot_mapping,
    const unsigned int num_kv_heads,
    const unsigned int head_dim,
    const unsigned int block_size,
    const unsigned int key_stride,
    const unsigned int value_stride,
    const unsigned long long k_block_stride_bytes,
    const unsigned long long k_data_section_bytes,
    const unsigned long long v_block_stride_bytes,
    const unsigned long long v_data_section_bytes
) {
    const unsigned int token_idx = blockIdx.x;
    const long long slot = slot_mapping[token_idx];
    if (slot < 0) return;

    const unsigned int block_idx = (unsigned int)(slot / block_size);
    const unsigned int block_offset = (unsigned int)(slot % block_size);
    const unsigned int n_elems = num_kv_heads * head_dim;
    const unsigned int num_groups = n_elems / GROUP_SIZE;

    const __nv_bfloat16* key_src = key + (unsigned long long)token_idx * key_stride;
    const __nv_bfloat16* val_src = value + (unsigned long long)token_idx * value_stride;


    unsigned char* block_k = k_cache + (unsigned long long)block_idx * k_block_stride_bytes;
    unsigned long long k_data_off = (unsigned long long)block_offset * (n_elems / 2);
    unsigned long long k_scale_off = k_data_section_bytes
        + (unsigned long long)block_offset * num_groups;


    unsigned char* block_v = v_cache + (unsigned long long)block_idx * v_block_stride_bytes;
    unsigned long long v_data_off = (unsigned long long)block_offset * (n_elems * 3 / 8);
    unsigned long long v_scale_off = v_data_section_bytes
        + (unsigned long long)block_offset * num_groups;

    for (unsigned int g = threadIdx.x; g < num_groups; g += blockDim.x) {
        unsigned int elem_offset = g * GROUP_SIZE;

        float kf[16], vf[16];
        float k_norm_sq = 0.0f, v_norm_sq = 0.0f;
        for (int i = 0; i < 16; i++) {
            kf[i] = __bfloat162float(key_src[elem_offset + i]);
            vf[i] = __bfloat162float(val_src[elem_offset + i]);
            k_norm_sq += kf[i] * kf[i];
            v_norm_sq += vf[i] * vf[i];
        }
        float k_max = 0.0f, v_max = 0.0f;
        for (int i = 0; i < 16; i++) {
            k_max = fmaxf(k_max, fabsf(kf[i]));
            v_max = fmaxf(v_max, fabsf(vf[i]));
        }

        float k_inv = (k_max > 1e-12f) ? (TURBO4_MAX / k_max) : 1.0f;
        float v_inv = (v_max > 1e-12f) ? (TURBO3_MAX / v_max) : 1.0f;

        unsigned char ki[16], vi[16];
        float k_recon_sq = 0.0f, v_recon_sq = 0.0f;
        for (int i = 0; i < 16; i++) {
            ki[i] = turbo4_quantize(kf[i] * k_inv);
            vi[i] = turbo3_quantize(vf[i] * v_inv);
            float kc = TURBO4_CODEBOOK[ki[i]];
            float vc = TURBO3_CODEBOOK[vi[i]];
            k_recon_sq += kc * kc;
            v_recon_sq += vc * vc;
        }
        float k_recon_norm = sqrtf(k_recon_sq);
        float v_recon_norm = sqrtf(v_recon_sq);

        float ks = (k_recon_norm > 1e-10f) ? (sqrtf(k_norm_sq) / k_recon_norm) : (k_max / TURBO4_MAX);
        float vs = (v_recon_norm > 1e-10f) ? (sqrtf(v_norm_sq) / v_recon_norm) : (v_max / TURBO3_MAX);
        if (ks > FP8_E4M3_MAX) ks = FP8_E4M3_MAX;
        if (vs > FP8_E4M3_MAX) vs = FP8_E4M3_MAX;

        ((__nv_fp8_storage_t*)(block_k + k_scale_off))[g] = float_to_fp8(ks);
        ((__nv_fp8_storage_t*)(block_v + v_scale_off))[g] = float_to_fp8(vs);


        unsigned char* kd = block_k + k_data_off + elem_offset / 2;
        for (int i = 0; i < 16; i += 2) {
            kd[i/2] = ki[i] | (ki[i+1] << 4);
        }


        unsigned int v_byte_base = elem_offset * 3 / 8;
        unsigned char* vd = block_v + v_data_off + v_byte_base;
        vd[0] = (vi[0]) | (vi[1] << 3) | (vi[2] << 6);
        vd[1] = (vi[2] >> 2) | (vi[3] << 1) | (vi[4] << 4) | (vi[5] << 7);
        vd[2] = (vi[5] >> 1) | (vi[6] << 2) | (vi[7] << 5);
        vd[3] = (vi[8]) | (vi[9] << 3) | (vi[10] << 6);
        vd[4] = (vi[10] >> 2) | (vi[11] << 1) | (vi[12] << 4) | (vi[13] << 7);
        vd[5] = (vi[13] >> 1) | (vi[14] << 2) | (vi[15] << 5);
    }
}




// 2026-09-25: turbo4 K and turbo8 V, each with its own block stride and data-section size.
extern "C" __global__ void reshape_and_cache_flash_turbo4k_turbo8v(
    const __nv_bfloat16* __restrict__ key,
    const __nv_bfloat16* __restrict__ value,
    unsigned char* __restrict__ k_cache,
    unsigned char* __restrict__ v_cache,
    const long long* __restrict__ slot_mapping,
    const unsigned int num_kv_heads,
    const unsigned int head_dim,
    const unsigned int block_size,
    const unsigned int key_stride,
    const unsigned int value_stride,
    const unsigned long long k_block_stride_bytes,
    const unsigned long long k_data_section_bytes,
    const unsigned long long v_block_stride_bytes,
    const unsigned long long v_data_section_bytes
) {
    const unsigned int token_idx = blockIdx.x;
    const long long slot = slot_mapping[token_idx];
    if (slot < 0) return;

    const unsigned int block_idx = (unsigned int)(slot / block_size);
    const unsigned int block_offset = (unsigned int)(slot % block_size);
    const unsigned int n_elems = num_kv_heads * head_dim;
    const unsigned int num_groups = n_elems / GROUP_SIZE;

    const __nv_bfloat16* key_src = key + (unsigned long long)token_idx * key_stride;
    const __nv_bfloat16* val_src = value + (unsigned long long)token_idx * value_stride;


    unsigned char* block_k = k_cache + (unsigned long long)block_idx * k_block_stride_bytes;
    unsigned long long k_data_off = (unsigned long long)block_offset * (n_elems / 2);
    unsigned long long k_scale_off = k_data_section_bytes
        + (unsigned long long)block_offset * num_groups;


    unsigned char* block_v = v_cache + (unsigned long long)block_idx * v_block_stride_bytes;
    unsigned long long v_data_off = (unsigned long long)block_offset * n_elems;
    unsigned long long v_scale_off = v_data_section_bytes
        + (unsigned long long)block_offset * num_groups * 2;

    for (unsigned int g = threadIdx.x; g < num_groups; g += blockDim.x) {
        unsigned int elem_offset = g * GROUP_SIZE;

        float kf[16], vf[16];
        float k_norm_sq = 0.0f;
        for (int i = 0; i < 16; i++) {
            kf[i] = __bfloat162float(key_src[elem_offset + i]);
            vf[i] = __bfloat162float(val_src[elem_offset + i]);
            k_norm_sq += kf[i] * kf[i];
        }
        float k_max = 0.0f, v_max = 0.0f;
        for (int i = 0; i < 16; i++) {
            k_max = fmaxf(k_max, fabsf(kf[i]));
            v_max = fmaxf(v_max, fabsf(vf[i]));
        }


        float k_inv = (k_max > 1e-12f) ? (TURBO4_MAX / k_max) : 1.0f;
        unsigned char ki[16];
        float k_recon_sq = 0.0f;
        for (int i = 0; i < 16; i++) {
            ki[i] = turbo4_quantize(kf[i] * k_inv);
            float kc = TURBO4_CODEBOOK[ki[i]];
            k_recon_sq += kc * kc;
        }
        float k_recon_norm = sqrtf(k_recon_sq);
        float ks = (k_recon_norm > 1e-10f) ? (sqrtf(k_norm_sq) / k_recon_norm) : (k_max / TURBO4_MAX);
        if (ks > FP8_E4M3_MAX) ks = FP8_E4M3_MAX;
        ((__nv_fp8_storage_t*)(block_k + k_scale_off))[g] = float_to_fp8(ks);

        unsigned char* kd = block_k + k_data_off + elem_offset / 2;
        for (int i = 0; i < 16; i += 2) {
            kd[i/2] = ki[i] | (ki[i+1] << 4);
        }


        float v_scale = v_max / FP8_E4M3_MAX;
        if (v_scale < 1e-12f) v_scale = 1e-12f;
        ((__nv_bfloat16*)(block_v + v_scale_off))[g] = __float2bfloat16(v_scale);

        float v_inv = 1.0f / v_scale;
        unsigned char* vd = block_v + v_data_off + elem_offset;
        for (int i = 0; i < 16; i++) {
            float vs = fminf(fmaxf(vf[i] * v_inv, -FP8_E4M3_MAX), FP8_E4M3_MAX);
            vd[i] = (unsigned char)float_to_fp8(vs);
        }
    }
}




// 2026-09-25: turbo3 K and turbo8 V, each with its own block stride and data-section size.
extern "C" __global__ void reshape_and_cache_flash_turbo3k_turbo8v(
    const __nv_bfloat16* __restrict__ key,
    const __nv_bfloat16* __restrict__ value,
    unsigned char* __restrict__ k_cache,
    unsigned char* __restrict__ v_cache,
    const long long* __restrict__ slot_mapping,
    const unsigned int num_kv_heads,
    const unsigned int head_dim,
    const unsigned int block_size,
    const unsigned int key_stride,
    const unsigned int value_stride,
    const unsigned long long k_block_stride_bytes,
    const unsigned long long k_data_section_bytes,
    const unsigned long long v_block_stride_bytes,
    const unsigned long long v_data_section_bytes
) {
    const unsigned int token_idx = blockIdx.x;
    const long long slot = slot_mapping[token_idx];
    if (slot < 0) return;

    const unsigned int block_idx = (unsigned int)(slot / block_size);
    const unsigned int block_offset = (unsigned int)(slot % block_size);
    const unsigned int n_elems = num_kv_heads * head_dim;
    const unsigned int num_groups = n_elems / GROUP_SIZE;

    const __nv_bfloat16* key_src = key + (unsigned long long)token_idx * key_stride;
    const __nv_bfloat16* val_src = value + (unsigned long long)token_idx * value_stride;


    unsigned char* block_k = k_cache + (unsigned long long)block_idx * k_block_stride_bytes;
    unsigned long long k_data_off = (unsigned long long)block_offset * (n_elems * 3 / 8);
    unsigned long long k_scale_off = k_data_section_bytes
        + (unsigned long long)block_offset * num_groups;


    unsigned char* block_v = v_cache + (unsigned long long)block_idx * v_block_stride_bytes;
    unsigned long long v_data_off = (unsigned long long)block_offset * n_elems;
    unsigned long long v_scale_off = v_data_section_bytes
        + (unsigned long long)block_offset * num_groups * 2;

    for (unsigned int g = threadIdx.x; g < num_groups; g += blockDim.x) {
        unsigned int elem_offset = g * GROUP_SIZE;

        float kf[16], vf[16];
        float k_norm_sq = 0.0f;
        for (int i = 0; i < 16; i++) {
            kf[i] = __bfloat162float(key_src[elem_offset + i]);
            vf[i] = __bfloat162float(val_src[elem_offset + i]);
            k_norm_sq += kf[i] * kf[i];
        }
        float k_max = 0.0f, v_max = 0.0f;
        for (int i = 0; i < 16; i++) {
            k_max = fmaxf(k_max, fabsf(kf[i]));
            v_max = fmaxf(v_max, fabsf(vf[i]));
        }


        float k_inv = (k_max > 1e-12f) ? (TURBO3_MAX / k_max) : 1.0f;
        unsigned char ki[16];
        float k_recon_sq = 0.0f;
        for (int i = 0; i < 16; i++) {
            ki[i] = turbo3_quantize(kf[i] * k_inv);
            float kc = TURBO3_CODEBOOK[ki[i]];
            k_recon_sq += kc * kc;
        }
        float k_recon_norm = sqrtf(k_recon_sq);
        float ks = (k_recon_norm > 1e-10f) ? (sqrtf(k_norm_sq) / k_recon_norm) : (k_max / TURBO3_MAX);
        if (ks > FP8_E4M3_MAX) ks = FP8_E4M3_MAX;
        ((__nv_fp8_storage_t*)(block_k + k_scale_off))[g] = float_to_fp8(ks);


        unsigned int k_byte_base = elem_offset * 3 / 8;
        unsigned char* kd = block_k + k_data_off + k_byte_base;
        kd[0] = (ki[0]) | (ki[1] << 3) | (ki[2] << 6);
        kd[1] = (ki[2] >> 2) | (ki[3] << 1) | (ki[4] << 4) | (ki[5] << 7);
        kd[2] = (ki[5] >> 1) | (ki[6] << 2) | (ki[7] << 5);
        kd[3] = (ki[8]) | (ki[9] << 3) | (ki[10] << 6);
        kd[4] = (ki[10] >> 2) | (ki[11] << 1) | (ki[12] << 4) | (ki[13] << 7);
        kd[5] = (ki[13] >> 1) | (ki[14] << 2) | (ki[15] << 5);


        float v_scale = v_max / FP8_E4M3_MAX;
        if (v_scale < 1e-12f) v_scale = 1e-12f;
        ((__nv_bfloat16*)(block_v + v_scale_off))[g] = __float2bfloat16(v_scale);

        float v_inv = 1.0f / v_scale;
        unsigned char* vd = block_v + v_data_off + elem_offset;
        for (int i = 0; i < 16; i++) {
            float vs = fminf(fmaxf(vf[i] * v_inv, -FP8_E4M3_MAX), FP8_E4M3_MAX);
            vd[i] = (unsigned char)float_to_fp8(vs);
        }
    }
}
