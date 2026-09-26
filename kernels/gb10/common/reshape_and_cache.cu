// SPDX-License-Identifier: MIT OR Apache-2.0

// 2026-09-25: Writes of BF16 K and V rows into the paged KV cache as BF16, FP8 E4M3 or NVFP4, and
// the BF16 max-|x| reductions for FP8 KV-scale calibration.
//
// key and value are [num_tokens, num_kv_heads, head_dim] BF16 with row strides key_stride and
// value_stride, in elements. slot_mapping[t] is block * block_size + offset within the block. The
// write kernels run one block per token; their launchers in crates/model-layers/src/layers/ops
// (kv_cache.rs, prefill_attn_b.rs) use grid (num_tokens, 1, 1) and block (256, 1, 1).
//
// Owner: gb10 kernels.
// Invariants:
// - A token whose slot is negative writes nothing.
// - BF16 and FP8 slots are [num_kv_heads, head_dim] inside a block of block_size slots. An NVFP4
//   block is a data section [block_size, num_kv_heads, head_dim / 2] followed, at
//   data_section_bytes, by a scale section [block_size, num_kv_heads, head_dim / 16], the layout
//   crates/cache/src/kv_dequant.rs reads.







#include <cuda_bf16.h>


// 2026-09-25: Writes only V, for callers whose K is written by a fused_k_norm_rope_cache_write_*
// launch (reshape_and_cache_flash_v_only in crates/model-layers/src/layers/ops/kv_cache.rs).
extern "C" __global__ void reshape_and_cache_flash_v_only(
    const __nv_bfloat16* __restrict__ value,
    __nv_bfloat16* __restrict__ v_cache,
    const long long* __restrict__ slot_mapping,
    const unsigned int num_kv_heads,
    const unsigned int head_dim,
    const unsigned int block_size,
    const unsigned int value_stride
) {
    const unsigned int token_idx = blockIdx.x;
    const long long slot = slot_mapping[token_idx];
    if (slot < 0) return;

    const unsigned int block_idx = (unsigned int)(slot / block_size);
    const unsigned int block_offset = (unsigned int)(slot % block_size);
    const unsigned int n_elems = num_kv_heads * head_dim;

    const __nv_bfloat16* val_src = value + (unsigned long long)token_idx * value_stride;
    const unsigned long long cache_stride = (unsigned long long)block_size * n_elems;
    __nv_bfloat16* val_dst = v_cache + (unsigned long long)block_idx * cache_stride
                                      + (unsigned long long)block_offset * n_elems;

    const unsigned int n_vec = n_elems / 4;
    const unsigned int n_rem = n_elems % 4;
    const uint2* val_src_vec = (const uint2*)val_src;
    uint2* val_dst_vec = (uint2*)val_dst;
    for (unsigned int i = threadIdx.x; i < n_vec; i += blockDim.x) {
        val_dst_vec[i] = val_src_vec[i];
    }
    if (n_rem > 0) {
        unsigned int base = n_vec * 4;
        for (unsigned int i = threadIdx.x; i < n_rem; i += blockDim.x) {
            val_dst[base + i] = val_src[base + i];
        }
    }
}
// 2026-09-25: Copies 4 BF16 per uint2, so source rows and cache slots must be 8-byte aligned.
extern "C" __global__ void reshape_and_cache_flash(
    const __nv_bfloat16* __restrict__ key,
    const __nv_bfloat16* __restrict__ value,
    __nv_bfloat16* __restrict__ k_cache,
    __nv_bfloat16* __restrict__ v_cache,
    const long long* __restrict__ slot_mapping,
    const unsigned int num_kv_heads,
    const unsigned int head_dim,
    const unsigned int block_size,
    const unsigned int key_stride,
    const unsigned int value_stride
) {
    const unsigned int token_idx = blockIdx.x;
    const long long slot = slot_mapping[token_idx];


    if (slot < 0) return;

    const unsigned int block_idx = (unsigned int)(slot / block_size);
    const unsigned int block_offset = (unsigned int)(slot % block_size);


    const unsigned int n_elems = num_kv_heads * head_dim;


    const __nv_bfloat16* key_src = key + (unsigned long long)token_idx * key_stride;
    const __nv_bfloat16* val_src = value + (unsigned long long)token_idx * value_stride;


    const unsigned long long cache_stride = (unsigned long long)block_size * n_elems;
    __nv_bfloat16* key_dst = k_cache + (unsigned long long)block_idx * cache_stride
                                      + (unsigned long long)block_offset * n_elems;
    __nv_bfloat16* val_dst = v_cache + (unsigned long long)block_idx * cache_stride
                                      + (unsigned long long)block_offset * n_elems;



    const unsigned int n_vec = n_elems / 4;
    const unsigned int n_rem = n_elems % 4;

    const uint2* key_src_vec = (const uint2*)key_src;
    const uint2* val_src_vec = (const uint2*)val_src;
    uint2* key_dst_vec = (uint2*)key_dst;
    uint2* val_dst_vec = (uint2*)val_dst;

    for (unsigned int i = threadIdx.x; i < n_vec; i += blockDim.x) {
        key_dst_vec[i] = key_src_vec[i];
        val_dst_vec[i] = val_src_vec[i];
    }


    if (n_rem > 0) {
        unsigned int base = n_vec * 4;
        for (unsigned int i = threadIdx.x; i < n_rem; i += blockDim.x) {
            key_dst[base + i] = key_src[base + i];
            val_dst[base + i] = val_src[base + i];
        }
    }
}









// 2026-09-25: FP8 E4M3 cache write, byte = E4M3(x / scale) with saturation. k_scale and v_scale
// are the dequant scales, and cache_stride is the cache block stride in elements, both from the
// host (reshape_and_cache_fp8 in crates/model-layers/src/layers/ops/kv_cache.rs).

#include <cuda_fp8.h>


__device__ __forceinline__ __nv_fp8x2_storage_t
bf16x2_to_fp8x2(unsigned int packed_bf16, float inv_scale) {

    float v0 = __bfloat162float(__ushort_as_bfloat16((unsigned short)(packed_bf16 & 0xFFFF)));
    float v1 = __bfloat162float(__ushort_as_bfloat16((unsigned short)(packed_bf16 >> 16)));

    float2 scaled = make_float2(v0 * inv_scale, v1 * inv_scale);
    return __nv_cvt_float2_to_fp8x2(scaled, __NV_SATFINITE, __NV_E4M3);
}

extern "C" __global__ void reshape_and_cache_flash_fp8(
    const __nv_bfloat16* __restrict__ key,
    const __nv_bfloat16* __restrict__ value,
    __nv_fp8_storage_t* __restrict__ k_cache,
    __nv_fp8_storage_t* __restrict__ v_cache,
    const long long* __restrict__ slot_mapping,
    const unsigned int num_kv_heads,
    const unsigned int head_dim,
    const unsigned int block_size,
    const float k_scale,
    const float v_scale,
    const unsigned int key_stride,
    const unsigned int value_stride,
    const unsigned long long cache_stride
) {
    const unsigned int token_idx = blockIdx.x;
    const long long slot = slot_mapping[token_idx];

    if (slot < 0) return;

    const unsigned int block_idx = (unsigned int)(slot / block_size);
    const unsigned int block_offset = (unsigned int)(slot % block_size);
    const unsigned int n_elems = num_kv_heads * head_dim;


    const __nv_bfloat16* key_src = key + (unsigned long long)token_idx * key_stride;
    const __nv_bfloat16* val_src = value + (unsigned long long)token_idx * value_stride;




    __nv_fp8_storage_t* key_dst = k_cache + (unsigned long long)block_idx * cache_stride
                                           + (unsigned long long)block_offset * n_elems;
    __nv_fp8_storage_t* val_dst = v_cache + (unsigned long long)block_idx * cache_stride
                                           + (unsigned long long)block_offset * n_elems;


    const float inv_k_scale = 1.0f / k_scale;
    const float inv_v_scale = 1.0f / v_scale;



    const unsigned int n_pairs = n_elems / 2;
    const unsigned int n_rem = n_elems % 2;

    const unsigned int* key_src32 = (const unsigned int*)key_src;
    const unsigned int* val_src32 = (const unsigned int*)val_src;
    __nv_fp8x2_storage_t* key_dst16 = (__nv_fp8x2_storage_t*)key_dst;
    __nv_fp8x2_storage_t* val_dst16 = (__nv_fp8x2_storage_t*)val_dst;

    for (unsigned int i = threadIdx.x; i < n_pairs; i += blockDim.x) {
        key_dst16[i] = bf16x2_to_fp8x2(key_src32[i], inv_k_scale);
        val_dst16[i] = bf16x2_to_fp8x2(val_src32[i], inv_v_scale);
    }


    if (n_rem > 0 && threadIdx.x == 0) {
        unsigned int base = n_pairs * 2;
        float kf = __bfloat162float(key_src[base]) * inv_k_scale;
        float vf = __bfloat162float(val_src[base]) * inv_v_scale;
        key_dst[base] = __nv_cvt_float_to_fp8(kf, __NV_SATFINITE, __NV_E4M3);
        val_dst[base] = __nv_cvt_float_to_fp8(vf, __NV_SATFINITE, __NV_E4M3);
    }
}


















// 2026-09-25: NVFP4 cache write. Each thread quantizes 16-element groups: the group's E4M3 scale
// is absmax / 6, with no second-level scale, and its elements are divided by the decoded scale
// and rounded to E2M1, two per byte with the first in the low nibble.

#define NVFP4_GROUP_SIZE 16

// 2026-09-25: The same comparisons as branchless_float_to_e2m1 in e2m1_branchless.cu.
__device__ __forceinline__ unsigned char nvfp4_float_to_e2m1(float x) {
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


__device__ void nvfp4_quantize_group(
    const __nv_bfloat16* __restrict__ src,
    unsigned char* __restrict__ data_dst,
    __nv_fp8_storage_t* __restrict__ scale_dst
) {

    float vals[NVFP4_GROUP_SIZE];
    float absmax = 0.0f;
    #pragma unroll
    for (int i = 0; i < NVFP4_GROUP_SIZE; i++) {
        vals[i] = __bfloat162float(src[i]);
        float av = fabsf(vals[i]);
        absmax = fmaxf(absmax, av);
    }



    float fp8_scale_f = absmax * (1.0f / 6.0f);

    __nv_fp8_storage_t fp8_scale = __nv_cvt_float_to_fp8(
        fp8_scale_f, __NV_SATFINITE, __NV_E4M3
    );
    *scale_dst = fp8_scale;

    // 2026-09-25: Divide by the decoded scale, the value the dequant multiplies by. A scale byte that
    // decodes to 0 gives inv_scale 0, so every element of that group encodes as a zero.
    float dequant_scale = __half2float(__nv_cvt_fp8_to_halfraw(fp8_scale, __NV_E4M3));
    float inv_scale = (dequant_scale > 0.0f) ? (1.0f / dequant_scale) : 0.0f;


    #pragma unroll
    for (int i = 0; i < NVFP4_GROUP_SIZE; i += 2) {
        unsigned char lo = nvfp4_float_to_e2m1(vals[i] * inv_scale);
        unsigned char hi = nvfp4_float_to_e2m1(vals[i + 1] * inv_scale);
        data_dst[i / 2] = lo | (hi << 4);
    }
}

extern "C" __global__ void reshape_and_cache_flash_nvfp4(
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
    const unsigned int num_groups = n_elems / NVFP4_GROUP_SIZE;


    const __nv_bfloat16* key_src = key + (unsigned long long)token_idx * key_stride;
    const __nv_bfloat16* val_src = value + (unsigned long long)token_idx * value_stride;




    unsigned char* block_base_k = k_cache + (unsigned long long)block_idx * block_stride_bytes;
    unsigned char* block_base_v = v_cache + (unsigned long long)block_idx * block_stride_bytes;


    unsigned long long data_offset = (unsigned long long)block_offset * (n_elems / 2);

    unsigned long long scale_offset = data_section_bytes
        + (unsigned long long)block_offset * num_groups;

    unsigned char* key_data = block_base_k + data_offset;
    unsigned char* val_data = block_base_v + data_offset;
    __nv_fp8_storage_t* key_scales = (__nv_fp8_storage_t*)(block_base_k + scale_offset);
    __nv_fp8_storage_t* val_scales = (__nv_fp8_storage_t*)(block_base_v + scale_offset);


    for (unsigned int g = threadIdx.x; g < num_groups; g += blockDim.x) {
        unsigned int elem_offset = g * NVFP4_GROUP_SIZE;
        nvfp4_quantize_group(
            key_src + elem_offset,
            key_data + elem_offset / 2,
            key_scales + g
        );
        nvfp4_quantize_group(
            val_src + elem_offset,
            val_data + elem_offset / 2,
            val_scales + g
        );
    }
}















// 2026-09-25: Max |x| reductions for FP8 KV-scale calibration. bf16_absmax is launched by
// crates/model-layers/src/layers/fp8_calibration.rs through ops::bf16_absmax (grid-stride, at
// most 256 blocks); no launcher in crates/ resolves bf16_absmax_per_head. Both fold a block
// result into out_max with atomicMaxFloat, so the caller initialises out_max (0 for a fresh
// maximum), and both assume 256 threads per block: warp_max has 8 slots, and the per-head
// kernel strides by 256.

// 2026-09-25: Float max by a CAS loop. A value <= 0 never updates *addr.
__device__ __forceinline__ void atomicMaxFloat(float* addr, float val) {
    if (val <= 0.0f) return;
    unsigned int* addr_as_ui = (unsigned int*)addr;
    unsigned int old = *addr_as_ui;
    unsigned int assumed;
    do {
        assumed = old;
        float old_val = __uint_as_float(assumed);
        if (old_val >= val) return;
        old = atomicCAS(addr_as_ui, assumed, __float_as_uint(val));
    } while (assumed != old);
}






// 2026-09-25: Per-head max |x| of [num_tokens, num_kv_heads, head_dim] into out_max[kv_head],
// one block per KV head.
extern "C" __global__ void bf16_absmax_per_head(
    const __nv_bfloat16* __restrict__ data,
    float* __restrict__ out_max,
    const unsigned int num_tokens,
    const unsigned int num_kv_heads,
    const unsigned int head_dim
) {
    const unsigned int kv_head = blockIdx.x;
    if (kv_head >= num_kv_heads) return;

    const unsigned int tid = threadIdx.x;
    const unsigned int n_per_token = head_dim;
    const unsigned int head_stride = num_kv_heads * head_dim;

    float local_max = 0.0f;

    for (unsigned int t = 0; t < num_tokens; t++) {
        const __nv_bfloat16* row = data + (unsigned long long)t * head_stride + kv_head * head_dim;
        for (unsigned int d = tid; d < n_per_token; d += 256) {
            float v = fabsf(__bfloat162float(row[d]));
            local_max = fmaxf(local_max, v);
        }
    }


    for (int offset = 16; offset > 0; offset >>= 1) {
        local_max = fmaxf(local_max, __shfl_down_sync(0xffffffff, local_max, offset));
    }

    __shared__ float warp_max[8];
    unsigned int warp_id = tid / 32;
    unsigned int lane_id = tid % 32;
    if (lane_id == 0) {
        warp_max[warp_id] = local_max;
    }
    __syncthreads();

    if (warp_id == 0 && lane_id < 8) {
        float val = warp_max[lane_id];
        for (int offset = 4; offset > 0; offset >>= 1) {
            val = fmaxf(val, __shfl_down_sync(0xff, val, offset));
        }
        if (lane_id == 0) {
            atomicMaxFloat(&out_max[kv_head], val);
        }
    }
}

extern "C" __global__ void bf16_absmax(
    const __nv_bfloat16* __restrict__ data,
    float* __restrict__ out_max,
    const unsigned int n_elems
) {

    float local_max = 0.0f;
    const unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int stride = gridDim.x * blockDim.x;


    const unsigned int n_pairs = n_elems / 2;
    const unsigned int* data32 = (const unsigned int*)data;
    for (unsigned int i = tid; i < n_pairs; i += stride) {
        unsigned int packed = data32[i];
        float v0 = __bfloat162float(__ushort_as_bfloat16((unsigned short)(packed & 0xFFFF)));
        float v1 = __bfloat162float(__ushort_as_bfloat16((unsigned short)(packed >> 16)));
        float a0 = fabsf(v0);
        float a1 = fabsf(v1);
        local_max = fmaxf(local_max, fmaxf(a0, a1));
    }

    if (n_elems % 2 != 0 && tid == 0) {
        float v = fabsf(__bfloat162float(data[n_elems - 1]));
        local_max = fmaxf(local_max, v);
    }


    for (int offset = 16; offset > 0; offset >>= 1) {
        local_max = fmaxf(local_max, __shfl_down_sync(0xffffffff, local_max, offset));
    }


    __shared__ float warp_max[8];
    unsigned int warp_id = threadIdx.x / 32;
    unsigned int lane_id = threadIdx.x % 32;
    if (lane_id == 0) {
        warp_max[warp_id] = local_max;
    }
    __syncthreads();


    if (warp_id == 0 && lane_id < 8) {
        float val = warp_max[lane_id];
        for (int offset = 4; offset > 0; offset >>= 1) {
            val = fmaxf(val, __shfl_down_sync(0xff, val, offset));
        }
        if (lane_id == 0) {
            atomicMaxFloat(out_max, val);
        }
    }
}
