// SPDX-License-Identifier: MIT OR Apache-2.0

// 2026-09-25: Fused K norm + RoPE + FP8 paged cache write of K and V, for single-token decode.
//
// It stands in for this chain on the FP8-KV decode path:
//   rms_norm (rms_norm.cu, module `norm`) on K, one row per KV head
//   rope_forward (rope.cu) on K
//   reshape_and_cache_flash_fp8 (reshape_and_cache.cu) on K and V
// The caller then launches rope_forward with num_kv_heads = 0, for Q only, and takes this path
// only when fused_fp8_kv_decode_eligible holds (crates/model-layers/src/layers/qwen3_attention/
// decode/attention_forward.rs and decode/write_kv_cache_fp8.rs).
//
// The kernel writes the chain's cache bytes, so it repeats the chain's roundings rather than
// keeping FP32 intermediates (KERNEL.toml builds this tree with --fmad=false):
// - the sum of squares uses rms_norm's thread assignment (thread t owns BF16 pair t) and its
//   shuffle + warp_sums[32] tree, with blockDim = head_dim as in rms_norm's per-head launch;
// - the normed value is rounded to BF16 where rms_norm stores it, and kept as BF16 bits in
//   shared memory;
// - the rotated value is rounded to BF16 where rope_forward stores it;
// - the FP8 cast is the same paired __nv_cvt_float2_to_fp8x2 on the same pairs, with the same
//   in-kernel 1.0f / scale.
// decode/write_kv_cache_fp8_gpu_tests.rs, a GPU test run with --ignored, compares the two paths
// byte for byte. Removing a rounding here would change the cache bytes.
//
// Grid (num_tokens, num_kv_heads, 1), block (head_dim, 1, 1).
//
// Owner: gb10 kernels.
// Invariants:
// - blockDim.x == head_dim, a multiple of 32 and at most FUSED_KFP8_MAX_HEAD_DIM, and rotary_dim
//   is even and at most head_dim. The kernel checks none of this; fused_fp8_kv_decode_eligible
//   does.
// - A token whose slot is negative writes nothing.
















#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <math.h>

// 2026-09-25: Sizes the shared normed row; FUSED_FP8_MAX_HEAD_DIM in decode/write_kv_cache_fp8.rs matches it.
#define FUSED_KFP8_MAX_HEAD_DIM 256

// 2026-09-25: Copies of unpack_bf16x2, pack_bf16x2 and warp_reduce_sum in rms_norm.cu, renamed.
// The bodies must stay identical for the byte parity.
__device__ __forceinline__ void fkfp8_unpack_bf16x2(unsigned int packed, float& v0, float& v1) {
    v0 = __bfloat162float(__ushort_as_bfloat16((unsigned short)(packed & 0xFFFF)));
    v1 = __bfloat162float(__ushort_as_bfloat16((unsigned short)(packed >> 16)));
}

__device__ __forceinline__ unsigned int fkfp8_pack_bf16x2(float v0, float v1) {
    unsigned int lo = (unsigned int)__bfloat16_as_ushort(__float2bfloat16(v0));
    unsigned int hi = (unsigned int)__bfloat16_as_ushort(__float2bfloat16(v1));
    return lo | (hi << 16);
}

__device__ __forceinline__ float fkfp8_warp_reduce_sum(float val) {
    for (int offset = 16; offset > 0; offset >>= 1) {
        val += __shfl_xor_sync(0xFFFFFFFF, val, offset);
    }
    return val;
}

// 2026-09-25: Copy of bf16x2_to_fp8x2 in reshape_and_cache.cu.
__device__ __forceinline__ __nv_fp8x2_storage_t
fkfp8_bf16x2_to_fp8x2(unsigned int packed_bf16, float inv_scale) {
    float v0 = __bfloat162float(__ushort_as_bfloat16((unsigned short)(packed_bf16 & 0xFFFF)));
    float v1 = __bfloat162float(__ushort_as_bfloat16((unsigned short)(packed_bf16 >> 16)));
    float2 scaled = make_float2(v0 * inv_scale, v1 * inv_scale);
    return __nv_cvt_float2_to_fp8x2(scaled, __NV_SATFINITE, __NV_E4M3);
}













// 2026-09-25: One output element of rope_forward's rotation, read from the BF16 normed row:
// pairs (d, d + rotary_dim/2) for d < rotary_dim/2, and elements from rotary_dim on come back
// unrotated. s_cos and s_sin are computed once per block with rope_forward's FP64 frequency
// expression and angle.
__device__ __forceinline__ float fkfp8_rope_elem(
    const unsigned short* __restrict__ s_normed,
    const float* __restrict__ s_cos,
    const float* __restrict__ s_sin,
    const unsigned int e,
    const unsigned int rotary_dim
) {
    if (e >= rotary_dim) {
        return __bfloat162float(__ushort_as_bfloat16(s_normed[e]));
    }
    const unsigned int half_rot = rotary_dim / 2;
    const bool is_d0 = (e < half_rot);
    const unsigned int pair_idx = is_d0 ? e : (e - half_rot);
    const float cos_val = s_cos[pair_idx];
    const float sin_val = s_sin[pair_idx];
    const float x0 = __bfloat162float(__ushort_as_bfloat16(s_normed[pair_idx]));
    const float x1 = __bfloat162float(__ushort_as_bfloat16(s_normed[pair_idx + half_rot]));
    return is_d0 ? (x0 * cos_val - x1 * sin_val)
                 : (x1 * cos_val + x0 * sin_val);
}






// 2026-09-25: k_in is [num_tokens, key_stride] BF16 with the head at kv_head * head_dim, and
// value likewise with value_stride; this kernel only quantizes V. k_scale / v_scale are dequant
// scales (bf16 = fp8 * scale), and cache_stride is the cache block stride in elements, as for
// reshape_and_cache_flash_fp8.
extern "C" __global__ void fused_k_norm_rope_cache_write_fp8_kv(
    const __nv_bfloat16* __restrict__ k_in,
    const __nv_bfloat16* __restrict__ value,
    const __nv_bfloat16* __restrict__ k_norm_weight,
    const unsigned int*  __restrict__ positions,
    __nv_fp8_storage_t*  __restrict__ k_cache,
    __nv_fp8_storage_t*  __restrict__ v_cache,
    const long long*     __restrict__ slot_mapping,
    const unsigned int num_kv_heads,
    const unsigned int head_dim,
    const unsigned int rotary_dim,
    const unsigned int block_size,
    const float k_scale,
    const float v_scale,
    const unsigned int key_stride,
    const unsigned int value_stride,
    const unsigned long long cache_stride,
    const float rms_eps,
    const float theta
) {
    const unsigned int token_idx = blockIdx.x;
    const unsigned int kv_head   = blockIdx.y;
    const unsigned int tid       = threadIdx.x;

    // 2026-09-25: Uniform across the block (it depends only on blockIdx.x), so this return cannot
    // strand a __syncthreads().
    const long long slot = slot_mapping[token_idx];
    if (slot < 0) return;

    const unsigned int half_size = head_dim / 2;

    const __nv_bfloat16* k_row = k_in
        + (unsigned long long)token_idx * key_stride
        + (unsigned long long)kv_head * head_dim;
    const unsigned int* x32 = (const unsigned int*)k_row;


    float sum_sq = 0.0f;
    for (unsigned int i = tid; i < half_size; i += blockDim.x) {
        float v0, v1;
        fkfp8_unpack_bf16x2(x32[i], v0, v1);
        sum_sq += v0 * v0 + v1 * v1;
    }
    sum_sq = fkfp8_warp_reduce_sum(sum_sq);

    __shared__ float warp_sums[32];
    const unsigned int warp_id = tid / 32;
    const unsigned int lane_id = tid % 32;
    if (lane_id == 0) {
        warp_sums[warp_id] = sum_sq;
    }
    __syncthreads();
    if (warp_id == 0) {
        float val = (lane_id < (blockDim.x + 31) / 32) ? warp_sums[lane_id] : 0.0f;
        val = fkfp8_warp_reduce_sum(val);
        if (lane_id == 0) {
            warp_sums[0] = val;
        }
    }
    __syncthreads();


    const float rms = rsqrtf(warp_sums[0] / (float)head_dim + rms_eps);




    // 2026-09-25: __align__(4): stage 2 stores through an unsigned int view of this array, and an
    // unsigned short array is only guaranteed 2-byte alignment.
    __shared__ __align__(4) unsigned short s_normed[FUSED_KFP8_MAX_HEAD_DIM];
    const unsigned int* w32 = (const unsigned int*)k_norm_weight;
    for (unsigned int i = tid; i < half_size; i += blockDim.x) {
        float xv0, xv1, wv0, wv1;
        fkfp8_unpack_bf16x2(x32[i], xv0, xv1);
        fkfp8_unpack_bf16x2(w32[i], wv0, wv1);
        ((unsigned int*)s_normed)[i] =
            fkfp8_pack_bf16x2(xv0 * rms * (1.0f + wv0), xv1 * rms * (1.0f + wv1));
    }


    // 2026-09-25: cos and sin once per pair, since the block covers one token. This loop shares the
    // barrier below with the store above.
    __shared__ float s_cos[FUSED_KFP8_MAX_HEAD_DIM / 2];
    __shared__ float s_sin[FUSED_KFP8_MAX_HEAD_DIM / 2];
    const unsigned int half_rot = rotary_dim / 2;
    const unsigned int abs_pos = positions[token_idx];
    for (unsigned int i = tid; i < half_rot; i += blockDim.x) {

        const double fe = (double)(2 * i) / (double)rotary_dim;
        const float freq = (float)(1.0 / pow((double)theta, fe));
        const float angle = (float)abs_pos * freq;
        s_cos[i] = cosf(angle);
        s_sin[i] = sinf(angle);
    }
    __syncthreads();


    // 2026-09-25: Threads from head_dim / 2 on have no pair to write and no barrier left.
    if (tid >= half_size) return;


    const float y0 = fkfp8_rope_elem(s_normed, s_cos, s_sin, 2u * tid, rotary_dim);
    const float y1 = fkfp8_rope_elem(s_normed, s_cos, s_sin, 2u * tid + 1u, rotary_dim);
    const unsigned int packed_k = fkfp8_pack_bf16x2(y0, y1);


    const unsigned int block_idx    = (unsigned int)(slot / block_size);
    const unsigned int block_offset = (unsigned int)(slot % block_size);
    const unsigned int n_elems      = num_kv_heads * head_dim;

    __nv_fp8_storage_t* key_dst = k_cache
        + (unsigned long long)block_idx * cache_stride
        + (unsigned long long)block_offset * n_elems;
    __nv_fp8_storage_t* val_dst = v_cache
        + (unsigned long long)block_idx * cache_stride
        + (unsigned long long)block_offset * n_elems;

    const float inv_k_scale = 1.0f / k_scale;
    const float inv_v_scale = 1.0f / v_scale;

    // 2026-09-25: head_dim is even, so `pair` is the index reshape_and_cache_flash_fp8 uses for
    // these two elements.
    const unsigned int pair = (kv_head * head_dim) / 2u + tid;

    ((__nv_fp8x2_storage_t*)key_dst)[pair] =
        fkfp8_bf16x2_to_fp8x2(packed_k, inv_k_scale);

    const __nv_bfloat16* v_row = value
        + (unsigned long long)token_idx * value_stride
        + (unsigned long long)kv_head * head_dim;
    ((__nv_fp8x2_storage_t*)val_dst)[pair] =
        fkfp8_bf16x2_to_fp8x2(((const unsigned int*)v_row)[tid], inv_v_scale);
}
