// SPDX-License-Identifier: MIT OR Apache-2.0

// 2026-09-25: Fused K path: RMSNorm, RoPE and the paged-cache write in one pass, with K kept
// in FP32 from the BF16 load until the single rounding to the cache dtype. V is not
// processed here.
//
// Owner: gb10 kernels.
// Invariants:
// - Launch: grid (num_tokens, num_kv_heads, 1), block (head_dim, 1, 1); one thread per
//   element. head_dim must be <= FUSED_KV_MAX_HEAD_DIM (256): smem_normed and the eight
//   warp_sums slots are sized for it.
// - A token whose slot_mapping entry is negative is skipped (the whole block returns).
// - k_in is [num_tokens, num_kv_heads, head_dim]; the cache is
//   [num_blocks, block_size, num_kv_heads, head_dim], slot = block * block_size + offset.










#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <math.h>

// 2026-09-25: Largest supported head_dim (shared-memory arrays).
#define FUSED_KV_MAX_HEAD_DIM 256

__device__ __forceinline__ float warp_reduce_sum_fkv(float v) {
    for (int offset = 16; offset > 0; offset >>= 1) {
        v += __shfl_xor_sync(0xFFFFFFFF, v, offset);
    }
    return v;
}

/// 2026-09-25: K path into a BF16 paged cache.
///
/// out = RoPE(x * rsqrt(mean(x^2) + eps) * (1 + k_norm_weight)), stored as BF16. The first
/// rotary_dim dims are rotated in pairs (d, d + rotary_dim / 2) for d < rotary_dim / 2 at
/// the absolute position positions[token]; the rest pass through. Frequencies use an FP64
/// pow, as common/rope.cu does. slot_mapping is i64 [num_tokens].








extern "C" __global__ void fused_k_norm_rope_cache_write_bf16(
    const __nv_bfloat16* __restrict__ k_in,
    const __nv_bfloat16* __restrict__ k_norm_weight,
    const unsigned int* __restrict__ positions,
    __nv_bfloat16* __restrict__ k_cache,
    const long long* __restrict__ slot_mapping,
    const unsigned int num_kv_heads,
    const unsigned int head_dim,
    const unsigned int rotary_dim,
    const unsigned int block_size,
    const float rms_eps,
    const float theta
) {
    const unsigned int token_idx = blockIdx.x;
    const unsigned int kv_head = blockIdx.y;
    const unsigned int t = threadIdx.x;
    if (t >= head_dim) return;


    const long long slot = slot_mapping[token_idx];
    if (slot < 0) return;


    const unsigned long long src_off =
        (unsigned long long)token_idx * num_kv_heads * head_dim
        + (unsigned long long)kv_head * head_dim;
    const __nv_bfloat16* k_row = k_in + src_off;


    float x = __bfloat162float(k_row[t]);


    float sum_sq = x * x;
    sum_sq = warp_reduce_sum_fkv(sum_sq);

    __shared__ float warp_sums[8];
    const unsigned int warp_id = t / 32;
    const unsigned int lane_id = t % 32;
    if (lane_id == 0) warp_sums[warp_id] = sum_sq;
    __syncthreads();

    if (warp_id == 0) {
        const unsigned int num_warps = (head_dim + 31) / 32;
        float v = (lane_id < num_warps) ? warp_sums[lane_id] : 0.0f;
        v = warp_reduce_sum_fkv(v);
        if (lane_id == 0) warp_sums[0] = v;
    }
    __syncthreads();
    const float rms = rsqrtf(warp_sums[0] / (float)head_dim + rms_eps);


    const float w = __bfloat162float(k_norm_weight[t]);
    const float normed = x * rms * (1.0f + w);

    // 2026-09-25: The normed values go through shared memory so each thread can read its
    // rotation partner.
    __shared__ float smem_normed[FUSED_KV_MAX_HEAD_DIM];
    smem_normed[t] = normed;
    __syncthreads();

    float out_val;
    if (t < rotary_dim) {
        const unsigned int half_rot = rotary_dim / 2;
        const bool is_d0 = (t < half_rot);
        const unsigned int pair_idx = is_d0 ? t : (t - half_rot);
        const float x0 = is_d0 ? smem_normed[t] : smem_normed[t - half_rot];
        const float x1 = is_d0 ? smem_normed[t + half_rot] : smem_normed[t];

        const double freq_exp_d = (double)(2u * pair_idx) / (double)rotary_dim;
        const float freq = (float)(1.0 / pow((double)theta, freq_exp_d));
        const unsigned int pos = positions[token_idx];
        const float angle = (float)pos * freq;
        const float cos_val = cosf(angle);
        const float sin_val = sinf(angle);
        out_val = is_d0
            ? (x0 * cos_val - x1 * sin_val)
            : (x1 * cos_val + x0 * sin_val);
    } else {
        out_val = normed;
    }


    const unsigned int block_idx = (unsigned int)(slot / (long long)block_size);
    const unsigned int block_offset = (unsigned int)(slot % (long long)block_size);
    const unsigned int n_elems = num_kv_heads * head_dim;
    const unsigned long long cache_stride = (unsigned long long)block_size * n_elems;
    const unsigned long long dst_off =
        (unsigned long long)block_idx * cache_stride
        + (unsigned long long)block_offset * n_elems
        + (unsigned long long)kv_head * head_dim
        + t;
    k_cache[dst_off] = __float2bfloat16(out_val);
}

/// 2026-09-25: MRoPE variant of `fused_k_norm_rope_cache_write_bf16`: rotation pair
/// pair_idx takes its position from pos_t, pos_h or pos_w by pair_idx % 3. With
/// pos_h == pos_w == pos_t the arithmetic is identical to the scalar-position kernel.



extern "C" __global__ void fused_k_norm_rope_mrope_cache_write_bf16(
    const __nv_bfloat16* __restrict__ k_in,
    const __nv_bfloat16* __restrict__ k_norm_weight,
    const unsigned int* __restrict__ pos_t,
    const unsigned int* __restrict__ pos_h,
    const unsigned int* __restrict__ pos_w,
    __nv_bfloat16* __restrict__ k_cache,
    const long long* __restrict__ slot_mapping,
    const unsigned int num_kv_heads,
    const unsigned int head_dim,
    const unsigned int rotary_dim,
    const unsigned int block_size,
    const float rms_eps,
    const float theta
) {
    const unsigned int token_idx = blockIdx.x;
    const unsigned int kv_head = blockIdx.y;
    const unsigned int t = threadIdx.x;
    if (t >= head_dim) return;

    const long long slot = slot_mapping[token_idx];
    if (slot < 0) return;

    const unsigned long long src_off =
        (unsigned long long)token_idx * num_kv_heads * head_dim
        + (unsigned long long)kv_head * head_dim;
    const __nv_bfloat16* k_row = k_in + src_off;

    float x = __bfloat162float(k_row[t]);

    float sum_sq = x * x;
    sum_sq = warp_reduce_sum_fkv(sum_sq);

    __shared__ float warp_sums[8];
    const unsigned int warp_id = t / 32;
    const unsigned int lane_id = t % 32;
    if (lane_id == 0) warp_sums[warp_id] = sum_sq;
    __syncthreads();

    if (warp_id == 0) {
        const unsigned int num_warps = (head_dim + 31) / 32;
        float v = (lane_id < num_warps) ? warp_sums[lane_id] : 0.0f;
        v = warp_reduce_sum_fkv(v);
        if (lane_id == 0) warp_sums[0] = v;
    }
    __syncthreads();
    const float rms = rsqrtf(warp_sums[0] / (float)head_dim + rms_eps);

    const float w = __bfloat162float(k_norm_weight[t]);
    const float normed = x * rms * (1.0f + w);

    __shared__ float smem_normed[FUSED_KV_MAX_HEAD_DIM];
    smem_normed[t] = normed;
    __syncthreads();

    float out_val;
    if (t < rotary_dim) {
        const unsigned int half_rot = rotary_dim / 2;
        const bool is_d0 = (t < half_rot);
        const unsigned int pair_idx = is_d0 ? t : (t - half_rot);
        const float x0 = is_d0 ? smem_normed[t] : smem_normed[t - half_rot];
        const float x1 = is_d0 ? smem_normed[t + half_rot] : smem_normed[t];

        const unsigned int section = pair_idx % 3u;
        unsigned int abs_pos;
        if (section == 0u) abs_pos = pos_t[token_idx];
        else if (section == 1u) abs_pos = pos_h[token_idx];
        else                    abs_pos = pos_w[token_idx];
        const double freq_exp_d = (double)(2u * pair_idx) / (double)rotary_dim;
        const float freq = (float)(1.0 / pow((double)theta, freq_exp_d));
        const float angle = (float)abs_pos * freq;
        const float cos_val = cosf(angle);
        const float sin_val = sinf(angle);
        out_val = is_d0
            ? (x0 * cos_val - x1 * sin_val)
            : (x1 * cos_val + x0 * sin_val);
    } else {
        out_val = normed;
    }

    const unsigned int block_idx = (unsigned int)(slot / (long long)block_size);
    const unsigned int block_offset = (unsigned int)(slot % (long long)block_size);
    const unsigned int n_elems = num_kv_heads * head_dim;
    const unsigned long long cache_stride = (unsigned long long)block_size * n_elems;
    const unsigned long long dst_off =
        (unsigned long long)block_idx * cache_stride
        + (unsigned long long)block_offset * n_elems
        + (unsigned long long)kv_head * head_dim
        + t;
    k_cache[dst_off] = __float2bfloat16(out_val);
}

/// 2026-09-25: `fused_k_norm_rope_cache_write_bf16` into an FP8 E4M3 paged cache, the same
/// layout as reshape_and_cache_flash_fp8. The FP32 value is multiplied by inv_scale
/// (1 / k_scale) and converted once with a saturating cast (__NV_SATFINITE).




extern "C" __global__ void fused_k_norm_rope_cache_write_fp8(
    const __nv_bfloat16* __restrict__ k_in,
    const __nv_bfloat16* __restrict__ k_norm_weight,
    const unsigned int* __restrict__ positions,
    __nv_fp8_storage_t* __restrict__ k_cache_fp8,
    const long long* __restrict__ slot_mapping,
    const unsigned int num_kv_heads,
    const unsigned int head_dim,
    const unsigned int rotary_dim,
    const unsigned int block_size,
    const float rms_eps,
    const float theta,
    const float inv_scale
) {
    const unsigned int token_idx = blockIdx.x;
    const unsigned int kv_head = blockIdx.y;
    const unsigned int t = threadIdx.x;
    if (t >= head_dim) return;

    const long long slot = slot_mapping[token_idx];
    if (slot < 0) return;

    const unsigned long long src_off =
        (unsigned long long)token_idx * num_kv_heads * head_dim
        + (unsigned long long)kv_head * head_dim;
    const __nv_bfloat16* k_row = k_in + src_off;

    float x = __bfloat162float(k_row[t]);

    float sum_sq = x * x;
    sum_sq = warp_reduce_sum_fkv(sum_sq);

    __shared__ float warp_sums[8];
    const unsigned int warp_id = t / 32;
    const unsigned int lane_id = t % 32;
    if (lane_id == 0) warp_sums[warp_id] = sum_sq;
    __syncthreads();

    if (warp_id == 0) {
        const unsigned int num_warps = (head_dim + 31) / 32;
        float v = (lane_id < num_warps) ? warp_sums[lane_id] : 0.0f;
        v = warp_reduce_sum_fkv(v);
        if (lane_id == 0) warp_sums[0] = v;
    }
    __syncthreads();
    const float rms = rsqrtf(warp_sums[0] / (float)head_dim + rms_eps);

    const float w = __bfloat162float(k_norm_weight[t]);
    const float normed = x * rms * (1.0f + w);

    __shared__ float smem_normed[FUSED_KV_MAX_HEAD_DIM];
    smem_normed[t] = normed;
    __syncthreads();

    float out_val;
    if (t < rotary_dim) {
        const unsigned int half_rot = rotary_dim / 2;
        const bool is_d0 = (t < half_rot);
        const unsigned int pair_idx = is_d0 ? t : (t - half_rot);
        const float x0 = is_d0 ? smem_normed[t] : smem_normed[t - half_rot];
        const float x1 = is_d0 ? smem_normed[t + half_rot] : smem_normed[t];
        const double freq_exp_d = (double)(2u * pair_idx) / (double)rotary_dim;
        const float freq = (float)(1.0 / pow((double)theta, freq_exp_d));
        const unsigned int pos = positions[token_idx];
        const float angle = (float)pos * freq;
        const float cos_val = cosf(angle);
        const float sin_val = sinf(angle);
        out_val = is_d0
            ? (x0 * cos_val - x1 * sin_val)
            : (x1 * cos_val + x0 * sin_val);
    } else {
        out_val = normed;
    }

    const unsigned int block_idx = (unsigned int)(slot / (long long)block_size);
    const unsigned int block_offset = (unsigned int)(slot % (long long)block_size);
    const unsigned int n_elems = num_kv_heads * head_dim;
    const unsigned long long cache_stride = (unsigned long long)block_size * n_elems;
    const unsigned long long dst_off =
        (unsigned long long)block_idx * cache_stride
        + (unsigned long long)block_offset * n_elems
        + (unsigned long long)kv_head * head_dim
        + t;

    const float scaled = out_val * inv_scale;
    k_cache_fp8[dst_off] = __nv_cvt_float_to_fp8(scaled, __NV_SATFINITE, __NV_E4M3);
}
