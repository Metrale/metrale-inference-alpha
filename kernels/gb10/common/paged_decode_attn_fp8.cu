// SPDX-License-Identifier: MIT OR Apache-2.0

// 2026-09-25: Kernels `paged_decode_attn_fp8`, `paged_decode_attn_splitk_fp8` and `paged_decode_attn_reduce_fp8`: decode
// attention over an FP8 E4M3 paged K/V cache in NHD layout, [num_blocks, block_size, num_kv_heads, head_dim] with
// physical blocks `cache_stride` elements apart, for a BF16 query, writing BF16 output.
//
// `paged_decode_attn_fp8` has the structure of `paged_decode_attn` (paged_decode_attn.cu): one CTA of 8 warps per
// (q_head, seq), grid (num_q_heads, num_seqs, 1); each warp takes a contiguous chunk of [window_start, seq_len),
// BC positions at a time within a physical block, and the warps merge through shared memory. A lane reads its
// K/V elements as VEC_U32_FP8 = HDIM / 128 uint32 loads of 4 bytes each. HDIM is 256 unless the build defines it.
//
// Owner: gb10 kernels.
// Invariants: none beyond the types. All three assume head_dim == HDIM, the first two blockDim.x == 256
// (NUM_WARPS * WARP_SIZE) and the reduce blockDim.x == 32; none checks it.








#include <cuda_bf16.h>
#include <cuda_fp8.h>

#define WARP_SIZE 32
#ifndef HDIM
#define HDIM 256
#endif
#define VEC_BF16 (HDIM / WARP_SIZE)
#define VEC_U32  (HDIM / (WARP_SIZE * 2))
#define VEC_U32_FP8 (HDIM / (WARP_SIZE * 4))
#define NUM_WARPS 8
#define BC 4


__device__ __forceinline__ void unpack2_bf16(unsigned int packed, float& v0, float& v1) {
    v0 = __bfloat162float(__ushort_as_bfloat16((unsigned short)(packed & 0xFFFF)));
    v1 = __bfloat162float(__ushort_as_bfloat16((unsigned short)(packed >> 16)));
}

// 2026-09-25: Converts 4 FP8 E4M3 bytes to F32, low byte first, with two paired fp8x2 conversions and no scale.
// The kernels apply the scales outside the position loops: k_scale in score_scale, and v_scale once per element
// when o_reg is written to shared memory.







__device__ __forceinline__ void unpack4_fp8_raw(
    unsigned int packed,
    float& v0, float& v1, float& v2, float& v3
) {
    __half2_raw h01 = __nv_cvt_fp8x2_to_halfraw2(
        (__nv_fp8x2_storage_t)(packed & 0xFFFF), __NV_E4M3);
    __half2_raw h23 = __nv_cvt_fp8x2_to_halfraw2(
        (__nv_fp8x2_storage_t)(packed >> 16), __NV_E4M3);
    const float2 f01 = __half22float2(*reinterpret_cast<const __half2*>(&h01));
    const float2 f23 = __half22float2(*reinterpret_cast<const __half2*>(&h23));
    v0 = f01.x;
    v1 = f01.y;
    v2 = f23.x;
    v3 = f23.y;
}





extern "C" __global__ void paged_decode_attn_fp8(
    const __nv_bfloat16* __restrict__ Q,             // 2026-09-25: BF16, sequence s at s * q_stride, heads contiguous
    const __nv_fp8_storage_t* __restrict__ K_cache,
    const __nv_fp8_storage_t* __restrict__ V_cache,
    __nv_bfloat16* __restrict__ O,                   // 2026-09-25: [num_seqs, num_q_heads, head_dim] BF16
    const int* __restrict__ block_tables,
    const int* __restrict__ seq_lens,
    const unsigned int max_blocks_per_seq,
    const unsigned int num_q_heads,
    const unsigned int num_kv_heads,
    const unsigned int head_dim,
    const unsigned int block_size,
    const float inv_sqrt_d,
    const float k_scale,
    const float v_scale,
    const unsigned int q_stride,              // 2026-09-25: elements between sequences' Q rows
    const unsigned long long cache_stride,    // 2026-09-25: elements between physical blocks, K and V
    const unsigned int sliding_window         // 2026-09-25: 0 attends every position; otherwise the last N
) {
    const unsigned int q_head = blockIdx.x;
    const unsigned int seq_idx = blockIdx.y;
    const unsigned int tid = threadIdx.x;
    const unsigned int warp_id = tid / WARP_SIZE;
    const unsigned int lane_id = tid % WARP_SIZE;

    if (q_head >= num_q_heads) return;

    const unsigned int seq_len = (unsigned int)seq_lens[seq_idx];
    if (seq_len == 0) return;
    const unsigned int window_start =
        (sliding_window > 0 && seq_len > sliding_window) ? (seq_len - sliding_window) : 0u;

    const unsigned int gqa_ratio = num_q_heads / num_kv_heads;
    const unsigned int kv_head = q_head / gqa_ratio;
    // 2026-09-25: K/V take one byte per element, so a lane's K/V bytes and its Q/O elements share the offset
    // lane_id * VEC_BF16.
    const unsigned int vec_offset_bf16 = lane_id * VEC_BF16;
    const unsigned int vec_offset_fp8 = lane_id * VEC_BF16;

    const int* my_block_table = block_tables + seq_idx * max_blocks_per_seq;


    const unsigned int* q32 = (const unsigned int*)(Q + (unsigned long long)seq_idx * q_stride
                                                       + (unsigned long long)q_head * head_dim + vec_offset_bf16);
    float q_reg[VEC_BF16];
    #pragma unroll
    for (int i = 0; i < VEC_U32; i++) {
        unpack2_bf16(q32[i], q_reg[2*i], q_reg[2*i+1]);
    }

    const unsigned int attended = seq_len - window_start;
    unsigned int chunk_size = (attended + NUM_WARPS - 1) / NUM_WARPS;
    unsigned int my_start = window_start + warp_id * chunk_size;
    unsigned int my_end = my_start + chunk_size;
    if (my_end > seq_len) my_end = seq_len;
    if (my_start > seq_len) my_start = seq_len;

    // 2026-09-25: Dots use raw FP8 K, so k_scale is applied here, once per score; o_reg accumulates raw V and gets
    // v_scale at the shared-memory write below.

    const float score_scale = inv_sqrt_d * k_scale;

    float m = -1e30f;
    float l = 0.0f;
    float o_reg[VEC_BF16];
    #pragma unroll
    for (int i = 0; i < VEC_BF16; i++) o_reg[i] = 0.0f;

    // 2026-09-25: Physical blocks are cache_stride elements apart (from the host); positions within a block are
    // num_kv_heads * head_dim elements apart.

    unsigned long long head_stride_kv = (unsigned long long)num_kv_heads * head_dim;

    unsigned int pos = my_start;
    while (pos < my_end) {
        unsigned int logical_block = pos / block_size;
        unsigned int block_offset = pos % block_size;
        unsigned int remaining_in_block = block_size - block_offset;
        unsigned int remaining_total = my_end - pos;
        unsigned int batch_count = remaining_in_block < remaining_total ? remaining_in_block : remaining_total;

        unsigned int physical_block = (unsigned int)my_block_table[logical_block];
        const __nv_fp8_storage_t* k_block_base = K_cache + (unsigned long long)physical_block * cache_stride
                                                          + (unsigned long long)block_offset * head_stride_kv
                                                          + (unsigned long long)kv_head * head_dim;
        const __nv_fp8_storage_t* v_block_base = V_cache + (unsigned long long)physical_block * cache_stride
                                                          + (unsigned long long)block_offset * head_stride_kv
                                                          + (unsigned long long)kv_head * head_dim;

        unsigned int processed = 0;
        unsigned int aligned_count = (batch_count / BC) * BC;


        for (; processed < aligned_count; processed += BC) {

            unsigned int k_packed[BC][VEC_U32_FP8];
            #pragma unroll
            for (int b = 0; b < BC; b++) {
                const unsigned int* k32 = (const unsigned int*)(k_block_base
                    + (unsigned long long)(processed + b) * head_stride_kv + vec_offset_fp8);
                #pragma unroll
                for (int i = 0; i < VEC_U32_FP8; i++)
                    k_packed[b][i] = k32[i];
            }



            float scores[BC];
            #pragma unroll
            for (int b = 0; b < BC; b++) {
                float dot = 0.0f;
                #pragma unroll
                for (int i = 0; i < VEC_U32_FP8; i++) {
                    float k0, k1, k2, k3;
                    unpack4_fp8_raw(k_packed[b][i], k0, k1, k2, k3);
                    dot += q_reg[4*i]   * k0 + q_reg[4*i+1] * k1
                         + q_reg[4*i+2] * k2 + q_reg[4*i+3] * k3;
                }
                #pragma unroll
                for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1)
                    dot += __shfl_xor_sync(0xffffffff, dot, offset);
                scores[b] = dot * score_scale;
            }


            unsigned int v_packed[BC][VEC_U32_FP8];
            #pragma unroll
            for (int b = 0; b < BC; b++) {
                const unsigned int* v32 = (const unsigned int*)(v_block_base
                    + (unsigned long long)(processed + b) * head_stride_kv + vec_offset_fp8);
                #pragma unroll
                for (int i = 0; i < VEC_U32_FP8; i++)
                    v_packed[b][i] = v32[i];
            }


            float m_new = m;
            #pragma unroll
            for (int b = 0; b < BC; b++)
                m_new = fmaxf(m_new, scores[b]);

            float exp_old = __expf(m - m_new);
            #pragma unroll
            for (int i = 0; i < VEC_BF16; i++)
                o_reg[i] *= exp_old;
            l *= exp_old;

            float exp_factors[BC];
            #pragma unroll
            for (int b = 0; b < BC; b++) {
                exp_factors[b] = __expf(scores[b] - m_new);
                l += exp_factors[b];
            }
            m = m_new;


            #pragma unroll
            for (int b = 0; b < BC; b++) {
                float ef = exp_factors[b];
                #pragma unroll
                for (int i = 0; i < VEC_U32_FP8; i++) {
                    float v0, v1, v2, v3;
                    unpack4_fp8_raw(v_packed[b][i], v0, v1, v2, v3);
                    o_reg[4*i]   += ef * v0;
                    o_reg[4*i+1] += ef * v1;
                    o_reg[4*i+2] += ef * v2;
                    o_reg[4*i+3] += ef * v3;
                }
            }
        }


        for (; processed < batch_count; processed++) {
            const unsigned int* k32 = (const unsigned int*)(k_block_base
                + (unsigned long long)processed * head_stride_kv + vec_offset_fp8);
            float dot = 0.0f;
            #pragma unroll
            for (int i = 0; i < VEC_U32_FP8; i++) {
                float k0, k1, k2, k3;
                unpack4_fp8_raw(k32[i], k0, k1, k2, k3);
                dot += q_reg[4*i] * k0 + q_reg[4*i+1] * k1
                     + q_reg[4*i+2] * k2 + q_reg[4*i+3] * k3;
            }
            #pragma unroll
            for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1)
                dot += __shfl_xor_sync(0xffffffff, dot, offset);

            float score = dot * score_scale;
            float m_new = fmaxf(m, score);
            float exp_old = __expf(m - m_new);
            float exp_new = __expf(score - m_new);
            l = l * exp_old + exp_new;

            const unsigned int* v32 = (const unsigned int*)(v_block_base
                + (unsigned long long)processed * head_stride_kv + vec_offset_fp8);
            #pragma unroll
            for (int i = 0; i < VEC_U32_FP8; i++) {
                float v0, v1, v2, v3;
                unpack4_fp8_raw(v32[i], v0, v1, v2, v3);
                o_reg[4*i]   = o_reg[4*i]   * exp_old + exp_new * v0;
                o_reg[4*i+1] = o_reg[4*i+1] * exp_old + exp_new * v1;
                o_reg[4*i+2] = o_reg[4*i+2] * exp_old + exp_new * v2;
                o_reg[4*i+3] = o_reg[4*i+3] * exp_old + exp_new * v3;
            }
            m = m_new;
        }

        pos += batch_count;
    }

    // 2026-09-25: Merge the warps' (m, l, o) states pairwise; a warp that attended no position (l == 0) is skipped.
    __shared__ float smem_m[NUM_WARPS];
    __shared__ float smem_l[NUM_WARPS];
    __shared__ float smem_o[NUM_WARPS][HDIM];

    if (lane_id == 0) {
        smem_m[warp_id] = m;
        smem_l[warp_id] = l;
    }
    // 2026-09-25: v_scale is applied once per element here. The merge below is linear in o, so this equals scaling
    // every V term, up to rounding.

    #pragma unroll
    for (int i = 0; i < VEC_BF16; i++) {
        smem_o[warp_id][vec_offset_bf16 + i] = o_reg[i] * v_scale;
    }
    __syncthreads();

    #pragma unroll
    for (int stride = NUM_WARPS / 2; stride > 0; stride >>= 1) {
        if (warp_id < (unsigned int)stride) {
            unsigned int other = warp_id + stride;
            float lw = smem_l[other];
            if (lw > 0.0f) {
                float mw = smem_m[other];
                float my_m = smem_m[warp_id];
                float my_l = smem_l[warp_id];
                float m_new = fmaxf(my_m, mw);
                float scale_me = __expf(my_m - m_new);
                float scale_w = __expf(mw - m_new);
                smem_l[warp_id] = my_l * scale_me + lw * scale_w;
                smem_m[warp_id] = m_new;
                #pragma unroll
                for (int i = 0; i < VEC_BF16; i++) {
                    smem_o[warp_id][vec_offset_bf16 + i] =
                        smem_o[warp_id][vec_offset_bf16 + i] * scale_me +
                        smem_o[other][vec_offset_bf16 + i] * scale_w;
                }
            }
        }
        __syncthreads();
    }

    if (warp_id == 0) {
        float final_l = smem_l[0];
        float inv_l = (final_l > 0.0f) ? (1.0f / final_l) : 0.0f;
        unsigned int* o32 = (unsigned int*)(O + (unsigned long long)seq_idx * num_q_heads * head_dim
                                              + (unsigned long long)q_head * head_dim + vec_offset_bf16);
        #pragma unroll
        for (int i = 0; i < VEC_U32; i++) {
            float v0 = smem_o[0][vec_offset_bf16 + 2*i]     * inv_l;
            float v1 = smem_o[0][vec_offset_bf16 + 2*i + 1] * inv_l;
            unsigned int lo = (unsigned int)__bfloat16_as_ushort(__float2bfloat16(v0));
            unsigned int hi = (unsigned int)__bfloat16_as_ushort(__float2bfloat16(v1));
            o32[i] = lo | (hi << 16);
        }
    }
}

// 2026-09-25: `paged_decode_attn_splitk_fp8` splits [window_start, seq_len) into num_splits contiguous ranges, one CTA
// per (q_head, split, seq), grid (num_q_heads, num_splits, num_seqs). Each CTA writes its range's unnormalised
// output and (m, l) to `workspace`; `paged_decode_attn_reduce_fp8` merges them.


extern "C" __global__ void paged_decode_attn_splitk_fp8(
    const __nv_bfloat16* __restrict__ Q,
    const __nv_fp8_storage_t* __restrict__ K_cache,
    const __nv_fp8_storage_t* __restrict__ V_cache,
    float* __restrict__ workspace,
    const int* __restrict__ block_tables,
    const int* __restrict__ seq_lens,
    const unsigned int max_blocks_per_seq,
    const unsigned int num_q_heads,
    const unsigned int num_kv_heads,
    const unsigned int head_dim,
    const unsigned int block_size,
    const float inv_sqrt_d,
    const unsigned int num_splits,
    const float k_scale,
    const float v_scale,
    const unsigned int q_stride,              // 2026-09-25: elements between sequences' Q rows
    const unsigned long long cache_stride,    // 2026-09-25: elements between physical blocks, K and V
    const unsigned int sliding_window         // 2026-09-25: 0 attends every position; otherwise the last N
) {
    const unsigned int q_head = blockIdx.x;
    const unsigned int split_id = blockIdx.y;
    const unsigned int seq_idx = blockIdx.z;
    const unsigned int tid = threadIdx.x;
    const unsigned int warp_id = tid / WARP_SIZE;
    const unsigned int lane_id = tid % WARP_SIZE;

    if (q_head >= num_q_heads) return;

    const unsigned int seq_len = (unsigned int)seq_lens[seq_idx];
    if (seq_len == 0) return;
    const unsigned int window_start =
        (sliding_window > 0 && seq_len > sliding_window) ? (seq_len - sliding_window) : 0u;

    const unsigned int attended = seq_len - window_start;
    unsigned int split_size = (attended + num_splits - 1) / num_splits;
    unsigned int kv_start = window_start + split_id * split_size;
    unsigned int kv_end = kv_start + split_size;
    if (kv_end > seq_len) kv_end = seq_len;
    if (kv_start >= seq_len) kv_start = kv_end;

    const unsigned int gqa_ratio = num_q_heads / num_kv_heads;
    const unsigned int kv_head = q_head / gqa_ratio;
    const unsigned int vec_offset_bf16 = lane_id * VEC_BF16;
    const unsigned int vec_offset_fp8 = lane_id * VEC_BF16;

    const int* my_block_table = block_tables + seq_idx * max_blocks_per_seq;


    const unsigned int* q32 = (const unsigned int*)(Q + (unsigned long long)seq_idx * q_stride
                                                       + (unsigned long long)q_head * head_dim + vec_offset_bf16);
    float q_reg[VEC_BF16];
    #pragma unroll
    for (int i = 0; i < VEC_U32; i++) {
        unpack2_bf16(q32[i], q_reg[2*i], q_reg[2*i+1]);
    }

    unsigned int local_len = kv_end - kv_start;
    unsigned int chunk_size = (local_len + NUM_WARPS - 1) / NUM_WARPS;
    unsigned int my_start = kv_start + warp_id * chunk_size;
    unsigned int my_end = my_start + chunk_size;
    if (my_end > kv_end) my_end = kv_end;
    if (my_start > kv_end) my_start = kv_end;

    float m_val = -1e30f;
    float l_val = 0.0f;
    float o_reg[VEC_BF16];
    #pragma unroll
    for (int i = 0; i < VEC_BF16; i++) o_reg[i] = 0.0f;

    // 2026-09-25: k_scale folds into score_scale and v_scale into the shared-memory write, as in paged_decode_attn_fp8.
    const float score_scale = inv_sqrt_d * k_scale;


    unsigned long long head_stride_kv = (unsigned long long)num_kv_heads * head_dim;

    for (unsigned int pos = my_start; pos < my_end; pos++) {
        unsigned int logical_block = pos / block_size;
        unsigned int block_offset = pos % block_size;
        unsigned int physical_block = (unsigned int)my_block_table[logical_block];

        const unsigned int* k32 = (const unsigned int*)(K_cache
            + (unsigned long long)physical_block * cache_stride
            + (unsigned long long)block_offset * head_stride_kv
            + (unsigned long long)kv_head * head_dim + vec_offset_fp8);

        float dot = 0.0f;
        #pragma unroll
        for (int i = 0; i < VEC_U32_FP8; i++) {
            float k0, k1, k2, k3;
            unpack4_fp8_raw(k32[i], k0, k1, k2, k3);
            dot += q_reg[4*i] * k0 + q_reg[4*i+1] * k1
                 + q_reg[4*i+2] * k2 + q_reg[4*i+3] * k3;
        }
        #pragma unroll
        for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1)
            dot += __shfl_xor_sync(0xffffffff, dot, offset);

        float score = dot * score_scale;
        float m_new = fmaxf(m_val, score);
        float exp_old = __expf(m_val - m_new);
        float exp_new = __expf(score - m_new);
        l_val = l_val * exp_old + exp_new;

        const unsigned int* v32 = (const unsigned int*)(V_cache
            + (unsigned long long)physical_block * cache_stride
            + (unsigned long long)block_offset * head_stride_kv
            + (unsigned long long)kv_head * head_dim + vec_offset_fp8);

        #pragma unroll
        for (int i = 0; i < VEC_U32_FP8; i++) {
            float v0, v1, v2, v3;
            unpack4_fp8_raw(v32[i], v0, v1, v2, v3);
            o_reg[4*i]   = o_reg[4*i]   * exp_old + exp_new * v0;
            o_reg[4*i+1] = o_reg[4*i+1] * exp_old + exp_new * v1;
            o_reg[4*i+2] = o_reg[4*i+2] * exp_old + exp_new * v2;
            o_reg[4*i+3] = o_reg[4*i+3] * exp_old + exp_new * v3;
        }
        m_val = m_new;
    }


    __shared__ float smem_m[NUM_WARPS];
    __shared__ float smem_l[NUM_WARPS];
    __shared__ float smem_o[NUM_WARPS][HDIM];

    if (lane_id == 0) {
        smem_m[warp_id] = m_val;
        smem_l[warp_id] = l_val;
    }
    // 2026-09-25: v_scale is applied once per element here; the merge below and the reduce kernel are linear in o.

    #pragma unroll
    for (int i = 0; i < VEC_BF16; i++) {
        smem_o[warp_id][vec_offset_bf16 + i] = o_reg[i] * v_scale;
    }
    __syncthreads();

    #pragma unroll
    for (int stride = NUM_WARPS / 2; stride > 0; stride >>= 1) {
        if (warp_id < (unsigned int)stride) {
            unsigned int other = warp_id + stride;
            float lw = smem_l[other];
            if (lw > 0.0f) {
                float mw = smem_m[other];
                float my_m = smem_m[warp_id];
                float my_l = smem_l[warp_id];
                float m_new = fmaxf(my_m, mw);
                float scale_me = __expf(my_m - m_new);
                float scale_w = __expf(mw - m_new);
                smem_l[warp_id] = my_l * scale_me + lw * scale_w;
                smem_m[warp_id] = m_new;
                #pragma unroll
                for (int i = 0; i < VEC_BF16; i++) {
                    smem_o[warp_id][vec_offset_bf16 + i] =
                        smem_o[warp_id][vec_offset_bf16 + i] * scale_me +
                        smem_o[other][vec_offset_bf16 + i] * scale_w;
                }
            }
        }
        __syncthreads();
    }


    unsigned int ws_stride = (head_dim + 2);
    float* ws_base = workspace + ((unsigned long long)seq_idx * num_q_heads + q_head) * num_splits * ws_stride
                   + split_id * ws_stride;

    if (warp_id == 0) {
        #pragma unroll
        for (int i = 0; i < VEC_BF16; i++) {
            ws_base[vec_offset_bf16 + i] = smem_o[0][vec_offset_bf16 + i];
        }
        if (lane_id == 0) {
            ws_base[head_dim] = smem_m[0];
            ws_base[head_dim + 1] = smem_l[0];
        }
    }
}

// 2026-09-25: `paged_decode_attn_reduce_fp8`: merges one (q_head, seq)'s num_splits partials, skipping splits with
// l <= 0, and writes normalised BF16 O. Workspace row [seq_idx, q_head, split] is head_dim F32 values, then m,
// then l. Grid (num_q_heads, num_seqs, 1), one warp per CTA. A sequence with seq_len 0 is left unwritten.






extern "C" __global__ void paged_decode_attn_reduce_fp8(
    const float* __restrict__ workspace,
    __nv_bfloat16* __restrict__ O,          // 2026-09-25: [num_seqs, num_q_heads, head_dim] BF16
    const int* __restrict__ seq_lens,
    const unsigned int num_q_heads,
    const unsigned int head_dim,
    const unsigned int num_splits
) {
    const unsigned int q_head = blockIdx.x;
    const unsigned int seq_idx = blockIdx.y;
    const unsigned int lane_id = threadIdx.x;

    if (q_head >= num_q_heads) return;
    if (seq_lens[seq_idx] == 0) return;

    const unsigned int vec_off = lane_id * VEC_BF16;
    const unsigned int ws_stride = head_dim + 2;
    const float* ws_base = workspace
        + ((unsigned long long)seq_idx * num_q_heads + q_head) * num_splits * ws_stride;


    float m = ws_base[head_dim];
    float l = ws_base[head_dim + 1];
    float o_reg[VEC_BF16];
    #pragma unroll
    for (int i = 0; i < VEC_BF16; i++)
        o_reg[i] = ws_base[vec_off + i];


    for (unsigned int s = 1; s < num_splits; s++) {
        const float* ws = ws_base + s * ws_stride;
        float ms = ws[head_dim];
        float ls = ws[head_dim + 1];

        if (ls <= 0.0f) continue;

        float m_new = fmaxf(m, ms);
        float scale_me = __expf(m - m_new);
        float scale_s = __expf(ms - m_new);

        #pragma unroll
        for (int i = 0; i < VEC_BF16; i++)
            o_reg[i] = o_reg[i] * scale_me + ws[vec_off + i] * scale_s;

        l = l * scale_me + ls * scale_s;
        m = m_new;
    }


    float inv_l = (l > 0.0f) ? (1.0f / l) : 0.0f;
    unsigned int* o32 = (unsigned int*)(O + (unsigned long long)seq_idx * num_q_heads * head_dim
                                          + (unsigned long long)q_head * head_dim + vec_off);
    #pragma unroll
    for (int i = 0; i < VEC_U32; i++) {
        float v0 = o_reg[2*i] * inv_l;
        float v1 = o_reg[2*i + 1] * inv_l;
        unsigned int lo = (unsigned int)__bfloat16_as_ushort(__float2bfloat16(v0));
        unsigned int hi = (unsigned int)__bfloat16_as_ushort(__float2bfloat16(v1));
        o32[i] = lo | (hi << 16);
    }
}
