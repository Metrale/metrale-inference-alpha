// SPDX-License-Identifier: MIT OR Apache-2.0
// forked-from: kernels/gb10/deepseek-v4-flash/nvfp4/paged_decode_attn_512.cu (2026-09-24; 2 of 527 lines differ, see kernels/FORKS.md)

// 2026-09-25: HDIM=512 paged decode attention over a BF16 KV cache. Owner: gb10 kernels (gemma-4-26b-a4b). Invariants: none checked; head_dim must equal HDIM and num_kv_heads must divide num_q_heads.
#undef HDIM
#define HDIM 512

#include <cuda_bf16.h>

#define WARP_SIZE 32
#ifndef HDIM
#define HDIM 256
#endif
#define VEC_BF16 (HDIM / WARP_SIZE)
#define VEC_U32  (HDIM / (WARP_SIZE * 2))
#define NUM_WARPS 8
#define BC 4

__device__ __forceinline__ void unpack2_pd(unsigned int packed, float& v0, float& v1) {
    v0 = __bfloat162float(__ushort_as_bfloat16((unsigned short)(packed & 0xFFFF)));
    v1 = __bfloat162float(__ushort_as_bfloat16((unsigned short)(packed >> 16)));
}


__device__ __forceinline__ const __nv_bfloat16* paged_kv_ptr(
    const __nv_bfloat16* __restrict__ cache,
    const int* __restrict__ block_table,
    unsigned int pos,
    unsigned int block_size,
    unsigned int num_kv_heads,
    unsigned int head_dim,
    unsigned int kv_head
) {
    unsigned int logical_block = pos / block_size;
    unsigned int block_offset = pos % block_size;
    unsigned int physical_block = (unsigned int)block_table[logical_block];
    unsigned long long page_stride = (unsigned long long)block_size * num_kv_heads * head_dim;
    return cache + (unsigned long long)physical_block * page_stride
                 + (unsigned long long)block_offset * num_kv_heads * head_dim
                 + (unsigned long long)kv_head * head_dim;
}
// 2026-09-25: One 256-thread block per (q_head, seq) (ops::paged_decode_attn_bf16), looked up as paged_decode_attn_512::paged_decode_attn for a BF16 KV cache and used when head_dim > 256. With sliding_window > 0 only the last sliding_window positions are attended.
extern "C" __global__ void paged_decode_attn(
    const __nv_bfloat16* __restrict__ Q,
    const __nv_bfloat16* __restrict__ K_cache,
    const __nv_bfloat16* __restrict__ V_cache,
    __nv_bfloat16* __restrict__ O,
    const int* __restrict__ block_tables,
    const int* __restrict__ seq_lens,
    const unsigned int max_blocks_per_seq,
    const unsigned int num_q_heads,
    const unsigned int num_kv_heads,
    const unsigned int head_dim,
    const unsigned int block_size,
    const float inv_sqrt_d,
    const unsigned int q_stride,
    const unsigned int sliding_window
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
    const unsigned int vec_offset = lane_id * VEC_BF16;


    const int* my_block_table = block_tables + seq_idx * max_blocks_per_seq;

    // 2026-09-25: Q rows are q_stride elements apart; O rows are num_q_heads * head_dim.
    const unsigned int* q32 = (const unsigned int*)(Q + (unsigned long long)seq_idx * q_stride
                                                       + (unsigned long long)q_head * head_dim + vec_offset);
    float q_reg[VEC_BF16];
    #pragma unroll
    for (int i = 0; i < VEC_U32; i++) {
        unpack2_pd(q32[i], q_reg[2*i], q_reg[2*i+1]);
    }

    // 2026-09-25: Each warp takes a contiguous 1/NUM_WARPS share of [window_start, seq_len).
    const unsigned int attended = seq_len - window_start;
    unsigned int chunk_size = (attended + NUM_WARPS - 1) / NUM_WARPS;
    unsigned int my_start = window_start + warp_id * chunk_size;
    unsigned int my_end = my_start + chunk_size;
    if (my_end > seq_len) my_end = seq_len;
    if (my_start > seq_len) my_start = seq_len;


    float m = -1e30f;
    float l = 0.0f;
    float o_reg[VEC_BF16];
    #pragma unroll
    for (int i = 0; i < VEC_BF16; i++) o_reg[i] = 0.0f;

    // 2026-09-25: Positions that share a physical block are read BC at a time from that block's base; the tail
    // of a block is read one position at a time.

    unsigned int pos = my_start;

    while (pos < my_end) {

        unsigned int logical_block = pos / block_size;
        unsigned int block_offset = pos % block_size;
        unsigned int remaining_in_block = block_size - block_offset;
        unsigned int remaining_total = my_end - pos;
        unsigned int batch_count = remaining_in_block < remaining_total ? remaining_in_block : remaining_total;


        unsigned int physical_block = (unsigned int)my_block_table[logical_block];
        unsigned long long page_stride = (unsigned long long)block_size * num_kv_heads * head_dim;
        unsigned long long head_stride_kv = (unsigned long long)num_kv_heads * head_dim;
        const __nv_bfloat16* k_block_base = K_cache + (unsigned long long)physical_block * page_stride
                                                     + (unsigned long long)block_offset * head_stride_kv
                                                     + (unsigned long long)kv_head * head_dim;
        const __nv_bfloat16* v_block_base = V_cache + (unsigned long long)physical_block * page_stride
                                                     + (unsigned long long)block_offset * head_stride_kv
                                                     + (unsigned long long)kv_head * head_dim;


        unsigned int processed = 0;
        unsigned int aligned_count = (batch_count / BC) * BC;


        for (; processed < aligned_count; processed += BC) {

            unsigned int k_packed[BC][VEC_U32];
            #pragma unroll
            for (int b = 0; b < BC; b++) {
                const unsigned int* k32 = (const unsigned int*)(k_block_base
                    + (unsigned long long)(processed + b) * head_stride_kv + vec_offset);
                #pragma unroll
                for (int i = 0; i < VEC_U32; i++)
                    k_packed[b][i] = k32[i];
            }


            float scores[BC];
            #pragma unroll
            for (int b = 0; b < BC; b++) {
                float dot = 0.0f;
                #pragma unroll
                for (int i = 0; i < VEC_U32; i++) {
                    float k0, k1;
                    unpack2_pd(k_packed[b][i], k0, k1);
                    dot += q_reg[2*i] * k0 + q_reg[2*i+1] * k1;
                }
                #pragma unroll
                for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1)
                    dot += __shfl_xor_sync(0xffffffff, dot, offset);
                scores[b] = dot * inv_sqrt_d;
            }


            unsigned int v_packed[BC][VEC_U32];
            #pragma unroll
            for (int b = 0; b < BC; b++) {
                const unsigned int* v32 = (const unsigned int*)(v_block_base
                    + (unsigned long long)(processed + b) * head_stride_kv + vec_offset);
                #pragma unroll
                for (int i = 0; i < VEC_U32; i++)
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
                for (int i = 0; i < VEC_U32; i++) {
                    float v0, v1;
                    unpack2_pd(v_packed[b][i], v0, v1);
                    o_reg[2*i]   += ef * v0;
                    o_reg[2*i+1] += ef * v1;
                }
            }
        }


        for (; processed < batch_count; processed++) {
            const unsigned int* k32 = (const unsigned int*)(k_block_base
                + (unsigned long long)processed * head_stride_kv + vec_offset);
            float dot = 0.0f;
            #pragma unroll
            for (int i = 0; i < VEC_U32; i++) {
                float k0, k1;
                unpack2_pd(k32[i], k0, k1);
                dot += q_reg[2*i] * k0 + q_reg[2*i+1] * k1;
            }
            #pragma unroll
            for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1)
                dot += __shfl_xor_sync(0xffffffff, dot, offset);

            float score = dot * inv_sqrt_d;
            float m_new = fmaxf(m, score);
            float exp_old = __expf(m - m_new);
            float exp_new = __expf(score - m_new);
            l = l * exp_old + exp_new;

            const unsigned int* v32 = (const unsigned int*)(v_block_base
                + (unsigned long long)processed * head_stride_kv + vec_offset);
            #pragma unroll
            for (int i = 0; i < VEC_U32; i++) {
                float v0, v1;
                unpack2_pd(v32[i], v0, v1);
                o_reg[2*i]   = o_reg[2*i]   * exp_old + exp_new * v0;
                o_reg[2*i+1] = o_reg[2*i+1] * exp_old + exp_new * v1;
            }
            m = m_new;
        }

        pos += batch_count;
    }


    __shared__ float smem_m[NUM_WARPS];
    __shared__ float smem_l[NUM_WARPS];
    __shared__ float smem_o[NUM_WARPS][HDIM];

    if (lane_id == 0) {
        smem_m[warp_id] = m;
        smem_l[warp_id] = l;
    }
    #pragma unroll
    for (int i = 0; i < VEC_BF16; i++) {
        smem_o[warp_id][vec_offset + i] = o_reg[i];
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
                    smem_o[warp_id][vec_offset + i] =
                        smem_o[warp_id][vec_offset + i] * scale_me +
                        smem_o[other][vec_offset + i] * scale_w;
                }
            }
        }
        __syncthreads();
    }


    if (warp_id == 0) {
        float final_l = smem_l[0];
        float inv_l = (final_l > 0.0f) ? (1.0f / final_l) : 0.0f;
        unsigned int* o32 = (unsigned int*)(O + (unsigned long long)seq_idx * num_q_heads * head_dim
                                              + (unsigned long long)q_head * head_dim + vec_offset);
        #pragma unroll
        for (int i = 0; i < VEC_U32; i++) {
            float v0 = smem_o[0][vec_offset + 2*i]     * inv_l;
            float v1 = smem_o[0][vec_offset + 2*i + 1] * inv_l;
            unsigned int lo = (unsigned int)__bfloat16_as_ushort(__float2bfloat16(v0));
            unsigned int hi = (unsigned int)__bfloat16_as_ushort(__float2bfloat16(v1));
            o32[i] = lo | (hi << 16);
        }
    }
}

// 2026-09-25: Split-K form: block (q_head, split, seq) covers its 1/num_splits share of [0, seq_len) (no sliding
// window) and writes unnormalised O, m and l to workspace[seq][head][split][head_dim + 2]; paged_decode_attn_reduce
// merges the splits into BF16 O. Nothing in crates/ looks up either kernel in this module.


extern "C" __global__ void paged_decode_attn_splitk(
    const __nv_bfloat16* __restrict__ Q,
    const __nv_bfloat16* __restrict__ K_cache,
    const __nv_bfloat16* __restrict__ V_cache,
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
    const unsigned int q_stride
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


    unsigned int split_size = (seq_len + num_splits - 1) / num_splits;
    unsigned int kv_start = split_id * split_size;
    unsigned int kv_end = kv_start + split_size;
    if (kv_end > seq_len) kv_end = seq_len;
    if (kv_start >= seq_len) kv_start = kv_end;

    const unsigned int gqa_ratio = num_q_heads / num_kv_heads;
    const unsigned int kv_head = q_head / gqa_ratio;
    const unsigned int vec_offset = lane_id * VEC_BF16;

    const int* my_block_table = block_tables + seq_idx * max_blocks_per_seq;


    const unsigned int* q32 = (const unsigned int*)(Q + (unsigned long long)seq_idx * q_stride
                                                       + (unsigned long long)q_head * head_dim + vec_offset);
    float q_reg[VEC_BF16];
    #pragma unroll
    for (int i = 0; i < VEC_U32; i++) {
        unpack2_pd(q32[i], q_reg[2*i], q_reg[2*i+1]);
    }


    unsigned int local_len = kv_end - kv_start;
    unsigned int chunk_size = (local_len + NUM_WARPS - 1) / NUM_WARPS;
    unsigned int my_start = kv_start + warp_id * chunk_size;
    unsigned int my_end = my_start + chunk_size;
    if (my_end > kv_end) my_end = kv_end;
    if (my_start > kv_end) my_start = kv_end;

    float m = -1e30f;
    float l = 0.0f;
    float o_reg[VEC_BF16];
    #pragma unroll
    for (int i = 0; i < VEC_BF16; i++) o_reg[i] = 0.0f;

    unsigned long long head_stride_kv = (unsigned long long)num_kv_heads * head_dim;
    unsigned long long page_stride = (unsigned long long)block_size * head_stride_kv;


    for (unsigned int pos = my_start; pos < my_end; pos++) {
        unsigned int logical_block = pos / block_size;
        unsigned int block_offset = pos % block_size;
        unsigned int physical_block = (unsigned int)my_block_table[logical_block];

        const unsigned int* k32 = (const unsigned int*)(K_cache
            + (unsigned long long)physical_block * page_stride
            + (unsigned long long)block_offset * head_stride_kv
            + (unsigned long long)kv_head * head_dim + vec_offset);

        float dot = 0.0f;
        #pragma unroll
        for (int i = 0; i < VEC_U32; i++) {
            float k0, k1;
            unpack2_pd(k32[i], k0, k1);
            dot += q_reg[2*i] * k0 + q_reg[2*i+1] * k1;
        }
        #pragma unroll
        for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1)
            dot += __shfl_xor_sync(0xffffffff, dot, offset);

        float score = dot * inv_sqrt_d;
        float m_new = fmaxf(m, score);
        float exp_old = __expf(m - m_new);
        float exp_new = __expf(score - m_new);
        l = l * exp_old + exp_new;

        const unsigned int* v32 = (const unsigned int*)(V_cache
            + (unsigned long long)physical_block * page_stride
            + (unsigned long long)block_offset * head_stride_kv
            + (unsigned long long)kv_head * head_dim + vec_offset);

        #pragma unroll
        for (int i = 0; i < VEC_U32; i++) {
            float v0, v1;
            unpack2_pd(v32[i], v0, v1);
            o_reg[2*i]   = o_reg[2*i]   * exp_old + exp_new * v0;
            o_reg[2*i+1] = o_reg[2*i+1] * exp_old + exp_new * v1;
        }
        m = m_new;
    }


    __shared__ float smem_m[NUM_WARPS];
    __shared__ float smem_l[NUM_WARPS];
    __shared__ float smem_o[NUM_WARPS][HDIM];

    if (lane_id == 0) {
        smem_m[warp_id] = m;
        smem_l[warp_id] = l;
    }
    #pragma unroll
    for (int i = 0; i < VEC_BF16; i++) {
        smem_o[warp_id][vec_offset + i] = o_reg[i];
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
                    smem_o[warp_id][vec_offset + i] =
                        smem_o[warp_id][vec_offset + i] * scale_me +
                        smem_o[other][vec_offset + i] * scale_w;
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
            ws_base[vec_offset + i] = smem_o[0][vec_offset + i];
        }
        if (lane_id == 0) {
            ws_base[head_dim] = smem_m[0];
            ws_base[head_dim + 1] = smem_l[0];
        }
    }
}


extern "C" __global__ void paged_decode_attn_reduce(
    const float* __restrict__ workspace,
    __nv_bfloat16* __restrict__ O,
    const unsigned int num_q_heads,
    const unsigned int head_dim,
    const unsigned int num_splits
) {
    const unsigned int q_head = blockIdx.x;
    const unsigned int seq_idx = blockIdx.y;
    const unsigned int tid = threadIdx.x;
    const unsigned int lane_id = tid % WARP_SIZE;
    const unsigned int vec_offset = lane_id * VEC_BF16;

    if (q_head >= num_q_heads) return;

    unsigned int ws_stride = (head_dim + 2);
    const float* ws_head = workspace + ((unsigned long long)seq_idx * num_q_heads + q_head) * num_splits * ws_stride;

    float m = ws_head[head_dim];
    float l = ws_head[head_dim + 1];
    float o_reg[VEC_BF16];
    #pragma unroll
    for (int i = 0; i < VEC_BF16; i++) {
        o_reg[i] = ws_head[vec_offset + i];
    }

    for (unsigned int s = 1; s < num_splits; s++) {
        const float* ws_s = ws_head + s * ws_stride;
        float ms = ws_s[head_dim];
        float ls = ws_s[head_dim + 1];
        if (ls > 0.0f) {
            float m_new = fmaxf(m, ms);
            float scale_me = __expf(m - m_new);
            float scale_s = __expf(ms - m_new);
            l = l * scale_me + ls * scale_s;
            #pragma unroll
            for (int i = 0; i < VEC_BF16; i++) {
                o_reg[i] = o_reg[i] * scale_me + ws_s[vec_offset + i] * scale_s;
            }
            m = m_new;
        }
    }

    float inv_l = (l > 0.0f) ? (1.0f / l) : 0.0f;
    unsigned int* o32 = (unsigned int*)(O + (unsigned long long)seq_idx * num_q_heads * head_dim
                                          + (unsigned long long)q_head * head_dim + vec_offset);
    #pragma unroll
    for (int i = 0; i < VEC_U32; i++) {
        float v0 = o_reg[2*i]     * inv_l;
        float v1 = o_reg[2*i + 1] * inv_l;
        unsigned int lo = (unsigned int)__bfloat16_as_ushort(__float2bfloat16(v0));
        unsigned int hi = (unsigned int)__bfloat16_as_ushort(__float2bfloat16(v1));
        o32[i] = lo | (hi << 16);
    }
}
