// SPDX-License-Identifier: MIT OR Apache-2.0

// 2026-09-25: GQA-packed FP8 paged decode attention: the twin of paged_decode_attn_fp8
// (paged_decode_attn_fp8.cu) that runs one CTA per (kv_head, seq) and computes all PD_GQA
// query heads of that group, so each K and V row is loaded once per group instead of
// once per query head.
//
// Owner: gb10 kernels.
// Invariants:
// - Launch: grid (num_kv_heads, num_seqs, 1) and 256 threads (NUM_WARPS * WARP_SIZE). It
//   takes the same arguments as paged_decode_attn_fp8.
// - Assumes num_q_heads == num_kv_heads * PD_GQA and head_dim == PD_HDIM. The host checks
//   both with gqa_pack_shape_ok (crates/kernels/src/attn_splitk.rs) before choosing it.
// - A sequence with seq_len == 0 returns before any write; its O rows keep their bytes.
// - Per query head it executes the same floating-point operations, in the same order, as
//   paged_decode_attn_fp8: the same warp partition of [window_start, seq_len), the same
//   BC = 4 block loop, dot-product term order, __shfl_xor_sync butterfly, softmax rescale
//   and 8-warp tree merge. Only the K/V loads move, above the per-head loop. With
//   --fmad=false (kernels/gb10/common/KERNEL.toml) nothing is contracted, so the BF16
//   output equals that kernel's byte for byte; gqa_pack_gpu_tests.rs (an ignored GPU
//   test) compares the two.
//
// The host launches it only on the branch that does not take split-K
// (qwen3_attention/decode/run_paged_decode.rs). The split count is derived from
// num_q_heads, so a packed split-K grid would need its own count and merge tree.























#include <cuda_bf16.h>
#include <cuda_fp8.h>

#define WARP_SIZE 32
// 2026-09-25: Named PD_HDIM rather than HDIM, and not under #ifndef: some targets pass
// -DHDIM=128 (the laguna-s-2.1 and minimax-m2-229b KERNEL.toml), which must not resize
// the register arrays below. gqa_pack_shape_ok requires head_dim ==
// DECODE_GQA_PACK_HEAD_DIM (256), so a 128-wide model never selects this kernel.



#define PD_HDIM 256
#define VEC_BF16 (PD_HDIM / WARP_SIZE)
#define VEC_U32 (PD_HDIM / (WARP_SIZE * 2))
#define VEC_U32_FP8 (PD_HDIM / (WARP_SIZE * 4))
#define NUM_WARPS 8
#define BC 4

// 2026-09-25: Query heads per CTA. A compile-time constant because it sizes the q_reg,
// o_reg, m_acc and l_acc arrays. gqa_pack_shape_ok refuses any launch with num_q_heads
// != num_kv_heads * PD_GQA, and cuda_sources_declare_the_pack_width_rust_dispatches_on
// (crates/kernels/src/attn_splitk_tests.rs) fails if PD_GQA differs from
// DECODE_GQA_PACK_WIDTH.


#define PD_GQA 6

// 2026-09-25: Copies of paged_decode_attn_fp8.cu's unpack2_bf16 and unpack4_fp8_raw: each
// .cu file is compiled on its own (crates/kernels/build.rs), so device helpers are not
// shared. Output identity with that kernel needs the bodies to stay the same, and
// gqa_kernels_copy_the_unpack_helpers_verbatim (crates/kernels/src/attn_splitk_tests.rs)
// fails when either copy differs from its original.





__device__ __forceinline__ void unpack2_bf16(unsigned int packed, float& v0, float& v1) {
    v0 = __bfloat162float(__ushort_as_bfloat16((unsigned short)(packed & 0xFFFF)));
    v1 = __bfloat162float(__ushort_as_bfloat16((unsigned short)(packed >> 16)));
}

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

// 2026-09-25: __launch_bounds__(256, 1): each thread holds q_reg and o_reg, PD_GQA *
// VEC_BF16 floats each, and a minimum of one block per SM lets the compiler give them
// the full per-thread register budget.


extern "C" __global__ void __launch_bounds__(NUM_WARPS* WARP_SIZE, 1)
    paged_decode_attn_fp8_gqa(
        const __nv_bfloat16* __restrict__ Q,
        const __nv_fp8_storage_t* __restrict__ K_cache,
        const __nv_fp8_storage_t* __restrict__ V_cache,
        __nv_bfloat16* __restrict__ O,
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
        const unsigned int q_stride,
        const unsigned long long cache_stride,
        const unsigned int sliding_window
    ) {
    const unsigned int kv_head = blockIdx.x;
    const unsigned int seq_idx = blockIdx.y;
    const unsigned int tid = threadIdx.x;
    const unsigned int warp_id = tid / WARP_SIZE;
    const unsigned int lane_id = tid % WARP_SIZE;

    if (kv_head >= num_kv_heads) return;

    const unsigned int seq_len = (unsigned int)seq_lens[seq_idx];
    if (seq_len == 0) return;
    const unsigned int window_start =
        (sliding_window > 0 && seq_len > sliding_window) ? (seq_len - sliding_window) : 0u;

    // 2026-09-25: Inverse of paged_decode_attn_fp8's kv_head = q_head / gqa_ratio. Exact
    // because the host refuses any shape with num_q_heads != num_kv_heads * PD_GQA.

    const unsigned int q_head_base = kv_head * PD_GQA;
    const unsigned int vec_offset_bf16 = lane_id * VEC_BF16;
    const unsigned int vec_offset_fp8 = lane_id * VEC_BF16;

    const int* my_block_table = block_tables + seq_idx * max_blocks_per_seq;

    // 2026-09-25: The group's PD_GQA query vectors, held in registers so that each K and V
    // row loaded below serves all of them.
    float q_reg[PD_GQA][VEC_BF16];
    #pragma unroll
    for (int h = 0; h < PD_GQA; h++) {
        const unsigned int* q32 = (const unsigned int*)(Q
            + (unsigned long long)seq_idx * q_stride
            + (unsigned long long)(q_head_base + h) * head_dim + vec_offset_bf16);
        #pragma unroll
        for (int i = 0; i < VEC_U32; i++) {
            unpack2_bf16(q32[i], q_reg[h][2 * i], q_reg[h][2 * i + 1]);
        }
    }

    // 2026-09-25: The warp partition of paged_decode_attn_fp8, so each head's per-warp states
    // and merge match that kernel's.
    const unsigned int attended = seq_len - window_start;
    unsigned int chunk_size = (attended + NUM_WARPS - 1) / NUM_WARPS;
    unsigned int my_start = window_start + warp_id * chunk_size;
    unsigned int my_end = my_start + chunk_size;
    if (my_end > seq_len) my_end = seq_len;
    if (my_start > seq_len) my_start = seq_len;

    const float score_scale = inv_sqrt_d * k_scale;

    float m_acc[PD_GQA];
    float l_acc[PD_GQA];
    float o_reg[PD_GQA][VEC_BF16];
    #pragma unroll
    for (int h = 0; h < PD_GQA; h++) {
        m_acc[h] = -1e30f;
        l_acc[h] = 0.0f;
        #pragma unroll
        for (int i = 0; i < VEC_BF16; i++) o_reg[h][i] = 0.0f;
    }

    unsigned long long head_stride_kv = (unsigned long long)num_kv_heads * head_dim;

    unsigned int pos = my_start;
    while (pos < my_end) {
        unsigned int logical_block = pos / block_size;
        unsigned int block_offset = pos % block_size;
        unsigned int remaining_in_block = block_size - block_offset;
        unsigned int remaining_total = my_end - pos;
        unsigned int batch_count =
            remaining_in_block < remaining_total ? remaining_in_block : remaining_total;

        unsigned int physical_block = (unsigned int)my_block_table[logical_block];
        const __nv_fp8_storage_t* k_block_base = K_cache
            + (unsigned long long)physical_block * cache_stride
            + (unsigned long long)block_offset * head_stride_kv
            + (unsigned long long)kv_head * head_dim;
        const __nv_fp8_storage_t* v_block_base = V_cache
            + (unsigned long long)physical_block * cache_stride
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
                for (int i = 0; i < VEC_U32_FP8; i++) k_packed[b][i] = k32[i];
            }
            unsigned int v_packed[BC][VEC_U32_FP8];
            #pragma unroll
            for (int b = 0; b < BC; b++) {
                const unsigned int* v32 = (const unsigned int*)(v_block_base
                    + (unsigned long long)(processed + b) * head_stride_kv + vec_offset_fp8);
                #pragma unroll
                for (int i = 0; i < VEC_U32_FP8; i++) v_packed[b][i] = v32[i];
            }

            #pragma unroll
            for (int h = 0; h < PD_GQA; h++) {
                float scores[BC];
                #pragma unroll
                for (int b = 0; b < BC; b++) {
                    float dot = 0.0f;
                    #pragma unroll
                    for (int i = 0; i < VEC_U32_FP8; i++) {
                        float k0, k1, k2, k3;
                        unpack4_fp8_raw(k_packed[b][i], k0, k1, k2, k3);
                        dot += q_reg[h][4 * i] * k0 + q_reg[h][4 * i + 1] * k1
                             + q_reg[h][4 * i + 2] * k2 + q_reg[h][4 * i + 3] * k3;
                    }
                    #pragma unroll
                    for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1)
                        dot += __shfl_xor_sync(0xffffffff, dot, offset);
                    scores[b] = dot * score_scale;
                }

                float m_new = m_acc[h];
                #pragma unroll
                for (int b = 0; b < BC; b++) m_new = fmaxf(m_new, scores[b]);

                float exp_old = __expf(m_acc[h] - m_new);
                #pragma unroll
                for (int i = 0; i < VEC_BF16; i++) o_reg[h][i] *= exp_old;
                l_acc[h] *= exp_old;

                float exp_factors[BC];
                #pragma unroll
                for (int b = 0; b < BC; b++) {
                    exp_factors[b] = __expf(scores[b] - m_new);
                    l_acc[h] += exp_factors[b];
                }
                m_acc[h] = m_new;

                #pragma unroll
                for (int b = 0; b < BC; b++) {
                    float ef = exp_factors[b];
                    #pragma unroll
                    for (int i = 0; i < VEC_U32_FP8; i++) {
                        float v0, v1, v2, v3;
                        unpack4_fp8_raw(v_packed[b][i], v0, v1, v2, v3);
                        o_reg[h][4 * i]     += ef * v0;
                        o_reg[h][4 * i + 1] += ef * v1;
                        o_reg[h][4 * i + 2] += ef * v2;
                        o_reg[h][4 * i + 3] += ef * v3;
                    }
                }
            }
        }


        for (; processed < batch_count; processed++) {
            const unsigned int* k32 = (const unsigned int*)(k_block_base
                + (unsigned long long)processed * head_stride_kv + vec_offset_fp8);
            unsigned int k_one[VEC_U32_FP8];
            #pragma unroll
            for (int i = 0; i < VEC_U32_FP8; i++) k_one[i] = k32[i];
            const unsigned int* v32 = (const unsigned int*)(v_block_base
                + (unsigned long long)processed * head_stride_kv + vec_offset_fp8);
            unsigned int v_one[VEC_U32_FP8];
            #pragma unroll
            for (int i = 0; i < VEC_U32_FP8; i++) v_one[i] = v32[i];

            #pragma unroll
            for (int h = 0; h < PD_GQA; h++) {
                float dot = 0.0f;
                #pragma unroll
                for (int i = 0; i < VEC_U32_FP8; i++) {
                    float k0, k1, k2, k3;
                    unpack4_fp8_raw(k_one[i], k0, k1, k2, k3);
                    dot += q_reg[h][4 * i] * k0 + q_reg[h][4 * i + 1] * k1
                         + q_reg[h][4 * i + 2] * k2 + q_reg[h][4 * i + 3] * k3;
                }
                #pragma unroll
                for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1)
                    dot += __shfl_xor_sync(0xffffffff, dot, offset);

                float score = dot * score_scale;
                float m_new = fmaxf(m_acc[h], score);
                float exp_old = __expf(m_acc[h] - m_new);
                float exp_new = __expf(score - m_new);
                l_acc[h] = l_acc[h] * exp_old + exp_new;

                #pragma unroll
                for (int i = 0; i < VEC_U32_FP8; i++) {
                    float v0, v1, v2, v3;
                    unpack4_fp8_raw(v_one[i], v0, v1, v2, v3);
                    o_reg[h][4 * i]     = o_reg[h][4 * i]     * exp_old + exp_new * v0;
                    o_reg[h][4 * i + 1] = o_reg[h][4 * i + 1] * exp_old + exp_new * v1;
                    o_reg[h][4 * i + 2] = o_reg[h][4 * i + 2] * exp_old + exp_new * v2;
                    o_reg[h][4 * i + 3] = o_reg[h][4 * i + 3] * exp_old + exp_new * v3;
                }
                m_acc[h] = m_new;
            }
        }

        pos += batch_count;
    }

    // 2026-09-25: One head at a time through one shared buffer: PD_GQA copies of smem_o
    // (8 KB each) would be 48 KB, which with smem_m and smem_l exceeds the 48 KB static
    // shared-memory limit. Each head runs the tree merge of paged_decode_attn_fp8.




    __shared__ float smem_m[NUM_WARPS];
    __shared__ float smem_l[NUM_WARPS];
    __shared__ float smem_o[NUM_WARPS][PD_HDIM];

    #pragma unroll
    for (int h = 0; h < PD_GQA; h++) {
        // 2026-09-25: The previous head's merge, and warp 0's read of smem_o[0], must finish
        // before this head overwrites the buffer. At h == 0 the barrier is not needed.
        __syncthreads();

        if (lane_id == 0) {
            smem_m[warp_id] = m_acc[h];
            smem_l[warp_id] = l_acc[h];
        }
        // 2026-09-25: v_scale is applied once per output element, as in paged_decode_attn_fp8.
        #pragma unroll
        for (int i = 0; i < VEC_BF16; i++) {
            smem_o[warp_id][vec_offset_bf16 + i] = o_reg[h][i] * v_scale;
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
            unsigned int* o32 = (unsigned int*)(O
                + (unsigned long long)seq_idx * num_q_heads * head_dim
                + (unsigned long long)(q_head_base + h) * head_dim + vec_offset_bf16);
            #pragma unroll
            for (int i = 0; i < VEC_U32; i++) {
                float v0 = smem_o[0][vec_offset_bf16 + 2 * i]     * inv_l;
                float v1 = smem_o[0][vec_offset_bf16 + 2 * i + 1] * inv_l;
                unsigned int lo = (unsigned int)__bfloat16_as_ushort(__float2bfloat16(v0));
                unsigned int hi = (unsigned int)__bfloat16_as_ushort(__float2bfloat16(v1));
                o32[i] = lo | (hi << 16);
            }
        }
    }
}
