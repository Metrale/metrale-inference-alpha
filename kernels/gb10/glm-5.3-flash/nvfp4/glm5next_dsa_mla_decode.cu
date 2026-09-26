// SPDX-License-Identifier: MIT OR Apache-2.0

// 2026-09-25: GLM-5.3-Flash selected-index MLA decode over the paged FP8 latent cache
// (module `glm5next_dsa_mla_decode`, entry `glm5next_dsa_mla_decode_fp8`).
// Owner: gb10 kernels (glm-5.3-flash).
// Invariants: none beyond the launch contract below.
//
// NoPE: qk_rope_head_dim is 0, so a cache token is the latent alone and there is no rope
// arm. The MLA decode kernels in deepseek-v4-flash/nvfp4/ assume a 64-dim rope tail
// (`ROPE_DIM 64`), so GLM does not use them.
//
// One block per (q_head, row), 8 warps (blockDim 256). The warps split the row's selection
// `sel_indices[row, 0..sel_width)` and gather each selected token through the block table,
// one token at a time, with a per-warp online softmax and a cross-warp merge.
//
// Launch contract:
//   * kv_lora_dim == GLM_KV_LORA_DIM (512).
//   * A selection entry of -1, or any index outside [0, seq_len), is skipped.
//   * Nothing deduplicates: an index listed twice is attended twice.
//   * A row with no valid index writes zeros; a row whose seq_len is 0 writes nothing
//     to O.







































#include <cuda_bf16.h>
#include <cuda_fp8.h>

#define WARP_SIZE 32
#define VEC_BF16 16
#define VEC_U32  8
#define NUM_WARPS 8

// 2026-09-25: The latent width, fixed at compile time: 32 lanes * VEC_BF16 (16) covers
// exactly 512. The host refuses a kv_lora_rank that differs (Glm5NextDsaConfig::validate,
// KERNEL_KV_LORA_DIM).
#define GLM_KV_LORA_DIM 512

#define DSA_INVALID (-1)

__device__ __forceinline__ float fp8e4m3_to_f32(__nv_fp8_storage_t b) {
    return __half2float(__nv_cvt_fp8_to_halfraw(b, __NV_E4M3));
}


// 2026-09-25: Decode this lane's VEC_BF16 FP8 bytes of one cache token, times `scale`.
__device__ __forceinline__ void load_kv_fp8(
    const unsigned char* __restrict__ token_base,
    unsigned int lane_offset,
    float scale,
    float* __restrict__ out
) {
    const unsigned char* p = token_base + lane_offset;
    #pragma unroll
    for (int i = 0; i < VEC_BF16; i++)
        out[i] = fp8e4m3_to_f32((__nv_fp8_storage_t)p[i]) * scale;
}

extern "C" __global__ void glm5next_dsa_mla_decode_fp8(
    const __nv_bfloat16* __restrict__ Q,           // 2026-09-25: [rows, num_q_heads, kv_lora_dim] bf16
    const unsigned char* __restrict__ K_cache,     // 2026-09-25: FP8 latent cache
    const unsigned char* __restrict__ V_cache,     // 2026-09-25: the caller passes the K buffer
    __nv_bfloat16* __restrict__ O,                 // 2026-09-25: [rows, num_q_heads, kv_lora_dim] bf16
    const int* __restrict__ block_tables,          // 2026-09-25: row r at r * max_blocks_per_seq
    const int* __restrict__ seq_lens,              // 2026-09-25: [rows]
    const int* __restrict__ sel_indices,           // 2026-09-25: [rows, sel_width] i32, -1 = unused
    const unsigned int sel_width,
    const unsigned int max_blocks_per_seq,
    const unsigned int num_q_heads,
    const unsigned int num_kv_heads,
    const unsigned int kv_lora_dim,                // 2026-09-25: latent width; a token is num_kv_heads * kv_lora_dim bytes
    const unsigned int block_size,
    const float inv_sqrt_d,
    const float k_scale,
    const float v_scale,
    const unsigned long long cache_stride_bytes
) {
    const unsigned int q_head  = blockIdx.x;
    const unsigned int seq_idx = blockIdx.y;
    const unsigned int tid     = threadIdx.x;
    const unsigned int warp_id = tid / WARP_SIZE;
    const unsigned int lane_id = tid % WARP_SIZE;

    if (q_head >= num_q_heads) return;

    const unsigned int seq_len = (unsigned int)seq_lens[seq_idx];
    if (seq_len == 0) return;

    const unsigned int lane_offset = lane_id * VEC_BF16;

    // 2026-09-25: No rope term in the token stride (NoPE).
    const unsigned int token_stride = num_kv_heads * kv_lora_dim;

    const int* my_block_table = block_tables + (size_t)seq_idx * max_blocks_per_seq;
    const int* my_sel         = sel_indices  + (size_t)seq_idx * sel_width;

    // 2026-09-25: Q and O are [rows, num_q_heads, kv_lora_dim], and blockIdx.y is the row.
    // The layer passes all verify rows of a step as rows of one launch
    // (glm5next_dsa/layer.rs `attend_rows`).

    const unsigned long long row_off = (unsigned long long)seq_idx * num_q_heads * kv_lora_dim;


    const unsigned int* q32 =
        (const unsigned int*)(Q + row_off + (unsigned long long)q_head * kv_lora_dim + lane_offset);
    float q_reg[VEC_BF16];
    #pragma unroll
    for (int i = 0; i < VEC_U32; i++) {
        unsigned int v = q32[i];
        q_reg[2*i]     = __bfloat162float(__ushort_as_bfloat16((unsigned short)(v & 0xFFFF)));
        q_reg[2*i + 1] = __bfloat162float(__ushort_as_bfloat16((unsigned short)(v >> 16)));
    }

    // 2026-09-25: Each warp takes a contiguous chunk of the selection row. A warp with no
    // valid index ends with l == 0 and contributes nothing to the merge.
    const unsigned int chunk = (sel_width + NUM_WARPS - 1) / NUM_WARPS;
    unsigned int j     = warp_id * chunk;
    unsigned int j_end = j + chunk;
    if (j_end > sel_width) j_end = sel_width;

    float m = -1e30f;
    float l = 0.0f;
    float o_reg[VEC_BF16];
    #pragma unroll
    for (int i = 0; i < VEC_BF16; i++) o_reg[i] = 0.0f;

    for (; j < j_end; j++) {
        const int t = my_sel[j];
        // 2026-09-25: Skipped rather than clamped: a clamped index would attend a real but
        // wrong token.
        if (t == DSA_INVALID || t < 0 || (unsigned int)t >= seq_len) continue;

        const unsigned int logical_block = (unsigned int)t / block_size;
        const unsigned int p             = (unsigned int)t % block_size;
        const unsigned int physical_block = (unsigned int)my_block_table[logical_block];

        const unsigned char* k_tok =
            K_cache + (unsigned long long)physical_block * cache_stride_bytes + p * token_stride;
        const unsigned char* v_tok =
            V_cache + (unsigned long long)physical_block * cache_stride_bytes + p * token_stride;

        float k_tmp[VEC_BF16];
        load_kv_fp8(k_tok, lane_offset, k_scale, k_tmp);

        float dot = 0.0f;
        #pragma unroll
        for (int i = 0; i < VEC_BF16; i++)
            if (lane_offset + i < kv_lora_dim) dot += q_reg[i] * k_tmp[i];
        #pragma unroll
        for (int off = WARP_SIZE / 2; off > 0; off >>= 1)
            dot += __shfl_xor_sync(0xffffffff, dot, off);

        const float score   = dot * inv_sqrt_d;
        const float m_new   = fmaxf(m, score);
        const float exp_old = __expf(m - m_new);
        const float exp_new = __expf(score - m_new);
        l = l * exp_old + exp_new;

        // 2026-09-25: Absorbed MLA: the layer passes one pool as both K and V and one scale
        // as both scales (glm5next_dsa/layer.rs `attend_rows`), so V is the K values just
        // decoded, bit for bit. The copy saves the second load; `__restrict__` on both
        // pointers stops the compiler from merging the loads itself. Distinct K/V buffers or
        // scales take the second load. Both operands are kernel arguments, so the branch is
        // uniform.











        const bool same_kv = (K_cache == V_cache) && (k_scale == v_scale);
        float v_tmp[VEC_BF16];
        if (same_kv) {
            #pragma unroll
            for (int i = 0; i < VEC_BF16; i++) v_tmp[i] = k_tmp[i];
        } else {
            load_kv_fp8(v_tok, lane_offset, v_scale, v_tmp);
        }

        #pragma unroll
        for (int i = 0; i < VEC_BF16; i++)
            o_reg[i] = o_reg[i] * exp_old + exp_new * v_tmp[i];
        m = m_new;
    }

    // 2026-09-25: Cross-warp merge of (m, l, o); no attention-sink term.
    __shared__ float smem_m[NUM_WARPS];
    __shared__ float smem_l[NUM_WARPS];
    __shared__ float smem_o[NUM_WARPS][GLM_KV_LORA_DIM];

    if (lane_id == 0) {
        smem_m[warp_id] = m;
        smem_l[warp_id] = l;
    }
    #pragma unroll
    for (int i = 0; i < VEC_BF16; i++)
        if (lane_offset + i < GLM_KV_LORA_DIM) smem_o[warp_id][lane_offset + i] = o_reg[i];
    __syncthreads();

    #pragma unroll
    for (int stride = NUM_WARPS / 2; stride > 0; stride >>= 1) {
        if (warp_id < (unsigned int)stride) {
            const unsigned int other = warp_id + stride;
            const float lw = smem_l[other];
            if (lw > 0.0f) {
                const float mw     = smem_m[other];
                const float my_m   = smem_m[warp_id];
                const float my_l   = smem_l[warp_id];
                const float m_new  = fmaxf(my_m, mw);
                const float sc_me  = __expf(my_m - m_new);
                const float sc_w   = __expf(mw - m_new);
                smem_l[warp_id] = my_l * sc_me + lw * sc_w;
                smem_m[warp_id] = m_new;
                #pragma unroll
                for (int i = 0; i < GLM_KV_LORA_DIM; i++)
                    smem_o[warp_id][i] = smem_o[warp_id][i] * sc_me + smem_o[other][i] * sc_w;
            }
        }
        __syncthreads();
    }

    if (warp_id == 0) {
        const float final_l = smem_l[0];
        // 2026-09-25: final_l == 0 means no valid index in the row: write zeros rather than
        // divide by zero.
        const float inv_l = (final_l > 0.0f) ? (1.0f / final_l) : 0.0f;
        unsigned int* o32 =
            (unsigned int*)(O + row_off + (unsigned long long)q_head * kv_lora_dim + lane_offset);
        #pragma unroll
        for (int i = 0; i < VEC_U32; i++) {
            const float v0 = smem_o[0][lane_offset + 2*i]     * inv_l;
            const float v1 = smem_o[0][lane_offset + 2*i + 1] * inv_l;
            const unsigned int lo = (unsigned int)__bfloat16_as_ushort(__float2bfloat16(v0));
            const unsigned int hi = (unsigned int)__bfloat16_as_ushort(__float2bfloat16(v1));
            o32[i] = lo | (hi << 16);
        }
    }
}
