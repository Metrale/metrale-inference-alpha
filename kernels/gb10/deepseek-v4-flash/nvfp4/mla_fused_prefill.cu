// SPDX-License-Identifier: MIT OR Apache-2.0

// 2026-09-25: Fused absorbed-MLA prefill, one (head, query token) per block: Q absorption through
// W_UK, causal online-softmax attention over the latent keys, and V extraction through W_UV.
//
// Owner: gb10 kernels (deepseek-v4-flash).
// Invariants: none beyond the types.

// 2026-09-25: Grid (nq, seq_len), block 256 (ops::mla_fused_prefill). q_full is [N, nq * hd] with each
// head's nope dims first, q_rope [N, nq * rope_dim], kv_latent [N, kv_lora], k_rope [N, rope_dim], v_out
// [N, nq * v_dim]. The W_UK rows [kv_lora, nope] and W_UV rows [v_dim, kv_lora] are indexed by kv head,
// not q head. The shared buffers fix kv_lora <= 512 and kv_lora + rope_dim <= 576. When k_cache_out is
// non-null, head 0's blocks also write K rows [latent | k_rope] and V rows [latent | zeros], each
// kv_lora + rope_dim wide.



#include <cuda_bf16.h>
#include <float.h>

extern "C" __global__ void mla_fused_prefill(

    const __nv_bfloat16* __restrict__ q_full,
    const __nv_bfloat16* __restrict__ q_rope,

    const __nv_bfloat16* __restrict__ kv_latent,
    const __nv_bfloat16* __restrict__ k_rope,

    const __nv_bfloat16* __restrict__ w_uk,
    const __nv_bfloat16* __restrict__ w_uv,

    __nv_bfloat16* __restrict__ v_out,

    __nv_bfloat16* __restrict__ k_cache_out,
    __nv_bfloat16* __restrict__ v_cache_out,

    unsigned int seq_len,
    unsigned int nq,
    unsigned int nope,
    unsigned int rope_dim,
    unsigned int kv_lora,
    unsigned int v_dim,
    unsigned int hd,
    unsigned int num_kv_heads,
    float inv_sqrt_d
) {
    const unsigned int head = blockIdx.x;
    const unsigned int q_pos = blockIdx.y;
    const unsigned int tid = threadIdx.x;

    if (head >= nq || q_pos >= seq_len) return;

    const unsigned int mla_cache_dim = kv_lora + rope_dim;
    const unsigned int gqa_ratio = nq / max(num_kv_heads, 1u);
    const unsigned int kv_head = head / gqa_ratio;







    const __nv_bfloat16* q_nope_ptr = q_full + (unsigned long long)q_pos * nq * hd + head * hd;

    const __nv_bfloat16* w_uk_head = w_uk + (unsigned long long)kv_head * kv_lora * nope;

    __shared__ float smem_q[576];

    for (unsigned int idx = tid; idx < kv_lora; idx += blockDim.x) {

        const __nv_bfloat16* w_row = w_uk_head + (unsigned long long)idx * nope;
        float q_absorbed_val = 0.0f;
        for (unsigned int k = 0; k < nope; k++) {
            q_absorbed_val += __bfloat162float(w_row[k]) * __bfloat162float(q_nope_ptr[k]);
        }
        smem_q[idx] = q_absorbed_val;
    }


    const __nv_bfloat16* q_rope_ptr = q_rope + (unsigned long long)q_pos * nq * rope_dim + head * rope_dim;
    if (tid < rope_dim) {
        smem_q[kv_lora + tid] = __bfloat162float(q_rope_ptr[tid]);
    }
    __syncthreads();






    if (head == 0 && k_cache_out != 0) {


        for (unsigned int idx = tid; idx < kv_lora; idx += blockDim.x) {
            __nv_bfloat16 lat_val = kv_latent[q_pos * kv_lora + idx];
            k_cache_out[q_pos * mla_cache_dim + idx] = lat_val;
            v_cache_out[q_pos * mla_cache_dim + idx] = lat_val;
        }

        for (unsigned int idx = tid + kv_lora; idx < mla_cache_dim; idx += blockDim.x) {
            unsigned int r = idx - kv_lora;
            k_cache_out[q_pos * mla_cache_dim + idx] = (r < rope_dim) ?
                k_rope[q_pos * rope_dim + r] : __float2bfloat16(0.0f);
            v_cache_out[q_pos * mla_cache_dim + idx] = __float2bfloat16(0.0f);
        }
    }








    float m_prev = -FLT_MAX;
    float l_prev = 0.0f;

    float acc_latent[2] = {0.0f, 0.0f};

    unsigned int kv_end = min(q_pos + 1, seq_len);
    for (unsigned int kv_pos = 0; kv_pos < kv_end; kv_pos++) {

        const __nv_bfloat16* kv_lat_row = kv_latent + (unsigned long long)kv_pos * kv_lora;
        const __nv_bfloat16* k_rope_row = k_rope + (unsigned long long)kv_pos * rope_dim;


        float dot = 0.0f;

        for (unsigned int idx = tid; idx < kv_lora; idx += blockDim.x) {
            dot += smem_q[idx] * __bfloat162float(kv_lat_row[idx]);
        }

        if (tid < rope_dim) {
            dot += smem_q[kv_lora + tid] * __bfloat162float(k_rope_row[tid]);
        }


        for (int offset = 16; offset > 0; offset >>= 1) {
            dot += __shfl_down_sync(0xFFFFFFFF, dot, offset);
        }

        __shared__ float smem_dot[8];
        unsigned int warp_id = tid / 32;
        unsigned int lane_id = tid % 32;
        if (lane_id == 0) {
            smem_dot[warp_id] = dot;
        }
        __syncthreads();

        float score;
        if (tid == 0) {
            score = 0.0f;
            for (int w = 0; w < 8; w++) score += smem_dot[w];
            score *= inv_sqrt_d;
            smem_dot[0] = score;
        }
        __syncthreads();
        score = smem_dot[0];


        float m_new = fmaxf(m_prev, score);
        float alpha = expf(m_prev - m_new);
        float p = expf(score - m_new);
        float l_new = alpha * l_prev + p;


        for (unsigned int i = 0; i < 2; i++) {
            unsigned int idx = tid + i * blockDim.x;
            if (idx < kv_lora) {
                acc_latent[i] = alpha * acc_latent[i] + p * __bfloat162float(kv_lat_row[idx]);
            }
        }

        m_prev = m_new;
        l_prev = l_new;
        __syncthreads();
    }


    float inv_l = (l_prev > 0.0f) ? (1.0f / l_prev) : 0.0f;
    for (unsigned int i = 0; i < 2; i++) {
        unsigned int idx = tid + i * blockDim.x;
        if (idx < kv_lora) {
            acc_latent[i] *= inv_l;
        }
    }





    __shared__ float smem_latent[512];
    for (unsigned int i = 0; i < 2; i++) {
        unsigned int idx = tid + i * blockDim.x;
        if (idx < kv_lora) {
            smem_latent[idx] = acc_latent[i];
        }
    }
    __syncthreads();


    const __nv_bfloat16* w_uv_head = w_uv + (unsigned long long)kv_head * v_dim * kv_lora;

    for (unsigned int idx = tid; idx < v_dim; idx += blockDim.x) {

        const __nv_bfloat16* w_row = w_uv_head + (unsigned long long)idx * kv_lora;
        float v_val = 0.0f;
        for (unsigned int l = 0; l < kv_lora; l++) {
            v_val += __bfloat162float(w_row[l]) * smem_latent[l];
        }
        v_out[(unsigned long long)q_pos * nq * v_dim + head * v_dim + idx] = __float2bfloat16(v_val);
    }
}
