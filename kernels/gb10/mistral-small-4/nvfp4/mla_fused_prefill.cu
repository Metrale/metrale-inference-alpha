// SPDX-License-Identifier: MIT OR Apache-2.0
// forked-from: kernels/gb10/deepseek-v4-flash/nvfp4/mla_fused_prefill.cu (2026-09-24; 79 of 213 lines differ, see kernels/FORKS.md)

// 2026-09-25: MLA prefill in one kernel for the mistral-small-4 tree. One 256-thread block per (q head,
// query token) absorbs Q (q_nope . w_uk), runs causal online-softmax attention over [kv_latent | k_rope],
// and extracts V (attention-weighted latent . w_uv).
//
// Owner: gb10 kernels (mistral-small-4).
// Invariants: none beyond the types. The fixed shared arrays and one thread per output need
// kv_lora <= 256, kv_lora + rope_dim <= 320 and v_dim <= 256.
//
// Layouts: q_full [N, nq * hd] with q_nope first in each head's hd; q_rope [N, nq * rope_dim];
// kv_latent [N, kv_lora]; k_rope [N, rope_dim]; w_uk [nq, kv_lora, nope] and w_uv [nq, v_dim, kv_lora],
// indexed by q head; v_out [N, nq * v_dim].



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
    float inv_sqrt_d
) {
    const unsigned int head = blockIdx.x;
    const unsigned int q_pos = blockIdx.y;
    const unsigned int tid = threadIdx.x;

    if (head >= nq || q_pos >= seq_len) return;

    const unsigned int mla_cache_dim = kv_lora + rope_dim;








    const __nv_bfloat16* q_nope_ptr = q_full + (unsigned long long)q_pos * nq * hd + head * hd;

    const __nv_bfloat16* w_uk_head = w_uk + (unsigned long long)head * kv_lora * nope;

    float q_absorbed_val = 0.0f;
    if (tid < kv_lora) {

        const __nv_bfloat16* w_row = w_uk_head + (unsigned long long)tid * nope;
        for (unsigned int k = 0; k < nope; k++) {
            q_absorbed_val += __bfloat162float(w_row[k]) * __bfloat162float(q_nope_ptr[k]);
        }
    }


    __shared__ float smem_q[320];
    if (tid < kv_lora) {
        smem_q[tid] = q_absorbed_val;
    }


    const __nv_bfloat16* q_rope_ptr = q_rope + (unsigned long long)q_pos * nq * rope_dim + head * rope_dim;
    if (tid < rope_dim) {
        smem_q[kv_lora + tid] = __bfloat162float(q_rope_ptr[tid]);
    }
    __syncthreads();




// 2026-09-25: Block (0, t) writes cache row t when k_cache_out is non-null: K = [kv_latent | k_rope],
// V = [kv_latent | zeros]. Only positions below blockDim.x (256) are written.
    if (head == 0 && k_cache_out != 0) {

        if (tid < kv_lora) {
            __nv_bfloat16 lat_val = kv_latent[q_pos * kv_lora + tid];
            k_cache_out[q_pos * mla_cache_dim + tid] = lat_val;
            v_cache_out[q_pos * mla_cache_dim + tid] = lat_val;
        } else if (tid < mla_cache_dim) {
            unsigned int r = tid - kv_lora;
            k_cache_out[q_pos * mla_cache_dim + tid] = (r < rope_dim) ?
                k_rope[q_pos * rope_dim + r] : __float2bfloat16(0.0f);
            v_cache_out[q_pos * mla_cache_dim + tid] = __float2bfloat16(0.0f);
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

        if (tid < kv_lora) {
            dot += smem_q[tid] * __bfloat162float(kv_lat_row[tid]);
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


        if (tid < kv_lora) {
            acc_latent[0] = alpha * acc_latent[0] + p * __bfloat162float(kv_lat_row[tid]);
        }

        m_prev = m_new;
        l_prev = l_new;
        __syncthreads();
    }


    float inv_l = (l_prev > 0.0f) ? (1.0f / l_prev) : 0.0f;
    if (tid < kv_lora) {
        acc_latent[0] *= inv_l;
    }





    __shared__ float smem_latent[256];
    if (tid < kv_lora) {
        smem_latent[tid] = acc_latent[0];
    }
    __syncthreads();


    const __nv_bfloat16* w_uv_head = w_uv + (unsigned long long)head * v_dim * kv_lora;

    if (tid < v_dim) {

        const __nv_bfloat16* w_row = w_uv_head + (unsigned long long)tid * kv_lora;
        float v_val = 0.0f;
        for (unsigned int l = 0; l < kv_lora; l++) {
            v_val += __bfloat162float(w_row[l]) * smem_latent[l];
        }
        v_out[(unsigned long long)q_pos * nq * v_dim + head * v_dim + tid] = __float2bfloat16(v_val);
    }
}
