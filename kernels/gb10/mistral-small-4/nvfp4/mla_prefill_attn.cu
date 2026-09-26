// SPDX-License-Identifier: MIT OR Apache-2.0
// forked-from: kernels/gb10/deepseek-v4-flash/nvfp4/mla_prefill_attn.cu (2026-09-24; 17 of 143 lines differ, see kernels/FORKS.md)

// 2026-09-25: Absorbed-MLA prefill attention for head dim 320: scalar BF16 dot products with FP32
// accumulation, no tensor-core MMA.
//
// Owner: gb10 kernels (mistral-small-4).
// Invariants: none beyond the types.

// 2026-09-25: Q and O are [batch, seq_len, num_q_heads, head_dim], K and V [batch, seq_len, num_kv_heads,
// head_dim]; q head h reads kv head h / (num_q_heads / num_kv_heads). Each 16-lane group owns one query
// row and 20 dims per lane, so head_dim is at most 320. causal != 0 limits row t to keys 0..=t.
// Grid (num_q_heads, ceil(seq_len / 16), batch), block 256 (ops::mla_prefill_attention_320).



#include <cuda_bf16.h>
#include <float.h>

#define MLA_HDIM 320
#define MLA_BR 16
#define MLA_BC 16

extern "C" __global__ void mla_prefill_attn_320(
    const __nv_bfloat16* __restrict__ Q,
    const __nv_bfloat16* __restrict__ K,
    const __nv_bfloat16* __restrict__ V,
    __nv_bfloat16* __restrict__ O,
    unsigned int seq_len,
    unsigned int num_q_heads,
    unsigned int num_kv_heads,
    unsigned int head_dim,
    float inv_sqrt_d,
    unsigned int causal
) {
    const unsigned int q_head = blockIdx.x;
    const unsigned int q_block = blockIdx.y;
    const unsigned int batch = blockIdx.z;
    const unsigned int tid = threadIdx.x;

    if (q_head >= num_q_heads) return;

    const unsigned int q_start = q_block * MLA_BR;
    if (q_start >= seq_len) return;
    const unsigned int q_end = min(q_start + MLA_BR, seq_len);

    const unsigned int gqa_ratio = num_q_heads / max(num_kv_heads, 1u);
    const unsigned int kv_head = q_head / gqa_ratio;

    const unsigned int q_stride = num_q_heads * head_dim;
    const unsigned int kv_stride = num_kv_heads * head_dim;

    const __nv_bfloat16* Q_base = Q + (unsigned long long)batch * seq_len * q_stride;
    const __nv_bfloat16* K_base = K + (unsigned long long)batch * seq_len * kv_stride;
    const __nv_bfloat16* V_base = V + (unsigned long long)batch * seq_len * kv_stride;
    __nv_bfloat16* O_base = O + (unsigned long long)batch * seq_len * q_stride;

    // 2026-09-25: 16 lanes per query row, two rows per warp. The reductions use the full-warp
    // mask with offsets 8, 4, 2, 1, so lanes 0 and 16 each end with their own row's sum.



    const unsigned int q_row = tid / 16;
    const unsigned int lane = tid % 16;
    const unsigned int warp_lane = tid % 32;

    if (q_row >= (q_end - q_start)) return;

    const unsigned int q_pos = q_start + q_row;
    const __nv_bfloat16* Q_row = Q_base + (unsigned long long)q_pos * q_stride + q_head * head_dim;


    float m_prev = -FLT_MAX;
    float l_prev = 0.0f;
    float acc_o[20];
    for (int i = 0; i < 20; i++) acc_o[i] = 0.0f;


    unsigned int kv_end = causal ? min(q_pos + 1, seq_len) : seq_len;
    for (unsigned int kv_start = 0; kv_start < kv_end; kv_start += MLA_BC) {
        unsigned int kv_block_end = min(kv_start + MLA_BC, kv_end);

        for (unsigned int kv_pos = kv_start; kv_pos < kv_block_end; kv_pos++) {

            const __nv_bfloat16* K_row = K_base + (unsigned long long)kv_pos * kv_stride + kv_head * head_dim;


            float dot = 0.0f;
            for (unsigned int d = lane * 20; d < min((lane + 1) * 20, head_dim); d++) {
                float q_val = __bfloat162float(Q_row[d]);
                float k_val = __bfloat162float(K_row[d]);
                dot += q_val * k_val;
            }



            for (int offset = 8; offset > 0; offset >>= 1) {
                dot += __shfl_down_sync(0xFFFFFFFF, dot, offset);
            }



            float score = dot * inv_sqrt_d;


            if (causal && kv_pos > q_pos) score = -FLT_MAX;


            // 2026-09-25: Broadcast each row's score from its group leader (lane 0 or 16).
            score = __shfl_sync(0xFFFFFFFF, score, (warp_lane / 16) * 16);


            float m_new = fmaxf(m_prev, score);
            float alpha = expf(m_prev - m_new);
            float p = expf(score - m_new);
            float l_new = alpha * l_prev + p;


            const __nv_bfloat16* V_row = V_base + (unsigned long long)kv_pos * kv_stride + kv_head * head_dim;
            for (int i = 0; i < 20; i++) {
                unsigned int d = lane * 20 + i;
                if (d < head_dim) {
                    float v_val = __bfloat162float(V_row[d]);
                    acc_o[i] = alpha * acc_o[i] + p * v_val;
                }
            }
            m_prev = m_new;
            l_prev = l_new;
        }
    }


    float inv_l = (l_prev > 0.0f) ? (1.0f / l_prev) : 0.0f;
    __nv_bfloat16* O_row = O_base + (unsigned long long)q_pos * q_stride + q_head * head_dim;
    for (int i = 0; i < 20; i++) {
        unsigned int d = lane * 20 + i;
        if (d < head_dim) {
            O_row[d] = __float2bfloat16(acc_o[i] * inv_l);
        }
    }
}
