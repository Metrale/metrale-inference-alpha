// SPDX-License-Identifier: MIT OR Apache-2.0
// forked-from: kernels/gb10/gemma-4-31b/nvfp4/attn_prefill_512.cu (2026-09-24; 38 of 141 lines differ, see kernels/FORKS.md)

// 2026-09-25: Scalar prefill attention at head_dim 512 with an optional sliding window and per-head sink, run
// by `ops::prefill_attention_512_sink` for DeepSeek-V4; a CTA takes BR query rows of one (q_head, batch), 8 threads a row.
// Owner: gb10 kernels (deepseek-v4-flash). Grid: (num_q_heads, ceil(seq_len/BR), batch)  Block: (128, 1, 1).
// Invariants: assumes 448 < head_dim <= 512: every lane must reach the full-warp shuffles, and 8 x 64 dims cover a row.
#include <cuda_bf16.h>

#define BR 16
#define HDIM 512

extern "C" __global__ void attn_prefill_512(
    const __nv_bfloat16* __restrict__ Q,
    const __nv_bfloat16* __restrict__ K,
    const __nv_bfloat16* __restrict__ V,
    __nv_bfloat16* __restrict__ O,
    const unsigned int seq_len,
    const unsigned int num_q_heads,
    const unsigned int num_kv_heads,
    const unsigned int head_dim,
    const float inv_sqrt_d,
    const unsigned int causal,
    const unsigned int sliding_window,
    const float* __restrict__ sinks  // 2026-09-25: [num_q_heads] sink logits, or nullptr for none
) {
    const unsigned int q_head = blockIdx.x;
    const unsigned int q_block = blockIdx.y;
    const unsigned int batch = blockIdx.z;
    const unsigned int tid = threadIdx.x;

    if (q_head >= num_q_heads) return;

    const unsigned int q_row = q_block * BR + (tid / 8);
    const bool valid = q_row < seq_len;

    const unsigned int dim_lane = tid % 8;
    const unsigned int dim_start = dim_lane * 64;
    if (dim_start >= head_dim) return;
    const unsigned int dim_end = min(dim_start + 64, head_dim);

    const unsigned int gqa_ratio = num_q_heads / num_kv_heads;
    const unsigned int kv_head = q_head / gqa_ratio;

    const unsigned int q_stride = num_q_heads * head_dim;
    const unsigned int kv_stride = num_kv_heads * head_dim;

    const __nv_bfloat16* Q_row = Q + (unsigned long long)batch * seq_len * q_stride
                                    + (unsigned long long)q_row * q_stride
                                    + (unsigned long long)q_head * head_dim;

    // 2026-09-25: Each of a row's 8 lanes takes a partial dot product over its 64 dims; the shuffles below sum them.



    unsigned int kv_len = seq_len;
    if (causal) kv_len = min(kv_len, q_row + 1);

    // 2026-09-25: Attend only the last `sliding_window` of the `kv_len` keys; 0 attends all of them.

    unsigned int kv_start = 0;
    if (sliding_window > 0u && kv_len > sliding_window) kv_start = kv_len - sliding_window;

    // 2026-09-25: Online softmax: m is the running max score, l the running sum of exp(score - m).
    float m = -1e30f;
    float l = 0.0f;
    float o_acc[64];
    for (unsigned int d = 0; d < 64 && dim_start + d < head_dim; d++) {
        o_acc[d] = 0.0f;
    }

    for (unsigned int kv_pos = kv_start; kv_pos < kv_len; kv_pos++) {
        const __nv_bfloat16* K_row = K + (unsigned long long)batch * seq_len * kv_stride
                                        + (unsigned long long)kv_pos * kv_stride
                                        + (unsigned long long)kv_head * head_dim;


        float dot = 0.0f;
        if (valid) {
            for (unsigned int d = dim_start; d < dim_end; d++) {
                dot += __bfloat162float(Q_row[d]) * __bfloat162float(K_row[d]);
            }
        }

        // 2026-09-25: XOR shuffles over lane offsets 1, 2 and 4 sum a row's 8 lanes; every lane gets the full dot.

        dot += __shfl_xor_sync(0xFFFFFFFF, dot, 1);
        dot += __shfl_xor_sync(0xFFFFFFFF, dot, 2);
        dot += __shfl_xor_sync(0xFFFFFFFF, dot, 4);


        float score = dot * inv_sqrt_d;


        float m_new = fmaxf(m, score);
        float exp_old = __expf(m - m_new);
        float exp_new = __expf(score - m_new);


        for (unsigned int d = 0; d < 64 && dim_start + d < head_dim; d++) {
            o_acc[d] *= exp_old;
        }
        l = l * exp_old + exp_new;
        m = m_new;


        const __nv_bfloat16* V_row = V + (unsigned long long)batch * seq_len * kv_stride
                                        + (unsigned long long)kv_pos * kv_stride
                                        + (unsigned long long)kv_head * head_dim;
        for (unsigned int d = 0; d < 64 && dim_start + d < head_dim; d++) {
            o_acc[d] += exp_new * __bfloat162float(V_row[dim_start + d]);
        }
    }

    // 2026-09-25: The per-head sink: a logit added to the softmax denominator only, with no value row.




    if (sinks != nullptr) {
        float sg = sinks[q_head];
        float m_new = fmaxf(m, sg);
        float exp_old = __expf(m - m_new);
        float exp_sink = __expf(sg - m_new);
        for (unsigned int d = 0; d < 64 && dim_start + d < head_dim; d++) {
            o_acc[d] *= exp_old;
        }
        l = l * exp_old + exp_sink;
        m = m_new;
    }


    if (valid) {
        float inv_l = (l > 0.0f) ? (1.0f / l) : 0.0f;
        __nv_bfloat16* O_row = O + (unsigned long long)batch * seq_len * q_stride
                                  + (unsigned long long)q_row * q_stride
                                  + (unsigned long long)q_head * head_dim;
        for (unsigned int d = 0; d < 64 && dim_start + d < head_dim; d++) {
            O_row[dim_start + d] = __float2bfloat16(o_acc[d] * inv_l);
        }
    }
}
