// SPDX-License-Identifier: MIT OR Apache-2.0

// 2026-09-25: DeepSeek-V4 CSA prefill attention: one online softmax over the raw sliding-window
// keys, the compressed keys and a per-head sink logit.
//
// Owner: gb10 kernels (deepseek-v4-flash).
// Invariants: none beyond the types.

// 2026-09-25: Query row t (of seq_len) attends to raw keys max(0, t + 1 - sliding_window)..=t (all of
// 0..=t when sliding_window is 0), to compressed rows w with (w + 1) * ratio <= t + 1 (at most n_comp),
// and to sinks[q_head], which enters the denominator only and is skipped when sinks is null.
// Q and O are [S, num_q_heads, head_dim], K and V [S, num_kv_heads, head_dim], Kc and Vc [n_comp,
// head_dim], sinks FP32 [num_q_heads]. 128 threads are 16 query rows x 8 lanes of 64 dims, which
// covers head_dim up to 512. Grid (num_q_heads, ceil(S / 16)); cache_skip_v4.rs passes K as V.




#include <cuda_bf16.h>

#define BR 16

extern "C" __global__ void prefill_attn_compressed(
    const __nv_bfloat16* __restrict__ Q,
    const __nv_bfloat16* __restrict__ K,
    const __nv_bfloat16* __restrict__ V,
    const __nv_bfloat16* __restrict__ Kc,
    const __nv_bfloat16* __restrict__ Vc,
    const float* __restrict__ sinks,
    __nv_bfloat16* __restrict__ O,
    const unsigned int seq_len,
    const unsigned int num_q_heads,
    const unsigned int num_kv_heads,
    const unsigned int head_dim,
    const unsigned int n_comp,
    const unsigned int ratio,
    const unsigned int sliding_window,
    const float inv_sqrt_d
) {
    const unsigned int q_head = blockIdx.x;
    const unsigned int q_block = blockIdx.y;
    const unsigned int tid = threadIdx.x;
    if (q_head >= num_q_heads) return;

    const unsigned int q_row = q_block * BR + (tid / 8);
    const bool valid = q_row < seq_len;
    const unsigned int dim_lane = tid % 8;
    const unsigned int dim_start = dim_lane * 64;
    if (dim_start >= head_dim) return;

    const unsigned int gqa = num_q_heads / num_kv_heads;
    const unsigned int kv_head = q_head / gqa;
    const unsigned int q_stride = num_q_heads * head_dim;
    const unsigned int kv_stride = num_kv_heads * head_dim;

    const __nv_bfloat16* Qr = Q + (size_t)q_row * q_stride + (size_t)q_head * head_dim;

    float m = -1e30f, l = 0.0f;
    float o_acc[64];
    for (unsigned int d = 0; d < 64 && dim_start + d < head_dim; ++d) o_acc[d] = 0.0f;


    #define ATTEND(KROW, VROW)                                                   \
    do {                                                                          \
        float dot = 0.0f;                                                         \
        if (valid) {                                                              \
            for (unsigned int d = dim_start; d < dim_start + 64 && d < head_dim; ++d) \
                dot += __bfloat162float(Qr[d]) * __bfloat162float((KROW)[d]);     \
        }                                                                         \
        dot += __shfl_xor_sync(0xFFFFFFFF, dot, 1);                               \
        dot += __shfl_xor_sync(0xFFFFFFFF, dot, 2);                               \
        dot += __shfl_xor_sync(0xFFFFFFFF, dot, 4);                               \
        float score = dot * inv_sqrt_d;                                           \
        float m_new = fmaxf(m, score);                                            \
        float eo = __expf(m - m_new);                                            \
        float en = __expf(score - m_new);                                        \
        for (unsigned int d = 0; d < 64 && dim_start + d < head_dim; ++d)         \
            o_acc[d] = o_acc[d] * eo + en * __bfloat162float((VROW)[dim_start + d]); \
        l = l * eo + en;                                                          \
        m = m_new;                                                               \
    } while (0)


    unsigned int kv_len = valid ? (q_row + 1) : 0;
    unsigned int kv_start = 0;
    if (sliding_window > 0u && kv_len > sliding_window) kv_start = kv_len - sliding_window;
    for (unsigned int kp = kv_start; kp < kv_len; ++kp) {
        const __nv_bfloat16* Kr = K + (size_t)kp * kv_stride + (size_t)kv_head * head_dim;
        const __nv_bfloat16* Vr = V + (size_t)kp * kv_stride + (size_t)kv_head * head_dim;
        ATTEND(Kr, Vr);
    }


    unsigned int comp_vis = valid ? ((q_row + 1) / ratio) : 0;
    if (comp_vis > n_comp) comp_vis = n_comp;
    for (unsigned int w = 0; w < comp_vis; ++w) {
        const __nv_bfloat16* Kr = Kc + (size_t)w * head_dim;
        const __nv_bfloat16* Vr = Vc + (size_t)w * head_dim;
        ATTEND(Kr, Vr);
    }


    if (valid && sinks != nullptr) {
        float sg = sinks[q_head];
        float m_new = fmaxf(m, sg);
        float eo = __expf(m - m_new);
        for (unsigned int d = 0; d < 64 && dim_start + d < head_dim; ++d) o_acc[d] *= eo;
        l = l * eo + __expf(sg - m_new);
        m = m_new;
    }

    if (valid) {
        float inv_l = (l > 0.0f) ? (1.0f / l) : 0.0f;
        __nv_bfloat16* Or = O + (size_t)q_row * q_stride + (size_t)q_head * head_dim;
        for (unsigned int d = 0; d < 64 && dim_start + d < head_dim; ++d)
            Or[dim_start + d] = __float2bfloat16(o_acc[d] * inv_l);
    }
    #undef ATTEND
}
