// SPDX-License-Identifier: MIT OR Apache-2.0

// 2026-09-25: DeepSeek-V4 compressor: one compressed KV row per window of `ratio` source tokens, a per-dim
// softmax-gated sum: out[w, d] = sum_s softmax_s(gate[s, c] + ape[r, c]) * kv[s, c], where s runs over the
// window's slots, r is the slot's row within its source window and c its column.
//
// CSA (`is_csa`; the loader sets it for ratio < 128, with proj_dim = 2 * head_dim): 2 * ratio slots, the
// previous window's rows at columns [0, head_dim), then the current window's at [head_dim, 2 * head_dim);
// window 0 has only the second half. HCA (`is_csa` 0, proj_dim == head_dim): the window's `ratio` rows.
//
// `out` is the compressed KV before its RMS norm and RoPE, which the callers apply (qwen3_attention
// prefill/cache_skip_v4.rs, decode/attention_forward_v4.rs).
// kv, gate: [S, proj_dim] bf16; ape: [ratio, proj_dim] f32; out: [n_win, head_dim] bf16.
// Owner: gb10 kernels (deepseek-v4-flash). Grid: (n_win, 1, 1)  Block: (256, 1, 1).
// Invariants: n_win = seq_len / ratio; tokens after the last whole window are not read.


#include <cuda_bf16.h>

extern "C" __global__ void csa_compress(
    const __nv_bfloat16* __restrict__ kv,
    const __nv_bfloat16* __restrict__ gate,
    const float* __restrict__ ape,



    __nv_bfloat16* __restrict__ out,
    const unsigned int seq_len,
    const unsigned int ratio,
    const unsigned int head_dim,
    const unsigned int proj_dim,
    const unsigned int is_csa
) {
    const unsigned int w = blockIdx.x;
    const unsigned int n_win = seq_len / ratio;
    if (w >= n_win) return;

    for (unsigned int d = threadIdx.x; d < head_dim; d += blockDim.x) {
        // 2026-09-25: Online softmax over the slots, per output dim d.
        float m = -1e30f, l = 0.0f, acc = 0.0f;

        if (is_csa) {
            // 2026-09-25: The previous window's rows at columns [0, head_dim); window 0 has none.

            if (w > 0) {
                for (unsigned int r = 0; r < ratio; ++r) {
                    const unsigned int tok = (w - 1) * ratio + r;
                    const float g = __bfloat162float(gate[(size_t)tok * proj_dim + d])
                                  + ape[(size_t)r * proj_dim + d];
                    const float v = __bfloat162float(kv[(size_t)tok * proj_dim + d]);
                    const float mn = fmaxf(m, g);
                    const float eo = __expf(m - mn);
                    const float en = __expf(g - mn);
                    l = l * eo + en;
                    acc = acc * eo + en * v;
                    m = mn;
                }
            }
            // 2026-09-25: The current window's rows at columns [head_dim, 2 * head_dim).
            for (unsigned int r = 0; r < ratio; ++r) {
                const unsigned int tok = w * ratio + r;
                const unsigned int c = head_dim + d;
                const float g = __bfloat162float(gate[(size_t)tok * proj_dim + c])
                              + ape[(size_t)r * proj_dim + c];
                const float v = __bfloat162float(kv[(size_t)tok * proj_dim + c]);
                const float mn = fmaxf(m, g);
                const float eo = __expf(m - mn);
                const float en = __expf(g - mn);
                l = l * eo + en;
                acc = acc * eo + en * v;
                m = mn;
            }
        } else {
            // 2026-09-25: HCA: the window's `ratio` rows, proj_dim == head_dim.
            for (unsigned int r = 0; r < ratio; ++r) {
                const unsigned int tok = w * ratio + r;
                const float g = __bfloat162float(gate[(size_t)tok * proj_dim + d])
                              + ape[(size_t)r * proj_dim + d];
                const float v = __bfloat162float(kv[(size_t)tok * proj_dim + d]);
                const float mn = fmaxf(m, g);
                const float eo = __expf(m - mn);
                const float en = __expf(g - mn);
                l = l * eo + en;
                acc = acc * eo + en * v;
                m = mn;
            }
        }

        out[(size_t)w * head_dim + d] = __float2bfloat16(l > 0.0f ? acc / l : 0.0f);
    }
}
