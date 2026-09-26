// SPDX-License-Identifier: MIT OR Apache-2.0

// 2026-09-25: Non-causal self-attention: every query token attends to all
// `seq_len` keys. One threadgroup per (token, head) on a flat 1-D grid of
// num_tokens * num_heads threadgroups; kv_h = h / (num_heads / num_kv_heads).
//
// Layout:
//   q   : bfloat [num_tokens, num_heads,   head_dim]
//   k   : bfloat [seq_len,    num_kv_heads, head_dim]
//   v   : bfloat [seq_len,    num_kv_heads, head_dim]
//   out : bfloat [num_tokens, num_heads,   head_dim]
//
// Owner: metal kernels. Invariants: assumes seq_len <= MAX_SEQ_FULL; the
// score loop stops at the cap but the softmax and V loops index `scores` up
// to seq_len.

#include <metal_stdlib>
using namespace metal;

constant uint MAX_SEQ_FULL = 4096;

kernel void attention_full(
    constant uint  &num_tokens   [[buffer(0)]],
    constant uint  &seq_len      [[buffer(1)]],
    constant uint  &num_heads    [[buffer(2)]],
    constant uint  &num_kv_heads [[buffer(3)]],
    constant uint  &head_dim     [[buffer(4)]],
    constant float &scale        [[buffer(5)]],
    device const bfloat *q       [[buffer(6)]],
    device const bfloat *k       [[buffer(7)]],
    device const bfloat *v       [[buffer(8)]],
    device bfloat       *out     [[buffer(9)]],
    uint  tg_idx  [[threadgroup_position_in_grid]],
    uint  tid     [[thread_position_in_threadgroup]],
    uint  tg_size [[threads_per_threadgroup]])
{
    threadgroup float scores[MAX_SEQ_FULL];
    threadgroup float max_score;
    threadgroup float sum_exp;

    uint h = tg_idx % num_heads;
    uint m = tg_idx / num_heads;
    if (m >= num_tokens || h >= num_heads) {
        return;
    }
    uint group = num_heads / num_kv_heads;
    uint kv_h  = h / group;


    for (uint s = tid; s < seq_len && s < MAX_SEQ_FULL; s += tg_size) {
        float dot = 0.0f;
        for (uint d = 0; d < head_dim; ++d) {
            float qv = float(q[(m * num_heads + h) * head_dim + d]);
            float kvv = float(k[(s * num_kv_heads + kv_h) * head_dim + d]);
            dot += qv * kvv;
        }
        scores[s] = dot * scale;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);


    if (tid == 0) {
        float mx = -INFINITY;
        for (uint s = 0; s < seq_len; ++s) {
            if (scores[s] > mx) mx = scores[s];
        }
        max_score = mx;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);


    for (uint s = tid; s < seq_len; s += tg_size) {
        scores[s] = exp(scores[s] - max_score);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);


    if (tid == 0) {
        float sum = 0.0f;
        for (uint s = 0; s < seq_len; ++s) {
            sum += scores[s];
        }
        sum_exp = sum;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);


    float inv_sum = 1.0f / sum_exp;
    for (uint d = tid; d < head_dim; d += tg_size) {
        float acc = 0.0f;
        for (uint s = 0; s < seq_len; ++s) {
            float vv = float(v[(s * num_kv_heads + kv_h) * head_dim + d]);
            acc += scores[s] * inv_sum * vv;
        }
        out[(m * num_heads + h) * head_dim + d] = bfloat(acc);
    }
}
