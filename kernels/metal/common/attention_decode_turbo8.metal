// SPDX-License-Identifier: MIT OR Apache-2.0

// 2026-09-25: Decode attention for one query token against a contiguous Turbo8
// KV cache: FP8 E4M3 values with one BF16 scale per 16 values. One
// threadgroup per query head. K and V are decoded without an inverse
// rotation, so `q` must be in the basis K was stored in and `out` is in the
// basis V was stored in. A position whose exp(score - max) <=
// sparse_v_threshold is left out of the V sum.
//
// Layout:
//   q        : bfloat [num_heads, head_dim]
//   k_data   : uchar  [seq_len, num_kv_heads * head_dim]
//   v_data   : uchar  [seq_len, num_kv_heads * head_dim]
//   k_scales : bfloat [seq_len, num_kv_heads * head_dim / 16]
//   v_scales : bfloat [seq_len, num_kv_heads * head_dim / 16]
//   out      : bfloat [num_heads, head_dim]
//
// Owner: metal kernels. Invariants: reads only the cache positions below
// min(seq_len, MAX_SEQ_DECODE_TQ8).

#include <metal_stdlib>
using namespace metal;

constant uint MAX_SEQ_DECODE_TQ8 = 4096;
constant uint TQ8_GROUP_SIZE = 16;

// 2026-09-25: FP8 E4M3 byte → float: exponent bias 7; exponent field 0 is subnormal, m * 2^-9.
static inline float e4m3_to_f32(uchar b) {
    float sign = (b & 0x80) ? -1.0f : 1.0f;
    uint e = (b >> 3) & 0xF;
    uint m = b & 7;
    if (e == 0) {
        return sign * float(m) * 0.001953125f;
    }
    return sign * (1.0f + float(m) * 0.125f) * exp2(float(int(e) - 7));
}

kernel void attention_decode_turbo8(
    constant uint  &seq_len      [[buffer(0)]],
    constant uint  &num_heads    [[buffer(1)]],
    constant uint  &num_kv_heads [[buffer(2)]],
    constant uint  &head_dim     [[buffer(3)]],
    constant float &scale        [[buffer(4)]],


    constant float &sparse_v_threshold [[buffer(5)]],
    device const bfloat *q       [[buffer(6)]],
    device const uchar  *k_data  [[buffer(7)]],
    device const uchar  *v_data  [[buffer(8)]],
    device const bfloat *k_scales [[buffer(9)]],
    device const bfloat *v_scales [[buffer(10)]],
    device bfloat       *out     [[buffer(11)]],
    uint h       [[threadgroup_position_in_grid]],
    uint tid     [[thread_position_in_threadgroup]],
    uint tg_size [[threads_per_threadgroup]])
{
    threadgroup float scores[MAX_SEQ_DECODE_TQ8];
    threadgroup float max_score;
    threadgroup float sum_exp;

    if (h >= num_heads) {
        return;
    }



    uint seq = min(seq_len, MAX_SEQ_DECODE_TQ8);
    uint group = num_heads / num_kv_heads;
    uint kv_h  = h / group;
    uint n_elems = num_kv_heads * head_dim;
    uint num_groups = n_elems / TQ8_GROUP_SIZE;


    for (uint s = tid; s < seq; s += tg_size) {
        device const uchar  *k_row = k_data + (ulong)s * n_elems + kv_h * head_dim;
        device const bfloat *k_srow =
            k_scales + (ulong)s * num_groups + kv_h * head_dim / TQ8_GROUP_SIZE;
        float dot = 0.0f;
        for (uint d = 0; d < head_dim; d += TQ8_GROUP_SIZE) {
            float gs = float(k_srow[d / TQ8_GROUP_SIZE]);
            for (uint i = 0; i < TQ8_GROUP_SIZE; ++i) {
                float qv = float(q[h * head_dim + d + i]);
                dot += qv * e4m3_to_f32(k_row[d + i]) * gs;
            }
        }
        scores[s] = dot * scale;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);


    if (tid == 0) {
        float m = -INFINITY;
        for (uint s = 0; s < seq; ++s) {
            if (scores[s] > m) {
                m = scores[s];
            }
        }
        max_score = m;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);


    for (uint s = tid; s < seq; s += tg_size) {
        scores[s] = exp(scores[s] - max_score);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);


    if (tid == 0) {
        float sum = 0.0f;
        for (uint s = 0; s < seq; ++s) {
            sum += scores[s];
        }
        sum_exp = sum;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);



    float inv_sum = 1.0f / sum_exp;
    for (uint d = tid; d < head_dim; d += tg_size) {
        uint sg = (kv_h * head_dim + d) / TQ8_GROUP_SIZE;
        float acc = 0.0f;
        for (uint s = 0; s < seq; ++s) {
            if (scores[s] <= sparse_v_threshold) {
                continue;
            }
            float vv = e4m3_to_f32(v_data[(ulong)s * n_elems + kv_h * head_dim + d])
                * float(v_scales[(ulong)s * num_groups + sg]);
            acc += scores[s] * inv_sum * vv;
        }
        out[h * head_dim + d] = bfloat(acc);
    }
}
