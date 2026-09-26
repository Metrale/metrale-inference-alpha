// SPDX-License-Identifier: MIT OR Apache-2.0

// 2026-09-25: Decode attention for one query token against a contiguous Turbo4
// KV cache: 4-bit indices into a unit-Gaussian Lloyd-Max codebook, 2 per
// byte (low nibble first), with one E4M3 scale per 16 values. One
// threadgroup per query head. K and V are decoded without an inverse
// rotation, so `q` must be in the basis K was stored in and `out` is in the
// basis V was stored in. A position whose exp(score - max) <=
// sparse_v_threshold is left out of the V sum.
//
// Layout: q and out are bfloat [num_heads, head_dim]; k_data and v_data are
// uchar [seq_len, num_kv_heads * head_dim / 2]; k_scales and v_scales are
// E4M3 uchar [seq_len, num_kv_heads * head_dim / 16].
//
// Owner: metal kernels. Invariants: reads only the cache positions below
// min(seq_len, MAX_SEQ_DECODE_TQ4).



#include <metal_stdlib>
using namespace metal;

constant uint MAX_SEQ_DECODE_TQ4 = 4096;
constant uint TQ4_GROUP_SIZE = 16;

constant float TURBO4_CODEBOOK[16] = {
    -2.7326f, -2.0690f, -1.6180f, -1.2562f, -0.9423f, -0.6568f, -0.3880f, -0.1284f,
     0.1284f,  0.3880f,  0.6568f,  0.9423f,  1.2562f,  1.6180f,  2.0690f,  2.7326f
};

static inline float e4m3_to_f32(uchar b) {
    float sign = (b & 0x80) ? -1.0f : 1.0f;
    uint e = (b >> 3) & 0xF;
    uint m = b & 7;
    if (e == 0) {
        return sign * float(m) * 0.001953125f;
    }
    return sign * (1.0f + float(m) * 0.125f) * exp2(float(int(e) - 7));
}

kernel void attention_decode_turbo4(
    constant uint  &seq_len      [[buffer(0)]],
    constant uint  &num_heads    [[buffer(1)]],
    constant uint  &num_kv_heads [[buffer(2)]],
    constant uint  &head_dim     [[buffer(3)]],
    constant float &scale        [[buffer(4)]],


    constant float &sparse_v_threshold [[buffer(5)]],
    device const bfloat *q       [[buffer(6)]],
    device const uchar  *k_data  [[buffer(7)]],
    device const uchar  *v_data  [[buffer(8)]],
    device const uchar  *k_scales [[buffer(9)]],
    device const uchar  *v_scales [[buffer(10)]],
    device bfloat       *out     [[buffer(11)]],
    uint h       [[threadgroup_position_in_grid]],
    uint tid     [[thread_position_in_threadgroup]],
    uint tg_size [[threads_per_threadgroup]])
{
    threadgroup float scores[MAX_SEQ_DECODE_TQ4];
    threadgroup float max_score;
    threadgroup float sum_exp;

    if (h >= num_heads) {
        return;
    }



    uint seq = min(seq_len, MAX_SEQ_DECODE_TQ4);
    uint group = num_heads / num_kv_heads;
    uint kv_h  = h / group;
    uint n_elems = num_kv_heads * head_dim;
    uint row_bytes = n_elems / 2;
    uint num_groups = n_elems / TQ4_GROUP_SIZE;


    for (uint s = tid; s < seq; s += tg_size) {
        device const uchar *k_row = k_data + (ulong)s * row_bytes + kv_h * head_dim / 2;
        device const uchar *k_srow =
            k_scales + (ulong)s * num_groups + kv_h * head_dim / TQ4_GROUP_SIZE;
        float dot = 0.0f;
        for (uint d = 0; d < head_dim; d += TQ4_GROUP_SIZE) {
            float gs = e4m3_to_f32(k_srow[d / TQ4_GROUP_SIZE]);
            for (uint i = 0; i < TQ4_GROUP_SIZE; i += 2) {
                uchar packed = k_row[(d + i) / 2];
                float qv0 = float(q[h * head_dim + d + i]);
                float qv1 = float(q[h * head_dim + d + i + 1]);
                dot += qv0 * TURBO4_CODEBOOK[packed & 0xF] * gs;
                dot += qv1 * TURBO4_CODEBOOK[packed >> 4] * gs;
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
        uint sg = (kv_h * head_dim + d) / TQ4_GROUP_SIZE;
        uint byte_idx = (kv_h * head_dim + d) / 2;
        bool high = (d & 1) != 0;
        float acc = 0.0f;
        for (uint s = 0; s < seq; ++s) {
            if (scores[s] <= sparse_v_threshold) {
                continue;
            }
            uchar packed = v_data[(ulong)s * row_bytes + byte_idx];
            uchar idx = high ? (packed >> 4) : (packed & 0xF);
            float vv = TURBO4_CODEBOOK[idx]
                * e4m3_to_f32(v_scales[(ulong)s * num_groups + sg]);
            acc += scores[s] * inv_sum * vv;
        }
        out[h * head_dim + d] = bfloat(acc);
    }
}
