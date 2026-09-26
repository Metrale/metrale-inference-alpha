// SPDX-License-Identifier: MIT OR Apache-2.0

// 2026-09-25: Decode attention for one query token against a contiguous Turbo3
// KV cache: 3-bit indices into a unit-Gaussian Lloyd-Max codebook, 8 per
// 3 bytes, with one E4M3 scale per 16 values. One threadgroup per query head.
// K and V are decoded without an inverse rotation, so `q` must be in the basis
// K was stored in and `out` is in the basis V was stored in. A position whose
// exp(score - max) <= sparse_v_threshold is left out of the V sum.
//
// Layout: q and out are bfloat [num_heads, head_dim]; k_data and v_data are
// uchar [seq_len, num_kv_heads * head_dim * 3 / 8]; k_scales and v_scales
// are E4M3 uchar [seq_len, num_kv_heads * head_dim / 16].
//
// Owner: metal kernels. Invariants: reads only the cache positions below
// min(seq_len, MAX_SEQ_DECODE_TQ3).



#include <metal_stdlib>
using namespace metal;

constant uint MAX_SEQ_DECODE_TQ3 = 4096;
constant uint TQ3_GROUP_SIZE = 16;

constant float TURBO3_CODEBOOK[8] = {
    -2.1520f, -1.3440f, -0.7560f, -0.2451f, 0.2451f, 0.7560f, 1.3440f, 2.1520f
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

// 2026-09-25: Index i (0..7) of a 3-byte pack of eight 3-bit indices, LSB first.
static inline uchar unpack3(device const uchar *b, uint i) {
    switch (i) {
        case 0: return b[0] & 7;
        case 1: return (b[0] >> 3) & 7;
        case 2: return ((b[0] >> 6) | (b[1] << 2)) & 7;
        case 3: return (b[1] >> 1) & 7;
        case 4: return (b[1] >> 4) & 7;
        case 5: return ((b[1] >> 7) | (b[2] << 1)) & 7;
        case 6: return (b[2] >> 2) & 7;
        default: return (b[2] >> 5) & 7;
    }
}

kernel void attention_decode_turbo3(
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
    threadgroup float scores[MAX_SEQ_DECODE_TQ3];
    threadgroup float max_score;
    threadgroup float sum_exp;

    if (h >= num_heads) {
        return;
    }



    uint seq = min(seq_len, MAX_SEQ_DECODE_TQ3);
    uint group = num_heads / num_kv_heads;
    uint kv_h  = h / group;
    uint n_elems = num_kv_heads * head_dim;
    uint row_bytes = n_elems * 3 / 8;
    uint num_groups = n_elems / TQ3_GROUP_SIZE;


    for (uint s = tid; s < seq; s += tg_size) {
        device const uchar *k_row =
            k_data + (ulong)s * row_bytes + kv_h * head_dim * 3 / 8;
        device const uchar *k_srow =
            k_scales + (ulong)s * num_groups + kv_h * head_dim / TQ3_GROUP_SIZE;
        float dot = 0.0f;
        for (uint d = 0; d < head_dim; d += TQ3_GROUP_SIZE) {
            float gs = e4m3_to_f32(k_srow[d / TQ3_GROUP_SIZE]);
            // 2026-09-25: One group of 16 indices is two 3-byte packs.
            device const uchar *p = k_row + d * 3 / 8;
            for (uint half_g = 0; half_g < 2; ++half_g) {
                for (uint i = 0; i < 8; ++i) {
                    float qv = float(q[h * head_dim + d + half_g * 8 + i]);
                    dot += qv * TURBO3_CODEBOOK[unpack3(p + half_g * 3, i)] * gs;
                }
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
        uint elem = kv_h * head_dim + d;
        uint sg = elem / TQ3_GROUP_SIZE;

        uint pack_base = (elem / 8) * 3;
        uint in_pack = elem & 7;
        float acc = 0.0f;
        for (uint s = 0; s < seq; ++s) {
            if (scores[s] <= sparse_v_threshold) {
                continue;
            }
            device const uchar *p = v_data + (ulong)s * row_bytes + pack_base;
            float vv = TURBO3_CODEBOOK[unpack3(p, in_pack)]
                * e4m3_to_f32(v_scales[(ulong)s * num_groups + sg]);
            acc += scores[s] * inv_sum * vv;
        }
        out[h * head_dim + d] = bfloat(acc);
    }
}
