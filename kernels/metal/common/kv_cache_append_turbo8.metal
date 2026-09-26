// SPDX-License-Identifier: MIT OR Apache-2.0

// 2026-09-25: Append one token's K and V to a contiguous Turbo8 cache at row
// cache_pos: FP8 E4M3 values (1 byte each) with one BF16 scale per group of
// 16, scale = max(max|x| / 448, 1e-12). K and V are quantized as given: this
// kernel does not rotate them.
//
// Layout:
//   new_k    : bfloat [num_kv_heads, head_dim]
//   new_v    : bfloat [num_kv_heads, head_dim]
//   k_data   : uchar  [max_seq, num_kv_heads * head_dim]
//   v_data   : uchar  [max_seq, num_kv_heads * head_dim]
//   k_scales : bfloat [max_seq, num_kv_heads * head_dim / 16]
//   v_scales : bfloat [max_seq, num_kv_heads * head_dim / 16]
//
// Grid: (num_groups, 1, 1) threads, one per group of 16 elements.
//
// Owner: metal kernels.
// Invariants: none beyond the types.



#include <metal_stdlib>
using namespace metal;

constant uint TQ8_GROUP_SIZE = 16;
constant float FP8_E4M3_MAX = 448.0f;

// 2026-09-25: float → FP8 E4M3 byte, saturating at ±448 (0x7E), mantissa rounded
// half away from zero.
static inline uchar f32_to_e4m3(float f) {
    uchar sign = f < 0.0f ? 0x80 : 0x00;
    float a = fabs(f);
    if (a >= FP8_E4M3_MAX) return sign | 0x7E;
    if (a < 0.001953125f) {
        uint m = uint(round(a * 512.0f));
        return sign | uchar(m);
    }
    int e = int(floor(log2(a)));
    if (e < -6) e = -6;
    float man = a / exp2(float(e));
    uint m3 = uint(round((man - 1.0f) * 8.0f));
    if (m3 == 8) { e += 1; m3 = 0; }
    return sign | uchar((e + 7) << 3) | uchar(m3);
}

kernel void kv_cache_append_turbo8(
    constant uint &num_kv_heads [[buffer(0)]],
    constant uint &head_dim     [[buffer(1)]],
    constant uint &cache_pos    [[buffer(2)]],
    device const bfloat *new_k  [[buffer(3)]],
    device const bfloat *new_v  [[buffer(4)]],
    device uchar  *k_data       [[buffer(5)]],
    device uchar  *v_data       [[buffer(6)]],
    device bfloat *k_scales     [[buffer(7)]],
    device bfloat *v_scales     [[buffer(8)]],
    uint g [[thread_position_in_grid]])
{
    uint n_elems = num_kv_heads * head_dim;
    uint num_groups = n_elems / TQ8_GROUP_SIZE;
    if (g >= num_groups) {
        return;
    }
    uint elem_off = g * TQ8_GROUP_SIZE;

    float kf[TQ8_GROUP_SIZE], vf[TQ8_GROUP_SIZE];
    float k_max = 0.0f, v_max = 0.0f;
    for (uint i = 0; i < TQ8_GROUP_SIZE; ++i) {
        kf[i] = float(new_k[elem_off + i]);
        vf[i] = float(new_v[elem_off + i]);
        k_max = max(k_max, fabs(kf[i]));
        v_max = max(v_max, fabs(vf[i]));
    }

    float k_scale = max(k_max / FP8_E4M3_MAX, 1e-12f);
    float v_scale = max(v_max / FP8_E4M3_MAX, 1e-12f);

    uint scale_row = cache_pos * num_groups;
    k_scales[scale_row + g] = bfloat(k_scale);
    v_scales[scale_row + g] = bfloat(v_scale);
    // 2026-09-25: Divide by the BF16-rounded scale, the value the decode
    // kernel reads back, so the scale's own rounding does not bias the
    // dequantized values.
    float k_inv = 1.0f / float(bfloat(k_scale));
    float v_inv = 1.0f / float(bfloat(v_scale));

    uint data_row = cache_pos * n_elems;
    for (uint i = 0; i < TQ8_GROUP_SIZE; ++i) {
        k_data[data_row + elem_off + i] = f32_to_e4m3(kf[i] * k_inv);
        v_data[data_row + elem_off + i] = f32_to_e4m3(vf[i] * v_inv);
    }
}
