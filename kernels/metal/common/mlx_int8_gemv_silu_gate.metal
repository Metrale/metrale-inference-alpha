// SPDX-License-Identifier: MIT OR Apache-2.0

// 2026-09-25: GEMV over a SwiGLU input with on-the-fly MLX 8-bit affine
// dequantization:
//
//   y[n] = sum_k W[n, k] * (silu(gate[k]) * up[k])
//
// silu(gate) * up is computed in FP32 per element, so no intermediate
// activation buffer exists. W, the packed/scales/biases layout and the
// threadgroup shape are those of mlx_int8_gemv.metal: one simdgroup per
// row, 4 rows per threadgroup of 128 threads.
//
// Layout:
//   packed         : uint32 [N, K / 4]
//   scales, biases : bfloat [N, K / group_size]
//   gate, up       : bfloat [K]
//   y              : bfloat [N]
//
// Owner: metal kernels.
// Invariants: assumes K % 4 == 0, group_size % 4 == 0, and gate and up are
// 8-byte aligned for the bfloat4 reads.

#include <metal_stdlib>
using namespace metal;

constant uint ROWS_PER_TG_SG    = 4u;
constant uint SIMDGROUP_SIZE_SG = 32u;

kernel void mlx_int8_gemv_silu_gate(
    constant uint &N          [[buffer(0)]],
    constant uint &K          [[buffer(1)]],
    constant uint &group_size [[buffer(2)]],
    device const uint   *packed [[buffer(3)]],
    device const bfloat *scales [[buffer(4)]],
    device const bfloat *biases [[buffer(5)]],
    device const bfloat *gate   [[buffer(6)]],
    device const bfloat *up     [[buffer(7)]],
    device bfloat       *y      [[buffer(8)]],
    uint   tg_idx          [[threadgroup_position_in_grid]],
    uint   simd_lane_id    [[thread_index_in_simdgroup]],
    uint   simd_group_id   [[simdgroup_index_in_threadgroup]])
{
    const uint row = tg_idx * ROWS_PER_TG_SG + simd_group_id;
    if (row >= N) {
        return;
    }

    const uint K4              = K >> 2u;
    const uint groups_per_row  = K / group_size;
    const uint group_words     = group_size >> 2u;

    device const bfloat4 *gate4 = reinterpret_cast<device const bfloat4*>(gate);
    device const bfloat4 *up4   = reinterpret_cast<device const bfloat4*>(up);
    device const uint    *prow  = packed + row * K4;
    device const bfloat  *srow  = scales + row * groups_per_row;
    device const bfloat  *brow  = biases + row * groups_per_row;

    float acc = 0.0f;
    for (uint k4 = simd_lane_id; k4 < K4; k4 += SIMDGROUP_SIZE_SG) {
        const uint    word = prow[k4];
        const uint    g    = k4 / group_words;
        const float   s    = float(srow[g]);
        const float   b    = float(brow[g]);
        const bfloat4 gv   = gate4[k4];
        const bfloat4 uv   = up4[k4];


        const float w0 = float((word >>  0) & 0xFFu) * s + b;
        const float w1 = float((word >>  8) & 0xFFu) * s + b;
        const float w2 = float((word >> 16) & 0xFFu) * s + b;
        const float w3 = float((word >> 24) & 0xFFu) * s + b;



        const float g0 = float(gv.x), u0 = float(uv.x);
        const float g1 = float(gv.y), u1 = float(uv.y);
        const float g2 = float(gv.z), u2 = float(uv.z);
        const float g3 = float(gv.w), u3 = float(uv.w);
        const float f0 = (g0 / (1.0f + exp(-g0))) * u0;
        const float f1 = (g1 / (1.0f + exp(-g1))) * u1;
        const float f2 = (g2 / (1.0f + exp(-g2))) * u2;
        const float f3 = (g3 / (1.0f + exp(-g3))) * u3;

        acc += w0 * f0 + w1 * f1 + w2 * f2 + w3 * f3;
    }

    const float row_sum = simd_sum(acc);
    if (simd_lane_id == 0) {
        y[row] = bfloat(row_sum);
    }
}

// 2026-09-25: mlx_int8_gemv_silu_gate plus the residual:
//   y[n] = x_resid[n] + sum_k W[n, k] * (silu(gate[k]) * up[k]),
// added in FP32 before the one BF16 rounding.


kernel void mlx_int8_gemv_silu_gate_resid(
    constant uint &N          [[buffer(0)]],
    constant uint &K          [[buffer(1)]],
    constant uint &group_size [[buffer(2)]],
    device const uint   *packed  [[buffer(3)]],
    device const bfloat *scales  [[buffer(4)]],
    device const bfloat *biases  [[buffer(5)]],
    device const bfloat *gate    [[buffer(6)]],
    device const bfloat *up      [[buffer(7)]],
    device const bfloat *x_resid [[buffer(8)]],
    device bfloat       *y       [[buffer(9)]],
    uint   tg_idx          [[threadgroup_position_in_grid]],
    uint   simd_lane_id    [[thread_index_in_simdgroup]],
    uint   simd_group_id   [[simdgroup_index_in_threadgroup]])
{
    const uint row = tg_idx * ROWS_PER_TG_SG + simd_group_id;
    if (row >= N) {
        return;
    }

    const uint K4              = K >> 2u;
    const uint groups_per_row  = K / group_size;
    const uint group_words     = group_size >> 2u;

    device const bfloat4 *gate4 = reinterpret_cast<device const bfloat4*>(gate);
    device const bfloat4 *up4   = reinterpret_cast<device const bfloat4*>(up);
    device const uint    *prow  = packed + row * K4;
    device const bfloat  *srow  = scales + row * groups_per_row;
    device const bfloat  *brow  = biases + row * groups_per_row;

    float acc = 0.0f;
    for (uint k4 = simd_lane_id; k4 < K4; k4 += SIMDGROUP_SIZE_SG) {
        const uint    word = prow[k4];
        const uint    g    = k4 / group_words;
        const float   s    = float(srow[g]);
        const float   b    = float(brow[g]);
        const bfloat4 gv   = gate4[k4];
        const bfloat4 uv   = up4[k4];

        const float w0 = float((word >>  0) & 0xFFu) * s + b;
        const float w1 = float((word >>  8) & 0xFFu) * s + b;
        const float w2 = float((word >> 16) & 0xFFu) * s + b;
        const float w3 = float((word >> 24) & 0xFFu) * s + b;

        const float g0 = float(gv.x), u0 = float(uv.x);
        const float g1 = float(gv.y), u1 = float(uv.y);
        const float g2 = float(gv.z), u2 = float(uv.z);
        const float g3 = float(gv.w), u3 = float(uv.w);
        const float f0 = (g0 / (1.0f + exp(-g0))) * u0;
        const float f1 = (g1 / (1.0f + exp(-g1))) * u1;
        const float f2 = (g2 / (1.0f + exp(-g2))) * u2;
        const float f3 = (g3 / (1.0f + exp(-g3))) * u3;

        acc += w0 * f0 + w1 * f1 + w2 * f2 + w3 * f3;
    }

    const float row_sum = simd_sum(acc);
    if (simd_lane_id == 0) {
        y[row] = bfloat(row_sum + float(x_resid[row]));
    }
}
