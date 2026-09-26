// SPDX-License-Identifier: MIT OR Apache-2.0

// 2026-09-25: GEMV with on-the-fly MLX 8-bit affine dequantization:
//
//   y[n] = sum_k (q[n, k] * scale[n, g] + bias[n, g]) * x[k],  g = k / group_size
//
// where q[n, k] is the unsigned byte k of row n. One simdgroup per output
// row and 4 rows per threadgroup: dispatch ceil(N / 4) threadgroups of 128
// threads. Each lane reads whole packed words (4 weights) and 4 x values as
// one bfloat4.
//
// Layout:
//   packed : uint32  [N, K / 4]
//   scales : bfloat  [N, K / group_size]
//   biases : bfloat  [N, K / group_size]
//   x      : bfloat  [K]
//   y      : bfloat  [N]
//
// Owner: metal kernels.
// Invariants: assumes K % 4 == 0 and group_size % 4 == 0, so the 4 bytes of
// a packed word share one scale and bias, and x is 8-byte aligned for the
// bfloat4 reads.










#include <metal_stdlib>
using namespace metal;

constant uint ROWS_PER_TG    = 4u;
constant uint SIMDGROUP_SIZE = 32u;

kernel void mlx_int8_gemv(
    constant uint &N          [[buffer(0)]],
    constant uint &K          [[buffer(1)]],
    constant uint &group_size [[buffer(2)]],
    device const uint   *packed [[buffer(3)]],
    device const bfloat *scales [[buffer(4)]],
    device const bfloat *biases [[buffer(5)]],
    device const bfloat *x      [[buffer(6)]],
    device bfloat       *y      [[buffer(7)]],
    uint   tg_idx          [[threadgroup_position_in_grid]],
    uint   simd_lane_id    [[thread_index_in_simdgroup]],
    uint   simd_group_id   [[simdgroup_index_in_threadgroup]])
{
    const uint row = tg_idx * ROWS_PER_TG + simd_group_id;
    if (row >= N) {
        return;
    }

    const uint K4              = K >> 2u;
    const uint groups_per_row  = K / group_size;
    const uint group_words     = group_size >> 2u;

    device const bfloat4 *x4 = reinterpret_cast<device const bfloat4*>(x);
    device const uint    *prow = packed + row * K4;
    device const bfloat  *srow = scales + row * groups_per_row;
    device const bfloat  *brow = biases + row * groups_per_row;

    float acc = 0.0f;

    for (uint k4 = simd_lane_id; k4 < K4; k4 += SIMDGROUP_SIZE) {
        const uint   word = prow[k4];
        const uint   g    = k4 / group_words;
        const float  s    = float(srow[g]);
        const float  b    = float(brow[g]);
        const bfloat4 xv  = x4[k4];


        const float w0 = float((word >>  0) & 0xFFu) * s + b;
        const float w1 = float((word >>  8) & 0xFFu) * s + b;
        const float w2 = float((word >> 16) & 0xFFu) * s + b;
        const float w3 = float((word >> 24) & 0xFFu) * s + b;

        acc += w0 * float(xv.x) + w1 * float(xv.y)
             + w2 * float(xv.z) + w3 * float(xv.w);
    }




    const float row_sum = simd_sum(acc);
    if (simd_lane_id == 0) {
        y[row] = bfloat(row_sum);
    }
}
