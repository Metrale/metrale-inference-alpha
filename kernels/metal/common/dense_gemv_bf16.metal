// SPDX-License-Identifier: MIT OR Apache-2.0

// 2026-09-26: BF16 matrix-vector multiply for unquantized weights, with FP32 accumulation:
//
//   y[n] = sum_k(W[n, k] * x[k])
//
// There is no bias term; a caller that has one adds it separately.
//
// One threadgroup per output row. Its threads stride over K; each simdgroup sums its lanes
// with simd_sum, then simdgroup 0 sums the per-simdgroup partials. A threadgroup may hold
// at most MAX_SIMDGROUPS simdgroups.
//
// Layout:
//   w : bfloat [N, K]
//   x : bfloat [K]
//   y : bfloat [N]





#include <metal_stdlib>
using namespace metal;

constant uint MAX_SIMDGROUPS = 32;

kernel void dense_gemv_bf16(
    constant uint &N        [[buffer(0)]],
    constant uint &K        [[buffer(1)]],
    device const bfloat *w  [[buffer(2)]],
    device const bfloat *x  [[buffer(3)]],
    device bfloat       *y  [[buffer(4)]],
    uint   row     [[threadgroup_position_in_grid]],
    uint   tid     [[thread_position_in_threadgroup]],
    uint   tg_size [[threads_per_threadgroup]],
    uint   simd_lane_id  [[thread_index_in_simdgroup]],
    uint   simd_group_id [[simdgroup_index_in_threadgroup]])
{
    if (row >= N) {
        return;
    }
    threadgroup float partial[MAX_SIMDGROUPS];

    float acc = 0.0f;
    for (uint k = tid; k < K; k += tg_size) {
        float wv = float(w[row * K + k]);
        float xv = float(x[k]);
        acc += wv * xv;
    }

    float simd_acc = simd_sum(acc);
    if (simd_lane_id == 0) {
        partial[simd_group_id] = simd_acc;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    uint num_simds = (tg_size + 31u) / 32u;
    if (simd_group_id == 0) {
        float v = (tid < num_simds) ? partial[tid] : 0.0f;
        v = simd_sum(v);
        if (tid == 0) {
            y[row] = bfloat(v);
        }
    }
}
