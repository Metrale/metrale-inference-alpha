// SPDX-License-Identifier: MIT OR Apache-2.0

// 2026-09-25: Argmax over a bfloat vector: writes the index of the largest
// element to result[0]. One threadgroup walks all `n` elements; each thread
// keeps its first strict maximum, then two simdgroup tournaments reduce
// through threadgroup memory. Equal values go to the smaller index.
//
// Layout:
//   logits : bfloat [n]
//   result : uint32 [1]
//
// Owner: metal kernels.
// Invariants: assumes 32-lane simdgroups and at most 32 simdgroups per
// threadgroup (`partial_val`, `partial_idx`).


#include <metal_stdlib>
using namespace metal;

constant uint MAX_SIMDGROUPS = 32;

kernel void argmax_bf16(
    constant uint &n           [[buffer(0)]],
    device const bfloat *logits [[buffer(1)]],
    device uint         *result [[buffer(2)]],
    uint  tid     [[thread_position_in_threadgroup]],
    uint  tg_size [[threads_per_threadgroup]],
    uint  simd_lane_id  [[thread_index_in_simdgroup]],
    uint  simd_group_id [[simdgroup_index_in_threadgroup]])
{
    threadgroup float partial_val[MAX_SIMDGROUPS];
    threadgroup uint  partial_idx[MAX_SIMDGROUPS];


    float best_val = -INFINITY;
    uint  best_idx = 0;
    for (uint i = tid; i < n; i += tg_size) {
        float v = float(logits[i]);
        if (v > best_val) {
            best_val = v;
            best_idx = i;
        }
    }




    for (uint offset = 16u; offset > 0u; offset >>= 1u) {
        float other_val = simd_shuffle_xor(best_val, offset);
        uint  other_idx = simd_shuffle_xor(best_idx, offset);
        bool other_wins = other_val > best_val ||
                          (other_val == best_val && other_idx < best_idx);
        best_val = other_wins ? other_val : best_val;
        best_idx = other_wins ? other_idx : best_idx;
    }

    if (simd_lane_id == 0) {
        partial_val[simd_group_id] = best_val;
        partial_idx[simd_group_id] = best_idx;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);



    uint num_simds = (tg_size + 31u) / 32u;
    if (simd_group_id == 0) {
        float v = (tid < num_simds) ? partial_val[tid] : -INFINITY;
        uint  i = (tid < num_simds) ? partial_idx[tid] : 0u;
        for (uint offset = 16u; offset > 0u; offset >>= 1u) {
            float other_v = simd_shuffle_xor(v, offset);
            uint  other_i = simd_shuffle_xor(i, offset);
            bool other_wins = other_v > v ||
                              (other_v == v && other_i < i);
            v = other_wins ? other_v : v;
            i = other_wins ? other_i : i;
        }
        if (tid == 0) {
            result[0] = i;
        }
    }
}
