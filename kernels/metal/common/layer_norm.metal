// SPDX-License-Identifier: MIT OR Apache-2.0

// 2026-09-25: LayerNorm with bias, one threadgroup per token:
//
//   out[i] = ((x[i] - mean(x)) / sqrt(var(x) + eps)) * weight[i] + bias[i]
//
// var is the biased variance (divided by hidden_size), computed around the
// mean in a second pass. All reductions accumulate in FP32.
//
// Layout:
//   x      : bfloat [num_tokens, hidden_size]
//   weight : bfloat [hidden_size]
//   bias   : bfloat [hidden_size]
//   out    : bfloat [num_tokens, hidden_size]
//
// Owner: metal kernels.
// Invariants: assumes 32-lane simdgroups and at most 32 simdgroups per
// threadgroup (`partial`).



#include <metal_stdlib>
using namespace metal;

constant uint MAX_SIMDGROUPS = 32;

kernel void layer_norm(
    constant uint  &hidden_size [[buffer(0)]],
    constant float &eps         [[buffer(1)]],
    device const bfloat *x      [[buffer(2)]],
    device const bfloat *weight [[buffer(3)]],
    device const bfloat *bias   [[buffer(4)]],
    device bfloat       *out    [[buffer(5)]],
    uint   tok_idx [[threadgroup_position_in_grid]],
    uint   tid     [[thread_position_in_threadgroup]],
    uint   tg_size [[threads_per_threadgroup]],
    uint   simd_lane_id  [[thread_index_in_simdgroup]],
    uint   simd_group_id [[simdgroup_index_in_threadgroup]])
{
    threadgroup float partial[MAX_SIMDGROUPS];
    threadgroup float shared_mean;
    threadgroup float shared_var;


    float local_sum = 0.0f;
    for (uint i = tid; i < hidden_size; i += tg_size) {
        local_sum += float(x[tok_idx * hidden_size + i]);
    }
    float simd_sum_v = simd_sum(local_sum);
    if (simd_lane_id == 0) {
        partial[simd_group_id] = simd_sum_v;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    uint num_simds = (tg_size + 31u) / 32u;
    if (simd_group_id == 0) {
        float v = (tid < num_simds) ? partial[tid] : 0.0f;
        v = simd_sum(v);
        if (tid == 0) {
            shared_mean = v / float(hidden_size);
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float mean = shared_mean;


    float local_var = 0.0f;
    for (uint i = tid; i < hidden_size; i += tg_size) {
        float c = float(x[tok_idx * hidden_size + i]) - mean;
        local_var += c * c;
    }
    float simd_var = simd_sum(local_var);
    if (simd_lane_id == 0) {
        partial[simd_group_id] = simd_var;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (simd_group_id == 0) {
        float v = (tid < num_simds) ? partial[tid] : 0.0f;
        v = simd_sum(v);
        if (tid == 0) {
            shared_var = v / float(hidden_size);
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float var = shared_var;


    float inv_std = rsqrt(var + eps);
    for (uint i = tid; i < hidden_size; i += tg_size) {
        float xi = float(x[tok_idx * hidden_size + i]);
        float w  = float(weight[i]);
        float b  = float(bias[i]);
        out[tok_idx * hidden_size + i] = bfloat((xi - mean) * inv_std * w + b);
    }
}
