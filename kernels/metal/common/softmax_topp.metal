// SPDX-License-Identifier: MIT OR Apache-2.0

// 2026-09-25: Top-p (nucleus) sampler over the full vocabulary in one
// threadgroup. With z[i] = exp((logits[i] - max) / temp), which lies in
// (0, 1]:
//   1. find max(logits), then total = sum_i z[i];
//   2. binary-search (BSEARCH_ITERS halvings of [0, 1]) for the largest
//      threshold T whose surviving mass sum_{z[i] >= T} z[i] is still
//      >= p * total; no sorted copy of the vocabulary is needed;
//   3. walk the vocabulary in index order over tokens with z >= T and pick
//      the first at which the running sum reaches uniform * surviving mass
//      (the last surviving token if none does).
// Steps 2 and 3 run on thread 0 alone. result[0] receives the token id.
//
// Layout:
//   logits : bfloat [vocab]
//   result : uint32 [1]
//
// Owner: metal kernels.
// Invariants: assumes 32-lane simdgroups and at most 32 simdgroups per
// threadgroup (`partial`).


#include <metal_stdlib>
using namespace metal;

constant uint MAX_SIMDGROUPS = 32;


constant uint BSEARCH_ITERS = 24;

kernel void softmax_topp(
    constant uint  &vocab    [[buffer(0)]],
    constant float &temp     [[buffer(1)]],
    constant float &p        [[buffer(2)]],
    constant float &uniform  [[buffer(3)]],
    device const bfloat *logits [[buffer(4)]],
    device uint         *result [[buffer(5)]],
    uint  tid     [[thread_position_in_threadgroup]],
    uint  tg_size [[threads_per_threadgroup]],
    uint  simd_lane_id  [[thread_index_in_simdgroup]],
    uint  simd_group_id [[simdgroup_index_in_threadgroup]])
{
    threadgroup float partial[MAX_SIMDGROUPS];
    threadgroup float shared_max;
    threadgroup float shared_total;
    threadgroup float shared_threshold;
    threadgroup float shared_surviving_sum;
    threadgroup uint  shared_pick;


    float local_max = -INFINITY;
    for (uint i = tid; i < vocab; i += tg_size) {
        float v = float(logits[i]);
        if (v > local_max) local_max = v;
    }
    float simd_max_v = simd_max(local_max);
    if (simd_lane_id == 0) {
        partial[simd_group_id] = simd_max_v;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (simd_group_id == 0) {
        uint num_simds = (tg_size + 31u) / 32u;
        float v = (tid < num_simds) ? partial[tid] : -INFINITY;
        v = simd_max(v);
        if (tid == 0) shared_max = v;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float mx = shared_max;


    float local_sum = 0.0f;
    for (uint i = tid; i < vocab; i += tg_size) {
        float z = exp((float(logits[i]) - mx) / temp);
        local_sum += z;
    }
    float simd_sum_v = simd_sum(local_sum);
    if (simd_lane_id == 0) {
        partial[simd_group_id] = simd_sum_v;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (simd_group_id == 0) {
        uint num_simds = (tg_size + 31u) / 32u;
        float v = (tid < num_simds) ? partial[tid] : 0.0f;
        v = simd_sum(v);
        if (tid == 0) shared_total = v;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float total = shared_total;
    float p_target = p * total;



    if (tid == 0) {
        float lo = 0.0f;
        float hi = 1.0f;
        for (uint iter = 0; iter < BSEARCH_ITERS; ++iter) {
            float mid = 0.5f * (lo + hi);
            float surviving = 0.0f;
            for (uint i = 0; i < vocab; ++i) {
                float z = exp((float(logits[i]) - mx) / temp);
                if (z >= mid) {
                    surviving += z;
                }
            }
            if (surviving >= p_target) {
                lo = mid;
            } else {
                hi = mid;
            }
        }
        shared_threshold = lo;

        float surv = 0.0f;
        for (uint i = 0; i < vocab; ++i) {
            float z = exp((float(logits[i]) - mx) / temp);
            if (z >= lo) surv += z;
        }
        shared_surviving_sum = surv;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);


    if (tid == 0) {
        float threshold = shared_threshold;
        float target = uniform * shared_surviving_sum;
        float run = 0.0f;
        uint pick = 0;
        for (uint i = 0; i < vocab; ++i) {
            float z = exp((float(logits[i]) - mx) / temp);
            if (z >= threshold) {
                run += z;
                if (run >= target) {
                    pick = i;
                    break;
                }
                pick = i;
            }
        }
        shared_pick = pick;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid == 0) {
        result[0] = shared_pick;
    }
}
