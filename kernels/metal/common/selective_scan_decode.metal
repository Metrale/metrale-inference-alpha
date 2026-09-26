// SPDX-License-Identifier: MIT OR Apache-2.0

// 2026-09-25: Single-token selective-scan (state-space) decode step, one
// threadgroup per channel c:
//
//   dt[h]    = softplus(dt_raw[h] + dt_bias[h])   (dt_pre itself above 20)
//   decay[h] = exp(-dt[h] * exp(A_log[h]))
//   S[h, c]  = S[h, c] * decay[h] + dt[h] * B[h] * x[c]
//   y[c]     = sum_h S[h, c] * C[h]
//
// Arithmetic is FP32; y uses the updated state before it is rounded to BF16
// for storage.
//
// Layout:
//   A_log    : float  [num_heads]
//   dt_bias  : bfloat [num_heads]
//   dt_raw   : bfloat [num_heads]
//   B        : bfloat [num_heads]
//   C        : bfloat [num_heads]
//   x        : bfloat [num_channels]
//   state    : bfloat [num_heads, num_channels]    (in/out)
//   y        : bfloat [num_channels]
//
// Owner: metal kernels.
// Invariants: assumes 32-lane simdgroups and at most 32 simdgroups per
// threadgroup (`partial`).








#include <metal_stdlib>
using namespace metal;

constant uint MAX_SIMDGROUPS_SSM = 32;

kernel void selective_scan_decode(
    constant uint  &num_heads    [[buffer(0)]],
    constant uint  &num_channels [[buffer(1)]],
    device const float  *A_log   [[buffer(2)]],
    device const bfloat *dt_bias [[buffer(3)]],
    device const bfloat *dt_raw  [[buffer(4)]],
    device const bfloat *B       [[buffer(5)]],
    device const bfloat *C       [[buffer(6)]],
    device const bfloat *x       [[buffer(7)]],
    device bfloat       *state   [[buffer(8)]],
    device bfloat       *y       [[buffer(9)]],
    uint   ch_idx [[threadgroup_position_in_grid]],
    uint   tid    [[thread_position_in_threadgroup]],
    uint   tg_size [[threads_per_threadgroup]],
    uint   simd_lane_id  [[thread_index_in_simdgroup]],
    uint   simd_group_id [[simdgroup_index_in_threadgroup]])
{
    threadgroup float partial[MAX_SIMDGROUPS_SSM];

    if (ch_idx >= num_channels) {
        return;
    }
    float xc = float(x[ch_idx]);



    float local_y = 0.0f;
    for (uint h = tid; h < num_heads; h += tg_size) {
        float a_eff = -exp(A_log[h]);
        float dt_pre = float(dt_raw[h]) + float(dt_bias[h]);

        float dt = (dt_pre > 20.0f) ? dt_pre : log(1.0f + exp(dt_pre));
        float decay = exp(dt * a_eff);
        float bv = float(B[h]);
        float cv = float(C[h]);

        uint  state_off = h * num_channels + ch_idx;
        float old_s = float(state[state_off]);
        float new_s = old_s * decay + dt * bv * xc;
        state[state_off] = bfloat(new_s);

        local_y += new_s * cv;
    }


    float simd_sum_v = simd_sum(local_y);
    if (simd_lane_id == 0) {
        partial[simd_group_id] = simd_sum_v;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    uint num_simds = (tg_size + 31u) / 32u;
    if (simd_group_id == 0) {
        float v = (tid < num_simds) ? partial[tid] : 0.0f;
        v = simd_sum(v);
        if (tid == 0) {
            y[ch_idx] = bfloat(v);
        }
    }
}
