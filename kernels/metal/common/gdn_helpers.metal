// SPDX-License-Identifier: MIT OR Apache-2.0

// 2026-09-25: Per-head and element-wise helpers for the gated delta rule path:
// gdn_compute_gate, sigmoid_bf16_to_f32, silu_apply and bf16_mul. Each runs
// one thread per element and computes in FP32.
//
// Owner: metal kernels.
// Invariants: none beyond the types.

#include <metal_stdlib>
using namespace metal;

// 2026-09-25: gdn_compute_gate, one thread per head:
//
//   dt[h]   = softplus(dt_raw[h] + dt_bias[h])   (dt_pre itself above 20)
//   gate[h] = exp(-dt[h] * exp(A_log[h]))        in (0, 1]
//
// Layout:
//   dt_raw  : bfloat [num_heads]
//   dt_bias : bfloat [num_heads]
//   A_log   : float  [num_heads]
//   gate    : float  [num_heads]





kernel void gdn_compute_gate(
    constant uint &num_heads     [[buffer(0)]],
    device const bfloat *dt_raw  [[buffer(1)]],
    device const bfloat *dt_bias [[buffer(2)]],
    device const float  *A_log   [[buffer(3)]],
    device float        *gate    [[buffer(4)]],
    uint h [[thread_position_in_grid]])
{
    if (h >= num_heads) {
        return;
    }
    float dt_pre = float(dt_raw[h]) + float(dt_bias[h]);

    float dt = (dt_pre > 20.0f) ? dt_pre : log(1.0f + exp(dt_pre));
    float a_eff = -exp(A_log[h]);
    gate[h] = exp(dt * a_eff);
}











kernel void sigmoid_bf16_to_f32(
    constant uint &n      [[buffer(0)]],
    device const bfloat *x [[buffer(1)]],
    device float        *out [[buffer(2)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid >= n) {
        return;
    }
    float v = float(x[gid]);
    out[gid] = 1.0f / (1.0f + exp(-v));
}











kernel void silu_apply(
    constant uint &n         [[buffer(0)]],
    device const bfloat *x   [[buffer(1)]],
    device bfloat       *out [[buffer(2)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid >= n) {
        return;
    }
    float v = float(x[gid]);
    float sig = 1.0f / (1.0f + exp(-v));
    out[gid] = bfloat(v * sig);
}











kernel void bf16_mul(
    constant uint &n         [[buffer(0)]],
    device const bfloat *a   [[buffer(1)]],
    device const bfloat *b   [[buffer(2)]],
    device bfloat       *out [[buffer(3)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid >= n) {
        return;
    }
    out[gid] = bfloat(float(a[gid]) * float(b[gid]));
}
