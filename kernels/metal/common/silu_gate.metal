// SPDX-License-Identifier: MIT OR Apache-2.0

// 2026-09-25: SwiGLU activation, one thread per element, in FP32:
//
//   out[i] = gate[i] * sigmoid(gate[i]) * up[i]
//
// Owner: metal kernels. Invariants: none beyond the types.









#include <metal_stdlib>
using namespace metal;

kernel void silu_gate(
    constant uint &n           [[buffer(0)]],
    device const bfloat *gate  [[buffer(1)]],
    device const bfloat *up    [[buffer(2)]],
    device bfloat       *out   [[buffer(3)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid >= n) {
        return;
    }
    float g = float(gate[gid]);
    float u = float(up[gid]);


    float sig = 1.0f / (1.0f + exp(-g));
    out[gid] = bfloat(g * sig * u);
}
