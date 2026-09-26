// SPDX-License-Identifier: MIT OR Apache-2.0

// 2026-09-26: Sigmoid-gated element-wise multiply, `out[i] = sigmoid(gate[i]) * x[i]`,
// computed in FP32.
//
// Layout:
//   gate, x, out : bfloat [n]





#include <metal_stdlib>
using namespace metal;

kernel void sigmoid_gate(
    constant uint &n         [[buffer(0)]],
    device const bfloat *gate [[buffer(1)]],
    device const bfloat *x   [[buffer(2)]],
    device bfloat       *out [[buffer(3)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid >= n) {
        return;
    }
    float g = float(gate[gid]);
    float v = float(x[gid]);
    float sig = 1.0f / (1.0f + exp(-g));
    out[gid] = bfloat(sig * v);
}
