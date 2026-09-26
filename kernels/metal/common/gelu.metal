// SPDX-License-Identifier: MIT OR Apache-2.0

// 2026-09-26: GeLU, tanh approximation, computed in FP32:
//
//   gelu(x) = 0.5 * x * (1 + tanh( sqrt(2/π) * (x + 0.044715 * x³) ))
//
// Layout:
//   x, out : bfloat [n]    (in place is safe: out may be x)







#include <metal_stdlib>
using namespace metal;

constant float SQRT_2_OVER_PI = 0.7978845608028654f;
constant float GELU_C        = 0.044715f;

kernel void gelu(
    constant uint &n         [[buffer(0)]],
    device const bfloat *x   [[buffer(1)]],
    device bfloat       *out [[buffer(2)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid >= n) {
        return;
    }
    float v = float(x[gid]);
    float v3 = v * v * v;
    float arg = SQRT_2_OVER_PI * (v + GELU_C * v3);
    float t = tanh(arg);
    out[gid] = bfloat(0.5f * v * (1.0f + t));
}
