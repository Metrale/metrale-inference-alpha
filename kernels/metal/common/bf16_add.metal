// SPDX-License-Identifier: MIT OR Apache-2.0

// 2026-09-26: Element-wise BF16 addition, `out[i] = a[i] + b[i]` for i < n, added in FP32
// and rounded once to BF16.
//
// Layout:
//   a, b, out : bfloat [n]








#include <metal_stdlib>
using namespace metal;

kernel void bf16_add(
    constant uint &n         [[buffer(0)]],
    device const bfloat *a   [[buffer(1)]],
    device const bfloat *b   [[buffer(2)]],
    device bfloat       *out [[buffer(3)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid >= n) {
        return;
    }
    out[gid] = bfloat(float(a[gid]) + float(b[gid]));
}
