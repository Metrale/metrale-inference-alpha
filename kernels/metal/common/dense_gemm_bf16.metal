// SPDX-License-Identifier: MIT OR Apache-2.0

// 2026-09-26: BF16 matrix multiply for unquantized weights, with FP32 accumulation:
//
//   y[m, n] = sum_k(x[m, k] * w[n, k])
//
// One thread per (m, n) output; the grid's x is n and its y is m.
//
// Layout:
//   x : bfloat [M, K]
//   w : bfloat [N, K]
//   y : bfloat [M, N]







#include <metal_stdlib>
using namespace metal;

kernel void dense_gemm_bf16(
    constant uint &M  [[buffer(0)]],
    constant uint &N  [[buffer(1)]],
    constant uint &K  [[buffer(2)]],
    device const bfloat *x [[buffer(3)]],
    device const bfloat *w [[buffer(4)]],
    device bfloat       *y [[buffer(5)]],
    uint2 gid [[thread_position_in_grid]])
{
    uint m = gid.y;
    uint n = gid.x;
    if (m >= M || n >= N) {
        return;
    }
    float acc = 0.0f;
    for (uint k = 0; k < K; ++k) {
        acc += float(x[m * K + k]) * float(w[n * K + k]);
    }
    y[m * N + n] = bfloat(acc);
}
