// SPDX-License-Identifier: MIT OR Apache-2.0

// 2026-09-26: Split Qwen3.5's q_proj output into separate Q and gate buffers. The input
// is interleaved per head, `[Q_h0 (head_dim), gate_h0 (head_dim), Q_h1, gate_h1, ...]`;
// the outputs are two contiguous `[num_heads, head_dim]` buffers.
//
// Grid: (head_dim, num_heads, 1), one thread per (head, element).









#include <metal_stdlib>
using namespace metal;

kernel void qwen35_qkv_split(
    constant uint &num_heads [[buffer(0)]],
    constant uint &head_dim  [[buffer(1)]],
    device const bfloat *q_full [[buffer(2)]],
    device bfloat       *q_out  [[buffer(3)]],
    device bfloat       *gate_out [[buffer(4)]],
    uint2 gid [[thread_position_in_grid]])
{
    uint h = gid.y;
    uint d = gid.x;
    if (h >= num_heads || d >= head_dim) {
        return;
    }
    uint stride = 2u * head_dim;
    uint q_src    = h * stride + d;
    uint gate_src = h * stride + head_dim + d;
    uint dst      = h * head_dim + d;
    q_out[dst]    = q_full[q_src];
    gate_out[dst] = q_full[gate_src];
}
