// SPDX-License-Identifier: MIT OR Apache-2.0

// 2026-09-25: q_lowrank_project: Q_proj = Q @ P for the high-speed-swap
// predictor.
//
// Owner: metrale-storage kernels.
// Invariants: none beyond the types.
//
// Shapes (BF16):
//   Q     : [num_q_heads, head_dim]
//   P     : [head_dim, r]
//   Q_proj: [num_q_heads, r]
//
// Launch: grid = (num_q_heads, 1, 1), block = (r, 1, 1); one thread per
// output element.




#include <cuda_bf16.h>

extern "C" __global__ void q_lowrank_project(
    const __nv_bfloat16* __restrict__ Q,
    const __nv_bfloat16* __restrict__ P,
    __nv_bfloat16*       __restrict__ Q_proj,
    int num_q_heads,
    int head_dim,
    int r
) {
    const int q_head  = blockIdx.x;
    const int out_idx = threadIdx.x;
    if (q_head >= num_q_heads || out_idx >= r) return;

    const __nv_bfloat16* q_row = Q + (size_t)q_head * head_dim;
    float acc = 0.0f;
    #pragma unroll 8
    for (int i = 0; i < head_dim; ++i) {
        float q_val = __bfloat162float(q_row[i]);
        float p_val = __bfloat162float(P[(size_t)i * r + out_idx]);
        acc = fmaf(q_val, p_val, acc);
    }
    Q_proj[(size_t)q_head * r + out_idx] = __float2bfloat16(acc);
}
