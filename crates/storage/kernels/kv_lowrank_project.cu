// SPDX-License-Identifier: MIT OR Apache-2.0

// 2026-09-25: kv_lowrank_project: K_lr[kv_head, tok, :] = K_block[tok, kv_head, :] @ P
// for one paged block. Every token keeps its own row; predictor_score takes
// the max over them.
//
// Owner: metrale-storage kernels.
// Invariants: none beyond the types.
//
// Shapes (BF16):
//   K_block : [block_size, num_kv_heads, head_dim], one paged block
//   P       : [head_dim, r]
//   K_lr    : [num_kv_heads, block_size, r], the per-block layout predictor_score reads
//
// Launch: grid = (num_kv_heads, block_size, 1), block = (r, 1, 1); one
// thread per output element.





#include <cuda_bf16.h>

extern "C" __global__ void kv_lowrank_project(
    const __nv_bfloat16* __restrict__ K_block,
    const __nv_bfloat16* __restrict__ P,
    __nv_bfloat16*       __restrict__ K_lr,
    int block_size,
    int num_kv_heads,
    int head_dim,
    int r
) {
    const int kv_head = blockIdx.x;
    const int tok     = blockIdx.y;
    const int out_idx = threadIdx.x;
    if (kv_head >= num_kv_heads || tok >= block_size || out_idx >= r) return;

    const __nv_bfloat16* k_row = K_block
        + (size_t)tok * (size_t)num_kv_heads * (size_t)head_dim
        + (size_t)kv_head * (size_t)head_dim;

    float acc = 0.0f;
    #pragma unroll 8
    for (int i = 0; i < head_dim; ++i) {
        float k_val = __bfloat162float(k_row[i]);
        float p_val = __bfloat162float(P[(size_t)i * r + out_idx]);
        acc = fmaf(k_val, p_val, acc);
    }
    K_lr[((size_t)kv_head * block_size + tok) * r + out_idx] = __float2bfloat16(acc);
}
