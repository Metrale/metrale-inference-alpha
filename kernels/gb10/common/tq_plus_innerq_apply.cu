// SPDX-License-Identifier: MIT OR Apache-2.0

// 2026-09-25: InnerQ kernels: tq_plus_innerq_apply_q multiplies Q by d_innerq_scale_inv and
// tq_plus_innerq_apply_k multiplies K by d_innerq_scale, both only while d_innerq_active is
// nonzero and only for head_dim 128. While d_innerq_calibrating is 1, apply_k first adds
// block 0's squared values into d_innerq_sq_accum and 1 into d_innerq_count.
//
// Owner: gb10 kernels.
// Invariants: none beyond the types.
//
// apply_q runs on Q before the WHT (qwen3_attention decode/attention_forward.rs) and apply_k
// on K after it (decode/write_kv_cache.rs).

#include <cuda_bf16.h>
#include "tq_plus_innerq.cuh"

namespace tq_plus {

// 2026-09-25: The InnerQ globals declared in tq_plus_innerq.cuh. The scales start at 1.0;
// innerq_driver.rs writes them through the registry's device_symbol.




__device__ float d_innerq_scale[INNERQ_MAX_CHANNELS] = {
    1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f,
    1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f,
    1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f,
    1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f,
    1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f,
    1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f,
    1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f,
    1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f,
    1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f,
    1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f,
    1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f,
    1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f,
    1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f,
    1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f,
    1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f,
    1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f,
};
__device__ float d_innerq_scale_inv[INNERQ_MAX_CHANNELS] = {
    1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f,
    1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f,
    1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f,
    1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f,
    1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f,
    1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f,
    1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f,
    1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f,
    1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f,
    1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f,
    1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f,
    1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f,
    1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f,
    1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f,
    1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f,
    1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f,
};
__device__ float d_innerq_sq_accum[INNERQ_MAX_CHANNELS] = {0};
__device__ int   d_innerq_count = 0;
__device__ int   d_innerq_active = 0;
__device__ int   d_innerq_calibrating = 0;

}

// 2026-09-25: One warp per head, channels lane * 4 + i; attention_forward.rs launches grid
// num_q_heads, block 32.

extern "C" __global__ void tq_plus_innerq_apply_q(
    __nv_bfloat16* __restrict__ data,
    const unsigned int head_dim
) {
    if (tq_plus::d_innerq_active == 0) return;
    if (head_dim != 128) return;

    const unsigned int head = blockIdx.x;
    const unsigned int lane = threadIdx.x;
    if (lane >= 32) return;

    __nv_bfloat16* head_data = data + (unsigned long long)head * head_dim;


    #pragma unroll
    for (unsigned int i = 0; i < 4; i++) {
        unsigned int ch = lane * 4 + i;
        float v = __bfloat162float(head_data[ch]);
        v *= tq_plus::d_innerq_scale_inv[ch];
        head_data[ch] = __float2bfloat16(v);
    }
}

// 2026-09-25: write_kv_cache.rs launches one 32-thread block per (token, kv_head) row, grid
// num_kv_heads * num_tokens.

extern "C" __global__ void tq_plus_innerq_apply_k(
    __nv_bfloat16* __restrict__ data,
    const unsigned int head_dim
) {
    if (head_dim != 128) return;

    const unsigned int head = blockIdx.x;
    const unsigned int lane = threadIdx.x;
    if (lane >= 32) return;

    __nv_bfloat16* head_data = data + (unsigned long long)head * head_dim;

    // 2026-09-25: Only block 0 accumulates, so each launch adds the squares of one row (the
    // first token's first kv head) and adds 1 to d_innerq_count.

    if (tq_plus::d_innerq_calibrating == 1 && head == 0) {
        #pragma unroll
        for (unsigned int i = 0; i < 4; i++) {
            unsigned int ch = lane * 4 + i;
            float v = __bfloat162float(head_data[ch]);
            atomicAdd(&tq_plus::d_innerq_sq_accum[ch], v * v);
        }
        if (lane == 0) {
            atomicAdd(&tq_plus::d_innerq_count, 1);
        }
    }

    if (tq_plus::d_innerq_active == 0) return;


    #pragma unroll
    for (unsigned int i = 0; i < 4; i++) {
        unsigned int ch = lane * 4 + i;
        float v = __bfloat162float(head_data[ch]);
        v *= tq_plus::d_innerq_scale[ch];
        head_data[ch] = __float2bfloat16(v);
    }
}
