// SPDX-License-Identifier: MIT OR Apache-2.0

// 2026-09-25: Token-sequential GDN prefill with the state held in registers
// (gated_delta_rule_prefill_regresident).
//
// One warp owns one value column col; lane l holds rows l, l + 32, l + 64 and l + 96 of it. The
// column is read from global memory once, all seq_len tokens run on registers with warp shuffles
// for the two sums, and the column is written back once. There is no shared memory and no block
// barrier. Per token:
//   kv   = sum_i H[i][col] * k[i]
//   vnew = (v[col] - g * kv) * beta
//   H[i][col] = g * H[i][col] + k[i] * vnew
//   out[col]  = (sum_i H[i][col] * q[i]) * rsqrt(k_dim)
// The sums are warp butterflies, so their order differs from gated_delta_rule_decode's per-thread
// loops.
//
// There is no SSM_STATE_NORM_ENABLED clamp: a head's columns are spread over grid.z blocks, and no
// block sees the whole head's norm.
//
// The Qwen3 SSM prefill recurrence launches it when the gdn_regresident lever is on, kd == vd == 128
// and the FLA chunked path was not taken, ahead of gated_delta_rule_prefill_persistent_wy4.
//
// Owner: gb10 kernels.
// Invariants:
// - Launch: grid (num_v_heads, batch_size, v_dim / GDN_RR_WARPS_PER_BLOCK) and
//   32 * GDN_RR_WARPS_PER_BLOCK threads (ops::gdn_prefill_regresident: v_dim / 4, 128 threads).
// - k_dim == 128.
// - batch_size == 1: token t is read at row t whatever b is. The only caller passes 1.
// - qk_stride, v_stride and gb_stride count elements between consecutive tokens.







#include <cuda_bf16.h>

#ifndef GDN_RR_WARPS_PER_BLOCK
#define GDN_RR_WARPS_PER_BLOCK 4
#endif




extern "C" __global__ void __launch_bounds__(32 * GDN_RR_WARPS_PER_BLOCK, 4)
gated_delta_rule_prefill_regresident(
    float* __restrict__ h_state,
    const __nv_bfloat16* __restrict__ query,
    const __nv_bfloat16* __restrict__ key,
    const __nv_bfloat16* __restrict__ value,
    const float* __restrict__ gate,
    const float* __restrict__ beta,
    __nv_bfloat16* __restrict__ output,
    unsigned int batch_size,
    unsigned int seq_len,
    unsigned int num_k_heads,
    unsigned int num_v_heads,
    unsigned int k_dim,
    unsigned int v_dim,
    unsigned int qk_stride,
    unsigned int v_stride,
    unsigned int gb_stride
) {
    const unsigned int vh = blockIdx.x;
    const unsigned int b  = blockIdx.y;
    if (vh >= num_v_heads || b >= batch_size) return;

    const unsigned int warp_id = threadIdx.x >> 5;
    const unsigned int lane    = threadIdx.x & 31;
    const unsigned int col     = blockIdx.z * GDN_RR_WARPS_PER_BLOCK + warp_id;
    if (col >= v_dim) return;

    const unsigned int head_repeat = num_v_heads / num_k_heads;
    const unsigned int kh = vh / head_repeat;


    const unsigned int r0 = lane, r1 = lane + 32, r2 = lane + 64, r3 = lane + 96;

    float* H = h_state + ((unsigned long long)(b * num_v_heads + vh) * k_dim * v_dim);


    float s0 = H[r0 * v_dim + col];
    float s1 = H[r1 * v_dim + col];
    float s2 = H[r2 * v_dim + col];
    float s3 = H[r3 * v_dim + col];

    const float inv_sqrt_d = rsqrtf((float)k_dim);

    for (unsigned int t = 0; t < seq_len; t++) {
        const __nv_bfloat16* k_t = key   + (unsigned long long)t * qk_stride + kh * k_dim;
        const __nv_bfloat16* q_t = query + (unsigned long long)t * qk_stride + kh * k_dim;

        // 2026-09-25: Each warp reads its own K and Q values from global memory, so no block
        // barrier is needed per token.
        float k0 = (float)k_t[r0], k1 = (float)k_t[r1], k2 = (float)k_t[r2], k3 = (float)k_t[r3];
        float q0 = (float)q_t[r0], q1 = (float)q_t[r1], q2 = (float)q_t[r2], q3 = (float)q_t[r3];

        float g_raw = gate[(unsigned long long)t * gb_stride + vh];
        const float g  = fminf(fmaxf(g_raw, 1e-6f), 1.0f - 1e-6f);   // 2026-09-25: same clamp as gated_delta_rule_decode.
        const float bt = beta[(unsigned long long)t * gb_stride + vh];
        float v_col = (float)value[(unsigned long long)t * v_stride + vh * v_dim + col];


        float kv = s0 * k0 + s1 * k1 + s2 * k2 + s3 * k3;
        #pragma unroll
        for (int off = 16; off > 0; off >>= 1) kv += __shfl_xor_sync(0xFFFFFFFFu, kv, off);

        float v_new = (v_col - g * kv) * bt;


        s0 = g * s0 + k0 * v_new;
        s1 = g * s1 + k1 * v_new;
        s2 = g * s2 + k2 * v_new;
        s3 = g * s3 + k3 * v_new;

        float q_dot = s0 * q0 + s1 * q1 + s2 * q2 + s3 * q3;
        #pragma unroll
        for (int off = 16; off > 0; off >>= 1) q_dot += __shfl_xor_sync(0xFFFFFFFFu, q_dot, off);

        if (lane == 0) {
            output[((unsigned long long)(b * seq_len + t) * num_v_heads + vh) * v_dim + col] =
                __float2bfloat16(q_dot * inv_sqrt_d);
        }
    }


    H[r0 * v_dim + col] = s0;
    H[r1 * v_dim + col] = s1;
    H[r2 * v_dim + col] = s2;
    H[r3 * v_dim + col] = s3;
}
