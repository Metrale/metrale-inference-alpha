// SPDX-License-Identifier: MIT OR Apache-2.0

// 2026-09-25: Gated delta rule decode step, one threadgroup per (batch row,
// value head) on a flat 1-D grid of num_v_heads * batch_size threadgroups,
// one thread per value column c, with kh = vh / (num_v_heads / num_k_heads):
//
//   hk[c]    = sum_j H[j, c] * k[j]
//   v_new[c] = (v[c] - g * hk[c]) * beta
//   H[j, c]  = g * H[j, c] + k[j] * v_new[c]
//   y[c]     = (sum_j H[j, c] * q[j]) / sqrt(k_dim)
//
// g is the gate clamped to [1e-6, 1 - 1e-6]. After the update, a head whose
// state Frobenius norm exceeds SSM_STATE_MAX_NORM is scaled down to it; y is
// computed before that scaling. The arguments, update and gate clamp match
// gated_delta_rule_decode in kernels/gb10/common/gated_delta_rule.cu.
//
// Layout:
//   h_state : float  [batch, num_v_heads, k_dim, v_dim]   (in/out)
//   query   : bfloat [batch, num_k_heads,   k_dim]
//   key     : bfloat [batch, num_k_heads,   k_dim]
//   value   : bfloat [batch, num_v_heads,   v_dim]
//   gate    : float  [batch, num_v_heads]
//   beta    : float  [batch, num_v_heads]
//   output  : bfloat [batch, num_v_heads,   v_dim]
//
// Owner: metal kernels.
// Invariants: assumes threadgroup size == v_dim <= 128 and k_dim <= 128, a
// multiple of 4 (`smem_k`, `smem_q`, `norm_sums`).

#include <metal_stdlib>
using namespace metal;




constant float SSM_STATE_MAX_NORM = 1000.0f;

kernel void gated_delta_rule_decode(
    device float        *h_state    [[buffer(0)]],
    device const bfloat *query      [[buffer(1)]],
    device const bfloat *key        [[buffer(2)]],
    device const bfloat *value      [[buffer(3)]],
    device const float  *gate       [[buffer(4)]],
    device const float  *beta       [[buffer(5)]],
    device bfloat       *output     [[buffer(6)]],
    constant uint &batch_size       [[buffer(7)]],
    constant uint &num_k_heads      [[buffer(8)]],
    constant uint &num_v_heads      [[buffer(9)]],
    constant uint &k_dim            [[buffer(10)]],
    constant uint &v_dim            [[buffer(11)]],
    uint  tg_idx    [[threadgroup_position_in_grid]],
    uint  tid       [[thread_position_in_threadgroup]],
    uint  simd_lane [[thread_index_in_simdgroup]],
    uint  simd_grp  [[simdgroup_index_in_threadgroup]])
{



    const uint vh = tg_idx % num_v_heads;
    const uint b  = tg_idx / num_v_heads;
    if (vh >= num_v_heads || b >= batch_size) {
        return;
    }
    const uint head_repeat = num_v_heads / num_k_heads;
    const uint kh = vh / head_repeat;


    device float *H = h_state + ((b * num_v_heads + vh) * k_dim * v_dim);
    device const bfloat *q_ptr = query + (b * num_k_heads + kh) * k_dim;
    device const bfloat *k_ptr = key   + (b * num_k_heads + kh) * k_dim;
    device const bfloat *v_ptr = value + (b * num_v_heads + vh) * v_dim;


    float g_raw = gate[b * num_v_heads + vh];
    const float g  = fmin(fmax(g_raw, 1e-6f), 1.0f - 1e-6f);
    const float bt = beta[b * num_v_heads + vh];



    threadgroup float smem_k[128];
    threadgroup float smem_q[128];
    if (tid < k_dim) {
        smem_k[tid] = float(k_ptr[tid]);
        smem_q[tid] = float(q_ptr[tid]);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (tid >= v_dim) {
        return;
    }
    float v_i = float(v_ptr[tid]);


    float hk_dot = 0.0f;
    for (uint j = 0; j < k_dim; j += 4) {
        float h0 = H[(j + 0) * v_dim + tid];
        float h1 = H[(j + 1) * v_dim + tid];
        float h2 = H[(j + 2) * v_dim + tid];
        float h3 = H[(j + 3) * v_dim + tid];
        hk_dot += h0 * smem_k[j] + h1 * smem_k[j + 1]
                + h2 * smem_k[j + 2] + h3 * smem_k[j + 3];
    }



    float v_new_i = (v_i - g * hk_dot) * bt;


    float q_dot = 0.0f;
    for (uint j = 0; j < k_dim; j += 4) {
        float h0 = H[(j + 0) * v_dim + tid];
        float h1 = H[(j + 1) * v_dim + tid];
        float h2 = H[(j + 2) * v_dim + tid];
        float h3 = H[(j + 3) * v_dim + tid];
        h0 = g * h0 + smem_k[j]     * v_new_i;
        h1 = g * h1 + smem_k[j + 1] * v_new_i;
        h2 = g * h2 + smem_k[j + 2] * v_new_i;
        h3 = g * h3 + smem_k[j + 3] * v_new_i;
        H[(j + 0) * v_dim + tid] = h0;
        H[(j + 1) * v_dim + tid] = h1;
        H[(j + 2) * v_dim + tid] = h2;
        H[(j + 3) * v_dim + tid] = h3;
        q_dot += h0 * smem_q[j] + h1 * smem_q[j + 1]
               + h2 * smem_q[j + 2] + h3 * smem_q[j + 3];
    }




    {
        float local_sq = 0.0f;
        for (uint j = 0; j < k_dim; ++j) {
            float hv = H[j * v_dim + tid];
            local_sq += hv * hv;
        }

        float warp_sum = simd_sum(local_sq);
        threadgroup float norm_sums[4];
        if (simd_lane == 0) {
            norm_sums[simd_grp] = warp_sum;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        threadgroup float head_norm_sq_storage;
        if (simd_grp == 0) {

            float s = (tid < 4u) ? norm_sums[tid] : 0.0f;
            s = simd_sum(s);
            if (tid == 0) {
                head_norm_sq_storage = s;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        float head_norm_sq = head_norm_sq_storage;
        if (head_norm_sq > SSM_STATE_MAX_NORM * SSM_STATE_MAX_NORM) {
            float scale = SSM_STATE_MAX_NORM * rsqrt(head_norm_sq);
            for (uint j = 0; j < k_dim; ++j) {
                H[j * v_dim + tid] *= scale;
            }
        }
    }


    float inv_sqrt_d = rsqrt(float(k_dim));
    output[(b * num_v_heads + vh) * v_dim + tid] = bfloat(q_dot * inv_sqrt_d);
}
