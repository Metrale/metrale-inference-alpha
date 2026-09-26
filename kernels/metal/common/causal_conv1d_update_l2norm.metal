// SPDX-License-Identifier: MIT OR Apache-2.0

// 2026-09-25: Fused causal-conv1d update + SiLU + per-head L2 norm. For each
// channel ch of batch row b:
//   1. shift conv_state[b][ch] left by one and append new_input[b][ch];
//   2. acc = sum_k conv_state[b][ch][k] * weight[ch][k] (no bias);
//   3. y = acc * sigmoid(acc);
//   4. in a threadgroup that starts below qk_channels, scale y by
//      rsqrt(sum of y^2 over its head_dim-channel head + l2_eps);
//   5. output[b][ch] = bf16(y).
//
// Layout:
//   conv_state : float  [batch, dim, d_conv]                (in/out)
//   new_input  : bfloat [batch, dim]
//   weight     : bfloat [dim, d_conv]
//   output     : bfloat [batch, dim]
//
// Grid: flat 1-D, ceil(dim / tg_size) * batch threadgroups.
//
// Owner: metal kernels.
// Invariants: assumes tg_size is a multiple of head_dim, head_dim is a
// multiple of 32, tg_size <= min(4 * head_dim, 512) (`head_inv_norm`,
// `partial`), and qk_channels is a multiple of tg_size, so that no head
// straddles a threadgroup and no threadgroup mixes Q/K and V channels.










#include <metal_stdlib>
using namespace metal;

constant uint MAX_HEADS_PER_BLOCK = 4;
constant uint MAX_SIMDGROUPS_LN = 16;

kernel void causal_conv1d_update_l2norm(
    device float        *conv_state [[buffer(0)]],
    device const bfloat *new_input  [[buffer(1)]],
    device const bfloat *weight     [[buffer(2)]],
    device bfloat       *output     [[buffer(3)]],
    constant uint  &batch        [[buffer(4)]],
    constant uint  &dim          [[buffer(5)]],
    constant uint  &d_conv       [[buffer(6)]],
    constant uint  &qk_channels  [[buffer(7)]],
    constant uint  &head_dim     [[buffer(8)]],
    constant float &l2_eps       [[buffer(9)]],
    uint  tg_idx        [[threadgroup_position_in_grid]],
    uint  tid           [[thread_position_in_threadgroup]],
    uint  tg_size       [[threads_per_threadgroup]],
    uint  simd_lane     [[thread_index_in_simdgroup]],
    uint  simd_grp      [[simdgroup_index_in_threadgroup]])
{


    uint blocks_per_batch = (dim + tg_size - 1) / tg_size;
    uint block_x_idx = tg_idx % blocks_per_batch;
    uint b           = tg_idx / blocks_per_batch;

    uint block_start = block_x_idx * tg_size;
    uint ch = block_start + tid;
    bool block_needs_l2 = (block_start < qk_channels);



    threadgroup float partial[MAX_SIMDGROUPS_LN];
    threadgroup float head_inv_norm[MAX_HEADS_PER_BLOCK];

    bool valid = (ch < dim && b < batch);
    float silu = 0.0f;


    if (valid) {
        device float *state = conv_state + (b * dim + ch) * d_conv;
        for (uint i = 0; i + 1 < d_conv; ++i) {
            state[i] = state[i + 1];
        }
        state[d_conv - 1] = float(new_input[b * dim + ch]);

        device const bfloat *w = weight + ch * d_conv;
        float acc = 0.0f;
        for (uint k = 0; k < d_conv; ++k) {
            acc += state[k] * float(w[k]);
        }
        float sig = 1.0f / (1.0f + exp(-acc));
        silu = acc * sig;
    }




    if (block_needs_l2) {
        float sq = valid ? (silu * silu) : 0.0f;


        float simd_sum_v = simd_sum(sq);
        if (simd_lane == 0) {
            partial[simd_grp] = simd_sum_v;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);



        uint simds_per_head = head_dim / 32u;
        if (simds_per_head == 0u) simds_per_head = 1u;

        uint head_in_block = tid / head_dim;
        uint pos_in_head = tid % head_dim;
        if (pos_in_head == 0 && head_in_block < MAX_HEADS_PER_BLOCK) {
            float total = 0.0f;
            uint base_simd = head_in_block * simds_per_head;
            for (uint i = 0; i < simds_per_head; ++i) {
                total += partial[base_simd + i];
            }
            head_inv_norm[head_in_block] = rsqrt(total + l2_eps);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        if (valid) {
            silu *= head_inv_norm[head_in_block];
        }
    }

    if (valid) {
        output[b * dim + ch] = bfloat(silu);
    }
}
