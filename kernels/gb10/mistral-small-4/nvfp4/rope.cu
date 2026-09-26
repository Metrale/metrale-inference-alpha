// SPDX-License-Identifier: MIT OR Apache-2.0
// forked-from: kernels/gb10/common/rope.cu (2026-09-24; 428 of 590 lines differ, see kernels/FORKS.md)

// 2026-09-25: RoPE for the mistral-small-4 tree, in place on Q and K. rope_forward and rope_forward_yarn
// rotate adjacent pairs (2i, 2i + 1) of the first rotary_dim values of each head by angle pos x freq_i;
// common/rope.cu's rope_forward rotates (i, i + rotary_dim / 2) instead.
//
// Owner: gb10 kernels (mistral-small-4).
// Invariants: none beyond the types.
//
// Q is [batch, seq_len, num_q_heads, head_dim] and K [batch, seq_len, num_kv_heads, head_dim], BF16;
// positions is [batch, seq_len] u32. blockIdx.x indexes the Q heads, then the K heads; blockIdx.y a group
// of 128 / (rotary_dim / 2) positions; blockIdx.z the batch. One thread per rotation pair in a 128-thread
// block, so rotary_dim / 2 must divide 128 (ops::rope and ops::rope_yarn compute the same grid with z = 1).












#include <cuda_bf16.h>

extern "C" __global__ void rope_forward(
    __nv_bfloat16* __restrict__ Q,
    __nv_bfloat16* __restrict__ K,
    const unsigned int* __restrict__ positions,
    const unsigned int seq_len,
    const unsigned int num_q_heads,
    const unsigned int num_kv_heads,
    const unsigned int head_dim,
    const unsigned int rotary_dim,
    const float* __restrict__ inv_freq,
    const float theta
) {
    const unsigned int head_idx = blockIdx.x;
    const unsigned int seq_block = blockIdx.y;
    const unsigned int batch = blockIdx.z;
    const unsigned int tid = threadIdx.x;


    const bool is_q = (head_idx < num_q_heads);
    const unsigned int head = is_q ? head_idx : (head_idx - num_q_heads);

    if (!is_q && head >= num_kv_heads) return;




    const unsigned int pairs_per_pos = rotary_dim / 2;
    const unsigned int pos_per_block = 128 / pairs_per_pos;



    const unsigned int local_pos = tid / pairs_per_pos;
    const unsigned int pair_idx = tid % pairs_per_pos;

    const unsigned int seq_pos = seq_block * pos_per_block + local_pos;
    if (seq_pos >= seq_len) return;


    const unsigned int abs_pos = positions[batch * seq_len + seq_pos];
    // 2026-09-25: freq_i = inv_freq[i], or theta^(-2i / rotary_dim) when inv_freq is null. ops::rope passes
    // no inv_freq: it packs nine arguments, and this kernel declares ten.
    const float freq = (inv_freq != 0) ? inv_freq[pair_idx]
        : (1.0f / powf(theta, (float)(2 * pair_idx) / (float)rotary_dim));
    const float angle = (float)abs_pos * freq;
    const float cos_val = cosf(angle);
    const float sin_val = sinf(angle);


    __nv_bfloat16* ptr;
    if (is_q) {
        ptr = Q + batch * seq_len * (num_q_heads * head_dim)
                + seq_pos * (num_q_heads * head_dim)
                + head * head_dim;
    } else {
        ptr = K + batch * seq_len * (num_kv_heads * head_dim)
                + seq_pos * (num_kv_heads * head_dim)
                + head * head_dim;
    }




    const unsigned int d0 = 2 * pair_idx;
    const unsigned int d1 = 2 * pair_idx + 1;
    float x0 = (float)ptr[d0];
    float x1 = (float)ptr[d1];


    float y0 = x0 * cos_val - x1 * sin_val;
    float y1 = x1 * cos_val + x0 * sin_val;


    ptr[d0] = __float2bfloat16(y0);
    ptr[d1] = __float2bfloat16(y1);
}






// 2026-09-25: rope_forward with freq_i always read from inv_freq [rotary_dim / 2]; theta is ignored.
// The MLA attention paths launch it with the layer's YaRN table (ops::rope_yarn).
extern "C" __global__ void rope_forward_yarn(
    __nv_bfloat16* __restrict__ Q,
    __nv_bfloat16* __restrict__ K,
    const unsigned int* __restrict__ positions,
    const unsigned int seq_len,
    const unsigned int num_q_heads,
    const unsigned int num_kv_heads,
    const unsigned int head_dim,
    const unsigned int rotary_dim,
    const float* __restrict__ inv_freq,
    const float theta
) {
    (void)theta;

    const unsigned int head_idx = blockIdx.x;
    const unsigned int seq_block = blockIdx.y;
    const unsigned int batch = blockIdx.z;
    const unsigned int tid = threadIdx.x;

    const bool is_q = (head_idx < num_q_heads);
    const unsigned int head = is_q ? head_idx : (head_idx - num_q_heads);

    if (!is_q && head >= num_kv_heads) return;

    const unsigned int pairs_per_pos = rotary_dim / 2;
    const unsigned int pos_per_block = 128 / pairs_per_pos;

    const unsigned int local_pos = tid / pairs_per_pos;
    const unsigned int pair_idx = tid % pairs_per_pos;

    const unsigned int seq_pos = seq_block * pos_per_block + local_pos;
    if (seq_pos >= seq_len) return;

    const unsigned int abs_pos = positions[batch * seq_len + seq_pos];


    const float freq = inv_freq[pair_idx];
    const float angle = (float)abs_pos * freq;
    const float cos_val = cosf(angle);
    const float sin_val = sinf(angle);

    __nv_bfloat16* ptr;
    if (is_q) {
        ptr = Q + batch * seq_len * (num_q_heads * head_dim)
                + seq_pos * (num_q_heads * head_dim)
                + head * head_dim;
    } else {
        ptr = K + batch * seq_len * (num_kv_heads * head_dim)
                + seq_pos * (num_kv_heads * head_dim)
                + head * head_dim;
    }






    const unsigned int d0 = 2 * pair_idx;
    const unsigned int d1 = 2 * pair_idx + 1;
    float x0 = (float)ptr[d0];
    float x1 = (float)ptr[d1];

    float y0 = x0 * cos_val - x1 * sin_val;
    float y1 = x1 * cos_val + x0 * sin_val;

    ptr[d0] = __float2bfloat16(y0);
    ptr[d1] = __float2bfloat16(y1);
}
