// SPDX-License-Identifier: MIT OR Apache-2.0

// 2026-09-25: Interleaved multimodal RoPE (MRoPE), in place on BF16 Q and K: rotary pair i
// takes its position from stream i % 3 (0: pos_t, 1: pos_h, 2: pos_w) and is rotated
// rotate-half style, dims (i, i + rotary_dim/2), with freq_i = 1 / theta^(2i / rotary_dim)
// evaluated in FP64. The _k_only kernel rotates K alone, for the prefill where
// deinterleave_qg_split_qnorm_mrope (ssm_preprocess.cu) already rotated Q.
//
// Owner: gb10 kernels.
// Invariants: none beyond the types.
//
// The kernels read no mrope_section: i % 3 alone gives 11/11/10 of the 32 pairs at
// rotary_dim 64, the mrope_section [11, 11, 10] with head_dim 256 and
// partial_rotary_factor 0.25 in kernels/gb10/qwen3.6-35b-a3b/MODEL.toml.
//
// With pos_t == pos_h == pos_w the arithmetic is that of rope_forward (rope.cu): the same
// FP64 freq expression and the same rotation, so the output is bit-identical.
//
// Q is [batch, seq_len, num_q_heads, head_dim], K [batch, seq_len, num_kv_heads, head_dim],
// positions [batch, seq_len] u32. Grid (num_q_heads + num_kv_heads, ceil(seq_len /
// pos_per_block), batch) with pos_per_block = 128 / (rotary_dim / 2), block 128; the
// launchers (model-layers ops/embeddings.rs) pass batch 1. rotary_dim must be at most 256:
// above that pos_per_block is 0 and the kernel returns without writing.








#include <cuda_bf16.h>

extern "C" __global__ void rope_forward_mrope_interleaved(
    __nv_bfloat16* __restrict__ Q,
    __nv_bfloat16* __restrict__ K,
    const unsigned int* __restrict__ pos_t,
    const unsigned int* __restrict__ pos_h,
    const unsigned int* __restrict__ pos_w,
    const unsigned int seq_len,
    const unsigned int num_q_heads,
    const unsigned int num_kv_heads,
    const unsigned int head_dim,
    const unsigned int rotary_dim,
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
    if (pos_per_block == 0) return;

    const unsigned int local_pos = tid / pairs_per_pos;
    const unsigned int pair_idx = tid % pairs_per_pos;

    const unsigned int seq_pos = seq_block * pos_per_block + local_pos;
    if (seq_pos >= seq_len) return;
    if (local_pos >= pos_per_block) return;


    const unsigned int section = pair_idx % 3;
    const unsigned int tok_idx = batch * seq_len + seq_pos;
    unsigned int abs_pos;
    if (section == 0) abs_pos = pos_t[tok_idx];
    else if (section == 1) abs_pos = pos_h[tok_idx];
    else                   abs_pos = pos_w[tok_idx];


    const double freq_exp_d = (double)(2 * pair_idx) / (double)rotary_dim;
    const float freq = (float)(1.0 / pow((double)theta, freq_exp_d));
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


    const unsigned int half_rot = rotary_dim / 2;
    const unsigned int d0 = pair_idx;
    const unsigned int d1 = pair_idx + half_rot;
    const float x0 = (float)ptr[d0];
    const float x1 = (float)ptr[d1];

    const float y0 = x0 * cos_val - x1 * sin_val;
    const float y1 = x1 * cos_val + x0 * sin_val;

    ptr[d0] = __float2bfloat16(y0);
    ptr[d1] = __float2bfloat16(y1);
}

extern "C" __global__ void rope_forward_mrope_interleaved_k_only(
    __nv_bfloat16* __restrict__ K,
    const unsigned int* __restrict__ pos_t,
    const unsigned int* __restrict__ pos_h,
    const unsigned int* __restrict__ pos_w,
    const unsigned int seq_len,
    const unsigned int num_kv_heads,
    const unsigned int head_dim,
    const unsigned int rotary_dim,
    const float theta
) {
    const unsigned int head = blockIdx.x;
    const unsigned int seq_block = blockIdx.y;
    const unsigned int batch = blockIdx.z;
    const unsigned int tid = threadIdx.x;
    if (head >= num_kv_heads) return;

    const unsigned int pairs_per_pos = rotary_dim / 2;
    const unsigned int pos_per_block = 128 / pairs_per_pos;
    if (pos_per_block == 0) return;

    const unsigned int local_pos = tid / pairs_per_pos;
    const unsigned int pair_idx = tid % pairs_per_pos;
    const unsigned int seq_pos = seq_block * pos_per_block + local_pos;
    if (seq_pos >= seq_len) return;
    if (local_pos >= pos_per_block) return;

    const unsigned int section = pair_idx % 3;
    const unsigned int tok_idx = batch * seq_len + seq_pos;
    unsigned int abs_pos;
    if (section == 0) abs_pos = pos_t[tok_idx];
    else if (section == 1) abs_pos = pos_h[tok_idx];
    else                   abs_pos = pos_w[tok_idx];

    const double freq_exp_d = (double)(2 * pair_idx) / (double)rotary_dim;
    const float freq = (float)(1.0 / pow((double)theta, freq_exp_d));
    const float angle = (float)abs_pos * freq;
    const float cos_val = cosf(angle);
    const float sin_val = sinf(angle);

    __nv_bfloat16* ptr = K + batch * seq_len * (num_kv_heads * head_dim)
        + seq_pos * (num_kv_heads * head_dim)
        + head * head_dim;

    const unsigned int half_rot = rotary_dim / 2;
    const unsigned int d0 = pair_idx;
    const unsigned int d1 = pair_idx + half_rot;
    const float x0 = (float)ptr[d0];
    const float x1 = (float)ptr[d1];

    ptr[d0] = __float2bfloat16(x0 * cos_val - x1 * sin_val);
    ptr[d1] = __float2bfloat16(x1 * cos_val + x0 * sin_val);
}
