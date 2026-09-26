// SPDX-License-Identifier: MIT OR Apache-2.0

// 2026-09-25: Rotary position embedding, applied in place to BF16 Q and K with FP32 math.
//
// Q is [batch, seq_len, num_q_heads, head_dim] and K is [batch, seq_len, num_kv_heads, head_dim]
// (rope_forward_strided takes the row strides instead); positions is u32 [batch, seq_len]. Grid
// (num_q_heads + num_kv_heads, seq_blocks, batch), block (128, 1, 1): blocks below num_q_heads
// rotate a Q head, the rest a K head. Each thread rotates one pair at one position, so a block
// covers 128 / pairs positions; the launchers in crates/model-layers/src/layers/ops/embeddings.rs
// size seq_blocks from that.
// - rope_forward, rope_forward_strided: pairs (i, i + rotary_dim/2) for i < rotary_dim/2, with
//   freq_i = theta^(-2i / rotary_dim). Channels past rotary_dim are not touched.
// - rope_forward_proportional: pairs (i, i + head_dim/2) for i < rope_angles, freq_i =
//   theta^(-2i / head_dim).
// - rope_forward_yarn, rope_forward_yarn_scaled: pairs as rope_forward, freq_i = inv_freq[i]
//   from the caller's table.
// - rope_forward_yarn_interleaved and _inv: adjacent pairs (2i, 2i + 1), freq_i = inv_freq[i].
//
// Owner: gb10 kernels.
// Invariants:
// - Every kernel except rope_forward_proportional assumes rotary_dim / 2 divides 128. Otherwise
//   the threads past a block's last whole position also rotate part of the next block's first
//   position, so that position is rotated twice.



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





    // 2026-09-25: Each distinct freq is computed once per block, in FP64, and shared through s_freq.
    // This comes before the position bounds return so that every thread reaches __syncthreads().
    __shared__ float s_freq[128];
    if (tid < pairs_per_pos) {
        const double fe = (double)(2 * tid) / (double)rotary_dim;
        s_freq[tid] = (float)(1.0 / pow((double)theta, fe));
    }
    __syncthreads();

    const unsigned int seq_pos = seq_block * pos_per_block + local_pos;
    if (seq_pos >= seq_len) return;


    const unsigned int abs_pos = positions[batch * seq_len + seq_pos];




    const float freq = s_freq[pair_idx];
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
    float x0 = (float)ptr[d0];
    float x1 = (float)ptr[d1];


    float y0 = x0 * cos_val - x1 * sin_val;
    float y1 = x1 * cos_val + x0 * sin_val;


    ptr[d0] = __float2bfloat16(y0);
    ptr[d1] = __float2bfloat16(y1);
}


























// 2026-09-25: rope_forward with explicit row strides, the elements between consecutive tokens of Q
// and of K. The multi-seq decode path rotates all n sequences of its interleaved QKV buffer in one
// launch with it (crates/model-layers/src/layers/qwen3_attention/trait_impl/multi_seq/attn.rs).
// With the packed strides, num_q_heads * head_dim and num_kv_heads * head_dim, it computes
// exactly what rope_forward does.
extern "C" __global__ void rope_forward_strided(
    __nv_bfloat16* __restrict__ Q,
    __nv_bfloat16* __restrict__ K,
    const unsigned int* __restrict__ positions,
    const unsigned int seq_len,
    const unsigned int num_q_heads,
    const unsigned int num_kv_heads,
    const unsigned int head_dim,
    const unsigned int rotary_dim,
    const float theta,





    const unsigned int q_row_stride,
    const unsigned int k_row_stride
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






    // 2026-09-25: Computed once per block before the bounds return, as in rope_forward.
    __shared__ float s_freq[128];
    if (tid < pairs_per_pos) {
        const double fe = (double)(2 * tid) / (double)rotary_dim;
        s_freq[tid] = (float)(1.0 / pow((double)theta, fe));
    }
    __syncthreads();

    const unsigned int seq_pos = seq_block * pos_per_block + local_pos;
    if (seq_pos >= seq_len) return;


    const unsigned int abs_pos = positions[batch * seq_len + seq_pos];




    const float freq = s_freq[pair_idx];
    const float angle = (float)abs_pos * freq;
    const float cos_val = cosf(angle);
    const float sin_val = sinf(angle);


    __nv_bfloat16* ptr;
    if (is_q) {
        ptr = Q + (unsigned long long)(batch * seq_len + seq_pos) * q_row_stride
                + head * head_dim;
    } else {
        ptr = K + (unsigned long long)(batch * seq_len + seq_pos) * k_row_stride
                + head * head_dim;
    }


    const unsigned int half_rot = rotary_dim / 2;
    const unsigned int d0 = pair_idx;
    const unsigned int d1 = pair_idx + half_rot;
    float x0 = (float)ptr[d0];
    float x1 = (float)ptr[d1];


    float y0 = x0 * cos_val - x1 * sin_val;
    float y1 = x1 * cos_val + x0 * sin_val;


    ptr[d0] = __float2bfloat16(y0);
    ptr[d1] = __float2bfloat16(y1);
}




















// 2026-09-25: Proportional RoPE, for the Gemma-4 full-attention layers
// (crates/model-arch/src/weight_loader/gemma4/loader_a.rs passes rope_angles as the rotary
// dim override). rope_forward cannot express it: it pairs i with i + rotary_dim/2 and divides
// the exponent by rotary_dim, while this pairs i with i + head_dim/2 and divides by head_dim.
// The channels from rope_angles to head_dim/2, and their partners, are not touched.

extern "C" __global__ void rope_forward_proportional(
    __nv_bfloat16* __restrict__ Q,
    __nv_bfloat16* __restrict__ K,
    const unsigned int* __restrict__ positions,
    const unsigned int seq_len,
    const unsigned int num_q_heads,
    const unsigned int num_kv_heads,
    const unsigned int head_dim,
    const unsigned int rope_angles,
    const float theta
) {
    const unsigned int head_idx = blockIdx.x;
    const unsigned int seq_block = blockIdx.y;
    const unsigned int batch = blockIdx.z;
    const unsigned int tid = threadIdx.x;

    const bool is_q = (head_idx < num_q_heads);
    const unsigned int head = is_q ? head_idx : (head_idx - num_q_heads);

    if (!is_q && head >= num_kv_heads) return;



    const unsigned int pairs_per_pos = rope_angles;
    const unsigned int pos_per_block = (128 / pairs_per_pos) > 0 ? (128 / pairs_per_pos) : 1;

    const unsigned int local_pos = tid / pairs_per_pos;
    const unsigned int pair_idx = tid % pairs_per_pos;

    const unsigned int seq_pos = seq_block * pos_per_block + local_pos;
    if (seq_pos >= seq_len) return;
    if (local_pos >= pos_per_block) return;
    if (pair_idx >= rope_angles) return;

    const unsigned int abs_pos = positions[batch * seq_len + seq_pos];


    const double freq_exp_d = (double)(2 * pair_idx) / (double)head_dim;
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


    const unsigned int half_hd = head_dim / 2;
    const unsigned int d0 = pair_idx;
    const unsigned int d1 = pair_idx + half_hd;
    float x0 = (float)ptr[d0];
    float x1 = (float)ptr[d1];

    float y0 = x0 * cos_val - x1 * sin_val;
    float y1 = x1 * cos_val + x0 * sin_val;

    ptr[d0] = __float2bfloat16(y0);
    ptr[d1] = __float2bfloat16(y1);
}








// 2026-09-25: rope_forward with freq_i = inv_freq[i] from the caller's [rotary_dim/2] table,
// which the loader builds (for example with YaRN scaling). theta is unused.
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

    const unsigned int half_rot = rotary_dim / 2;
    const unsigned int d0 = pair_idx;
    const unsigned int d1 = pair_idx + half_rot;
    float x0 = (float)ptr[d0];
    float x1 = (float)ptr[d1];

    float y0 = x0 * cos_val - x1 * sin_val;
    float y1 = x1 * cos_val + x0 * sin_val;

    ptr[d0] = __float2bfloat16(y0);
    ptr[d1] = __float2bfloat16(y1);
}

// 2026-09-25: rope_forward_yarn with cos and sin multiplied by attention_factor. Only the rotated
// channels are scaled; the channels past rotary_dim are not touched.
extern "C" __global__ void rope_forward_yarn_scaled(
    __nv_bfloat16* __restrict__ Q,
    __nv_bfloat16* __restrict__ K,
    const unsigned int* __restrict__ positions,
    const unsigned int seq_len,
    const unsigned int num_q_heads,
    const unsigned int num_kv_heads,
    const unsigned int head_dim,
    const unsigned int rotary_dim,
    const float* __restrict__ inv_freq,
    const float attention_factor
) {
    const unsigned int head_idx = blockIdx.x;
    const unsigned int seq_block = blockIdx.y;
    const unsigned int batch = blockIdx.z;
    const unsigned int tid = threadIdx.x;
    const bool is_q = head_idx < num_q_heads;
    const unsigned int head = is_q ? head_idx : head_idx - num_q_heads;
    if (!is_q && head >= num_kv_heads) return;

    const unsigned int pairs_per_pos = rotary_dim / 2;
    const unsigned int pos_per_block = 128 / pairs_per_pos;
    const unsigned int local_pos = tid / pairs_per_pos;
    const unsigned int pair_idx = tid % pairs_per_pos;
    const unsigned int seq_pos = seq_block * pos_per_block + local_pos;
    if (seq_pos >= seq_len) return;

    const unsigned int num_heads = is_q ? num_q_heads : num_kv_heads;
    __nv_bfloat16* ptr = (is_q ? Q : K)
        + batch * seq_len * num_heads * head_dim
        + seq_pos * num_heads * head_dim
        + head * head_dim;
    const float angle = (float)positions[batch * seq_len + seq_pos] * inv_freq[pair_idx];
    const float cos_val = cosf(angle) * attention_factor;
    const float sin_val = sinf(angle) * attention_factor;
    const unsigned int d0 = pair_idx;
    const unsigned int d1 = pair_idx + pairs_per_pos;
    const float x0 = __bfloat162float(ptr[d0]);
    const float x1 = __bfloat162float(ptr[d1]);
    ptr[d0] = __float2bfloat16(x0 * cos_val - x1 * sin_val);
    ptr[d1] = __float2bfloat16(x1 * cos_val + x0 * sin_val);
}













// 2026-09-25: YaRN RoPE on adjacent pairs (2i, 2i + 1) with freq_i = inv_freq[i], and cos and sin
// multiplied by mscale. The DeepSeek-V4 attention paths use it (qwen3_attention/*_v4.rs).
extern "C" __global__ void rope_forward_yarn_interleaved(
    __nv_bfloat16* __restrict__ Q,
    __nv_bfloat16* __restrict__ K,
    const unsigned int* __restrict__ positions,
    const unsigned int seq_len,
    const unsigned int num_q_heads,
    const unsigned int num_kv_heads,
    const unsigned int head_dim,
    const unsigned int rotary_dim,
    const float* __restrict__ inv_freq,
    const float mscale
) {
    const unsigned int head_idx = blockIdx.x;
    const unsigned int seq_block = blockIdx.y;
    const unsigned int batch = blockIdx.z;
    const unsigned int tid = threadIdx.x;

    const bool is_q = (head_idx < num_q_heads);
    const unsigned int head = is_q ? head_idx : (head_idx - num_q_heads);
    const unsigned int num_heads = is_q ? num_q_heads : num_kv_heads;
    (void)num_heads;

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

    const float cos_val = cosf(angle) * mscale;
    const float sin_val = sinf(angle) * mscale;

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



// 2026-09-25: rope_forward_yarn_interleaved with sin negated, which rotates by -angle. The
// DeepSeek-V4 paths apply it to the attention output through ops::rope_yarn, which launches
// either kernel with the same arguments.
extern "C" __global__ void rope_forward_yarn_interleaved_inv(
    __nv_bfloat16* __restrict__ Q,
    __nv_bfloat16* __restrict__ K,
    const unsigned int* __restrict__ positions,
    const unsigned int seq_len,
    const unsigned int num_q_heads,
    const unsigned int num_kv_heads,
    const unsigned int head_dim,
    const unsigned int rotary_dim,
    const float* __restrict__ inv_freq,
    const float mscale
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
    const float freq = inv_freq[pair_idx];
    const float angle = (float)abs_pos * freq;
    const float cos_val = cosf(angle) * mscale;
    const float sin_val = sinf(angle) * mscale;

    __nv_bfloat16* ptr;
    if (is_q) {
        ptr = Q + batch * seq_len * (num_q_heads * head_dim)
                + seq_pos * (num_q_heads * head_dim) + head * head_dim;
    } else {
        ptr = K + batch * seq_len * (num_kv_heads * head_dim)
                + seq_pos * (num_kv_heads * head_dim) + head * head_dim;
    }

    const unsigned int d0 = 2 * pair_idx;
    const unsigned int d1 = 2 * pair_idx + 1;
    float x0 = (float)ptr[d0];
    float x1 = (float)ptr[d1];


    float y0 = x0 * cos_val + x1 * sin_val;
    float y1 = x1 * cos_val - x0 * sin_val;

    ptr[d0] = __float2bfloat16(y0);
    ptr[d1] = __float2bfloat16(y1);
}
