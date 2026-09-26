// SPDX-License-Identifier: MIT OR Apache-2.0

// 2026-09-25: paged_decode_attn_tiled: online-softmax decode attention over one
// tile of KV blocks. Each launch folds its tile into a per-(seq, q_head)
// running state (m, l, o), so the high-speed swap streams a sequence through
// scratch one tile at a time; attention_finalize then writes o / l.
// Per token, with s = q.k / sqrt(head_dim):
//   m' = max(m, s), l' = l*exp(m - m') + exp(s - m'), o' = o*exp(m - m') + v*exp(s - m')
//
// Owner: metrale-storage kernels.
// Invariants:
// - The caller sets m = -inf and l = o = 0 before a step's first tile
//   (TiledAttention::begin_step_on_stream).
// - blockDim.x == head_dim <= MAX_HEAD_DIM: TiledAttention launches head_dim
//   threads, and TiledAttentionDims::validate rejects head_dim > 256.
//
// Shapes: Q [num_seqs, num_q_heads, head_dim] BF16; tile_blocks
// [num_seqs, tile_capacity] block ids; tile_block_counts [num_seqs] valid
// entries; m_state, l_state [num_seqs, num_q_heads] f32; o_state
// [num_seqs, num_q_heads, head_dim] f32. Grid (num_seqs, num_q_heads, 1).




#include <cuda_bf16.h>

constexpr int MAX_HEAD_DIM = 256;
constexpr int MAX_WARPS    = MAX_HEAD_DIM / 32;

static __device__ inline float warp_sum(float v) {
    #pragma unroll
    for (int off = 16; off > 0; off >>= 1) {
        v += __shfl_down_sync(0xffffffff, v, off);
    }
    return v;
}

// 2026-09-25: blk_stride, tok_stride and kvh_stride are in BF16 elements:
// K[blk, tok, kv_head, dim] = K_pool + blk*blk_stride + tok*tok_stride +
// kv_head*kvh_stride + dim, and V_pool uses the same strides.
// TiledAttention::paged_strides, for [num_blocks, block_size, num_kv_heads, head_dim]:
//   blk_stride = block_size * num_kv_heads * head_dim
//   tok_stride =              num_kv_heads * head_dim
//   kvh_stride =                             head_dim
// TiledAttention::scratch_pool_strides, for [slot][K|V][kv_head][tok][dim]:
//   blk_stride = 2 * num_kv_heads * block_size * head_dim
//   tok_stride =                                  head_dim
//   kvh_stride =                  block_size *    head_dim
// with V_pool = K_pool + num_kv_heads * block_size * head_dim elements.
extern "C" __global__ void paged_decode_attn_tiled(
    const __nv_bfloat16* __restrict__ Q,
    const __nv_bfloat16* __restrict__ K_pool,
    const __nv_bfloat16* __restrict__ V_pool,
    const int*           __restrict__ tile_blocks,
    const int*           __restrict__ tile_block_counts,
    float*               __restrict__ m_state,
    float*               __restrict__ l_state,
    float*               __restrict__ o_state,
    int num_q_heads,
    int num_kv_heads,
    int head_dim,
    int block_size,
    int tile_capacity,
    int gqa_ratio,
    long long blk_stride,
    long long tok_stride,
    long long kvh_stride,
    // 2026-09-25: In the tile's last block only the first
    // min(last_block_valid_slots, block_size) token slots are read; earlier
    // blocks are read in full. block_size or more disables the mask.




    int last_block_valid_slots
) {
    const int seq = blockIdx.x;
    const int qh  = blockIdx.y;
    const int tid = threadIdx.x;
    const int kh  = qh / gqa_ratio;

    __shared__ float Q_sm[MAX_HEAD_DIM];
    __shared__ float O_sm[MAX_HEAD_DIM];
    __shared__ float warp_buf[MAX_WARPS];
    __shared__ float logit_sh;


    if (tid < head_dim) {
        Q_sm[tid] = __bfloat162float(Q[((size_t)seq * num_q_heads + qh) * head_dim + tid]);
        O_sm[tid] = o_state[((size_t)seq * num_q_heads + qh) * head_dim + tid];
    }
    float m_run = m_state[seq * num_q_heads + qh];
    float l_run = l_state[seq * num_q_heads + qh];
    __syncthreads();

    const int n_blocks = tile_block_counts[seq];
    const float inv_sqrt_d = rsqrtf((float)head_dim);
    const int n_warps = (blockDim.x + 31) / 32;






    const int t_max_last = last_block_valid_slots < block_size
        ? last_block_valid_slots : block_size;
    for (int b = 0; b < n_blocks; ++b) {
        const int blk_id = tile_blocks[(size_t)seq * tile_capacity + b];
        const size_t blk_base = (size_t)blk_id * (size_t)blk_stride;
        const int t_lim = (b == n_blocks - 1) ? t_max_last : block_size;

        for (int t = 0; t < t_lim; ++t) {
            const size_t kv_base = blk_base
                + (size_t)t * (size_t)tok_stride
                + (size_t)kh * (size_t)kvh_stride;


            float partial = 0.0f;
            if (tid < head_dim) {
                partial = Q_sm[tid] * __bfloat162float(K_pool[kv_base + tid]);
            }


            float w = warp_sum(partial);
            const int lane = tid & 31;
            const int warp = tid >> 5;
            if (lane == 0) warp_buf[warp] = w;
            __syncthreads();
            if (warp == 0) {
                float v = (lane < n_warps) ? warp_buf[lane] : 0.0f;
                v = warp_sum(v);
                if (lane == 0) logit_sh = v * inv_sqrt_d;
            }
            __syncthreads();
            const float logit = logit_sh;



            const float m_new = fmaxf(m_run, logit);
            const float scale_old = __expf(m_run - m_new);
            const float scale_new = __expf(logit - m_new);
            const float l_new = l_run * scale_old + scale_new;


            if (tid < head_dim) {
                const float v = __bfloat162float(V_pool[kv_base + tid]);
                O_sm[tid] = O_sm[tid] * scale_old + v * scale_new;
            }
            __syncthreads();

            m_run = m_new;
            l_run = l_new;
        }
    }


    if (tid == 0) {
        m_state[seq * num_q_heads + qh] = m_run;
        l_state[seq * num_q_heads + qh] = l_run;
    }
    if (tid < head_dim) {
        o_state[((size_t)seq * num_q_heads + qh) * head_dim + tid] = O_sm[tid];
    }
}
