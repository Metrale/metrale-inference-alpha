// SPDX-License-Identifier: MIT OR Apache-2.0

// 2026-09-25: Kernels `moe_topk_sqrtsoftplus` and `moe_topk_sqrtsoftplus_batched`: MoE routing on sqrt(softplus).
//   scores    = sqrt(log(1 + exp(logits)))
//   indices   = top_k of (scores + bias), highest first
//   weights   = scores[indices], divided by their sum when `normalize` and the sum exceeds 1e-20,
//               then multiplied by scaling_factor
// One 256-thread block per token; only the first MAX_EXPERTS logits are read. The warp and cross-warp steps
// keep the lower lane's or warp's candidate on an equal value, not necessarily the lower expert index.
//
// Owner: gb10 kernels.
// Invariants: none beyond the types. The kernels assume blockDim.x == 256 (BLOCK_SIZE) and top_k <= MAX_TOP_K
// (MoeLayer::new_with_hash refuses a larger num_experts_per_tok), and check neither.

#include <cuda_bf16.h>

#define BLOCK_SIZE 256
#define MAX_EXPERTS 512
#define MAX_TOP_K 32
#define WARP_SIZE 32

extern "C" __global__ void moe_topk_sqrtsoftplus(
    const __nv_bfloat16* __restrict__ gate_logits,
    const float* __restrict__ bias,
    unsigned int* __restrict__ expert_indices,
    float* __restrict__ expert_weights,
    unsigned int num_experts,
    unsigned int top_k,
    unsigned int normalize,
    float scaling_factor
) {
    __shared__ float s_score[MAX_EXPERTS];
    __shared__ float s_selection[MAX_EXPERTS];
    __shared__ float s_top_vals[MAX_TOP_K];
    __shared__ unsigned int s_top_idxs[MAX_TOP_K];
    __shared__ float s_warp_val[8];
    __shared__ unsigned int s_warp_idx[8];

    const unsigned int tid = threadIdx.x;
    const unsigned int warp_id = tid / 32;
    const unsigned int lane = tid % 32;
    const unsigned int num_warps = BLOCK_SIZE / 32;

    unsigned int actual_n = num_experts < MAX_EXPERTS ? num_experts : MAX_EXPERTS;


    for (unsigned int i = tid; i < actual_n; i += BLOCK_SIZE) {
        float logit = __bfloat162float(gate_logits[i]);
        float score = sqrtf(logf(1.0f + __expf(logit)));
        s_score[i] = score;
        s_selection[i] = score + bias[i];
    }
    for (unsigned int i = actual_n + tid; i < MAX_EXPERTS; i += BLOCK_SIZE) {
        s_score[i] = -1e30f;
        s_selection[i] = -1e30f;
    }
    __syncthreads();


    for (unsigned int t = 0; t < top_k && t < actual_n; t++) {
        float local_max = -1e30f;
        unsigned int local_idx = 0;
        for (unsigned int i = tid; i < actual_n; i += BLOCK_SIZE) {
            float v = s_selection[i];
            if (v > local_max) {
                local_max = v;
                local_idx = i;
            }
        }

        #pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1) {
            float other_val = __shfl_down_sync(0xFFFFFFFF, local_max, offset);
            unsigned int other_idx = __shfl_down_sync(0xFFFFFFFF, local_idx, offset);
            if (other_val > local_max) {
                local_max = other_val;
                local_idx = other_idx;
            }
        }

        if (lane == 0) {
            s_warp_val[warp_id] = local_max;
            s_warp_idx[warp_id] = local_idx;
        }
        __syncthreads();

        if (tid == 0) {
            float best_val = s_warp_val[0];
            unsigned int best_idx = s_warp_idx[0];
            for (unsigned int w = 1; w < num_warps; w++) {
                if (s_warp_val[w] > best_val) {
                    best_val = s_warp_val[w];
                    best_idx = s_warp_idx[w];
                }
            }
            s_top_vals[t] = best_val;
            s_top_idxs[t] = best_idx;
            s_selection[best_idx] = -1e30f;
        }
        __syncthreads();
    }


    if (tid == 0) {
        float topk_sum = 0.0f;
        for (unsigned int t = 0; t < top_k && t < actual_n; t++) {
            unsigned int idx = s_top_idxs[t];
            float w = s_score[idx];
            s_top_vals[t] = w;
            topk_sum += w;
        }

        if (normalize && topk_sum > 1e-20f) {
            for (unsigned int t = 0; t < top_k && t < actual_n; t++) {
                s_top_vals[t] /= topk_sum;
            }
        }

        for (unsigned int t = 0; t < top_k && t < actual_n; t++) {
            expert_indices[t] = s_top_idxs[t];
            expert_weights[t] = s_top_vals[t] * scaling_factor;
        }
    }
}

// 2026-09-25: `moe_topk_sqrtsoftplus_batched`: N tokens, one block per token, grid (N, 1, 1); `bias` is shared.


extern "C" __global__ void moe_topk_sqrtsoftplus_batched(
    const __nv_bfloat16* __restrict__ gate_logits,  // 2026-09-25: [N, num_experts]
    const float* __restrict__ bias,
    unsigned int* __restrict__ expert_indices,
    float* __restrict__ expert_weights,
    unsigned int num_experts,
    unsigned int top_k,
    unsigned int normalize,
    float scaling_factor
) {
    __shared__ float s_score[MAX_EXPERTS];
    __shared__ float s_selection[MAX_EXPERTS];
    __shared__ float s_top_vals[MAX_TOP_K];
    __shared__ unsigned int s_top_idxs[MAX_TOP_K];
    __shared__ float s_warp_val[8];
    __shared__ unsigned int s_warp_idx[8];

    const unsigned int token = blockIdx.x;
    const unsigned int tid = threadIdx.x;
    const unsigned int warp_id = tid / 32;
    const unsigned int lane = tid % 32;
    const unsigned int num_warps = BLOCK_SIZE / 32;

    const __nv_bfloat16* my_gate = gate_logits + token * num_experts;
    unsigned int* my_indices = expert_indices + token * top_k;
    float* my_weights = expert_weights + token * top_k;

    unsigned int actual_n = num_experts < MAX_EXPERTS ? num_experts : MAX_EXPERTS;
    for (unsigned int i = tid; i < actual_n; i += BLOCK_SIZE) {
        float logit = __bfloat162float(my_gate[i]);
        float score = sqrtf(logf(1.0f + __expf(logit)));
        s_score[i] = score;
        s_selection[i] = score + bias[i];
    }
    for (unsigned int i = actual_n + tid; i < MAX_EXPERTS; i += BLOCK_SIZE) {
        s_score[i] = -1e30f;
        s_selection[i] = -1e30f;
    }
    __syncthreads();

    for (unsigned int t = 0; t < top_k && t < actual_n; t++) {
        float local_max = -1e30f;
        unsigned int local_idx = 0;
        for (unsigned int i = tid; i < actual_n; i += BLOCK_SIZE) {
            float v = s_selection[i];
            if (v > local_max) {
                local_max = v;
                local_idx = i;
            }
        }
        #pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1) {
            float other_val = __shfl_down_sync(0xFFFFFFFF, local_max, offset);
            unsigned int other_idx = __shfl_down_sync(0xFFFFFFFF, local_idx, offset);
            if (other_val > local_max) {
                local_max = other_val;
                local_idx = other_idx;
            }
        }
        if (lane == 0) {
            s_warp_val[warp_id] = local_max;
            s_warp_idx[warp_id] = local_idx;
        }
        __syncthreads();
        if (tid == 0) {
            float best_val = s_warp_val[0];
            unsigned int best_idx = s_warp_idx[0];
            for (unsigned int w = 1; w < num_warps; w++) {
                if (s_warp_val[w] > best_val) {
                    best_val = s_warp_val[w];
                    best_idx = s_warp_idx[w];
                }
            }
            s_top_vals[t] = best_val;
            s_top_idxs[t] = best_idx;
            s_selection[best_idx] = -1e30f;
        }
        __syncthreads();
    }

    if (tid == 0) {
        float topk_sum = 0.0f;
        for (unsigned int t = 0; t < top_k && t < actual_n; t++) {
            unsigned int idx = s_top_idxs[t];
            float w = s_score[idx];
            s_top_vals[t] = w;
            topk_sum += w;
        }
        if (normalize && topk_sum > 1e-20f) {
            for (unsigned int t = 0; t < top_k && t < actual_n; t++) {
                s_top_vals[t] /= topk_sum;
            }
        }
        for (unsigned int t = 0; t < top_k && t < actual_n; t++) {
            my_indices[t] = s_top_idxs[t];
            my_weights[t] = s_top_vals[t] * scaling_factor;
        }
    }
}
