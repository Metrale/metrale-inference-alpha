// SPDX-License-Identifier: MIT OR Apache-2.0

// 2026-09-25: Kernels `moe_topk_softmax_bias`, `moe_topk_softmax_bias_batched` and `moe_zero_expert_add`: softmax
// MoE routing with a selection bias and identity (zero-computation) experts.
//
//   scores    = softmax(logits) over the first min(num_experts, MAX_EXPERTS) logits
//   indices   = top_k of (scores + bias), highest first; on equal values the lower expert index wins
//   weights   = scores[indices], divided by their sum when `normalize` and the sum exceeds 1e-20,
//               then multiplied by scaling_factor
//
// Expert ids >= num_routed are identity experts, whose output is weight * input. A chosen identity expert's
// weight is added to zero_accum[token] and its slot is written as (expert 0, weight 0), so the expert GEMMs
// see a no-op slot. The MoE layer (moe/forward.rs) then runs `moe_zero_expert_add` to add
// zero_accum[token] * input[token] to the MoE output.
//
// Owner: gb10 kernels.
// Invariants: top_k is clamped to MAX_TOP_K, and every routing launch writes zero_accum[token]. The routing
// kernels assume blockDim.x == 256 (BLOCK_SIZE); grid (1, 1, 1) single, (N, 1, 1) batched.





#include <cuda_bf16.h>

#define BLOCK_SIZE 256
#define MAX_EXPERTS 512
#define MAX_TOP_K 32
#define WARP_SIZE 32


static __device__ __forceinline__ void softmax_bias_route_one(
    const __nv_bfloat16* __restrict__ my_gate,
    const float* __restrict__ bias,
    unsigned int* __restrict__ out_indices,
    float* __restrict__ out_weights,
    float* __restrict__ out_zero_accum,
    unsigned int num_experts,  // 2026-09-25: all logits, routed and identity
    unsigned int num_routed,
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
    __shared__ float s_bcast;

    const unsigned int tid = threadIdx.x;
    const unsigned int warp_id = tid / 32;
    const unsigned int lane = tid % 32;
    const unsigned int num_warps = BLOCK_SIZE / 32;

    unsigned int actual_n = num_experts < MAX_EXPERTS ? num_experts : MAX_EXPERTS;
    const unsigned int top_k_c = top_k < MAX_TOP_K ? top_k : MAX_TOP_K;




    float lmax = -1e30f;
    for (unsigned int i = tid; i < actual_n; i += BLOCK_SIZE) {
        float v = __bfloat162float(my_gate[i]);
        s_score[i] = v;
        lmax = fmaxf(lmax, v);
    }
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        lmax = fmaxf(lmax, __shfl_down_sync(0xFFFFFFFF, lmax, offset));
    }
    if (lane == 0) s_warp_val[warp_id] = lmax;
    __syncthreads();
    if (tid == 0) {
        float m = s_warp_val[0];
        for (unsigned int w = 1; w < num_warps; w++) m = fmaxf(m, s_warp_val[w]);
        s_bcast = m;
    }
    __syncthreads();
    const float gmax = s_bcast;


    float lsum = 0.0f;
    for (unsigned int i = tid; i < actual_n; i += BLOCK_SIZE) {
        float e = __expf(s_score[i] - gmax);
        s_score[i] = e;
        lsum += e;
    }
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        lsum += __shfl_down_sync(0xFFFFFFFF, lsum, offset);
    }
    if (lane == 0) s_warp_val[warp_id] = lsum;
    __syncthreads();
    if (tid == 0) {
        float s = 0.0f;
        for (unsigned int w = 0; w < num_warps; w++) s += s_warp_val[w];
        s_bcast = s;
    }
    __syncthreads();
    const float inv_sum = 1.0f / s_bcast;


    for (unsigned int i = tid; i < actual_n; i += BLOCK_SIZE) {
        float sm = s_score[i] * inv_sum;
        s_score[i] = sm;
        s_selection[i] = sm + bias[i];
    }
    for (unsigned int i = actual_n + tid; i < MAX_EXPERTS; i += BLOCK_SIZE) {
        s_score[i] = -1e30f;
        s_selection[i] = -1e30f;
    }
    __syncthreads();


    for (unsigned int t = 0; t < top_k_c && t < actual_n; t++) {
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
            if (other_val > local_max || (other_val == local_max && other_idx < local_idx)) {
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
                if (s_warp_val[w] > best_val
                    || (s_warp_val[w] == best_val && s_warp_idx[w] < best_idx)) {
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
        for (unsigned int t = 0; t < top_k_c && t < actual_n; t++) {
            float w = s_score[s_top_idxs[t]];
            s_top_vals[t] = w;
            topk_sum += w;
        }
        if (normalize && topk_sum > 1e-20f) {
            for (unsigned int t = 0; t < top_k_c && t < actual_n; t++) {
                s_top_vals[t] /= topk_sum;
            }
        }
        float zero_sum = 0.0f;
        for (unsigned int t = 0; t < top_k_c && t < actual_n; t++) {
            unsigned int idx = s_top_idxs[t];
            float w = s_top_vals[t] * scaling_factor;
            if (idx >= num_routed) {


                zero_sum += w;
                idx = 0;
                w = 0.0f;
            }
            out_indices[t] = idx;
            out_weights[t] = w;
        }
        out_zero_accum[0] = zero_sum;
    }
}

extern "C" __global__ void moe_topk_softmax_bias(
    const __nv_bfloat16* __restrict__ gate_logits,
    const float* __restrict__ bias,
    unsigned int* __restrict__ expert_indices,
    float* __restrict__ expert_weights,
    float* __restrict__ zero_accum,
    unsigned int num_experts,
    unsigned int num_routed,
    unsigned int top_k,
    unsigned int normalize,
    float scaling_factor
) {
    softmax_bias_route_one(
        gate_logits, bias, expert_indices, expert_weights, zero_accum,
        num_experts, num_routed, top_k, normalize, scaling_factor);
}



extern "C" __global__ void moe_topk_softmax_bias_batched(
    const __nv_bfloat16* __restrict__ gate_logits,
    const float* __restrict__ bias,
    unsigned int* __restrict__ expert_indices,
    float* __restrict__ expert_weights,
    float* __restrict__ zero_accum,
    unsigned int num_experts,
    unsigned int num_routed,
    unsigned int top_k,
    unsigned int normalize,
    float scaling_factor
) {
    const unsigned int token = blockIdx.x;
    softmax_bias_route_one(
        gate_logits + (unsigned long long)token * num_experts,
        bias,
        expert_indices + (unsigned long long)token * top_k,
        expert_weights + (unsigned long long)token * top_k,
        zero_accum + token,
        num_experts, num_routed, top_k, normalize, scaling_factor);
}

// 2026-09-25: out[t, :] += zero_accum[t] * x[t, :] in F32, rounded to BF16; tokens with zero_accum 0 are skipped.
// One thread per element, so the grid must cover n * h threads.

extern "C" __global__ void moe_zero_expert_add(
    __nv_bfloat16* __restrict__ out,        // 2026-09-25: [N, h], updated in place
    const __nv_bfloat16* __restrict__ x,    // 2026-09-25: [N, h] MoE input
    const float* __restrict__ zero_accum,
    unsigned int n,
    unsigned int h
) {
    unsigned long long i = (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;
    unsigned long long total = (unsigned long long)n * h;
    if (i >= total) return;
    unsigned int token = (unsigned int)(i / h);
    float z = zero_accum[token];
    if (z == 0.0f) return;
    float v = __bfloat162float(out[i]) + z * __bfloat162float(x[i]);
    out[i] = __float2bfloat16(v);
}
