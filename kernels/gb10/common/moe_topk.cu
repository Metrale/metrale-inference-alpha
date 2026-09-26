// SPDX-License-Identifier: MIT OR Apache-2.0

// 2026-09-25: Kernels `moe_topk_softmax`, `moe_topk_softmax_f32` and `moe_topk_softmax_batched`: softmax MoE routing.
// Each picks the top_k experts by gate logit, highest first, and returns their softmax weights.
//
// One 256-thread block per token. Only the first MAX_EXPERTS logits are read. The weight of a chosen expert is
// the softmax over all read logits taken at that expert; with `normalize` the top_k weights are then divided by
// their sum. When top_k exceeds num_experts, the output entries past num_experts are left unwritten.
//
// Owner: gb10 kernels.
// Invariants: none beyond the types. The kernels assume blockDim.x == 256 (BLOCK_SIZE) and top_k <= MAX_TOP_K,
// and check neither.

#include <cuda_bf16.h>

#define BLOCK_SIZE 256
#define MAX_EXPERTS 512
#define MAX_TOP_K 32  // 2026-09-25: MoeLayer::new_with_hash refuses num_experts_per_tok above MOE_TOPK_SIGMOID_MAX_TOP_K (32)
#define WARP_SIZE 32

// 2026-09-25: `moe_topk_softmax`: one token, BF16 logits, grid (1, 1, 1). Experts are picked one per round by a
// block-wide argmax; on equal logits the lower expert index wins, in the thread, warp and cross-warp steps.







extern "C" __global__ void moe_topk_softmax(
    const __nv_bfloat16* __restrict__ gate_logits,
    unsigned int* __restrict__ expert_indices,
    float* __restrict__ expert_weights,
    unsigned int num_experts,
    unsigned int top_k,
    unsigned int normalize
) {
    __shared__ float s_vals[MAX_EXPERTS];
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
        s_vals[i] = __bfloat162float(gate_logits[i]);
    }

    for (unsigned int i = actual_n + tid; i < MAX_EXPERTS; i += BLOCK_SIZE) {
        s_vals[i] = -1e30f;
    }
    __syncthreads();


    for (unsigned int t = 0; t < top_k && t < actual_n; t++) {

        float local_max = -1e30f;
        unsigned int local_idx = 0;
        for (unsigned int i = tid; i < actual_n; i += BLOCK_SIZE) {
            float v = s_vals[i];
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
                if (s_warp_val[w] > best_val || (s_warp_val[w] == best_val && s_warp_idx[w] < best_idx)) {
                    best_val = s_warp_val[w];
                    best_idx = s_warp_idx[w];
                }
            }
            s_top_vals[t] = best_val;
            s_top_idxs[t] = best_idx;

            s_vals[best_idx] = -1e30f;
        }
        __syncthreads();
    }

    // 2026-09-25: The softmax denominator runs over all read experts. Chosen experts are -1e30 in s_vals by now, so
    // they add zero to the parallel sum and thread 0 adds their terms back from s_top_vals. s_top_vals[0] is the
    // largest logit, the softmax max.








    float global_max = s_top_vals[0];



    {
        float local_sum = 0.0f;
        for (unsigned int i = tid; i < actual_n; i += BLOCK_SIZE) {
            local_sum += __expf(s_vals[i] - global_max);
        }

        #pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1) {
            local_sum += __shfl_down_sync(0xFFFFFFFF, local_sum, offset);
        }
        if (lane == 0) s_warp_val[warp_id] = local_sum;
        __syncthreads();


        if (tid == 0) {
            float total = 0.0f;
            for (unsigned int w = 0; w < num_warps; w++) {
                total += s_warp_val[w];
            }

            for (unsigned int t = 0; t < top_k && t < actual_n; t++) {
                total += __expf(s_top_vals[t] - global_max);
            }
            s_warp_val[0] = total;
        }
        __syncthreads();
    }

    float exp_sum = s_warp_val[0];


    if (tid == 0) {
        for (unsigned int t = 0; t < top_k && t < actual_n; t++) {
            expert_indices[t] = s_top_idxs[t];
            float softmax_weight = __expf(s_top_vals[t] - global_max) / exp_sum;

            if (normalize) {
                s_top_vals[t] = softmax_weight;
            } else {
                expert_weights[t] = softmax_weight;
            }
        }

        if (normalize) {
            float topk_sum = 0.0f;
            for (unsigned int t = 0; t < top_k && t < actual_n; t++) {
                topk_sum += s_top_vals[t];
            }
            for (unsigned int t = 0; t < top_k && t < actual_n; t++) {
                expert_weights[t] = s_top_vals[t] / topk_sum;
            }
        }
    }
}

// 2026-09-25: `moe_topk_softmax_f32`: `moe_topk_softmax` with F32 logits, same tie-break. The MoE layer uses it when
// the router logits are kept in F32 (the `fp32_gate` or `fp32_routing` lever, METRALE_FP32_GATE /
// METRALE_FP32_ROUTING).






extern "C" __global__ void moe_topk_softmax_f32(
    const float* __restrict__ gate_logits,
    unsigned int* __restrict__ expert_indices,
    float* __restrict__ expert_weights,
    unsigned int num_experts,
    unsigned int top_k,
    unsigned int normalize
) {
    __shared__ float s_vals[MAX_EXPERTS];
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
        s_vals[i] = gate_logits[i];
    }
    for (unsigned int i = actual_n + tid; i < MAX_EXPERTS; i += BLOCK_SIZE) {
        s_vals[i] = -1e30f;
    }
    __syncthreads();


    for (unsigned int t = 0; t < top_k && t < actual_n; t++) {
        float local_max = -1e30f;
        unsigned int local_idx = 0;
        for (unsigned int i = tid; i < actual_n; i += BLOCK_SIZE) {
            float v = s_vals[i];
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
                if (s_warp_val[w] > best_val || (s_warp_val[w] == best_val && s_warp_idx[w] < best_idx)) {
                    best_val = s_warp_val[w];
                    best_idx = s_warp_idx[w];
                }
            }
            s_top_vals[t] = best_val;
            s_top_idxs[t] = best_idx;
            s_vals[best_idx] = -1e30f;
        }
        __syncthreads();
    }


    float global_max = s_top_vals[0];
    {
        float local_sum = 0.0f;
        for (unsigned int i = tid; i < actual_n; i += BLOCK_SIZE) {
            local_sum += __expf(s_vals[i] - global_max);
        }
        #pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1) {
            local_sum += __shfl_down_sync(0xFFFFFFFF, local_sum, offset);
        }
        if (lane == 0) s_warp_val[warp_id] = local_sum;
        __syncthreads();
        if (tid == 0) {
            float total = 0.0f;
            for (unsigned int w = 0; w < num_warps; w++) {
                total += s_warp_val[w];
            }
            for (unsigned int t = 0; t < top_k && t < actual_n; t++) {
                total += __expf(s_top_vals[t] - global_max);
            }
            s_warp_val[0] = total;
        }
        __syncthreads();
    }

    float exp_sum = s_warp_val[0];
    if (tid == 0) {
        for (unsigned int t = 0; t < top_k && t < actual_n; t++) {
            expert_indices[t] = s_top_idxs[t];
            float softmax_weight = __expf(s_top_vals[t] - global_max) / exp_sum;
            if (normalize) {
                s_top_vals[t] = softmax_weight;
            } else {
                expert_weights[t] = softmax_weight;
            }
        }
        if (normalize) {
            float topk_sum = 0.0f;
            for (unsigned int t = 0; t < top_k && t < actual_n; t++) {
                topk_sum += s_top_vals[t];
            }
            for (unsigned int t = 0; t < top_k && t < actual_n; t++) {
                expert_weights[t] = s_top_vals[t] / topk_sum;
            }
        }
    }
}

// 2026-09-25: `moe_topk_softmax_batched`: BF16 logits for N tokens, one block per token, grid (N, 1, 1). Its warp
// and cross-warp steps keep the lower lane's or warp's candidate on an equal logit, which need not be the
// lower expert index that the single-token kernels pick.
// gate_logits [N, num_experts]; expert_indices and expert_weights [N, top_k].



extern "C" __global__ void moe_topk_softmax_batched(
    const __nv_bfloat16* __restrict__ gate_logits,
    unsigned int* __restrict__ expert_indices,
    float* __restrict__ expert_weights,
    unsigned int num_experts,
    unsigned int top_k,
    unsigned int normalize
) {
    __shared__ float s_vals[MAX_EXPERTS];
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
        s_vals[i] = __bfloat162float(my_gate[i]);
    }
    for (unsigned int i = actual_n + tid; i < MAX_EXPERTS; i += BLOCK_SIZE) {
        s_vals[i] = -1e30f;
    }
    __syncthreads();


    for (unsigned int t = 0; t < top_k && t < actual_n; t++) {
        float local_max = -1e30f;
        unsigned int local_idx = 0;
        for (unsigned int i = tid; i < actual_n; i += BLOCK_SIZE) {
            float v = s_vals[i];
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
            s_vals[best_idx] = -1e30f;
        }
        __syncthreads();
    }


    float global_max = s_top_vals[0];
    {
        float local_sum = 0.0f;
        for (unsigned int i = tid; i < actual_n; i += BLOCK_SIZE) {
            local_sum += __expf(s_vals[i] - global_max);
        }
        #pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1) {
            local_sum += __shfl_down_sync(0xFFFFFFFF, local_sum, offset);
        }
        if (lane == 0) s_warp_val[warp_id] = local_sum;
        __syncthreads();
        if (tid == 0) {
            float total = 0.0f;
            for (unsigned int w = 0; w < num_warps; w++) {
                total += s_warp_val[w];
            }
            for (unsigned int t = 0; t < top_k && t < actual_n; t++) {
                total += __expf(s_top_vals[t] - global_max);
            }
            s_warp_val[0] = total;
        }
        __syncthreads();
    }

    float exp_sum = s_warp_val[0];
    if (tid == 0) {
        for (unsigned int t = 0; t < top_k && t < actual_n; t++) {
            my_indices[t] = s_top_idxs[t];
            float softmax_weight = __expf(s_top_vals[t] - global_max) / exp_sum;
            if (normalize) {
                s_top_vals[t] = softmax_weight;
            } else {
                my_weights[t] = softmax_weight;
            }
        }
        if (normalize) {
            float topk_sum = 0.0f;
            for (unsigned int t = 0; t < top_k && t < actual_n; t++) {
                topk_sum += s_top_vals[t];
            }
            for (unsigned int t = 0; t < top_k && t < actual_n; t++) {
                my_weights[t] = s_top_vals[t] / topk_sum;
            }
        }
    }
}
