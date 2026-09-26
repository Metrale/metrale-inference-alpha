// SPDX-License-Identifier: MIT OR Apache-2.0

// 2026-09-25: MoE hash routing: a token's experts are the static row tid2eid[token_id], not a
// top-k of the gate, and each is weighted by sqrt(softplus(gate_logit)).
//
// Owner: gb10 kernels.
// Invariants:
// - Only thread 0 of a block works; the host launches 256-thread blocks. moe_hash_route
//   routes one token (token_id_ptr[0]); moe_hash_route_batched routes token blockIdx.x of
//   N, reading gate_logits [N, num_experts] and writing [N, top_k] outputs.
// - tid2eid is [vocab, top_k] of `long`. An id >= num_experts is replaced by expert 0,
//   with no error raised.
// - weight = score / sum(scores) when normalize is set and the sum exceeds 1e-20, else
//   score; then times scaling_factor. At most MAX_TOP_K (32) slots are written.






#include <cuda_bf16.h>

#define MAX_TOP_K 32

extern "C" __global__ void moe_hash_route(
    const __nv_bfloat16* __restrict__ gate_logits,
    const long* __restrict__ tid2eid,
    const unsigned int* __restrict__ token_id_ptr,
    unsigned int* __restrict__ expert_indices,
    float* __restrict__ expert_weights,
    unsigned int num_experts,
    unsigned int top_k,
    unsigned int normalize,
    float scaling_factor
) {
    if (threadIdx.x != 0) return;

    const unsigned int tok = token_id_ptr[0];
    const long* row = tid2eid + (size_t)tok * (size_t)top_k;

    float w_local[MAX_TOP_K];
    unsigned int idx_local[MAX_TOP_K];
    float topk_sum = 0.0f;

    for (unsigned int t = 0; t < top_k && t < MAX_TOP_K; t++) {
        unsigned int e = (unsigned int)row[t];
        if (e >= num_experts) e = 0;
        idx_local[t] = e;
        float logit = __bfloat162float(gate_logits[e]);
        float score = sqrtf(logf(1.0f + __expf(logit)));
        w_local[t] = score;
        topk_sum += score;
    }

    if (normalize && topk_sum > 1e-20f) {
        for (unsigned int t = 0; t < top_k && t < MAX_TOP_K; t++) {
            w_local[t] /= topk_sum;
        }
    }

    for (unsigned int t = 0; t < top_k && t < MAX_TOP_K; t++) {
        expert_indices[t] = idx_local[t];
        expert_weights[t] = w_local[t] * scaling_factor;
    }
}




extern "C" __global__ void moe_hash_route_batched(
    const __nv_bfloat16* __restrict__ gate_logits,
    const long* __restrict__ tid2eid,
    const unsigned int* __restrict__ token_ids,
    unsigned int* __restrict__ expert_indices,
    float* __restrict__ expert_weights,
    unsigned int num_experts,
    unsigned int top_k,
    unsigned int normalize,
    float scaling_factor
) {
    if (threadIdx.x != 0) return;

    const unsigned int token = blockIdx.x;
    const __nv_bfloat16* my_gate = gate_logits + (size_t)token * num_experts;
    unsigned int* my_indices = expert_indices + (size_t)token * top_k;
    float* my_weights = expert_weights + (size_t)token * top_k;

    const unsigned int tok = token_ids[token];
    const long* row = tid2eid + (size_t)tok * (size_t)top_k;

    float w_local[MAX_TOP_K];
    unsigned int idx_local[MAX_TOP_K];
    float topk_sum = 0.0f;

    for (unsigned int t = 0; t < top_k && t < MAX_TOP_K; t++) {
        unsigned int e = (unsigned int)row[t];
        if (e >= num_experts) e = 0;
        idx_local[t] = e;
        float logit = __bfloat162float(my_gate[e]);
        float score = sqrtf(logf(1.0f + __expf(logit)));
        w_local[t] = score;
        topk_sum += score;
    }

    if (normalize && topk_sum > 1e-20f) {
        for (unsigned int t = 0; t < top_k && t < MAX_TOP_K; t++) {
            w_local[t] /= topk_sum;
        }
    }

    for (unsigned int t = 0; t < top_k && t < MAX_TOP_K; t++) {
        my_indices[t] = idx_local[t];
        my_weights[t] = w_local[t] * scaling_factor;
    }
}
