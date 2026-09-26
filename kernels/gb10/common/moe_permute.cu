// SPDX-License-Identifier: MIT OR Apache-2.0

#include <cuda_bf16.h>
#include <assert.h>

// 2026-09-25: MoE token routing: sort the (token, k) slots by expert, gather rows, reduce
// expert outputs back to tokens, blend in the shared expert, and build the grouped-GEMM
// work-list.
//
// Owner: gb10 kernels.
// Invariants:
// - A slot is token * topk + k. topk_ids, topk_weights and token_to_perm are indexed by slot.
// - After moe_sort_by_expert, expert e owns the sorted rows
//   [expert_offsets[e], expert_offsets[e + 1]).
// - The row order inside one expert's range follows shared-memory atomic order, so it
//   is not deterministic from run to run.

// 2026-09-25: permuted[row] = hidden_states[sorted_token_ids[row]]. One block per row;
// the threads stride over hidden_size.
extern "C" __global__ void moe_permute_tokens(
    const __nv_bfloat16* __restrict__ hidden_states,
    __nv_bfloat16* __restrict__ permuted,
    const int* __restrict__ sorted_token_ids,
    unsigned int hidden_size,
    unsigned int total_expanded
) {
    unsigned int row = blockIdx.x;
    unsigned int col = threadIdx.x;

    if (row >= total_expanded) return;

    int src_token = sorted_token_ids[row];


    for (unsigned int c = col; c < hidden_size; c += blockDim.x) {
        permuted[row * hidden_size + c] = hidden_states[src_token * hidden_size + c];
    }
}

// 2026-09-25: output[token] = sum over k of topk_weights[token, k] *
// expert_output[token * topk + k]. It reads expert_output in slot order, not sorted
// order: sorted_token_ids is not read. Slots at or past total_expanded are skipped.


extern "C" __global__ void moe_unpermute_reduce(
    const __nv_bfloat16* __restrict__ expert_output,
    __nv_bfloat16* __restrict__ output,
    const int* __restrict__ sorted_token_ids,
    const float* __restrict__ topk_weights,
    unsigned int hidden_size,
    unsigned int num_tokens,
    unsigned int topk,
    unsigned int total_expanded
) {

    unsigned int token = blockIdx.x;
    unsigned int col = threadIdx.x;

    if (token >= num_tokens) return;

    for (unsigned int c = col; c < hidden_size; c += blockDim.x) {
        float acc = 0.0f;
        for (unsigned int k = 0; k < topk; k++) {
            unsigned int perm_row = token * topk + k;
            if (perm_row < total_expanded) {
                float w = topk_weights[token * topk + k];
                float val = __bfloat162float(expert_output[perm_row * hidden_size + c]);
                acc += w * val;
            }
        }
        output[token * hidden_size + c] = __float2bfloat16(acc);
    }
}

// 2026-09-25: expert_counts[topk_ids[slot]] += 1 for each of the num_tokens * topk slots.
// The kernel only adds, so the caller zeroes expert_counts first.
extern "C" __global__ void moe_count_experts(
    const int* __restrict__ topk_ids,
    int* __restrict__ expert_counts,
    unsigned int num_tokens,
    unsigned int topk
) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int total = num_tokens * topk;
    if (idx < total) {
        int expert_id = topk_ids[idx];
        atomicAdd(&expert_counts[expert_id], 1);
    }
}

// 2026-09-25: output[token] = sum over k of topk_weights[token, k] *
// expert_output[token_to_perm[token * topk + k]], with token_to_perm from
// moe_sort_by_expert. One block per token; the threads stride over hidden_size.

extern "C" __global__ void moe_unpermute_reduce_indexed(
    const __nv_bfloat16* __restrict__ expert_output,
    __nv_bfloat16* __restrict__ output,
    const int* __restrict__ token_to_perm,
    const float* __restrict__ topk_weights,
    unsigned int hidden_size,
    unsigned int num_tokens,
    unsigned int topk
) {
    unsigned int token = blockIdx.x;
    if (token >= num_tokens) return;

    for (unsigned int c = threadIdx.x; c < hidden_size; c += blockDim.x) {
        float acc = 0.0f;
        for (unsigned int k = 0; k < topk; k++) {
            int perm_row = token_to_perm[token * topk + k];
            float w = topk_weights[token * topk + k];
            float val = __bfloat162float(expert_output[perm_row * hidden_size + c]);
            acc += w * val;
        }
        output[token * hidden_size + c] = __float2bfloat16(acc);
    }
}

// 2026-09-25: output[token] += sigmoid(dot(normed[token], gate_weight)) * shared_out[token].
// A null gate_weight uses a factor of 1. One block per token. blockDim.x must be a
// multiple of 32 and at most 256: s_dot_partial holds one partial sum per warp, 8 slots.




extern "C" __global__ void moe_batched_blend(
    __nv_bfloat16* __restrict__ output,
    const __nv_bfloat16* __restrict__ shared_out,
    const __nv_bfloat16* __restrict__ normed,
    const __nv_bfloat16* __restrict__ gate_weight,
    unsigned int hidden_size,
    unsigned int num_tokens
) {
    __shared__ float s_dot_partial[8];

    unsigned int token = blockIdx.x;
    if (token >= num_tokens) return;

    unsigned int tid = threadIdx.x;
    unsigned int warp_id = tid / 32;
    unsigned int lane = tid % 32;

    const __nv_bfloat16* my_normed = normed + token * hidden_size;
    const __nv_bfloat16* my_shared = shared_out + token * hidden_size;
    __nv_bfloat16* my_output = output + token * hidden_size;



    float local_dot = 0.0f;
    if (gate_weight != 0) {
        for (unsigned int i = tid; i < hidden_size; i += blockDim.x) {
            float n = __bfloat162float(my_normed[i]);
            float g = __bfloat162float(gate_weight[i]);
            local_dot += n * g;
        }
    }


    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        local_dot += __shfl_down_sync(0xFFFFFFFF, local_dot, offset);
    }
    if (lane == 0) s_dot_partial[warp_id] = local_dot;
    __syncthreads();


    float gate_scalar;
    if (tid == 0) {
        if (gate_weight == 0) {

            gate_scalar = 1.0f;
        } else {
            float total = 0.0f;
            for (unsigned int w = 0; w < blockDim.x / 32; w++) {
                total += s_dot_partial[w];
            }
            gate_scalar = 1.0f / (1.0f + __expf(-total));
        }
        s_dot_partial[0] = gate_scalar;
    }
    __syncthreads();
    gate_scalar = s_dot_partial[0];


    for (unsigned int i = tid; i < hidden_size; i += blockDim.x) {
        float o = __bfloat162float(my_output[i]);
        float s = __bfloat162float(my_shared[i]);
        my_output[i] = __float2bfloat16(o + gate_scalar * s);
    }
}

// 2026-09-25: Counting sort of the total_expanded slots by expert id (topk_ids is indexed
// by slot). Writes expert_offsets[0..=num_experts] as the prefix sum of the per-expert
// counts, and for each slot at sorted position pos: sorted_token_ids[pos] = slot / topk,
// sorted_expert_ids[pos] = its expert, token_to_perm[slot] = pos.
// Runs as one block: thread 0 builds the prefix sum.



extern "C" __global__ void moe_sort_by_expert(
    const unsigned int* __restrict__ topk_ids,
    int* __restrict__ sorted_token_ids,
    int* __restrict__ sorted_expert_ids,
    int* __restrict__ expert_offsets,
    int* __restrict__ token_to_perm,
    unsigned int total_expanded,
    unsigned int num_experts,
    unsigned int topk
) {
    // 2026-09-25: counts and offsets hold 1024 experts; num_experts is not checked against that.
    __shared__ unsigned int counts[1024];
    __shared__ unsigned int offsets[1025];


    for (unsigned int i = threadIdx.x; i < num_experts; i += blockDim.x)
        counts[i] = 0;
    __syncthreads();


    for (unsigned int i = threadIdx.x; i < total_expanded; i += blockDim.x)
        atomicAdd(&counts[topk_ids[i]], 1);
    __syncthreads();


    if (threadIdx.x == 0) {
        offsets[0] = 0;
        for (unsigned int e = 0; e < num_experts; e++)
            offsets[e + 1] = offsets[e] + counts[e];
        for (unsigned int e = 0; e <= num_experts; e++)
            expert_offsets[e] = (int)offsets[e];
    }
    __syncthreads();


    for (unsigned int i = threadIdx.x; i < num_experts; i += blockDim.x)
        counts[i] = 0;
    __syncthreads();


    for (unsigned int i = threadIdx.x; i < total_expanded; i += blockDim.x) {
        unsigned int expert_id = topk_ids[i];
        unsigned int pos = offsets[expert_id] + atomicAdd(&counts[expert_id], 1);
        sorted_token_ids[pos] = (int)(i / topk);
        sorted_expert_ids[pos] = (int)expert_id;
        token_to_perm[i] = (int)pos;
    }
}

// 2026-09-25: Build the compacted work-list read by moe_fp8_grouped_gemm and
// moe_w8a8_grouped_gemm_pm4: one item per (expert, m_tile, n_tile) tile of an expert that
// has rows and a non-null weight pointer. The GEMMs skip a null weight pointer as well.
//   worklist[w*2 + 0] = expert_id
//   worklist[w*2 + 1] = (mt << 6) | nt  (m-tile and n-tile indices; the GEMMs decode >> 6 and & 0x3F)
//   total_tiles[0]    = number of items
// m_tile and n_tiles must match the GEMM's tile shape (PM4_M_TILE rows, PM4_N_TILE
// columns). The worklist capacity is not checked: the caller sizes it for the worst case.
// Only thread 0 works, emitting items in expert, then m_tile, then n_tile order.
// The GEMM that reads total_tiles and worklist must run on the same stream after this
// kernel; no event orders them.













extern "C" __global__ void moe_build_tile_worklist(
    const int* __restrict__ expert_offsets,
    const unsigned long long* __restrict__ B_weight_ptrs,
    unsigned int* __restrict__ worklist,
    int* __restrict__ total_tiles,
    unsigned int num_experts,
    unsigned int n_tiles,
    unsigned int m_tile
) {
    if (threadIdx.x != 0) return;

    unsigned int w = 0;
    for (unsigned int e = 0; e < num_experts; e++) {
        int m_start = expert_offsets[e];
        int M_e = expert_offsets[e + 1] - m_start;
        if (M_e <= 0 || B_weight_ptrs[e] == 0) continue;

        unsigned int mt_e = ((unsigned int)M_e + m_tile - 1) / m_tile;
        for (unsigned int mt = 0; mt < mt_e; mt++) {
            for (unsigned int nt = 0; nt < n_tiles; nt++) {
                // 2026-09-25: the packed word holds nt in its low 6 bits and mt in the upper 26.

                assert(mt < (1u << 26) && nt < 64u);
                worklist[w * 2 + 0] = e;
                worklist[w * 2 + 1] = (mt << 6) | nt;
                w++;
            }
        }
    }
    total_tiles[0] = (int)w;
}
