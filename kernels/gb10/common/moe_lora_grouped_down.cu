// SPDX-License-Identifier: MIT OR Apache-2.0

// 2026-09-25: MoE expert LoRA fold over expert-sorted rows, in two kernels (shrink, then
// expand and fold), for one row window [row_offset, row_end):
//   base_out[r, :] += scale_e * bf16( bf16(x[r', :] @ A_e^T) @ B_e^T )
// for each sorted row r of expert e, where r' is r (x_gather == 0: the down projection's
// sorted activations) or sorted_token_ids[r] (x_gather == 1: token-major gate/up input).
// expert_offsets is read on the device, so the launch needs no host copy of it.
//
// Owner: gb10 kernels.
// Invariants:
// - Per row, the arithmetic is lora_bgmv.cu's: uint4 loads, FP32 accumulation, the same
//   __shfl_down_sync order and two-warp shared-memory sum, a BF16 xa, a BF16 delta, then
//   base + scale_e * delta in FP32 as residual_add.cu bf16_scaled_add does.
// - One thread writes each output element, so no atomics are used, and a row's result
//   does not depend on which window holds it.
// - xa is indexed by r - row_offset. No other buffer is rebased by the window: base_out and
//   sorted_token_ids by r, x by r', moe_row_adapter by sorted_token_ids[r], expert_offsets by e.
// - A block returns early when its expert is at or past num_experts, has no rows in the
//   window, or has a zero table entry (no adapter for that expert).
// - With a non-null moe_row_adapter, a row whose token maps to a value below 0 is skipped.
//   The test depends only on r, so the whole block skips the row together and every
//   __syncthreads is reached by all threads that have not returned.





























#include <cuda_bf16.h>

#define MLG_BLOCK_SIZE 256
#define MLG_N_PER_BLOCK 4
#define MLG_WARP_SIZE 32
#define MLG_VEC_SIZE 8
#define MLG_M_TILE 64

// 2026-09-25: Shrink: xa[r - row_offset, n] = bf16(x[r', :] . A_e[n, :]) for n < max_rank,
// where A_e = a_expert_table[e] is a row-major [max_rank, k_in] BF16 matrix.
// Launch: grid (ceil(max_rank / 4), ceil((row_end - row_offset) / 64), table length),
// block 256; each block computes four outputs with 64 threads (two warps) apiece.


extern "C" __global__ void moe_lora_grouped_down_shrink(
    const __nv_bfloat16* __restrict__ x,
    const int* __restrict__ expert_offsets,
    const int* __restrict__ sorted_token_ids,
    const int* __restrict__ moe_row_adapter,
    const unsigned long long* __restrict__ a_expert_table,
    __nv_bfloat16* __restrict__ xa,
    unsigned int num_experts,
    unsigned int max_rank,
    unsigned int k_in,
    unsigned int x_gather,
    unsigned int row_offset,
    unsigned int row_end
) {
    const unsigned int e = blockIdx.z;
    if (e >= num_experts) return;
    const int m_start = expert_offsets[e];
    const int m_end = expert_offsets[e + 1];
    if (m_end <= m_start) return;
    const unsigned long long a_addr = a_expert_table[e];
    if (a_addr == 0ULL) return;

    // 2026-09-25: Clamp the expert's rows to the window. Tiles start at the clamped start, so
    // grid.y = ceil((row_end - row_offset) / 64) covers every expert's part of the window.



    const int win_start = max(m_start, (int)row_offset);
    const int win_end = min(m_end, (int)row_end);
    const int r0 = win_start + (int)blockIdx.y * MLG_M_TILE;
    if (r0 >= win_end) return;
    const int r1 = min(r0 + MLG_M_TILE, win_end);

    const __nv_bfloat16* A_base = (const __nv_bfloat16*)a_addr;

    const unsigned int threads_per_out = MLG_BLOCK_SIZE / MLG_N_PER_BLOCK;
    const unsigned int local_out = threadIdx.x / threads_per_out;
    const unsigned int lane = threadIdx.x % threads_per_out;
    const unsigned int warp_lane = threadIdx.x % MLG_WARP_SIZE;

    const unsigned int n = blockIdx.x * MLG_N_PER_BLOCK + local_out;
    if (n >= max_rank) return;

    const __nv_bfloat16* Brow = A_base + (unsigned long long)n * k_in;
    const uint4* B_vec = (const uint4*)Brow;
    const unsigned int K_VEC = k_in / MLG_VEC_SIZE;
    const unsigned int tail_start = K_VEC * MLG_VEC_SIZE;

    __shared__ float smem[MLG_N_PER_BLOCK * 2];

    for (int r = r0; r < r1; ++r) {

        if (moe_row_adapter != nullptr && moe_row_adapter[sorted_token_ids[r]] < 0) {
            continue;
        }





        const int x_row = x_gather ? sorted_token_ids[r] : r;
        const __nv_bfloat16* Arow = x + (unsigned long long)x_row * k_in;
        const uint4* A_vec = (const uint4*)Arow;

        float acc = 0.0f;
        for (unsigned int kv = lane; kv < K_VEC; kv += threads_per_out) {
            uint4 a_data = A_vec[kv];
            uint4 b_data = B_vec[kv];
            const unsigned int a_raw[4] = {a_data.x, a_data.y, a_data.z, a_data.w};
            const unsigned int b_raw[4] = {b_data.x, b_data.y, b_data.z, b_data.w};
            #pragma unroll
            for (int i = 0; i < 4; i++) {
                __nv_bfloat16 a_lo, a_hi, b_lo, b_hi;
                *(unsigned short*)&a_lo = (unsigned short)(a_raw[i] & 0xFFFF);
                *(unsigned short*)&a_hi = (unsigned short)(a_raw[i] >> 16);
                *(unsigned short*)&b_lo = (unsigned short)(b_raw[i] & 0xFFFF);
                *(unsigned short*)&b_hi = (unsigned short)(b_raw[i] >> 16);
                acc += __bfloat162float(a_lo) * __bfloat162float(b_lo);
                acc += __bfloat162float(a_hi) * __bfloat162float(b_hi);
            }
        }
        for (unsigned int k = tail_start + lane; k < k_in; k += threads_per_out) {
            acc += __bfloat162float(Arow[k]) * __bfloat162float(Brow[k]);
        }

        #pragma unroll
        for (int offset = MLG_WARP_SIZE / 2; offset > 0; offset >>= 1) {
            acc += __shfl_down_sync(0xFFFFFFFF, acc, offset);
        }
        if (warp_lane == 0) {
            smem[local_out * 2 + (lane / MLG_WARP_SIZE)] = acc;
        }
        __syncthreads();
        if (lane == 0) {
            float result = smem[local_out * 2] + smem[local_out * 2 + 1];
            xa[(unsigned long long)(r - (int)row_offset) * max_rank + n] = __float2bfloat16(result);
        }
        __syncthreads();
    }
}

// 2026-09-25: Expand and fold: base_out[r, n] += scale_e * bf16(xa[r - row_offset, :] . B_e[n, :])
// for n < n_out, where B_e = b_expert_table[e] is an [n_out, max_rank] BF16 matrix and
// scale_e = scale_expert_table[e].
// Launch: grid (ceil(n_out / 4), ceil((row_end - row_offset) / 64), table length), block 256.



extern "C" __global__ void moe_lora_grouped_down_expand_fold(
    const __nv_bfloat16* __restrict__ xa,
    const int* __restrict__ expert_offsets,
    const int* __restrict__ sorted_token_ids,
    const int* __restrict__ moe_row_adapter,
    const unsigned long long* __restrict__ b_expert_table,
    const float* __restrict__ scale_expert_table,
    __nv_bfloat16* __restrict__ base_out,
    unsigned int num_experts,
    unsigned int n_out,
    unsigned int max_rank,
    unsigned int row_offset,
    unsigned int row_end
) {
    const unsigned int e = blockIdx.z;
    if (e >= num_experts) return;
    const int m_start = expert_offsets[e];
    const int m_end = expert_offsets[e + 1];
    if (m_end <= m_start) return;
    const unsigned long long b_addr = b_expert_table[e];
    if (b_addr == 0ULL) return;



    const int win_start = max(m_start, (int)row_offset);
    const int win_end = min(m_end, (int)row_end);
    const int r0 = win_start + (int)blockIdx.y * MLG_M_TILE;
    if (r0 >= win_end) return;
    const int r1 = min(r0 + MLG_M_TILE, win_end);

    const __nv_bfloat16* B_base = (const __nv_bfloat16*)b_addr;
    const float scale_e = scale_expert_table[e];

    const unsigned int threads_per_out = MLG_BLOCK_SIZE / MLG_N_PER_BLOCK;
    const unsigned int local_out = threadIdx.x / threads_per_out;
    const unsigned int lane = threadIdx.x % threads_per_out;
    const unsigned int warp_lane = threadIdx.x % MLG_WARP_SIZE;

    const unsigned int n = blockIdx.x * MLG_N_PER_BLOCK + local_out;
    if (n >= n_out) return;

    const __nv_bfloat16* Brow = B_base + (unsigned long long)n * max_rank;
    const uint4* B_vec = (const uint4*)Brow;
    const unsigned int K_VEC = max_rank / MLG_VEC_SIZE;
    const unsigned int tail_start = K_VEC * MLG_VEC_SIZE;

    __shared__ float smem[MLG_N_PER_BLOCK * 2];

    for (int r = r0; r < r1; ++r) {
        if (moe_row_adapter != nullptr && moe_row_adapter[sorted_token_ids[r]] < 0) {
            continue;
        }
        const __nv_bfloat16* Arow = xa + (unsigned long long)(r - (int)row_offset) * max_rank;
        const uint4* A_vec = (const uint4*)Arow;

        float acc = 0.0f;
        for (unsigned int kv = lane; kv < K_VEC; kv += threads_per_out) {
            uint4 a_data = A_vec[kv];
            uint4 b_data = B_vec[kv];
            const unsigned int a_raw[4] = {a_data.x, a_data.y, a_data.z, a_data.w};
            const unsigned int b_raw[4] = {b_data.x, b_data.y, b_data.z, b_data.w};
            #pragma unroll
            for (int i = 0; i < 4; i++) {
                __nv_bfloat16 a_lo, a_hi, b_lo, b_hi;
                *(unsigned short*)&a_lo = (unsigned short)(a_raw[i] & 0xFFFF);
                *(unsigned short*)&a_hi = (unsigned short)(a_raw[i] >> 16);
                *(unsigned short*)&b_lo = (unsigned short)(b_raw[i] & 0xFFFF);
                *(unsigned short*)&b_hi = (unsigned short)(b_raw[i] >> 16);
                acc += __bfloat162float(a_lo) * __bfloat162float(b_lo);
                acc += __bfloat162float(a_hi) * __bfloat162float(b_hi);
            }
        }
        for (unsigned int k = tail_start + lane; k < max_rank; k += threads_per_out) {
            acc += __bfloat162float(Arow[k]) * __bfloat162float(Brow[k]);
        }

        #pragma unroll
        for (int offset = MLG_WARP_SIZE / 2; offset > 0; offset >>= 1) {
            acc += __shfl_down_sync(0xFFFFFFFF, acc, offset);
        }
        if (warp_lane == 0) {
            smem[local_out * 2 + (lane / MLG_WARP_SIZE)] = acc;
        }
        __syncthreads();
        if (lane == 0) {
            float result = smem[local_out * 2] + smem[local_out * 2 + 1];





            __nv_bfloat16 delta_bf = __float2bfloat16(result);
            float d = __bfloat162float(delta_bf);
            __nv_bfloat16* dst = base_out + (unsigned long long)r * n_out + n;
            float o = __bfloat162float(*dst);
            *dst = __float2bfloat16(o + scale_e * d);
        }
        __syncthreads();
    }
}
