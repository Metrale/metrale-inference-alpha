// SPDX-License-Identifier: MIT OR Apache-2.0

// 2026-09-25: LoRA fold for the MoE experts on the decode path, over unsorted slot-major rows:
//   base_out[row, :] += scale[e] * bf16(bf16(x[x_row, :] @ A_e^T) @ B_e^T),   e = indices[row]
// for every flat (token, slot) row below n_slots. moe_lora_gather_bgmv_shrink writes
// xa = bf16(x @ A_e^T); moe_lora_gather_bgmv_expand_fold adds the delta into base_out.
//
// Owner: gb10 kernels.
// Invariants:
// - Grid (ceil(outputs / 4), n_slots), block 256: blockIdx.y is the row, and each of the
//   four outputs of a block gets 64 threads (two warps).
// - A row is left unchanged when row_adapter is non-NULL and row_adapter[row / top_k] < 0,
//   when e >= n_experts, or when the expert's a_table / b_table entry is 0.
// - x_row is row when x_gather is 0 (x is [n_slots, k_in], the down input) and row / top_k
//   when it is 1 (x is one row per token, the gate/up input).
// - Both contractions run at max_rank: A_e is [max_rank, k_in] and B_e is [n_out, max_rank];
//   base_out is [n_slots, n_out].
// - Per row, the operations and their order match moe_lora_grouped_down.cu, the prefill
//   fold: the same dot-product reduction, BF16 rounding of xa and of the delta, then
//   o + scale * delta in FP32.
// - Each (row, output) element is written by one thread, so the fold needs no atomics.





















#include <cuda_bf16.h>

#define GBGMV_BLOCK_SIZE 256
#define GBGMV_N_PER_BLOCK 4
#define GBGMV_WARP_SIZE 32
#define GBGMV_VEC_SIZE 8








extern "C" __global__ void moe_lora_gather_bgmv_shrink(
    const __nv_bfloat16* __restrict__ x,
    const unsigned int* __restrict__ indices,
    const int* __restrict__ row_adapter,
    const unsigned long long* __restrict__ a_table,
    __nv_bfloat16* __restrict__ xa,
    unsigned int n_slots,
    unsigned int top_k,
    unsigned int n_experts,
    unsigned int max_rank,
    unsigned int k_in,
    unsigned int x_gather
) {
    const unsigned int row = blockIdx.y;
    if (row >= n_slots) return;

    if (row_adapter != nullptr && row_adapter[row / top_k] < 0) return;
    const unsigned int e = indices[row];
    if (e >= n_experts) return;
    const unsigned long long a_addr = a_table[e];
    if (a_addr == 0ULL) return;

    const __nv_bfloat16* A_base = (const __nv_bfloat16*)a_addr;

    const unsigned int threads_per_out = GBGMV_BLOCK_SIZE / GBGMV_N_PER_BLOCK;
    const unsigned int local_out = threadIdx.x / threads_per_out;
    const unsigned int lane = threadIdx.x % threads_per_out;

    const unsigned int n = blockIdx.x * GBGMV_N_PER_BLOCK + local_out;
    if (n >= max_rank) return;





    const unsigned int x_row = x_gather ? (row / top_k) : row;
    const __nv_bfloat16* Arow = x + (unsigned long long)x_row * k_in;
    const __nv_bfloat16* Brow = A_base + (unsigned long long)n * k_in;

    float acc = 0.0f;
    const unsigned int K_VEC = k_in / GBGMV_VEC_SIZE;
    const uint4* A_vec = (const uint4*)Arow;
    const uint4* B_vec = (const uint4*)Brow;

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
    {
        const unsigned int tail_start = K_VEC * GBGMV_VEC_SIZE;
        for (unsigned int k = tail_start + lane; k < k_in; k += threads_per_out) {
            acc += __bfloat162float(Arow[k]) * __bfloat162float(Brow[k]);
        }
    }

    const unsigned int warp_lane = threadIdx.x % GBGMV_WARP_SIZE;
    #pragma unroll
    for (int offset = GBGMV_WARP_SIZE / 2; offset > 0; offset >>= 1) {
        acc += __shfl_down_sync(0xFFFFFFFF, acc, offset);
    }

    __shared__ float smem[GBGMV_N_PER_BLOCK * 2];
    if (warp_lane == 0) {
        unsigned int smem_idx = local_out * 2 + (lane / GBGMV_WARP_SIZE);
        smem[smem_idx] = acc;
    }
    __syncthreads();

    if (lane == 0) {
        float result = smem[local_out * 2] + smem[local_out * 2 + 1];
        xa[(unsigned long long)row * max_rank + n] = __float2bfloat16(result);
    }
}









extern "C" __global__ void moe_lora_gather_bgmv_expand_fold(
    const __nv_bfloat16* __restrict__ xa,
    const unsigned int* __restrict__ indices,
    const int* __restrict__ row_adapter,
    const unsigned long long* __restrict__ b_table,
    const float* __restrict__ scale_table,
    __nv_bfloat16* __restrict__ base_out,
    unsigned int n_slots,
    unsigned int top_k,
    unsigned int n_experts,
    unsigned int n_out,
    unsigned int max_rank
) {
    const unsigned int row = blockIdx.y;
    if (row >= n_slots) return;
    if (row_adapter != nullptr && row_adapter[row / top_k] < 0) return;
    const unsigned int e = indices[row];
    if (e >= n_experts) return;
    const unsigned long long b_addr = b_table[e];
    if (b_addr == 0ULL) return;

    const __nv_bfloat16* B_base = (const __nv_bfloat16*)b_addr;

    const unsigned int threads_per_out = GBGMV_BLOCK_SIZE / GBGMV_N_PER_BLOCK;
    const unsigned int local_out = threadIdx.x / threads_per_out;
    const unsigned int lane = threadIdx.x % threads_per_out;

    const unsigned int n = blockIdx.x * GBGMV_N_PER_BLOCK + local_out;
    if (n >= n_out) return;

    const __nv_bfloat16* Arow = xa + (unsigned long long)row * max_rank;
    const __nv_bfloat16* Brow = B_base + (unsigned long long)n * max_rank;

    float acc = 0.0f;
    const unsigned int K_VEC = max_rank / GBGMV_VEC_SIZE;
    const uint4* A_vec = (const uint4*)Arow;
    const uint4* B_vec = (const uint4*)Brow;

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
    {
        const unsigned int tail_start = K_VEC * GBGMV_VEC_SIZE;
        for (unsigned int k = tail_start + lane; k < max_rank; k += threads_per_out) {
            acc += __bfloat162float(Arow[k]) * __bfloat162float(Brow[k]);
        }
    }

    const unsigned int warp_lane = threadIdx.x % GBGMV_WARP_SIZE;
    #pragma unroll
    for (int offset = GBGMV_WARP_SIZE / 2; offset > 0; offset >>= 1) {
        acc += __shfl_down_sync(0xFFFFFFFF, acc, offset);
    }

    __shared__ float smem[GBGMV_N_PER_BLOCK * 2];
    if (warp_lane == 0) {
        unsigned int smem_idx = local_out * 2 + (lane / GBGMV_WARP_SIZE);
        smem[smem_idx] = acc;
    }
    __syncthreads();

    if (lane == 0) {
        float result = smem[local_out * 2] + smem[local_out * 2 + 1];




        __nv_bfloat16 delta_bf = __float2bfloat16(result);
        float d = __bfloat162float(delta_bf);
        float sc = scale_table[e];
        __nv_bfloat16* dst = base_out + (unsigned long long)row * n_out + n;
        float o = __bfloat162float(*dst);
        *dst = __float2bfloat16(o + sc * d);
    }
}
