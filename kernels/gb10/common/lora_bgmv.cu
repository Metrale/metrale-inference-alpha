// SPDX-License-Identifier: MIT OR Apache-2.0

// 2026-09-25: Batched LoRA delta for decode, with one adapter slot per row:
//   out[i, :] += scale[s] * bf16(bf16(x[i, :] @ A_s^T) @ B_s^T),   s = seq_slot[i]
// lora_bgmv_shrink writes xa = bf16(x @ A_s^T); lora_bgmv_expand_fold adds the scaled
// delta into out in place.
//
// Owner: gb10 kernels.
// Invariants:
// - Grid (ceil(outputs / 4), N), block 256: blockIdx.y is the row, and each of the four
//   outputs of a block gets 64 threads (two warps).
// - A row whose seq_slot is negative, or whose slot has a 0 entry in a_table / b_table,
//   is left unchanged.
// - Both contractions run at max_rank: A_s is [max_rank, k_in] and B_s is
//   [n_out, max_rank], row-major. The pool zeroes the pad rows of A and pad columns of B
//   (LoraPair::max_rank in crates/model-layers/src/layers/ops/lora_delta.rs).
// - Per row, the operations and their order match apply_lora_delta at m = 1 in
//   lora_delta.rs: the dense_gemv_bf16 reduction for the shrink and for the expand, BF16
//   rounding of xa and of the delta, then the bf16_scaled_add fold o + scale * delta in
//   FP32. common/KERNEL.toml builds all three files with --fmad=false.
















#include <cuda_bf16.h>

#define BGMV_BLOCK_SIZE 256
#define BGMV_N_PER_BLOCK 4
#define BGMV_WARP_SIZE 32
#define BGMV_VEC_SIZE 8







extern "C" __global__ void lora_bgmv_shrink(
    const __nv_bfloat16* __restrict__ x,
    const int* __restrict__ seq_slot,
    const unsigned long long* __restrict__ a_table,
    __nv_bfloat16* __restrict__ xa,
    unsigned int N,
    unsigned int max_rank,
    unsigned int k_in,
    unsigned int x_row_stride
) {
    const unsigned int row = blockIdx.y;
    if (row >= N) return;
    const int s = seq_slot[row];
    if (s < 0) return;
    const unsigned long long a_addr = a_table[s];
    if (a_addr == 0ULL) return;

    const __nv_bfloat16* A_base = (const __nv_bfloat16*)a_addr;

    const unsigned int threads_per_out = BGMV_BLOCK_SIZE / BGMV_N_PER_BLOCK;
    const unsigned int local_out = threadIdx.x / threads_per_out;
    const unsigned int lane = threadIdx.x % threads_per_out;

    const unsigned int n = blockIdx.x * BGMV_N_PER_BLOCK + local_out;
    if (n >= max_rank) return;



    const __nv_bfloat16* Arow = x + (unsigned long long)row * x_row_stride;
    const __nv_bfloat16* Brow = A_base + (unsigned long long)n * k_in;

    float acc = 0.0f;
    const unsigned int K_VEC = k_in / BGMV_VEC_SIZE;
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
        const unsigned int tail_start = K_VEC * BGMV_VEC_SIZE;
        for (unsigned int k = tail_start + lane; k < k_in; k += threads_per_out) {
            acc += __bfloat162float(Arow[k]) * __bfloat162float(Brow[k]);
        }
    }

    const unsigned int warp_lane = threadIdx.x % BGMV_WARP_SIZE;
    #pragma unroll
    for (int offset = BGMV_WARP_SIZE / 2; offset > 0; offset >>= 1) {
        acc += __shfl_down_sync(0xFFFFFFFF, acc, offset);
    }

    __shared__ float smem[BGMV_N_PER_BLOCK * 2];
    if (warp_lane == 0) {
        unsigned int smem_idx = local_out * 2 + (lane / BGMV_WARP_SIZE);
        smem[smem_idx] = acc;
    }
    __syncthreads();

    if (lane == 0) {
        float result = smem[local_out * 2] + smem[local_out * 2 + 1];
        xa[(unsigned long long)row * max_rank + n] = __float2bfloat16(result);
    }
}












extern "C" __global__ void lora_bgmv_expand_fold(
    const __nv_bfloat16* __restrict__ xa,
    const int* __restrict__ seq_slot,
    const unsigned long long* __restrict__ b_table,
    const float* __restrict__ scale_table,
    __nv_bfloat16* __restrict__ base_out,
    unsigned int N,
    unsigned int n_out,
    unsigned int max_rank,
    unsigned int out_row_stride
) {
    const unsigned int row = blockIdx.y;
    if (row >= N) return;
    const int s = seq_slot[row];
    if (s < 0) return;
    const unsigned long long b_addr = b_table[s];
    if (b_addr == 0ULL) return;

    const __nv_bfloat16* B_base = (const __nv_bfloat16*)b_addr;

    const unsigned int threads_per_out = BGMV_BLOCK_SIZE / BGMV_N_PER_BLOCK;
    const unsigned int local_out = threadIdx.x / threads_per_out;
    const unsigned int lane = threadIdx.x % threads_per_out;

    const unsigned int n = blockIdx.x * BGMV_N_PER_BLOCK + local_out;
    if (n >= n_out) return;



    const __nv_bfloat16* Arow = xa + (unsigned long long)row * max_rank;
    const __nv_bfloat16* Brow = B_base + (unsigned long long)n * max_rank;

    float acc = 0.0f;
    const unsigned int K_VEC = max_rank / BGMV_VEC_SIZE;
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
        const unsigned int tail_start = K_VEC * BGMV_VEC_SIZE;
        for (unsigned int k = tail_start + lane; k < max_rank; k += threads_per_out) {
            acc += __bfloat162float(Arow[k]) * __bfloat162float(Brow[k]);
        }
    }

    const unsigned int warp_lane = threadIdx.x % BGMV_WARP_SIZE;
    #pragma unroll
    for (int offset = BGMV_WARP_SIZE / 2; offset > 0; offset >>= 1) {
        acc += __shfl_down_sync(0xFFFFFFFF, acc, offset);
    }

    __shared__ float smem[BGMV_N_PER_BLOCK * 2];
    if (warp_lane == 0) {
        unsigned int smem_idx = local_out * 2 + (lane / BGMV_WARP_SIZE);
        smem[smem_idx] = acc;
    }
    __syncthreads();

    if (lane == 0) {
        float result = smem[local_out * 2] + smem[local_out * 2 + 1];




        __nv_bfloat16 delta_bf = __float2bfloat16(result);
        float d = __bfloat162float(delta_bf);
        float sc = scale_table[s];
        __nv_bfloat16* dst = base_out + (unsigned long long)row * out_row_stride + n;
        float o = __bfloat162float(*dst);
        *dst = __float2bfloat16(o + sc * d);
    }
}
