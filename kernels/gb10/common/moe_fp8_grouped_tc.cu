// SPDX-License-Identifier: MIT OR Apache-2.0

// 2026-09-28: Tensor-core grouped FP8 MoE decode: the gate+up (SiLU product) and down
// projections of moe_shared_expert_fused_fp8_grouped.cu on mma.sync m16n8k16 BF16 tiles,
// FP32 accumulation. 2026-10-02: the FP8 block-128 point of the weight-format-parameterized
// warp in moe_grouped_tc.cuh (Fp8Block128 below); moe_nvfp4_grouped_tc.cu is the NVFP4 point. Same grid contract (one block row per active expert, the first
// ceil(num_tokens / TC_ROWS) block rows the shared expert), same weights and scales, same
// outputs by sorted position.
//
// Why: on GB10 the CUDA-core kernels spend their power on the per-byte E4M3 table lookup,
// the scale multiply and the per-row FP32 products; here a weight byte costs a byte permute
// and two logic ops, and the products run on the tensor cores. Standalone at the
// Qwen3.6-35B-A3B shape (256 experts, top-8, uniform routing, dgx3): the same bytes per
// second within 1-5% and 30-45% fewer GPU-rail joules per layer (M = 1..64).
//
// Owner: gb10 kernels.
// Invariants:
// - Weights are row-major [N, K] FP8 E4M3 with FP32 block scales [N / 128, K / 128]; K and
//   N are multiples of 128 (the host checks `fp8_grouped_tc_shape_ok`).
// - A weight byte b becomes the BF16 whose bits are sign(b) | (b & 0x7F) << 4, which equals
//   E4M3(b) * 2^-120 exactly (normals and subnormals). Activations are multiplied by 2^60
//   (exact), and the 128-K block scale by 2^60, so every product and partial sum stays an
//   FP32 normal: products are E4M3 * A * 2^-60.
// - 2026-09-28: gate+up computes the FP32 SiLU product a and stores it as two BF16 terms of
//   a * 2^60, hi = BF16(a * 2^60) and lo = BF16(a * 2^60 - hi), in the SiLU buffer's FP32 row
//   space: row r holds N hi values then N lo values (2N BF16 = N FP32). The down projection
//   runs one MMA on each (hi first), so the product keeps about 16 bits of its mantissa where
//   one BF16 would keep 8, and the split is paid once per element, not once per down warp.
// - Inside a 64-wide K chunk, lane t = lane & 3 holds K = 16t .. 16t + 15 of its rows,
//   weights and activations alike (one 16-byte load each), and MMA j (0..3) takes
//   K = 16t + 4j + {0,1} as fragment slots 2t, 2t+1 and K = 16t + 4j + {2,3} as 2t+8, 2t+9.
//   The K order of a row's sum is therefore fixed by K alone.
// - Rows run TC_ROWS at a time as the MMA's N columns. Column r of an MMA depends only on
//   row r's activations, so a row's output bits do not depend on which rows share its
//   launch, its pass or its expert (padding rows are zero).
// - Grids: gate+up (N / TC_GU_COLS, cap + S), down (N / TC_DOWN_COLS, cap + S), block
//   TC_THREADS, S = ceil(num_tokens / TC_ROWS). TC_* must equal FP8_GROUPED_TC_* in
//   fp8_moe_grouped.rs.

#include <cuda_bf16.h>

#include "moe_grouped_tc.cuh"
#include "tc_weight_formats.cuh"

#define TC_WARPS 4
#define TC_THREADS (TC_WARPS * 32)
// 2026-09-28: m-tiles (16 output columns) per warp and 64-K chunks per load group. A warp
// keeps 2 groups in flight (prefetch); gate+up tiles are the gate and up rows of the same
// columns, so its 16 columns feed the SiLU product in registers.
#define TC_GU_MT 1
#define TC_DOWN_MT 2
#define TC_G 2
#define TC_GU_COLS (TC_WARPS * 16 * TC_GU_MT)
#define TC_DOWN_COLS (TC_WARPS * 16 * TC_DOWN_MT)

// 2026-09-28: Gate+up and SiLU of the routed experts and the shared expert. A: [num_tokens, K]
// BF16. act: routed hi|lo rows [pos, 2N] BF16 by sorted position; sh_act: shared hi|lo rows
// [token, 2N]; both in FP32-sized buffers.
// A null routed gate or up pointer makes that expert's act 0.
extern "C" __global__ void __launch_bounds__(TC_THREADS) moe_expert_gate_up_act_fp8_grouped_tc(
    const __nv_bfloat16* __restrict__ A,
    const unsigned long long* __restrict__ gate_weight_ptrs,
    const unsigned long long* __restrict__ gate_block_scale_ptrs,
    const unsigned long long* __restrict__ up_weight_ptrs,
    const unsigned long long* __restrict__ up_block_scale_ptrs,
    __nv_bfloat16* __restrict__ act,
    const int* __restrict__ expert_offsets,
    const int* __restrict__ sorted_token_ids,
    const int* __restrict__ active_experts,
    const int* __restrict__ active_count,
    const unsigned char* __restrict__ sh_gate_weight,
    const float* __restrict__ sh_gate_block_scale,
    const unsigned char* __restrict__ sh_up_weight,
    const float* __restrict__ sh_up_block_scale,
    __nv_bfloat16* __restrict__ sh_act,
    unsigned int N, unsigned int K, unsigned int cap, unsigned int num_tokens
) {
    bool is_shared;
    unsigned int expert, begin, end;
    if (!tc_block_rows(expert_offsets, active_experts, active_count, num_tokens,
                       &is_shared, &expert, &begin, &end)) return;
    const unsigned int f0 = blockIdx.x * TC_GU_COLS + (threadIdx.x >> 5) * 16 * TC_GU_MT;
    if (is_shared) {
        gtc_warp<Fp8Block128, true, TC_GU_MT, 1, TC_G>(
            A, sorted_token_ids, true, begin, end, {sh_gate_weight, sh_gate_block_scale},
            {sh_up_weight, sh_up_block_scale}, sh_act, N, K, f0);
        return;
    }
    const unsigned char* Wg = (const unsigned char*)gate_weight_ptrs[expert];
    const unsigned char* Wu = (const unsigned char*)up_weight_ptrs[expert];
    if (Wg == 0 || Wu == 0) {
        for (unsigned int pos = begin; pos < end; pos++)
            for (unsigned int i = threadIdx.x; i < TC_GU_COLS; i += TC_THREADS)
                for (unsigned int hl = 0; hl < 2; hl++)
                    act[(unsigned long long)pos * 2 * N + hl * N + blockIdx.x * TC_GU_COLS + i] = __float2bfloat16(0.0f);
        return;
    }
    gtc_warp_routed<Fp8Block128, true, TC_GU_MT, TC_G>(
        A, sorted_token_ids, false, begin, end, {Wg, (const float*)gate_block_scale_ptrs[expert]},
        {Wu, (const float*)up_block_scale_ptrs[expert]}, act, N, K, f0);
}

// 2026-09-28: Down projection of the hi|lo SiLU rows (act routed by position, sh_act
// shared by token) into C [pos, N] and sh_down_out [token, N] BF16. A null routed down pointer
// zeroes that expert's rows.
extern "C" __global__ void __launch_bounds__(TC_THREADS) moe_expert_down_act_fp8_grouped_tc(
    const __nv_bfloat16* __restrict__ act,
    const unsigned long long* __restrict__ weight_ptrs,
    const unsigned long long* __restrict__ block_scale_ptrs,
    __nv_bfloat16* __restrict__ C,
    const int* __restrict__ expert_offsets,
    const int* __restrict__ active_experts,
    const int* __restrict__ active_count,
    const __nv_bfloat16* __restrict__ sh_act,
    const unsigned char* __restrict__ sh_down_weight,
    const float* __restrict__ sh_down_block_scale,
    __nv_bfloat16* __restrict__ sh_down_out,
    unsigned int N, unsigned int K, unsigned int cap, unsigned int num_tokens
) {
    bool is_shared;
    unsigned int expert, begin, end;
    if (!tc_block_rows(expert_offsets, active_experts, active_count, num_tokens,
                       &is_shared, &expert, &begin, &end)) return;
    const unsigned int f0 = blockIdx.x * TC_DOWN_COLS + (threadIdx.x >> 5) * 16 * TC_DOWN_MT;
    if (is_shared) {
        gtc_warp<Fp8Block128, false, TC_DOWN_MT, 1, TC_G>(
            sh_act, nullptr, true, begin, end, {sh_down_weight, sh_down_block_scale}, {nullptr, nullptr},
            sh_down_out, N, K, f0);
        return;
    }
    const unsigned char* W = (const unsigned char*)weight_ptrs[expert];
    if (W == 0) {
        for (unsigned int pos = begin; pos < end; pos++)
            for (unsigned int i = threadIdx.x; i < TC_DOWN_COLS; i += TC_THREADS)
                C[(unsigned long long)pos * N + blockIdx.x * TC_DOWN_COLS + i] = __float2bfloat16(0.0f);
        return;
    }
    gtc_warp_routed<Fp8Block128, false, TC_DOWN_MT, TC_G>(
        act, nullptr, true, begin, end, {W, (const float*)block_scale_ptrs[expert]}, {nullptr, nullptr},
        C, N, K, f0);
}
