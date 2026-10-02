// SPDX-License-Identifier: MIT OR Apache-2.0

// 2026-10-02: Tensor-core grouped BF16 MoE decode (W16A16): the BF16 point (Bf16Dense) of the
// weight-format-parameterized warp in moe_grouped_tc.cuh, whose FP8 and NVFP4 points are
// moe_fp8_grouped_tc.cu and moe_nvfp4_grouped_tc.cu. It serves BF16 experts, among them the MTP
// drafter of nvidia/Qwen3.6-35B-A3B-NVFP4 (its MTP head is excluded from quantization), so the
// drafter's MoE runs once for every sequence of a batched propose.
//
// Owner: gb10 kernels.
// Invariants:
// - Weights are row-major [N, K] BF16 per expert (pointer tables), the shared expert's likewise;
//   K % 128 == 0 and N a multiple of the CTA's columns (the host checks
//   `bf16_grouped_tc_shape_ok`).
// - Products and sums: BF16 x BF16 on the tensor cores, FP32 accumulation; gate+up rounds gate and
//   up to BF16, stores the FP32 SiLU product as BF16 hi + lo; down runs one MMA on each.
// - A row's output bits do not depend on the other rows (moe_grouped_tc.cuh).
// - Grids: gate+up (N / BTC_GU_COLS, cap + S), down (N / BTC_DOWN_COLS, cap + S), block
//   BTC_THREADS, S = ceil(num_tokens / TC_ROWS). BTC_* must equal BF16_GROUPED_TC_* in
//   bf16_moe_grouped.rs.

#include <cuda_bf16.h>

#include "moe_grouped_tc.cuh"
#include "tc_weight_formats.cuh"

#define BTC_WARPS 4
#define BTC_THREADS (BTC_WARPS * 32)
// 2026-10-02: m-tiles (16 output columns) per warp and 32-K chunks per load group (a warp
// keeps 2 groups in flight). gate+up tiles are the gate and up rows of the same columns.
// K must be a multiple of 128 (`bf16_grouped_tc_shape_ok`).
#define BTC_GU_MT 1
#define BTC_DOWN_MT 2
// 2026-10-02: 32-K chunks, two per load group (four took 236-240 registers, two take 128-140).
#define BTC_GU_G 2
#define BTC_DOWN_G 2
#define BTC_GU_COLS (BTC_WARPS * 16 * BTC_GU_MT)
#define BTC_DOWN_COLS (BTC_WARPS * 16 * BTC_DOWN_MT)

// 2026-10-02: Gate+up and SiLU of the routed experts and the shared expert. A: [num_tokens, K]
// BF16. act: routed hi|lo rows [pos, 2N] BF16 by sorted position; sh_act: shared hi|lo rows
// [token, 2N]; both in FP32-sized buffers. Arguments as moe_expert_gate_up_act_nvfp4_grouped.
extern "C" __global__ void __launch_bounds__(BTC_THREADS) moe_expert_gate_up_act_bf16_grouped_tc(
    const __nv_bfloat16* __restrict__ A,
    const unsigned long long* __restrict__ gate_ptrs,
    const unsigned long long* __restrict__ up_ptrs,
    __nv_bfloat16* __restrict__ act,
    const int* __restrict__ expert_offsets,
    const int* __restrict__ sorted_token_ids,
    const int* __restrict__ active_experts,
    const int* __restrict__ active_count,
    const unsigned char* __restrict__ sh_gate,
    const unsigned char* __restrict__ sh_up,
    __nv_bfloat16* __restrict__ sh_act,
    unsigned int N, unsigned int K, unsigned int cap, unsigned int num_tokens
) {
    bool is_shared;
    unsigned int expert, begin, end;
    if (!tc_block_rows(expert_offsets, active_experts, active_count, num_tokens,
                       &is_shared, &expert, &begin, &end)) return;
    const unsigned int f0 = blockIdx.x * BTC_GU_COLS + (threadIdx.x >> 5) * 16 * BTC_GU_MT;
    if (is_shared) {
        gtc_warp<Bf16Dense, true, BTC_GU_MT, 1, BTC_GU_G>(
            A, sorted_token_ids, true, begin, end, {sh_gate}, {sh_up}, sh_act, N, K, f0);
        return;
    }
    const unsigned char* Pg = (const unsigned char*)gate_ptrs[expert];
    const unsigned char* Pu = (const unsigned char*)up_ptrs[expert];
    if (Pg == 0 || Pu == 0) {
        for (unsigned int pos = begin; pos < end; pos++)
            for (unsigned int i = threadIdx.x; i < BTC_GU_COLS; i += BTC_THREADS)
                for (unsigned int hl = 0; hl < 2; hl++)
                    act[(unsigned long long)pos * 2 * N + hl * N + blockIdx.x * BTC_GU_COLS + i] = __float2bfloat16(0.0f);
        return;
    }
    gtc_warp_routed<Bf16Dense, true, BTC_GU_MT, BTC_GU_G>(
        A, sorted_token_ids, false, begin, end, {Pg}, {Pu}, act, N, K, f0);
}

// 2026-10-02: Down projection of the hi|lo SiLU rows (act routed by position, sh_act shared by
// token) into C [pos, N] and sh_down_out [token, N] BF16. A null routed pointer zeroes that
// expert's rows.
extern "C" __global__ void __launch_bounds__(BTC_THREADS) moe_expert_down_act_bf16_grouped_tc(
    const __nv_bfloat16* __restrict__ act,
    const unsigned long long* __restrict__ down_ptrs,
    __nv_bfloat16* __restrict__ C,
    const int* __restrict__ expert_offsets,
    const int* __restrict__ active_experts,
    const int* __restrict__ active_count,
    const __nv_bfloat16* __restrict__ sh_act,
    const unsigned char* __restrict__ sh_down,
    __nv_bfloat16* __restrict__ sh_down_out,
    unsigned int N, unsigned int K, unsigned int cap, unsigned int num_tokens
) {
    bool is_shared;
    unsigned int expert, begin, end;
    if (!tc_block_rows(expert_offsets, active_experts, active_count, num_tokens,
                       &is_shared, &expert, &begin, &end)) return;
    const unsigned int f0 = blockIdx.x * BTC_DOWN_COLS + (threadIdx.x >> 5) * 16 * BTC_DOWN_MT;
    if (is_shared) {
        gtc_warp<Bf16Dense, false, BTC_DOWN_MT, 1, BTC_DOWN_G>(
            sh_act, nullptr, true, begin, end, {sh_down}, {nullptr}, sh_down_out, N, K, f0);
        return;
    }
    const unsigned char* P = (const unsigned char*)down_ptrs[expert];
    if (P == 0) {
        for (unsigned int pos = begin; pos < end; pos++)
            for (unsigned int i = threadIdx.x; i < BTC_DOWN_COLS; i += BTC_THREADS)
                C[(unsigned long long)pos * N + blockIdx.x * BTC_DOWN_COLS + i] = __float2bfloat16(0.0f);
        return;
    }
    gtc_warp_routed<Bf16Dense, false, BTC_DOWN_MT, BTC_DOWN_G>(
        act, nullptr, true, begin, end, {P}, {nullptr}, C, N, K, f0);
}
