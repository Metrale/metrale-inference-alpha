// SPDX-License-Identifier: MIT OR Apache-2.0
// forked-from: kernels/gb10/common/moe_w4a16_grouped_gemm.cu (2026-09-24; 1001 of 1031 lines differ, see kernels/FORKS.md)

// 2026-09-25: Compile stubs for the MoE W4A16 grouped GEMM entry points on HIP
// (gfx1151): every body returns without reading or writing memory.
//
// Owner: strix-hip kernels.
// Invariants:
// - Both strix-hip model targets replace this file ([shadow] in their
//   KERNEL.toml), so neither of them compiles it.





#include <cuda_bf16.h>

extern "C" __global__ void moe_w4a16_grouped_gemm(
    const __nv_bfloat16* __restrict__ A,
    const unsigned char* __restrict__ B_packed,
    const unsigned char* __restrict__ B_scale,
    const float scale2,
    __nv_bfloat16* __restrict__ C,
    const int* __restrict__ expert_offsets,
    unsigned int num_experts,
    unsigned int N,
    unsigned int K
) {

    (void)A; (void)B_packed; (void)B_scale; (void)scale2; (void)C;
    (void)expert_offsets; (void)num_experts; (void)N; (void)K;
}

extern "C" __global__ void moe_w4a16_grouped_gemm_ptrtable(
    const __nv_bfloat16* __restrict__ A,
    const unsigned long long* __restrict__ B_packed_ptrs,
    const unsigned long long* __restrict__ B_scale_ptrs,
    const float* __restrict__ scale2_vals,
    __nv_bfloat16* __restrict__ C,
    const int* __restrict__ expert_offsets,
    const int* __restrict__ sorted_token_ids,
    unsigned int num_experts,
    unsigned int N,
    unsigned int K
) {

    (void)A; (void)B_packed_ptrs; (void)B_scale_ptrs; (void)scale2_vals; (void)C;
    (void)expert_offsets; (void)sorted_token_ids; (void)num_experts; (void)N; (void)K;
}

extern "C" __global__ void moe_w4a16_grouped_gemm_ptrtable_t(
    const __nv_bfloat16* __restrict__ A,
    const unsigned long long* __restrict__ B_packed_ptrs,
    const unsigned long long* __restrict__ B_scale_ptrs,
    const float* __restrict__ scale2_vals,
    __nv_bfloat16* __restrict__ C,
    const int* __restrict__ expert_offsets,
    const int* __restrict__ sorted_token_ids,
    unsigned int num_experts,
    unsigned int N,
    unsigned int K
) {

    (void)A; (void)B_packed_ptrs; (void)B_scale_ptrs; (void)scale2_vals; (void)C;
    (void)expert_offsets; (void)sorted_token_ids; (void)num_experts; (void)N; (void)K;
}
