// SPDX-License-Identifier: MIT OR Apache-2.0

// 2026-09-25: Two-token BF16 MoE GEMVs: gate+up, then SiLU(gate) * up and down. Weights
// are row-major [N, K] BF16, unscaled.
//
// Owner: gb10 kernels.
// Invariants:
// - blockIdx.y < 2 * top_k is routed slot y (token y / top_k, slot y % top_k) and writes
//   row y of gate_out/up_out/C. blockIdx.y == 2 * top_k is the shared expert: one block
//   reads the shared weight once and computes both tokens, writing rows 0 and 1 of
//   sh_gate_out/sh_up_out/sh_down_out. moe_weighted_sum_blend_batch2 reads this layout.
// - A null routed or shared weight pointer writes zeros.
// - K must be a multiple of 8, and at most 1024 in the down kernel (s_act holds 2048
//   floats and the shared block stores 2K); neither is checked.
// - Launch: grid (ceil(N / 8), 2 * top_k + 1, 2 for gate+up or 1 for down), block 128.













#include <cuda_bf16.h>

#define BLOCK_SIZE 128
#define N_PER_BLOCK 4
#define WARP_SIZE 32

// 2026-09-25: Gate+up for two tokens. blockIdx.z selects gate (0) or up (1).



extern "C" __global__ void moe_expert_gate_up_shared_bf16_batch2(
    const __nv_bfloat16* __restrict__ A,

    const unsigned long long* __restrict__ gate_weight_ptrs,
    __nv_bfloat16* __restrict__ gate_out,
    const unsigned long long* __restrict__ up_weight_ptrs,
    __nv_bfloat16* __restrict__ up_out,
    const unsigned int* __restrict__ expert_indices,

    const __nv_bfloat16* __restrict__ sh_gate_weight,
    __nv_bfloat16* __restrict__ sh_gate_out,
    const __nv_bfloat16* __restrict__ sh_up_weight,
    __nv_bfloat16* __restrict__ sh_up_out,
    unsigned int N, unsigned int K, unsigned int top_k
) {
    const unsigned int total_routed = 2 * top_k;
    const unsigned int y = blockIdx.y;
    const unsigned int proj = blockIdx.z;
    const bool is_shared = (y == total_routed);

    const unsigned int threads_per_out = BLOCK_SIZE / N_PER_BLOCK;
    const unsigned int local_out = threadIdx.x / threads_per_out;
    const unsigned int lane = threadIdx.x % threads_per_out;
    const unsigned int n1 = blockIdx.x * (N_PER_BLOCK * 2) + local_out * 2;
    const unsigned int n2 = n1 + 1;
    const unsigned int K8 = K / 8;

    if (is_shared) {


        const __nv_bfloat16* B_weight = (proj == 0) ? sh_gate_weight : sh_up_weight;
        __nv_bfloat16* C0 = ((proj == 0) ? sh_gate_out : sh_up_out);
        __nv_bfloat16* C1 = C0 + (unsigned long long)N;

        if (B_weight == 0) {
            const unsigned int n_base = blockIdx.x * (N_PER_BLOCK * 2);
            for (unsigned int i = threadIdx.x; i < N_PER_BLOCK * 2 && n_base + i < N; i += BLOCK_SIZE) {
                C0[n_base + i] = __float2bfloat16(0.0f);
                C1[n_base + i] = __float2bfloat16(0.0f);
            }
            return;
        }
        if (n1 >= N) return;
        const bool have_n2 = (n2 < N);
        const __nv_bfloat16* A0 = A;
        const __nv_bfloat16* A1 = A + (unsigned long long)K;

        float a0n1 = 0.0f, a0n2 = 0.0f, a1n1 = 0.0f, a1n2 = 0.0f;
        for (unsigned int k8 = lane; k8 < K8; k8 += threads_per_out) {
            uint4 w_n1 = ((const uint4*)(B_weight + (unsigned long long)n1 * K))[k8];
            uint4 w_n2;
            if (have_n2) w_n2 = ((const uint4*)(B_weight + (unsigned long long)n2 * K))[k8];
            else { w_n2.x = 0; w_n2.y = 0; w_n2.z = 0; w_n2.w = 0; }
            uint4 a0 = ((const uint4*)A0)[k8];
            uint4 a1 = ((const uint4*)A1)[k8];
            const unsigned int w1[4] = {w_n1.x, w_n1.y, w_n1.z, w_n1.w};
            const unsigned int w2[4] = {w_n2.x, w_n2.y, w_n2.z, w_n2.w};
            const unsigned int a0w[4] = {a0.x, a0.y, a0.z, a0.w};
            const unsigned int a1w[4] = {a1.x, a1.y, a1.z, a1.w};
            #pragma unroll
            for (int b = 0; b < 4; b++) {
                __nv_bfloat16 w1v0, w1v1, w2v0, w2v1, a0v0, a0v1, a1v0, a1v1;
                *(unsigned short*)&w1v0 = (unsigned short)(w1[b] & 0xFFFF);
                *(unsigned short*)&w1v1 = (unsigned short)(w1[b] >> 16);
                *(unsigned short*)&w2v0 = (unsigned short)(w2[b] & 0xFFFF);
                *(unsigned short*)&w2v1 = (unsigned short)(w2[b] >> 16);
                *(unsigned short*)&a0v0 = (unsigned short)(a0w[b] & 0xFFFF);
                *(unsigned short*)&a0v1 = (unsigned short)(a0w[b] >> 16);
                *(unsigned short*)&a1v0 = (unsigned short)(a1w[b] & 0xFFFF);
                *(unsigned short*)&a1v1 = (unsigned short)(a1w[b] >> 16);
                float wf1_0 = __bfloat162float(w1v0), wf1_1 = __bfloat162float(w1v1);
                float wf2_0 = __bfloat162float(w2v0), wf2_1 = __bfloat162float(w2v1);
                float a0f0 = __bfloat162float(a0v0), a0f1 = __bfloat162float(a0v1);
                float a1f0 = __bfloat162float(a1v0), a1f1 = __bfloat162float(a1v1);
                a0n1 += a0f0 * wf1_0 + a0f1 * wf1_1;
                a0n2 += a0f0 * wf2_0 + a0f1 * wf2_1;
                a1n1 += a1f0 * wf1_0 + a1f1 * wf1_1;
                a1n2 += a1f0 * wf2_0 + a1f1 * wf2_1;
            }
        }
        #pragma unroll
        for (int off = WARP_SIZE / 2; off > 0; off >>= 1) {
            a0n1 += __shfl_down_sync(0xFFFFFFFF, a0n1, off);
            a1n1 += __shfl_down_sync(0xFFFFFFFF, a1n1, off);
        }
        if (lane == 0) { C0[n1] = __float2bfloat16(a0n1); C1[n1] = __float2bfloat16(a1n1); }
        if (have_n2) {
            #pragma unroll
            for (int off = WARP_SIZE / 2; off > 0; off >>= 1) {
                a0n2 += __shfl_down_sync(0xFFFFFFFF, a0n2, off);
                a1n2 += __shfl_down_sync(0xFFFFFFFF, a1n2, off);
            }
            if (lane == 0) { C0[n2] = __float2bfloat16(a0n2); C1[n2] = __float2bfloat16(a1n2); }
        }
        return;
    }


    const unsigned int token = y / top_k;
    const unsigned int expert_slot = y % top_k;
    const __nv_bfloat16* A_token = A + (unsigned long long)token * K;
    const unsigned int expert_id = expert_indices[token * top_k + expert_slot];
    const unsigned int flat_slot = token * top_k + expert_slot;
    const __nv_bfloat16* B_weight;
    __nv_bfloat16* C;
    if (proj == 0) {
        B_weight = (const __nv_bfloat16*)gate_weight_ptrs[expert_id];
        C = gate_out + (unsigned long long)flat_slot * N;
    } else {
        B_weight = (const __nv_bfloat16*)up_weight_ptrs[expert_id];
        C = up_out + (unsigned long long)flat_slot * N;
    }

    if (B_weight == 0) {
        const unsigned int n_base = blockIdx.x * (N_PER_BLOCK * 2);
        for (unsigned int i = threadIdx.x; i < N_PER_BLOCK * 2 && n_base + i < N; i += BLOCK_SIZE)
            C[n_base + i] = __float2bfloat16(0.0f);
        return;
    }

    if (n1 >= N) return;
    const bool have_n2 = (n2 < N);
    float acc1 = 0.0f, acc2 = 0.0f;
    for (unsigned int k8 = lane; k8 < K8; k8 += threads_per_out) {
        uint4 a_data = ((const uint4*)A_token)[k8];
        uint4 w_n1 = ((const uint4*)(B_weight + (unsigned long long)n1 * K))[k8];
        uint4 w_n2;
        if (have_n2) w_n2 = ((const uint4*)(B_weight + (unsigned long long)n2 * K))[k8];
        else { w_n2.x = 0; w_n2.y = 0; w_n2.z = 0; w_n2.w = 0; }
        const unsigned int aw[4] = {a_data.x, a_data.y, a_data.z, a_data.w};
        const unsigned int w1[4] = {w_n1.x, w_n1.y, w_n1.z, w_n1.w};
        const unsigned int w2[4] = {w_n2.x, w_n2.y, w_n2.z, w_n2.w};
        #pragma unroll
        for (int b = 0; b < 4; b++) {
            __nv_bfloat16 av0, av1, w1v0, w1v1, w2v0, w2v1;
            *(unsigned short*)&av0 = (unsigned short)(aw[b] & 0xFFFF);
            *(unsigned short*)&av1 = (unsigned short)(aw[b] >> 16);
            *(unsigned short*)&w1v0 = (unsigned short)(w1[b] & 0xFFFF);
            *(unsigned short*)&w1v1 = (unsigned short)(w1[b] >> 16);
            *(unsigned short*)&w2v0 = (unsigned short)(w2[b] & 0xFFFF);
            *(unsigned short*)&w2v1 = (unsigned short)(w2[b] >> 16);
            float af0 = __bfloat162float(av0), af1 = __bfloat162float(av1);
            float wf1_0 = __bfloat162float(w1v0), wf1_1 = __bfloat162float(w1v1);
            float wf2_0 = __bfloat162float(w2v0), wf2_1 = __bfloat162float(w2v1);
            acc1 += af0 * wf1_0 + af1 * wf1_1;
            acc2 += af0 * wf2_0 + af1 * wf2_1;
        }
    }
    #pragma unroll
    for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1)
        acc1 += __shfl_down_sync(0xFFFFFFFF, acc1, offset);
    if (lane == 0) C[n1] = __float2bfloat16(acc1);
    if (have_n2) {
        #pragma unroll
        for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1)
            acc2 += __shfl_down_sync(0xFFFFFFFF, acc2, offset);
        if (lane == 0) C[n2] = __float2bfloat16(acc2);
    }
}

// 2026-09-25: SiLU(gate) * up into s_act, then the down GEMV, for two tokens.





extern "C" __global__ void moe_expert_silu_down_shared_bf16_batch2(
    const __nv_bfloat16* __restrict__ gate_out,
    const __nv_bfloat16* __restrict__ up_out,
    const unsigned long long* __restrict__ weight_ptrs,
    __nv_bfloat16* __restrict__ C,
    const unsigned int* __restrict__ expert_indices,

    const __nv_bfloat16* __restrict__ sh_gate_in,
    const __nv_bfloat16* __restrict__ sh_up_in,
    const __nv_bfloat16* __restrict__ sh_down_weight,
    __nv_bfloat16* __restrict__ sh_down_out,
    unsigned int N, unsigned int K, unsigned int top_k
) {
    const unsigned int total_routed = 2 * top_k;
    const unsigned int y = blockIdx.y;
    const bool is_shared = (y == total_routed);

    const unsigned int threads_per_out = BLOCK_SIZE / N_PER_BLOCK;
    const unsigned int local_out = threadIdx.x / threads_per_out;
    const unsigned int lane = threadIdx.x % threads_per_out;
    const unsigned int n1 = blockIdx.x * (N_PER_BLOCK * 2) + local_out * 2;
    const unsigned int n2 = n1 + 1;
    const unsigned int K8 = K / 8;






    __shared__ float s_act[2048];

    if (is_shared) {

        __nv_bfloat16* O0 = sh_down_out;
        __nv_bfloat16* O1 = sh_down_out + (unsigned long long)N;
        if (sh_down_weight == 0) {
            const unsigned int n_base = blockIdx.x * (N_PER_BLOCK * 2);
            for (unsigned int i = threadIdx.x; i < N_PER_BLOCK * 2 && n_base + i < N; i += BLOCK_SIZE) {
                O0[n_base + i] = __float2bfloat16(0.0f);
                O1[n_base + i] = __float2bfloat16(0.0f);
            }
            return;
        }
        const __nv_bfloat16* B_weight = sh_down_weight;
        // 2026-09-25: s_act[0, K) holds token 0's activation and s_act[K, 2K) token 1's.
        for (unsigned int i = threadIdx.x; i < K; i += BLOCK_SIZE) {
            float g0 = __bfloat162float(sh_gate_in[i]);
            float u0 = __bfloat162float(sh_up_in[i]);
            s_act[i] = (g0 / (1.0f + __expf(-g0))) * u0;
            float g1 = __bfloat162float(sh_gate_in[(unsigned long long)K + i]);
            float u1 = __bfloat162float(sh_up_in[(unsigned long long)K + i]);
            s_act[K + i] = (g1 / (1.0f + __expf(-g1))) * u1;
        }
        __syncthreads();
        if (n1 >= N) return;
        const bool have_n2 = (n2 < N);
        float a0n1 = 0.0f, a0n2 = 0.0f, a1n1 = 0.0f, a1n2 = 0.0f;
        for (unsigned int k8 = lane; k8 < K8; k8 += threads_per_out) {
            const unsigned int base_k = k8 * 8;
            uint4 w_n1 = ((const uint4*)(B_weight + (unsigned long long)n1 * K))[k8];
            uint4 w_n2;
            if (have_n2) w_n2 = ((const uint4*)(B_weight + (unsigned long long)n2 * K))[k8];
            else { w_n2.x = 0; w_n2.y = 0; w_n2.z = 0; w_n2.w = 0; }
            const unsigned int w1[4] = {w_n1.x, w_n1.y, w_n1.z, w_n1.w};
            const unsigned int w2[4] = {w_n2.x, w_n2.y, w_n2.z, w_n2.w};
            #pragma unroll
            for (int b = 0; b < 4; b++) {
                __nv_bfloat16 w1v0, w1v1, w2v0, w2v1;
                *(unsigned short*)&w1v0 = (unsigned short)(w1[b] & 0xFFFF);
                *(unsigned short*)&w1v1 = (unsigned short)(w1[b] >> 16);
                *(unsigned short*)&w2v0 = (unsigned short)(w2[b] & 0xFFFF);
                *(unsigned short*)&w2v1 = (unsigned short)(w2[b] >> 16);
                float wf1_0 = __bfloat162float(w1v0), wf1_1 = __bfloat162float(w1v1);
                float wf2_0 = __bfloat162float(w2v0), wf2_1 = __bfloat162float(w2v1);
                float al0 = s_act[base_k + b * 2],       al1 = s_act[base_k + b * 2 + 1];
                float bl0 = s_act[K + base_k + b * 2],   bl1 = s_act[K + base_k + b * 2 + 1];
                a0n1 += al0 * wf1_0 + al1 * wf1_1;
                a0n2 += al0 * wf2_0 + al1 * wf2_1;
                a1n1 += bl0 * wf1_0 + bl1 * wf1_1;
                a1n2 += bl0 * wf2_0 + bl1 * wf2_1;
            }
        }
        #pragma unroll
        for (int off = WARP_SIZE / 2; off > 0; off >>= 1) {
            a0n1 += __shfl_down_sync(0xFFFFFFFF, a0n1, off);
            a1n1 += __shfl_down_sync(0xFFFFFFFF, a1n1, off);
        }
        if (lane == 0) { O0[n1] = __float2bfloat16(a0n1); O1[n1] = __float2bfloat16(a1n1); }
        if (have_n2) {
            #pragma unroll
            for (int off = WARP_SIZE / 2; off > 0; off >>= 1) {
                a0n2 += __shfl_down_sync(0xFFFFFFFF, a0n2, off);
                a1n2 += __shfl_down_sync(0xFFFFFFFF, a1n2, off);
            }
            if (lane == 0) { O0[n2] = __float2bfloat16(a0n2); O1[n2] = __float2bfloat16(a1n2); }
        }
        return;
    }


    const unsigned int token = y / top_k;
    const unsigned int expert_slot = y % top_k;
    const unsigned int expert_id = expert_indices[token * top_k + expert_slot];
    const unsigned int flat_slot = token * top_k + expert_slot;
    const __nv_bfloat16* B_weight = (const __nv_bfloat16*)weight_ptrs[expert_id];
    const __nv_bfloat16* g_ptr = gate_out + (unsigned long long)flat_slot * K;
    const __nv_bfloat16* u_ptr = up_out + (unsigned long long)flat_slot * K;
    if (B_weight == 0) {
        const unsigned int n_base = blockIdx.x * (N_PER_BLOCK * 2);
        for (unsigned int i = threadIdx.x; i < N_PER_BLOCK * 2 && n_base + i < N; i += BLOCK_SIZE)
            C[(unsigned long long)flat_slot * N + n_base + i] = __float2bfloat16(0.0f);
        return;
    }


    for (unsigned int i = threadIdx.x; i < K; i += BLOCK_SIZE) {
        float gf = __bfloat162float(g_ptr[i]);
        float uf = __bfloat162float(u_ptr[i]);
        s_act[i] = (gf / (1.0f + __expf(-gf))) * uf;
    }
    __syncthreads();

    if (n1 >= N) return;
    const bool have_n2 = (n2 < N);
    float acc1 = 0.0f, acc2 = 0.0f;
    for (unsigned int k8 = lane; k8 < K8; k8 += threads_per_out) {
        const unsigned int base_k = k8 * 8;
        uint4 w_n1 = ((const uint4*)(B_weight + (unsigned long long)n1 * K))[k8];
        uint4 w_n2;
        if (have_n2) w_n2 = ((const uint4*)(B_weight + (unsigned long long)n2 * K))[k8];
        else { w_n2.x = 0; w_n2.y = 0; w_n2.z = 0; w_n2.w = 0; }
        const unsigned int w1[4] = {w_n1.x, w_n1.y, w_n1.z, w_n1.w};
        const unsigned int w2[4] = {w_n2.x, w_n2.y, w_n2.z, w_n2.w};
        #pragma unroll
        for (int b = 0; b < 4; b++) {
            __nv_bfloat16 w1v0, w1v1, w2v0, w2v1;
            *(unsigned short*)&w1v0 = (unsigned short)(w1[b] & 0xFFFF);
            *(unsigned short*)&w1v1 = (unsigned short)(w1[b] >> 16);
            *(unsigned short*)&w2v0 = (unsigned short)(w2[b] & 0xFFFF);
            *(unsigned short*)&w2v1 = (unsigned short)(w2[b] >> 16);
            float wf1_0 = __bfloat162float(w1v0), wf1_1 = __bfloat162float(w1v1);
            float wf2_0 = __bfloat162float(w2v0), wf2_1 = __bfloat162float(w2v1);
            float al0 = s_act[base_k + b * 2];
            float al1 = s_act[base_k + b * 2 + 1];
            acc1 += al0 * wf1_0 + al1 * wf1_1;
            acc2 += al0 * wf2_0 + al1 * wf2_1;
        }
    }

    __nv_bfloat16* out = C + (unsigned long long)flat_slot * N;
    #pragma unroll
    for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1)
        acc1 += __shfl_down_sync(0xFFFFFFFF, acc1, offset);
    if (lane == 0) out[n1] = __float2bfloat16(acc1);
    if (have_n2) {
        #pragma unroll
        for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1)
            acc2 += __shfl_down_sync(0xFFFFFFFF, acc2, offset);
        if (lane == 0) out[n2] = __float2bfloat16(acc2);
    }
}
