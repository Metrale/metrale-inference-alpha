// SPDX-License-Identifier: MIT OR Apache-2.0

// 2026-09-25: Single-token BF16 MoE GEMVs with the shared expert as blockIdx.y == top_k:
// gate+up, then SiLU(gate) * up and down. Weights are row-major [N, K] BF16, unscaled.
//
// Owner: gb10 kernels.
// Invariants:
// - blockIdx.y < top_k is routed slot y (expert expert_indices[y], output row y).
// - A null routed weight pointer writes zeros; a null sh_gate_weight zeroes the shared
//   gate+up outputs. sh_down_weight is not checked.
// - K must be a multiple of 8: inputs past the last multiple of 8 are not read.
// - Launch: grid (ceil(N / 8), top_k + 1, 2 for gate+up or 1 for down), block 128.

#include <cuda_bf16.h>

#define BLOCK_SIZE 128
#define N_PER_BLOCK 4
#define WARP_SIZE 32

// 2026-09-25: Gate+up. blockIdx.z selects gate (0) or up (1); each warp computes two
// adjacent outputs.




extern "C" __global__ void moe_expert_gate_up_shared_bf16(
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
    const unsigned int expert_slot = blockIdx.y;
    const unsigned int proj = blockIdx.z;
    const bool is_shared = (expert_slot == top_k);

    const __nv_bfloat16* B_weight;
    __nv_bfloat16* C;

    if (is_shared) {
        if (sh_gate_weight == 0) {
            __nv_bfloat16* out = (proj == 0) ? sh_gate_out : sh_up_out;
            const unsigned int n_base = blockIdx.x * (N_PER_BLOCK * 2);
            for (unsigned int i = threadIdx.x; i < N_PER_BLOCK * 2 && n_base + i < N; i += BLOCK_SIZE)
                out[n_base + i] = __float2bfloat16(0.0f);
            return;
        }
        if (proj == 0) {
            B_weight = sh_gate_weight; C = sh_gate_out;
        } else {
            B_weight = sh_up_weight; C = sh_up_out;
        }
    } else {
        const unsigned int expert_id = expert_indices[expert_slot];
        if (proj == 0) {
            B_weight = (const __nv_bfloat16*)gate_weight_ptrs[expert_id];
            C = gate_out;
        } else {
            B_weight = (const __nv_bfloat16*)up_weight_ptrs[expert_id];
            C = up_out;
        }
        if (B_weight == 0) {
            const unsigned int n_base = blockIdx.x * (N_PER_BLOCK * 2);
            for (unsigned int i = threadIdx.x; i < N_PER_BLOCK * 2 && n_base + i < N; i += BLOCK_SIZE) {
                C[expert_slot * N + n_base + i] = __float2bfloat16(0.0f);
            }
            return;
        }
    }

    const unsigned int threads_per_out = BLOCK_SIZE / N_PER_BLOCK;
    const unsigned int local_out = threadIdx.x / threads_per_out;
    const unsigned int lane = threadIdx.x % threads_per_out;

    const unsigned int n1 = blockIdx.x * (N_PER_BLOCK * 2) + local_out * 2;
    const unsigned int n2 = n1 + 1;
    if (n1 >= N) return;
    const bool have_n2 = (n2 < N);

    const unsigned int K8 = K / 8;

    float acc1 = 0.0f, acc2 = 0.0f;




    for (unsigned int k8 = lane; k8 < K8; k8 += threads_per_out) {
        const unsigned int base_k = k8 * 8;

        uint4 a_data = ((const uint4*)A)[k8];
        uint4 w_n1 = ((const uint4*)(B_weight + (unsigned long long)n1 * K))[k8];
        uint4 w_n2;
        if (have_n2) {
            w_n2 = ((const uint4*)(B_weight + (unsigned long long)n2 * K))[k8];
        } else {
            w_n2.x = 0; w_n2.y = 0; w_n2.z = 0; w_n2.w = 0;
        }

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
        (void)base_k;
    }

    const unsigned long long base = is_shared ? 0 : (unsigned long long)expert_slot * N;

    #pragma unroll
    for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1)
        acc1 += __shfl_down_sync(0xFFFFFFFF, acc1, offset);
    if (lane == 0) C[base + n1] = __float2bfloat16(acc1);

    if (have_n2) {
        #pragma unroll
        for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1)
            acc2 += __shfl_down_sync(0xFFFFFFFF, acc2, offset);
        if (lane == 0) C[base + n2] = __float2bfloat16(acc2);
    }
}

// 2026-09-25: SiLU(gate) * up into shared memory (s_act), then the down GEMV; each warp
// computes two adjacent outputs.




extern "C" __global__ void moe_expert_silu_down_shared_bf16(
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
    const unsigned int expert_slot = blockIdx.y;
    const bool is_shared = (expert_slot == top_k);

    const __nv_bfloat16* B_weight;
    const __nv_bfloat16* g_ptr;
    const __nv_bfloat16* u_ptr;

    if (is_shared) {
        B_weight = sh_down_weight;
        g_ptr = sh_gate_in; u_ptr = sh_up_in;
    } else {
        const unsigned int expert_id = expert_indices[expert_slot];
        B_weight = (const __nv_bfloat16*)weight_ptrs[expert_id];
        g_ptr = gate_out + (unsigned long long)expert_slot * K;
        u_ptr = up_out + (unsigned long long)expert_slot * K;
        if (B_weight == 0) {
            const unsigned int n_base = blockIdx.x * (N_PER_BLOCK * 2);
            for (unsigned int i = threadIdx.x; i < N_PER_BLOCK * 2 && n_base + i < N; i += BLOCK_SIZE) {
                C[expert_slot * N + n_base + i] = __float2bfloat16(0.0f);
            }
            return;
        }
    }

    const unsigned int threads_per_out = BLOCK_SIZE / N_PER_BLOCK;
    const unsigned int local_out = threadIdx.x / threads_per_out;
    const unsigned int lane = threadIdx.x % threads_per_out;

    const unsigned int n1 = blockIdx.x * (N_PER_BLOCK * 2) + local_out * 2;
    const unsigned int n2 = n1 + 1;
    if (n1 >= N) return;
    const bool have_n2 = (n2 < N);

    const unsigned int K8 = K / 8;

    // 2026-09-25: K must be at most 2048, the size of s_act; the kernel does not check it.

    __shared__ float s_act[2048];


    for (unsigned int i = threadIdx.x; i < K; i += BLOCK_SIZE) {
        float gf = __bfloat162float(g_ptr[i]);
        float uf = __bfloat162float(u_ptr[i]);
        s_act[i] = (gf / (1.0f + __expf(-gf))) * uf;
    }
    __syncthreads();


    float acc1 = 0.0f, acc2 = 0.0f;

    for (unsigned int k8 = lane; k8 < K8; k8 += threads_per_out) {
        const unsigned int base_k = k8 * 8;

        uint4 w_n1 = ((const uint4*)(B_weight + (unsigned long long)n1 * K))[k8];
        uint4 w_n2;
        if (have_n2) {
            w_n2 = ((const uint4*)(B_weight + (unsigned long long)n2 * K))[k8];
        } else {
            w_n2.x = 0; w_n2.y = 0; w_n2.z = 0; w_n2.w = 0;
        }
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

    __nv_bfloat16* out = is_shared ? sh_down_out : (C + (unsigned long long)expert_slot * N);

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
