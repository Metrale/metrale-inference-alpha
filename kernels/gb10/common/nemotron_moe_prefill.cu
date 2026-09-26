// SPDX-License-Identifier: MIT OR Apache-2.0

// 2026-09-25: Nemotron-H MoE prefill kernels for N tokens: batched sigmoid top-k routing; NVFP4 W4A16 GEMVs for the
// experts' up projection and relu^2 + down projection (the experts have no gate projection); and the weighted
// sum of the routed outputs plus the shared expert's.
//
// Weights are E2M1 nibbles, low nibble first, with one FP8 E4M3 scale per GROUP_SIZE values times a per-tensor
// scale2. The up and down kernels pack (token, expert slot) into blockIdx.y:
//   y < num_tokens * top_k:   routed, token = y / top_k, slot = y % top_k
//   y >= num_tokens * top_k:  shared expert, token = y - num_tokens * top_k
//
// Owner: gb10 kernels.
// Invariants: none beyond the types.












#include <cuda_bf16.h>
#include <cuda_fp8.h>

#define BLOCK_SIZE 128
#define N_PER_BLOCK 4
#define WARP_SIZE 32
#define GROUP_SIZE 16

__device__ __constant__ float E2M1_LUT_NMP[16] = {
    0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f,
    -0.0f, -0.5f, -1.0f, -1.5f, -2.0f, -3.0f, -4.0f, -6.0f
};

// 2026-09-25: `nemotron_moe_topk_sigmoid_batched`: grid (1, num_tokens, 1), block 256 (the cross-warp step reads 8
// warps). Per token: the top_k experts by sigmoid(logit) + bias, highest first; their weights are the sigmoid
// scores, divided by their sum when `normalize` and the sum is positive, times scaling_factor. top_k is not
// clamped to MAX_TOP_K, and an equal value keeps the lower lane's or warp's candidate.







#define MAX_EXPERTS 512
#define MAX_TOP_K 32

extern "C" __global__ void nemotron_moe_topk_sigmoid_batched(
    const __nv_bfloat16* __restrict__ gate_logits,
    const float* __restrict__ bias,
    unsigned int* __restrict__ expert_indices,
    float* __restrict__ expert_weights,
    unsigned int num_experts,
    unsigned int top_k,
    unsigned int normalize,
    float scaling_factor,
    unsigned int num_tokens
) {
    const unsigned int token = blockIdx.y;
    if (token >= num_tokens) return;

    const unsigned int tid = threadIdx.x;

    __shared__ float s_sigmoid[MAX_EXPERTS];
    __shared__ float s_selection[MAX_EXPERTS];
    __shared__ unsigned int s_top_idxs[MAX_TOP_K];
    __shared__ float s_warp_val[8];
    __shared__ unsigned int s_warp_idx[8];

    unsigned int actual_n = num_experts < MAX_EXPERTS ? num_experts : MAX_EXPERTS;
    const __nv_bfloat16* logits = gate_logits + (unsigned long long)token * num_experts;
    unsigned int* out_idx = expert_indices + (unsigned long long)token * top_k;
    float* out_wt = expert_weights + (unsigned long long)token * top_k;


    for (unsigned int i = tid; i < actual_n; i += blockDim.x) {
        float logit = __bfloat162float(logits[i]);
        float sig = 1.0f / (1.0f + __expf(-logit));
        s_sigmoid[i] = sig;
        s_selection[i] = sig + bias[i];
    }
    __syncthreads();


    for (unsigned int k = 0; k < top_k; k++) {

        float local_max = -1e30f;
        unsigned int local_idx = 0;
        for (unsigned int i = tid; i < actual_n; i += blockDim.x) {
            if (s_selection[i] > local_max) {
                local_max = s_selection[i];
                local_idx = i;
            }
        }

        unsigned int warp_id = tid / WARP_SIZE;
        unsigned int lane = tid % WARP_SIZE;
        #pragma unroll
        for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1) {
            float other_val = __shfl_down_sync(0xFFFFFFFF, local_max, offset);
            unsigned int other_idx = __shfl_down_sync(0xFFFFFFFF, local_idx, offset);
            if (other_val > local_max) {
                local_max = other_val;
                local_idx = other_idx;
            }
        }
        if (lane == 0) {
            s_warp_val[warp_id] = local_max;
            s_warp_idx[warp_id] = local_idx;
        }
        __syncthreads();

        if (tid == 0) {
            float best_val = s_warp_val[0];
            unsigned int best_idx = s_warp_idx[0];
            for (int w = 1; w < 8; w++) {
                if (s_warp_val[w] > best_val) {
                    best_val = s_warp_val[w];
                    best_idx = s_warp_idx[w];
                }
            }
            s_top_idxs[k] = best_idx;
            s_selection[best_idx] = -1e30f;
        }
        __syncthreads();
    }


    if (tid < top_k) {
        out_idx[tid] = s_top_idxs[tid];
        float w = s_sigmoid[s_top_idxs[tid]];

        if (normalize) {
            float sum = 0.0f;
            for (unsigned int k = 0; k < top_k; k++) {
                sum += s_sigmoid[s_top_idxs[k]];
            }
            w = (sum > 0.0f) ? (w / sum) : w;
        }
        out_wt[tid] = w * scaling_factor;
    }
}

// 2026-09-25: `nemotron_moe_up_prefill`: out = A[token] @ W^T for one (token, expert slot) per blockIdx.y. Block 128:
// each warp computes 2 adjacent outputs, 8 per block, so grid x must cover the larger of N and N_shared in
// steps of 8; grid y is num_tokens * (top_k + 1). Routed rows write up_out [num_tokens * top_k, N], shared
// rows sh_up_out [num_tokens, N_shared].








extern "C" __global__ void nemotron_moe_up_prefill(
    const __nv_bfloat16* __restrict__ A,             // 2026-09-25: [num_tokens, K] BF16

    const unsigned long long* __restrict__ packed_ptrs,
    const unsigned long long* __restrict__ scale_ptrs,
    const float* __restrict__ scale2_vals,
    __nv_bfloat16* __restrict__ up_out,
    const unsigned int* __restrict__ expert_indices,  // 2026-09-25: [num_tokens * top_k]

    const unsigned char* __restrict__ sh_up_packed,
    const unsigned char* __restrict__ sh_up_scale,
    float sh_up_s2,
    __nv_bfloat16* __restrict__ sh_up_out,
    unsigned int N,
    unsigned int K,
    unsigned int N_shared,
    unsigned int top_k,
    unsigned int num_tokens
) {
    const unsigned int total_routed = num_tokens * top_k;
    const unsigned int y = blockIdx.y;
    const bool is_shared = (y >= total_routed);

    unsigned int token;
    if (is_shared) {
        token = y - total_routed;
    } else {
        token = y / top_k;
    }

    const __nv_bfloat16* A_token = A + (unsigned long long)token * K;

    const unsigned char* B_packed;
    const unsigned char* B_scale;
    float s2;
    __nv_bfloat16* C;
    unsigned int N_out;

    if (is_shared) {
        B_packed = sh_up_packed;
        B_scale = sh_up_scale;
        s2 = sh_up_s2;
        C = sh_up_out + (unsigned long long)token * N_shared;
        N_out = N_shared;
    } else {
        unsigned int expert_slot = y % top_k;
        const unsigned int expert_id = expert_indices[token * top_k + expert_slot];
        B_packed = (const unsigned char*)packed_ptrs[expert_id];
        B_scale = (const unsigned char*)scale_ptrs[expert_id];
        s2 = scale2_vals[expert_id];
        C = up_out + (unsigned long long)(token * top_k + expert_slot) * N;
        N_out = N;
        // 2026-09-25: A NULL expert pointer writes zeros to this block's outputs.
        if (B_packed == 0) {
            const unsigned int n_base = blockIdx.x * (N_PER_BLOCK * 2);
            for (unsigned int i = threadIdx.x; i < N_PER_BLOCK * 2 && n_base + i < N_out; i += BLOCK_SIZE) {
                C[n_base + i] = __float2bfloat16(0.0f);
            }
            return;
        }
    }

    // 2026-09-25: One warp per 2 adjacent outputs: lanes stride over K 8 values at a time, then a warp shuffle reduces.
    const unsigned int threads_per_out = BLOCK_SIZE / N_PER_BLOCK;
    const unsigned int local_out = threadIdx.x / threads_per_out;
    const unsigned int lane = threadIdx.x % threads_per_out;

    const unsigned int n1 = blockIdx.x * (N_PER_BLOCK * 2) + local_out * 2;
    const unsigned int n2 = n1 + 1;
    if (n1 >= N_out) return;
    const bool have_n2 = (n2 < N_out);

    const unsigned int half_K = K / 2;
    const unsigned int num_groups = K / GROUP_SIZE;
    const unsigned int K8 = K / 8;

    __shared__ float s_lut[16];
    if (threadIdx.x < 16) s_lut[threadIdx.x] = E2M1_LUT_NMP[threadIdx.x];
    __syncthreads();

    float acc1 = 0.0f, acc2 = 0.0f;

    for (unsigned int k8 = lane; k8 < K8; k8 += threads_per_out) {
        uint4 a_data = ((const uint4*)A_token)[k8];
        const unsigned int a_raw[4] = {a_data.x, a_data.y, a_data.z, a_data.w};
        const unsigned int base_k = k8 * 8;

        unsigned int packed4_1 = *(const unsigned int*)(B_packed + (unsigned long long)n1 * half_K + k8 * 4);
        unsigned int sg = base_k / GROUP_SIZE;
        unsigned char sb1 = B_scale[(unsigned long long)n1 * num_groups + sg];
        __nv_fp8_e4m3 fp8_1; *(unsigned char*)&fp8_1 = sb1;
        float sc1 = (float)fp8_1 * s2;

        unsigned int packed4_2 = have_n2 ?
            *(const unsigned int*)(B_packed + (unsigned long long)n2 * half_K + k8 * 4) : 0;
        unsigned char sb2 = have_n2 ? B_scale[(unsigned long long)n2 * num_groups + sg] : 0;
        __nv_fp8_e4m3 fp8_2; *(unsigned char*)&fp8_2 = sb2;
        float sc2 = have_n2 ? (float)fp8_2 * s2 : 0.0f;

        #pragma unroll
        for (int b = 0; b < 4; b++) {
            unsigned char bv1 = (packed4_1 >> (b * 8)) & 0xFF;
            float w1l = s_lut[bv1 & 0xF] * sc1, w1h = s_lut[bv1 >> 4] * sc1;
            unsigned char bv2 = (packed4_2 >> (b * 8)) & 0xFF;
            float w2l = s_lut[bv2 & 0xF] * sc2, w2h = s_lut[bv2 >> 4] * sc2;
            __nv_bfloat16 al, ah;
            *(unsigned short*)&al = (unsigned short)(a_raw[b] & 0xFFFF);
            *(unsigned short*)&ah = (unsigned short)(a_raw[b] >> 16);
            float afl = __bfloat162float(al), afh = __bfloat162float(ah);
            acc1 += afl * w1l + afh * w1h;
            acc2 += afl * w2l + afh * w2h;
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

// 2026-09-25: `nemotron_moe_relu2_down_prefill`: out = relu(u)^2 @ W^T for one (token, expert slot) per blockIdx.y,
// with the grid, block and row packing of `nemotron_moe_up_prefill`. relu(u)^2 is staged in dynamic shared
// memory, K floats.






extern "C" __global__ void nemotron_moe_relu2_down_prefill(
    const __nv_bfloat16* __restrict__ up_out,        // 2026-09-25: [num_tokens * top_k, K_routed] BF16
    const unsigned long long* __restrict__ packed_ptrs,
    const unsigned long long* __restrict__ scale_ptrs,
    const float* __restrict__ scale2_vals,
    __nv_bfloat16* __restrict__ C,                    // 2026-09-25: [num_tokens * top_k, N]
    const unsigned int* __restrict__ expert_indices,   // 2026-09-25: [num_tokens * top_k]

    const __nv_bfloat16* __restrict__ sh_up_in,      // 2026-09-25: [num_tokens, K_shared] BF16
    const unsigned char* __restrict__ sh_down_packed,
    const unsigned char* __restrict__ sh_down_scale,
    float sh_down_s2,
    __nv_bfloat16* __restrict__ sh_down_out,          // 2026-09-25: [num_tokens, N_shared] BF16
    unsigned int N,
    unsigned int K_routed,
    unsigned int K_shared,
    unsigned int N_shared,
    unsigned int top_k,
    unsigned int num_tokens
) {
    const unsigned int total_routed = num_tokens * top_k;
    const unsigned int y = blockIdx.y;
    const bool is_shared = (y >= total_routed);

    unsigned int token;
    const unsigned char* B_packed;
    const unsigned char* B_scale;
    float s2;
    unsigned int K;
    unsigned int N_out;
    const __nv_bfloat16* u_ptr;

    if (is_shared) {
        token = y - total_routed;
        B_packed = sh_down_packed;
        B_scale = sh_down_scale;
        s2 = sh_down_s2;
        u_ptr = sh_up_in + (unsigned long long)token * K_shared;
        K = K_shared;
        N_out = N_shared;
    } else {
        token = y / top_k;
        unsigned int expert_slot = y % top_k;
        const unsigned int expert_id = expert_indices[token * top_k + expert_slot];
        B_packed = (const unsigned char*)packed_ptrs[expert_id];
        B_scale = (const unsigned char*)scale_ptrs[expert_id];
        s2 = scale2_vals[expert_id];
        u_ptr = up_out + (unsigned long long)(token * top_k + expert_slot) * K_routed;
        K = K_routed;
        N_out = N;
        // 2026-09-25: A NULL expert pointer writes zeros to this block's outputs.
        if (B_packed == 0) {
            __nv_bfloat16* out = C + (unsigned long long)(token * top_k + expert_slot) * N;
            const unsigned int n_base = blockIdx.x * (N_PER_BLOCK * 2);
            for (unsigned int i = threadIdx.x; i < N_PER_BLOCK * 2 && n_base + i < N_out; i += BLOCK_SIZE) {
                out[n_base + i] = __float2bfloat16(0.0f);
            }
            return;
        }
    }

    const unsigned int threads_per_out = BLOCK_SIZE / N_PER_BLOCK;
    const unsigned int local_out = threadIdx.x / threads_per_out;
    const unsigned int lane = threadIdx.x % threads_per_out;

    const unsigned int n1 = blockIdx.x * (N_PER_BLOCK * 2) + local_out * 2;
    const unsigned int n2 = n1 + 1;
    if (n1 >= N_out) return;
    const bool have_n2 = (n2 < N_out);

    const unsigned int half_K = K / 2;
    const unsigned int num_groups = K / GROUP_SIZE;
    const unsigned int K8 = K / 8;

    __shared__ float s_lut[16];
    extern __shared__ float s_act[];

    if (threadIdx.x < 16) s_lut[threadIdx.x] = E2M1_LUT_NMP[threadIdx.x];


    for (unsigned int i = threadIdx.x; i < K; i += BLOCK_SIZE) {
        float u = __bfloat162float(u_ptr[i]);
        float r = fmaxf(u, 0.0f);
        s_act[i] = r * r;
    }
    __syncthreads();


    float acc1 = 0.0f, acc2 = 0.0f;

    for (unsigned int k8 = lane; k8 < K8; k8 += threads_per_out) {
        const unsigned int base_k = k8 * 8;

        unsigned int packed4_1 = *(const unsigned int*)(B_packed + (unsigned long long)n1 * half_K + k8 * 4);
        unsigned int sg = base_k / GROUP_SIZE;
        unsigned char sb1 = B_scale[(unsigned long long)n1 * num_groups + sg];
        __nv_fp8_e4m3 fp8_1; *(unsigned char*)&fp8_1 = sb1;
        float sc1 = (float)fp8_1 * s2;

        unsigned int packed4_2 = have_n2 ?
            *(const unsigned int*)(B_packed + (unsigned long long)n2 * half_K + k8 * 4) : 0;
        unsigned char sb2 = have_n2 ? B_scale[(unsigned long long)n2 * num_groups + sg] : 0;
        __nv_fp8_e4m3 fp8_2; *(unsigned char*)&fp8_2 = sb2;
        float sc2 = have_n2 ? (float)fp8_2 * s2 : 0.0f;

        #pragma unroll
        for (int b = 0; b < 4; b++) {
            float al = s_act[base_k + b * 2];
            float ah = s_act[base_k + b * 2 + 1];

            unsigned char bv1 = (packed4_1 >> (b * 8)) & 0xFF;
            float w1l = s_lut[bv1 & 0xF] * sc1, w1h = s_lut[bv1 >> 4] * sc1;
            unsigned char bv2 = (packed4_2 >> (b * 8)) & 0xFF;
            float w2l = s_lut[bv2 & 0xF] * sc2, w2h = s_lut[bv2 >> 4] * sc2;

            acc1 += al * w1l + ah * w1h;
            acc2 += al * w2l + ah * w2h;
        }
    }


    __nv_bfloat16* out = is_shared ?
        (sh_down_out + (unsigned long long)token * N_shared) :
        (C + (unsigned long long)(token * top_k + (y % top_k)) * N);

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

// 2026-09-25: `nemotron_moe_weighted_sum_prefill`: grid (ceil(hidden / blockDim.x), num_tokens, 1).
// output[t, d] = routed_scaling_factor * sum_e(expert_weights[t * top_k + e] * expert_down[t * top_k + e, d])
//               + shared_down[t, d]






extern "C" __global__ void nemotron_moe_weighted_sum_prefill(
    __nv_bfloat16* __restrict__ output,
    const __nv_bfloat16* __restrict__ expert_down,
    const float* __restrict__ expert_weights,
    const __nv_bfloat16* __restrict__ shared_down,
    unsigned int hidden,
    unsigned int top_k,
    float routed_scaling_factor,
    unsigned int num_tokens
) {
    const unsigned int token = blockIdx.y;
    const unsigned int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j >= hidden || token >= num_tokens) return;

    float routed_sum = 0.0f;
    for (unsigned int e = 0; e < top_k; e++) {
        float w = expert_weights[token * top_k + e];
        routed_sum += w * __bfloat162float(expert_down[(unsigned long long)(token * top_k + e) * hidden + j]);
    }
    float shared_val = __bfloat162float(shared_down[(unsigned long long)token * hidden + j]);
    output[(unsigned long long)token * hidden + j] = __float2bfloat16(routed_scaling_factor * routed_sum + shared_val);
}
