// SPDX-License-Identifier: MIT OR Apache-2.0

// 2026-09-25: MoE router for one token in a single block: an NVFP4 gate GEMV with one expert
// per thread, top-k by repeated block-wide max, then a softmax over the k selected logits.
//
// Owner: gb10 kernels.
// Invariants:
// - Grid 1, block BLOCK_SIZE (256); dynamic shared memory holds A as K BF16 values.
// - Only the first MAX_EXPERTS (256) experts are scored, and top_k must be at most
//   MAX_TOP_K (32), the size of s_top_vals / s_top_idxs.
// - expert_weights[t] = exp(v_t - v_0) over the selected logits, divided by their sum only
//   when normalize is non-zero. Gate weights: [num_experts, K / 2] E2M1 pairs and
//   [num_experts, K / 16] E4M3 scales, times scale2.

#include <cuda_bf16.h>
#include <cuda_fp8.h>

#define BLOCK_SIZE 256
#define MAX_EXPERTS 256
#define MAX_TOP_K 32
#define WARP_SIZE 32
#define GROUP_SIZE 16

__device__ __constant__ float E2M1_LUT_GATE[16] = {
    0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f,
    -0.0f, -0.5f, -1.0f, -1.5f, -2.0f, -3.0f, -4.0f, -6.0f
};

// 2026-09-25: Decode an FP8 E4M3 scale byte. The SCALE/HIP branch decodes in software and
// maps the NaN encoding (exponent 15, mantissa 7) to 0.

#if defined(__SCALE__) || defined(__HIP_PLATFORM_AMD__)
__device__ __forceinline__ float metrale_dec_e4m3(unsigned char b) {
    unsigned int s = (b >> 7) & 1u, e = (b >> 3) & 0xFu, m = b & 0x7u; float v;
    if (e == 0u)               v = (float)m * 0.001953125f;
    else if (e == 15u && m == 7u) v = 0.0f;
    else                       v = __uint_as_float(((e + 120u) << 23) | (m << 20));
    return s ? -v : v;
}
#else
__device__ __forceinline__ float metrale_dec_e4m3(unsigned char b) {
    __nv_fp8_e4m3 f; *(unsigned char*)&f = b; return (float)f;
}
#endif

extern "C" __global__ void moe_gate_topk_fused(
    const __nv_bfloat16* __restrict__ A,
    const unsigned char* __restrict__ B_packed,
    const unsigned char* __restrict__ B_scale,
    float scale2,
    unsigned int* __restrict__ expert_indices,
    float* __restrict__ expert_weights,
    unsigned int num_experts,
    unsigned int K,
    unsigned int top_k,
    unsigned int normalize
) {
    extern __shared__ __nv_bfloat16 s_A[];
    __shared__ float s_lut[16];
    __shared__ float s_vals[MAX_EXPERTS];
    __shared__ float s_top_vals[MAX_TOP_K];
    __shared__ unsigned int s_top_idxs[MAX_TOP_K];
    __shared__ float s_warp_val[8];
    __shared__ unsigned int s_warp_idx[8];

    const unsigned int tid = threadIdx.x;
    const unsigned int warp_id = tid / WARP_SIZE;
    const unsigned int lane = tid % WARP_SIZE;
    const unsigned int num_warps = BLOCK_SIZE / WARP_SIZE;


    if (tid < 16) s_lut[tid] = E2M1_LUT_GATE[tid];


    for (unsigned int i = tid; i < K; i += BLOCK_SIZE) {
        s_A[i] = A[i];
    }
    __syncthreads();



    const unsigned int actual_n = num_experts < MAX_EXPERTS ? num_experts : MAX_EXPERTS;

    if (tid < actual_n) {
        const unsigned int half_K = K / 2;
        const unsigned int num_groups = K / GROUP_SIZE;
        const unsigned int K8 = K / 8;

        float acc = 0.0f;

        for (unsigned int k8 = 0; k8 < K8; k8++) {
            unsigned int packed4 = *(const unsigned int*)(B_packed + (unsigned long long)tid * half_K + k8 * 4);
            unsigned int sg = (k8 * 8) / GROUP_SIZE;
            unsigned char sb = B_scale[(unsigned long long)tid * num_groups + sg];
            float sc = metrale_dec_e4m3(sb) * scale2;

            #pragma unroll
            for (int b = 0; b < 4; b++) {
                unsigned char bv = (packed4 >> (b * 8)) & 0xFF;
                float wl = s_lut[bv & 0xF] * sc, wh = s_lut[bv >> 4] * sc;
                float al = __bfloat162float(s_A[k8 * 8 + b * 2]);
                float ah = __bfloat162float(s_A[k8 * 8 + b * 2 + 1]);
                acc += al * wl + ah * wh;
            }
        }

        s_vals[tid] = acc;
    } else {
        s_vals[tid] = -1e30f;
    }
    __syncthreads();


    for (unsigned int t = 0; t < top_k && t < actual_n; t++) {
        float local_max = -1e30f;
        unsigned int local_idx = 0;
        for (unsigned int i = tid; i < actual_n; i += BLOCK_SIZE) {
            float v = s_vals[i];
            if (v > local_max) {
                local_max = v;
                local_idx = i;
            }
        }

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
            for (unsigned int w = 1; w < num_warps; w++) {
                if (s_warp_val[w] > best_val) {
                    best_val = s_warp_val[w];
                    best_idx = s_warp_idx[w];
                }
            }
            s_top_vals[t] = best_val;
            s_top_idxs[t] = best_idx;
            s_vals[best_idx] = -1e30f;
        }
        __syncthreads();
    }


    if (tid == 0) {
        float max_val = s_top_vals[0];

        float exp_sum = 0.0f;
        for (unsigned int t = 0; t < top_k && t < actual_n; t++) {
            float e = __expf(s_top_vals[t] - max_val);
            s_top_vals[t] = e;
            exp_sum += e;
        }

        for (unsigned int t = 0; t < top_k && t < actual_n; t++) {
            expert_indices[t] = s_top_idxs[t];
            if (normalize) {
                expert_weights[t] = s_top_vals[t] / exp_sum;
            } else {
                expert_weights[t] = s_top_vals[t];
            }
        }
    }
}
