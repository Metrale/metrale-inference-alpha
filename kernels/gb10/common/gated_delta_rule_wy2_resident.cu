// SPDX-License-Identifier: MIT OR Apache-2.0

// 2026-09-25: K=2 speculative-verify GDN step with the state column held in registers
// (gated_delta_rule_wy2_resident).
//
// Same arguments, outputs and float expressions as gated_delta_rule_wy2 (gated_delta_rule_wy.cu),
// but pass 1 keeps each loaded H value in H_reg and pass 2 reads H_reg instead of H: one read and
// two writes of the per-head FP32 state instead of two reads and two writes. Pass 2 runs as two
// loops, token 0 then token 1, where wy2 interleaves them; each value keeps its expression and
// accumulation order. The resident-parity leg of gdn_wy_verify_microtest compares output,
// intermediate and final H with wy2 bit for bit.
//
// Qwen3SsmLayer::wy2_kernel selects it for an FP32 h-state when kd == vd == 128, the launch
// width n is at least wy_resident_min_width(), and METRALE_NO_GDN_WY2_RESIDENT is unset.
//
// Owner: gb10 kernels.
// Invariants:
// - k_dim == v_dim == 128: the loops run to the compile-time WY2R_KD and index with WY2R_VD.
// - Launch, token rows and state_is_table are those of gated_delta_rule_wy2 (ops::gdn_decode_wy2).
// - The state after token 0 is written to h_state_intermediate, after token 1 to h_state.































#include <cuda_bf16.h>
#include "gdn_reduce.cuh"
#define BLOCK_SIZE 128
#define WY2R_KD 128u
#define WY2R_VD 128u

extern "C" __global__ void __launch_bounds__(128, 1)
gated_delta_rule_wy2_resident(
    float* __restrict__ h_state,
    const __nv_bfloat16* __restrict__ query,
    const __nv_bfloat16* __restrict__ key,
    const __nv_bfloat16* __restrict__ value,
    const float* __restrict__ gate,
    const float* __restrict__ beta,
    __nv_bfloat16* __restrict__ output,
    float* __restrict__ h_state_intermediate,
    unsigned int batch_size,
    unsigned int num_k_heads,
    unsigned int num_v_heads,
    unsigned int k_dim,
    unsigned int v_dim,
    unsigned int qk_stride,
    unsigned int v_stride,
    unsigned int gb_stride,
    // 2026-09-25: Same meaning as gated_delta_rule_wy2's state_is_table.





    unsigned int state_is_table
) {
    const unsigned int vh = blockIdx.x;
    const unsigned int b = blockIdx.y;
    if (vh >= num_v_heads || b >= batch_size) return;

    const unsigned int tid = threadIdx.x;
    const unsigned int head_repeat = num_v_heads / num_k_heads;
    const unsigned int kh = vh / head_repeat;

    const unsigned int hv_size = k_dim * v_dim;
    const unsigned long long head_off = (unsigned long long)vh * hv_size;
    const unsigned long long flat_off =
        (unsigned long long)(b * num_v_heads + vh) * hv_size;
    float* H = state_is_table ? ((float* const*)h_state)[b] + head_off
                              : h_state + flat_off;
    float* H_inter = state_is_table
                         ? ((float* const*)h_state_intermediate)[b] + head_off
                         : h_state_intermediate + flat_off;


    const __nv_bfloat16* q0 = query + (b * 2) * qk_stride + kh * k_dim;
    const __nv_bfloat16* k0 = key   + (b * 2) * qk_stride + kh * k_dim;
    const __nv_bfloat16* v0 = value + (b * 2) * v_stride  + vh * v_dim;
    // 2026-09-25: Same gate clamp as gated_delta_rule_decode and gated_delta_rule_wy2.

    const float g0 = fminf(fmaxf(gate[(b * 2) * gb_stride + vh], 1e-6f), 1.0f - 1e-6f);
    const float bt0 = beta[(b * 2) * gb_stride + vh];

    const __nv_bfloat16* q1 = query + (b * 2 + 1) * qk_stride + kh * k_dim;
    const __nv_bfloat16* k1 = key   + (b * 2 + 1) * qk_stride + kh * k_dim;
    const __nv_bfloat16* v1 = value + (b * 2 + 1) * v_stride  + vh * v_dim;
    const float g1 = fminf(fmaxf(gate[(b * 2 + 1) * gb_stride + vh], 1e-6f), 1.0f - 1e-6f);
    const float bt1 = beta[(b * 2 + 1) * gb_stride + vh];

    __shared__ float smem_k0[128], smem_q0[128];
    __shared__ float smem_k1[128], smem_q1[128];
    __shared__ float smem_kdot;
    __shared__ float smem_warp[4];

    if (tid < k_dim) {
        smem_k0[tid] = (float)k0[tid]; smem_q0[tid] = (float)q0[tid];
        smem_k1[tid] = (float)k1[tid]; smem_q1[tid] = (float)q1[tid];
    }
    __syncthreads();


    {
        float partial = (tid < k_dim) ? smem_k1[tid] * smem_k0[tid] : 0.0f;
        float result = metrale_block_reduce_sum(partial, smem_warp, tid);
        if (tid == 0) smem_kdot = result;
    }
    __syncthreads();

    if (tid < v_dim) {
        float vi0 = (float)v0[tid];
        float vi1 = (float)v1[tid];
        float kdot_10 = smem_kdot;

        // 2026-09-25: Thread tid owns state column tid: H[j][tid] for j = 0..127.
        float H_reg[WY2R_KD];



        float hk0 = 0.0f, hk1_prev = 0.0f;
        #pragma unroll
        for (unsigned int j = 0; j < WY2R_KD; j += 4) {
            float h0 = H[(j+0) * WY2R_VD + tid];
            float h1 = H[(j+1) * WY2R_VD + tid];
            float h2 = H[(j+2) * WY2R_VD + tid];
            float h3 = H[(j+3) * WY2R_VD + tid];
            H_reg[j+0] = h0; H_reg[j+1] = h1;
            H_reg[j+2] = h2; H_reg[j+3] = h3;
            hk0      += h0*smem_k0[j] + h1*smem_k0[j+1] + h2*smem_k0[j+2] + h3*smem_k0[j+3];
            hk1_prev += h0*smem_k1[j] + h1*smem_k1[j+1] + h2*smem_k1[j+2] + h3*smem_k1[j+3];
        }


        float v_new_0 = (vi0 - g0 * hk0) * bt0;
        float hk1_corr = g0 * hk1_prev + kdot_10 * v_new_0;
        float v_new_1 = (vi1 - g1 * hk1_corr) * bt1;




        float q0_dot = 0.0f, q1_dot = 0.0f;
        #pragma unroll
        for (unsigned int j = 0; j < WY2R_KD; j += 4) {
            float h0 = H_reg[j+0];
            float h1 = H_reg[j+1];
            float h2 = H_reg[j+2];
            float h3 = H_reg[j+3];
            h0 = g0*h0 + smem_k0[j]  *v_new_0;
            h1 = g0*h1 + smem_k0[j+1]*v_new_0;
            h2 = g0*h2 + smem_k0[j+2]*v_new_0;
            h3 = g0*h3 + smem_k0[j+3]*v_new_0;
            H_inter[(j+0)*WY2R_VD+tid]=h0; H_inter[(j+1)*WY2R_VD+tid]=h1;
            H_inter[(j+2)*WY2R_VD+tid]=h2; H_inter[(j+3)*WY2R_VD+tid]=h3;
            H_reg[j+0] = h0; H_reg[j+1] = h1;
            H_reg[j+2] = h2; H_reg[j+3] = h3;
            q0_dot += h0*smem_q0[j] + h1*smem_q0[j+1] + h2*smem_q0[j+2] + h3*smem_q0[j+3];
        }


        #pragma unroll
        for (unsigned int j = 0; j < WY2R_KD; j += 4) {
            float h0 = H_reg[j+0];
            float h1 = H_reg[j+1];
            float h2 = H_reg[j+2];
            float h3 = H_reg[j+3];
            h0 = g1*h0 + smem_k1[j]  *v_new_1;
            h1 = g1*h1 + smem_k1[j+1]*v_new_1;
            h2 = g1*h2 + smem_k1[j+2]*v_new_1;
            h3 = g1*h3 + smem_k1[j+3]*v_new_1;
            H[(j+0)*WY2R_VD+tid]=h0; H[(j+1)*WY2R_VD+tid]=h1;
            H[(j+2)*WY2R_VD+tid]=h2; H[(j+3)*WY2R_VD+tid]=h3;
            q1_dot += h0*smem_q1[j] + h1*smem_q1[j+1] + h2*smem_q1[j+2] + h3*smem_q1[j+3];
        }

        float inv_sqrt_d = rsqrtf((float)k_dim);
        unsigned int out0 = (b * 2 * num_v_heads + vh) * v_dim;
        unsigned int out1 = ((b * 2 + 1) * num_v_heads + vh) * v_dim;
        output[out0 + tid] = __float2bfloat16(q0_dot * inv_sqrt_d);
        output[out1 + tid] = __float2bfloat16(q1_dot * inv_sqrt_d);
    }
}
