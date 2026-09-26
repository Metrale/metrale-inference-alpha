// SPDX-License-Identifier: MIT OR Apache-2.0

// 2026-09-25: FP16 h-state twin of gated_delta_rule_wy2 (K=2 GDN verify step).
//
// Owner: gb10 kernels.
// Invariants:
// - Apart from the FP16 round trip below, every float expression, gate clamp and
//   accumulation order is gated_delta_rule_wy2's (gated_delta_rule_wy.cu). H and H_inter
//   are `__half` in memory, read with __half2float and written with gdn_f16_store
//   (gdn_f16_state.cuh).
// - Each token's updated state is rounded to FP16 before it is stored, carried to the next
//   token and dotted with q, so the state the forward chain carries equals the stored
//   rollback intermediate bit for bit.
// - Pass 1 reads H once; pass 2 reads it again and writes H_1 to H_inter, H_2 to H.
//
// Qwen3SsmLayer::wy2_kernel (qwen3_ssm/trait_decode_batched_conv_gdn.rs) returns this
// kernel when the FP16 h-state is on and the register-resident FP16 twin is not selected.
// Grid (num_v_heads, batch), block 128. Needs k_dim, v_dim <= 128 and k_dim % 4 == 0.













#include <cuda_bf16.h>
#include "gdn_reduce.cuh"
#include "gdn_f16_state.cuh"
#define BLOCK_SIZE 128









extern "C" __global__ void gated_delta_rule_wy2_f16(
    __half* __restrict__ h_state,
    const __nv_bfloat16* __restrict__ query,
    const __nv_bfloat16* __restrict__ key,
    const __nv_bfloat16* __restrict__ value,
    const float* __restrict__ gate,
    const float* __restrict__ beta,
    __nv_bfloat16* __restrict__ output,
    __half* __restrict__ h_state_intermediate,
    unsigned int batch_size,
    unsigned int num_k_heads,
    unsigned int num_v_heads,
    unsigned int k_dim,
    unsigned int v_dim,
    unsigned int qk_stride,
    unsigned int v_stride,
    unsigned int gb_stride,








    // 2026-09-25: 0: both state arguments are contiguous bases indexed by
    // (b * num_v_heads + vh) * k_dim * v_dim halves. That matches neither the FP32-sized
    // pool's slot pitch nor the placement of intermediates, so ops::gdn_decode_wy2 accepts
    // 0 only at batch_size 1.
    // 1: each is a device table of `batch_size` per-sequence base pointers; head vh starts
    // at table[b] + vh * k_dim * v_dim.
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
    __half* H = state_is_table ? ((__half* const*)h_state)[b] + head_off
                              : h_state + flat_off;
    __half* H_inter = state_is_table
                         ? ((__half* const*)h_state_intermediate)[b] + head_off
                         : h_state_intermediate + flat_off;

    // 2026-09-25: Token t of sequence b is input row b*2+t.
    const __nv_bfloat16* q0 = query + (b * 2) * qk_stride + kh * k_dim;
    const __nv_bfloat16* k0 = key   + (b * 2) * qk_stride + kh * k_dim;
    const __nv_bfloat16* v0 = value + (b * 2) * v_stride  + vh * v_dim;


    // 2026-09-25: The gate is clamped to [1e-6, 1 - 1e-6], the clamp gated_delta_rule_decode
    // applies.
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


        float hk0 = 0.0f, hk1_prev = 0.0f;
        #pragma unroll 4
        for (unsigned int j = 0; j < k_dim; j += 4) {
            float h0 = __half2float(H[(j+0) * v_dim + tid]);
            float h1 = __half2float(H[(j+1) * v_dim + tid]);
            float h2 = __half2float(H[(j+2) * v_dim + tid]);
            float h3 = __half2float(H[(j+3) * v_dim + tid]);
            hk0      += h0*smem_k0[j] + h1*smem_k0[j+1] + h2*smem_k0[j+2] + h3*smem_k0[j+3];
            hk1_prev += h0*smem_k1[j] + h1*smem_k1[j+1] + h2*smem_k1[j+2] + h3*smem_k1[j+3];
        }


        float v_new_0 = (vi0 - g0 * hk0) * bt0;
        float hk1_corr = g0 * hk1_prev + kdot_10 * v_new_0;
        float v_new_1 = (vi1 - g1 * hk1_corr) * bt1;


        float q0_dot = 0.0f, q1_dot = 0.0f;
        #pragma unroll 4
        for (unsigned int j = 0; j < k_dim; j += 4) {
            float h0 = __half2float(H[(j+0) * v_dim + tid]);
            float h1 = __half2float(H[(j+1) * v_dim + tid]);
            float h2 = __half2float(H[(j+2) * v_dim + tid]);
            float h3 = __half2float(H[(j+3) * v_dim + tid]);


            h0 = g0*h0 + smem_k0[j]  *v_new_0;
            h1 = g0*h1 + smem_k0[j+1]*v_new_0;
            h2 = g0*h2 + smem_k0[j+2]*v_new_0;
            h3 = g0*h3 + smem_k0[j+3]*v_new_0;
            h0=__half2float(gdn_f16_store(h0)); h1=__half2float(gdn_f16_store(h1));
            H_inter[(j+0)*v_dim+tid]=gdn_f16_store(h0); H_inter[(j+1)*v_dim+tid]=gdn_f16_store(h1);
            h2=__half2float(gdn_f16_store(h2)); h3=__half2float(gdn_f16_store(h3));
            H_inter[(j+2)*v_dim+tid]=gdn_f16_store(h2); H_inter[(j+3)*v_dim+tid]=gdn_f16_store(h3);
            q0_dot += h0*smem_q0[j] + h1*smem_q0[j+1] + h2*smem_q0[j+2] + h3*smem_q0[j+3];


            h0 = g1*h0 + smem_k1[j]  *v_new_1;
            h1 = g1*h1 + smem_k1[j+1]*v_new_1;
            h2 = g1*h2 + smem_k1[j+2]*v_new_1;
            h3 = g1*h3 + smem_k1[j+3]*v_new_1;
            h0=__half2float(gdn_f16_store(h0)); h1=__half2float(gdn_f16_store(h1));
            H[(j+0)*v_dim+tid]=gdn_f16_store(h0); H[(j+1)*v_dim+tid]=gdn_f16_store(h1);
            h2=__half2float(gdn_f16_store(h2)); h3=__half2float(gdn_f16_store(h3));
            H[(j+2)*v_dim+tid]=gdn_f16_store(h2); H[(j+3)*v_dim+tid]=gdn_f16_store(h3);
            q1_dot += h0*smem_q1[j] + h1*smem_q1[j+1] + h2*smem_q1[j+2] + h3*smem_q1[j+3];
        }

        float inv_sqrt_d = rsqrtf((float)k_dim);
        unsigned int out0 = (b * 2 * num_v_heads + vh) * v_dim;
        unsigned int out1 = ((b * 2 + 1) * num_v_heads + vh) * v_dim;
        output[out0 + tid] = __float2bfloat16(q0_dot * inv_sqrt_d);
        output[out1 + tid] = __float2bfloat16(q1_dot * inv_sqrt_d);
    }
}

