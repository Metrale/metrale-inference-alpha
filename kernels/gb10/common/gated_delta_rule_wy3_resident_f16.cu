// SPDX-License-Identifier: MIT OR Apache-2.0

// 2026-09-25: FP16 h-state twin of gated_delta_rule_wy3_resident (K=3 GDN verify step,
// register-resident pass 2).
//
// Owner: gb10 kernels.
// Invariants:
// - Apart from the FP16 round trip, every float expression and accumulation order is
//   gated_delta_rule_wy3_resident's. H_reg stays `float[128]`; H, Hi0 and Hi1 are `__half`
//   in memory, read with __half2float and written with gdn_f16_store (gdn_f16_state.cuh).
// - Each token's updated state is rounded to FP16 once, and that rounded value is what is
//   stored, carried to the next token and dotted with q, so each rollback intermediate
//   equals the state the forward chain carried.
// - H is read from memory once: one read plus three writes of FP16 state per launch.
//
// With the FP16 h-state on, Qwen3SsmLayer::wy3_kernel (qwen3_ssm/
// trait_decode_batched_conv_gdn.rs) selects this kernel under the conditions that select
// gated_delta_rule_wy3_resident, plus this handle being linked; otherwise it selects
// gated_delta_rule_wy3_f16.
// Grid (num_v_heads, batch), block 128; k_dim and v_dim are the compile-time 128.













#include <cuda_bf16.h>
#include "gdn_reduce.cuh"
#include "gdn_f16_state.cuh"
#define BLOCK_SIZE 128
#define WY3RF_KD 128u
#define WY3RF_VD 128u








extern "C" __global__ void __launch_bounds__(128, 1)
gated_delta_rule_wy3_resident_f16(
    __half* __restrict__ h_state,
    const __nv_bfloat16* __restrict__ query,
    const __nv_bfloat16* __restrict__ key,
    const __nv_bfloat16* __restrict__ value,
    const float* __restrict__ gate,
    const float* __restrict__ beta,
    __nv_bfloat16* __restrict__ output,
    __half* __restrict__ h_state_inter0,
    __half* __restrict__ h_state_inter1,
    unsigned int batch_size,
    unsigned int num_k_heads,
    unsigned int num_v_heads,
    unsigned int k_dim,
    unsigned int v_dim,
    unsigned int qk_stride,
    unsigned int v_stride,
    unsigned int gb_stride,



    // 2026-09-25: Same meaning as gated_delta_rule_wy3's: 0 = contiguous bases, accepted by
    // ops::gdn_decode_wy3 only at batch_size 1; 1 = per-sequence device pointer tables.
    unsigned int state_is_table
) {
    const unsigned int vh = blockIdx.x;
    const unsigned int b = blockIdx.y;
    if (vh >= num_v_heads || b >= batch_size) return;

    const unsigned int tid = threadIdx.x;
    const unsigned int hr = num_v_heads / num_k_heads;
    const unsigned int kh = vh / hr;
    const unsigned int hv = k_dim * v_dim;

    const unsigned long long head_off = (unsigned long long)vh * hv;
    const unsigned long long flat_off = (unsigned long long)(b * num_v_heads + vh) * hv;
    __half* H   = state_is_table ? ((__half* const*)h_state)[b]        + head_off
                                 : h_state        + flat_off;
    __half* Hi0 = state_is_table ? ((__half* const*)h_state_inter0)[b] + head_off
                                 : h_state_inter0 + flat_off;
    __half* Hi1 = state_is_table ? ((__half* const*)h_state_inter1)[b] + head_off
                                 : h_state_inter1 + flat_off;



    // 2026-09-25: Input rows, strides and the gate clamp are gated_delta_rule_wy3's.
    #define TP(T) \
        const __nv_bfloat16* q##T = query + (b*3+T)*qk_stride + kh*k_dim; \
        const __nv_bfloat16* k##T = key   + (b*3+T)*qk_stride + kh*k_dim; \
        const __nv_bfloat16* v##T = value + (b*3+T)*v_stride  + vh*v_dim; \
        const float g##T = fminf(fmaxf(gate[(b*3+T)*gb_stride + vh], 1e-6f), 1.0f - 1e-6f); \
        const float bt##T = beta[(b*3+T)*gb_stride + vh];
    TP(0) TP(1) TP(2)
    #undef TP

    __shared__ float sk0[128], sq0[128], sk1[128], sq1[128], sk2[128], sq2[128];
    __shared__ float smem_warp[4];
    __shared__ float kd10, kd20, kd21;

    if (tid < k_dim) {
        sk0[tid]=(float)k0[tid]; sq0[tid]=(float)q0[tid];
        sk1[tid]=(float)k1[tid]; sq1[tid]=(float)q1[tid];
        sk2[tid]=(float)k2[tid]; sq2[tid]=(float)q2[tid];
    }
    __syncthreads();


    {
        float p = (tid<k_dim) ? sk1[tid]*sk0[tid] : 0.0f;
        float r = metrale_block_reduce_sum(p, smem_warp, tid);
        if (tid==0) kd10 = r;
    }
    __syncthreads();
    {
        float p = (tid<k_dim) ? sk2[tid]*sk0[tid] : 0.0f;
        float r = metrale_block_reduce_sum(p, smem_warp, tid);
        if (tid==0) kd20 = r;
    }
    __syncthreads();
    {
        float p = (tid<k_dim) ? sk2[tid]*sk1[tid] : 0.0f;
        float r = metrale_block_reduce_sum(p, smem_warp, tid);
        if (tid==0) kd21 = r;
    }
    __syncthreads();

    if (tid < v_dim) {
        float vi0=(float)v0[tid], vi1=(float)v1[tid], vi2=(float)v2[tid];

        // 2026-09-25: Thread tid owns state column tid (H[j][tid], j < 128) and keeps it in H_reg.
        float H_reg[WY3RF_KD];



        float hk0=0, hk1p=0, hk2p=0;
        #pragma unroll
        for (unsigned int j=0; j<WY3RF_KD; j+=4) {
            float h0=__half2float(H[(j+0)*WY3RF_VD+tid]);
            float h1=__half2float(H[(j+1)*WY3RF_VD+tid]);
            float h2=__half2float(H[(j+2)*WY3RF_VD+tid]);
            float h3=__half2float(H[(j+3)*WY3RF_VD+tid]);
            H_reg[j+0]=h0; H_reg[j+1]=h1;
            H_reg[j+2]=h2; H_reg[j+3]=h3;
            hk0  += h0*sk0[j]+h1*sk0[j+1]+h2*sk0[j+2]+h3*sk0[j+3];
            hk1p += h0*sk1[j]+h1*sk1[j+1]+h2*sk1[j+2]+h3*sk1[j+3];
            hk2p += h0*sk2[j]+h1*sk2[j+1]+h2*sk2[j+2]+h3*sk2[j+3];
        }


        float vn0 = (vi0 - g0*hk0) * bt0;
        float hk1c = g0*hk1p + kd10*vn0;
        float vn1 = (vi1 - g1*hk1c) * bt1;
        float hk2c = g0*g1*hk2p + g1*kd20*vn0 + kd21*vn1;
        float vn2 = (vi2 - g2*hk2c) * bt2;


        float qd0=0, qd1=0, qd2=0;
        #pragma unroll
        for (unsigned int j=0; j<WY3RF_KD; j+=4) {
            float h0=H_reg[j+0], h1=H_reg[j+1];
            float h2=H_reg[j+2], h3=H_reg[j+3];
            h0=g0*h0+sk0[j]*vn0; h1=g0*h1+sk0[j+1]*vn0;
            h2=g0*h2+sk0[j+2]*vn0; h3=g0*h3+sk0[j+3]*vn0;
            __half s0 = gdn_f16_store(h0);
            __half s1 = gdn_f16_store(h1);
            __half s2 = gdn_f16_store(h2);
            __half s3 = gdn_f16_store(h3);
            Hi0[(j+0)*WY3RF_VD+tid]=s0; Hi0[(j+1)*WY3RF_VD+tid]=s1;
            Hi0[(j+2)*WY3RF_VD+tid]=s2; Hi0[(j+3)*WY3RF_VD+tid]=s3;
            h0=__half2float(s0); h1=__half2float(s1);
            h2=__half2float(s2); h3=__half2float(s3);
            H_reg[j+0]=h0; H_reg[j+1]=h1;
            H_reg[j+2]=h2; H_reg[j+3]=h3;
            qd0 += h0*sq0[j]+h1*sq0[j+1]+h2*sq0[j+2]+h3*sq0[j+3];
        }


        #pragma unroll
        for (unsigned int j=0; j<WY3RF_KD; j+=4) {
            float h0=H_reg[j+0], h1=H_reg[j+1];
            float h2=H_reg[j+2], h3=H_reg[j+3];
            h0=g1*h0+sk1[j]*vn1; h1=g1*h1+sk1[j+1]*vn1;
            h2=g1*h2+sk1[j+2]*vn1; h3=g1*h3+sk1[j+3]*vn1;
            __half s0 = gdn_f16_store(h0);
            __half s1 = gdn_f16_store(h1);
            __half s2 = gdn_f16_store(h2);
            __half s3 = gdn_f16_store(h3);
            Hi1[(j+0)*WY3RF_VD+tid]=s0; Hi1[(j+1)*WY3RF_VD+tid]=s1;
            Hi1[(j+2)*WY3RF_VD+tid]=s2; Hi1[(j+3)*WY3RF_VD+tid]=s3;
            h0=__half2float(s0); h1=__half2float(s1);
            h2=__half2float(s2); h3=__half2float(s3);
            H_reg[j+0]=h0; H_reg[j+1]=h1;
            H_reg[j+2]=h2; H_reg[j+3]=h3;
            qd1 += h0*sq1[j]+h1*sq1[j+1]+h2*sq1[j+2]+h3*sq1[j+3];
        }


        #pragma unroll
        for (unsigned int j=0; j<WY3RF_KD; j+=4) {
            float h0=H_reg[j+0], h1=H_reg[j+1];
            float h2=H_reg[j+2], h3=H_reg[j+3];
            h0=g2*h0+sk2[j]*vn2; h1=g2*h1+sk2[j+1]*vn2;
            h2=g2*h2+sk2[j+2]*vn2; h3=g2*h3+sk2[j+3]*vn2;
            __half s0 = gdn_f16_store(h0);
            __half s1 = gdn_f16_store(h1);
            __half s2 = gdn_f16_store(h2);
            __half s3 = gdn_f16_store(h3);
            H[(j+0)*WY3RF_VD+tid]=s0; H[(j+1)*WY3RF_VD+tid]=s1;
            H[(j+2)*WY3RF_VD+tid]=s2; H[(j+3)*WY3RF_VD+tid]=s3;
            h0=__half2float(s0); h1=__half2float(s1);
            h2=__half2float(s2); h3=__half2float(s3);
            qd2 += h0*sq2[j]+h1*sq2[j+1]+h2*sq2[j+2]+h3*sq2[j+3];
        }

        float s = rsqrtf((float)k_dim);
        output[(b*3*num_v_heads+vh)*v_dim+tid]     = __float2bfloat16(qd0*s);
        output[((b*3+1)*num_v_heads+vh)*v_dim+tid] = __float2bfloat16(qd1*s);
        output[((b*3+2)*num_v_heads+vh)*v_dim+tid] = __float2bfloat16(qd2*s);
    }
}
