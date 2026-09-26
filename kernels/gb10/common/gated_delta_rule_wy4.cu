// SPDX-License-Identifier: MIT OR Apache-2.0

// 2026-09-25: K=4 GDN verify step in WY form: four tokens, two passes over the h-state.
//
// Owner: gb10 kernels.
// Invariants:
// - Pass 1 reads H once; pass 2 reads it again and writes H_1..H_3 to Hi0..Hi2, H_4 to H.
// - Output row b*4+t holds H_{t+1}^T q_t / sqrt(k_dim) for head vh.
// Grid (num_v_heads, batch), block 128. Needs k_dim, v_dim <= 128 and k_dim % 4 == 0.

#include <cuda_bf16.h>
#include "gdn_reduce.cuh"
#define BLOCK_SIZE 128




extern "C" __global__ void gated_delta_rule_wy4(
    float* __restrict__ h_state,
    const __nv_bfloat16* __restrict__ query,
    const __nv_bfloat16* __restrict__ key,
    const __nv_bfloat16* __restrict__ value,
    const float* __restrict__ gate,
    const float* __restrict__ beta,
    __nv_bfloat16* __restrict__ output,
    float* __restrict__ h_state_inter0,
    float* __restrict__ h_state_inter1,
    float* __restrict__ h_state_inter2,
    unsigned int batch_size,
    unsigned int num_k_heads,
    unsigned int num_v_heads,
    unsigned int k_dim,
    unsigned int v_dim,
    unsigned int qk_stride,
    unsigned int v_stride,
    unsigned int gb_stride,







    // 2026-09-25: 0: the four state arguments are contiguous bases indexed by
    // (b * num_v_heads + vh) * k_dim * v_dim, which assumes Hi0..Hi2 share h_state's
    // per-sequence stride. The pool does not place intermediates that way (h_intermediate
    // in model-engine ssm_pool_slots.rs), so ops::gdn_decode_wy4 accepts 0 only at
    // batch_size 1.
    // 1: each is a device table of `batch_size` per-sequence base pointers; head vh starts
    // at table[b] + vh * k_dim * v_dim. Tables also free the sequences of a batch from
    // occupying consecutive slots.
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
    float* H   = state_is_table ? ((float* const*)h_state)[b]        + head_off
                                : h_state        + flat_off;
    float* Hi0 = state_is_table ? ((float* const*)h_state_inter0)[b] + head_off
                                : h_state_inter0 + flat_off;
    float* Hi1 = state_is_table ? ((float* const*)h_state_inter1)[b] + head_off
                                : h_state_inter1 + flat_off;
    float* Hi2 = state_is_table ? ((float* const*)h_state_inter2)[b] + head_off
                                : h_state_inter2 + flat_off;


    // 2026-09-25: Token t of sequence b is input row b*4+t: q and k at row*qk_stride +
    // kh*k_dim, v at row*v_stride + vh*v_dim, gate and beta at row*gb_stride + vh. The gate
    // is clamped to [1e-6, 1 - 1e-6], the clamp gated_delta_rule_decode applies.
    #define TP(T) \
        const __nv_bfloat16* q##T = query + (b*4+T)*qk_stride + kh*k_dim; \
        const __nv_bfloat16* k##T = key   + (b*4+T)*qk_stride + kh*k_dim; \
        const __nv_bfloat16* v##T = value + (b*4+T)*v_stride  + vh*v_dim; \
        const float g##T = fminf(fmaxf(gate[(b*4+T)*gb_stride + vh], 1e-6f), 1.0f - 1e-6f); \
        const float bt##T = beta[(b*4+T)*gb_stride + vh];
    TP(0) TP(1) TP(2) TP(3)
    #undef TP

    __shared__ float sk0[128], sq0[128], sk1[128], sq1[128];
    __shared__ float sk2[128], sq2[128], sk3[128], sq3[128];
    __shared__ float smem_warp[4];
    __shared__ float kd10, kd20, kd21, kd30, kd31, kd32;

    if (tid < k_dim) {
        sk0[tid]=(float)k0[tid]; sq0[tid]=(float)q0[tid];
        sk1[tid]=(float)k1[tid]; sq1[tid]=(float)q1[tid];
        sk2[tid]=(float)k2[tid]; sq2[tid]=(float)q2[tid];
        sk3[tid]=(float)k3[tid]; sq3[tid]=(float)q3[tid];
    }
    __syncthreads();


    #define KDOT(NAME, A, B) { \
        float p = (tid<k_dim) ? s##A[tid]*s##B[tid] : 0.0f; \
        float r = metrale_block_reduce_sum(p, smem_warp, tid); \
        if (tid==0) NAME = r; \
        __syncthreads(); \
    }
    KDOT(kd10, k1, k0)
    KDOT(kd20, k2, k0)
    KDOT(kd21, k2, k1)
    KDOT(kd30, k3, k0)
    KDOT(kd31, k3, k1)
    KDOT(kd32, k3, k2)
    #undef KDOT

    if (tid < v_dim) {
        float vi0=(float)v0[tid], vi1=(float)v1[tid];
        float vi2=(float)v2[tid], vi3=(float)v3[tid];


        float hk0=0, hk1p=0, hk2p=0, hk3p=0;
        #pragma unroll 4
        for (unsigned int j=0; j<k_dim; j+=4) {
            float h0=H[(j+0)*v_dim+tid], h1=H[(j+1)*v_dim+tid];
            float h2=H[(j+2)*v_dim+tid], h3=H[(j+3)*v_dim+tid];
            hk0  += h0*sk0[j]+h1*sk0[j+1]+h2*sk0[j+2]+h3*sk0[j+3];
            hk1p += h0*sk1[j]+h1*sk1[j+1]+h2*sk1[j+2]+h3*sk1[j+3];
            hk2p += h0*sk2[j]+h1*sk2[j+1]+h2*sk2[j+2]+h3*sk2[j+3];
            hk3p += h0*sk3[j]+h1*sk3[j+1]+h2*sk3[j+2]+h3*sk3[j+3];
        }


        float vn0 = (vi0 - g0*hk0) * bt0;
        float hk1c = g0*hk1p + kd10*vn0;
        float vn1 = (vi1 - g1*hk1c) * bt1;
        float hk2c = g0*g1*hk2p + g1*kd20*vn0 + kd21*vn1;
        float vn2 = (vi2 - g2*hk2c) * bt2;
        float hk3c = g0*g1*g2*hk3p + g1*g2*kd30*vn0 + g2*kd31*vn1 + kd32*vn2;
        float vn3 = (vi3 - g3*hk3c) * bt3;


        float qd0=0, qd1=0, qd2=0, qd3=0;
        #pragma unroll 4
        for (unsigned int j=0; j<k_dim; j+=4) {
            float h0=H[(j+0)*v_dim+tid], h1=H[(j+1)*v_dim+tid];
            float h2=H[(j+2)*v_dim+tid], h3=H[(j+3)*v_dim+tid];

            h0=g0*h0+sk0[j]*vn0; h1=g0*h1+sk0[j+1]*vn0;
            h2=g0*h2+sk0[j+2]*vn0; h3=g0*h3+sk0[j+3]*vn0;
            Hi0[(j+0)*v_dim+tid]=h0; Hi0[(j+1)*v_dim+tid]=h1;
            Hi0[(j+2)*v_dim+tid]=h2; Hi0[(j+3)*v_dim+tid]=h3;
            qd0 += h0*sq0[j]+h1*sq0[j+1]+h2*sq0[j+2]+h3*sq0[j+3];

            h0=g1*h0+sk1[j]*vn1; h1=g1*h1+sk1[j+1]*vn1;
            h2=g1*h2+sk1[j+2]*vn1; h3=g1*h3+sk1[j+3]*vn1;
            Hi1[(j+0)*v_dim+tid]=h0; Hi1[(j+1)*v_dim+tid]=h1;
            Hi1[(j+2)*v_dim+tid]=h2; Hi1[(j+3)*v_dim+tid]=h3;
            qd1 += h0*sq1[j]+h1*sq1[j+1]+h2*sq1[j+2]+h3*sq1[j+3];

            h0=g2*h0+sk2[j]*vn2; h1=g2*h1+sk2[j+1]*vn2;
            h2=g2*h2+sk2[j+2]*vn2; h3=g2*h3+sk2[j+3]*vn2;
            Hi2[(j+0)*v_dim+tid]=h0; Hi2[(j+1)*v_dim+tid]=h1;
            Hi2[(j+2)*v_dim+tid]=h2; Hi2[(j+3)*v_dim+tid]=h3;
            qd2 += h0*sq2[j]+h1*sq2[j+1]+h2*sq2[j+2]+h3*sq2[j+3];

            h0=g3*h0+sk3[j]*vn3; h1=g3*h1+sk3[j+1]*vn3;
            h2=g3*h2+sk3[j+2]*vn3; h3=g3*h3+sk3[j+3]*vn3;
            H[(j+0)*v_dim+tid]=h0; H[(j+1)*v_dim+tid]=h1;
            H[(j+2)*v_dim+tid]=h2; H[(j+3)*v_dim+tid]=h3;
            qd3 += h0*sq3[j]+h1*sq3[j+1]+h2*sq3[j+2]+h3*sq3[j+3];
        }

        float s = rsqrtf((float)k_dim);
        output[(b*4*num_v_heads+vh)*v_dim+tid]     = __float2bfloat16(qd0*s);
        output[((b*4+1)*num_v_heads+vh)*v_dim+tid] = __float2bfloat16(qd1*s);
        output[((b*4+2)*num_v_heads+vh)*v_dim+tid] = __float2bfloat16(qd2*s);
        output[((b*4+3)*num_v_heads+vh)*v_dim+tid] = __float2bfloat16(qd3*s);
    }
}
