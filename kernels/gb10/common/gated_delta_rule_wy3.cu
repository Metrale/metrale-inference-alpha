// SPDX-License-Identifier: MIT OR Apache-2.0

// 2026-09-25: K=3 GDN verify step in WY form: three tokens, two passes over the h-state.
//
// Owner: gb10 kernels.
// Invariants:
// - Pass 1 reads H once; pass 2 reads it again and writes H_1 to Hi0, H_2 to Hi1, H_3 to H.
// - Output row b*3+t holds H_{t+1}^T q_t / sqrt(k_dim) for head vh.
// Grid (num_v_heads, batch), block 128. Needs k_dim, v_dim <= 128 and k_dim % 4 == 0.

#include <cuda_bf16.h>
#include "gdn_reduce.cuh"
#define BLOCK_SIZE 128




extern "C" __global__ void gated_delta_rule_wy3(
    float* __restrict__ h_state,
    const __nv_bfloat16* __restrict__ query,
    const __nv_bfloat16* __restrict__ key,
    const __nv_bfloat16* __restrict__ value,
    const float* __restrict__ gate,
    const float* __restrict__ beta,
    __nv_bfloat16* __restrict__ output,
    float* __restrict__ h_state_inter0,
    float* __restrict__ h_state_inter1,
    unsigned int batch_size,
    unsigned int num_k_heads,
    unsigned int num_v_heads,
    unsigned int k_dim,
    unsigned int v_dim,
    unsigned int qk_stride,
    unsigned int v_stride,
    unsigned int gb_stride,







    // 2026-09-25: 0: the three state arguments are contiguous bases indexed by
    // (b * num_v_heads + vh) * k_dim * v_dim, which assumes Hi0 and Hi1 share h_state's
    // per-sequence stride. The pool does not place intermediates that way (h_intermediate
    // in model-engine ssm_pool_slots.rs), so ops::gdn_decode_wy3 accepts 0 only at
    // batch_size 1.
    // 1: each is a device table of `batch_size` per-sequence base pointers; head vh starts
    // at table[b] + vh * k_dim * v_dim.
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


    // 2026-09-25: Token t of sequence b is input row b*3+t: q and k at row*qk_stride +
    // kh*k_dim, v at row*v_stride + vh*v_dim, gate and beta at row*gb_stride + vh. The gate
    // is clamped to [1e-6, 1 - 1e-6], the clamp gated_delta_rule_decode applies.
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


        float hk0=0, hk1p=0, hk2p=0;
        #pragma unroll 4
        for (unsigned int j=0; j<k_dim; j+=4) {
            float h0=H[(j+0)*v_dim+tid], h1=H[(j+1)*v_dim+tid];
            float h2=H[(j+2)*v_dim+tid], h3=H[(j+3)*v_dim+tid];
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
            H[(j+0)*v_dim+tid]=h0; H[(j+1)*v_dim+tid]=h1;
            H[(j+2)*v_dim+tid]=h2; H[(j+3)*v_dim+tid]=h3;
            qd2 += h0*sq2[j]+h1*sq2[j+1]+h2*sq2[j+2]+h3*sq2[j+3];
        }

        float s = rsqrtf((float)k_dim);
        output[(b*3*num_v_heads+vh)*v_dim+tid]     = __float2bfloat16(qd0*s);
        output[((b*3+1)*num_v_heads+vh)*v_dim+tid] = __float2bfloat16(qd1*s);
        output[((b*3+2)*num_v_heads+vh)*v_dim+tid] = __float2bfloat16(qd2*s);
    }
}
