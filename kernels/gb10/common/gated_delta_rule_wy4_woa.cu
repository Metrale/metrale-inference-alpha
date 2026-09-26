// SPDX-License-Identifier: MIT OR Apache-2.0













// 2026-09-25: Write-on-accept K=4 GDN verify: the verify pass writes no h-state, and a
// fold applies only the accepted rows once the verdict is known.
//
// Owner: gb10 kernels.
// Invariants:
// - gated_delta_rule_wy4_woa computes `output` with gated_delta_rule_wy4's expressions in
//   wy4's order and writes no h-state. It stashes what the fold needs: the four vn
//   vectors, the four clamped gates and the four key rows, as the floats it used.
// - gated_delta_rule_wy4_fold applies rows 0..na-1 to H with wy4's update expression in
//   row order, reading and writing H once, so H ends holding what wy4 leaves in Hi(na-1),
//   or in its final H when na == 4.
// - When `engaged_flag` is 0 the fold instead restores H from Hi(na-1) for na < k_rows
//   and leaves H alone otherwise, which is the parent kernels' partial-accept restore.
// gdn_woa_oracle (model-arch examples) checks the output and every na bitwise against wy4.
//
// State traffic per step: wy4 reads H twice and writes Hi0, Hi1, Hi2 and H; this pair
// reads H once in the verify and once more, plus one write, in the fold.
// The verify takes k_dim * v_dim * 4 bytes (64 KiB) of dynamic shared memory. The CUDA
// launch path (gpu-runtime registry.rs) raises the function's limit for any launch above
// 48 KiB, and a refused launch is returned as an error.
//
// woa_decision (model-layers qwen3_ssm/woa.rs) runs the pair only when the caller asks for
// it (only the DFlash batched verify does), K == 4, METRALE_GDN_WOA=1, the h-state is
// FP32, k_dim == v_dim == 128, all three kernels are linked and the bound stash covers
// the batch. Grid (num_v_heads, batch), block 128. The state arguments are device pointer
// tables only: slab 0 of the verify WY tables, the slab wy4's state_is_table = 1 form reads.
// provenance-id: 526f6e616c6420522e205374657369616b

#include <cuda_bf16.h>
#include "gdn_reduce.cuh"
#define BLOCK_SIZE 128
#define WOA_KD 128

// 2026-09-25: Stash per sequence, in floats: vn[4][num_v_heads][v_dim] | g[4][num_v_heads] |
// sk[4][num_k_heads][k_dim]. Sequence b's stash starts at b * stash_seq_floats, a count the
// host computes with stash_seq_floats (qwen3_ssm/woa.rs).
#define WOA_VN(base, T, VH, VD)      ((base) + ((T) * num_v_heads + (VH)) * (VD))
#define WOA_G(base, T, VH)           ((base) + 4 * num_v_heads * v_dim + (T) * num_v_heads + (VH))
#define WOA_SK(base, T, KH, KD)      ((base) + 4 * num_v_heads * v_dim + 4 * num_v_heads + ((T) * num_k_heads + (KH)) * (KD))

extern "C" __global__ void gated_delta_rule_wy4_woa(
    const float* __restrict__ h_state_table,
    const __nv_bfloat16* __restrict__ query,
    const __nv_bfloat16* __restrict__ key,
    const __nv_bfloat16* __restrict__ value,
    const float* __restrict__ gate,
    const float* __restrict__ beta,
    __nv_bfloat16* __restrict__ output,
    float* __restrict__ stash,
    unsigned int batch_size,
    unsigned int num_k_heads,
    unsigned int num_v_heads,
    unsigned int k_dim,
    unsigned int v_dim,
    unsigned int qk_stride,
    unsigned int v_stride,
    unsigned int gb_stride,
    unsigned int stash_seq_floats,

    // 2026-09-25: This layer's engaged word. The kernel sets it to 1 inside the captured
    // graph, so a replay sets it too. The host clears it with gated_delta_rule_wy4_flag_clear
    // at the start of every batched verify that requests write-on-accept, in the same
    // capture, so the word describes that launch only; 0 means a parent WY kernel ran.
    unsigned int* __restrict__ engaged_flag
) {
    const unsigned int vh = blockIdx.x;
    const unsigned int b = blockIdx.y;
    if (vh >= num_v_heads || b >= batch_size) return;

    // 2026-09-25: k_dim == v_dim == 128 is checked on the host (woa_decision). There is no
    // in-kernel guard: an early return here would leave `output` unwritten.
    if (threadIdx.x == 0) *engaged_flag = 1u;

    const unsigned int tid = threadIdx.x;
    const unsigned int hr = num_v_heads / num_k_heads;
    const unsigned int kh = vh / hr;
    const unsigned int hv = k_dim * v_dim;
    const unsigned long long head_off = (unsigned long long)vh * hv;
    const float* H = ((const float* const*)h_state_table)[b] + head_off;
    float* sb = stash + (unsigned long long)b * stash_seq_floats;

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
        // 2026-09-25: Key rows for the fold, written by one v-head per k-head (vh % hr == 0),
        // as the same floats used below.
        if (vh % hr == 0) {
            WOA_SK(sb, 0, kh, k_dim)[tid] = sk0[tid];
            WOA_SK(sb, 1, kh, k_dim)[tid] = sk1[tid];
            WOA_SK(sb, 2, kh, k_dim)[tid] = sk2[tid];
            WOA_SK(sb, 3, kh, k_dim)[tid] = sk3[tid];
        }
    }
    if (tid == 0) {
        *WOA_G(sb, 0, vh) = g0; *WOA_G(sb, 1, vh) = g1;
        *WOA_G(sb, 2, vh) = g2; *WOA_G(sb, 3, vh) = g3;
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

    // 2026-09-25: The head's H slice (k_dim x v_dim floats) lives in dynamic shared memory:
    // pass 1 fills it from global memory once and pass 2 reads it back. Column-per-thread
    // indexing (j*v_dim + tid) puts consecutive threads on consecutive words, so it is
    // bank-conflict-free.
    extern __shared__ float hs[];

    if (tid < v_dim) {
        float vi0=(float)v0[tid], vi1=(float)v1[tid];
        float vi2=(float)v2[tid], vi3=(float)v3[tid];


        float hk0=0, hk1p=0, hk2p=0, hk3p=0;
        #pragma unroll 8
        for (unsigned int j=0; j<WOA_KD; j+=4) {
            float h0=H[(j+0)*v_dim+tid], h1=H[(j+1)*v_dim+tid];
            float h2=H[(j+2)*v_dim+tid], h3=H[(j+3)*v_dim+tid];
            hs[(j+0)*v_dim+tid]=h0; hs[(j+1)*v_dim+tid]=h1;
            hs[(j+2)*v_dim+tid]=h2; hs[(j+3)*v_dim+tid]=h3;
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


        WOA_VN(sb, 0, vh, v_dim)[tid] = vn0;
        WOA_VN(sb, 1, vh, v_dim)[tid] = vn1;
        WOA_VN(sb, 2, vh, v_dim)[tid] = vn2;
        WOA_VN(sb, 3, vh, v_dim)[tid] = vn3;


        float qd0=0, qd1=0, qd2=0, qd3=0;
        #pragma unroll 8
        for (unsigned int j=0; j<WOA_KD; j+=4) {
            float h0=hs[(j+0)*v_dim+tid], h1=hs[(j+1)*v_dim+tid];
            float h2=hs[(j+2)*v_dim+tid], h3=hs[(j+3)*v_dim+tid];
            h0=g0*h0+sk0[j]*vn0; h1=g0*h1+sk0[j+1]*vn0;
            h2=g0*h2+sk0[j+2]*vn0; h3=g0*h3+sk0[j+3]*vn0;
            qd0 += h0*sq0[j]+h1*sq0[j+1]+h2*sq0[j+2]+h3*sq0[j+3];
            h0=g1*h0+sk1[j]*vn1; h1=g1*h1+sk1[j+1]*vn1;
            h2=g1*h2+sk1[j+2]*vn1; h3=g1*h3+sk1[j+3]*vn1;
            qd1 += h0*sq1[j]+h1*sq1[j+1]+h2*sq1[j+2]+h3*sq1[j+3];
            h0=g2*h0+sk2[j]*vn2; h1=g2*h1+sk2[j+1]*vn2;
            h2=g2*h2+sk2[j+2]*vn2; h3=g2*h3+sk2[j+3]*vn2;
            qd2 += h0*sq2[j]+h1*sq2[j+1]+h2*sq2[j+2]+h3*sq2[j+3];
            h0=g3*h0+sk3[j]*vn3; h1=g3*h1+sk3[j+1]*vn3;
            h2=g3*h2+sk3[j+2]*vn3; h3=g3*h3+sk3[j+3]*vn3;
            qd3 += h0*sq3[j]+h1*sq3[j+1]+h2*sq3[j+2]+h3*sq3[j+3];
        }

        float s = rsqrtf((float)k_dim);
        output[(b*4*num_v_heads+vh)*v_dim+tid]     = __float2bfloat16(qd0*s);
        output[((b*4+1)*num_v_heads+vh)*v_dim+tid] = __float2bfloat16(qd1*s);
        output[((b*4+2)*num_v_heads+vh)*v_dim+tid] = __float2bfloat16(qd2*s);
        output[((b*4+3)*num_v_heads+vh)*v_dim+tid] = __float2bfloat16(qd3*s);
    }
}


// 2026-09-25: Apply the accepted rows 0..na-1 (na = na_tab[b]) to H, reading and writing it
// once; na == 0 leaves H unchanged. `hi_tables` is the verify's Hi0 slab, and Hi(t) is
// `slab_entries` pointers further on. `k_rows` is the verify width of the launch that ran.
// When `engaged_flag` is 0 a parent WY kernel ran, and this launch performs the parent's
// partial-accept restore instead, so the host need not know which kernel the graph replayed.
extern "C" __global__ void gated_delta_rule_wy4_fold(
    float* __restrict__ h_state_table,
    const float* __restrict__ stash,
    const unsigned int* __restrict__ na_tab,
    const float* __restrict__ hi_tables,
    unsigned int slab_entries,
    const unsigned int* __restrict__ engaged_flag,
    unsigned int k_rows,
    unsigned int batch_size,
    unsigned int num_k_heads,
    unsigned int num_v_heads,
    unsigned int k_dim,
    unsigned int v_dim,
    unsigned int stash_seq_floats
) {
    const unsigned int vh = blockIdx.x;
    const unsigned int b = blockIdx.y;
    if (vh >= num_v_heads || b >= batch_size) return;
    const unsigned int na = na_tab[b];
    if (na == 0) return;

    const unsigned int tid = threadIdx.x;
    const unsigned int hr = num_v_heads / num_k_heads;
    const unsigned int kh = vh / hr;
    const unsigned int hv = k_dim * v_dim;
    float* H = ((float* const*)h_state_table)[b] + (unsigned long long)vh * hv;
    const float* sb = stash + (unsigned long long)b * stash_seq_floats;

    if (*engaged_flag == 0u) {
        // 2026-09-25: A parent WY kernel ran (wy2, wy3, wy4 or a wyN table twin) and wrote
        // the final H and Hi0..Hi(k_rows-2). A full accept keeps H; a partial accept of na
        // rows restores Hi(na-1).
        if (na >= k_rows) return;
        const float* src = ((const float* const*)hi_tables)[(na - 1) * slab_entries + b]
                           + (unsigned long long)vh * hv;
        for (unsigned int i = tid; i < hv; i += BLOCK_SIZE) H[i] = src[i];
        return;
    }

    __shared__ float sk0[128], sk1[128], sk2[128], sk3[128];
    if (tid < k_dim) {
        sk0[tid] = WOA_SK(sb, 0, kh, k_dim)[tid];
        sk1[tid] = WOA_SK(sb, 1, kh, k_dim)[tid];
        sk2[tid] = WOA_SK(sb, 2, kh, k_dim)[tid];
        sk3[tid] = WOA_SK(sb, 3, kh, k_dim)[tid];
    }
    __syncthreads();
    if (tid >= v_dim) return;

    const float g0 = *WOA_G(sb, 0, vh), g1 = *WOA_G(sb, 1, vh);
    const float g2 = *WOA_G(sb, 2, vh), g3 = *WOA_G(sb, 3, vh);
    const float vn0 = WOA_VN(sb, 0, vh, v_dim)[tid], vn1 = WOA_VN(sb, 1, vh, v_dim)[tid];
    const float vn2 = WOA_VN(sb, 2, vh, v_dim)[tid], vn3 = WOA_VN(sb, 3, vh, v_dim)[tid];

    // 2026-09-25: Dims are checked on the host, as for the verify kernel above.
    #pragma unroll
    for (unsigned int j=0; j<WOA_KD; j+=4) {
        float h0=H[(j+0)*v_dim+tid], h1=H[(j+1)*v_dim+tid];
        float h2=H[(j+2)*v_dim+tid], h3=H[(j+3)*v_dim+tid];
        h0=g0*h0+sk0[j]*vn0; h1=g0*h1+sk0[j+1]*vn0;
        h2=g0*h2+sk0[j+2]*vn0; h3=g0*h3+sk0[j+3]*vn0;
        if (na > 1) {
            h0=g1*h0+sk1[j]*vn1; h1=g1*h1+sk1[j+1]*vn1;
            h2=g1*h2+sk1[j+2]*vn1; h3=g1*h3+sk1[j+3]*vn1;
        }
        if (na > 2) {
            h0=g2*h0+sk2[j]*vn2; h1=g2*h1+sk2[j+1]*vn2;
            h2=g2*h2+sk2[j+2]*vn2; h3=g2*h3+sk2[j+3]*vn2;
        }
        if (na > 3) {
            h0=g3*h0+sk3[j]*vn3; h1=g3*h1+sk3[j+1]*vn3;
            h2=g3*h2+sk3[j+2]*vn3; h3=g3*h3+sk3[j+3]*vn3;
        }
        H[(j+0)*v_dim+tid]=h0; H[(j+1)*v_dim+tid]=h1;
        H[(j+2)*v_dim+tid]=h2; H[(j+3)*v_dim+tid]=h3;
    }
}
// 2026-09-25: Clear the layer's engaged word. The host runs it at the start of every batched
// verify that requests write-on-accept (woa_clear_at_entry in qwen3_ssm/woa.rs).
extern "C" __global__ void gated_delta_rule_wy4_flag_clear(unsigned int* __restrict__ engaged_flag) {
    if (threadIdx.x == 0 && blockIdx.x == 0) *engaged_flag = 0u;
}
