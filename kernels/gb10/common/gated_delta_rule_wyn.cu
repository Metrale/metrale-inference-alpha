// SPDX-License-Identifier: MIT OR Apache-2.0

// 2026-09-25: GDN verify step in WY form for K = 5..16, one template instantiated per K.
//
// Owner: gb10 kernels.
// Invariants:
// - Pass 1 reads H once for the K dots H^T k_t; pass 2 reads H again and writes the state
//   after token t to Hi_t for t = 0..K-2 and the state after token K-1 to H.
// - Output row b*K+t holds (state after token t)^T q_t / sqrt(k_dim) for head vh.
// - The gate clamp [1e-6, 1 - 1e-6] and the k.k reductions (metrale_block_reduce_sum) are
//   the ones gated_delta_rule_wy2/wy3/wy4 use.
// - The `_f16` twins differ from the FP32 impl only by the FP16 round trip described there.
//
// Entry points, for each K = 5..16: gated_delta_rule_wy{K} (contiguous state),
// gated_delta_rule_wy{K}_table (pointer tables), and the same two with an `_f16` suffix.
// The contiguous argument list is gated_delta_rule_wy17's (qwen3.6-35b-a3b), so
// ops::gdn_decode_wyn launches both. Static shared memory: sk and sq hold 2 * K * 128
// floats (16 KiB at K = 16).
// Grid (num_v_heads, batch), block 128. Needs k_dim, v_dim <= 128 and k_dim % 4 == 0.











#include <cuda_bf16.h>
#include "gdn_reduce.cuh"
#include "gdn_f16_state.cuh"
#define BLOCK_SIZE 128



// 2026-09-25: Contiguous form: H is at h_state + (b*num_v_heads + vh) * k_dim * v_dim, and
// Hi_t at the same offset from h_state_inter_base plus t * inter_stride_floats. The host
// passes the pool pitch h_bytes / 4 (qwen3_ssm/trait_decode_batched_conv_gdn_wyn.rs), and
// ops::gdn_decode_wyn launches this form at batch_size 1 only.

template <int K_TOKENS>
__device__ __forceinline__ void gated_delta_rule_wyn_impl(
    float* __restrict__ h_state,
    const __nv_bfloat16* __restrict__ query,
    const __nv_bfloat16* __restrict__ key,
    const __nv_bfloat16* __restrict__ value,
    const float* __restrict__ gate,
    const float* __restrict__ beta,
    __nv_bfloat16* __restrict__ output,
    float* __restrict__ h_state_inter_base,
    unsigned int inter_stride_floats,
    unsigned int batch_size,
    unsigned int num_k_heads,
    unsigned int num_v_heads,
    unsigned int k_dim,
    unsigned int v_dim,
    unsigned int qk_stride,
    unsigned int v_stride,
    unsigned int gb_stride,





    // 2026-09-25: 0: contiguous bases, as described above the template.
    // 1: h_state is a table of `batch_size` per-sequence H base pointers, h_state_inter_base
    //    points at the Hi0 table, and inter_stride_floats is read as the number of pointer
    //    entries between consecutive Hi tables (ops::gdn_decode_wyn_table passes
    //    VERIFY_WY_TABLE_SEQS). Head vh starts at the entry plus vh * k_dim * v_dim.
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
    const unsigned long long flat_off =
        (unsigned long long)(b * num_v_heads + vh) * hv;
    float* H = state_is_table ? ((float* const*)h_state)[b] + head_off
                              : h_state + flat_off;



    float* Hi_ptrs[K_TOKENS - 1];
    #pragma unroll
    for (int t = 0; t < K_TOKENS - 1; t++) {
        Hi_ptrs[t] = state_is_table
            ? ((float* const*)h_state_inter_base)[b + (unsigned long long)t * inter_stride_floats]
                  + head_off
            : h_state_inter_base + flat_off
                  + (unsigned long long)t * inter_stride_floats;
    }


    __shared__ float sk[K_TOKENS][128];
    __shared__ float sq[K_TOKENS][128];
    __shared__ float sg[K_TOKENS];
    __shared__ float sbt[K_TOKENS];
    __shared__ float smem_warp[4];

    if (tid < k_dim) {
        #pragma unroll
        for (int t = 0; t < K_TOKENS; t++) {
            const __nv_bfloat16* q_t = query + (b * K_TOKENS + t) * qk_stride + kh * k_dim;
            const __nv_bfloat16* k_t = key   + (b * K_TOKENS + t) * qk_stride + kh * k_dim;
            sq[t][tid] = (float)q_t[tid];
            sk[t][tid] = (float)k_t[tid];
        }
    }
    if (tid < K_TOKENS) {

        float g_raw = gate[(b * K_TOKENS + tid) * gb_stride + vh];
        sg[tid] = fminf(fmaxf(g_raw, 1e-6f), 1.0f - 1e-6f);
        sbt[tid] = beta[(b * K_TOKENS + tid) * gb_stride + vh];
    }
    __syncthreads();



    // 2026-09-25: kd_flat[t*(t-1)/2 + s] = k_t . k_s for s < t.
    __shared__ float kd_flat[K_TOKENS * (K_TOKENS - 1) / 2];

    #pragma unroll
    for (int t = 1; t < K_TOKENS; t++) {
        #pragma unroll
        for (int s = 0; s < t; s++) {
            float p = (tid < k_dim) ? sk[t][tid] * sk[s][tid] : 0.0f;
            float r = metrale_block_reduce_sum(p, smem_warp, tid);
            if (tid == 0) {
                kd_flat[t * (t - 1) / 2 + s] = r;
            }
            __syncthreads();
        }
    }

    if (tid < v_dim) {

        float vi[K_TOKENS];
        #pragma unroll
        for (int t = 0; t < K_TOKENS; t++) {
            const __nv_bfloat16* v_t = value + (b * K_TOKENS + t) * v_stride + vh * v_dim;
            vi[t] = (float)v_t[tid];
        }


        float hk[K_TOKENS];
        #pragma unroll
        for (int t = 0; t < K_TOKENS; t++) hk[t] = 0.0f;

        #pragma unroll 4
        for (unsigned int j = 0; j < k_dim; j += 4) {
            float h0 = H[(j + 0) * v_dim + tid];
            float h1 = H[(j + 1) * v_dim + tid];
            float h2 = H[(j + 2) * v_dim + tid];
            float h3 = H[(j + 3) * v_dim + tid];
            #pragma unroll
            for (int t = 0; t < K_TOKENS; t++) {
                hk[t] += h0 * sk[t][j + 0] + h1 * sk[t][j + 1]
                       + h2 * sk[t][j + 2] + h3 * sk[t][j + 3];
            }
        }


        // 2026-09-25: WY correction, in token order:
        //   corrected[t] = prod(g[0..t-1]) * hk[t] + sum_{s<t} prod(g[s+1..t-1]) * kd[t][s] * vn[s]
        //   vn[t]        = (v[t] - g[t] * corrected[t]) * beta[t]
        float vn[K_TOKENS];
        vn[0] = (vi[0] - sg[0] * hk[0]) * sbt[0];
        for (int t = 1; t < K_TOKENS; t++) {
            float lead_prod = 1.0f;
            for (int u = 0; u < t; u++) lead_prod *= sg[u];
            float corrected = lead_prod * hk[t];
            for (int s = 0; s < t; s++) {
                float gprod = 1.0f;
                for (int u = s + 1; u < t; u++) gprod *= sg[u];
                corrected += gprod * kd_flat[t * (t - 1) / 2 + s] * vn[s];
            }
            vn[t] = (vi[t] - sg[t] * corrected) * sbt[t];
        }




        float qd[K_TOKENS];
        #pragma unroll
        for (int t = 0; t < K_TOKENS; t++) qd[t] = 0.0f;

        #pragma unroll 4
        for (unsigned int j = 0; j < k_dim; j += 4) {
            float h0 = H[(j + 0) * v_dim + tid];
            float h1 = H[(j + 1) * v_dim + tid];
            float h2 = H[(j + 2) * v_dim + tid];
            float h3 = H[(j + 3) * v_dim + tid];

            #pragma unroll
            for (int t = 0; t < K_TOKENS; t++) {
                h0 = sg[t] * h0 + sk[t][j + 0] * vn[t];
                h1 = sg[t] * h1 + sk[t][j + 1] * vn[t];
                h2 = sg[t] * h2 + sk[t][j + 2] * vn[t];
                h3 = sg[t] * h3 + sk[t][j + 3] * vn[t];
                if (t < K_TOKENS - 1) {
                    float* Hi_t = Hi_ptrs[t];
                    Hi_t[(j + 0) * v_dim + tid] = h0;
                    Hi_t[(j + 1) * v_dim + tid] = h1;
                    Hi_t[(j + 2) * v_dim + tid] = h2;
                    Hi_t[(j + 3) * v_dim + tid] = h3;
                } else {
                    H[(j + 0) * v_dim + tid] = h0;
                    H[(j + 1) * v_dim + tid] = h1;
                    H[(j + 2) * v_dim + tid] = h2;
                    H[(j + 3) * v_dim + tid] = h3;
                }
                qd[t] += h0 * sq[t][j + 0] + h1 * sq[t][j + 1]
                       + h2 * sq[t][j + 2] + h3 * sq[t][j + 3];
            }
        }


        float s = rsqrtf((float)k_dim);
        #pragma unroll
        for (int t = 0; t < K_TOKENS; t++) {
            output[((b * K_TOKENS + t) * num_v_heads + vh) * v_dim + tid] =
                __float2bfloat16(qd[t] * s);
        }
    }
}

// 2026-09-25: FP32 entry points. The host picks the handle by verify width
// (Qwen3SsmLayer::wyn_kernel, qwen3_ssm/trait_decode_batched_conv_gdn_wyn.rs).
#define METRALE_WYN_INSTANTIATE(K)                                              \
    extern "C" __global__ void gated_delta_rule_wy##K(                        \
        float* __restrict__ h_state,                                          \
        const __nv_bfloat16* __restrict__ query,                              \
        const __nv_bfloat16* __restrict__ key,                                \
        const __nv_bfloat16* __restrict__ value,                              \
        const float* __restrict__ gate,                                       \
        const float* __restrict__ beta,                                       \
        __nv_bfloat16* __restrict__ output,                                   \
        float* __restrict__ h_state_inter_base,                               \
        unsigned int inter_stride_floats,                                     \
        unsigned int batch_size,                                              \
        unsigned int num_k_heads,                                             \
        unsigned int num_v_heads,                                             \
        unsigned int k_dim,                                                   \
        unsigned int v_dim,                                                   \
        unsigned int qk_stride,                                               \
        unsigned int v_stride,                                                \
        unsigned int gb_stride                                                \
    ) {                                                                       \
        gated_delta_rule_wyn_impl<K>(                                         \
            h_state, query, key, value, gate, beta, output,                   \
            h_state_inter_base, inter_stride_floats, batch_size,              \
            num_k_heads, num_v_heads, k_dim, v_dim, qk_stride, v_stride,      \
            gb_stride, 0u);                                                   \
    }                                                                         \
    /* 2026-09-25: Pointer-table twin: this impl with state_is_table = 1. */  \
                                                                              \
                                                                              \
                                                                              \
    extern "C" __global__ void gated_delta_rule_wy##K##_table(                \
        float* __restrict__ h_state,                                          \
        const __nv_bfloat16* __restrict__ query,                              \
        const __nv_bfloat16* __restrict__ key,                                \
        const __nv_bfloat16* __restrict__ value,                              \
        const float* __restrict__ gate,                                       \
        const float* __restrict__ beta,                                       \
        __nv_bfloat16* __restrict__ output,                                   \
        float* __restrict__ h_state_inter_base,                               \
        unsigned int inter_stride_floats,                                     \
        unsigned int batch_size,                                              \
        unsigned int num_k_heads,                                             \
        unsigned int num_v_heads,                                             \
        unsigned int k_dim,                                                   \
        unsigned int v_dim,                                                   \
        unsigned int qk_stride,                                               \
        unsigned int v_stride,                                                \
        unsigned int gb_stride                                                \
    ) {                                                                       \
        gated_delta_rule_wyn_impl<K>(                                         \
            h_state, query, key, value, gate, beta, output,                   \
            h_state_inter_base, inter_stride_floats, batch_size,              \
            num_k_heads, num_v_heads, k_dim, v_dim, qk_stride, v_stride,      \
            gb_stride, 1u);                                                   \
    }

METRALE_WYN_INSTANTIATE(5)
METRALE_WYN_INSTANTIATE(6)
METRALE_WYN_INSTANTIATE(7)
METRALE_WYN_INSTANTIATE(8)






// 2026-09-25: K = 9..16.
// provenance-id: 526f6e616c6420522e205374657369616b
METRALE_WYN_INSTANTIATE(9)
METRALE_WYN_INSTANTIATE(10)
METRALE_WYN_INSTANTIATE(11)
METRALE_WYN_INSTANTIATE(12)
METRALE_WYN_INSTANTIATE(13)
METRALE_WYN_INSTANTIATE(14)
METRALE_WYN_INSTANTIATE(15)
METRALE_WYN_INSTANTIATE(16)

#undef METRALE_WYN_INSTANTIATE












// 2026-09-25: FP16 h-state twins of gated_delta_rule_wyn_impl.
// Apart from the FP16 round trip, every float expression, gate clamp, accumulation order
// and reduction is gated_delta_rule_wyn_impl's. H and the Hi_t are `__half` in memory, read
// with __half2float and written with gdn_f16_store (gdn_f16_state.cuh). Each token's updated
// state is rounded to FP16 before it is stored, carried to the next token and dotted with q,
// so the state the forward chain carries equals the stored rollback intermediate bit for bit.
// Contiguous form: `inter_stride_halves` is the pool pitch in halves; the host passes
// h_bytes / 2 (qwen3_ssm/trait_decode_batched_conv_gdn_wyn.rs).
// provenance-id: 526f6e616c6420522e205374657369616b

template <int K_TOKENS>
__device__ __forceinline__ void gated_delta_rule_wyn_f16_impl(
    __half* __restrict__ h_state,
    const __nv_bfloat16* __restrict__ query,
    const __nv_bfloat16* __restrict__ key,
    const __nv_bfloat16* __restrict__ value,
    const float* __restrict__ gate,
    const float* __restrict__ beta,
    __nv_bfloat16* __restrict__ output,
    __half* __restrict__ h_state_inter_base,
    unsigned int inter_stride_halves,
    unsigned int batch_size,
    unsigned int num_k_heads,
    unsigned int num_v_heads,
    unsigned int k_dim,
    unsigned int v_dim,
    unsigned int qk_stride,
    unsigned int v_stride,
    unsigned int gb_stride,


    // 2026-09-25: Same meaning as the FP32 impl's `state_is_table`. In the table form the
    // entries are `__half*` and `inter_stride_halves` is read as the number of pointer
    // entries between consecutive Hi tables.
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
    const unsigned long long flat_off =
        (unsigned long long)(b * num_v_heads + vh) * hv;
    __half* H = state_is_table ? ((__half* const*)h_state)[b] + head_off
                               : h_state + flat_off;
    __half* Hi_ptrs[K_TOKENS - 1];
    #pragma unroll
    for (int t = 0; t < K_TOKENS - 1; t++) {
        Hi_ptrs[t] = state_is_table
            ? ((__half* const*)h_state_inter_base)[b + (unsigned long long)t * inter_stride_halves]
                  + head_off
            : h_state_inter_base + flat_off
                  + (unsigned long long)t * inter_stride_halves;
    }

    __shared__ float sk[K_TOKENS][128];
    __shared__ float sq[K_TOKENS][128];
    __shared__ float sg[K_TOKENS];
    __shared__ float sbt[K_TOKENS];
    __shared__ float smem_warp[4];

    if (tid < k_dim) {
        #pragma unroll
        for (int t = 0; t < K_TOKENS; t++) {
            const __nv_bfloat16* q_t = query + (b * K_TOKENS + t) * qk_stride + kh * k_dim;
            const __nv_bfloat16* k_t = key   + (b * K_TOKENS + t) * qk_stride + kh * k_dim;
            sq[t][tid] = (float)q_t[tid];
            sk[t][tid] = (float)k_t[tid];
        }
    }
    if (tid < K_TOKENS) {
        float g_raw = gate[(b * K_TOKENS + tid) * gb_stride + vh];
        sg[tid] = fminf(fmaxf(g_raw, 1e-6f), 1.0f - 1e-6f);
        sbt[tid] = beta[(b * K_TOKENS + tid) * gb_stride + vh];
    }
    __syncthreads();

    __shared__ float kd_flat[K_TOKENS * (K_TOKENS - 1) / 2];

    #pragma unroll
    for (int t = 1; t < K_TOKENS; t++) {
        #pragma unroll
        for (int s = 0; s < t; s++) {
            float p = (tid < k_dim) ? sk[t][tid] * sk[s][tid] : 0.0f;
            float r = metrale_block_reduce_sum(p, smem_warp, tid);
            if (tid == 0) {
                kd_flat[t * (t - 1) / 2 + s] = r;
            }
            __syncthreads();
        }
    }

    if (tid < v_dim) {
        float vi[K_TOKENS];
        #pragma unroll
        for (int t = 0; t < K_TOKENS; t++) {
            const __nv_bfloat16* v_t = value + (b * K_TOKENS + t) * v_stride + vh * v_dim;
            vi[t] = (float)v_t[tid];
        }

        float hk[K_TOKENS];
        #pragma unroll
        for (int t = 0; t < K_TOKENS; t++) hk[t] = 0.0f;

        #pragma unroll 4
        for (unsigned int j = 0; j < k_dim; j += 4) {
            float h0 = __half2float(H[(j + 0) * v_dim + tid]);
            float h1 = __half2float(H[(j + 1) * v_dim + tid]);
            float h2 = __half2float(H[(j + 2) * v_dim + tid]);
            float h3 = __half2float(H[(j + 3) * v_dim + tid]);
            #pragma unroll
            for (int t = 0; t < K_TOKENS; t++) {
                hk[t] += h0 * sk[t][j + 0] + h1 * sk[t][j + 1]
                       + h2 * sk[t][j + 2] + h3 * sk[t][j + 3];
            }
        }

        float vn[K_TOKENS];
        vn[0] = (vi[0] - sg[0] * hk[0]) * sbt[0];
        for (int t = 1; t < K_TOKENS; t++) {
            float lead_prod = 1.0f;
            for (int u = 0; u < t; u++) lead_prod *= sg[u];
            float corrected = lead_prod * hk[t];
            for (int s = 0; s < t; s++) {
                float gprod = 1.0f;
                for (int u = s + 1; u < t; u++) gprod *= sg[u];
                corrected += gprod * kd_flat[t * (t - 1) / 2 + s] * vn[s];
            }
            vn[t] = (vi[t] - sg[t] * corrected) * sbt[t];
        }

        float qd[K_TOKENS];
        #pragma unroll
        for (int t = 0; t < K_TOKENS; t++) qd[t] = 0.0f;

        #pragma unroll 4
        for (unsigned int j = 0; j < k_dim; j += 4) {
            float h0 = __half2float(H[(j + 0) * v_dim + tid]);
            float h1 = __half2float(H[(j + 1) * v_dim + tid]);
            float h2 = __half2float(H[(j + 2) * v_dim + tid]);
            float h3 = __half2float(H[(j + 3) * v_dim + tid]);

            #pragma unroll
            for (int t = 0; t < K_TOKENS; t++) {
                h0 = sg[t] * h0 + sk[t][j + 0] * vn[t];
                h1 = sg[t] * h1 + sk[t][j + 1] * vn[t];
                h2 = sg[t] * h2 + sk[t][j + 2] * vn[t];
                h3 = sg[t] * h3 + sk[t][j + 3] * vn[t];


                h0 = __half2float(gdn_f16_store(h0));
                h1 = __half2float(gdn_f16_store(h1));
                h2 = __half2float(gdn_f16_store(h2));
                h3 = __half2float(gdn_f16_store(h3));
                if (t < K_TOKENS - 1) {
                    __half* Hi_t = Hi_ptrs[t];
                    Hi_t[(j + 0) * v_dim + tid] = gdn_f16_store(h0);
                    Hi_t[(j + 1) * v_dim + tid] = gdn_f16_store(h1);
                    Hi_t[(j + 2) * v_dim + tid] = gdn_f16_store(h2);
                    Hi_t[(j + 3) * v_dim + tid] = gdn_f16_store(h3);
                } else {
                    H[(j + 0) * v_dim + tid] = gdn_f16_store(h0);
                    H[(j + 1) * v_dim + tid] = gdn_f16_store(h1);
                    H[(j + 2) * v_dim + tid] = gdn_f16_store(h2);
                    H[(j + 3) * v_dim + tid] = gdn_f16_store(h3);
                }
                qd[t] += h0 * sq[t][j + 0] + h1 * sq[t][j + 1]
                       + h2 * sq[t][j + 2] + h3 * sq[t][j + 3];
            }
        }

        float s = rsqrtf((float)k_dim);
        #pragma unroll
        for (int t = 0; t < K_TOKENS; t++) {
            output[((b * K_TOKENS + t) * num_v_heads + vh) * v_dim + tid] =
                __float2bfloat16(qd[t] * s);
        }
    }
}

#define METRALE_WYN_F16_INSTANTIATE(K)                                          \
    extern "C" __global__ void gated_delta_rule_wy##K##_f16(                  \
        __half* __restrict__ h_state,                                         \
        const __nv_bfloat16* __restrict__ query,                              \
        const __nv_bfloat16* __restrict__ key,                                \
        const __nv_bfloat16* __restrict__ value,                              \
        const float* __restrict__ gate,                                       \
        const float* __restrict__ beta,                                       \
        __nv_bfloat16* __restrict__ output,                                   \
        __half* __restrict__ h_state_inter_base,                              \
        unsigned int inter_stride_halves,                                      \
        unsigned int batch_size,                                              \
        unsigned int num_k_heads,                                             \
        unsigned int num_v_heads,                                             \
        unsigned int k_dim,                                                   \
        unsigned int v_dim,                                                   \
        unsigned int qk_stride,                                               \
        unsigned int v_stride,                                                \
        unsigned int gb_stride                                                \
    ) {                                                                       \
        gated_delta_rule_wyn_f16_impl<K>(                                     \
            h_state, query, key, value, gate, beta, output,                   \
            h_state_inter_base, inter_stride_halves, batch_size,               \
            num_k_heads, num_v_heads, k_dim, v_dim, qk_stride, v_stride,      \
            gb_stride, 0u);                                                   \
    }                                                                         \
    /* 2026-09-25: Pointer-table twin: this impl with state_is_table = 1. */  \
    extern "C" __global__ void gated_delta_rule_wy##K##_f16_table(            \
        __half* __restrict__ h_state,                                         \
        const __nv_bfloat16* __restrict__ query,                              \
        const __nv_bfloat16* __restrict__ key,                                \
        const __nv_bfloat16* __restrict__ value,                              \
        const float* __restrict__ gate,                                       \
        const float* __restrict__ beta,                                       \
        __nv_bfloat16* __restrict__ output,                                   \
        __half* __restrict__ h_state_inter_base,                              \
        unsigned int inter_stride_halves,                                      \
        unsigned int batch_size,                                              \
        unsigned int num_k_heads,                                             \
        unsigned int num_v_heads,                                             \
        unsigned int k_dim,                                                   \
        unsigned int v_dim,                                                   \
        unsigned int qk_stride,                                               \
        unsigned int v_stride,                                                \
        unsigned int gb_stride                                                \
    ) {                                                                       \
        gated_delta_rule_wyn_f16_impl<K>(                                     \
            h_state, query, key, value, gate, beta, output,                   \
            h_state_inter_base, inter_stride_halves, batch_size,               \
            num_k_heads, num_v_heads, k_dim, v_dim, qk_stride, v_stride,      \
            gb_stride, 1u);                                                   \
    }

METRALE_WYN_F16_INSTANTIATE(5)
METRALE_WYN_F16_INSTANTIATE(6)
METRALE_WYN_F16_INSTANTIATE(7)
METRALE_WYN_F16_INSTANTIATE(8)
METRALE_WYN_F16_INSTANTIATE(9)
METRALE_WYN_F16_INSTANTIATE(10)
METRALE_WYN_F16_INSTANTIATE(11)
METRALE_WYN_F16_INSTANTIATE(12)
METRALE_WYN_F16_INSTANTIATE(13)
METRALE_WYN_F16_INSTANTIATE(14)
METRALE_WYN_F16_INSTANTIATE(15)
METRALE_WYN_F16_INSTANTIATE(16)

#undef METRALE_WYN_F16_INSTANTIATE
