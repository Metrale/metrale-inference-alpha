// SPDX-License-Identifier: MIT OR Apache-2.0

// 2026-09-25: Gated delta rule verify step for exactly K = 17 rows per sequence, in one launch
// (WY form): the 17 output rows, the state after each of rows 0..15, and the final state.
// trait_decode_batched_conv_gdn.rs dispatches it when num_tokens == 17, the handle resolved,
// METRALE_GDN_WY17 is not 0 and the h-state is FP32; ops::gdn_decode_wyn launches it with
// grid (num_v_heads, batch, 1), block (128, 1, 1). qwen3.6-27b compiles this file too
// ([sources] in its KERNEL.toml).
//
// Steps:
//   1. Load q[K], k[K] into shared memory.
//   2. K*(K-1)/2 = 136 dot products k_t . k_s (s < t) by block reduction.
//   3. Pass 1: read H once; hk[t] = H . k_t with the pre-update H.
//   4. WY correction, sequential over t: vn[t].
//   5. Pass 2: apply the K updates in one loop, writing Hi_t for t = 0..K-2 and the
//      final state to H.
//
// Shared memory at K = 17: sk, sq 2*17*128*4 = 17,408 B; kd_flat 136*4 = 544 B; sg, sbt
// 2*17*4 = 136 B; smem_warp 16 B; 18,104 B in all.
//
// Owner: gb10 kernels (qwen3.6-35b-a3b, and the targets that list this file in `[sources] use`).
// Invariants: k_dim <= 128 with k_dim % 4 == 0, and v_dim <= 128 (thread tid owns state
// column tid); a block writes only its own (b, vh) state, intermediates and output rows.





#include <cuda_bf16.h>
#include "../../common/gdn_reduce.cuh"
#define BLOCK_SIZE 128
#define K_TOKENS 17

// 2026-09-25: Hi_t, the state after row t for t = 0..15, is written at h_state_inter_base
// + t * inter_stride_floats plus the (b, vh) offset; the state after row 16 goes back
// to h_state.




extern "C" __global__ void gated_delta_rule_wy17(
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
    unsigned int gb_stride
) {
    const unsigned int vh = blockIdx.x;
    const unsigned int b = blockIdx.y;
    if (vh >= num_v_heads || b >= batch_size) return;

    const unsigned int tid = threadIdx.x;
    const unsigned int hr = num_v_heads / num_k_heads;
    const unsigned int kh = vh / hr;
    const unsigned int hv = k_dim * v_dim;

    float* H = h_state + ((b * num_v_heads + vh) * hv);


    float* Hi_base = h_state_inter_base + ((b * num_v_heads + vh) * hv);

    // 2026-09-25: q and k rows, the clamped gate and beta, into shared memory.
    __shared__ float sk[K_TOKENS][128];
    __shared__ float sq[K_TOKENS][128];
    __shared__ float sg[K_TOKENS];
    __shared__ float sbt[K_TOKENS];
    __shared__ float smem_warp[4];

    // 2026-09-25: Row t of batch entry b holds its q/k at (b*K + t) * qk_stride + kh*k_dim.
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
        // 2026-09-25: The gate clamp of gated_delta_rule_decode: [1e-6, 1 - 1e-6].
        float g_raw = gate[(b * K_TOKENS + tid) * gb_stride + vh];
        sg[tid] = fminf(fmaxf(g_raw, 1e-6f), 1.0f - 1e-6f);
        sbt[tid] = beta[(b * K_TOKENS + tid) * gb_stride + vh];
    }
    __syncthreads();

    // 2026-09-25: K*(K-1)/2 = 136 k-dot products by block reduction: kd[t][s] = k_t . k_s for
    // s < t, stored at kd_flat[t*(t-1)/2 + s].

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
        // 2026-09-25: v[t] for this thread's column, every t.
        float vi[K_TOKENS];
        #pragma unroll
        for (int t = 0; t < K_TOKENS; t++) {
            const __nv_bfloat16* v_t = value + (b * K_TOKENS + t) * v_stride + vh * v_dim;
            vi[t] = (float)v_t[tid];
        }

        // 2026-09-25: Pass 1: read H once; hk[t] = H . k_t for every t.
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

        // 2026-09-25: WY correction, sequential over t:
        // hk_corrected[t] = prod(g[0..t-1]) * hk[t]
        //                 + sum_{s<t} prod(g[s+1..t-1]) * kd[t][s] * vn[s]
        // vn[t]           = (v[t] - g[t] * hk_corrected[t]) * beta[t]


        float vn[K_TOKENS];

        // 2026-09-25: Row 0: no correction.
        vn[0] = (vi[0] - sg[0] * hk[0]) * sbt[0];


        for (int t = 1; t < K_TOKENS; t++) {


            float corrected = 0.0f;

            float lead_prod = 1.0f;
            for (int u = 0; u < t; u++) lead_prod *= sg[u];
            corrected = lead_prod * hk[t];

            for (int s = 0; s < t; s++) {
                float gprod = 1.0f;
                for (int u = s + 1; u < t; u++) gprod *= sg[u];
                corrected += gprod * kd_flat[t * (t - 1) / 2 + s] * vn[s];
            }
            vn[t] = (vi[t] - sg[t] * corrected) * sbt[t];
        }

        // 2026-09-25: Pass 2: H_t = g[t] * H_{t-1} + k[t] * vn[t]. Hi_t = H_t for t = 0..K-2;
        // H = H_{K-1}.

        float qd[K_TOKENS];
        #pragma unroll
        for (int t = 0; t < K_TOKENS; t++) qd[t] = 0.0f;

        #pragma unroll 4
        for (unsigned int j = 0; j < k_dim; j += 4) {
            float h0 = H[(j + 0) * v_dim + tid];
            float h1 = H[(j + 1) * v_dim + tid];
            float h2 = H[(j + 2) * v_dim + tid];
            float h3 = H[(j + 3) * v_dim + tid];

            // 2026-09-25: Apply the updates row by row, writing the intermediates.
            #pragma unroll
            for (int t = 0; t < K_TOKENS; t++) {
                h0 = sg[t] * h0 + sk[t][j + 0] * vn[t];
                h1 = sg[t] * h1 + sk[t][j + 1] * vn[t];
                h2 = sg[t] * h2 + sk[t][j + 2] * vn[t];
                h3 = sg[t] * h3 + sk[t][j + 3] * vn[t];
                if (t < K_TOKENS - 1) {

                    float* Hi_t = Hi_base + t * inter_stride_floats;
                    Hi_t[(j + 0) * v_dim + tid] = h0;
                    Hi_t[(j + 1) * v_dim + tid] = h1;
                    Hi_t[(j + 2) * v_dim + tid] = h2;
                    Hi_t[(j + 3) * v_dim + tid] = h3;
                } else {
                    // 2026-09-25: The final state goes to H.
                    H[(j + 0) * v_dim + tid] = h0;
                    H[(j + 1) * v_dim + tid] = h1;
                    H[(j + 2) * v_dim + tid] = h2;
                    H[(j + 3) * v_dim + tid] = h3;
                }
                qd[t] += h0 * sq[t][j + 0] + h1 * sq[t][j + 1]
                       + h2 * sq[t][j + 2] + h3 * sq[t][j + 3];
            }
        }

        // 2026-09-25: Outputs: K rows x v_dim.
        float s = rsqrtf((float)k_dim);
        #pragma unroll
        for (int t = 0; t < K_TOKENS; t++) {
            output[((b * K_TOKENS + t) * num_v_heads + vh) * v_dim + tid] =
                __float2bfloat16(qd[t] * s);
        }
    }
}
