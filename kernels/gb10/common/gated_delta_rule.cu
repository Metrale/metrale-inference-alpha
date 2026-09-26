// SPDX-License-Identifier: MIT OR Apache-2.0

// 2026-09-25: Gated delta rule (GDN) recurrence kernels: single-token decode variants, the
// fused conv + recurrence + gated-norm decode, two- and three-token steps that keep the
// intermediate states, and a sequential prefill.
//
// Owner: gb10 kernels.
// Invariants:
// - Per head, with g the gate input (a decay factor) and beta the write strength:
//     v' = (v - g * (h_{t-1}^T k)) * beta,   h_t = g * h_{t-1} + k (outer) v',
//     out = (h_t^T q) / sqrt(k_dim).
// - State h is FP32 [batch, num_v_heads, k_dim, v_dim] with v_dim contiguous; thread tid
//   owns column tid, so every state access is coalesced across the warp.
// - Value head vh reads key head vh / (num_v_heads / num_k_heads).
// - k_dim <= 128 and k_dim % 4 == 0 (the shared q/k arrays hold 128 floats, and the
//   loops step j by 4); v_dim <= blockDim.x.













#include <cuda_bf16.h>


#define BLOCK_SIZE 128

__device__ __forceinline__ void gdn_unpack_bf16x2(unsigned int packed, float& v0, float& v1) {
    v0 = __bfloat162float(__ushort_as_bfloat16((unsigned short)(packed & 0xFFFF)));
    v1 = __bfloat162float(__ushort_as_bfloat16((unsigned short)(packed >> 16)));
}

__device__ __forceinline__ unsigned int gdn_pack_bf16x2(float v0, float v1) {
    unsigned int lo = (unsigned int)__bfloat16_as_ushort(__float2bfloat16(v0));
    unsigned int hi = (unsigned int)__bfloat16_as_ushort(__float2bfloat16(v1));
    return lo | (hi << 16);
}

__device__ __forceinline__ float gdn_warp_reduce_sum(float val) {
    for (int offset = 16; offset > 0; offset >>= 1) {
        val += __shfl_xor_sync(0xFFFFFFFF, val, offset);
    }
    return val;
}

// 2026-09-25: State-norm clamp, applied by every single-token decode kernel below (not by
// chunk2, chunk3 or prefill): after the update, a head whose state Frobenius norm exceeds
// SSM_STATE_MAX_NORM is scaled down to that norm. The output of that step uses the state
// before the clamp.




#ifndef SSM_STATE_NORM_ENABLED
#define SSM_STATE_NORM_ENABLED
#define SSM_STATE_MAX_NORM 1000.0f
#endif

// 2026-09-25: Single-token decode, BF16 q/k/v and output, one block per (value head, batch).
// Launch: grid (num_v_heads, batch, 1), block (128, 1, 1).










extern "C" __global__ void gated_delta_rule_decode(

    float* __restrict__ h_state,

    const __nv_bfloat16* __restrict__ query,
    const __nv_bfloat16* __restrict__ key,
    const __nv_bfloat16* __restrict__ value,

    const float* __restrict__ gate,
    const float* __restrict__ beta,

    __nv_bfloat16* __restrict__ output,

    unsigned int batch_size,
    unsigned int num_k_heads,
    unsigned int num_v_heads,
    unsigned int k_dim,
    unsigned int v_dim
) {
    const unsigned int vh = blockIdx.x;
    const unsigned int b = blockIdx.y;
    if (vh >= num_v_heads || b >= batch_size) return;

    const unsigned int tid = threadIdx.x;


    const unsigned int head_repeat = num_v_heads / num_k_heads;
    const unsigned int kh = vh / head_repeat;


    float* H = h_state + ((b * num_v_heads + vh) * k_dim * v_dim);
    const __nv_bfloat16* q_ptr = query + (b * num_k_heads + kh) * k_dim;
    const __nv_bfloat16* k_ptr = key + (b * num_k_heads + kh) * k_dim;
    const __nv_bfloat16* v_ptr = value + (b * num_v_heads + vh) * v_dim;

    // 2026-09-25: The decay is clamped to [1e-6, 1 - 1e-6], so it can neither grow the state
    // nor flip its sign.


    float g_raw = gate[b * num_v_heads + vh];
    const float g = fminf(fmaxf(g_raw, 1e-6f), 1.0f - 1e-6f);
    const float bt = beta[b * num_v_heads + vh];


    __shared__ float smem_k[128];
    __shared__ float smem_q[128];


    if (tid < k_dim) {
        smem_k[tid] = (float)k_ptr[tid];
        smem_q[tid] = (float)q_ptr[tid];
    }
    __syncthreads();



    if (tid < v_dim) {
        float v_i = (float)v_ptr[tid];


        float hk_dot = 0.0f;
        #pragma unroll 4
        for (unsigned int j = 0; j < k_dim; j += 4) {
            float h0 = H[(j + 0) * v_dim + tid];
            float h1 = H[(j + 1) * v_dim + tid];
            float h2 = H[(j + 2) * v_dim + tid];
            float h3 = H[(j + 3) * v_dim + tid];
            hk_dot += h0 * smem_k[j] + h1 * smem_k[j + 1]
                    + h2 * smem_k[j + 2] + h3 * smem_k[j + 3];
        }

        // 2026-09-25: The decay is applied before the correction:
        // (g * H)^T k = g * hk_dot, so v' = (v - g * hk_dot) * beta.


        float v_new_i = (v_i - g * hk_dot) * bt;

        // 2026-09-25: State update and output dot product in one pass over H.

        float q_dot = 0.0f;
        #pragma unroll 4
        for (unsigned int j = 0; j < k_dim; j += 4) {
            float h0 = H[(j + 0) * v_dim + tid];
            float h1 = H[(j + 1) * v_dim + tid];
            float h2 = H[(j + 2) * v_dim + tid];
            float h3 = H[(j + 3) * v_dim + tid];
            h0 = g * h0 + smem_k[j]     * v_new_i;
            h1 = g * h1 + smem_k[j + 1] * v_new_i;
            h2 = g * h2 + smem_k[j + 2] * v_new_i;
            h3 = g * h3 + smem_k[j + 3] * v_new_i;
            H[(j + 0) * v_dim + tid] = h0;
            H[(j + 1) * v_dim + tid] = h1;
            H[(j + 2) * v_dim + tid] = h2;
            H[(j + 3) * v_dim + tid] = h3;
            q_dot += h0 * smem_q[j] + h1 * smem_q[j + 1]
                   + h2 * smem_q[j + 2] + h3 * smem_q[j + 3];
        }




        #ifdef SSM_STATE_NORM_ENABLED
        {

            float local_sq = 0.0f;
            for (unsigned int j = 0; j < k_dim; j++) {
                float hv = H[j * v_dim + tid];
                local_sq += hv * hv;
            }


            unsigned int mask = __activemask();
            float warp_sum = local_sq;
            warp_sum += __shfl_down_sync(mask, warp_sum, 16);
            warp_sum += __shfl_down_sync(mask, warp_sum, 8);
            warp_sum += __shfl_down_sync(mask, warp_sum, 4);
            warp_sum += __shfl_down_sync(mask, warp_sum, 2);
            warp_sum += __shfl_down_sync(mask, warp_sum, 1);

            __shared__ float norm_sums[4];
            unsigned int warp_id = tid / 32;
            unsigned int lane_id = tid % 32;
            if (lane_id == 0) norm_sums[warp_id] = warp_sum;
            __syncthreads();

            float head_norm_sq;
            if (tid < 4) {
                float s = norm_sums[tid];
                s += __shfl_down_sync(0xf, s, 2);
                s += __shfl_down_sync(0xf, s, 1);
                norm_sums[0] = s;
            }
            __syncthreads();
            head_norm_sq = norm_sums[0];

            if (head_norm_sq > SSM_STATE_MAX_NORM * SSM_STATE_MAX_NORM) {
                float scale = SSM_STATE_MAX_NORM * rsqrtf(head_norm_sq);
                for (unsigned int j = 0; j < k_dim; j++) {
                    H[j * v_dim + tid] *= scale;
                }
            }
        }
        #endif


        float inv_sqrt_d = rsqrtf((float)k_dim);
        output[(b * num_v_heads + vh) * v_dim + tid] = __float2bfloat16(q_dot * inv_sqrt_d);
    }
}

// 2026-09-25: gated_delta_rule_decode with FP32 q/k/v/output. The state-norm sum of squares
// is accumulated from the just-stored registers instead of a second read of H.



extern "C" __global__ void gated_delta_rule_decode_f32(
    float* __restrict__ h_state,
    const float* __restrict__ query,
    const float* __restrict__ key,
    const float* __restrict__ value,
    const float* __restrict__ gate,
    const float* __restrict__ beta,
    float* __restrict__ output,
    unsigned int batch_size,
    unsigned int num_k_heads,
    unsigned int num_v_heads,
    unsigned int k_dim,
    unsigned int v_dim
) {
    const unsigned int vh = blockIdx.x;
    const unsigned int b = blockIdx.y;
    if (vh >= num_v_heads || b >= batch_size) return;

    const unsigned int tid = threadIdx.x;
    if (tid >= v_dim) return;

    const unsigned int head_repeat = num_v_heads / num_k_heads;
    const unsigned int kh = vh / head_repeat;

    float* H = h_state + ((b * num_v_heads + vh) * k_dim * v_dim);
    const float* q_ptr = query + (b * num_k_heads + kh) * k_dim;
    const float* k_ptr = key + (b * num_k_heads + kh) * k_dim;
    const float* v_ptr = value + (b * num_v_heads + vh) * v_dim;

    float g_raw = gate[b * num_v_heads + vh];
    const float g = fminf(fmaxf(g_raw, 1e-6f), 1.0f - 1e-6f);
    const float bt = beta[b * num_v_heads + vh];

    __shared__ float smem_k[128];
    __shared__ float smem_q[128];

    if (tid < k_dim) {
        smem_k[tid] = k_ptr[tid];
        smem_q[tid] = q_ptr[tid];
    }
    __syncthreads();

    float v_i = v_ptr[tid];
    float hk_dot = 0.0f;
    #pragma unroll 4
    for (unsigned int j = 0; j < k_dim; j += 4) {
        float h0 = H[(j + 0) * v_dim + tid];
        float h1 = H[(j + 1) * v_dim + tid];
        float h2 = H[(j + 2) * v_dim + tid];
        float h3 = H[(j + 3) * v_dim + tid];
        hk_dot += h0 * smem_k[j] + h1 * smem_k[j+1] + h2 * smem_k[j+2] + h3 * smem_k[j+3];
    }

    float v_new_i = (v_i - g * hk_dot) * bt;

    float q_dot = 0.0f;
#ifdef SSM_STATE_NORM_ENABLED
    float norm_acc = 0.0f;
#endif
    #pragma unroll 4
    for (unsigned int j = 0; j < k_dim; j += 4) {
        float h0 = H[(j + 0) * v_dim + tid];
        float h1 = H[(j + 1) * v_dim + tid];
        float h2 = H[(j + 2) * v_dim + tid];
        float h3 = H[(j + 3) * v_dim + tid];
        h0 = g * h0 + smem_k[j]     * v_new_i;
        h1 = g * h1 + smem_k[j + 1] * v_new_i;
        h2 = g * h2 + smem_k[j + 2] * v_new_i;
        h3 = g * h3 + smem_k[j + 3] * v_new_i;
        H[(j + 0) * v_dim + tid] = h0;
        H[(j + 1) * v_dim + tid] = h1;
        H[(j + 2) * v_dim + tid] = h2;
        H[(j + 3) * v_dim + tid] = h3;
        q_dot += h0 * smem_q[j] + h1 * smem_q[j+1] + h2 * smem_q[j+2] + h3 * smem_q[j+3];
#ifdef SSM_STATE_NORM_ENABLED
        // 2026-09-25: Sum of squares of the stored state, in ascending j, for the clamp below.




        norm_acc += h0 * h0;
        norm_acc += h1 * h1;
        norm_acc += h2 * h2;
        norm_acc += h3 * h3;
#endif
    }

    #ifdef SSM_STATE_NORM_ENABLED
    {
        float local_sq = norm_acc;
        for (int offset = 16; offset >= 1; offset >>= 1)
            local_sq += __shfl_down_sync(0xFFFFFFFF, local_sq, offset);
        __shared__ float norm_sums[4];
        if (tid % 32 == 0) norm_sums[tid / 32] = local_sq;
        __syncthreads();
        if (tid == 0) {
            float total = 0.0f;
            for (int w = 0; w < 4; w++) total += norm_sums[w];
            norm_sums[0] = total;
        }
        __syncthreads();
        float head_norm_sq = norm_sums[0];
        if (head_norm_sq > SSM_STATE_MAX_NORM * SSM_STATE_MAX_NORM) {
            float scale = SSM_STATE_MAX_NORM * rsqrtf(head_norm_sq);
            for (unsigned int j = 0; j < k_dim; j++) {
                H[j * v_dim + tid] *= scale;
            }
        }
    }
    #endif

    float inv_sqrt_d = rsqrtf((float)k_dim);
    output[(b * num_v_heads + vh) * v_dim + tid] = q_dot * inv_sqrt_d;
}

// 2026-09-25: gated_delta_rule_decode_f32 followed, in the same block, by the gated RMS norm
// of its per-head output: out = x * rsqrt(mean(x^2) + eps) * w * silu(z), written as BF16
// without an FP32 row in global memory. z_gate and output are indexed by head only, not by
// batch row, so batch_size must be 1 (every host call site passes 1).


extern "C" __global__ void gated_delta_rule_decode_f32_norm(
    float* __restrict__ h_state,
    const float* __restrict__ query,
    const float* __restrict__ key,
    const float* __restrict__ value,
    const float* __restrict__ gate,
    const float* __restrict__ beta,
    const __nv_bfloat16* __restrict__ z_gate,
    const __nv_bfloat16* __restrict__ norm_weight,
    __nv_bfloat16* __restrict__ output,
    unsigned int batch_size,
    unsigned int num_k_heads,
    unsigned int num_v_heads,
    unsigned int k_dim,
    unsigned int v_dim,
    float eps
) {
    const unsigned int vh = blockIdx.x;
    const unsigned int b = blockIdx.y;
    if (vh >= num_v_heads || b >= batch_size) return;

    const unsigned int tid = threadIdx.x;
    if (tid >= v_dim) return;

    const unsigned int head_repeat = num_v_heads / num_k_heads;
    const unsigned int kh = vh / head_repeat;

    float* H = h_state + ((b * num_v_heads + vh) * k_dim * v_dim);
    const float* q_ptr = query + (b * num_k_heads + kh) * k_dim;
    const float* k_ptr = key + (b * num_k_heads + kh) * k_dim;
    const float* v_ptr = value + (b * num_v_heads + vh) * v_dim;

    float g_raw = gate[b * num_v_heads + vh];
    const float g = fminf(fmaxf(g_raw, 1e-6f), 1.0f - 1e-6f);
    const float bt = beta[b * num_v_heads + vh];

    __shared__ float smem_k[128];
    __shared__ float smem_q[128];

    if (tid < k_dim) {
        smem_k[tid] = k_ptr[tid];
        smem_q[tid] = q_ptr[tid];
    }
    __syncthreads();

    float v_i = v_ptr[tid];
    float hk_dot = 0.0f;
    #pragma unroll 4
    for (unsigned int j = 0; j < k_dim; j += 4) {
        float h0 = H[(j + 0) * v_dim + tid];
        float h1 = H[(j + 1) * v_dim + tid];
        float h2 = H[(j + 2) * v_dim + tid];
        float h3 = H[(j + 3) * v_dim + tid];
        hk_dot += h0 * smem_k[j] + h1 * smem_k[j+1] + h2 * smem_k[j+2] + h3 * smem_k[j+3];
    }

    float v_new_i = (v_i - g * hk_dot) * bt;

    float q_dot = 0.0f;
#ifdef SSM_STATE_NORM_ENABLED
    float norm_acc = 0.0f;
#endif
    #pragma unroll 4
    for (unsigned int j = 0; j < k_dim; j += 4) {
        float h0 = H[(j + 0) * v_dim + tid];
        float h1 = H[(j + 1) * v_dim + tid];
        float h2 = H[(j + 2) * v_dim + tid];
        float h3 = H[(j + 3) * v_dim + tid];
        h0 = g * h0 + smem_k[j]     * v_new_i;
        h1 = g * h1 + smem_k[j + 1] * v_new_i;
        h2 = g * h2 + smem_k[j + 2] * v_new_i;
        h3 = g * h3 + smem_k[j + 3] * v_new_i;
        H[(j + 0) * v_dim + tid] = h0;
        H[(j + 1) * v_dim + tid] = h1;
        H[(j + 2) * v_dim + tid] = h2;
        H[(j + 3) * v_dim + tid] = h3;
        q_dot += h0 * smem_q[j] + h1 * smem_q[j+1] + h2 * smem_q[j+2] + h3 * smem_q[j+3];
#ifdef SSM_STATE_NORM_ENABLED
        // 2026-09-25: Sum of squares of the stored state, in ascending j, for the clamp below.




        norm_acc += h0 * h0;
        norm_acc += h1 * h1;
        norm_acc += h2 * h2;
        norm_acc += h3 * h3;
#endif
    }

    #ifdef SSM_STATE_NORM_ENABLED
    {
        float local_sq = norm_acc;
        for (int offset = 16; offset >= 1; offset >>= 1)
            local_sq += __shfl_down_sync(0xFFFFFFFF, local_sq, offset);
        __shared__ float norm_sums[4];
        if (tid % 32 == 0) norm_sums[tid / 32] = local_sq;
        __syncthreads();
        if (tid == 0) {
            float total = 0.0f;
            for (int w = 0; w < 4; w++) total += norm_sums[w];
            norm_sums[0] = total;
        }
        __syncthreads();
        float head_norm_sq = norm_sums[0];
        if (head_norm_sq > SSM_STATE_MAX_NORM * SSM_STATE_MAX_NORM) {
            float scale = SSM_STATE_MAX_NORM * rsqrtf(head_norm_sq);
            for (unsigned int j = 0; j < k_dim; j++) {
                H[j * v_dim + tid] *= scale;
            }
        }
    }
    #endif

    const float inv_sqrt_d = rsqrtf((float)k_dim);
    const float x = q_dot * inv_sqrt_d;

    __shared__ float x_cache[128];
    x_cache[tid] = x;

    float sum_sq = x * x;
    sum_sq = gdn_warp_reduce_sum(sum_sq);
    __shared__ float rms_sums[4];
    const unsigned int warp_id = tid / 32;
    const unsigned int lane_id = tid % 32;
    if (lane_id == 0) rms_sums[warp_id] = sum_sq;
    __syncthreads();
    if (warp_id == 0) {
        float val = (lane_id < (blockDim.x + 31) / 32) ? rms_sums[lane_id] : 0.0f;
        val = gdn_warp_reduce_sum(val);
        if (lane_id == 0) rms_sums[0] = val;
    }
    __syncthreads();

    const float rms = rsqrtf(rms_sums[0] / (float)v_dim + eps);

    const unsigned int quad_size = v_dim / 4;
    const unsigned long long* g64 = (const unsigned long long*)(z_gate + vh * v_dim);
    const unsigned long long* w64 = (const unsigned long long*)norm_weight;
    unsigned long long* out64 = (unsigned long long*)(output + vh * v_dim);
    for (unsigned int i = tid; i < quad_size; i += blockDim.x) {
        unsigned int base = i * 4;
        float f0 = x_cache[base];
        float f1 = x_cache[base + 1];
        float f2 = x_cache[base + 2];
        float f3 = x_cache[base + 3];

        unsigned long long wv = w64[i];
        float w0, w1, w2, w3;
        gdn_unpack_bf16x2((unsigned int)wv, w0, w1);
        gdn_unpack_bf16x2((unsigned int)(wv >> 32), w2, w3);

        unsigned long long gv = g64[i];
        float g0, g1, g2, g3;
        gdn_unpack_bf16x2((unsigned int)gv, g0, g1);
        gdn_unpack_bf16x2((unsigned int)(gv >> 32), g2, g3);

        float s0 = g0 / (1.0f + expf(-g0));
        float s1 = g1 / (1.0f + expf(-g1));
        float s2 = g2 / (1.0f + expf(-g2));
        float s3 = g3 / (1.0f + expf(-g3));

        unsigned int lo = gdn_pack_bf16x2(f0 * rms * w0 * s0, f1 * rms * w1 * s1);
        unsigned int hi = gdn_pack_bf16x2(f2 * rms * w2 * s2, f3 * rms * w3 * s3);
        out64[i] = ((unsigned long long)hi << 32) | (unsigned long long)lo;
    }
}

// 2026-09-25: Fused decode: conv1d update + SiLU (+ L2 norm for q and k), the recurrence, and
// the gated RMS norm, in one launch. One block per (key head kh, batch) owns kh and its
// head_repeat value heads, so it is the only writer of their conv_state rows.
// Launch: grid (num_k_heads, batch, 1), block (head_repeat * v_dim, 1, 1); thread tid is
// value head kh * head_repeat + tid / v_dim, column tid % v_dim.
// Requires k_dim == v_dim == 128 and head_repeat == 2 (256 threads): the q/k groups are
// tid >> 7 and warp_sums holds eight warps. The Rust dispatch checks exactly these.
// new_input and conv_state rows are [Q (num_k_heads * k_dim) | K | V (num_v_heads * v_dim)].







extern "C" __global__ void gated_delta_rule_decode_f32_conv_norm(
    float* __restrict__ h_state,
    float* __restrict__ conv_state,
    const __nv_bfloat16* __restrict__ new_input,
    const __nv_bfloat16* __restrict__ conv_weight,
    const float* __restrict__ conv_bias,
    const float* __restrict__ gate,
    const float* __restrict__ beta,
    const __nv_bfloat16* __restrict__ z_gate,
    const __nv_bfloat16* __restrict__ norm_weight,
    __nv_bfloat16* __restrict__ output,
    unsigned int batch_size,
    unsigned int num_k_heads,
    unsigned int num_v_heads,
    unsigned int k_dim,
    unsigned int v_dim,
    unsigned int conv_dim,
    unsigned int d_conv,
    float l2_eps,
    float eps
) {
    const unsigned int kh = blockIdx.x;
    const unsigned int b  = blockIdx.y;
    if (kh >= num_k_heads || b >= batch_size) return;

    const unsigned int head_repeat = num_v_heads / num_k_heads;
    const unsigned int tid     = threadIdx.x;
    const unsigned int which_v = tid / v_dim;
    const unsigned int vlocal  = tid % v_dim;
    const unsigned int vh      = kh * head_repeat + which_v;
    const unsigned int key_dim_total = num_k_heads * k_dim;

    __shared__ float smem_q[128];
    __shared__ float smem_k[128];
    __shared__ float warp_sums[8];

    // 2026-09-25: Step 1: conv update + SiLU of this block's q/k channel; threads [0, k_dim) take
    // q head kh and [k_dim, 2 k_dim) take k head kh.
    float qk_silu = 0.0f;
    const bool is_q = (tid < k_dim);
    const bool is_k = (tid >= k_dim && tid < 2 * k_dim);
    if (is_q || is_k) {
        const unsigned int ch = is_q ? (kh * k_dim + tid)
                                     : (key_dim_total + kh * k_dim + (tid - k_dim));
        float* state = conv_state + ((unsigned long long)(b * conv_dim + ch)) * d_conv;
        for (unsigned int i = 0; i < d_conv - 1; i++) state[i] = state[i + 1];
        state[d_conv - 1] = (float)new_input[b * conv_dim + ch];
        const __nv_bfloat16* w = conv_weight + (unsigned long long)ch * d_conv;
        float acc = (conv_bias != nullptr) ? conv_bias[ch] : 0.0f;
        for (unsigned int k = 0; k < d_conv; k++) acc += state[k] * (float)w[k];
        qk_silu = acc / (1.0f + __expf(-acc));
    }
    // 2026-09-25: L2-normalize q (warps 0-3) and k (warps 4-7) separately.
    {
        float sq = qk_silu * qk_silu;
        for (int off = 16; off >= 1; off >>= 1) sq += __shfl_down_sync(0xFFFFFFFF, sq, off);
        if ((tid & 31) == 0) warp_sums[tid >> 5] = sq;
        __syncthreads();
        const unsigned int grp = tid >> 7;
        float total = warp_sums[grp * 4 + 0] + warp_sums[grp * 4 + 1]
                    + warp_sums[grp * 4 + 2] + warp_sums[grp * 4 + 3];
        float inv = rsqrtf(total + l2_eps);
        if (is_q) smem_q[tid] = qk_silu * inv;
        else if (is_k) smem_k[tid - k_dim] = qk_silu * inv;
    }

    // 2026-09-25: Step 1b: conv update + SiLU of this thread's V channel (no L2 norm).
    float v_i = 0.0f;
    {
        const unsigned int vch = 2 * key_dim_total + vh * v_dim + vlocal;
        float* state = conv_state + ((unsigned long long)(b * conv_dim + vch)) * d_conv;
        for (unsigned int i = 0; i < d_conv - 1; i++) state[i] = state[i + 1];
        state[d_conv - 1] = (float)new_input[b * conv_dim + vch];
        const __nv_bfloat16* w = conv_weight + (unsigned long long)vch * d_conv;
        float acc = (conv_bias != nullptr) ? conv_bias[vch] : 0.0f;
        for (unsigned int k = 0; k < d_conv; k++) acc += state[k] * (float)w[k];
        v_i = acc / (1.0f + __expf(-acc));
    }
    __syncthreads();   // 2026-09-25: smem_q / smem_k are complete before the recurrence reads them

    // 2026-09-25: Step 2: the recurrence for value head vh, column vlocal.
    float* H = h_state + ((unsigned long long)(b * num_v_heads + vh)) * k_dim * v_dim;
    const float g_raw = gate[b * num_v_heads + vh];
    const float g  = fminf(fmaxf(g_raw, 1e-6f), 1.0f - 1e-6f);
    const float bt = beta[b * num_v_heads + vh];

    float hk_dot = 0.0f;
    #pragma unroll 4
    for (unsigned int j = 0; j < k_dim; j += 4) {
        float h0 = H[(j + 0) * v_dim + vlocal];
        float h1 = H[(j + 1) * v_dim + vlocal];
        float h2 = H[(j + 2) * v_dim + vlocal];
        float h3 = H[(j + 3) * v_dim + vlocal];
        hk_dot += h0 * smem_k[j] + h1 * smem_k[j+1] + h2 * smem_k[j+2] + h3 * smem_k[j+3];
    }
    const float v_new = (v_i - g * hk_dot) * bt;

    float q_dot = 0.0f;
    #pragma unroll 4
    for (unsigned int j = 0; j < k_dim; j += 4) {
        float h0 = g * H[(j+0)*v_dim+vlocal] + smem_k[j]   * v_new;
        float h1 = g * H[(j+1)*v_dim+vlocal] + smem_k[j+1] * v_new;
        float h2 = g * H[(j+2)*v_dim+vlocal] + smem_k[j+2] * v_new;
        float h3 = g * H[(j+3)*v_dim+vlocal] + smem_k[j+3] * v_new;
        H[(j+0)*v_dim+vlocal] = h0;
        H[(j+1)*v_dim+vlocal] = h1;
        H[(j+2)*v_dim+vlocal] = h2;
        H[(j+3)*v_dim+vlocal] = h3;
        q_dot += h0 * smem_q[j] + h1 * smem_q[j+1] + h2 * smem_q[j+2] + h3 * smem_q[j+3];
    }

    // 2026-09-25: State-norm clamp, reduced within this value head's 128-thread group.
    #ifdef SSM_STATE_NORM_ENABLED
    {
        float local_sq = 0.0f;
        for (unsigned int j = 0; j < k_dim; j++) {
            float hv = H[j * v_dim + vlocal];
            local_sq += hv * hv;
        }
        for (int off = 16; off >= 1; off >>= 1)
            local_sq += __shfl_down_sync(0xFFFFFFFF, local_sq, off);
        __syncthreads();   // 2026-09-25: warp_sums is reused
        if ((tid & 31) == 0) warp_sums[tid >> 5] = local_sq;
        __syncthreads();
        const unsigned int grp = tid >> 7;
        float head_norm_sq = warp_sums[grp*4+0] + warp_sums[grp*4+1]
                           + warp_sums[grp*4+2] + warp_sums[grp*4+3];
        if (head_norm_sq > SSM_STATE_MAX_NORM * SSM_STATE_MAX_NORM) {
            float scale = SSM_STATE_MAX_NORM * rsqrtf(head_norm_sq);
            for (unsigned int j = 0; j < k_dim; j++) H[j * v_dim + vlocal] *= scale;
        }
    }
    #endif

    // 2026-09-25: Step 3: gated RMS norm per value head, then the BF16 output.
    const float x = q_dot * rsqrtf((float)k_dim);
    float sum_sq = x * x;
    for (int off = 16; off >= 1; off >>= 1) sum_sq += __shfl_down_sync(0xFFFFFFFF, sum_sq, off);
    __syncthreads();   // 2026-09-25: warp_sums is reused
    if ((tid & 31) == 0) warp_sums[tid >> 5] = sum_sq;
    __syncthreads();
    {
        const unsigned int grp = tid >> 7;
        float total = warp_sums[grp*4+0] + warp_sums[grp*4+1]
                    + warp_sums[grp*4+2] + warp_sums[grp*4+3];
        const float rms = rsqrtf(total / (float)v_dim + eps);
        const float zg = (float)z_gate[(unsigned long long)(b * num_v_heads + vh) * v_dim + vlocal];
        const float wv = (float)norm_weight[vlocal];
        const float sg = zg / (1.0f + expf(-zg));
        output[(unsigned long long)(b * num_v_heads + vh) * v_dim + vlocal]
            = __float2bfloat16(x * rms * wv * sg);
    }
}

// 2026-09-25: gated_delta_rule_decode_f32 for several sequences: q/k, v and gate/beta rows of
// batch row b start at b * qk_stride, b * v_stride and b * gb_stride, and the output row at
// b * out_stride, so the multi-sequence path needs no repacking.


extern "C" __global__ void gated_delta_rule_decode_f32_strided(
    float* __restrict__ h_state,
    const float* __restrict__ query,
    const float* __restrict__ key,
    const float* __restrict__ value,
    const float* __restrict__ gate,
    const float* __restrict__ beta,
    float* __restrict__ output,
    unsigned int batch_size,
    unsigned int num_k_heads,
    unsigned int num_v_heads,
    unsigned int k_dim,
    unsigned int v_dim,
    unsigned int qk_stride,
    unsigned int v_stride,
    unsigned int gb_stride,
    unsigned int out_stride
) {
    const unsigned int vh = blockIdx.x;
    const unsigned int b = blockIdx.y;
    if (vh >= num_v_heads || b >= batch_size) return;

    const unsigned int tid = threadIdx.x;
    if (tid >= v_dim) return;

    const unsigned int head_repeat = num_v_heads / num_k_heads;
    const unsigned int kh = vh / head_repeat;

    float* H = h_state + ((b * num_v_heads + vh) * k_dim * v_dim);
    const float* q_ptr = query + (unsigned long long)b * qk_stride + kh * k_dim;
    const float* k_ptr = key + (unsigned long long)b * qk_stride + kh * k_dim;
    const float* v_ptr = value + (unsigned long long)b * v_stride + vh * v_dim;

    float g_raw = gate[(unsigned long long)b * gb_stride + vh];
    const float g = fminf(fmaxf(g_raw, 1e-6f), 1.0f - 1e-6f);
    const float bt = beta[(unsigned long long)b * gb_stride + vh];

    __shared__ float smem_k[128];
    __shared__ float smem_q[128];

    if (tid < k_dim) {
        smem_k[tid] = k_ptr[tid];
        smem_q[tid] = q_ptr[tid];
    }
    __syncthreads();

    float v_i = v_ptr[tid];
    float hk_dot = 0.0f;
    #pragma unroll 4
    for (unsigned int j = 0; j < k_dim; j += 4) {
        float h0 = H[(j + 0) * v_dim + tid];
        float h1 = H[(j + 1) * v_dim + tid];
        float h2 = H[(j + 2) * v_dim + tid];
        float h3 = H[(j + 3) * v_dim + tid];
        hk_dot += h0 * smem_k[j] + h1 * smem_k[j+1] + h2 * smem_k[j+2] + h3 * smem_k[j+3];
    }

    float v_new_i = (v_i - g * hk_dot) * bt;

    float q_dot = 0.0f;
#ifdef SSM_STATE_NORM_ENABLED
    float norm_acc = 0.0f;
#endif
    #pragma unroll 4
    for (unsigned int j = 0; j < k_dim; j += 4) {
        float h0 = H[(j + 0) * v_dim + tid];
        float h1 = H[(j + 1) * v_dim + tid];
        float h2 = H[(j + 2) * v_dim + tid];
        float h3 = H[(j + 3) * v_dim + tid];
        h0 = g * h0 + smem_k[j]     * v_new_i;
        h1 = g * h1 + smem_k[j + 1] * v_new_i;
        h2 = g * h2 + smem_k[j + 2] * v_new_i;
        h3 = g * h3 + smem_k[j + 3] * v_new_i;
        H[(j + 0) * v_dim + tid] = h0;
        H[(j + 1) * v_dim + tid] = h1;
        H[(j + 2) * v_dim + tid] = h2;
        H[(j + 3) * v_dim + tid] = h3;
        q_dot += h0 * smem_q[j] + h1 * smem_q[j+1] + h2 * smem_q[j+2] + h3 * smem_q[j+3];
#ifdef SSM_STATE_NORM_ENABLED
        // 2026-09-25: Sum of squares of the stored state, in ascending j, for the clamp below.




        norm_acc += h0 * h0;
        norm_acc += h1 * h1;
        norm_acc += h2 * h2;
        norm_acc += h3 * h3;
#endif
    }

    #ifdef SSM_STATE_NORM_ENABLED
    {
        float local_sq = norm_acc;
        for (int offset = 16; offset >= 1; offset >>= 1)
            local_sq += __shfl_down_sync(0xFFFFFFFF, local_sq, offset);
        __shared__ float norm_sums[4];
        if (tid % 32 == 0) norm_sums[tid / 32] = local_sq;
        __syncthreads();
        if (tid == 0) {
            float total = 0.0f;
            for (int w = 0; w < 4; w++) total += norm_sums[w];
            norm_sums[0] = total;
        }
        __syncthreads();
        float head_norm_sq = norm_sums[0];
        if (head_norm_sq > SSM_STATE_MAX_NORM * SSM_STATE_MAX_NORM) {
            float scale = SSM_STATE_MAX_NORM * rsqrtf(head_norm_sq);
            for (unsigned int j = 0; j < k_dim; j++) {
                H[j * v_dim + tid] *= scale;
            }
        }
    }
    #endif

    float inv_sqrt_d = rsqrtf((float)k_dim);
    output[(unsigned long long)b * out_stride + vh * v_dim + tid] = q_dot * inv_sqrt_d;
}

// 2026-09-25: gated_delta_rule_decode_f32_strided followed by the gated RMS norm of
// gated_delta_rule_decode_f32_norm, writing BF16 at b * out_stride; z rows start at
// b * z_stride.



extern "C" __global__ void gated_delta_rule_decode_f32_strided_norm(
    float* __restrict__ h_state,
    const float* __restrict__ query,
    const float* __restrict__ key,
    const float* __restrict__ value,
    const float* __restrict__ gate,
    const float* __restrict__ beta,
    const __nv_bfloat16* __restrict__ z_gate,
    const __nv_bfloat16* __restrict__ norm_weight,
    __nv_bfloat16* __restrict__ output,
    unsigned int batch_size,
    unsigned int num_k_heads,
    unsigned int num_v_heads,
    unsigned int k_dim,
    unsigned int v_dim,
    unsigned int qk_stride,
    unsigned int v_stride,
    unsigned int gb_stride,
    unsigned int z_stride,
    unsigned int out_stride,
    float eps
) {
    const unsigned int vh = blockIdx.x;
    const unsigned int b = blockIdx.y;
    if (vh >= num_v_heads || b >= batch_size) return;

    const unsigned int tid = threadIdx.x;
    if (tid >= v_dim) return;

    const unsigned int head_repeat = num_v_heads / num_k_heads;
    const unsigned int kh = vh / head_repeat;

    float* H = h_state + ((b * num_v_heads + vh) * k_dim * v_dim);
    const float* q_ptr = query + (unsigned long long)b * qk_stride + kh * k_dim;
    const float* k_ptr = key + (unsigned long long)b * qk_stride + kh * k_dim;
    const float* v_ptr = value + (unsigned long long)b * v_stride + vh * v_dim;

    float g_raw = gate[(unsigned long long)b * gb_stride + vh];
    const float g = fminf(fmaxf(g_raw, 1e-6f), 1.0f - 1e-6f);
    const float bt = beta[(unsigned long long)b * gb_stride + vh];

    __shared__ float smem_k[128];
    __shared__ float smem_q[128];

    if (tid < k_dim) {
        smem_k[tid] = k_ptr[tid];
        smem_q[tid] = q_ptr[tid];
    }
    __syncthreads();

    float v_i = v_ptr[tid];
    float hk_dot = 0.0f;
    #pragma unroll 4
    for (unsigned int j = 0; j < k_dim; j += 4) {
        float h0 = H[(j + 0) * v_dim + tid];
        float h1 = H[(j + 1) * v_dim + tid];
        float h2 = H[(j + 2) * v_dim + tid];
        float h3 = H[(j + 3) * v_dim + tid];
        hk_dot += h0 * smem_k[j] + h1 * smem_k[j+1] + h2 * smem_k[j+2] + h3 * smem_k[j+3];
    }

    float v_new_i = (v_i - g * hk_dot) * bt;

    float q_dot = 0.0f;
#ifdef SSM_STATE_NORM_ENABLED
    float norm_acc = 0.0f;
#endif
    #pragma unroll 4
    for (unsigned int j = 0; j < k_dim; j += 4) {
        float h0 = H[(j + 0) * v_dim + tid];
        float h1 = H[(j + 1) * v_dim + tid];
        float h2 = H[(j + 2) * v_dim + tid];
        float h3 = H[(j + 3) * v_dim + tid];
        h0 = g * h0 + smem_k[j]     * v_new_i;
        h1 = g * h1 + smem_k[j + 1] * v_new_i;
        h2 = g * h2 + smem_k[j + 2] * v_new_i;
        h3 = g * h3 + smem_k[j + 3] * v_new_i;
        H[(j + 0) * v_dim + tid] = h0;
        H[(j + 1) * v_dim + tid] = h1;
        H[(j + 2) * v_dim + tid] = h2;
        H[(j + 3) * v_dim + tid] = h3;
        q_dot += h0 * smem_q[j] + h1 * smem_q[j+1] + h2 * smem_q[j+2] + h3 * smem_q[j+3];
#ifdef SSM_STATE_NORM_ENABLED
        // 2026-09-25: Sum of squares of the stored state, in ascending j, for the clamp below.




        norm_acc += h0 * h0;
        norm_acc += h1 * h1;
        norm_acc += h2 * h2;
        norm_acc += h3 * h3;
#endif
    }

    #ifdef SSM_STATE_NORM_ENABLED
    {
        float local_sq = norm_acc;
        for (int offset = 16; offset >= 1; offset >>= 1)
            local_sq += __shfl_down_sync(0xFFFFFFFF, local_sq, offset);
        __shared__ float norm_sums[4];
        if (tid % 32 == 0) norm_sums[tid / 32] = local_sq;
        __syncthreads();
        if (tid == 0) {
            float total = 0.0f;
            for (int w = 0; w < 4; w++) total += norm_sums[w];
            norm_sums[0] = total;
        }
        __syncthreads();
        float head_norm_sq = norm_sums[0];
        if (head_norm_sq > SSM_STATE_MAX_NORM * SSM_STATE_MAX_NORM) {
            float scale = SSM_STATE_MAX_NORM * rsqrtf(head_norm_sq);
            for (unsigned int j = 0; j < k_dim; j++) {
                H[j * v_dim + tid] *= scale;
            }
        }
    }
    #endif

    const float inv_sqrt_d = rsqrtf((float)k_dim);
    const float x = q_dot * inv_sqrt_d;

    __shared__ float x_cache[128];
    x_cache[tid] = x;

    float sum_sq = x * x;
    sum_sq = gdn_warp_reduce_sum(sum_sq);
    __shared__ float rms_sums[4];
    const unsigned int warp_id = tid / 32;
    const unsigned int lane_id = tid % 32;
    if (lane_id == 0) rms_sums[warp_id] = sum_sq;
    __syncthreads();
    if (warp_id == 0) {
        float val = (lane_id < (blockDim.x + 31) / 32) ? rms_sums[lane_id] : 0.0f;
        val = gdn_warp_reduce_sum(val);
        if (lane_id == 0) rms_sums[0] = val;
    }
    __syncthreads();

    const float rms = rsqrtf(rms_sums[0] / (float)v_dim + eps);

    const unsigned int quad_size = v_dim / 4;
    const unsigned long long* g64 = (const unsigned long long*)(
        z_gate + (unsigned long long)b * z_stride + vh * v_dim
    );
    const unsigned long long* w64 = (const unsigned long long*)norm_weight;
    unsigned long long* out64 = (unsigned long long*)(
        output + (unsigned long long)b * out_stride + vh * v_dim
    );
    for (unsigned int i = tid; i < quad_size; i += blockDim.x) {
        unsigned int base = i * 4;
        float f0 = x_cache[base];
        float f1 = x_cache[base + 1];
        float f2 = x_cache[base + 2];
        float f3 = x_cache[base + 3];

        unsigned long long wv = w64[i];
        float w0, w1, w2, w3;
        gdn_unpack_bf16x2((unsigned int)wv, w0, w1);
        gdn_unpack_bf16x2((unsigned int)(wv >> 32), w2, w3);

        unsigned long long gv = g64[i];
        float g0, g1, g2, g3;
        gdn_unpack_bf16x2((unsigned int)gv, g0, g1);
        gdn_unpack_bf16x2((unsigned int)(gv >> 32), g2, g3);

        float s0 = g0 / (1.0f + expf(-g0));
        float s1 = g1 / (1.0f + expf(-g1));
        float s2 = g2 / (1.0f + expf(-g2));
        float s3 = g3 / (1.0f + expf(-g3));

        unsigned int lo = gdn_pack_bf16x2(f0 * rms * w0 * s0, f1 * rms * w1 * s1);
        unsigned int hi = gdn_pack_bf16x2(f2 * rms * w2 * s2, f3 * rms * w3 * s3);
        out64[i] = ((unsigned long long)hi << 32) | (unsigned long long)lo;
    }
}

// 2026-09-25: Two-token step. Runs tokens 0 and 1 through the recurrence in one launch, stores
// the intermediate state H_1 in h_state_intermediate and the final H_2 in h_state, so a
// caller can roll back to H_1. Token t of batch row b reads its q/k/v/gate/beta row
// (b * 2 + t) at the given strides; output is [batch, 2, num_v_heads, v_dim]. Unlike the
// decode kernels it applies neither the decay clamp nor the state-norm clamp.
// Launch: grid (num_v_heads, batch, 1); v_dim <= blockDim.x.







extern "C" __global__ void gated_delta_rule_chunk2(

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
    unsigned int gb_stride
) {
    const unsigned int vh = blockIdx.x;
    const unsigned int b = blockIdx.y;
    if (vh >= num_v_heads || b >= batch_size) return;

    const unsigned int tid = threadIdx.x;
    const unsigned int head_repeat = num_v_heads / num_k_heads;
    const unsigned int kh = vh / head_repeat;


    const unsigned int hv_size = k_dim * v_dim;
    float* H = h_state + ((b * num_v_heads + vh) * hv_size);
    float* H_inter = h_state_intermediate + ((b * num_v_heads + vh) * hv_size);





    const __nv_bfloat16* q0 = query + (b * 2) * qk_stride + kh * k_dim;
    const __nv_bfloat16* k0 = key   + (b * 2) * qk_stride + kh * k_dim;
    const __nv_bfloat16* v0 = value + (b * 2) * v_stride  + vh * v_dim;
    const float g0 = gate[(b * 2) * gb_stride + vh];
    const float bt0 = beta[(b * 2) * gb_stride + vh];


    const __nv_bfloat16* q1 = query + (b * 2 + 1) * qk_stride + kh * k_dim;
    const __nv_bfloat16* k1 = key   + (b * 2 + 1) * qk_stride + kh * k_dim;
    const __nv_bfloat16* v1 = value + (b * 2 + 1) * v_stride  + vh * v_dim;
    const float g1 = gate[(b * 2 + 1) * gb_stride + vh];
    const float bt1 = beta[(b * 2 + 1) * gb_stride + vh];


    __shared__ float smem_k0[128];
    __shared__ float smem_q0[128];
    __shared__ float smem_k1[128];
    __shared__ float smem_q1[128];

    if (tid < k_dim) {
        smem_k0[tid] = (float)k0[tid];
        smem_q0[tid] = (float)q0[tid];
        smem_k1[tid] = (float)k1[tid];
        smem_q1[tid] = (float)q1[tid];
    }
    __syncthreads();

    if (tid < v_dim) {
        float vi0 = (float)v0[tid];
        float vi1 = (float)v1[tid];

        // 2026-09-25: Pass 1: hk0 = H_0^T k_0.
        float hk0 = 0.0f;
        #pragma unroll 4
        for (unsigned int j = 0; j < k_dim; j += 4) {
            float h0 = H[(j + 0) * v_dim + tid];
            float h1 = H[(j + 1) * v_dim + tid];
            float h2 = H[(j + 2) * v_dim + tid];
            float h3 = H[(j + 3) * v_dim + tid];
            hk0 += h0 * smem_k0[j] + h1 * smem_k0[j + 1]
                 + h2 * smem_k0[j + 2] + h3 * smem_k0[j + 3];
        }


        float v_new_0 = (vi0 - g0 * hk0) * bt0;

        // 2026-09-25: Pass 2: H_0 -> H_1 (stored in H_inter), out_0, and hk1 on H_1.
        float q0_dot = 0.0f;
        float hk1 = 0.0f;
        #pragma unroll 4
        for (unsigned int j = 0; j < k_dim; j += 4) {
            float h0 = H[(j + 0) * v_dim + tid];
            float h1 = H[(j + 1) * v_dim + tid];
            float h2 = H[(j + 2) * v_dim + tid];
            float h3 = H[(j + 3) * v_dim + tid];


            h0 = g0 * h0 + smem_k0[j]     * v_new_0;
            h1 = g0 * h1 + smem_k0[j + 1] * v_new_0;
            h2 = g0 * h2 + smem_k0[j + 2] * v_new_0;
            h3 = g0 * h3 + smem_k0[j + 3] * v_new_0;


            H_inter[(j + 0) * v_dim + tid] = h0;
            H_inter[(j + 1) * v_dim + tid] = h1;
            H_inter[(j + 2) * v_dim + tid] = h2;
            H_inter[(j + 3) * v_dim + tid] = h3;


            q0_dot += h0 * smem_q0[j] + h1 * smem_q0[j + 1]
                    + h2 * smem_q0[j + 2] + h3 * smem_q0[j + 3];


            hk1 += h0 * smem_k1[j] + h1 * smem_k1[j + 1]
                 + h2 * smem_k1[j + 2] + h3 * smem_k1[j + 3];
        }


        float v_new_1 = (vi1 - g1 * hk1) * bt1;

        // 2026-09-25: Pass 3: H_1 -> H_2 (stored in H), out_1.
        float q1_dot = 0.0f;
        #pragma unroll 4
        for (unsigned int j = 0; j < k_dim; j += 4) {
            float h0 = H_inter[(j + 0) * v_dim + tid];
            float h1 = H_inter[(j + 1) * v_dim + tid];
            float h2 = H_inter[(j + 2) * v_dim + tid];
            float h3 = H_inter[(j + 3) * v_dim + tid];


            h0 = g1 * h0 + smem_k1[j]     * v_new_1;
            h1 = g1 * h1 + smem_k1[j + 1] * v_new_1;
            h2 = g1 * h2 + smem_k1[j + 2] * v_new_1;
            h3 = g1 * h3 + smem_k1[j + 3] * v_new_1;


            H[(j + 0) * v_dim + tid] = h0;
            H[(j + 1) * v_dim + tid] = h1;
            H[(j + 2) * v_dim + tid] = h2;
            H[(j + 3) * v_dim + tid] = h3;


            q1_dot += h0 * smem_q1[j] + h1 * smem_q1[j + 1]
                    + h2 * smem_q1[j + 2] + h3 * smem_q1[j + 3];
        }


        float inv_sqrt_d = rsqrtf((float)k_dim);
        unsigned int out_base0 = (b * 2 * num_v_heads + vh) * v_dim;
        unsigned int out_base1 = ((b * 2 + 1) * num_v_heads + vh) * v_dim;
        output[out_base0 + tid] = __float2bfloat16(q0_dot * inv_sqrt_d);
        output[out_base1 + tid] = __float2bfloat16(q1_dot * inv_sqrt_d);
    }
}

// 2026-09-25: Prefill: one block per (value head, batch) walks all seq_len tokens in order,
// with the head's state held in dynamic shared memory. Token t reads q/k at t * qk_stride,
// v at t * v_stride and gate/beta at t * gb_stride (not offset by batch row); the output
// is [batch, seq_len, num_v_heads, v_dim]. No decay clamp and no state-norm clamp.
// Dynamic shared memory must hold k_dim * v_dim + 2 * k_dim floats (H, then k and q).










extern "C" __global__ void gated_delta_rule_prefill(

    float* __restrict__ h_state,

    const __nv_bfloat16* __restrict__ query,
    const __nv_bfloat16* __restrict__ key,
    const __nv_bfloat16* __restrict__ value,

    const float* __restrict__ gate,
    const float* __restrict__ beta,

    __nv_bfloat16* __restrict__ output,

    unsigned int batch_size,
    unsigned int seq_len,
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
    const unsigned int head_repeat = num_v_heads / num_k_heads;
    const unsigned int kh = vh / head_repeat;



    extern __shared__ float H_smem[];


    float* H_global = h_state + ((b * num_v_heads + vh) * k_dim * v_dim);



    for (unsigned int i = tid; i < k_dim * v_dim; i += blockDim.x) {
        H_smem[i] = H_global[i];
    }



    float* smem_k = H_smem + k_dim * v_dim;
    float* smem_q = smem_k + k_dim;

    __syncthreads();

    float inv_sqrt_d = rsqrtf((float)k_dim);


    for (unsigned int t = 0; t < seq_len; t++) {
        const __nv_bfloat16* q_t = query + (unsigned long long)t * qk_stride + kh * k_dim;
        const __nv_bfloat16* k_t = key   + (unsigned long long)t * qk_stride + kh * k_dim;
        const __nv_bfloat16* v_t = value + (unsigned long long)t * v_stride  + vh * v_dim;

        float g_t = gate[(unsigned long long)t * gb_stride + vh];
        float bt = beta[(unsigned long long)t * gb_stride + vh];

        if (tid < k_dim) {
            smem_k[tid] = (float)k_t[tid];
            smem_q[tid] = (float)q_t[tid];
        }
        __syncthreads();

        if (tid < v_dim) {
            float v_i = (float)v_t[tid];

            float hk_dot = 0.0f;
            #pragma unroll 4
            for (unsigned int j = 0; j < k_dim; j += 4) {
                hk_dot += H_smem[(j + 0) * v_dim + tid] * smem_k[j]
                        + H_smem[(j + 1) * v_dim + tid] * smem_k[j + 1]
                        + H_smem[(j + 2) * v_dim + tid] * smem_k[j + 2]
                        + H_smem[(j + 3) * v_dim + tid] * smem_k[j + 3];
            }

            float v_new_i = (v_i - g_t * hk_dot) * bt;

            float q_dot = 0.0f;
            #pragma unroll 4
            for (unsigned int j = 0; j < k_dim; j += 4) {
                float h0 = g_t * H_smem[(j + 0) * v_dim + tid] + smem_k[j]     * v_new_i;
                float h1 = g_t * H_smem[(j + 1) * v_dim + tid] + smem_k[j + 1] * v_new_i;
                float h2 = g_t * H_smem[(j + 2) * v_dim + tid] + smem_k[j + 2] * v_new_i;
                float h3 = g_t * H_smem[(j + 3) * v_dim + tid] + smem_k[j + 3] * v_new_i;
                H_smem[(j + 0) * v_dim + tid] = h0;
                H_smem[(j + 1) * v_dim + tid] = h1;
                H_smem[(j + 2) * v_dim + tid] = h2;
                H_smem[(j + 3) * v_dim + tid] = h3;
                q_dot += h0 * smem_q[j] + h1 * smem_q[j + 1]
                       + h2 * smem_q[j + 2] + h3 * smem_q[j + 3];
            }

            output[((b * seq_len + t) * num_v_heads + vh) * v_dim + tid] =
                __float2bfloat16(q_dot * inv_sqrt_d);
        }
        __syncthreads();
    }


    for (unsigned int i = tid; i < k_dim * v_dim; i += blockDim.x) {
        H_global[i] = H_smem[i];
    }
}
// 2026-09-25: Three-token step: like gated_delta_rule_chunk2, storing H_1 and H_2 in
// h_state_inter0 / h_state_inter1 and H_3 in h_state. Token t of batch row b reads row
// (b * 3 + t); output is [batch, 3, num_v_heads, v_dim]. No decay or state-norm clamp.

extern "C" __global__ void gated_delta_rule_chunk3(

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
    unsigned int gb_stride
) {
    const unsigned int vh = blockIdx.x;
    const unsigned int b = blockIdx.y;
    if (vh >= num_v_heads || b >= batch_size) return;

    const unsigned int tid = threadIdx.x;
    const unsigned int head_repeat = num_v_heads / num_k_heads;
    const unsigned int kh = vh / head_repeat;

    const unsigned int hv_size = k_dim * v_dim;
    float* H = h_state + ((b * num_v_heads + vh) * hv_size);
    float* Hi0 = h_state_inter0 + ((b * num_v_heads + vh) * hv_size);
    float* Hi1 = h_state_inter1 + ((b * num_v_heads + vh) * hv_size);


    const __nv_bfloat16* q0 = query + (b * 3) * qk_stride + kh * k_dim;
    const __nv_bfloat16* k0 = key   + (b * 3) * qk_stride + kh * k_dim;
    const __nv_bfloat16* v0 = value + (b * 3) * v_stride  + vh * v_dim;
    const float g0 = gate[(b * 3) * gb_stride + vh];
    const float bt0 = beta[(b * 3) * gb_stride + vh];


    const __nv_bfloat16* q1 = query + (b * 3 + 1) * qk_stride + kh * k_dim;
    const __nv_bfloat16* k1 = key   + (b * 3 + 1) * qk_stride + kh * k_dim;
    const __nv_bfloat16* v1 = value + (b * 3 + 1) * v_stride  + vh * v_dim;
    const float g1 = gate[(b * 3 + 1) * gb_stride + vh];
    const float bt1 = beta[(b * 3 + 1) * gb_stride + vh];


    const __nv_bfloat16* q2 = query + (b * 3 + 2) * qk_stride + kh * k_dim;
    const __nv_bfloat16* k2 = key   + (b * 3 + 2) * qk_stride + kh * k_dim;
    const __nv_bfloat16* v2 = value + (b * 3 + 2) * v_stride  + vh * v_dim;
    const float g2 = gate[(b * 3 + 2) * gb_stride + vh];
    const float bt2 = beta[(b * 3 + 2) * gb_stride + vh];


    __shared__ float smem_k0[128];
    __shared__ float smem_q0[128];
    __shared__ float smem_k1[128];
    __shared__ float smem_q1[128];
    __shared__ float smem_k2[128];
    __shared__ float smem_q2[128];

    if (tid < k_dim) {
        smem_k0[tid] = (float)k0[tid]; smem_q0[tid] = (float)q0[tid];
        smem_k1[tid] = (float)k1[tid]; smem_q1[tid] = (float)q1[tid];
        smem_k2[tid] = (float)k2[tid]; smem_q2[tid] = (float)q2[tid];
    }
    __syncthreads();

    if (tid < v_dim) {
        float vi0 = (float)v0[tid];
        float vi1 = (float)v1[tid];
        float vi2 = (float)v2[tid];

        // 2026-09-25: Pass 1: hk0 = H_0^T k_0.
        float hk0 = 0.0f;
        #pragma unroll 4
        for (unsigned int j = 0; j < k_dim; j += 4) {
            float h0 = H[(j + 0) * v_dim + tid];
            float h1 = H[(j + 1) * v_dim + tid];
            float h2 = H[(j + 2) * v_dim + tid];
            float h3 = H[(j + 3) * v_dim + tid];
            hk0 += h0 * smem_k0[j] + h1 * smem_k0[j + 1]
                 + h2 * smem_k0[j + 2] + h3 * smem_k0[j + 3];
        }
        float v_new_0 = (vi0 - g0 * hk0) * bt0;

        // 2026-09-25: Pass 2: H_0 -> H_1 (stored in Hi0), out_0, and hk1 on H_1.
        float q0_dot = 0.0f;
        float hk1 = 0.0f;
        #pragma unroll 4
        for (unsigned int j = 0; j < k_dim; j += 4) {
            float h0 = H[(j + 0) * v_dim + tid];
            float h1 = H[(j + 1) * v_dim + tid];
            float h2 = H[(j + 2) * v_dim + tid];
            float h3 = H[(j + 3) * v_dim + tid];
            h0 = g0 * h0 + smem_k0[j]     * v_new_0;
            h1 = g0 * h1 + smem_k0[j + 1] * v_new_0;
            h2 = g0 * h2 + smem_k0[j + 2] * v_new_0;
            h3 = g0 * h3 + smem_k0[j + 3] * v_new_0;
            Hi0[(j + 0) * v_dim + tid] = h0;
            Hi0[(j + 1) * v_dim + tid] = h1;
            Hi0[(j + 2) * v_dim + tid] = h2;
            Hi0[(j + 3) * v_dim + tid] = h3;
            q0_dot += h0 * smem_q0[j] + h1 * smem_q0[j + 1]
                    + h2 * smem_q0[j + 2] + h3 * smem_q0[j + 3];
            hk1 += h0 * smem_k1[j] + h1 * smem_k1[j + 1]
                 + h2 * smem_k1[j + 2] + h3 * smem_k1[j + 3];
        }
        float v_new_1 = (vi1 - g1 * hk1) * bt1;

        // 2026-09-25: Pass 3: H_1 -> H_2 (stored in Hi1), out_1, and hk2 on H_2.
        float q1_dot = 0.0f;
        float hk2 = 0.0f;
        #pragma unroll 4
        for (unsigned int j = 0; j < k_dim; j += 4) {
            float h0 = Hi0[(j + 0) * v_dim + tid];
            float h1 = Hi0[(j + 1) * v_dim + tid];
            float h2 = Hi0[(j + 2) * v_dim + tid];
            float h3 = Hi0[(j + 3) * v_dim + tid];
            h0 = g1 * h0 + smem_k1[j]     * v_new_1;
            h1 = g1 * h1 + smem_k1[j + 1] * v_new_1;
            h2 = g1 * h2 + smem_k1[j + 2] * v_new_1;
            h3 = g1 * h3 + smem_k1[j + 3] * v_new_1;
            Hi1[(j + 0) * v_dim + tid] = h0;
            Hi1[(j + 1) * v_dim + tid] = h1;
            Hi1[(j + 2) * v_dim + tid] = h2;
            Hi1[(j + 3) * v_dim + tid] = h3;
            q1_dot += h0 * smem_q1[j] + h1 * smem_q1[j + 1]
                    + h2 * smem_q1[j + 2] + h3 * smem_q1[j + 3];
            hk2 += h0 * smem_k2[j] + h1 * smem_k2[j + 1]
                 + h2 * smem_k2[j + 2] + h3 * smem_k2[j + 3];
        }
        float v_new_2 = (vi2 - g2 * hk2) * bt2;

        // 2026-09-25: Pass 4: H_2 -> H_3 (stored in H), out_2.
        float q2_dot = 0.0f;
        #pragma unroll 4
        for (unsigned int j = 0; j < k_dim; j += 4) {
            float h0 = Hi1[(j + 0) * v_dim + tid];
            float h1 = Hi1[(j + 1) * v_dim + tid];
            float h2 = Hi1[(j + 2) * v_dim + tid];
            float h3 = Hi1[(j + 3) * v_dim + tid];
            h0 = g2 * h0 + smem_k2[j]     * v_new_2;
            h1 = g2 * h1 + smem_k2[j + 1] * v_new_2;
            h2 = g2 * h2 + smem_k2[j + 2] * v_new_2;
            h3 = g2 * h3 + smem_k2[j + 3] * v_new_2;
            H[(j + 0) * v_dim + tid] = h0;
            H[(j + 1) * v_dim + tid] = h1;
            H[(j + 2) * v_dim + tid] = h2;
            H[(j + 3) * v_dim + tid] = h3;
            q2_dot += h0 * smem_q2[j] + h1 * smem_q2[j + 1]
                    + h2 * smem_q2[j + 2] + h3 * smem_q2[j + 3];
        }


        float inv_sqrt_d = rsqrtf((float)k_dim);
        unsigned int out_base0 = (b * 3 * num_v_heads + vh) * v_dim;
        unsigned int out_base1 = ((b * 3 + 1) * num_v_heads + vh) * v_dim;
        unsigned int out_base2 = ((b * 3 + 2) * num_v_heads + vh) * v_dim;
        output[out_base0 + tid] = __float2bfloat16(q0_dot * inv_sqrt_d);
        output[out_base1 + tid] = __float2bfloat16(q1_dot * inv_sqrt_d);
        output[out_base2 + tid] = __float2bfloat16(q2_dot * inv_sqrt_d);
    }
}
