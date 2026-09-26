// SPDX-License-Identifier: MIT OR Apache-2.0

// 2026-09-25: Gated delta rule (GDN) kernels for qwen3.6-35b-a3b, in place of
// common/gated_delta_rule.cu ([shadow] in KERNEL.toml): sequential prefill (one, two
// or four CTAs per value head, and a batched four-CTA form), single-token decode (BF16
// or FP32 output, fused gated RMS norm, strided rows, fused conv), and two- and
// three-token steps (chunk2, chunk3).
// forked-from: kernels/gb10/common/gated_delta_rule.cu (2026-09-24; 2339 of 1713 lines differ, see kernels/FORKS.md)
//
// Owner: gb10 kernels (qwen3.6-35b-a3b, and the targets that list this file in `[sources] use`).
// Invariants:
// - Per head, with the gate g clamped to [1e-6, 1 - 1e-6]:
//     v' = (v - g * (h^T k)) * beta,   h' = g * h + k (outer) v',   out = (h'^T q) / sqrt(k_dim).
// - State h is FP32 [k_dim, v_dim] per (sequence, value head); value head vh reads key
//   head vh / (num_v_heads / num_k_heads).
// - The register-tiled kernels keep one state column per thread in H_reg[K_DIM] and
//   assume k_dim == K_DIM == 128.
// - gated_delta_rule_prefill, _split and _split4 read q, k, v, gate and beta at row t for every b.

#include <cuda_bf16.h>

#define K_DIM 128

// 2026-09-25: State-norm clamp, the same definition as common/gated_delta_rule.cu. Here only
// gated_delta_rule_decode_f32_strided_norm and gated_delta_rule_decode_f32_conv_norm
// apply it: after the update, a head whose state Frobenius norm exceeds
// SSM_STATE_MAX_NORM is scaled down to that norm. Their output uses the state before
// the clamp.







#ifndef SSM_STATE_NORM_ENABLED
#define SSM_STATE_NORM_ENABLED
#define SSM_STATE_MAX_NORM 1000.0f
#endif

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

extern "C" __global__ void __launch_bounds__(128, 1)
gated_delta_rule_prefill(
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

    // 2026-09-25: Double-buffered k and q: 4 * K_DIM floats (ops::gdn_prefill passes 4 * k_dim * 4 B).
    extern __shared__ float smem[];
    float* smem_k0 = smem;
    float* smem_q0 = smem + K_DIM;
    float* smem_k1 = smem + 2 * K_DIM;
    float* smem_q1 = smem + 3 * K_DIM;

    float* H_global = h_state + ((unsigned long long)(b * num_v_heads + vh) * K_DIM * v_dim);

    // 2026-09-25: Thread tid owns state column tid and keeps it in H_reg for the whole sequence.
    float H_reg[K_DIM];
    #pragma unroll
    for (int j = 0; j < K_DIM; j++) {
        H_reg[j] = H_global[j * v_dim + tid];
    }

    float inv_sqrt_d = rsqrtf((float)k_dim);

    if (seq_len == 0) goto store_h;


    {
        unsigned long long qk_off = (unsigned long long)0 * qk_stride + kh * k_dim;
        smem_k0[tid] = (float)key[qk_off + tid];
        smem_q0[tid] = (float)query[qk_off + tid];
    }
    __syncthreads();


    for (unsigned int t = 0; t < seq_len; t++) {

        float* cur_k = (t & 1) ? smem_k1 : smem_k0;
        float* cur_q = (t & 1) ? smem_q1 : smem_q0;
        float* nxt_k = (t & 1) ? smem_k0 : smem_k1;
        float* nxt_q = (t & 1) ? smem_q0 : smem_q1;


        if (t + 1 < seq_len) {
            unsigned long long qk_off_nxt = (unsigned long long)(t + 1) * qk_stride + kh * k_dim;
            nxt_k[tid] = (float)key[qk_off_nxt + tid];
            nxt_q[tid] = (float)query[qk_off_nxt + tid];
        }

        float v_i = (float)value[(unsigned long long)t * v_stride + vh * v_dim + tid];
        float g_t = fminf(fmaxf(gate[(unsigned long long)t * gb_stride + vh], 1e-6f), 1.0f - 1e-6f);
        float bt_t = beta[(unsigned long long)t * gb_stride + vh];

        // 2026-09-25: Pass 1: hk_dot = h^T k, in four independent accumulators.

        float hk0 = 0.0f, hk1 = 0.0f, hk2 = 0.0f, hk3 = 0.0f;
        #pragma unroll
        for (int j = 0; j < K_DIM; j += 4) {
            hk0 += H_reg[j]     * cur_k[j];
            hk1 += H_reg[j + 1] * cur_k[j + 1];
            hk2 += H_reg[j + 2] * cur_k[j + 2];
            hk3 += H_reg[j + 3] * cur_k[j + 3];
        }
        float hk_dot = (hk0 + hk1) + (hk2 + hk3);

        float v_new = (v_i - g_t * hk_dot) * bt_t;

        // 2026-09-25: Pass 2: update h in registers and accumulate q_dot = h'^T q.

        float qd0 = 0.0f, qd1 = 0.0f, qd2 = 0.0f, qd3 = 0.0f;
        #pragma unroll
        for (int j = 0; j < K_DIM; j += 4) {
            float h0 = g_t * H_reg[j]     + cur_k[j]     * v_new;
            float h1 = g_t * H_reg[j + 1] + cur_k[j + 1] * v_new;
            float h2 = g_t * H_reg[j + 2] + cur_k[j + 2] * v_new;
            float h3 = g_t * H_reg[j + 3] + cur_k[j + 3] * v_new;
            H_reg[j]     = h0;
            H_reg[j + 1] = h1;
            H_reg[j + 2] = h2;
            H_reg[j + 3] = h3;
            qd0 += h0 * cur_q[j];
            qd1 += h1 * cur_q[j + 1];
            qd2 += h2 * cur_q[j + 2];
            qd3 += h3 * cur_q[j + 3];
        }
        float q_dot = (qd0 + qd1) + (qd2 + qd3);

        output[((unsigned long long)(b * seq_len + t) * num_v_heads + vh) * v_dim + tid] =
            __float2bfloat16(q_dot * inv_sqrt_d);

        __syncthreads();  // 2026-09-25: token t+1's k/q are stored, and token t's buffers are free to refill
    }

store_h:
    // 2026-09-25: For seq_len <= 1, a head whose state Frobenius norm exceeds MAX_NORM (50)
    // is scaled down to it; longer sequences store the state unclamped.

    if (seq_len <= 1) {
        float local_sq = 0.0f;
        #pragma unroll
        for (int j = 0; j < K_DIM; j++) {
            local_sq += H_reg[j] * H_reg[j];
        }
        unsigned int mask = __activemask();
        float ws = local_sq;
        ws += __shfl_down_sync(mask, ws, 16);
        ws += __shfl_down_sync(mask, ws, 8);
        ws += __shfl_down_sync(mask, ws, 4);
        ws += __shfl_down_sync(mask, ws, 2);
        ws += __shfl_down_sync(mask, ws, 1);
        __shared__ float ns[4];
        if (tid % 32 == 0) ns[tid / 32] = ws;
        __syncthreads();
        if (tid < 4) {
            float s = ns[tid];
            s += __shfl_down_sync(0xf, s, 2);
            s += __shfl_down_sync(0xf, s, 1);
            ns[0] = s;
        }
        __syncthreads();
        const float MAX_NORM = 50.0f;
        float norm_sq = ns[0];
        if (norm_sq > MAX_NORM * MAX_NORM) {
            float scale = MAX_NORM * rsqrtf(norm_sq);
            #pragma unroll
            for (int j = 0; j < K_DIM; j++) {
                H_reg[j] *= scale;
            }
        }
    }


    #pragma unroll
    for (int j = 0; j < K_DIM; j++) {
        H_global[j * v_dim + tid] = H_reg[j];
    }
}

// 2026-09-25: Split prefill: the gated_delta_rule_prefill recurrence with v_dim split across
// two CTAs of 64 threads per value head (blockIdx.x = vh * 2 + split). Thread tid_local
// owns state column split * 64 + tid_local and still holds all K_DIM rows of it; each
// thread loads two of the k/q values per buffer. No state clamp.
// ops::gdn_prefill_split: grid (num_v_heads * 2, batch, 1), block (64, 1, 1).










extern "C" __global__ void __launch_bounds__(64, 1)
gated_delta_rule_prefill_split(
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

    const unsigned int vh    = blockIdx.x / 2;
    const unsigned int split = blockIdx.x % 2;
    const unsigned int b     = blockIdx.y;
    if (vh >= num_v_heads || b >= batch_size) return;

    const unsigned int tid_local  = threadIdx.x;
    const unsigned int half       = blockDim.x;
    const unsigned int tid        = split * half + tid_local;
    const unsigned int head_repeat = num_v_heads / num_k_heads;
    const unsigned int kh = vh / head_repeat;


    extern __shared__ float smem[];
    float* smem_k0 = smem;
    float* smem_q0 = smem + K_DIM;
    float* smem_k1 = smem + 2 * K_DIM;
    float* smem_q1 = smem + 3 * K_DIM;

    float* H_global = h_state + ((unsigned long long)(b * num_v_heads + vh) * K_DIM * v_dim);


    float H_reg[K_DIM];
    #pragma unroll
    for (int j = 0; j < K_DIM; j++) {
        H_reg[j] = H_global[j * v_dim + tid];
    }

    float inv_sqrt_d = rsqrtf((float)k_dim);

    if (seq_len == 0) goto store_h_split;

    // 2026-09-25: Each thread loads k/q elements tid_local and tid_local + 64.

    {
        unsigned long long qk_off = (unsigned long long)0 * qk_stride + kh * k_dim;
        smem_k0[tid_local]        = (float)key[qk_off + tid_local];
        smem_k0[tid_local + half] = (float)key[qk_off + tid_local + half];
        smem_q0[tid_local]        = (float)query[qk_off + tid_local];
        smem_q0[tid_local + half] = (float)query[qk_off + tid_local + half];
    }
    __syncthreads();

    for (unsigned int t = 0; t < seq_len; t++) {
        float* cur_k = (t & 1) ? smem_k1 : smem_k0;
        float* cur_q = (t & 1) ? smem_q1 : smem_q0;
        float* nxt_k = (t & 1) ? smem_k0 : smem_k1;
        float* nxt_q = (t & 1) ? smem_q0 : smem_q1;

        if (t + 1 < seq_len) {
            unsigned long long qk_off_nxt = (unsigned long long)(t + 1) * qk_stride + kh * k_dim;
            nxt_k[tid_local]        = (float)key[qk_off_nxt + tid_local];
            nxt_k[tid_local + half] = (float)key[qk_off_nxt + tid_local + half];
            nxt_q[tid_local]        = (float)query[qk_off_nxt + tid_local];
            nxt_q[tid_local + half] = (float)query[qk_off_nxt + tid_local + half];
        }

        float v_i  = (float)value[(unsigned long long)t * v_stride + vh * v_dim + tid];
        float g_t  = fminf(fmaxf(gate[(unsigned long long)t * gb_stride + vh], 1e-6f), 1.0f - 1e-6f);
        float bt_t = beta[(unsigned long long)t * gb_stride + vh];

        float hk0 = 0.0f, hk1 = 0.0f, hk2 = 0.0f, hk3 = 0.0f;
        #pragma unroll
        for (int j = 0; j < K_DIM; j += 4) {
            hk0 += H_reg[j]     * cur_k[j];
            hk1 += H_reg[j + 1] * cur_k[j + 1];
            hk2 += H_reg[j + 2] * cur_k[j + 2];
            hk3 += H_reg[j + 3] * cur_k[j + 3];
        }
        float hk_dot = (hk0 + hk1) + (hk2 + hk3);

        float v_new = (v_i - g_t * hk_dot) * bt_t;

        float qd0 = 0.0f, qd1 = 0.0f, qd2 = 0.0f, qd3 = 0.0f;
        #pragma unroll
        for (int j = 0; j < K_DIM; j += 4) {
            float h0 = g_t * H_reg[j]     + cur_k[j]     * v_new;
            float h1 = g_t * H_reg[j + 1] + cur_k[j + 1] * v_new;
            float h2 = g_t * H_reg[j + 2] + cur_k[j + 2] * v_new;
            float h3 = g_t * H_reg[j + 3] + cur_k[j + 3] * v_new;
            H_reg[j]     = h0;
            H_reg[j + 1] = h1;
            H_reg[j + 2] = h2;
            H_reg[j + 3] = h3;
            qd0 += h0 * cur_q[j];
            qd1 += h1 * cur_q[j + 1];
            qd2 += h2 * cur_q[j + 2];
            qd3 += h3 * cur_q[j + 3];
        }
        float q_dot = (qd0 + qd1) + (qd2 + qd3);

        output[((unsigned long long)(b * seq_len + t) * num_v_heads + vh) * v_dim + tid] =
            __float2bfloat16(q_dot * inv_sqrt_d);

        __syncthreads();
    }

store_h_split:
    #pragma unroll
    for (int j = 0; j < K_DIM; j++) {
        H_global[j * v_dim + tid] = H_reg[j];
    }
}

// 2026-09-25: Split4 prefill: as the split prefill with four CTAs of 32 threads per value head
// (blockIdx.x = vh * 4 + split); each thread loads four k/q values per buffer, stride 32.
// No state clamp. ops::gdn_prefill_split4: grid (num_v_heads * 4, batch, 1), block (32, 1, 1).







extern "C" __global__ void __launch_bounds__(32, 1)
gated_delta_rule_prefill_split4(
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

    const unsigned int vh    = blockIdx.x / 4;
    const unsigned int split = blockIdx.x % 4;
    const unsigned int b     = blockIdx.y;
    if (vh >= num_v_heads || b >= batch_size) return;

    const unsigned int tid_local  = threadIdx.x;
    const unsigned int quarter    = blockDim.x;
    const unsigned int tid        = split * quarter + tid_local;
    const unsigned int head_repeat = num_v_heads / num_k_heads;
    const unsigned int kh = vh / head_repeat;


    extern __shared__ float smem[];
    float* smem_k0 = smem;
    float* smem_q0 = smem + K_DIM;
    float* smem_k1 = smem + 2 * K_DIM;
    float* smem_q1 = smem + 3 * K_DIM;

    float* H_global = h_state + ((unsigned long long)(b * num_v_heads + vh) * K_DIM * v_dim);

    float H_reg[K_DIM];
    #pragma unroll
    for (int j = 0; j < K_DIM; j++) {
        H_reg[j] = H_global[j * v_dim + tid];
    }

    float inv_sqrt_d = rsqrtf((float)k_dim);

    if (seq_len == 0) goto store_h_split4;


    {
        unsigned long long qk_off = (unsigned long long)0 * qk_stride + kh * k_dim;
        smem_k0[tid_local]            = (float)key[qk_off + tid_local];
        smem_k0[tid_local + quarter]  = (float)key[qk_off + tid_local + quarter];
        smem_k0[tid_local + 2*quarter]= (float)key[qk_off + tid_local + 2*quarter];
        smem_k0[tid_local + 3*quarter]= (float)key[qk_off + tid_local + 3*quarter];
        smem_q0[tid_local]            = (float)query[qk_off + tid_local];
        smem_q0[tid_local + quarter]  = (float)query[qk_off + tid_local + quarter];
        smem_q0[tid_local + 2*quarter]= (float)query[qk_off + tid_local + 2*quarter];
        smem_q0[tid_local + 3*quarter]= (float)query[qk_off + tid_local + 3*quarter];
    }
    __syncthreads();

    for (unsigned int t = 0; t < seq_len; t++) {
        float* cur_k = (t & 1) ? smem_k1 : smem_k0;
        float* cur_q = (t & 1) ? smem_q1 : smem_q0;
        float* nxt_k = (t & 1) ? smem_k0 : smem_k1;
        float* nxt_q = (t & 1) ? smem_q0 : smem_q1;

        if (t + 1 < seq_len) {
            unsigned long long qk_off_nxt = (unsigned long long)(t + 1) * qk_stride + kh * k_dim;
            nxt_k[tid_local]            = (float)key[qk_off_nxt + tid_local];
            nxt_k[tid_local + quarter]  = (float)key[qk_off_nxt + tid_local + quarter];
            nxt_k[tid_local + 2*quarter]= (float)key[qk_off_nxt + tid_local + 2*quarter];
            nxt_k[tid_local + 3*quarter]= (float)key[qk_off_nxt + tid_local + 3*quarter];
            nxt_q[tid_local]            = (float)query[qk_off_nxt + tid_local];
            nxt_q[tid_local + quarter]  = (float)query[qk_off_nxt + tid_local + quarter];
            nxt_q[tid_local + 2*quarter]= (float)query[qk_off_nxt + tid_local + 2*quarter];
            nxt_q[tid_local + 3*quarter]= (float)query[qk_off_nxt + tid_local + 3*quarter];
        }

        float v_i  = (float)value[(unsigned long long)t * v_stride + vh * v_dim + tid];
        float g_t  = fminf(fmaxf(gate[(unsigned long long)t * gb_stride + vh], 1e-6f), 1.0f - 1e-6f);
        float bt_t = beta[(unsigned long long)t * gb_stride + vh];

        float hk0 = 0.0f, hk1 = 0.0f, hk2 = 0.0f, hk3 = 0.0f;
        #pragma unroll
        for (int j = 0; j < K_DIM; j += 4) {
            hk0 += H_reg[j]     * cur_k[j];
            hk1 += H_reg[j + 1] * cur_k[j + 1];
            hk2 += H_reg[j + 2] * cur_k[j + 2];
            hk3 += H_reg[j + 3] * cur_k[j + 3];
        }
        float hk_dot = (hk0 + hk1) + (hk2 + hk3);

        float v_new = (v_i - g_t * hk_dot) * bt_t;

        float qd0 = 0.0f, qd1 = 0.0f, qd2 = 0.0f, qd3 = 0.0f;
        #pragma unroll
        for (int j = 0; j < K_DIM; j += 4) {
            float h0 = g_t * H_reg[j]     + cur_k[j]     * v_new;
            float h1 = g_t * H_reg[j + 1] + cur_k[j + 1] * v_new;
            float h2 = g_t * H_reg[j + 2] + cur_k[j + 2] * v_new;
            float h3 = g_t * H_reg[j + 3] + cur_k[j + 3] * v_new;
            H_reg[j]     = h0;
            H_reg[j + 1] = h1;
            H_reg[j + 2] = h2;
            H_reg[j + 3] = h3;
            qd0 += h0 * cur_q[j];
            qd1 += h1 * cur_q[j + 1];
            qd2 += h2 * cur_q[j + 2];
            qd3 += h3 * cur_q[j + 3];
        }
        float q_dot = (qd0 + qd1) + (qd2 + qd3);

        output[((unsigned long long)(b * seq_len + t) * num_v_heads + vh) * v_dim + tid] =
            __float2bfloat16(q_dot * inv_sqrt_d);

        __syncthreads();
    }

store_h_split4:
    #pragma unroll
    for (int j = 0; j < K_DIM; j++) {
        H_global[j * v_dim + tid] = H_reg[j];
    }
}

// 2026-09-25: Batched split4 prefill: gated_delta_rule_prefill_split4 over batch_size streams in
// one launch. Stream b's state is h_state_ptrs[b], and its q/k/v/gate/beta rows and output
// start at row b * seq_len, so every stream has exactly seq_len tokens.
// ops::gdn_prefill_split4_batched: grid (num_v_heads * 4, batch_size, 1), block (32, 1, 1).


extern "C" __global__ void __launch_bounds__(32, 1)
gated_delta_rule_prefill_split4_batched(
    float* const* __restrict__ h_state_ptrs,
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
    const unsigned int vh    = blockIdx.x / 4;
    const unsigned int split = blockIdx.x % 4;
    const unsigned int b     = blockIdx.y;
    if (vh >= num_v_heads || b >= batch_size) return;

    const unsigned int tid_local  = threadIdx.x;
    const unsigned int quarter    = blockDim.x;
    const unsigned int tid        = split * quarter + tid_local;
    const unsigned int head_repeat = num_v_heads / num_k_heads;
    const unsigned int kh = vh / head_repeat;

    const unsigned long long qk_batch_off = (unsigned long long)b * seq_len * qk_stride;
    const unsigned long long v_batch_off  = (unsigned long long)b * seq_len * v_stride;
    const unsigned long long gb_batch_off = (unsigned long long)b * seq_len * gb_stride;
    const unsigned long long out_batch_off = (unsigned long long)b * seq_len * num_v_heads * v_dim;

    extern __shared__ float smem[];
    float* smem_k0 = smem;
    float* smem_q0 = smem + K_DIM;
    float* smem_k1 = smem + 2 * K_DIM;
    float* smem_q1 = smem + 3 * K_DIM;

    float* H_global = h_state_ptrs[b] + ((unsigned long long)vh * K_DIM * v_dim);

    float H_reg[K_DIM];
    #pragma unroll
    for (int j = 0; j < K_DIM; j++) {
        H_reg[j] = H_global[j * v_dim + tid];
    }

    float inv_sqrt_d = rsqrtf((float)k_dim);

    if (seq_len == 0) goto store_h_split4_batched;

    {
        unsigned long long qk_off = qk_batch_off + kh * k_dim;
        smem_k0[tid_local]            = (float)key[qk_off + tid_local];
        smem_k0[tid_local + quarter]  = (float)key[qk_off + tid_local + quarter];
        smem_k0[tid_local + 2*quarter]= (float)key[qk_off + tid_local + 2*quarter];
        smem_k0[tid_local + 3*quarter]= (float)key[qk_off + tid_local + 3*quarter];
        smem_q0[tid_local]            = (float)query[qk_off + tid_local];
        smem_q0[tid_local + quarter]  = (float)query[qk_off + tid_local + quarter];
        smem_q0[tid_local + 2*quarter]= (float)query[qk_off + tid_local + 2*quarter];
        smem_q0[tid_local + 3*quarter]= (float)query[qk_off + tid_local + 3*quarter];
    }
    __syncthreads();

    for (unsigned int t = 0; t < seq_len; t++) {
        float* cur_k = (t & 1) ? smem_k1 : smem_k0;
        float* cur_q = (t & 1) ? smem_q1 : smem_q0;
        float* nxt_k = (t & 1) ? smem_k0 : smem_k1;
        float* nxt_q = (t & 1) ? smem_q0 : smem_q1;

        if (t + 1 < seq_len) {
            unsigned long long qk_off_nxt = qk_batch_off + (unsigned long long)(t + 1) * qk_stride + kh * k_dim;
            nxt_k[tid_local]            = (float)key[qk_off_nxt + tid_local];
            nxt_k[tid_local + quarter]  = (float)key[qk_off_nxt + tid_local + quarter];
            nxt_k[tid_local + 2*quarter]= (float)key[qk_off_nxt + tid_local + 2*quarter];
            nxt_k[tid_local + 3*quarter]= (float)key[qk_off_nxt + tid_local + 3*quarter];
            nxt_q[tid_local]            = (float)query[qk_off_nxt + tid_local];
            nxt_q[tid_local + quarter]  = (float)query[qk_off_nxt + tid_local + quarter];
            nxt_q[tid_local + 2*quarter]= (float)query[qk_off_nxt + tid_local + 2*quarter];
            nxt_q[tid_local + 3*quarter]= (float)query[qk_off_nxt + tid_local + 3*quarter];
        }

        float v_i  = (float)value[v_batch_off + (unsigned long long)t * v_stride + vh * v_dim + tid];
        float g_t  = fminf(fmaxf(gate[gb_batch_off + (unsigned long long)t * gb_stride + vh], 1e-6f), 1.0f - 1e-6f);
        float bt_t = beta[gb_batch_off + (unsigned long long)t * gb_stride + vh];

        float hk0 = 0.0f, hk1 = 0.0f, hk2 = 0.0f, hk3 = 0.0f;
        #pragma unroll
        for (int j = 0; j < K_DIM; j += 4) {
            hk0 += H_reg[j]     * cur_k[j];
            hk1 += H_reg[j + 1] * cur_k[j + 1];
            hk2 += H_reg[j + 2] * cur_k[j + 2];
            hk3 += H_reg[j + 3] * cur_k[j + 3];
        }
        float hk_dot = (hk0 + hk1) + (hk2 + hk3);

        float v_new = (v_i - g_t * hk_dot) * bt_t;

        float qd0 = 0.0f, qd1 = 0.0f, qd2 = 0.0f, qd3 = 0.0f;
        #pragma unroll
        for (int j = 0; j < K_DIM; j += 4) {
            float h0 = g_t * H_reg[j]     + cur_k[j]     * v_new;
            float h1 = g_t * H_reg[j + 1] + cur_k[j + 1] * v_new;
            float h2 = g_t * H_reg[j + 2] + cur_k[j + 2] * v_new;
            float h3 = g_t * H_reg[j + 3] + cur_k[j + 3] * v_new;
            H_reg[j]     = h0;
            H_reg[j + 1] = h1;
            H_reg[j + 2] = h2;
            H_reg[j + 3] = h3;
            qd0 += h0 * cur_q[j];
            qd1 += h1 * cur_q[j + 1];
            qd2 += h2 * cur_q[j + 2];
            qd3 += h3 * cur_q[j + 3];
        }
        float q_dot = (qd0 + qd1) + (qd2 + qd3);

        output[out_batch_off + ((unsigned long long)t * num_v_heads + vh) * v_dim + tid] =
            __float2bfloat16(q_dot * inv_sqrt_d);

        __syncthreads();
    }

store_h_split4_batched:
    #pragma unroll
    for (int j = 0; j < K_DIM; j++) {
        H_global[j * v_dim + tid] = H_reg[j];
    }
}


// 2026-09-25: Single-token decode, FP32 q/k/v, BF16 output. Each thread loads its state column
// into H_reg once, runs both passes from registers, and stores it once.
// ops::gdn_decode: grid (num_v_heads, batch, 1), block (128, 1, 1).





















extern "C" __global__ void __launch_bounds__(128, 1)
gated_delta_rule_decode(
    float* __restrict__ h_state,
    const float* __restrict__ query,
    const float* __restrict__ key,
    const float* __restrict__ value,
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

    float* H_global = h_state + ((unsigned long long)(b * num_v_heads + vh) * K_DIM * v_dim);


    __shared__ float smem_k[K_DIM];
    __shared__ float smem_q[K_DIM];
    const float* k_ptr = key + (b * num_k_heads + kh) * k_dim;
    const float* q_ptr = query + (b * num_k_heads + kh) * k_dim;
    smem_k[tid] = k_ptr[tid];
    smem_q[tid] = q_ptr[tid];
    __syncthreads();


    float H_reg[K_DIM];
    #pragma unroll
    for (int j = 0; j < K_DIM; j++) {
        H_reg[j] = H_global[j * v_dim + tid];
    }

    float v_i = value[(b * num_v_heads + vh) * v_dim + tid];
    float g = fminf(fmaxf(gate[b * num_v_heads + vh], 1e-6f), 1.0f - 1e-6f);
    float bt = beta[b * num_v_heads + vh];

    // 2026-09-25: Pass 1: hk_dot = h^T k from registers.
    float hk0 = 0.0f, hk1 = 0.0f, hk2 = 0.0f, hk3 = 0.0f;
    #pragma unroll
    for (int j = 0; j < K_DIM; j += 4) {
        hk0 += H_reg[j]     * smem_k[j];
        hk1 += H_reg[j + 1] * smem_k[j + 1];
        hk2 += H_reg[j + 2] * smem_k[j + 2];
        hk3 += H_reg[j + 3] * smem_k[j + 3];
    }
    float hk_dot = (hk0 + hk1) + (hk2 + hk3);

    float v_new = (v_i - g * hk_dot) * bt;

    // 2026-09-25: Pass 2: update h in registers and accumulate q_dot.
    float qd0 = 0.0f, qd1 = 0.0f, qd2 = 0.0f, qd3 = 0.0f;
    #pragma unroll
    for (int j = 0; j < K_DIM; j += 4) {
        float h0 = g * H_reg[j]     + smem_k[j]     * v_new;
        float h1 = g * H_reg[j + 1] + smem_k[j + 1] * v_new;
        float h2 = g * H_reg[j + 2] + smem_k[j + 2] * v_new;
        float h3 = g * H_reg[j + 3] + smem_k[j + 3] * v_new;
        H_reg[j]     = h0;
        H_reg[j + 1] = h1;
        H_reg[j + 2] = h2;
        H_reg[j + 3] = h3;
        qd0 += h0 * smem_q[j];
        qd1 += h1 * smem_q[j + 1];
        qd2 += h2 * smem_q[j + 2];
        qd3 += h3 * smem_q[j + 3];
    }
    float q_dot = (qd0 + qd1) + (qd2 + qd3);

    float inv_sqrt_d = rsqrtf((float)k_dim);
    output[(b * num_v_heads + vh) * v_dim + tid] = __float2bfloat16(q_dot * inv_sqrt_d);


    #pragma unroll
    for (int j = 0; j < K_DIM; j++) {
        H_global[j * v_dim + tid] = H_reg[j];
    }
}

// 2026-09-25: gated_delta_rule_decode with an FP32 output.





extern "C" __global__ void __launch_bounds__(128, 1)
gated_delta_rule_decode_f32(
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
    const unsigned int head_repeat = num_v_heads / num_k_heads;
    const unsigned int kh = vh / head_repeat;

    float* H_global = h_state + ((unsigned long long)(b * num_v_heads + vh) * K_DIM * v_dim);

    __shared__ float smem_k[K_DIM];
    __shared__ float smem_q[K_DIM];
    const float* k_ptr = key + (b * num_k_heads + kh) * k_dim;
    const float* q_ptr = query + (b * num_k_heads + kh) * k_dim;
    smem_k[tid] = k_ptr[tid];
    smem_q[tid] = q_ptr[tid];
    __syncthreads();

    float H_reg[K_DIM];
    #pragma unroll
    for (int j = 0; j < K_DIM; j++) {
        H_reg[j] = H_global[j * v_dim + tid];
    }

    float v_i = value[(b * num_v_heads + vh) * v_dim + tid];
    float g = fminf(fmaxf(gate[b * num_v_heads + vh], 1e-6f), 1.0f - 1e-6f);
    float bt = beta[b * num_v_heads + vh];

    float hk0 = 0.0f, hk1 = 0.0f, hk2 = 0.0f, hk3 = 0.0f;
    #pragma unroll
    for (int j = 0; j < K_DIM; j += 4) {
        hk0 += H_reg[j]     * smem_k[j];
        hk1 += H_reg[j + 1] * smem_k[j + 1];
        hk2 += H_reg[j + 2] * smem_k[j + 2];
        hk3 += H_reg[j + 3] * smem_k[j + 3];
    }
    float hk_dot = (hk0 + hk1) + (hk2 + hk3);

    float v_new = (v_i - g * hk_dot) * bt;

    float qd0 = 0.0f, qd1 = 0.0f, qd2 = 0.0f, qd3 = 0.0f;
    #pragma unroll
    for (int j = 0; j < K_DIM; j += 4) {
        float h0 = g * H_reg[j]     + smem_k[j]     * v_new;
        float h1 = g * H_reg[j + 1] + smem_k[j + 1] * v_new;
        float h2 = g * H_reg[j + 2] + smem_k[j + 2] * v_new;
        float h3 = g * H_reg[j + 3] + smem_k[j + 3] * v_new;
        H_reg[j]     = h0;
        H_reg[j + 1] = h1;
        H_reg[j + 2] = h2;
        H_reg[j + 3] = h3;
        qd0 += h0 * smem_q[j];
        qd1 += h1 * smem_q[j + 1];
        qd2 += h2 * smem_q[j + 2];
        qd3 += h3 * smem_q[j + 3];
    }
    float q_dot = (qd0 + qd1) + (qd2 + qd3);

    float inv_sqrt_d = rsqrtf((float)k_dim);
    output[(b * num_v_heads + vh) * v_dim + tid] = q_dot * inv_sqrt_d;

    #pragma unroll
    for (int j = 0; j < K_DIM; j++) {
        H_global[j * v_dim + tid] = H_reg[j];
    }
}

// 2026-09-25: gated_delta_rule_decode_f32's recurrence, then the gated RMS norm in the same block:
// out = x * rsqrt(mean(x^2) + eps) * w * silu(z), BF16, with x the recurrence output.
// z_gate and output are indexed by value head only, with no batch offset, so batch_size
// must be 1; every ops::gdn_decode_f32_norm call in crates/model-layers passes 1.




extern "C" __global__ void __launch_bounds__(128, 1)
gated_delta_rule_decode_f32_norm(
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
    const unsigned int head_repeat = num_v_heads / num_k_heads;
    const unsigned int kh = vh / head_repeat;

    float* H_global = h_state + ((unsigned long long)(b * num_v_heads + vh) * K_DIM * v_dim);

    __shared__ float smem_k[K_DIM];
    __shared__ float smem_q[K_DIM];
    const float* k_ptr = key + (b * num_k_heads + kh) * k_dim;
    const float* q_ptr = query + (b * num_k_heads + kh) * k_dim;
    smem_k[tid] = k_ptr[tid];
    smem_q[tid] = q_ptr[tid];
    __syncthreads();

    float H_reg[K_DIM];
    #pragma unroll
    for (int j = 0; j < K_DIM; j++) {
        H_reg[j] = H_global[j * v_dim + tid];
    }

    float v_i = value[(b * num_v_heads + vh) * v_dim + tid];
    float g = fminf(fmaxf(gate[b * num_v_heads + vh], 1e-6f), 1.0f - 1e-6f);
    float bt = beta[b * num_v_heads + vh];

    float hk0 = 0.0f, hk1 = 0.0f, hk2 = 0.0f, hk3 = 0.0f;
    #pragma unroll
    for (int j = 0; j < K_DIM; j += 4) {
        hk0 += H_reg[j]     * smem_k[j];
        hk1 += H_reg[j + 1] * smem_k[j + 1];
        hk2 += H_reg[j + 2] * smem_k[j + 2];
        hk3 += H_reg[j + 3] * smem_k[j + 3];
    }
    float hk_dot = (hk0 + hk1) + (hk2 + hk3);

    float v_new = (v_i - g * hk_dot) * bt;

    float qd0 = 0.0f, qd1 = 0.0f, qd2 = 0.0f, qd3 = 0.0f;
    #pragma unroll
    for (int j = 0; j < K_DIM; j += 4) {
        float h0 = g * H_reg[j]     + smem_k[j]     * v_new;
        float h1 = g * H_reg[j + 1] + smem_k[j + 1] * v_new;
        float h2 = g * H_reg[j + 2] + smem_k[j + 2] * v_new;
        float h3 = g * H_reg[j + 3] + smem_k[j + 3] * v_new;
        H_reg[j]     = h0;
        H_reg[j + 1] = h1;
        H_reg[j + 2] = h2;
        H_reg[j + 3] = h3;
        qd0 += h0 * smem_q[j];
        qd1 += h1 * smem_q[j + 1];
        qd2 += h2 * smem_q[j + 2];
        qd3 += h3 * smem_q[j + 3];
    }
    float q_dot = (qd0 + qd1) + (qd2 + qd3);

    float inv_sqrt_d = rsqrtf((float)k_dim);
    float x = q_dot * inv_sqrt_d;

    __shared__ float x_cache[K_DIM];
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

    float rms = rsqrtf(rms_sums[0] / (float)v_dim + eps);

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

    #pragma unroll
    for (int j = 0; j < K_DIM; j++) {
        H_global[j * v_dim + tid] = H_reg[j];
    }
}

// 2026-09-25: gated_delta_rule_decode_f32 with strided rows: sequence b's q/k start at
// b * qk_stride, v at b * v_stride, gate/beta at b * gb_stride and the output at
// b * out_stride.






extern "C" __global__ void __launch_bounds__(128, 1)
gated_delta_rule_decode_f32_strided(
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
    const unsigned int head_repeat = num_v_heads / num_k_heads;
    const unsigned int kh = vh / head_repeat;

    float* H_global = h_state + ((unsigned long long)(b * num_v_heads + vh) * K_DIM * v_dim);

    __shared__ float smem_k[K_DIM];
    __shared__ float smem_q[K_DIM];
    const float* k_ptr = key + (unsigned long long)b * qk_stride + kh * k_dim;
    const float* q_ptr = query + (unsigned long long)b * qk_stride + kh * k_dim;
    smem_k[tid] = k_ptr[tid];
    smem_q[tid] = q_ptr[tid];
    __syncthreads();

    float H_reg[K_DIM];
    #pragma unroll
    for (int j = 0; j < K_DIM; j++) {
        H_reg[j] = H_global[j * v_dim + tid];
    }

    const float v_i = value[(unsigned long long)b * v_stride + vh * v_dim + tid];
    const float g = fminf(
        fmaxf(gate[(unsigned long long)b * gb_stride + vh], 1e-6f),
        1.0f - 1e-6f
    );
    const float bt = beta[(unsigned long long)b * gb_stride + vh];

    float hk0 = 0.0f, hk1 = 0.0f, hk2 = 0.0f, hk3 = 0.0f;
    #pragma unroll
    for (int j = 0; j < K_DIM; j += 4) {
        hk0 += H_reg[j]     * smem_k[j];
        hk1 += H_reg[j + 1] * smem_k[j + 1];
        hk2 += H_reg[j + 2] * smem_k[j + 2];
        hk3 += H_reg[j + 3] * smem_k[j + 3];
    }
    const float hk_dot = (hk0 + hk1) + (hk2 + hk3);

    const float v_new = (v_i - g * hk_dot) * bt;

    float qd0 = 0.0f, qd1 = 0.0f, qd2 = 0.0f, qd3 = 0.0f;
    #pragma unroll
    for (int j = 0; j < K_DIM; j += 4) {
        float h0 = g * H_reg[j]     + smem_k[j]     * v_new;
        float h1 = g * H_reg[j + 1] + smem_k[j + 1] * v_new;
        float h2 = g * H_reg[j + 2] + smem_k[j + 2] * v_new;
        float h3 = g * H_reg[j + 3] + smem_k[j + 3] * v_new;
        H_reg[j]     = h0;
        H_reg[j + 1] = h1;
        H_reg[j + 2] = h2;
        H_reg[j + 3] = h3;
        qd0 += h0 * smem_q[j];
        qd1 += h1 * smem_q[j + 1];
        qd2 += h2 * smem_q[j + 2];
        qd3 += h3 * smem_q[j + 3];
    }
    const float q_dot = (qd0 + qd1) + (qd2 + qd3);

    const float inv_sqrt_d = rsqrtf((float)k_dim);
    output[(unsigned long long)b * out_stride + vh * v_dim + tid] = q_dot * inv_sqrt_d;

    #pragma unroll
    for (int j = 0; j < K_DIM; j++) {
        H_global[j * v_dim + tid] = H_reg[j];
    }
}

// 2026-09-25: Two tokens (rows 2b and 2b + 1) in one launch, the state in registers throughout.
// Only the state after the second token is stored; h_state_intermediate is neither read
// nor written. ops::gdn_decode_chunk2: grid (num_v_heads, batch, 1), block (128, 1, 1).



















extern "C" __global__ void __launch_bounds__(128, 1)
gated_delta_rule_chunk2(
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

    float* H_global = h_state + ((unsigned long long)(b * num_v_heads + vh) * K_DIM * v_dim);


    __shared__ float sk0[K_DIM], sq0[K_DIM], sk1[K_DIM], sq1[K_DIM];
    {
        unsigned long long qk0 = (unsigned long long)(b * 2) * qk_stride + kh * k_dim;
        unsigned long long qk1 = (unsigned long long)(b * 2 + 1) * qk_stride + kh * k_dim;
        sk0[tid] = (float)key[qk0 + tid];   sq0[tid] = (float)query[qk0 + tid];
        sk1[tid] = (float)key[qk1 + tid];   sq1[tid] = (float)query[qk1 + tid];
    }
    __syncthreads();


    float H_reg[K_DIM];
    #pragma unroll
    for (int j = 0; j < K_DIM; j++) {
        H_reg[j] = H_global[j * v_dim + tid];
    }

    float vi0 = (float)value[(unsigned long long)(b * 2) * v_stride + vh * v_dim + tid];
    float vi1 = (float)value[(unsigned long long)(b * 2 + 1) * v_stride + vh * v_dim + tid];
    float g0 = fminf(fmaxf(gate[(unsigned long long)(b * 2) * gb_stride + vh], 1e-6f), 1.0f - 1e-6f);
    float bt0 = beta[(unsigned long long)(b * 2) * gb_stride + vh];
    float g1 = fminf(fmaxf(gate[(unsigned long long)(b * 2 + 1) * gb_stride + vh], 1e-6f), 1.0f - 1e-6f);
    float bt1 = beta[(unsigned long long)(b * 2 + 1) * gb_stride + vh];



    float hk_a = 0.0f, hk_b = 0.0f, hk_c = 0.0f, hk_d = 0.0f;
    #pragma unroll
    for (int j = 0; j < K_DIM; j += 4) {
        hk_a += H_reg[j]     * sk0[j];
        hk_b += H_reg[j + 1] * sk0[j + 1];
        hk_c += H_reg[j + 2] * sk0[j + 2];
        hk_d += H_reg[j + 3] * sk0[j + 3];
    }
    float v_new_0 = (vi0 - g0 * ((hk_a + hk_b) + (hk_c + hk_d))) * bt0;

    // 2026-09-25: Update h for token 0 and, in the same loop, accumulate q0_dot and token 1's hk_dot.
    float qd0a = 0.0f, qd0b = 0.0f, qd0c = 0.0f, qd0d = 0.0f;
    float hk1a = 0.0f, hk1b = 0.0f, hk1c = 0.0f, hk1d = 0.0f;
    #pragma unroll
    for (int j = 0; j < K_DIM; j += 4) {
        float h0 = g0 * H_reg[j]     + sk0[j]     * v_new_0;
        float h1 = g0 * H_reg[j + 1] + sk0[j + 1] * v_new_0;
        float h2 = g0 * H_reg[j + 2] + sk0[j + 2] * v_new_0;
        float h3 = g0 * H_reg[j + 3] + sk0[j + 3] * v_new_0;
        H_reg[j]     = h0;
        H_reg[j + 1] = h1;
        H_reg[j + 2] = h2;
        H_reg[j + 3] = h3;
        qd0a += h0 * sq0[j];     qd0b += h1 * sq0[j + 1];
        qd0c += h2 * sq0[j + 2]; qd0d += h3 * sq0[j + 3];
        hk1a += h0 * sk1[j];     hk1b += h1 * sk1[j + 1];
        hk1c += h2 * sk1[j + 2]; hk1d += h3 * sk1[j + 3];
    }
    float q0_dot = (qd0a + qd0b) + (qd0c + qd0d);
    float v_new_1 = (vi1 - g1 * ((hk1a + hk1b) + (hk1c + hk1d))) * bt1;



    float qd1a = 0.0f, qd1b = 0.0f, qd1c = 0.0f, qd1d = 0.0f;
    #pragma unroll
    for (int j = 0; j < K_DIM; j += 4) {
        float h0 = g1 * H_reg[j]     + sk1[j]     * v_new_1;
        float h1 = g1 * H_reg[j + 1] + sk1[j + 1] * v_new_1;
        float h2 = g1 * H_reg[j + 2] + sk1[j + 2] * v_new_1;
        float h3 = g1 * H_reg[j + 3] + sk1[j + 3] * v_new_1;
        H_reg[j]     = h0;
        H_reg[j + 1] = h1;
        H_reg[j + 2] = h2;
        H_reg[j + 3] = h3;
        qd1a += h0 * sq1[j];     qd1b += h1 * sq1[j + 1];
        qd1c += h2 * sq1[j + 2]; qd1d += h3 * sq1[j + 3];
    }
    float q1_dot = (qd1a + qd1b) + (qd1c + qd1d);

    float inv_sqrt_d = rsqrtf((float)k_dim);
    output[((unsigned long long)(b * 2) * num_v_heads + vh) * v_dim + tid] =
        __float2bfloat16(q0_dot * inv_sqrt_d);
    output[((unsigned long long)(b * 2 + 1) * num_v_heads + vh) * v_dim + tid] =
        __float2bfloat16(q1_dot * inv_sqrt_d);


    #pragma unroll
    for (int j = 0; j < K_DIM; j++) {
        H_global[j * v_dim + tid] = H_reg[j];
    }
}

// 2026-09-25: Three tokens (rows 3b to 3b + 2) in one launch, as chunk2: only the final state is
// stored; h_state_inter0 and h_state_inter1 are not used.
// ops::gdn_decode_chunk3: grid (num_v_heads, batch, 1), block (128, 1, 1).













extern "C" __global__ void __launch_bounds__(128, 1)
gated_delta_rule_chunk3(
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

    float* H_global = h_state + ((unsigned long long)(b * num_v_heads + vh) * K_DIM * v_dim);


    __shared__ float sk0[K_DIM], sq0[K_DIM], sk1[K_DIM], sq1[K_DIM], sk2[K_DIM], sq2[K_DIM];
    {
        unsigned long long qk0 = (unsigned long long)(b * 3) * qk_stride + kh * k_dim;
        unsigned long long qk1 = (unsigned long long)(b * 3 + 1) * qk_stride + kh * k_dim;
        unsigned long long qk2 = (unsigned long long)(b * 3 + 2) * qk_stride + kh * k_dim;
        sk0[tid] = (float)key[qk0 + tid]; sq0[tid] = (float)query[qk0 + tid];
        sk1[tid] = (float)key[qk1 + tid]; sq1[tid] = (float)query[qk1 + tid];
        sk2[tid] = (float)key[qk2 + tid]; sq2[tid] = (float)query[qk2 + tid];
    }
    __syncthreads();


    float H_reg[K_DIM];
    #pragma unroll
    for (int j = 0; j < K_DIM; j++) {
        H_reg[j] = H_global[j * v_dim + tid];
    }

    float vi0 = (float)value[(unsigned long long)(b * 3) * v_stride + vh * v_dim + tid];
    float vi1 = (float)value[(unsigned long long)(b * 3 + 1) * v_stride + vh * v_dim + tid];
    float vi2 = (float)value[(unsigned long long)(b * 3 + 2) * v_stride + vh * v_dim + tid];
    float g0 = fminf(fmaxf(gate[(unsigned long long)(b * 3) * gb_stride + vh], 1e-6f), 1.0f - 1e-6f);
    float bt0 = beta[(unsigned long long)(b * 3) * gb_stride + vh];
    float g1 = fminf(fmaxf(gate[(unsigned long long)(b * 3 + 1) * gb_stride + vh], 1e-6f), 1.0f - 1e-6f);
    float bt1 = beta[(unsigned long long)(b * 3 + 1) * gb_stride + vh];
    float g2 = fminf(fmaxf(gate[(unsigned long long)(b * 3 + 2) * gb_stride + vh], 1e-6f), 1.0f - 1e-6f);
    float bt2 = beta[(unsigned long long)(b * 3 + 2) * gb_stride + vh];


    float hk_a = 0.0f, hk_b = 0.0f, hk_c = 0.0f, hk_d = 0.0f;
    #pragma unroll
    for (int j = 0; j < K_DIM; j += 4) {
        hk_a += H_reg[j]     * sk0[j];
        hk_b += H_reg[j + 1] * sk0[j + 1];
        hk_c += H_reg[j + 2] * sk0[j + 2];
        hk_d += H_reg[j + 3] * sk0[j + 3];
    }
    float v_new_0 = (vi0 - g0 * ((hk_a + hk_b) + (hk_c + hk_d))) * bt0;

    // 2026-09-25: Update h for token 0 and accumulate q0_dot and token 1's hk_dot.
    float qd0a = 0.0f, qd0b = 0.0f, qd0c = 0.0f, qd0d = 0.0f;
    float hk1a = 0.0f, hk1b = 0.0f, hk1c = 0.0f, hk1d = 0.0f;
    #pragma unroll
    for (int j = 0; j < K_DIM; j += 4) {
        float h0 = g0 * H_reg[j]     + sk0[j]     * v_new_0;
        float h1 = g0 * H_reg[j + 1] + sk0[j + 1] * v_new_0;
        float h2 = g0 * H_reg[j + 2] + sk0[j + 2] * v_new_0;
        float h3 = g0 * H_reg[j + 3] + sk0[j + 3] * v_new_0;
        H_reg[j] = h0; H_reg[j+1] = h1; H_reg[j+2] = h2; H_reg[j+3] = h3;
        qd0a += h0 * sq0[j];     qd0b += h1 * sq0[j + 1];
        qd0c += h2 * sq0[j + 2]; qd0d += h3 * sq0[j + 3];
        hk1a += h0 * sk1[j];     hk1b += h1 * sk1[j + 1];
        hk1c += h2 * sk1[j + 2]; hk1d += h3 * sk1[j + 3];
    }
    float q0_dot = (qd0a + qd0b) + (qd0c + qd0d);
    float v_new_1 = (vi1 - g1 * ((hk1a + hk1b) + (hk1c + hk1d))) * bt1;

    // 2026-09-25: Update h for token 1 and accumulate q1_dot and token 2's hk_dot.

    float qd1a = 0.0f, qd1b = 0.0f, qd1c = 0.0f, qd1d = 0.0f;
    float hk2a = 0.0f, hk2b = 0.0f, hk2c = 0.0f, hk2d = 0.0f;
    #pragma unroll
    for (int j = 0; j < K_DIM; j += 4) {
        float h0 = g1 * H_reg[j]     + sk1[j]     * v_new_1;
        float h1 = g1 * H_reg[j + 1] + sk1[j + 1] * v_new_1;
        float h2 = g1 * H_reg[j + 2] + sk1[j + 2] * v_new_1;
        float h3 = g1 * H_reg[j + 3] + sk1[j + 3] * v_new_1;
        H_reg[j] = h0; H_reg[j+1] = h1; H_reg[j+2] = h2; H_reg[j+3] = h3;
        qd1a += h0 * sq1[j];     qd1b += h1 * sq1[j + 1];
        qd1c += h2 * sq1[j + 2]; qd1d += h3 * sq1[j + 3];
        hk2a += h0 * sk2[j];     hk2b += h1 * sk2[j + 1];
        hk2c += h2 * sk2[j + 2]; hk2d += h3 * sk2[j + 3];
    }
    float q1_dot = (qd1a + qd1b) + (qd1c + qd1d);
    float v_new_2 = (vi2 - g2 * ((hk2a + hk2b) + (hk2c + hk2d))) * bt2;



    float qd2a = 0.0f, qd2b = 0.0f, qd2c = 0.0f, qd2d = 0.0f;
    #pragma unroll
    for (int j = 0; j < K_DIM; j += 4) {
        float h0 = g2 * H_reg[j]     + sk2[j]     * v_new_2;
        float h1 = g2 * H_reg[j + 1] + sk2[j + 1] * v_new_2;
        float h2 = g2 * H_reg[j + 2] + sk2[j + 2] * v_new_2;
        float h3 = g2 * H_reg[j + 3] + sk2[j + 3] * v_new_2;
        H_reg[j] = h0; H_reg[j+1] = h1; H_reg[j+2] = h2; H_reg[j+3] = h3;
        qd2a += h0 * sq2[j];     qd2b += h1 * sq2[j + 1];
        qd2c += h2 * sq2[j + 2]; qd2d += h3 * sq2[j + 3];
    }
    float q2_dot = (qd2a + qd2b) + (qd2c + qd2d);

    float inv_sqrt_d = rsqrtf((float)k_dim);
    output[((unsigned long long)(b * 3) * num_v_heads + vh) * v_dim + tid] =
        __float2bfloat16(q0_dot * inv_sqrt_d);
    output[((unsigned long long)(b * 3 + 1) * num_v_heads + vh) * v_dim + tid] =
        __float2bfloat16(q1_dot * inv_sqrt_d);
    output[((unsigned long long)(b * 3 + 2) * num_v_heads + vh) * v_dim + tid] =
        __float2bfloat16(q2_dot * inv_sqrt_d);


    #pragma unroll
    for (int j = 0; j < K_DIM; j++) {
        H_global[j * v_dim + tid] = H_reg[j];
    }
}
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
    }

    #ifdef SSM_STATE_NORM_ENABLED
    {
        float local_sq = 0.0f;
        for (unsigned int j = 0; j < k_dim; j++) {
            float hv = H[j * v_dim + tid];
            local_sq += hv * hv;
        }
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





















// 2026-09-25: Fused decode: conv1d update + SiLU (+ L2 norm for q and k), the recurrence, the
// state-norm clamp and the gated RMS norm in one launch; the same code as the kernel of
// this name in common/gated_delta_rule.cu. One block per (key head kh, batch) owns kh and
// its head_repeat value heads, so it is the only writer of their conv_state rows.
// ops::gdn_decode_f32_conv_norm: grid (num_k_heads, batch, 1), block (head_repeat * v_dim,
// 1, 1); thread tid is value head kh * head_repeat + tid / v_dim, column tid % v_dim.
// Requires k_dim == v_dim == 128 and head_repeat == 2 (256 threads): the q/k groups are
// tid >> 7 and warp_sums holds eight warps. ssm_batched_recurrent.rs uses it only when
// nv == 2 * nk and kd == vd == 128.
// Layouts: conv_state [batch, conv_dim, d_conv]; new_input [batch, conv_dim], each row
// [Q (num_k_heads * k_dim) | K | V (num_v_heads * v_dim)]; conv_weight [conv_dim, d_conv];
// conv_bias [conv_dim] or null; gate, beta [batch, num_v_heads]; z_gate and output
// [batch, num_v_heads, v_dim]; norm_weight [v_dim].


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
    __syncthreads();  // 2026-09-25: smem_q / smem_k are complete before the recurrence reads them

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
        __syncthreads();  // 2026-09-25: warp_sums is reused
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
    __syncthreads();  // 2026-09-25: warp_sums is reused
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

