// SPDX-License-Identifier: MIT OR Apache-2.0

// 2026-09-25: Kernels `gated_delta_rule_decode_f32_norm_snap` and `gated_delta_rule_decode_f32_strided_norm_snap`:
// the fused-norm GDN decode kernels of this directory's gated_delta_rule.cu with one addition, an `h_inter` output.
// Every value stored to H, including the clamp's rescale, is also stored to h_inter: the per-token h-state snapshot
// that exact verify rolls back to. h_inter == nullptr skips those stores; the caller passes null for the last token
// (crates/model-layers/src/layers/qwen3_ssm/trait_decode_batched_conv_gdn_exact.rs).
//
// With the snapshot stores removed, each kernel's code is its parent's, and both files build under this directory's
// --fmad=false. examples/verify_exact_microtest compares H, the outputs and every snapshot byte for byte with the
// sequential chain that runs the parent kernel.
//
// Only kernels/gb10/qwen3.6-27b/nvfp4 defines these (qwen3.8-27b compiles this tree through kernel_source). Where the
// handle is 0 the caller runs the parent kernel and copies the snapshot (crates/model-layers/src/layers/ops/
// ssm_gdn_snap.rs).
//
// Owner: gb10 kernels (qwen3.6-27b).
// Invariants: none beyond the types.













#include <cuda_bf16.h>

#ifndef GDN_MS_HELPERS
#define GDN_MS_HELPERS
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

// 2026-09-25: The same clamp configuration as gated_delta_rule.cu; a different SSM_STATE_MAX_NORM here would break
// the bit match with the parent kernels.

#ifndef SSM_STATE_NORM_ENABLED
#define SSM_STATE_NORM_ENABLED
#define SSM_STATE_MAX_NORM 1000.0f
#endif
#endif

// 2026-09-25: `gated_delta_rule_decode_f32_norm` plus the snapshot. h_inter has h_state's layout,
// [batch, num_v_heads, k_dim, v_dim] FP32, or is nullptr.

extern "C" __global__ void gated_delta_rule_decode_f32_norm_snap(
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
    float eps,
    float* __restrict__ h_inter
) {
    const unsigned int vh = blockIdx.x;
    const unsigned int b = blockIdx.y;
    if (vh >= num_v_heads || b >= batch_size) return;

    const unsigned int tid = threadIdx.x;
    if (tid >= v_dim) return;

    const unsigned int head_repeat = num_v_heads / num_k_heads;
    const unsigned int kh = vh / head_repeat;

    float* H = h_state + ((b * num_v_heads + vh) * k_dim * v_dim);

    float* HI = (h_inter != nullptr)
        ? h_inter + ((b * num_v_heads + vh) * k_dim * v_dim)
        : nullptr;
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
        // 2026-09-25: Snapshot: the values just stored to H.
        if (HI != nullptr) {
            HI[(j + 0) * v_dim + tid] = h0;
            HI[(j + 1) * v_dim + tid] = h1;
            HI[(j + 2) * v_dim + tid] = h2;
            HI[(j + 3) * v_dim + tid] = h3;
        }
        q_dot += h0 * smem_q[j] + h1 * smem_q[j+1] + h2 * smem_q[j+2] + h3 * smem_q[j+3];
#ifdef SSM_STATE_NORM_ENABLED
        // 2026-09-25: The squared norm for the clamp, accumulated as in the parent kernel.


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
                // 2026-09-25: The parent's read-scale-store; the snapshot gets the same rescaled value.


                float hv = H[j * v_dim + tid] * scale;
                H[j * v_dim + tid] = hv;
                if (HI != nullptr) HI[j * v_dim + tid] = hv;
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

// 2026-09-25: `gated_delta_rule_decode_f32_strided_norm` plus the snapshot, for batch_size sequences. h_inter is
// sequence 0's snapshot for this token position, and sequence b's is b * h_inter_seq_stride FP32 elements further;
// the caller takes the stride from the pool's per-slot snapshot layout
// (qwen3_ssm/trait_decode_batched_conv_gdn_multi_exact.rs). nullptr skips the snapshot.


extern "C" __global__ void gated_delta_rule_decode_f32_strided_norm_snap(
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
    float eps,
    float* __restrict__ h_inter,
    unsigned long long h_inter_seq_stride
) {
    const unsigned int vh = blockIdx.x;
    const unsigned int b = blockIdx.y;
    if (vh >= num_v_heads || b >= batch_size) return;

    const unsigned int tid = threadIdx.x;
    if (tid >= v_dim) return;

    const unsigned int head_repeat = num_v_heads / num_k_heads;
    const unsigned int kh = vh / head_repeat;

    float* H = h_state + ((b * num_v_heads + vh) * k_dim * v_dim);
    float* HI = (h_inter != nullptr)
        ? h_inter + (unsigned long long)b * h_inter_seq_stride + vh * k_dim * v_dim
        : nullptr;
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
        // 2026-09-25: Snapshot: the values just stored to H.
        if (HI != nullptr) {
            HI[(j + 0) * v_dim + tid] = h0;
            HI[(j + 1) * v_dim + tid] = h1;
            HI[(j + 2) * v_dim + tid] = h2;
            HI[(j + 3) * v_dim + tid] = h3;
        }
        q_dot += h0 * smem_q[j] + h1 * smem_q[j+1] + h2 * smem_q[j+2] + h3 * smem_q[j+3];
#ifdef SSM_STATE_NORM_ENABLED
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
                float hv = H[j * v_dim + tid] * scale;
                H[j * v_dim + tid] = hv;
                if (HI != nullptr) HI[j * v_dim + tid] = hv;
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
