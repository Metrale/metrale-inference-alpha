// SPDX-License-Identifier: MIT OR Apache-2.0

// 2026-09-25: Sigmoid-gated RMS norm for GDN layers: out = x * rsqrt(mean(x^2) + eps) * weight * sigmoid(gate).
//
// `crates/model-layers/src/layers/qwen3_ssm/init.rs` loads these three entry
// points in place of the SiLU-gated `gated_rms_norm`, `gated_rms_norm_f32_input`
// and `gated_rms_norm_prefill` of `common/rms_norm.cu` when
// `config.gdn_norm_sigmoid` is set. The qwen4_exp config parser sets it when
// `output_gate_type` is "sigmoid", as this checkpoint's config.json does; the
// reference model passes `output_gate_type or hidden_act` as the gated norm's
// activation (bench/qwen4_exp/ref/modeling_qwen4_exp.py).
//
// Each kernel is its `common/rms_norm.cu` namesake with the gate term
// `g / (1 + exp(-g))` replaced by `1 / (1 + exp(-g))`. The common file's
// `gated_rms_norm_f32_input_strided` has no twin here: init.rs leaves that
// handle at 0, and the batched decode path then runs the norm per sequence.
//
// Owner: gb10 kernels (qwen3.8-flash-next).
// Invariants:
// - weight scales as plain `weight`, not `1 + weight`.
// - Only whole groups of 4 elements are read and written, so hidden_size and
//   head_dim must be multiples of 4.


#include <cuda_bf16.h>

__device__ __forceinline__ float warp_reduce_sum(float val) {
    for (int offset = 16; offset > 0; offset >>= 1) {
        val += __shfl_xor_sync(0xFFFFFFFF, val, offset);
    }
    return val;
}

__device__ __forceinline__ void unpack_bf16x2(unsigned int packed, float& v0, float& v1) {
    v0 = __bfloat162float(__ushort_as_bfloat16((unsigned short)(packed & 0xFFFF)));
    v1 = __bfloat162float(__ushort_as_bfloat16((unsigned short)(packed >> 16)));
}

__device__ __forceinline__ unsigned int pack_bf16x2(float v0, float v1) {
    unsigned int lo = (unsigned int)__bfloat16_as_ushort(__float2bfloat16(v0));
    unsigned int hi = (unsigned int)__bfloat16_as_ushort(__float2bfloat16(v1));
    return lo | (hi << 16);
}

// 2026-09-25: One block per token (grid.x = num_tokens). x stays in registers
// between the sum-of-squares pass and the output pass, 16 floats per thread,
// so hidden_size must be <= 16 * blockDim.x; ops/norm.rs launches
// min(hidden_size, 1024) threads.

extern "C" __global__ void gated_rms_norm_sigmoid(
    const __nv_bfloat16* __restrict__ input,   // 2026-09-25: [num_tokens, hidden_size]
    const __nv_bfloat16* __restrict__ gate,    // 2026-09-25: [num_tokens, gate_stride], gate_stride >= hidden_size
    const __nv_bfloat16* __restrict__ weight,  // 2026-09-25: [hidden_size]
    __nv_bfloat16* __restrict__ output,         // 2026-09-25: [num_tokens, hidden_size]
    unsigned int hidden_size,
    float eps,
    unsigned int gate_stride,                   // 2026-09-25: elements between gate rows
    unsigned int group_size                     // 2026-09-25: ignored
) {
    (void)group_size;
    unsigned int token = blockIdx.x;
    unsigned int tid = threadIdx.x;

    const __nv_bfloat16* x = input + token * hidden_size;
    const __nv_bfloat16* g = gate + (unsigned long long)token * gate_stride;
    __nv_bfloat16* out = output + token * hidden_size;




    const unsigned int quad_size = hidden_size / 4;
    const unsigned long long* x64 = (const unsigned long long*)x;

    float x_cache[16];
    float sum_sq = 0.0f;
    unsigned int n_cached = 0;

    for (unsigned int i = tid; i < quad_size; i += blockDim.x) {
        unsigned long long v = x64[i];
        float f0, f1, f2, f3;
        unpack_bf16x2((unsigned int)v, f0, f1);
        unpack_bf16x2((unsigned int)(v >> 32), f2, f3);
        x_cache[n_cached]     = f0;
        x_cache[n_cached + 1] = f1;
        x_cache[n_cached + 2] = f2;
        x_cache[n_cached + 3] = f3;
        n_cached += 4;
        sum_sq += f0 * f0 + f1 * f1 + f2 * f2 + f3 * f3;
    }


    sum_sq = warp_reduce_sum(sum_sq);

    __shared__ float warp_sums[32];
    unsigned int warp_id = tid / 32;
    unsigned int lane_id = tid % 32;

    if (lane_id == 0) warp_sums[warp_id] = sum_sq;
    __syncthreads();

    if (warp_id == 0) {
        float val = (lane_id < (blockDim.x + 31) / 32) ? warp_sums[lane_id] : 0.0f;
        val = warp_reduce_sum(val);
        if (lane_id == 0) warp_sums[0] = val;
    }
    __syncthreads();

    float rms = rsqrtf(warp_sums[0] / (float)hidden_size + eps);



    const unsigned long long* g64 = (const unsigned long long*)g;
    const unsigned long long* w64 = (const unsigned long long*)weight;
    unsigned long long* out64 = (unsigned long long*)out;

    unsigned int ci = 0;
    for (unsigned int i = tid; i < quad_size; i += blockDim.x) {
        float f0 = x_cache[ci];
        float f1 = x_cache[ci + 1];
        float f2 = x_cache[ci + 2];
        float f3 = x_cache[ci + 3];
        ci += 4;

        unsigned long long wv = w64[i];
        float w0, w1, w2, w3;
        unpack_bf16x2((unsigned int)wv, w0, w1);
        unpack_bf16x2((unsigned int)(wv >> 32), w2, w3);

        unsigned long long gv = g64[i];
        float g0, g1, g2, g3;
        unpack_bf16x2((unsigned int)gv, g0, g1);
        unpack_bf16x2((unsigned int)(gv >> 32), g2, g3);

        float s0 = 1.0f / (1.0f + expf(-g0));
        float s1 = 1.0f / (1.0f + expf(-g1));
        float s2 = 1.0f / (1.0f + expf(-g2));
        float s3 = 1.0f / (1.0f + expf(-g3));

        unsigned int lo = pack_bf16x2(f0 * rms * w0 * s0, f1 * rms * w1 * s1);
        unsigned int hi = pack_bf16x2(f2 * rms * w2 * s2, f3 * rms * w3 * s3);
        out64[i] = ((unsigned long long)hi << 32) | (unsigned long long)lo;
    }
}

// 2026-09-25: FP32-input variant: input is FP32; gate, weight and output are
// BF16. x is not cached: the output pass reads it from global memory again.

extern "C" __global__ void gated_rms_norm_f32_input_sigmoid(
    const float* __restrict__ input,              // 2026-09-25: [num_tokens, hidden_size]
    const __nv_bfloat16* __restrict__ gate,       // 2026-09-25: [num_tokens, gate_stride]
    const __nv_bfloat16* __restrict__ weight,     // 2026-09-25: [hidden_size]
    __nv_bfloat16* __restrict__ output,            // 2026-09-25: [num_tokens, hidden_size]
    unsigned int hidden_size,
    float eps,
    unsigned int gate_stride,
    unsigned int group_size
) {
    (void)group_size;
    unsigned int token = blockIdx.x;
    unsigned int tid = threadIdx.x;

    const float* x = input + token * hidden_size;
    const __nv_bfloat16* g = gate + (unsigned long long)token * gate_stride;
    __nv_bfloat16* out = output + token * hidden_size;


    float sum_sq = 0.0f;
    for (unsigned int i = tid; i < hidden_size; i += blockDim.x) {
        float f = x[i];
        sum_sq += f * f;
    }

    sum_sq = warp_reduce_sum(sum_sq);
    __shared__ float warp_sums[32];
    unsigned int warp_id = tid / 32;
    unsigned int lane_id = tid % 32;
    if (lane_id == 0) warp_sums[warp_id] = sum_sq;
    __syncthreads();
    if (warp_id == 0) {
        float val = (lane_id < (blockDim.x + 31) / 32) ? warp_sums[lane_id] : 0.0f;
        val = warp_reduce_sum(val);
        if (lane_id == 0) warp_sums[0] = val;
    }
    __syncthreads();

    float rms = rsqrtf(warp_sums[0] / (float)hidden_size + eps);


    const unsigned long long* g64 = (const unsigned long long*)g;
    const unsigned long long* w64 = (const unsigned long long*)weight;
    unsigned long long* out64 = (unsigned long long*)out;

    const unsigned int quad_size = hidden_size / 4;
    for (unsigned int i = tid; i < quad_size; i += blockDim.x) {
        unsigned int base = i * 4;
        float f0 = x[base];
        float f1 = x[base + 1];
        float f2 = x[base + 2];
        float f3 = x[base + 3];

        unsigned long long wv = w64[i];
        float w0, w1, w2, w3;
        unpack_bf16x2((unsigned int)wv, w0, w1);
        unpack_bf16x2((unsigned int)(wv >> 32), w2, w3);

        unsigned long long gv = g64[i];
        float g0, g1, g2, g3;
        unpack_bf16x2((unsigned int)gv, g0, g1);
        unpack_bf16x2((unsigned int)(gv >> 32), g2, g3);

        float s0 = 1.0f / (1.0f + expf(-g0));
        float s1 = 1.0f / (1.0f + expf(-g1));
        float s2 = 1.0f / (1.0f + expf(-g2));
        float s3 = 1.0f / (1.0f + expf(-g3));

        unsigned int lo = pack_bf16x2(f0 * rms * w0 * s0, f1 * rms * w1 * s1);
        unsigned int hi = pack_bf16x2(f2 * rms * w2 * s2, f3 * rms * w3 * s3);
        out64[i] = ((unsigned long long)hi << 32) | (unsigned long long)lo;
    }
}

// 2026-09-25: Prefill variant: one block per (head, token), grid
// (heads_per_token, num_actual_tokens), block min(head_dim, 1024)
// (ops/norm.rs). The register cache matches gated_rms_norm_sigmoid, so
// head_dim must be <= 16 * blockDim.x.

extern "C" __global__ void gated_rms_norm_prefill_sigmoid(
    const __nv_bfloat16* __restrict__ input,
    const __nv_bfloat16* __restrict__ gate,
    const __nv_bfloat16* __restrict__ weight,  // 2026-09-25: [head_dim]
    __nv_bfloat16* __restrict__ output,
    unsigned int head_dim,
    float eps,
    unsigned int input_token_stride,            // 2026-09-25: BF16 elements between tokens in input and output
    unsigned int gate_token_stride              // 2026-09-25: BF16 elements between tokens in gate
) {
    unsigned int head = blockIdx.x;
    unsigned int token = blockIdx.y;
    unsigned int tid = threadIdx.x;

    const __nv_bfloat16* x = input + (unsigned long long)token * input_token_stride + head * head_dim;
    const __nv_bfloat16* g = gate + (unsigned long long)token * gate_token_stride + head * head_dim;
    __nv_bfloat16* out = output + (unsigned long long)token * input_token_stride + head * head_dim;

    const unsigned int quad_size = head_dim / 4;
    const unsigned long long* x64 = (const unsigned long long*)x;

    float x_cache[16];
    float sum_sq = 0.0f;
    unsigned int n_cached = 0;

    for (unsigned int i = tid; i < quad_size; i += blockDim.x) {
        unsigned long long v = x64[i];
        float f0, f1, f2, f3;
        unpack_bf16x2((unsigned int)v, f0, f1);
        unpack_bf16x2((unsigned int)(v >> 32), f2, f3);
        x_cache[n_cached]     = f0;
        x_cache[n_cached + 1] = f1;
        x_cache[n_cached + 2] = f2;
        x_cache[n_cached + 3] = f3;
        n_cached += 4;
        sum_sq += f0 * f0 + f1 * f1 + f2 * f2 + f3 * f3;
    }

    sum_sq = warp_reduce_sum(sum_sq);

    __shared__ float warp_sums[32];
    unsigned int warp_id = tid / 32;
    unsigned int lane_id = tid % 32;

    if (lane_id == 0) warp_sums[warp_id] = sum_sq;
    __syncthreads();

    if (warp_id == 0) {
        float val = (lane_id < (blockDim.x + 31) / 32) ? warp_sums[lane_id] : 0.0f;
        val = warp_reduce_sum(val);
        if (lane_id == 0) warp_sums[0] = val;
    }
    __syncthreads();

    float rms = rsqrtf(warp_sums[0] / (float)head_dim + eps);

    const unsigned long long* g64 = (const unsigned long long*)g;
    const unsigned long long* w64 = (const unsigned long long*)weight;
    unsigned long long* out64 = (unsigned long long*)out;

    unsigned int ci = 0;
    for (unsigned int i = tid; i < quad_size; i += blockDim.x) {
        float f0 = x_cache[ci];
        float f1 = x_cache[ci + 1];
        float f2 = x_cache[ci + 2];
        float f3 = x_cache[ci + 3];
        ci += 4;

        unsigned long long wv = w64[i];
        float w0, w1, w2, w3;
        unpack_bf16x2((unsigned int)wv, w0, w1);
        unpack_bf16x2((unsigned int)(wv >> 32), w2, w3);

        unsigned long long gv = g64[i];
        float g0, g1, g2, g3;
        unpack_bf16x2((unsigned int)gv, g0, g1);
        unpack_bf16x2((unsigned int)(gv >> 32), g2, g3);

        float s0 = 1.0f / (1.0f + expf(-g0));
        float s1 = 1.0f / (1.0f + expf(-g1));
        float s2 = 1.0f / (1.0f + expf(-g2));
        float s3 = 1.0f / (1.0f + expf(-g3));

        unsigned int lo = pack_bf16x2(f0 * rms * w0 * s0, f1 * rms * w1 * s1);
        unsigned int hi = pack_bf16x2(f2 * rms * w2 * s2, f3 * rms * w3 * s3);
        out64[i] = ((unsigned long long)hi << 32) | (unsigned long long)lo;
    }
}
