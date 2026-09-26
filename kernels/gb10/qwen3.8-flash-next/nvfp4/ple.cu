// SPDX-License-Identifier: MIT OR Apache-2.0

// 2026-09-25: PLE: hashed n-gram features gated into the FP32 hyper-connection highway (Qwen3.8-Flash-Next).
// Reference: `Qwen4ExpTextPLELayer.forward` in bench/qwen4_exp/ref/modeling_qwen4_exp.py.
// It runs on the layers listed in `ple_layer_ids`, which the reference reads as one-indexed.
//
//   key_normed   = norm_key(key_proj(emb)).unflatten(-1, (hc, H))
//   value        = value_proj(emb)                            # [T, H]
//   query_normed = norm_query(hidden).unflatten(-1, (hc, H))  # hidden [T, hc*H]
//   gate  = (key_normed * query_normed).sum(-1) / sqrt(H)     # [T, hc]
//   gate  = gate.abs().clamp_min(1e-6).sqrt() * gate.sign()   # signed sqrt
//   gated = sigmoid(gate) * value.unsqueeze(-2)               # [T, hc, H]
//   out   = gated.flatten(-2) + silu(conv1d(norm_conv(gated.flatten(-2))))
//
// Three details the kernels follow:
//  1. All three norms are `Qwen4ExpTextRMSNorm`, `normed * (1.0 + weight)` with
//     zero-initialised weights, not `normed * weight` (bench/qwen4_exp/
//     ARCHITECTURE.md §6).
//  2. They are grouped, `group_size = hidden_size`: each of the `hc` streams is
//     normalised over its own H-wide slice of the `hc*H` vector.
//  3. The gate takes a signed square root before the sigmoid.
//
// The conv is depthwise (`groups = hc*H`) with kernel `ple_conv_kernel_size` and
// dilation `ngram_size` (4 and 3 in this checkpoint's config.json), so it carries
// (K-1)*D steps of state. `common/causal_conv1d.cu` takes no dilation.
//
// Owner: gb10 kernels (qwen3.8-flash-next).
// Invariants:
// - ple_gate needs blockDim.x == PLE_BLOCK (ops/ple.rs launches 256) and
//   hc <= PLE_MAX_STREAMS (its `smem_gate` array).
// - ple_conv needs (k_size - 1) * dilation <= 16 (its `carry` array).



// 2026-09-25: Precision: PLE's output is added to the FP32 highway, so every
// intermediate stays FP32 rather than rounding to BF16 on the way. Only `key`
// and `value` (the projection outputs) and the weights are BF16.













#include <cuda_bf16.h>
#include <cuda_runtime.h>

#define PLE_BLOCK 256
#define PLE_MAX_STREAMS 8

__device__ __forceinline__ float ple_silu(float v) {
    return v / (1.0f + __expf(-v));
}

__device__ __forceinline__ float ple_sigmoid(float v) {
    return 1.0f / (1.0f + __expf(-v));
}

// 2026-09-25: Block-wide sum of `val`, returned to every thread. Needs blockDim.x == PLE_BLOCK.
__device__ __forceinline__ float ple_block_sum(float val, float* smem_red) {
    const unsigned int tid = threadIdx.x;
    const unsigned int lane = tid & 31u;
    const unsigned int warp = tid >> 5;
    const unsigned int warps = PLE_BLOCK / 32;
    #pragma unroll
    for (int off = 16; off > 0; off >>= 1) {
        val += __shfl_down_sync(0xFFFFFFFFu, val, off);
    }
    if (lane == 0) smem_red[warp] = val;
    __syncthreads();
    float tot = 0.0f;
    for (unsigned int w = 0; w < warps; ++w) tot += smem_red[w];
    __syncthreads();
    return tot;
}

// 2026-09-25: ple_gate: one block per token. Computes the per-stream gate,
// applies it to `value`, and writes both the gated value (which the final add
// uses un-normalised) and its norm_conv output (which ple_conv consumes).
// `hidden` is the FP32 highway; `key` and `value` are BF16 projection outputs.
// Everything accumulates in FP32.


extern "C" __global__ void ple_gate(
    const float* __restrict__ hidden,          // 2026-09-25: [T, hc*H] FP32 highway
    const __nv_bfloat16* __restrict__ key,     // 2026-09-25: [T, hc*H] key_proj output
    const __nv_bfloat16* __restrict__ value,   // 2026-09-25: [T, H] value_proj output
    const __nv_bfloat16* __restrict__ norm_query_w, // 2026-09-25: [hc*H]
    const __nv_bfloat16* __restrict__ norm_key_w,   // 2026-09-25: [hc*H]
    const __nv_bfloat16* __restrict__ norm_conv_w,  // 2026-09-25: [hc*H]
    float* __restrict__ gated_out,             // 2026-09-25: [T, hc*H] FP32
    float* __restrict__ gated_normed,          // 2026-09-25: [T, hc*H] FP32
    const unsigned int hidden_size,
    const unsigned int hc,
    const float eps
) {
    const unsigned int t = blockIdx.x;
    const unsigned int tid = threadIdx.x;
    const unsigned int H = hidden_size;
    const unsigned int hc_dim = hc * H;

    const float* q = hidden + (size_t)t * hc_dim;
    const __nv_bfloat16* k = key + (size_t)t * hc_dim;
    const __nv_bfloat16* v = value + (size_t)t * H;
    float* g_out = gated_out + (size_t)t * hc_dim;
    float* gn_out = gated_normed + (size_t)t * hc_dim;

    __shared__ float smem_red[PLE_BLOCK / 32];
    __shared__ float smem_gate[PLE_MAX_STREAMS];

        // 2026-09-25: Per stream: the RMS of query and key in one pass over the
        // H-wide slice, then the dot product of the normed values in a second.

    for (unsigned int s = 0; s < hc; ++s) {
        const float* qs = q + (size_t)s * H;
        const __nv_bfloat16* ks = k + (size_t)s * H;
        float sq = 0.0f, sk = 0.0f;
        for (unsigned int d = tid; d < H; d += PLE_BLOCK) {
            float qv = qs[d];
            float kv = (float)ks[d];
            sq += qv * qv;
            sk += kv * kv;
        }
        float rq = ple_block_sum(sq, smem_red);
        float rk = ple_block_sum(sk, smem_red);
        rq = rsqrtf(rq / (float)H + eps);
        rk = rsqrtf(rk / (float)H + eps);

        // 2026-09-25: dot(norm_query(q), norm_key(k)) over this stream, with the
        // `1 + weight` convention of `Qwen4ExpTextRMSNorm`.
        float dot = 0.0f;
        for (unsigned int d = tid; d < H; d += PLE_BLOCK) {
            const unsigned int i = s * H + d;
            float qn = qs[d] * rq * (1.0f + (float)norm_query_w[i]);
            float kn = (float)ks[d] * rk * (1.0f + (float)norm_key_w[i]);
            dot += qn * kn;
        }
        dot = ple_block_sum(dot, smem_red);
        if (tid == 0) {
            float gate = dot * rsqrtf((float)H);
            // 2026-09-25: Signed sqrt: sign(g) * sqrt(max(|g|, 1e-6)), and sign(0) = 0.
            float mag = fabsf(gate);
            mag = mag < 1e-6f ? 1e-6f : mag;
            float sgn = (gate > 0.0f) ? 1.0f : ((gate < 0.0f) ? -1.0f : 0.0f);
            smem_gate[s] = ple_sigmoid(sgn * sqrtf(mag));
        }
        __syncthreads();
    }


    for (unsigned int i = tid; i < hc_dim; i += PLE_BLOCK) {
        float gv = smem_gate[i / H] * (float)v[i % H];
        g_out[i] = gv;
    }
    __syncthreads();

    // 2026-09-25: norm_conv of the gated value, per stream, with the same `1 + weight` convention.
    for (unsigned int s = 0; s < hc; ++s) {
        float acc = 0.0f;
        for (unsigned int d = tid; d < H; d += PLE_BLOCK) {
            float x = g_out[s * H + d];
            acc += x * x;
        }
        float r = ple_block_sum(acc, smem_red);
        r = rsqrtf(r / (float)H + eps);
        for (unsigned int d = tid; d < H; d += PLE_BLOCK) {
            const unsigned int i = s * H + d;
            float x = g_out[i];
            gn_out[i] = x * r * (1.0f + (float)norm_conv_w[i]);
        }
        __syncthreads();
    }
}

// 2026-09-25: ple_conv: depthwise causal conv1d over C = hc*H channels, kernel
// K = k_size, dilation D = dilation, over `num_tokens` new steps with
// `state_len = (K-1)*D` carried steps in front.
//
//   y[t][c] = sum_{j<K} w[c][j] * x[t - (K-1-j)*D][c]
//   out[t][c] = gated[t][c] + silu(y[t][c])
//
// `state` holds the last `state_len` rows of the previous call's normed input,
// and is updated in place to the last `state_len` rows of `state ++ x`, so
// prefill and decode run the same code.
//
// One thread per channel, looping over the tokens. Channels are the fast axis,
// so the loads coalesce.

extern "C" __global__ void ple_conv(
    const float* __restrict__ x,               // 2026-09-25: [T, C] norm_conv output, FP32
    const float* __restrict__ gated,           // 2026-09-25: [T, C] un-normalised, FP32
    const __nv_bfloat16* __restrict__ weight,  // 2026-09-25: [C, K] depthwise
    float* __restrict__ state,                 // 2026-09-25: [state_len, C] FP32, read and written
    float* __restrict__ out,                   // 2026-09-25: [T, C] FP32
    const unsigned int num_tokens,
    const unsigned int channels,
    const unsigned int k_size,
    const unsigned int dilation
) {
    const unsigned int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= channels) return;
    const unsigned int state_len = (k_size - 1) * dilation;

    for (unsigned int t = 0; t < num_tokens; ++t) {
        float acc = 0.0f;
        for (unsigned int j = 0; j < k_size; ++j) {
            // 2026-09-25: Offset back in time for tap j; j = K-1 is the current step.
            const int back = (int)((k_size - 1 - j) * dilation);
            const int src = (int)t - back;
            float xv;
            if (src >= 0) {
                xv = x[(size_t)src * channels + c];
            } else {
                // 2026-09-25: Into the carried state at `state_len + src`, which is
                // >= 0 because back <= state_len.
                const int si = (int)state_len + src;
                xv = (si >= 0) ? state[(size_t)si * channels + c] : 0.0f;
            }
            acc += xv * (float)weight[(size_t)c * k_size + j];
        }
        float base = gated[(size_t)t * channels + c];
        out[(size_t)t * channels + c] = base + ple_silu(acc);
    }

    // 2026-09-25: Roll the state to the last `state_len` rows of `state ++ x`.
    // Every value is read before any is written, so a short `x` that overlaps
    // the tail of the old state cannot overwrite its own source.
    float carry[16];
    const unsigned int keep = state_len;
    for (unsigned int i = 0; i < keep; ++i) {


        const int gi = (int)(num_tokens + state_len - keep + i);
        const int xi = gi - (int)state_len;
        carry[i] = (xi >= 0) ? x[(size_t)xi * channels + c]
                             : state[(size_t)gi * channels + c];
    }
    for (unsigned int i = 0; i < keep; ++i) {
        state[(size_t)i * channels + c] = carry[i];
    }
}

// 2026-09-25: ple_add_highway: `hidden_states = hidden_states + ple(...)` on the
// FP32 highway. The reference adds it before that layer's attention
// hyper-connection.
extern "C" __global__ void ple_add_highway(
    const float* __restrict__ ple_out,         // 2026-09-25: [T, C] FP32
    float* __restrict__ hidden,                // 2026-09-25: [T, C] FP32 highway, read and written
    const unsigned int n
) {
    const unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) hidden[i] += ple_out[i];
}
