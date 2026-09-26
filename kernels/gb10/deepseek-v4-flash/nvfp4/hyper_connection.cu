// SPDX-License-Identifier: MIT OR Apache-2.0

// 2026-09-25: DeepSeek-V4 manifold-constrained hyper-connection (mHC) kernels: hc_expand, hc_pre, hc_post, hc_head.
//
// Owner: gb10 kernels (deepseek-v4-flash).
// Invariants:
// - The stream state is FP32 [T, hc_mult, H], stream-major per token; the HC parameters are FP32; the
//   collapsed and sublayer hidden states are BF16 [T, H].
// - One HC_BLOCK (256) thread block per token, as ops::hyper_connection launches them; hc_mult is at most
//   HC_MAX_MULT (4), unchecked, and mix_hc = (2 + hc_mult) * hc_mult.
// - hc_pre's pre / post / comb split and its softmax-plus-Sinkhorn follow hc_split_sinkhorn in the
//   checkpoint's inference/kernel.py, plus one eps-free column normalisation (see hc_pre).







#include <cuda_bf16.h>

#define HC_BLOCK 256
#define HC_MAX_MULT 4
#define HC_MAX_MIX 24

// 2026-09-25: Block-wide sum over red[0..HC_BLOCK); every one of the HC_BLOCK threads must call it.
__device__ __forceinline__ float hc_block_reduce(float* red, unsigned int tid) {
    for (unsigned int s = HC_BLOCK / 2; s > 0; s >>= 1) {
        if (tid < s) red[tid] += red[tid + s];
        __syncthreads();
    }
    return red[0];
}

// 2026-09-25: hc_expand: streams[t, i, d] = hidden[t, d] for every i < hc_mult.


extern "C" __global__ void hc_expand(
    const __nv_bfloat16* __restrict__ hidden,
    float* __restrict__ streams,
    const unsigned int hidden_size,
    const unsigned int hc_mult
) {
    const unsigned int t = blockIdx.x;
    const unsigned int tid = threadIdx.x;
    const unsigned int H = hidden_size;
    const __nv_bfloat16* x = hidden + (size_t)t * H;
    float* s = streams + (size_t)t * hc_mult * H;
    for (unsigned int d = tid; d < H; d += HC_BLOCK) {
        float v = (float)x[d];
        for (unsigned int i = 0; i < hc_mult; ++i) s[i * H + d] = v;
    }
}

// 2026-09-25: hc_pre: streams [T, hc, H] -> y_out [T, H] (the pre-weighted collapse), post_out [T, hc] and
// comb_out [T, hc, hc]. hc_fn is [mix_hc, hc * H], hc_scale [3], hc_base [mix_hc]; the mixes are the hc_fn
// rows dotted with the streams, times the reciprocal RMS of the whole hc * H vector.
extern "C" __global__ void hc_pre(
    const float* __restrict__ streams,
    const float* __restrict__ hc_fn,
    const float* __restrict__ hc_scale,
    const float* __restrict__ hc_base,
    __nv_bfloat16* __restrict__ y_out,
    float* __restrict__ post_out,
    float* __restrict__ comb_out,
    const unsigned int hidden_size,
    const unsigned int hc_mult,
    const unsigned int sinkhorn_iters,
    const float norm_eps,
    const float hc_eps
) {
    const unsigned int t = blockIdx.x;
    const unsigned int tid = threadIdx.x;
    const unsigned int H = hidden_size;
    const unsigned int hc = hc_mult;
    const unsigned int hc_dim = hc * H;
    const unsigned int mix_hc = (2 + hc) * hc;

    const float* x = streams + (size_t)t * hc_dim;

    __shared__ float red[HC_BLOCK];
    __shared__ float s_rsqrt;
    __shared__ float s_mix[HC_MAX_MIX];
    __shared__ float s_pre[HC_MAX_MULT];


    float ss = 0.f;
    for (unsigned int k = tid; k < hc_dim; k += HC_BLOCK) {
        float v = (float)x[k];
        ss += v * v;
    }
    red[tid] = ss;
    __syncthreads();
    float ssum = hc_block_reduce(red, tid);
    if (tid == 0) s_rsqrt = rsqrtf(ssum / (float)hc_dim + norm_eps);
    __syncthreads();
    const float rsqrt = s_rsqrt;


    for (unsigned int m = 0; m < mix_hc; ++m) {
        const float* fn_row = hc_fn + (size_t)m * hc_dim;
        float acc = 0.f;
        for (unsigned int k = tid; k < hc_dim; k += HC_BLOCK) {
            acc += fn_row[k] * (float)x[k];
        }
        red[tid] = acc;
        __syncthreads();
        float r = hc_block_reduce(red, tid);
        if (tid == 0) s_mix[m] = r * rsqrt;
        __syncthreads();
    }


    if (tid == 0) {
        float comb[HC_MAX_MULT * HC_MAX_MULT];
        for (unsigned int i = 0; i < hc; ++i) {
            float pr = s_mix[i] * hc_scale[0] + hc_base[i];
            s_pre[i] = 1.f / (1.f + expf(-pr)) + hc_eps;
            float po = s_mix[hc + i] * hc_scale[1] + hc_base[hc + i];
            post_out[(size_t)t * hc + i] = 2.f * (1.f / (1.f + expf(-po)));
        }
        for (unsigned int i = 0; i < hc; ++i)
            for (unsigned int j = 0; j < hc; ++j)
                comb[i * hc + j] =
                    s_mix[2 * hc + i * hc + j] * hc_scale[2] + hc_base[2 * hc + i * hc + j];

        for (unsigned int i = 0; i < hc; ++i) {
            float mx = -1e30f;
            for (unsigned int j = 0; j < hc; ++j) mx = fmaxf(mx, comb[i * hc + j]);
            float sum = 0.f;
            for (unsigned int j = 0; j < hc; ++j) {
                float e = expf(comb[i * hc + j] - mx);
                comb[i * hc + j] = e;
                sum += e;
            }
            for (unsigned int j = 0; j < hc; ++j) comb[i * hc + j] = comb[i * hc + j] / sum + hc_eps;
        }

        for (unsigned int j = 0; j < hc; ++j) {
            float c = hc_eps;
            for (unsigned int i = 0; i < hc; ++i) c += comb[i * hc + j];
            for (unsigned int i = 0; i < hc; ++i) comb[i * hc + j] /= c;
        }

        for (unsigned int it = 0; it + 1 < sinkhorn_iters; ++it) {
            for (unsigned int i = 0; i < hc; ++i) {
                float r = hc_eps;
                for (unsigned int j = 0; j < hc; ++j) r += comb[i * hc + j];
                for (unsigned int j = 0; j < hc; ++j) comb[i * hc + j] /= r;
            }
            for (unsigned int j = 0; j < hc; ++j) {
                float c = hc_eps;
                for (unsigned int i = 0; i < hc; ++i) c += comb[i * hc + j];
                for (unsigned int i = 0; i < hc; ++i) comb[i * hc + j] /= c;
            }
        }
        // 2026-09-25: An extra column normalisation without eps, which the reference does not have: hc_post
        // mixes out[j] = sum_i comb[i][j] * res[i], so column sums of exactly 1 make each output stream a
        // convex combination of the residual streams. Measured 2026-07-05: dropping this pass cut the
        // coherent-output length from about 150 to about 90 tokens.











        for (unsigned int j = 0; j < hc; ++j) {
            float c = 0.f;
            for (unsigned int i = 0; i < hc; ++i) c += comb[i * hc + j];
            float inv = (c > 0.f) ? (1.f / c) : 0.f;
            for (unsigned int i = 0; i < hc; ++i) comb[i * hc + j] *= inv;
        }
        for (unsigned int i = 0; i < hc; ++i)
            for (unsigned int j = 0; j < hc; ++j)
                comb_out[(size_t)t * hc * hc + i * hc + j] = comb[i * hc + j];
    }
    __syncthreads();


    for (unsigned int d = tid; d < H; d += HC_BLOCK) {
        float acc = 0.f;
        for (unsigned int i = 0; i < hc; ++i) acc += s_pre[i] * (float)x[i * H + d];
        y_out[(size_t)t * H + d] = __float2bfloat16(acc);
    }
}

// 2026-09-25: hc_post: out[t, j, d] = post[t, j] * block_out[t, d] + sum_i comb[t, i, j] * residual[t, i, d].
// out may alias residual: each thread reads all hc residual values of its column d before writing them.


extern "C" __global__ void hc_post(
    const __nv_bfloat16* __restrict__ block_out,
    const float* __restrict__ residual,
    const float* __restrict__ post,
    const float* __restrict__ comb,
    float* __restrict__ out,
    const unsigned int hidden_size,
    const unsigned int hc_mult
) {
    const unsigned int t = blockIdx.x;
    const unsigned int tid = threadIdx.x;
    const unsigned int H = hidden_size;
    const unsigned int hc = hc_mult;

    const __nv_bfloat16* x = block_out + (size_t)t * H;
    const float* res = residual + (size_t)t * hc * H;
    const float* p = post + (size_t)t * hc;
    const float* c = comb + (size_t)t * hc * hc;
    float* o = out + (size_t)t * hc * H;

    for (unsigned int d = tid; d < H; d += HC_BLOCK) {
        float xd = (float)x[d];
        float rv[HC_MAX_MULT];
        for (unsigned int i = 0; i < hc; ++i) rv[i] = res[i * H + d];
        for (unsigned int j = 0; j < hc; ++j) {
            float acc = p[j] * xd;
            for (unsigned int i = 0; i < hc; ++i) acc += c[i * hc + j] * rv[i];
            o[j * H + d] = acc;
        }
    }
}

// 2026-09-25: hc_head: streams [T, hc, H] -> y_out [T, H], weighted per stream by sigmoid(mix * head_scale
// + head_base) + hc_eps, where the mixes are head_fn [hc, hc * H] rows dotted with the RMS-normalised streams.

extern "C" __global__ void hc_head(
    const float* __restrict__ streams,
    const float* __restrict__ head_fn,
    const float* __restrict__ head_scale,
    const float* __restrict__ head_base,
    __nv_bfloat16* __restrict__ y_out,
    const unsigned int hidden_size,
    const unsigned int hc_mult,
    const float norm_eps,
    const float hc_eps
) {
    const unsigned int t = blockIdx.x;
    const unsigned int tid = threadIdx.x;
    const unsigned int H = hidden_size;
    const unsigned int hc = hc_mult;
    const unsigned int hc_dim = hc * H;

    const float* x = streams + (size_t)t * hc_dim;

    __shared__ float red[HC_BLOCK];
    __shared__ float s_rsqrt;
    __shared__ float s_pre[HC_MAX_MULT];

    float ss = 0.f;
    for (unsigned int k = tid; k < hc_dim; k += HC_BLOCK) {
        float v = (float)x[k];
        ss += v * v;
    }
    red[tid] = ss;
    __syncthreads();
    float ssum = hc_block_reduce(red, tid);
    if (tid == 0) s_rsqrt = rsqrtf(ssum / (float)hc_dim + norm_eps);
    __syncthreads();
    const float rsqrt = s_rsqrt;
    const float scale = head_scale[0];

    for (unsigned int m = 0; m < hc; ++m) {
        const float* fn_row = head_fn + (size_t)m * hc_dim;
        float acc = 0.f;
        for (unsigned int k = tid; k < hc_dim; k += HC_BLOCK) {
            acc += fn_row[k] * (float)x[k];
        }
        red[tid] = acc;
        __syncthreads();
        float r = hc_block_reduce(red, tid);
        if (tid == 0) {
            float v = r * rsqrt * scale + head_base[m];
            s_pre[m] = 1.f / (1.f + expf(-v)) + hc_eps;
        }
        __syncthreads();
    }

    for (unsigned int d = tid; d < H; d += HC_BLOCK) {
        float acc = 0.f;
        for (unsigned int i = 0; i < hc; ++i) acc += s_pre[i] * (float)x[i * H + d];
        y_out[(size_t)t * H + d] = __float2bfloat16(acc);
    }
}
