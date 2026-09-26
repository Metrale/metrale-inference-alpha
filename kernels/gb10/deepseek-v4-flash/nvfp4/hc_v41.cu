// SPDX-License-Identifier: MIT OR Apache-2.0
// provenance-id: 526f6e616c6420522e205374657369616b

// 2026-09-25: DeepSeek-V4.1 Flash hyper-connections in the delayed-mix form. A block's attention collapses
// with the pre that its predecessor's FFN mixes produced, and its FFN with the pre of its own attention
// mixes (the CPU reference deepseek_v41_ref::hc does the same), so a site's mixes and the collapse that
// uses them are separate kernels.
//
// Owner: gb10 kernels (deepseek-v4-flash, built for deepseek-v4.1-flash through kernel_source).
// Invariants:
// - Streams are FP32 [T, hc, H]; pre and post are FP32 [T, hc], comb FP32 [T, hc, hc] row-major [j][k];
//   collapsed outputs are BF16 [T, H]. hc_mult is at most HCV_MAX_HC (4), unchecked.
// - The mixes are the mix_hc = (2 + hc) * hc rows of hc_fn dotted with the flattened hc * H stream, times
//   its reciprocal RMS. pre = sigmoid(m * s0 + b) + eps, post = 2 sigmoid(m * s1 + b), comb = row softmax
//   + eps, then a column normalisation and (row, column) x (sinkhorn_iters - 1), each dividing by sum + eps.







#include <cuda_bf16.h>

#include <math_constants.h>
#define HCV_BLOCK 256
#define HCV_MAX_HC 4
#define HCV_MAX_MIX 24

__device__ __forceinline__ float hcv_sigmoid(float x) { return 1.0f / (1.0f + expf(-x)); }

__device__ __forceinline__ float hcv_block_sum(float v, float* red) {
    const unsigned int tid = threadIdx.x;
    red[tid] = v;
    __syncthreads();
    for (unsigned int s = HCV_BLOCK / 2; s > 0; s >>= 1) {
        if (tid < s) red[tid] += red[tid + s];
        __syncthreads();
    }
    const float r = red[0];
    __syncthreads();
    return r;
}

// 2026-09-25: hc_v41_mixes: one 256-thread block per token (blockIdx.x) computes the mixes; thread 0 runs the finish.
extern "C" __global__ void hc_v41_mixes(
    const float* __restrict__ streams,
    const float* __restrict__ hc_fn,
    const float* __restrict__ hc_scale,
    const float* __restrict__ hc_base,
    float* __restrict__ pre,
    float* __restrict__ post,
    float* __restrict__ comb,
    const unsigned int hidden_size,
    const unsigned int hc_mult,
    const unsigned int sinkhorn_iters,
    const float norm_eps,
    const float hc_eps) {
    const unsigned int t = blockIdx.x;
    const unsigned int tid = threadIdx.x;
    const unsigned int hc = hc_mult;
    const unsigned int n = hc * hidden_size;
    const unsigned int mix_hc = (2 + hc) * hc;
    __shared__ float red[HCV_BLOCK];
    __shared__ float mixes[HCV_MAX_MIX];
    __shared__ float c[HCV_MAX_HC * HCV_MAX_HC];
    const float* x = streams + (size_t)t * n;

    float ss = 0.0f;
    for (unsigned int i = tid; i < n; i += HCV_BLOCK) ss += x[i] * x[i];
    ss = hcv_block_sum(ss, red);
    const float rsq = rsqrtf(ss / (float)n + norm_eps);
    for (unsigned int m = 0; m < mix_hc; ++m) {
        const float* w = hc_fn + (size_t)m * n;
        float acc = 0.0f;
        for (unsigned int i = tid; i < n; i += HCV_BLOCK) acc += w[i] * x[i];
        acc = hcv_block_sum(acc, red);
        if (tid == 0) mixes[m] = acc * rsq;
    }
    __syncthreads();
    if (tid == 0) {
        for (unsigned int j = 0; j < hc; ++j) {
            pre[(size_t)t * hc + j] = hcv_sigmoid(mixes[j] * hc_scale[0] + hc_base[j]) + hc_eps;
            post[(size_t)t * hc + j] = 2.0f * hcv_sigmoid(mixes[hc + j] * hc_scale[1] + hc_base[hc + j]);
        }
        for (unsigned int i = 0; i < hc * hc; ++i) c[i] = mixes[2 * hc + i] * hc_scale[2] + hc_base[2 * hc + i];

        for (unsigned int j = 0; j < hc; ++j) {
            float m = -CUDART_INF_F;
            for (unsigned int k = 0; k < hc; ++k) m = fmaxf(m, c[j * hc + k]);
            float sum = 0.0f;
            for (unsigned int k = 0; k < hc; ++k) { c[j * hc + k] = expf(c[j * hc + k] - m); sum += c[j * hc + k]; }
            for (unsigned int k = 0; k < hc; ++k) c[j * hc + k] = c[j * hc + k] / sum + hc_eps;
        }

        for (unsigned int it = 0; it < sinkhorn_iters; ++it) {
            if (it > 0) {
                for (unsigned int j = 0; j < hc; ++j) {
                    float s = 0.0f;
                    for (unsigned int k = 0; k < hc; ++k) s += c[j * hc + k];
                    for (unsigned int k = 0; k < hc; ++k) c[j * hc + k] /= s + hc_eps;
                }
            }
            for (unsigned int k = 0; k < hc; ++k) {
                float s = 0.0f;
                for (unsigned int j = 0; j < hc; ++j) s += c[j * hc + k];
                for (unsigned int j = 0; j < hc; ++j) c[j * hc + k] /= s + hc_eps;
            }
        }
        for (unsigned int i = 0; i < hc * hc; ++i) comb[(size_t)t * hc * hc + i] = c[i];
    }
}

// 2026-09-25: hc_v41_mixes_dot: block (t, m) recomputes token t's RMS and writes mixes_out[t][m], one mix
// per block. Grid (T, mix_hc), block 256 (deepseek_v41_layer/hc_launch.rs).


extern "C" __global__ void hc_v41_mixes_dot(
    const float* __restrict__ streams,
    const float* __restrict__ hc_fn,
    float* __restrict__ mixes_out,
    const unsigned int hidden_size,
    const unsigned int hc_mult,
    const float norm_eps) {
    const unsigned int t = blockIdx.x;
    const unsigned int m = blockIdx.y;
    const unsigned int tid = threadIdx.x;
    const unsigned int hc = hc_mult;
    const unsigned int n = hc * hidden_size;
    const unsigned int mix_hc = (2 + hc) * hc;
    __shared__ float red[HCV_BLOCK];
    const float* x = streams + (size_t)t * n;
    float ss = 0.0f;
    for (unsigned int i = tid; i < n; i += HCV_BLOCK) ss += x[i] * x[i];
    ss = hcv_block_sum(ss, red);
    const float rsq = rsqrtf(ss / (float)n + norm_eps);
    const float* w = hc_fn + (size_t)m * n;
    float acc = 0.0f;
    for (unsigned int i = tid; i < n; i += HCV_BLOCK) acc += w[i] * x[i];
    acc = hcv_block_sum(acc, red);
    if (tid == 0) mixes_out[(size_t)t * mix_hc + m] = acc * rsq;
}

// 2026-09-25: The Sinkhorn finish on one thread, with HC a compile-time constant so the loops unroll.
// Nothing in this file calls it.



template <int HC>
__device__ __forceinline__ void hcv_finish_t(
    const float* __restrict__ mixes,
    const float* __restrict__ hc_scale,
    const float* __restrict__ hc_base,
    float* __restrict__ pre,
    float* __restrict__ post,
    float* __restrict__ comb,
    const unsigned int sinkhorn_iters,
    const float hc_eps) {
    constexpr unsigned int hc = HC;
    float c[HC * HC];
#pragma unroll
    for (unsigned int j = 0; j < hc; ++j) {
        pre[j] = hcv_sigmoid(mixes[j] * hc_scale[0] + hc_base[j]) + hc_eps;
        post[j] = 2.0f * hcv_sigmoid(mixes[hc + j] * hc_scale[1] + hc_base[hc + j]);
    }
#pragma unroll
    for (unsigned int i = 0; i < hc * hc; ++i) c[i] = mixes[2 * hc + i] * hc_scale[2] + hc_base[2 * hc + i];
#pragma unroll
    for (unsigned int j = 0; j < hc; ++j) {
        float m = -CUDART_INF_F;
#pragma unroll
        for (unsigned int k = 0; k < hc; ++k) m = fmaxf(m, c[j * hc + k]);
        float sum = 0.0f;
#pragma unroll
        for (unsigned int k = 0; k < hc; ++k) { c[j * hc + k] = expf(c[j * hc + k] - m); sum += c[j * hc + k]; }
#pragma unroll
        for (unsigned int k = 0; k < hc; ++k) c[j * hc + k] = c[j * hc + k] / sum + hc_eps;
    }
    for (unsigned int it = 0; it < sinkhorn_iters; ++it) {
        if (it > 0) {
#pragma unroll
            for (unsigned int j = 0; j < hc; ++j) {
                float s = 0.0f;
#pragma unroll
                for (unsigned int k = 0; k < hc; ++k) s += c[j * hc + k];
#pragma unroll
                for (unsigned int k = 0; k < hc; ++k) c[j * hc + k] /= s + hc_eps;
            }
        }
#pragma unroll
        for (unsigned int k = 0; k < hc; ++k) {
            float s = 0.0f;
#pragma unroll
            for (unsigned int j = 0; j < hc; ++j) s += c[j * hc + k];
#pragma unroll
            for (unsigned int j = 0; j < hc; ++j) c[j * hc + k] /= s + hc_eps;
        }
    }
#pragma unroll
    for (unsigned int i = 0; i < hc * hc; ++i) comb[i] = c[i];
}
// 2026-09-25: The same finish on one warp; all 32 lanes must call it. Lane i owns comb element i (row i / HC,
// column i % HC), and row and column sums are gathered by shuffles in index order, starting from 0.0f.





template <int HC>
__device__ __forceinline__ void hcv_finish_lanes(
    const float* __restrict__ mixes, const float* __restrict__ hc_scale,
    const float* __restrict__ hc_base, float* __restrict__ pre, float* __restrict__ post,
    float* __restrict__ comb, const unsigned int sinkhorn_iters, const float hc_eps) {
    constexpr unsigned int hc = HC, n = HC * HC;
    const unsigned int lane = threadIdx.x & 31u;
    const unsigned int j = lane / hc, k = lane % hc;
    if (lane < hc) {
        pre[lane] = hcv_sigmoid(mixes[lane] * hc_scale[0] + hc_base[lane]) + hc_eps;
        post[lane] = 2.0f * hcv_sigmoid(mixes[hc + lane] * hc_scale[1] + hc_base[hc + lane]);
    }
    float c = lane < n ? mixes[2 * hc + lane] * hc_scale[2] + hc_base[2 * hc + lane] : 0.0f;

    float m = -CUDART_INF_F;
#pragma unroll
    for (unsigned int kk = 0; kk < hc; ++kk) m = fmaxf(m, __shfl_sync(0xFFFFFFFFu, c, j * hc + kk));
    c = expf(c - m);
    float s = 0.0f;
#pragma unroll
    for (unsigned int kk = 0; kk < hc; ++kk) s += __shfl_sync(0xFFFFFFFFu, c, j * hc + kk);
    c = c / s + hc_eps;
    for (unsigned int it = 0; it < sinkhorn_iters; ++it) {
        if (it > 0) {
            s = 0.0f;
#pragma unroll
            for (unsigned int kk = 0; kk < hc; ++kk) s += __shfl_sync(0xFFFFFFFFu, c, j * hc + kk);
            c /= s + hc_eps;
        }
        s = 0.0f;
#pragma unroll
        for (unsigned int jj = 0; jj < hc; ++jj) s += __shfl_sync(0xFFFFFFFFu, c, jj * hc + k);
        c /= s + hc_eps;
    }
    if (lane < n) comb[lane] = c;
}
// 2026-09-25: Runs hcv_finish_lanes for hc 2, 3 or 4, and for any other value as hc 1. Every lane of the
// calling warp must reach this call.
__device__ __forceinline__ void hcv_finish(
    const unsigned int hc, const float* mixes, const float* hc_scale, const float* hc_base,
    float* pre, float* post, float* comb, const unsigned int sinkhorn_iters, const float hc_eps) {
    switch (hc) {
        case 4: hcv_finish_lanes<4>(mixes, hc_scale, hc_base, pre, post, comb, sinkhorn_iters, hc_eps); break;
        case 3: hcv_finish_lanes<3>(mixes, hc_scale, hc_base, pre, post, comb, sinkhorn_iters, hc_eps); break;
        case 2: hcv_finish_lanes<2>(mixes, hc_scale, hc_base, pre, post, comb, sinkhorn_iters, hc_eps); break;
        default: hcv_finish_lanes<1>(mixes, hc_scale, hc_base, pre, post, comb, sinkhorn_iters, hc_eps); break;
    }
}
// 2026-09-25: hc_v41_mixes_finish: grid (T); one warp per token runs the finish on precomputed mixes.
extern "C" __global__ void hc_v41_mixes_finish(
    const float* __restrict__ mixes_in,
    const float* __restrict__ hc_scale,
    const float* __restrict__ hc_base,
    float* __restrict__ pre,
    float* __restrict__ post,
    float* __restrict__ comb,
    const unsigned int hc_mult,
    const unsigned int sinkhorn_iters,
    const float hc_eps) {
    const unsigned int t = blockIdx.x;
    if (threadIdx.x >= 32) return;
    const unsigned int hc = hc_mult;
    const unsigned int mix_hc = (2 + hc) * hc;
    hcv_finish(hc, mixes_in + (size_t)t * mix_hc, hc_scale, hc_base, pre + (size_t)t * hc,
               post + (size_t)t * hc, comb + (size_t)t * hc * hc, sinkhorn_iters, hc_eps);
}
// 2026-09-25: One column of the collapse: y[d] = sum_c pre[c] * x[c][d].
__device__ __forceinline__ void hcv_collapse_col(
    const float* __restrict__ x, const float* __restrict__ p, __nv_bfloat16* __restrict__ y,
    const unsigned int hidden_size, const unsigned int hc_mult, const unsigned int d) {
    float acc = 0.0f;
    for (unsigned int c = 0; c < hc_mult; ++c) acc += p[c] * x[(size_t)c * hidden_size + d];
    y[d] = __float2bfloat16(acc);
}
// 2026-09-25: hc_v41_finish_collapse: the site's finish (warp 0 of block (0, t)) and token t's collapse with the
// previous site's pre (pre_in), one column per thread, in one launch; pre_in and pre_out must be different
// buffers. Grid (ceil(H / 256), T), block 256 (deepseek_v41_layer/hc_launch.rs).


extern "C" __global__ void hc_v41_finish_collapse(
    const float* __restrict__ mixes_in,
    const float* __restrict__ hc_scale,
    const float* __restrict__ hc_base,
    float* __restrict__ pre_out,
    float* __restrict__ post,
    float* __restrict__ comb,
    const float* __restrict__ streams,
    const float* __restrict__ pre_in,
    __nv_bfloat16* __restrict__ y,
    const unsigned int hidden_size,
    const unsigned int hc_mult,
    const unsigned int sinkhorn_iters,
    const float hc_eps) {
    const unsigned int t = blockIdx.y;
    const unsigned int hc = hc_mult;
    if (blockIdx.x == 0 && threadIdx.x < 32) {
        const unsigned int mix_hc = (2 + hc) * hc;
        hcv_finish(hc, mixes_in + (size_t)t * mix_hc, hc_scale, hc_base, pre_out + (size_t)t * hc,
                   post + (size_t)t * hc, comb + (size_t)t * hc * hc, sinkhorn_iters, hc_eps);
    }
    const unsigned int d = blockIdx.x * blockDim.x + threadIdx.x;
    if (d >= hidden_size) return;
    hcv_collapse_col(streams + (size_t)t * hc * hidden_size, pre_in + (size_t)t * hc,
                     y + (size_t)t * hidden_size, hidden_size, hc, d);
}
// 2026-09-25: hc_v41_collapse with one column per thread: grid (ceil(H / 256), T), block 256.

extern "C" __global__ void hc_v41_collapse_wide(
    const float* __restrict__ streams,
    const float* __restrict__ pre,
    __nv_bfloat16* __restrict__ y,
    const unsigned int hidden_size,
    const unsigned int hc_mult) {
    const unsigned int t = blockIdx.y;
    const unsigned int d = blockIdx.x * blockDim.x + threadIdx.x;
    if (d >= hidden_size) return;
    hcv_collapse_col(streams + (size_t)t * hc_mult * hidden_size, pre + (size_t)t * hc_mult,
                     y + (size_t)t * hidden_size, hidden_size, hc_mult, d);
}
// 2026-09-25: hc_v41_post_wide: out[t, j, d] = post[t, j] * block_out[t, d] + sum_i comb[t, i, j] * residual[t, i, d],
// one column per thread. Each thread reads its column's hc residual values before writing it, and
// hc_launch.rs passes one buffer as both residual and out. Grid (ceil(H / 256), T), block 256.


extern "C" __global__ void hc_v41_post_wide(
    const __nv_bfloat16* __restrict__ block_out,
    const float* __restrict__ residual,
    const float* __restrict__ post,
    const float* __restrict__ comb,
    float* __restrict__ out,
    const unsigned int hidden_size,
    const unsigned int hc_mult) {
    const unsigned int t = blockIdx.y;
    const unsigned int H = hidden_size;
    const unsigned int hc = hc_mult;
    const unsigned int d = blockIdx.x * blockDim.x + threadIdx.x;
    if (d >= H) return;
    const __nv_bfloat16* x = block_out + (size_t)t * H;
    const float* res = residual + (size_t)t * hc * H;
    const float* p = post + (size_t)t * hc;
    const float* c = comb + (size_t)t * hc * hc;
    float* o = out + (size_t)t * hc * H;
    float xd = (float)x[d];
    float rv[HCV_MAX_HC];
    for (unsigned int i = 0; i < hc; ++i) rv[i] = res[i * H + d];
    for (unsigned int j = 0; j < hc; ++j) {
        float acc = p[j] * xd;
        for (unsigned int i = 0; i < hc; ++i) acc += c[i * hc + j] * rv[i];
        o[j * H + d] = acc;
    }
}
// 2026-09-25: hc_v41_collapse: y[t][d] = sum_c pre[t][c] * streams[t][c][d]; grid (T), block-stride over d.
extern "C" __global__ void hc_v41_collapse(
    const float* __restrict__ streams,
    const float* __restrict__ pre,
    __nv_bfloat16* __restrict__ y,
    const unsigned int hidden_size,
    const unsigned int hc_mult) {
    const unsigned int t = blockIdx.x;
    const float* x = streams + (size_t)t * hc_mult * hidden_size;
    const float* p = pre + (size_t)t * hc_mult;
    for (unsigned int d = threadIdx.x; d < hidden_size; d += blockDim.x) {
        float acc = 0.0f;
        for (unsigned int c = 0; c < hc_mult; ++c) acc += p[c] * x[(size_t)c * hidden_size + d];
        y[(size_t)t * hidden_size + d] = __float2bfloat16(acc);
    }
}
