// SPDX-License-Identifier: MIT OR Apache-2.0

// 2026-09-25: NLLB-200 / M2M-100 encoder-decoder kernels (module `nllb_encoder`), loaded by
// crates/model-engine/src/model/nllb/kernels.rs: embedding, layer norm, elementwise ops, attention, argmax,
// beam top-k and cache scatter/gather. The kernels above the BF16 section comment are F32 throughout; the ones
// below it store BF16 and compute in F32. The BF16 path's GEMMs use `dense_gemm_bf16_pipelined` (module `gemm`).
//
// Linear weights are [N, K] row-major: C[m, n] = bias[n] + sum_k A[m, k] * W[n, k].
//
// Owner: gb10 kernels.
// Invariants: none beyond the types.

#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <math.h>


extern "C" __global__ void nllb_embed(
    const unsigned int* __restrict__ ids,
    const float* __restrict__ table,
    float* __restrict__ out,
    unsigned int d) {
    unsigned int tok = blockIdx.x;
    unsigned long long id = ids[tok];
    for (unsigned int i = threadIdx.x; i < d; i += blockDim.x) {
        out[(unsigned long long)tok * d + i] = table[id * d + i];
    }
}


extern "C" __global__ void nllb_scale_inplace(float* __restrict__ x, unsigned int n, float s) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) x[i] *= s;
}


extern "C" __global__ void nllb_add_inplace(
    float* __restrict__ dst, const float* __restrict__ src, unsigned int n) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) dst[i] += src[i];
}


extern "C" __global__ void nllb_relu_inplace(float* __restrict__ x, unsigned int n) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) x[i] = fmaxf(x[i], 0.0f);
}

// 2026-09-25: Affine layer norm over the last dim, in place. One block per row; blockDim.x must be a power of two,
// and the dynamic shared memory must hold blockDim.x floats.
extern "C" __global__ void nllb_layernorm(
    float* __restrict__ x, const float* __restrict__ w, const float* __restrict__ b,
    unsigned int rows, unsigned int dim, float eps) {
    unsigned int row = blockIdx.x;
    if (row >= rows) return;
    extern __shared__ float sm[];
    unsigned int tid = threadIdx.x;
    float* rowp = x + (unsigned long long)row * dim;

    float local = 0.0f;
    for (unsigned int i = tid; i < dim; i += blockDim.x) local += rowp[i];
    sm[tid] = local;
    __syncthreads();
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) sm[tid] += sm[tid + s];
        __syncthreads();
    }
    float mean = sm[0] / dim;
    __syncthreads();

    local = 0.0f;
    for (unsigned int i = tid; i < dim; i += blockDim.x) {
        float dv = rowp[i] - mean;
        local += dv * dv;
    }
    sm[tid] = local;
    __syncthreads();
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) sm[tid] += sm[tid + s];
        __syncthreads();
    }
    float inv = rsqrtf(sm[0] / dim + eps);
    for (unsigned int i = tid; i < dim; i += blockDim.x) {
        rowp[i] = (rowp[i] - mean) * inv * w[i] + b[i];
    }
}

// 2026-09-25: C[M, N] = A[M, K] @ W[N, K]^T + bias[N]; bias may be null.
extern "C" __global__ void nllb_linear(
    const float* __restrict__ a, const float* __restrict__ w, const float* __restrict__ bias,
    float* __restrict__ c, unsigned int M, unsigned int N, unsigned int K) {
    unsigned int n = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int m = blockIdx.y * blockDim.y + threadIdx.y;
    if (m >= M || n >= N) return;
    const float* arow = a + (unsigned long long)m * K;
    const float* wrow = w + (unsigned long long)n * K;
    float acc = bias ? bias[n] : 0.0f;
    for (unsigned int k = 0; k < K; k++) acc += arow[k] * wrow[k];
    c[(unsigned long long)m * N + n] = acc;
}

// 2026-09-25: Multi-head attention with separate query and key lengths. q [tq, H*D], k and v [tk, H*D], out [tq, H*D],
// heads interleaved on the feature axis. One block per (query, head), grid tq * H; blockDim.x == D, a power of
// two. With causal != 0 query i attends keys 0..=i, otherwise all tk keys. Dynamic shared memory: tk + D floats.



extern "C" __global__ void nllb_attn_kv(
    const float* __restrict__ q, const float* __restrict__ k, const float* __restrict__ v,
    float* __restrict__ out, unsigned int tq, unsigned int tk, unsigned int H, unsigned int D,
    float scale, unsigned int causal) {
    unsigned int qh = blockIdx.x;
    unsigned int query = qh / H;
    unsigned int head = qh % H;
    unsigned int tid = threadIdx.x;
    extern __shared__ float sh2[];
    float* scores = sh2;
    float* red = sh2 + tk;
    unsigned int dmodel = H * D;
    unsigned long long bq = (unsigned long long)query * dmodel + (unsigned long long)head * D;
    unsigned int kmax = causal ? (query + 1) : tk;

    for (unsigned int j = 0; j < kmax; j++) {
        unsigned long long bk = (unsigned long long)j * dmodel + (unsigned long long)head * D;
        red[tid] = q[bq + tid] * k[bk + tid];
        __syncthreads();
        for (unsigned int s = D / 2; s > 0; s >>= 1) {
            if (tid < s) red[tid] += red[tid + s];
            __syncthreads();
        }
        if (tid == 0) scores[j] = red[0] * scale;
        __syncthreads();
    }
    if (tid == 0) {
        float m = -1e30f;
        for (unsigned int j = 0; j < kmax; j++) m = fmaxf(m, scores[j]);
        float su = 0.0f;
        for (unsigned int j = 0; j < kmax; j++) {
            scores[j] = expf(scores[j] - m);
            su += scores[j];
        }
        for (unsigned int j = 0; j < kmax; j++) scores[j] /= su;
    }
    __syncthreads();
    float acc = 0.0f;
    for (unsigned int j = 0; j < kmax; j++) {
        acc += scores[j] * v[(unsigned long long)j * dmodel + (unsigned long long)head * D + tid];
    }
    out[bq + tid] = acc;
}

// 2026-09-25: Non-causal multi-head attention; q, k, v and out are [seq, H*D], heads interleaved on the feature
// axis. One block per (query, head); blockDim.x == D, a power of two. `scale` multiplies the logits.

extern "C" __global__ void nllb_attention(
    const float* __restrict__ q, const float* __restrict__ k, const float* __restrict__ v,
    float* __restrict__ out, unsigned int seq, unsigned int H, unsigned int D, float scale) {
    unsigned int qh = blockIdx.x;
    unsigned int query = qh / H;
    unsigned int head = qh % H;
    unsigned int tid = threadIdx.x;
    extern __shared__ float sh[];   // 2026-09-25: dynamic: scores[seq], then red[D]
    float* scores = sh;
    float* red = sh + seq;
    unsigned int dmodel = H * D;
    unsigned long long bq = (unsigned long long)query * dmodel + (unsigned long long)head * D;

    for (unsigned int j = 0; j < seq; j++) {
        unsigned long long bk = (unsigned long long)j * dmodel + (unsigned long long)head * D;
        red[tid] = q[bq + tid] * k[bk + tid];
        __syncthreads();
        for (unsigned int s = D / 2; s > 0; s >>= 1) {
            if (tid < s) red[tid] += red[tid + s];
            __syncthreads();
        }
        if (tid == 0) scores[j] = red[0] * scale;
        __syncthreads();
    }
    if (tid == 0) {
        float m = -1e30f;
        for (unsigned int j = 0; j < seq; j++) m = fmaxf(m, scores[j]);
        float su = 0.0f;
        for (unsigned int j = 0; j < seq; j++) {
            scores[j] = expf(scores[j] - m);
            su += scores[j];
        }
        for (unsigned int j = 0; j < seq; j++) scores[j] /= su;
    }
    __syncthreads();
    float acc = 0.0f;
    for (unsigned int j = 0; j < seq; j++) {
        acc += scores[j] * v[(unsigned long long)j * dmodel + (unsigned long long)head * D + tid];
    }
    out[bq + tid] = acc;
}

// 2026-09-25: BF16 kernels: BF16 in memory, F32 arithmetic.





extern "C" __global__ void nllb_embed_bf16(
    const unsigned int* __restrict__ ids, const __nv_bfloat16* __restrict__ table,
    __nv_bfloat16* __restrict__ out, unsigned int d) {
    unsigned int tok = blockIdx.x;
    unsigned long long id = ids[tok];
    for (unsigned int i = threadIdx.x; i < d; i += blockDim.x)
        out[(unsigned long long)tok * d + i] = table[id * d + i];
}

extern "C" __global__ void nllb_scale_bf16(__nv_bfloat16* __restrict__ x, unsigned int n, float s) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) x[i] = __float2bfloat16(__bfloat162float(x[i]) * s);
}

extern "C" __global__ void nllb_add_bf16(
    __nv_bfloat16* __restrict__ dst, const __nv_bfloat16* __restrict__ src, unsigned int n) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) dst[i] = __float2bfloat16(__bfloat162float(dst[i]) + __bfloat162float(src[i]));
}

// 2026-09-25: c[m, n] += bias[n] for c [M, N].
extern "C" __global__ void nllb_bias_bf16(
    __nv_bfloat16* __restrict__ c, const __nv_bfloat16* __restrict__ bias,
    unsigned int M, unsigned int N) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= M * N) return;
    unsigned int n = idx % N;
    c[idx] = __float2bfloat16(__bfloat162float(c[idx]) + __bfloat162float(bias[n]));
}

extern "C" __global__ void nllb_relu_bf16(__nv_bfloat16* __restrict__ x, unsigned int n) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) x[i] = __float2bfloat16(fmaxf(__bfloat162float(x[i]), 0.0f));
}

extern "C" __global__ void nllb_layernorm_bf16(
    __nv_bfloat16* __restrict__ x, const __nv_bfloat16* __restrict__ w,
    const __nv_bfloat16* __restrict__ b, unsigned int rows, unsigned int dim, float eps) {
    unsigned int row = blockIdx.x;
    if (row >= rows) return;
    extern __shared__ float sm[];
    unsigned int tid = threadIdx.x;
    __nv_bfloat16* rowp = x + (unsigned long long)row * dim;
    float local = 0.0f;
    for (unsigned int i = tid; i < dim; i += blockDim.x) local += __bfloat162float(rowp[i]);
    sm[tid] = local;
    __syncthreads();
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) sm[tid] += sm[tid + s];
        __syncthreads();
    }
    float mean = sm[0] / dim;
    __syncthreads();
    local = 0.0f;
    for (unsigned int i = tid; i < dim; i += blockDim.x) {
        float dv = __bfloat162float(rowp[i]) - mean;
        local += dv * dv;
    }
    sm[tid] = local;
    __syncthreads();
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) sm[tid] += sm[tid + s];
        __syncthreads();
    }
    float inv = rsqrtf(sm[0] / dim + eps);
    for (unsigned int i = tid; i < dim; i += blockDim.x) {
        float v = (__bfloat162float(rowp[i]) - mean) * inv * __bfloat162float(w[i]) + __bfloat162float(b[i]);
        rowp[i] = __float2bfloat16(v);
    }
}

// 2026-09-25: `nllb_layernorm_bf16` out of place: the same arithmetic, reading `in` and writing `out`, which are
// __restrict__ and must not overlap.



extern "C" __global__ void nllb_layernorm_oop_bf16(
    const __nv_bfloat16* __restrict__ in, __nv_bfloat16* __restrict__ out,
    const __nv_bfloat16* __restrict__ w, const __nv_bfloat16* __restrict__ b,
    unsigned int rows, unsigned int dim, float eps) {
    unsigned int row = blockIdx.x;
    if (row >= rows) return;
    extern __shared__ float sm[];
    unsigned int tid = threadIdx.x;
    const __nv_bfloat16* inp = in + (unsigned long long)row * dim;
    __nv_bfloat16* outp = out + (unsigned long long)row * dim;
    float local = 0.0f;
    for (unsigned int i = tid; i < dim; i += blockDim.x) local += __bfloat162float(inp[i]);
    sm[tid] = local;
    __syncthreads();
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) sm[tid] += sm[tid + s];
        __syncthreads();
    }
    float mean = sm[0] / dim;
    __syncthreads();
    local = 0.0f;
    for (unsigned int i = tid; i < dim; i += blockDim.x) {
        float dv = __bfloat162float(inp[i]) - mean;
        local += dv * dv;
    }
    sm[tid] = local;
    __syncthreads();
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) sm[tid] += sm[tid + s];
        __syncthreads();
    }
    float inv = rsqrtf(sm[0] / dim + eps);
    for (unsigned int i = tid; i < dim; i += blockDim.x) {
        float v = (__bfloat162float(inp[i]) - mean) * inv * __bfloat162float(w[i]) + __bfloat162float(b[i]);
        outp[i] = __float2bfloat16(v);
    }
}

extern "C" __global__ void nllb_attn_kv_bf16(
    const __nv_bfloat16* __restrict__ q, const __nv_bfloat16* __restrict__ k,
    const __nv_bfloat16* __restrict__ v, __nv_bfloat16* __restrict__ out,
    unsigned int tq, unsigned int tk, unsigned int H, unsigned int D, float scale, unsigned int causal) {
    unsigned int qh = blockIdx.x;
    unsigned int query = qh / H;
    unsigned int head = qh % H;
    unsigned int tid = threadIdx.x;
    extern __shared__ float sh2b[];
    float* scores = sh2b;
    float* red = sh2b + tk;
    unsigned int dmodel = H * D;
    unsigned long long bq = (unsigned long long)query * dmodel + (unsigned long long)head * D;
    unsigned int kmax = causal ? (query + 1) : tk;
    for (unsigned int j = 0; j < kmax; j++) {
        unsigned long long bk = (unsigned long long)j * dmodel + (unsigned long long)head * D;
        red[tid] = __bfloat162float(q[bq + tid]) * __bfloat162float(k[bk + tid]);
        __syncthreads();
        for (unsigned int s = D / 2; s > 0; s >>= 1) {
            if (tid < s) red[tid] += red[tid + s];
            __syncthreads();
        }
        if (tid == 0) scores[j] = red[0] * scale;
        __syncthreads();
    }
    if (tid == 0) {
        float m = -1e30f;
        for (unsigned int j = 0; j < kmax; j++) m = fmaxf(m, scores[j]);
        float su = 0.0f;
        for (unsigned int j = 0; j < kmax; j++) {
            scores[j] = expf(scores[j] - m);
            su += scores[j];
        }
        for (unsigned int j = 0; j < kmax; j++) scores[j] /= su;
    }
    __syncthreads();
    float acc = 0.0f;
    for (unsigned int j = 0; j < kmax; j++)
        acc += scores[j] * __bfloat162float(v[(unsigned long long)j * dmodel + (unsigned long long)head * D + tid]);
    out[bq + tid] = __float2bfloat16(acc);
}

// 2026-09-25: y[N] = W[N, K] @ x[K] + bias[N], bias may be null. One warp per output row, F32 accumulation.



extern "C" __global__ void nllb_gemv_bf16(
    const __nv_bfloat16* __restrict__ x, const __nv_bfloat16* __restrict__ W,
    const __nv_bfloat16* __restrict__ bias, __nv_bfloat16* __restrict__ y,
    unsigned int N, unsigned int K) {
    unsigned int warp = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
    unsigned int lane = threadIdx.x & 31u;
    if (warp >= N) return;
    const __nv_bfloat16* wrow = W + (unsigned long long)warp * K;
    float acc = 0.0f;
    for (unsigned int k = lane; k < K; k += 32u)
        acc += __bfloat162float(x[k]) * __bfloat162float(wrow[k]);
    for (int o = 16; o > 0; o >>= 1)
        acc += __shfl_down_sync(0xffffffffu, acc, o);
    if (lane == 0) {
        if (bias) acc += __bfloat162float(bias[warp]);
        y[warp] = __float2bfloat16(acc);
    }
}

// 2026-09-25: Batched decode: B sequences step together, each with a batch-major [B, stride, H*D] K/V cache.







// 2026-09-25: Batched single-query attention. q and out [B, H*D]; row b attends rows 0..tk[b]-1 of cache slab
// b / group in kc and vc, [*, stride, H*D]. Grid B * H, block D (a power of two); dynamic shared memory
// D + max(tk) floats, red then scores.
extern "C" __global__ void nllb_attn_bdecode(
    const __nv_bfloat16* __restrict__ q, const __nv_bfloat16* __restrict__ kc,
    const __nv_bfloat16* __restrict__ vc, __nv_bfloat16* __restrict__ out,
    unsigned int B, unsigned int stride, unsigned int group,
    const unsigned int* __restrict__ tk, unsigned int H, unsigned int D, float scale) {
    unsigned int bh = blockIdx.x;
    unsigned int b = bh / H;
    unsigned int head = bh % H;
    unsigned int tid = threadIdx.x;
    if (b >= B) return;
    unsigned int t = tk[b];
    unsigned int dmodel = H * D;
    extern __shared__ float shd[];
    float* red = shd;
    float* scores = shd + D;
    // 2026-09-25: Rows b with the same b / group share one cache slab; group 1 gives each row its own.


    unsigned long long qbase = (unsigned long long)b * dmodel + (unsigned long long)head * D;
    unsigned long long cbase =
        (unsigned long long)(b / group) * stride * dmodel + (unsigned long long)head * D;
    for (unsigned int j = 0; j < t; j++) {
        red[tid] = __bfloat162float(q[qbase + tid]) *
                   __bfloat162float(kc[cbase + (unsigned long long)j * dmodel + tid]);
        __syncthreads();
        for (unsigned int s = D / 2; s > 0; s >>= 1) {
            if (tid < s) red[tid] += red[tid + s];
            __syncthreads();
        }
        if (tid == 0) scores[j] = red[0] * scale;
        __syncthreads();
    }
    if (tid == 0) {
        float m = -1e30f;
        for (unsigned int j = 0; j < t; j++) m = fmaxf(m, scores[j]);
        float su = 0.0f;
        for (unsigned int j = 0; j < t; j++) {
            scores[j] = expf(scores[j] - m);
            su += scores[j];
        }
        for (unsigned int j = 0; j < t; j++) scores[j] /= su;
    }
    __syncthreads();
    float acc = 0.0f;
    for (unsigned int j = 0; j < t; j++)
        acc += scores[j] * __bfloat162float(vc[cbase + (unsigned long long)j * dmodel + tid]);
    out[qbase + tid] = __float2bfloat16(acc);
}

// 2026-09-25: Writes src [B, d] into row `pos` of the batch-major cache dst [B, stride, d].
extern "C" __global__ void nllb_scatter_batched(
    const __nv_bfloat16* __restrict__ src, __nv_bfloat16* __restrict__ dst,
    unsigned int pos, unsigned int B, unsigned int stride, unsigned int d) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= B * d) return;
    unsigned int b = idx / d;
    unsigned int i = idx % d;
    dst[(unsigned long long)b * stride * d + (unsigned long long)pos * d + i] = src[idx];
}

// 2026-09-25: dst[n] += row[n % d]: adds one d-row to every row of dst.
extern "C" __global__ void nllb_add_row_bf16(
    __nv_bfloat16* __restrict__ dst, const __nv_bfloat16* __restrict__ row,
    unsigned int n, unsigned int d) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n)
        dst[idx] = __float2bfloat16(__bfloat162float(dst[idx]) + __bfloat162float(row[idx % d]));
}

// 2026-09-25: out[b] = argmax over v of logits[b, v]. Grid B; blockDim.x a power of two; dynamic shared memory
// blockDim.x * 8 bytes.
extern "C" __global__ void nllb_argmax_batched(
    const __nv_bfloat16* __restrict__ logits, unsigned int* __restrict__ out,
    unsigned int B, unsigned int vocab) {
    unsigned int b = blockIdx.x;
    unsigned int tid = threadIdx.x;
    extern __shared__ char smem[];
    float* sval = (float*)smem;
    unsigned int* sidx = (unsigned int*)(sval + blockDim.x);
    const __nv_bfloat16* row = logits + (unsigned long long)b * vocab;
    float best = -1e30f;
    unsigned int bi = 0;
    for (unsigned int v = tid; v < vocab; v += blockDim.x) {
        float x = __bfloat162float(row[v]);
        if (x > best) {
            best = x;
            bi = v;
        }
    }
    sval[tid] = best;
    sidx[tid] = bi;
    __syncthreads();
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s && sval[tid + s] > sval[tid]) {
            sval[tid] = sval[tid + s];
            sidx[tid] = sidx[tid + s];
        }
        __syncthreads();
    }
    if (tid == 0) out[b] = sidx[0];
}

// 2026-09-25: dst[i] = src[perm[i]] over rows 0..used of each batch-major [B, stride, d] cache slab.


extern "C" __global__ void nllb_gather_batched(
    const __nv_bfloat16* __restrict__ src, __nv_bfloat16* __restrict__ dst,
    const unsigned int* __restrict__ perm, unsigned int B, unsigned int used,
    unsigned int stride, unsigned int d) {
    unsigned long long idx = (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;
    unsigned long long total = (unsigned long long)B * used * d;
    if (idx >= total) return;
    unsigned int i = (unsigned int)(idx / ((unsigned long long)used * d));
    unsigned long long rem = idx % ((unsigned long long)used * d);
    unsigned int src_b = perm[i];
    dst[(unsigned long long)i * stride * d + rem] =
        src[(unsigned long long)src_b * stride * d + rem];
}

// 2026-09-25: For each row b of logits [B, vocab]: lse_out[b] = logsumexp over the whole vocab, and the K largest
// (value, token) pairs in val_out and idx_out [B, K], descending, with the lower token id first on equal
// values. Grid B; blockDim.x at most 128 (the size of sm and ss); K at most NLLB_TOPK_MAX, unchecked;
// dynamic shared memory blockDim.x * K * 8 bytes.



#define NLLB_TOPK_MAX 32
extern "C" __global__ void nllb_beam_topk(
    const __nv_bfloat16* __restrict__ logits, float* __restrict__ lse_out,
    float* __restrict__ val_out, unsigned int* __restrict__ idx_out,
    unsigned int B, unsigned int vocab, unsigned int K) {
    unsigned int b = blockIdx.x;
    unsigned int tid = threadIdx.x;
    unsigned int nt = blockDim.x;
    const __nv_bfloat16* row = logits + (unsigned long long)b * vocab;

    // 2026-09-25: Per-thread local top-K, descending, and streaming logsumexp state (m, s).
    float lv[NLLB_TOPK_MAX];
    unsigned int li[NLLB_TOPK_MAX];
    for (unsigned int j = 0; j < K; ++j) {
        lv[j] = -1e30f;
        li[j] = 0u;
    }
    float m = -1e30f, s = 0.0f;
    for (unsigned int v = tid; v < vocab; v += nt) {
        float x = __bfloat162float(row[v]);
        if (x > m) {
            s = s * __expf(m - x) + 1.0f;
            m = x;
        } else {
            s += __expf(x - m);
        }
        // 2026-09-25: Insert into the local top-K. A thread scans v ascending and inserts after equal values, so the
        // lower token id stays first.
        if (x > lv[K - 1]) {
            int j = (int)K - 1;
            while (j > 0 && lv[j - 1] < x) {
                lv[j] = lv[j - 1];
                li[j] = li[j - 1];
                --j;
            }
            lv[j] = x;
            li[j] = v;
        }
    }

    extern __shared__ char smem[];
    float* cval = (float*)smem;
    unsigned int* cidx = (unsigned int*)(cval + nt * K);
    for (unsigned int j = 0; j < K; ++j) {
        cval[tid * K + j] = lv[j];
        cidx[tid * K + j] = li[j];
    }
    __shared__ float sm[128];
    __shared__ float ss[128];
    sm[tid] = m;
    ss[tid] = s;
    __syncthreads();

    if (tid == 0) {
        // 2026-09-25: Combine the per-thread partials: lse = M + log(sum_t s_t * exp(m_t - M)).
        float M = -1e30f;
        for (unsigned int t = 0; t < nt; ++t)
            if (sm[t] > M) M = sm[t];
        float S = 0.0f;
        for (unsigned int t = 0; t < nt; ++t) S += ss[t] * __expf(sm[t] - M);
        lse_out[b] = M + logf(S);
        // 2026-09-25: Thread 0 extracts the top K from the nt * K candidates, lower token id first on equal values.
        unsigned int cand = nt * K;
        for (unsigned int r = 0; r < K; ++r) {
            float best = -1e30f;
            unsigned int bpos = 0, bid = 0xFFFFFFFFu;
            for (unsigned int c = 0; c < cand; ++c) {
                float cv = cval[c];
                if (cv > best || (cv == best && cidx[c] < bid)) {
                    best = cv;
                    bid = cidx[c];
                    bpos = c;
                }
            }
            val_out[b * K + r] = best;
            idx_out[b * K + r] = cidx[bpos];
            cval[bpos] = -1e30f;
        }
    }
}
