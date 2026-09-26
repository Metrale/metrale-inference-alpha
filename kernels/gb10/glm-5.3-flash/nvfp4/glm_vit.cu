// SPDX-License-Identifier: MIT OR Apache-2.0

// 2026-09-25: GLM-5.3-Flash vision tower kernels (module `glm_vit`), launched by
// crates/model-layers/src/layers/glm_vit/.
// Owner: gb10 kernels (glm-5.3-flash).
// Invariants: none beyond each kernel's launch shape, stated above it.
//
// Storage is BF16; reductions accumulate in f32. The GEMMs are not here: the layer runs
// them on `gemm::dense_gemm_bf16_pipelined` (C = A * B^T), adding a bias with
// `glm_vit_add_bias`, and the attention scores on `gemm::dense_gemm_bf16_f32out`.







#include <cuda_bf16.h>
#include <math.h>

__device__ inline float bf16_to_f32(__nv_bfloat16 v) { return __bfloat162float(v); }
__device__ inline __nv_bfloat16 f32_to_bf16(float v) { return __float2bfloat16(v); }

// 2026-09-25: Block-wide sum. `red` is `ceil(blockDim.x/32)` floats of shared scratch.
__device__ inline float glm_block_sum(float v, float* red) {
    unsigned lane = threadIdx.x & 31u, warp = threadIdx.x >> 5;
    for (int o = 16; o > 0; o >>= 1) v += __shfl_down_sync(0xffffffff, v, o);
    if (lane == 0) red[warp] = v;
    __syncthreads();
    if (threadIdx.x == 0) {
        float s = 0.0f;
        unsigned nw = (blockDim.x + 31u) / 32u;
        for (unsigned w = 0; w < nw; ++w) s += red[w];
        red[0] = s;
    }
    __syncthreads();
    return red[0];
}

// 2026-09-25: f32 -> bf16 (pixel upload). grid (ceil(n/256),1,1) block (256,1,1).

extern "C" __global__ void glm_vit_f32_to_bf16(
    const float* __restrict__ src, __nv_bfloat16* __restrict__ dst, unsigned int n)
{
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) dst[i] = f32_to_bf16(src[i]);
}


extern "C" __global__ void glm_vit_copy(
    const __nv_bfloat16* __restrict__ src, __nv_bfloat16* __restrict__ dst, unsigned int n)
{
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) dst[i] = src[i];
}

// 2026-09-25: dst += src, in place.
extern "C" __global__ void glm_vit_add_inplace(
    __nv_bfloat16* __restrict__ dst, const __nv_bfloat16* __restrict__ src, unsigned int n)
{
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) dst[i] = f32_to_bf16(bf16_to_f32(dst[i]) + bf16_to_f32(src[i]));
}

// 2026-09-25: C[m,n] += bias[n]; `dense_gemm_bf16_pipelined` takes no bias.
extern "C" __global__ void glm_vit_add_bias(
    __nv_bfloat16* __restrict__ C, const __nv_bfloat16* __restrict__ bias,
    unsigned int M, unsigned int N)
{
    unsigned long long idx = (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= (unsigned long long)M * N) return;
    C[idx] = f32_to_bf16(bf16_to_f32(C[idx]) + bf16_to_f32(bias[(unsigned int)(idx % N)]));
}

// 2026-09-25: Weight-only RMSNorm, in place: x = x * rsqrt(mean(x^2) + eps) * w, one block
// per row. The layer applies it as norm1, norm2 and post_layernorm (glm_vit/block.rs,
// forward.rs). grid (N,1,1) block (min(D,1024),1,1), shared ceil(blockDim.x/32) floats.


extern "C" __global__ void glm_vit_rmsnorm(
    __nv_bfloat16* __restrict__ x, const __nv_bfloat16* __restrict__ w,
    unsigned int N, unsigned int D, float eps)
{
    unsigned int row = blockIdx.x;
    if (row >= N) return;
    __nv_bfloat16* r = x + (size_t)row * D;
    extern __shared__ float red[];

    float ss = 0.0f;
    for (unsigned int i = threadIdx.x; i < D; i += blockDim.x) {
        float v = bf16_to_f32(r[i]);
        ss += v * v;
    }
    float total = glm_block_sum(ss, red);
    float inv = rsqrtf(total / (float)D + eps);
    for (unsigned int i = threadIdx.x; i < D; i += blockDim.x)
        r[i] = f32_to_bf16(bf16_to_f32(r[i]) * inv * bf16_to_f32(w[i]));
}

// 2026-09-25: LayerNorm with weight and bias, in place; only the merger's
// post_projection_norm uses it (glm_vit/merge.rs). grid (N,1,1) block (min(D,1024),1,1),
// shared ceil(blockDim.x/32) floats.


extern "C" __global__ void glm_vit_layernorm(
    __nv_bfloat16* __restrict__ x, const __nv_bfloat16* __restrict__ w,
    const __nv_bfloat16* __restrict__ b, unsigned int N, unsigned int D, float eps)
{
    unsigned int row = blockIdx.x;
    if (row >= N) return;
    __nv_bfloat16* r = x + (size_t)row * D;
    extern __shared__ float red[];

    float s = 0.0f;
    for (unsigned int i = threadIdx.x; i < D; i += blockDim.x) s += bf16_to_f32(r[i]);
    float mean = glm_block_sum(s, red) / (float)D;
    __syncthreads();

    float v = 0.0f;
    for (unsigned int i = threadIdx.x; i < D; i += blockDim.x) {
        float d = bf16_to_f32(r[i]) - mean;
        v += d * d;
    }
    float inv = rsqrtf(glm_block_sum(v, red) / (float)D + eps);
    for (unsigned int i = threadIdx.x; i < D; i += blockDim.x) {
        float y = (bf16_to_f32(r[i]) - mean) * inv;
        r[i] = f32_to_bf16(y * bf16_to_f32(w[i]) + bf16_to_f32(b[i]));
    }
}

// 2026-09-25: Exact (erf) GELU, in place; the merger's activation (glm_vit/merge.rs).



extern "C" __global__ void glm_vit_gelu_erf(__nv_bfloat16* __restrict__ x, unsigned int n)
{
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float v = bf16_to_f32(x[i]);
    x[i] = f32_to_bf16(0.5f * v * (1.0f + erff(v * 0.70710678118654752f)));
}

// 2026-09-25: Clamped SwiGLU: out = silu(min(gate, limit)) * clamp(up, -limit, limit). The
// layer passes the vision config's swiglu_limit (glm_vit/block.rs `swiglu`).


extern "C" __global__ void glm_vit_swiglu_clamp(
    const __nv_bfloat16* __restrict__ gate, const __nv_bfloat16* __restrict__ up,
    __nv_bfloat16* __restrict__ out, unsigned int n, float limit)
{
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float g = fminf(bf16_to_f32(gate[i]), limit);
    float u = fminf(fmaxf(bf16_to_f32(up[i]), -limit), limit);
    float silu = g / (1.0f + expf(-g));
    out[i] = f32_to_bf16(silu * u);
}

// 2026-09-25: Per-head QK-RMSNorm, then 2D axial RoPE, into head-contiguous Qr/Kr, and V
// transposed to Vt.
//
// One block per (token, head), blockDim.x == D (head_dim). q_norm/k_norm apply to each
// head before the rotation. `rope_cos`/`rope_sin` are the [seq, D] BF16 tables the host
// builds from block-major patch ids (glm_vit/rope.rs); the rotate-half partner is
// d +/- D/2 over the whole head.
//
// Vt is [H, D, seq] so the P * V GEMM, which computes A * B^T, needs no second transpose.
// grid (seq,H,1) block (D,1,1), shared (2*D + ceil(D/32)) floats.



extern "C" __global__ void glm_vit_qknorm_rope_deint(
    const __nv_bfloat16* __restrict__ QKV,      // 2026-09-25: [seq, 3*H*D] fused, q|k|v major
    const __nv_bfloat16* __restrict__ q_norm_w, // 2026-09-25: [D]
    const __nv_bfloat16* __restrict__ k_norm_w, // 2026-09-25: [D]
    __nv_bfloat16* __restrict__ Qr,             // 2026-09-25: [H, seq, D]
    __nv_bfloat16* __restrict__ Kr,             // 2026-09-25: [H, seq, D]
    __nv_bfloat16* __restrict__ Vt,             // 2026-09-25: [H, D, seq]
    const __nv_bfloat16* __restrict__ rope_cos, // 2026-09-25: [seq, D]
    const __nv_bfloat16* __restrict__ rope_sin, // 2026-09-25: [seq, D]
    unsigned int seq, unsigned int H, unsigned int D, float eps)
{
    unsigned int tok = blockIdx.x, h = blockIdx.y, d = threadIdx.x;
    if (tok >= seq || h >= H || d >= D) return;

    extern __shared__ float smem[];
    float* sq = smem;          // 2026-09-25: [D]
    float* sk = smem + D;      // 2026-09-25: [D]
    float* red = smem + 2 * D; // 2026-09-25: [ceil(D/32)]

    size_t stride = (size_t)3u * H * D;
    const __nv_bfloat16* base = QKV + (size_t)tok * stride + (size_t)h * D;
    float qv = bf16_to_f32(base[(size_t)0 * H * D + d]);
    float kv = bf16_to_f32(base[(size_t)1 * H * D + d]);
    float vv = bf16_to_f32(base[(size_t)2 * H * D + d]);

    float q_inv = rsqrtf(glm_block_sum(qv * qv, red) / (float)D + eps);
    __syncthreads();
    float k_inv = rsqrtf(glm_block_sum(kv * kv, red) / (float)D + eps);
    __syncthreads();

    sq[d] = qv * q_inv * bf16_to_f32(q_norm_w[d]);
    sk[d] = kv * k_inv * bf16_to_f32(k_norm_w[d]);
    __syncthreads();

    unsigned int half = D / 2;
    float c = bf16_to_f32(rope_cos[(size_t)tok * D + d]);
    float s = bf16_to_f32(rope_sin[(size_t)tok * D + d]);
    float q_rot = (d < half) ? -sq[d + half] : sq[d - half];
    float k_rot = (d < half) ? -sk[d + half] : sk[d - half];

    size_t hc = (size_t)h * seq * D + (size_t)tok * D + d;
    Qr[hc] = f32_to_bf16(sq[d] * c + q_rot * s);
    Kr[hc] = f32_to_bf16(sk[d] * c + k_rot * s);
    Vt[(size_t)h * D * seq + (size_t)d * seq + (size_t)tok] = f32_to_bf16(vv);
}

// 2026-09-25: Row softmax of raw f32 scores S[seq,seq] into bf16 P, with rsqrt(D) applied
// here because the score GEMM emits raw Q * K^T. grid (seq,1,1) block (256,1,1).

extern "C" __global__ void glm_vit_softmax_rows(
    const float* __restrict__ S, __nv_bfloat16* __restrict__ P,
    unsigned int seq, unsigned int D)
{
    unsigned int row = blockIdx.x;
    if (row >= seq) return;
    const float* srow = S + (size_t)row * seq;
    __nv_bfloat16* prow = P + (size_t)row * seq;
    float scale = rsqrtf((float)D);

    __shared__ float red[256 / 32];
    unsigned int tid = threadIdx.x, lane = tid & 31u, warp = tid >> 5;

    float m = -1e30f;
    for (unsigned int j = tid; j < seq; j += blockDim.x) m = fmaxf(m, srow[j] * scale);
    for (int o = 16; o > 0; o >>= 1) m = fmaxf(m, __shfl_down_sync(0xffffffff, m, o));
    if (lane == 0) red[warp] = m;
    __syncthreads();
    if (tid == 0) {
        float mm = -1e30f;
        for (unsigned int w = 0; w < blockDim.x / 32; ++w) mm = fmaxf(mm, red[w]);
        red[0] = mm;
    }
    __syncthreads();
    float row_max = red[0];
    __syncthreads();

    float acc = 0.0f;
    for (unsigned int j = tid; j < seq; j += blockDim.x) acc += expf(srow[j] * scale - row_max);
    for (int o = 16; o > 0; o >>= 1) acc += __shfl_down_sync(0xffffffff, acc, o);
    if (lane == 0) red[warp] = acc;
    __syncthreads();
    if (tid == 0) {
        float ss = 0.0f;
        for (unsigned int w = 0; w < blockDim.x / 32; ++w) ss += red[w];
        red[0] = (ss > 0.0f) ? (1.0f / ss) : 0.0f;
    }
    __syncthreads();
    float inv = red[0];

    for (unsigned int j = tid; j < seq; j += blockDim.x)
        prow[j] = f32_to_bf16(expf(srow[j] * scale - row_max) * inv);
}

// 2026-09-25: Copy one head's O[seq,D] into its columns of the [seq, dst_stride] output.
extern "C" __global__ void glm_vit_scatter_head(
    const __nv_bfloat16* __restrict__ Oh, __nv_bfloat16* __restrict__ O,
    unsigned int seq, unsigned int D, unsigned int dst_stride)
{
    unsigned int lin = blockIdx.x * blockDim.x + threadIdx.x;
    if (lin >= seq * D) return;
    unsigned int tok = lin / D, d = lin % D;
    O[(size_t)tok * dst_stride + d] = Oh[(size_t)tok * D + d];
}

// 2026-09-25: im2col for the 2x2 stride-2 downsample conv: dst[b, c*4 + t] = src[b*4 + t, c].
//
// The token stream is block-major, so four consecutive tokens are one 2x2 block and token
// t of a block is (kh, kw) = (t / 2, t % 2) (glm_vit/rope.rs `block_major_position_ids`).
// A conv weight [out_c, in_c, 2, 2] flattened to [out_c, in_c * 4] has the same
// (in_c, kh, kw) nesting, so the downsample is an ordinary A * B^T GEMM.
// grid (ceil(M/256),1,1) block (256,1,1) with M = merged_p * 4 * C.









extern "C" __global__ void glm_vit_im2col_2x2(
    const __nv_bfloat16* __restrict__ src, // 2026-09-25: [merged_p*4, C]
    __nv_bfloat16* __restrict__ dst,       // 2026-09-25: [merged_p, 4*C]
    unsigned int merged_p, unsigned int C)
{
    unsigned long long lin = (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;
    unsigned long long total = (unsigned long long)merged_p * 4ull * C;
    if (lin >= total) return;
    unsigned int c = (unsigned int)(lin % C);
    unsigned long long rest = lin / C;
    unsigned int t = (unsigned int)(rest % 4ull);
    unsigned int b = (unsigned int)(rest / 4ull);
    dst[(size_t)b * 4u * C + (size_t)c * 4u + t] = src[lin];
}
