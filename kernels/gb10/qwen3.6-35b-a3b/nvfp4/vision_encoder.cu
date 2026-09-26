// SPDX-License-Identifier: MIT OR Apache-2.0

// 2026-09-25: Vision encoder (ViT) kernels: BF16 storage, FP32 accumulation; launched from crates/model-layers/src/layers/vision_encoder/enc_impl.
// forked-from: kernels/gb10/qwen3-vl-30b-a3b/nvfp4/vision_encoder.cu (2026-09-24; 149 of 462 lines differ, see kernels/FORKS.md)
// Owner: gb10 kernels (qwen3.6-35b-a3b, and the targets that list this file in `[sources] use`).
// Invariants: none beyond the types.

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <math.h>



__device__ inline float bf16_to_f32(__nv_bfloat16 v) {
    return __bfloat162float(v);
}
__device__ inline __nv_bfloat16 f32_to_bf16(float v) {
    return __float2bfloat16(v);
}

// 2026-09-25: C[M,N] = A[M,K] @ B[N,K]^T + bias[N], all BF16. Grid (ceil(N/32), ceil(M/32)), block (32, 32).
// vit_block.rs runs it when dense_gemm_bf16_pipelined or vision_add_bias does not resolve.
extern "C" __global__
void vision_gemm_bias(
    const __nv_bfloat16* __restrict__ A,   // 2026-09-25: [M, K]
    const __nv_bfloat16* __restrict__ B,   // 2026-09-25: [N, K]
    const __nv_bfloat16* __restrict__ bias,// 2026-09-25: [N]
    __nv_bfloat16* __restrict__ C,         // 2026-09-25: [M, N]
    unsigned int M, unsigned int N, unsigned int K
) {
    unsigned int row = blockIdx.y * 32 + threadIdx.y;
    unsigned int col = blockIdx.x * 32 + threadIdx.x;
    if (row >= M || col >= N) return;

    float acc = 0.0f;
    for (unsigned int k = 0; k < K; ++k) {
        acc += bf16_to_f32(A[row * K + k]) * bf16_to_f32(B[col * K + k]);
    }
    acc += bf16_to_f32(bias[col]);
    C[row * N + col] = f32_to_bf16(acc);
}

// 2026-09-25: C[M,N] = A[M,K] @ B[K,N] + bias[N] (B not transposed). Grid (ceil(N/32), ceil(M/32)),
// block (32, 32). No Rust code looks this kernel up.
extern "C" __global__
void vision_gemm_bias_nn(
    const __nv_bfloat16* __restrict__ A,   // 2026-09-25: [M, K]
    const __nv_bfloat16* __restrict__ B,   // 2026-09-25: [K, N]
    const __nv_bfloat16* __restrict__ bias,// 2026-09-25: [N]
    __nv_bfloat16* __restrict__ C,         // 2026-09-25: [M, N]
    unsigned int M, unsigned int N, unsigned int K
) {
    unsigned int row = blockIdx.y * 32 + threadIdx.y;
    unsigned int col = blockIdx.x * 32 + threadIdx.x;
    if (row >= M || col >= N) return;

    float acc = 0.0f;
    for (unsigned int k = 0; k < K; ++k) {
        acc += bf16_to_f32(A[row * K + k]) * bf16_to_f32(B[k * N + col]);
    }
    acc += bf16_to_f32(bias[col]);
    C[row * N + col] = f32_to_bf16(acc);
}

// 2026-09-25: Row-broadcast bias add in place: C[m,n] += bias[n]. dense_gemm_bf16_pipelined
// has no bias epilogue, so vit_block.rs runs it and then this kernel.



extern "C" __global__ void vision_add_bias(
    __nv_bfloat16* __restrict__ C,         // 2026-09-25: [M, N], in place
    const __nv_bfloat16* __restrict__ bias,// 2026-09-25: [N]
    unsigned int M, unsigned int N
) {
    unsigned long long idx = (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= (unsigned long long)M * N) return;
    unsigned int col = (unsigned int)(idx % N);
    C[idx] = f32_to_bf16(bf16_to_f32(C[idx]) + bf16_to_f32(bias[col]));
}

// 2026-09-25: LayerNorm in place: x = (x - mean) / sqrt(var + eps) * w + b.
// One block per row; the launchers use min(D, 1024) threads.
extern "C" __global__
void vision_layer_norm(
    __nv_bfloat16* __restrict__ x,         // 2026-09-25: [N, D], in place
    const __nv_bfloat16* __restrict__ w,   // 2026-09-25: [D]
    const __nv_bfloat16* __restrict__ b,   // 2026-09-25: [D]
    unsigned int N, unsigned int D,
    float eps
) {
    unsigned int row = blockIdx.x;
    if (row >= N) return;

    __nv_bfloat16* row_ptr = x + row * D;


    float sum = 0.0f;
    for (unsigned int i = threadIdx.x; i < D; i += blockDim.x) {
        sum += bf16_to_f32(row_ptr[i]);
    }

    for (int offset = 16; offset > 0; offset >>= 1)
        sum += __shfl_down_sync(0xffffffff, sum, offset);
    __shared__ float smem_sum[32];
    if (threadIdx.x % 32 == 0) smem_sum[threadIdx.x / 32] = sum;
    __syncthreads();
    if (threadIdx.x < (blockDim.x + 31) / 32) sum = smem_sum[threadIdx.x];
    for (int offset = 16; offset > 0; offset >>= 1)
        sum += __shfl_down_sync(0xffffffff, sum, offset);
    __shared__ float mean_val;
    if (threadIdx.x == 0) mean_val = sum / D;
    __syncthreads();


    float var = 0.0f;
    for (unsigned int i = threadIdx.x; i < D; i += blockDim.x) {
        float diff = bf16_to_f32(row_ptr[i]) - mean_val;
        var += diff * diff;
    }
    for (int offset = 16; offset > 0; offset >>= 1)
        var += __shfl_down_sync(0xffffffff, var, offset);
    __shared__ float smem_var[32];
    if (threadIdx.x % 32 == 0) smem_var[threadIdx.x / 32] = var;
    __syncthreads();
    if (threadIdx.x < (blockDim.x + 31) / 32) var = smem_var[threadIdx.x];
    for (int offset = 16; offset > 0; offset >>= 1)
        var += __shfl_down_sync(0xffffffff, var, offset);
    __shared__ float inv_std;
    if (threadIdx.x == 0) inv_std = rsqrtf(var / D + eps);
    __syncthreads();


    for (unsigned int i = threadIdx.x; i < D; i += blockDim.x) {
        float val = (bf16_to_f32(row_ptr[i]) - mean_val) * inv_std;
        val = val * bf16_to_f32(w[i]) + bf16_to_f32(b[i]);
        row_ptr[i] = f32_to_bf16(val);
    }
}

// 2026-09-25: Residual add in place: dst[i] += src[i].
// Grid (ceil(n/256)); the indexing assumes 256 threads per block.
extern "C" __global__
void vision_add_inplace(
    __nv_bfloat16* __restrict__ dst,
    const __nv_bfloat16* __restrict__ src,
    unsigned int n
) {
    unsigned int i = blockIdx.x * 256 + threadIdx.x;
    if (i < n) dst[i] = f32_to_bf16(bf16_to_f32(dst[i]) + bf16_to_f32(src[i]));
}

// 2026-09-25: Positional embeddings are added with vision_add_inplace (patch_embed.rs).



// 2026-09-25: GELU in place, tanh approximation (below).
// Grid (ceil(n/256)); the indexing assumes 256 threads per block.
#define SQRT2F 1.41421356237f
extern "C" __global__
void vision_gelu(
    __nv_bfloat16* __restrict__ x,
    unsigned int n
) {
    unsigned int i = blockIdx.x * 256 + threadIdx.x;
    if (i >= n) return;
    float v = bf16_to_f32(x[i]);
    // 2026-09-25: tanh-approximation GELU, PyTorch's "gelu_pytorch_tanh", which
    // the vision config of the Qwen3.6-35B-A3B checkpoint declares as its hidden_act:
    //   GELU_tanh(x) = 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))



    const float SQRT_2_OVER_PI = 0.7978845608028654f;
    const float COEFF = 0.044715f;
    float inner = SQRT_2_OVER_PI * (v + COEFF * v * v * v);
    x[i] = f32_to_bf16(0.5f * v * (1.0f + tanhf(inner)));
}

// 2026-09-25: Scaled dot-product attention with 2D rotary embedding on Q and K, one
// warp per (query token, head). QKV is [seq, 3*H*D] fused; rope_cos and rope_sin
// are [seq, D] BF16 tables. For token t and dim d:
//   rotated[d] = x[d] * cos[t,d] - x[d + D/2] * sin[t,d]   (d < D/2)
//   rotated[d] = x[d] * cos[t,d] + x[d - D/2] * sin[t,d]   (d >= D/2)
// Grid (seq, num_heads), block (32). Dynamic shared memory holds seq + D floats.
// vit_block.rs uses it when METRALE_VISION_ATTN_LEGACY is set or
// vit_rope_deinterleave does not resolve.



extern "C" __global__
void vision_attention_rope(
    const __nv_bfloat16* __restrict__ QKV, // 2026-09-25: [seq, 3*H*D]
    __nv_bfloat16* __restrict__ O,          // 2026-09-25: [seq, H*D]
    const __nv_bfloat16* __restrict__ rope_cos, // 2026-09-25: [seq, D]
    const __nv_bfloat16* __restrict__ rope_sin, // 2026-09-25: [seq, D]
    unsigned int seq, unsigned int H, unsigned int D
) {
    unsigned int qi = blockIdx.x;
    unsigned int h  = blockIdx.y;
    if (qi >= seq || h >= H) return;

    unsigned int stride_qkv = 3 * H * D;
    float scale = rsqrtf((float)D);
    unsigned int half_D = D / 2;


    const __nv_bfloat16* Q_row = QKV + qi * stride_qkv + h * D;

    // 2026-09-25: Shared memory: seq floats of scores, then this query's rotated Q (D floats).

    extern __shared__ float smem[]; // 2026-09-25: [seq + D]
    float* scores  = smem;
    float* q_rope  = smem + seq;



    // 2026-09-25: 1. Rotate Q once per query; every key iteration reads it.

    for (unsigned int d = threadIdx.x; d < D; d += blockDim.x) {
        float q_val   = bf16_to_f32(Q_row[d]);
        float q_part  = (d < half_D) ? bf16_to_f32(Q_row[d + half_D])
                                     : bf16_to_f32(Q_row[d - half_D]);
        float q_rot   = (d < half_D) ? -q_part : q_part;
        float q_cos_v = bf16_to_f32(rope_cos[qi * D + d]);
        float q_sin_v = bf16_to_f32(rope_sin[qi * D + d]);
        q_rope[d] = q_val * q_cos_v + q_rot * q_sin_v;
    }
    __syncthreads();

    // 2026-09-25: 2. Scores: rotate each K on the fly and dot it with the cached Q.
    for (unsigned int kj = 0; kj < seq; ++kj) {
        const __nv_bfloat16* K_row = QKV + kj * stride_qkv + H * D + h * D;
        float dot = 0.0f;
        for (unsigned int d = threadIdx.x; d < D; d += blockDim.x) {
            float k_val   = bf16_to_f32(K_row[d]);
            float k_part  = (d < half_D) ? bf16_to_f32(K_row[d + half_D])
                                         : bf16_to_f32(K_row[d - half_D]);
            float k_rot   = (d < half_D) ? -k_part : k_part;
            float k_cos_v = bf16_to_f32(rope_cos[kj * D + d]);
            float k_sin_v = bf16_to_f32(rope_sin[kj * D + d]);
            float k_r     = k_val * k_cos_v + k_rot * k_sin_v;
            dot += q_rope[d] * k_r;
        }
        for (int offset = 16; offset > 0; offset >>= 1)
            dot += __shfl_down_sync(0xffffffff, dot, offset);
        if (threadIdx.x == 0) scores[kj] = dot * scale;
        __syncthreads();
    }

    // 2026-09-25: 3. Softmax over the scores, on thread 0.
    if (threadIdx.x == 0) {
        float max_s = -1e30f;
        for (unsigned int j = 0; j < seq; ++j) max_s = fmaxf(max_s, scores[j]);
        float sum_exp = 0.0f;
        for (unsigned int j = 0; j < seq; ++j) {
            scores[j] = expf(scores[j] - max_s);
            sum_exp += scores[j];
        }
        float inv_sum = 1.0f / sum_exp;
        for (unsigned int j = 0; j < seq; ++j) scores[j] *= inv_sum;
    }
    __syncthreads();

    // 2026-09-25: 4. Probability-weighted sum of V.
    for (unsigned int d = threadIdx.x; d < D; d += blockDim.x) {
        float out = 0.0f;
        for (unsigned int vj = 0; vj < seq; ++vj) {
            const __nv_bfloat16* V_row = QKV + vj * stride_qkv + 2 * H * D + h * D;
            out += scores[vj] * bf16_to_f32(V_row[d]);
        }
        O[qi * H * D + h * D + d] = f32_to_bf16(out);
    }
}

// 2026-09-25: GEMM-based ViT SDPA, used by vit_block.rs unless METRALE_VISION_ATTN_LEGACY is
// set or these kernels do not resolve. Per image: vit_rope_deinterleave, then per
// head dense_gemm_bf16_f32out (raw Q K^T) -> vit_softmax_rows ->
// dense_gemm_bf16_pipelined (P V) -> vit_scatter_head. Same math as
// vision_attention_rope (rotate-half RoPE on Q and K only, scale rsqrt(D),
// non-causal max-subtracted softmax), except that P is rounded to BF16 before
// the P V GEMM.



// 2026-09-25: (A) QKV[seq, 3*H*D] -> head-contiguous rotated Qr, Kr [H, seq, D] and
//     transposed Vt [H, D, seq], so the second GEMM (which computes A B^T)
//     yields P V. One thread per (token, head, d) element.
//     Grid (ceil(seq*D/256), H), block (256).
extern "C" __global__
void vit_rope_deinterleave(
    const __nv_bfloat16* __restrict__ QKV,      // 2026-09-25: [seq, 3*H*D]
    __nv_bfloat16* __restrict__ Qr,             // 2026-09-25: [H, seq, D]
    __nv_bfloat16* __restrict__ Kr,             // 2026-09-25: [H, seq, D]
    __nv_bfloat16* __restrict__ Vt,             // 2026-09-25: [H, D, seq]
    const __nv_bfloat16* __restrict__ rope_cos, // 2026-09-25: [seq, D]
    const __nv_bfloat16* __restrict__ rope_sin, // 2026-09-25: [seq, D]
    unsigned int seq, unsigned int H, unsigned int D)
{
    unsigned int h   = blockIdx.y;
    unsigned int lin = blockIdx.x * blockDim.x + threadIdx.x;
    if (h >= H || lin >= seq * D) return;
    unsigned int tok = lin / D;
    unsigned int d   = lin % D;
    unsigned int half_D = D / 2;

    unsigned int stride_qkv = 3u * H * D;
    const __nv_bfloat16* Q_row = QKV + (size_t)tok * stride_qkv + 0u * H * D + h * D;
    const __nv_bfloat16* K_row = QKV + (size_t)tok * stride_qkv + 1u * H * D + h * D;
    const __nv_bfloat16* V_row = QKV + (size_t)tok * stride_qkv + 2u * H * D + h * D;

    float cos_v = bf16_to_f32(rope_cos[(size_t)tok * D + d]);
    float sin_v = bf16_to_f32(rope_sin[(size_t)tok * D + d]);

    // 2026-09-25: Rotate-half, the same formula as vision_attention_rope.
    float qv    = bf16_to_f32(Q_row[d]);
    float qpart = (d < half_D) ? bf16_to_f32(Q_row[d + half_D])
                               : bf16_to_f32(Q_row[d - half_D]);
    float qrot  = (d < half_D) ? -qpart : qpart;
    float q_r   = qv * cos_v + qrot * sin_v;


    float kv    = bf16_to_f32(K_row[d]);
    float kpart = (d < half_D) ? bf16_to_f32(K_row[d + half_D])
                               : bf16_to_f32(K_row[d - half_D]);
    float krot  = (d < half_D) ? -kpart : kpart;
    float k_r   = kv * cos_v + krot * sin_v;


    float v_v   = bf16_to_f32(V_row[d]);

    size_t hc = (size_t)h * seq * D + (size_t)tok * D + d;
    Qr[hc] = f32_to_bf16(q_r);
    Kr[hc] = f32_to_bf16(k_r);

    Vt[(size_t)h * D * seq + (size_t)d * seq + (size_t)tok] = f32_to_bf16(v_v);
}

// 2026-09-25: (B) Row softmax over raw scores S[seq, seq] (FP32) -> P[seq, seq] (BF16),
//     with the rsqrt(D) scale applied here. One block per row, three passes
//     (max, sum of exp, normalise). `red` has 8 slots, so blockDim.x <= 256.
//     Grid (seq), block (256).

extern "C" __global__
void vit_softmax_rows(
    const float* __restrict__ S,        // 2026-09-25: [seq, seq] FP32 raw scores
    __nv_bfloat16* __restrict__ P,      // 2026-09-25: [seq, seq] BF16 probabilities
    unsigned int seq, unsigned int D)
{
    unsigned int row = blockIdx.x;
    if (row >= seq) return;
    const float* srow = S + (size_t)row * seq;
    __nv_bfloat16* prow = P + (size_t)row * seq;
    float scale = rsqrtf((float)D);

    __shared__ float red[256 / 32];
    unsigned int tid  = threadIdx.x;
    unsigned int lane = tid & 31u, warp = tid >> 5;


    float m = -1e30f;
    for (unsigned int j = tid; j < seq; j += blockDim.x)
        m = fmaxf(m, srow[j] * scale);
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


    float s = 0.0f;
    for (unsigned int j = tid; j < seq; j += blockDim.x)
        s += expf(srow[j] * scale - row_max);
    for (int o = 16; o > 0; o >>= 1) s += __shfl_down_sync(0xffffffff, s, o);
    if (lane == 0) red[warp] = s;
    __syncthreads();
    if (tid == 0) {
        float ss = 0.0f;
        for (unsigned int w = 0; w < blockDim.x / 32; ++w) ss += red[w];
        red[0] = (ss > 0.0f) ? (1.0f / ss) : 0.0f;
    }
    __syncthreads();
    float inv_sum = red[0];


    for (unsigned int j = tid; j < seq; j += blockDim.x)
        prow[j] = f32_to_bf16(expf(srow[j] * scale - row_max) * inv_sum);
}

// 2026-09-25: (C) Copy contiguous Oh[seq, D] into the interleaved O[seq, dst_stride] head slot.
//     Grid (ceil(seq*D/256)), block (256).
extern "C" __global__
void vit_scatter_head(
    const __nv_bfloat16* __restrict__ Oh,  // 2026-09-25: [seq, D]
    __nv_bfloat16* __restrict__ O,          // 2026-09-25: [seq, dst_stride], head-slot base
    unsigned int seq, unsigned int D, unsigned int dst_stride)
{
    unsigned int lin = blockIdx.x * blockDim.x + threadIdx.x;
    if (lin >= seq * D) return;
    unsigned int tok = lin / D, d = lin % D;
    O[(size_t)tok * dst_stride + d] = Oh[(size_t)tok * D + d];
}

// 2026-09-25: Spatial merge, src [P, D] -> dst [P/m^2, m^2*D] with m = merge_size:
// each output token concatenates the D features of an m x m group of patches
// (P = grid_h * grid_w). Grid (P/m^2); merger.rs launches min(m^2*D, 1024)
// threads, and each strides over the m^2*D outputs.
extern "C" __global__
void vision_spatial_merge(
    const __nv_bfloat16* __restrict__ src, // 2026-09-25: [P, D]
    __nv_bfloat16* __restrict__ dst,        // 2026-09-25: [P/m^2, m^2*D]
    unsigned int grid_h, unsigned int grid_w,
    unsigned int D, unsigned int merge_size
) {
    unsigned int out_idx = blockIdx.x;
    unsigned int m  = merge_size;
    unsigned int m2 = m * m;
    unsigned int out_gh = grid_h / m;
    unsigned int out_gw = grid_w / m;
    if (out_idx >= out_gh * out_gw) return;

    unsigned int oh = out_idx / out_gw;
    unsigned int ow = out_idx % out_gw;

    // 2026-09-25: Gather the m x m source patches and concatenate their D features.
    for (unsigned int pi = threadIdx.x; pi < m2 * D; pi += blockDim.x) {
        unsigned int p_local = pi / D;
        unsigned int d       = pi % D;
        unsigned int ph = oh * m + p_local / m;
        unsigned int pw = ow * m + p_local % m;
        unsigned int src_idx = (ph * grid_w + pw) * D + d;
        dst[out_idx * (m2 * D) + pi] = src[src_idx];
    }
}

// 2026-09-25: Copy FP32 -> BF16; patch_embed.rs converts the uploaded FP32 pixels with it.
// Grid (ceil(n/256)); the indexing assumes 256 threads per block.
extern "C" __global__
void vision_f32_to_bf16(
    const float* __restrict__ src,
    __nv_bfloat16* __restrict__ dst,
    unsigned int n
) {
    unsigned int i = blockIdx.x * 256 + threadIdx.x;
    if (i < n) dst[i] = f32_to_bf16(src[i]);
}

// 2026-09-25: Copy BF16 -> BF16 (utils.rs `gpu_copy_bf16`).
// Grid (ceil(n/256)); the indexing assumes 256 threads per block.
extern "C" __global__
void vision_bf16_copy(
    const __nv_bfloat16* __restrict__ src,
    __nv_bfloat16* __restrict__ dst,
    unsigned int n
) {
    unsigned int i = blockIdx.x * 256 + threadIdx.x;
    if (i < n) dst[i] = src[i];
}
