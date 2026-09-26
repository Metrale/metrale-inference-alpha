// SPDX-License-Identifier: MIT OR Apache-2.0

// 2026-09-25: Vision encoder (ViT) kernels for qwen3-vl-30b-a3b, also compiled for qwen3.5-122b-a10b
// and qwen3.5-397b-a17b through their [sources] use.
// Owner: gb10 kernels (qwen3-vl-30b-a3b).
// Invariants: tensors are BF16 in memory (vision_f32_to_bf16 reads FP32); arithmetic is FP32.

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <math.h>



__device__ inline float bf16_to_f32(__nv_bfloat16 v) {
    return __bfloat162float(v);
}
__device__ inline __nv_bfloat16 f32_to_bf16(float v) {
    return __float2bfloat16(v);
}

// 2026-09-25: C[M, N] = A[M, K] B[N, K]^T + bias[N], one output per thread. Grid (ceil(N / 32),
// ceil(M / 32)), block (32, 32) (vision_encoder/enc_impl/vit_block.rs vit_gemm_bias).
extern "C" __global__
void vision_gemm_bias(
    const __nv_bfloat16* __restrict__ A,
    const __nv_bfloat16* __restrict__ B,
    const __nv_bfloat16* __restrict__ bias,
    __nv_bfloat16* __restrict__ C,
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

// 2026-09-25: C[M, N] = A[M, K] B[K, N] + bias[N], with vision_gemm_bias's thread mapping.

extern "C" __global__
void vision_gemm_bias_nn(
    const __nv_bfloat16* __restrict__ A,
    const __nv_bfloat16* __restrict__ B,
    const __nv_bfloat16* __restrict__ bias,
    __nv_bfloat16* __restrict__ C,
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

// 2026-09-25: In-place LayerNorm per row, x = (x - mean) / sqrt(var + eps) * w + b. The second reduction stage leaves
// lanes past the last warp holding stale partial sums, so it is correct only for 32 full warps; callers pass min(D, 1024).
extern "C" __global__
void vision_layer_norm(
    __nv_bfloat16* __restrict__ x,
    const __nv_bfloat16* __restrict__ w,
    const __nv_bfloat16* __restrict__ b,
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

// 2026-09-25: dst[i] += src[i]; the index is blockIdx.x * 256 + threadIdx.x, so the block must be 256.

extern "C" __global__
void vision_add_inplace(
    __nv_bfloat16* __restrict__ dst,
    const __nv_bfloat16* __restrict__ src,
    unsigned int n
) {
    unsigned int i = blockIdx.x * 256 + threadIdx.x;
    if (i < n) dst[i] = f32_to_bf16(bf16_to_f32(dst[i]) + bf16_to_f32(src[i]));
}





// 2026-09-25: In-place tanh GELU, 0.5 * x * (1 + tanh(sqrt(2 / pi) * (x + 0.044715 * x^3))), the
// gelu_pytorch_tanh that the checkpoints' vision_config hidden_act names. Block 256, as vision_add_inplace.
#define SQRT2F 1.41421356237f
extern "C" __global__
void vision_gelu(
    __nv_bfloat16* __restrict__ x,
    unsigned int n
) {
    unsigned int i = blockIdx.x * 256 + threadIdx.x;
    if (i >= n) return;
    float v = bf16_to_f32(x[i]);






    const float SQRT_2_OVER_PI = 0.7978845608028654f;
    const float COEFF = 0.044715f;
    float inner = SQRT_2_OVER_PI * (v + COEFF * v * v * v);
    x[i] = f32_to_bf16(0.5f * v * (1.0f + tanhf(inner)));
}

// 2026-09-25: Attention of one query token for one head per 32-thread block (a single warp), with
// rotary embedding applied to Q and K from the [seq, D] rope_cos / rope_sin tables:
//   rotated[d] = x[d] * cos[t, d] - x[d + D/2] * sin[t, d]   for d < D/2
//   rotated[d] = x[d] * cos[t, d] + x[d - D/2] * sin[t, d]   for d >= D/2
// Each QKV row is [Q | K | V], H * D values each. Every query attends to all seq keys (no mask).
// Dynamic shared memory holds seq scores and the D rotated Q values, (seq + D) floats
// (enc_impl/vit_block.rs). Grid (seq, H), block 32.




extern "C" __global__
void vision_attention_rope(
    const __nv_bfloat16* __restrict__ QKV,
    __nv_bfloat16* __restrict__ O,
    const __nv_bfloat16* __restrict__ rope_cos,
    const __nv_bfloat16* __restrict__ rope_sin,
    unsigned int seq, unsigned int H, unsigned int D
) {
    unsigned int qi = blockIdx.x;
    unsigned int h  = blockIdx.y;
    if (qi >= seq || h >= H) return;

    unsigned int stride_qkv = 3 * H * D;
    float scale = rsqrtf((float)D);
    unsigned int half_D = D / 2;


    const __nv_bfloat16* Q_row = QKV + qi * stride_qkv + h * D;



    extern __shared__ float smem[];
    float* scores  = smem;
    float* q_rope  = smem + seq;





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


    for (unsigned int d = threadIdx.x; d < D; d += blockDim.x) {
        float out = 0.0f;
        for (unsigned int vj = 0; vj < seq; ++vj) {
            const __nv_bfloat16* V_row = QKV + vj * stride_qkv + 2 * H * D + h * D;
            out += scores[vj] * bf16_to_f32(V_row[d]);
        }
        O[qi * H * D + h * D + d] = f32_to_bf16(out);
    }
}

// 2026-09-25: Spatial merge from src [grid_h * grid_w, D] to dst [P / m^2, m^2 * D] (m = merge_size):
// output row (oh, ow) concatenates the m x m source patches (oh * m + i, ow * m + j) in row-major
// order. Grid (P / m^2), block min(m^2 * D, 1024) (enc_impl/merger.rs).

extern "C" __global__
void vision_spatial_merge(
    const __nv_bfloat16* __restrict__ src,
    __nv_bfloat16* __restrict__ dst,
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


    for (unsigned int pi = threadIdx.x; pi < m2 * D; pi += blockDim.x) {
        unsigned int p_local = pi / D;
        unsigned int d       = pi % D;
        unsigned int ph = oh * m + p_local / m;
        unsigned int pw = ow * m + p_local % m;
        unsigned int src_idx = (ph * grid_w + pw) * D + d;
        dst[out_idx * (m2 * D) + pi] = src[src_idx];
    }
}

// 2026-09-25: FP32 to BF16, one element per thread; block 256, as vision_add_inplace.

extern "C" __global__
void vision_f32_to_bf16(
    const float* __restrict__ src,
    __nv_bfloat16* __restrict__ dst,
    unsigned int n
) {
    unsigned int i = blockIdx.x * 256 + threadIdx.x;
    if (i < n) dst[i] = f32_to_bf16(src[i]);
}

// 2026-09-25: BF16 copy, one element per thread; block 256, as vision_add_inplace.

extern "C" __global__
void vision_bf16_copy(
    const __nv_bfloat16* __restrict__ src,
    __nv_bfloat16* __restrict__ dst,
    unsigned int n
) {
    unsigned int i = blockIdx.x * 256 + threadIdx.x;
    if (i < n) dst[i] = src[i];
}
