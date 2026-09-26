// SPDX-License-Identifier: MIT OR Apache-2.0

// 2026-09-25: Glue kernels for the KDA layer: the sigmoid-gated output RMSNorm, the q|k|v
// pack and split, the beta sigmoid, and a device-side fill.
//
// Owner: gb10 kernels.
// Invariants:
// - kda_o_norm_gated_*: one block per row of head_dim elements (blockIdx.x is the row);
//   out[i] = x[i] / sqrt(mean(x^2) + eps) * weight[i] * sigmoid(gate[i]). The block must
//   be a whole number of warps and at most 1024 threads: the reductions shuffle with a
//   full mask into red[32]. The host launches block = head_dim, which
//   Glm5NextKdaConfig::validate requires to be 128.
// - kda_split_widen and kda_pack_qkv_bf16: blockIdx.y is the token and each thread one
//   channel; tokens t >= T are not written.
//
// The norm does not reuse gated_rms_norm_f32_input (rms_norm.cu): that kernel applies
// SiLU to the gate, g / (1 + expf(-g)), and this one applies sigmoid.


#include <cuda_bf16.h>





#define KDA_ONORM_BODY(GATE_LD, W_LD, OUT_ST)                                                 \
    const unsigned int row = blockIdx.x;                                                      \
    const unsigned int tid = threadIdx.x;                                                     \
    const float* x = input + (size_t)row * head_dim;                                          \
    float acc = 0.0f;                                                                         \
    for (unsigned int i = tid; i < head_dim; i += blockDim.x) { float f = x[i]; acc += f * f; }\
    __shared__ float red[32];                                                                 \
    for (int off = 16; off > 0; off >>= 1) acc += __shfl_down_sync(0xffffffff, acc, off);     \
    if ((tid & 31) == 0) red[tid >> 5] = acc;                                                 \
    __syncthreads();                                                                          \
    if (tid < 32) {                                                                           \
        float v = (tid < ((blockDim.x + 31) / 32)) ? red[tid] : 0.0f;                          \
        for (int off = 16; off > 0; off >>= 1) v += __shfl_down_sync(0xffffffff, v, off);      \
        if (tid == 0) red[0] = v;                                                             \
    }                                                                                         \
    __syncthreads();                                                                          \
    const float inv = rsqrtf(red[0] / (float)head_dim + eps);                                 \
    for (unsigned int i = tid; i < head_dim; i += blockDim.x) {                                \
        const float g = (GATE_LD);                                                            \
        const float s = 1.0f / (1.0f + __expf(-g));                    \
        OUT_ST(x[i] * inv * (W_LD) * s);                                                      \
    }

// 2026-09-25: The entry point glm5next_kda resolves: FP32 input; BF16 gate, weight and output.
extern "C" __global__ void kda_o_norm_gated_bf16(
    const float* __restrict__ input,
    const __nv_bfloat16* __restrict__ gate,
    const __nv_bfloat16* __restrict__ weight,
    __nv_bfloat16* __restrict__ output,
    unsigned int head_dim,
    float eps
) {
#define KDA_ONORM_OUT_BF16(val) output[(size_t)row * head_dim + i] = __float2bfloat16(val)
    KDA_ONORM_BODY(__bfloat162float(gate[(size_t)row * head_dim + i]),
                   __bfloat162float(weight[i]),
                   KDA_ONORM_OUT_BF16)
#undef KDA_ONORM_OUT_BF16
}


extern "C" __global__ void kda_o_norm_gated_f32(
    const float* __restrict__ input,
    const float* __restrict__ gate,
    const float* __restrict__ weight,
    float* __restrict__ output,
    unsigned int head_dim,
    float eps
) {
#define KDA_ONORM_OUT_F32(val) output[(size_t)row * head_dim + i] = (val)
    KDA_ONORM_BODY(gate[(size_t)row * head_dim + i], weight[i], KDA_ONORM_OUT_F32)
#undef KDA_ONORM_OUT_F32
}

// 2026-09-25: Splits the [T, 3 * qkv] BF16 q|k|v rows into three FP32 [T_pad, qkv] buffers;
// BF16 to FP32 is exact. Tokens t >= T are not written, so the pad tail keeps what the
// caller put there: zero in production, or a poison value from the pad-guard test through
// prefill_with_pad_fill.




extern "C" __global__ void kda_split_widen(
    const __nv_bfloat16* __restrict__ src,
    float* __restrict__ q,
    float* __restrict__ k,
    float* __restrict__ v,
    unsigned int T,
    unsigned int qkv
) {
    const unsigned int ch = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int t = blockIdx.y;
    if (ch >= qkv || t >= T) return;
    const size_t s = (size_t)t * 3 * qkv + ch;
    const size_t d = (size_t)t * qkv + ch;
    q[d] = __bfloat162float(src[s]);
    k[d] = __bfloat162float(src[s + qkv]);
    v[d] = __bfloat162float(src[s + 2 * qkv]);
}



extern "C" __global__ void kda_fill_f32(float* __restrict__ dst, unsigned int n, float value) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) dst[i] = value;
}

// 2026-09-25: beta = sigmoid(b_proj(hidden)), BF16 in and FP32 out; the KDA kernels take
// beta already sigmoided.
extern "C" __global__ void kda_sigmoid_bf16_f32(
    const __nv_bfloat16* __restrict__ src,
    float* __restrict__ dst,
    unsigned int n
) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        float x = __bfloat162float(src[i]);
        dst[i] = 1.0f / (1.0f + __expf(-x));
    }
}

// 2026-09-25: Interleaves three [T, qkv] projections into the [T, 3 * qkv] q|k|v layout the
// depthwise conv reads. The projections are separate GEMMs because dense_gemm_bf16 writes
// C[row * N + col] with no output stride: aimed at offsets inside one [T, 3 * qkv] buffer
// they would overwrite each other for T > 1, and agree at T = 1, where a decode-only test
// would not see it.


extern "C" __global__ void kda_pack_qkv_bf16(
    const __nv_bfloat16* __restrict__ q,
    const __nv_bfloat16* __restrict__ k,
    const __nv_bfloat16* __restrict__ v,
    __nv_bfloat16* __restrict__ dst,
    unsigned int T,
    unsigned int qkv
) {
    const unsigned int ch = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int t = blockIdx.y;
    if (ch >= qkv || t >= T) return;
    const size_t s = (size_t)t * qkv + ch;
    const size_t d = (size_t)t * 3 * qkv + ch;
    dst[d] = q[s];
    dst[d + qkv] = k[s];
    dst[d + 2 * qkv] = v[s];
}
