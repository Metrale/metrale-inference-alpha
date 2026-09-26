// SPDX-License-Identifier: MIT OR Apache-2.0
// provenance-id: 526f6e616c6420522e205374657369616b

// 2026-09-25: DeepSeek-V4.1 Flash engram: the gate that adds the engram value to each mHC stream, and the
// single-token `wkv` GEMV over Q2_K blocks.
//
// Engram (the layers in `engram_layer_ids`) gathers `n_hash_cols` rows of a hashed n-gram table per token,
// projects their concatenation through `wkv` into one key per hyper-connection stream plus one shared value,
// and adds `gate * value` to each stream (model-arch engram_v41.rs). The gate follows the CPU reference
// `deepseek_v41_ref::engram::gate_and_add`:
//   rstd = rsqrt(mean(h^2) + eps) * rsqrt(mean(key^2) + eps)
//   dot  = sum_d h[d] * (q_w[c,d] * k_w[c,d]) * key[d] * rstd * dim^-0.5
//   g    = copysign(sqrt(max(|dot|, 1e-6)), dot);  gate = sigmoid(g)
//   h   += gate * value
// `qk` is the product q_w * k_w (`upload_qk` in engram_v41.rs); the gate uses the two weights only that way.
//
// streams: [T, hc, H] FP32 highway, updated in place; kv: [T, H * (hc + 1)] bf16, the hc keys then the
// value; qk: [hc, H] f32. Grid: (T, hc). Block: 256, one block per (token, stream).
// Owner: gb10 kernels (deepseek-v4-flash); launched by model-arch `engram_v41`.
// Invariants: none beyond the types.





#include <cuda_bf16.h>

#define ENGRAM_BLOCK 256

__device__ __forceinline__ float engram_block_sum(float v, float* red) {
    const unsigned int tid = threadIdx.x;
    red[tid] = v;
    __syncthreads();
    for (unsigned int s = ENGRAM_BLOCK / 2; s > 0; s >>= 1) {
        if (tid < s) red[tid] += red[tid + s];
        __syncthreads();
    }
    const float r = red[0];
    __syncthreads();
    return r;
}

extern "C" __global__ void engram_v41_gate(
    float* __restrict__ streams,
    const __nv_bfloat16* __restrict__ kv,
    const float* __restrict__ qk,
    const unsigned int hidden_size,
    const unsigned int hc_mult,
    const float norm_eps
) {
    const unsigned int t = blockIdx.x;
    const unsigned int c = blockIdx.y;
    const unsigned int tid = threadIdx.x;
    const unsigned int H = hidden_size;
    __shared__ float red[ENGRAM_BLOCK];

    float* h = streams + ((size_t)t * hc_mult + c) * H;
    const __nv_bfloat16* key = kv + (size_t)t * H * (hc_mult + 1) + (size_t)c * H;
    const __nv_bfloat16* value = kv + (size_t)t * H * (hc_mult + 1) + (size_t)hc_mult * H;
    const float* w = qk + (size_t)c * H;

    float hh = 0.0f, kk = 0.0f, dot = 0.0f;
    for (unsigned int d = tid; d < H; d += ENGRAM_BLOCK) {
        const float hv = h[d];
        const float kvv = __bfloat162float(key[d]);
        hh += hv * hv;
        kk += kvv * kvv;
        dot += hv * w[d] * kvv;
    }
    hh = engram_block_sum(hh, red);
    kk = engram_block_sum(kk, red);
    dot = engram_block_sum(dot, red);

    const float inv_h = rsqrtf(hh / (float)H + norm_eps);
    const float inv_k = rsqrtf(kk / (float)H + norm_eps);
    const float scaled = dot * inv_h * inv_k * rsqrtf((float)H);
    const float g = copysignf(sqrtf(fmaxf(fabsf(scaled), 1e-6f)), scaled);
    const float gate = 1.0f / (1.0f + expf(-g));

    for (unsigned int d = tid; d < H; d += ENGRAM_BLOCK) {
        h[d] += gate * __bfloat162float(value[d]);
    }
}

// 2026-09-25: The `wkv` projection of one token read straight from the Q2_K blocks the GGUF ships (84 bytes
// per 256 weights), instead of the bf16 expansion the loader keeps for the prefill GEMM. It matches
// common/dense_gemv_bf16.cu over that expansion bit for bit (engram_v41_tests.rs
// `gpu_engram_wkv_q2k_gemv_is_bitwise_the_bf16_gemv_over_the_expansion`): the same 64 threads per output,
// the same uint4 groups of 8 in the same order, each weight dequantised as common/dequant_gguf_bf16.cu's
// dequant_q2_k_to_bf16 does (dl = d * (sc & 0xF), ml = dmin * (sc >> 4), bf16(dl * code - ml), both
// built with --fmad=false) and rounded to bf16 before the product, and the same shuffle tree and two-warp sum.
//
// A: [1, K] bf16 (the concatenated engram rows); B: [N][K / 256] Q2_K blocks
// (GGUF row-major, dims [K, N]); C: [1, N] bf16. K % 256 == 0.
// Grid: (ceil(N / 4), 1, 1)  Block: (256, 1, 1)


#include <cuda_fp16.h>

__device__ __forceinline__ float engram_q2k_f16(const unsigned char* p) {
    unsigned short bits = (unsigned short)p[0] | ((unsigned short)p[1] << 8);
    return __half2float(__ushort_as_half(bits));
}

extern "C" __global__ void engram_v41_wkv_q2k_gemv(
    const __nv_bfloat16* __restrict__ A,
    const unsigned char* __restrict__ B,
    __nv_bfloat16* __restrict__ C,
    unsigned int N,
    unsigned int K
) {
    const unsigned int threads_per_out = 64;
    const unsigned int local_out = threadIdx.x / threads_per_out;
    const unsigned int lane = threadIdx.x % threads_per_out;
    const unsigned int n = blockIdx.x * 4 + local_out;
    if (n >= N) return;

    float acc = 0.0f;
    const unsigned int K_VEC = K / 8;
    const uint4* A_vec = (const uint4*)A;
    const unsigned char* row = B + (size_t)n * (size_t)(K / 256) * 84u;

    for (unsigned int kv = lane; kv < K_VEC; kv += threads_per_out) {
        const uint4 a_data = A_vec[kv];
        // 2026-09-25: Group g of 8 elements y = 8g..8g+7 inside block kv / 32:
        // half nh = y >> 7, shift group j = (y & 127) >> 5, run sub =
        // (y >> 4) & 1, lane-in-run l = y & 15 (the 8 share nh, j, sub)
        const unsigned int g = kv & 31u;
        const unsigned char* blk = row + (size_t)(kv >> 5) * 84u;
        const unsigned int nh = g >> 4, j = (g & 15u) >> 2, sub = (g >> 1) & 1u, l0 = (g & 1u) * 8u;
        const unsigned char sc = blk[nh * 8u + 2u * j + sub];
        const float d = engram_q2k_f16(blk + 80);
        const float dmin = engram_q2k_f16(blk + 82);
        const float dl = d * (float)(sc & 0x0F);
        const float ml = dmin * (float)(sc >> 4);
        const unsigned int* qw = (const unsigned int*)(blk + 16 + nh * 32u + sub * 16u + l0);
        const unsigned int q0 = qw[0], q1 = qw[1];
        const unsigned int a_raw[4] = {a_data.x, a_data.y, a_data.z, a_data.w};
        const unsigned int sh = 2u * j;

        #pragma unroll
        for (int i = 0; i < 4; i++) {
            const unsigned int qpair = (i < 2) ? (q0 >> (16 * i)) : (q1 >> (16 * (i - 2)));
            const int code_lo = (int)((qpair >> sh) & 3u);
            const int code_hi = (int)(((qpair >> 8) >> sh) & 3u);
            __nv_bfloat16 a_lo, a_hi;
            *(unsigned short*)&a_lo = (unsigned short)(a_raw[i] & 0xFFFF);
            *(unsigned short*)&a_hi = (unsigned short)(a_raw[i] >> 16);
            const __nv_bfloat16 b_lo = __float2bfloat16(dl * (float)code_lo - ml);
            const __nv_bfloat16 b_hi = __float2bfloat16(dl * (float)code_hi - ml);
            acc += __bfloat162float(a_lo) * __bfloat162float(b_lo);
            acc += __bfloat162float(a_hi) * __bfloat162float(b_hi);
        }
    }

    const unsigned int warp_lane = threadIdx.x % 32;
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        acc += __shfl_down_sync(0xFFFFFFFF, acc, offset);
    }
    __shared__ float smem[4 * 2];
    if (warp_lane == 0) smem[local_out * 2 + (lane / 32)] = acc;
    __syncthreads();
    if (lane == 0) {
        const float result = smem[local_out * 2] + smem[local_out * 2 + 1];
        C[n] = __float2bfloat16(result);
    }
}
