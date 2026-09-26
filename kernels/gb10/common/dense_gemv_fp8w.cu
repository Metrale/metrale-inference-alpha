// SPDX-License-Identifier: MIT OR Apache-2.0

// 2026-09-25: FP8-weight GEMV for one activation row, and the BF16 -> FP8 row quantizer
// that produces its weights: C[n] = row_scale[n] * sum_k A[k] * fp8(B[n, k]).
//
// Owner: gb10 kernels.
// Invariants:
// - A is [1, K] BF16, B is [N, K] FP8 E4M3 bytes row-major, row_scale is [N] FP32,
//   C is [1, N] BF16.
// - dense_gemv_fp8w launch: grid (ceil(N / 4), 1, 1), block (256, 1, 1); 64 threads
//   (2 warps) per output. quantize_bf16_to_fp8 launch: grid (N, 1, 1), block (256, 1, 1).
// - Assumes K % 16 == 0: dense_gemv_fp8w has no scalar tail, so a remainder would be
//   dropped, and rows start 16-byte aligned (B) and 32-byte aligned (A) only then.
// - One weight byte per element: a uint4 load carries 16 FP8 weights, matched by two
//   uint4 loads of 16 BF16 activations.





#include <cuda_bf16.h>
#include <cuda_fp8.h>

// 2026-09-25: Software E4M3 decode and encode, used on SCALE and HIP builds in place of the
// __nv_fp8_e4m3 conversions, so quantize_bf16_to_fp8 and dense_gemv_fp8w agree there.
// Decode: bias 7, subnormal m * 2^-9, and the NaN pattern (e = 15, m = 7) reads as 0.


#if defined(__SCALE__) || defined(__HIP_PLATFORM_AMD__)
__device__ __forceinline__ float scl_fp8(unsigned char b) {
    unsigned int s = (b >> 7) & 1u, e = (b >> 3) & 0xFu, m = b & 0x7u; float v;
    if (e == 0u)               v = (float)m * 0.001953125f;
    else if (e == 15u && m == 7u) v = 0.0f;
    else                       v = __uint_as_float(((e + 120u) << 23) | (m << 20));
    return s ? -v : v;
}
// 2026-09-25: Encode that pairs with scl_fp8. NaN encodes as 0x7F; |v| >= 512 saturates to
// 448 (e = 15, m = 6); the mantissa rounds half up. The caller clamps to +-448 first.


__device__ __forceinline__ unsigned char scl_enc_fp8(float v) {
    if (v != v) return 0x7F;
    unsigned int bb = __float_as_uint(v); unsigned int sign = (bb >> 31) & 1u;
    int e = (int)((bb >> 23) & 0xFF) - 127; unsigned int man = bb & 0x7FFFFFu;
    int ee = e + 7; unsigned int em;
    if (ee < 1) { ee = 0; em = 0; if (e >= -10) { float a = v < 0 ? -v : v; em = (unsigned int)(a / 0.001953125f + 0.5f); if (em > 7u) em = 7u; } }
    else if (ee > 15) { ee = 15; em = 6; }
    else { em = (man + (1u << 19)) >> 20; if (em > 7u) { em = 0; ee++; if (ee > 15) { ee = 15; em = 6; } } }
    return (unsigned char)((sign << 7) | ((unsigned)ee << 3) | em);
}
#endif

#define BLOCK_SIZE 256
#define N_PER_BLOCK 4
#define WARP_SIZE 32
#define VEC_SIZE 16

// 2026-09-25: Quantizes one [K] BF16 row per block to FP8 E4M3 with a per-row FP32 scale
// max|x| / 448 (1.0 for an all-zero row). Callers build FP8 copies while constructing
// heads and tables (for example factory/lm_head_setup.rs, mtp_head/new.rs).
// blockDim.x must be a multiple of 32 and at most 256 (smem_q holds 8 warp maxima).



extern "C" __global__ void quantize_bf16_to_fp8(
    const __nv_bfloat16* __restrict__ input,
    unsigned char* __restrict__ output,
    float* __restrict__ row_scales,
    unsigned int N,
    unsigned int K
) {
    unsigned int row = blockIdx.x;
    if (row >= N) return;

    const __nv_bfloat16* row_in = input + (unsigned long long)row * K;
    unsigned char* row_out = output + (unsigned long long)row * K;


    float local_max = 0.0f;
    for (unsigned int k = threadIdx.x; k < K; k += blockDim.x) {
        float absval = fabsf(__bfloat162float(row_in[k]));
        if (absval > local_max) local_max = absval;
    }


    #pragma unroll
    for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1) {
        float other = __shfl_down_sync(0xFFFFFFFF, local_max, offset);
        if (other > local_max) local_max = other;
    }


    __shared__ float smem_q[8];
    unsigned int warp_id = threadIdx.x / WARP_SIZE;
    unsigned int warp_lane = threadIdx.x % WARP_SIZE;
    if (warp_lane == 0) smem_q[warp_id] = local_max;
    __syncthreads();

    if (threadIdx.x == 0) {
        float max_val = 0.0f;
        for (int w = 0; w < (int)(blockDim.x / WARP_SIZE); w++) {
            if (smem_q[w] > max_val) max_val = smem_q[w];
        }

        float scale = (max_val > 0.0f) ? (max_val / 448.0f) : 1.0f;
        row_scales[row] = scale;
        smem_q[0] = 1.0f / scale;
    }
    __syncthreads();
    float inv_scale = smem_q[0];


    for (unsigned int k = threadIdx.x; k < K; k += blockDim.x) {
        float val = __bfloat162float(row_in[k]) * inv_scale;

        val = fminf(fmaxf(val, -448.0f), 448.0f);
#if defined(__SCALE__) || defined(__HIP_PLATFORM_AMD__)
        row_out[k] = scl_enc_fp8(val);
#else
        __nv_fp8_e4m3 fp8 = (__nv_fp8_e4m3)val;
        row_out[k] = *(unsigned char*)&fp8;
#endif
    }
}






extern "C" __global__ void dense_gemv_fp8w(
    const __nv_bfloat16* __restrict__ A,
    const unsigned char* __restrict__ B,
    const float* __restrict__ row_scale,
    __nv_bfloat16* __restrict__ C,
    unsigned int N,
    unsigned int K
) {
    const unsigned int threads_per_out = BLOCK_SIZE / N_PER_BLOCK;
    const unsigned int local_out = threadIdx.x / threads_per_out;
    const unsigned int lane = threadIdx.x % threads_per_out;

    const unsigned int n = blockIdx.x * N_PER_BLOCK + local_out;
    if (n >= N) return;


    const float scale = row_scale[n];

    float acc = 0.0f;


    const unsigned int K_VEC = K / VEC_SIZE;
    const uint4* B_vec = (const uint4*)(B + (unsigned long long)n * K);

    for (unsigned int kv = lane; kv < K_VEC; kv += threads_per_out) {

        uint4 b_data = B_vec[kv];

        // 2026-09-25: The 16 activations of vector kv are BF16 uint4 indices 2kv and 2kv+1.


        uint4 a_data0 = ((const uint4*)A)[kv * 2];
        uint4 a_data1 = ((const uint4*)A)[kv * 2 + 1];


        const unsigned int b_raw[4] = {b_data.x, b_data.y, b_data.z, b_data.w};


        const unsigned int a_raw0[4] = {a_data0.x, a_data0.y, a_data0.z, a_data0.w};

        #pragma unroll
        for (int i = 0; i < 2; i++) {

            unsigned int w32 = b_raw[i];
            unsigned int a32_lo = a_raw0[i * 2];
            unsigned int a32_hi = a_raw0[i * 2 + 1];


            __nv_fp8_e4m3 fp8_0, fp8_1, fp8_2, fp8_3;
            *(unsigned char*)&fp8_0 = (unsigned char)(w32 & 0xFF);
            *(unsigned char*)&fp8_1 = (unsigned char)((w32 >> 8) & 0xFF);
            *(unsigned char*)&fp8_2 = (unsigned char)((w32 >> 16) & 0xFF);
            *(unsigned char*)&fp8_3 = (unsigned char)((w32 >> 24) & 0xFF);

            __nv_bfloat16 a_lo, a_hi;
            *(unsigned short*)&a_lo = (unsigned short)(a32_lo & 0xFFFF);
            *(unsigned short*)&a_hi = (unsigned short)(a32_lo >> 16);
#if defined(__SCALE__) || defined(__HIP_PLATFORM_AMD__)
            acc += __bfloat162float(a_lo) * scl_fp8((unsigned char)(w32 & 0xFF));
            acc += __bfloat162float(a_hi) * scl_fp8((unsigned char)((w32 >> 8) & 0xFF));
#else
            acc += __bfloat162float(a_lo) * (float)fp8_0;
            acc += __bfloat162float(a_hi) * (float)fp8_1;
#endif

            *(unsigned short*)&a_lo = (unsigned short)(a32_hi & 0xFFFF);
            *(unsigned short*)&a_hi = (unsigned short)(a32_hi >> 16);
#if defined(__SCALE__) || defined(__HIP_PLATFORM_AMD__)
            acc += __bfloat162float(a_lo) * scl_fp8((unsigned char)((w32 >> 16) & 0xFF));
            acc += __bfloat162float(a_hi) * scl_fp8((unsigned char)((w32 >> 24) & 0xFF));
#else
            acc += __bfloat162float(a_lo) * (float)fp8_2;
            acc += __bfloat162float(a_hi) * (float)fp8_3;
#endif
        }


        const unsigned int a_raw1[4] = {a_data1.x, a_data1.y, a_data1.z, a_data1.w};

        #pragma unroll
        for (int i = 0; i < 2; i++) {
            unsigned int w32 = b_raw[i + 2];
            unsigned int a32_lo = a_raw1[i * 2];
            unsigned int a32_hi = a_raw1[i * 2 + 1];

            __nv_fp8_e4m3 fp8_0, fp8_1, fp8_2, fp8_3;
            *(unsigned char*)&fp8_0 = (unsigned char)(w32 & 0xFF);
            *(unsigned char*)&fp8_1 = (unsigned char)((w32 >> 8) & 0xFF);
            *(unsigned char*)&fp8_2 = (unsigned char)((w32 >> 16) & 0xFF);
            *(unsigned char*)&fp8_3 = (unsigned char)((w32 >> 24) & 0xFF);

            __nv_bfloat16 a_lo, a_hi;
            *(unsigned short*)&a_lo = (unsigned short)(a32_lo & 0xFFFF);
            *(unsigned short*)&a_hi = (unsigned short)(a32_lo >> 16);
#if defined(__SCALE__) || defined(__HIP_PLATFORM_AMD__)
            acc += __bfloat162float(a_lo) * scl_fp8((unsigned char)(w32 & 0xFF));
            acc += __bfloat162float(a_hi) * scl_fp8((unsigned char)((w32 >> 8) & 0xFF));
#else
            acc += __bfloat162float(a_lo) * (float)fp8_0;
            acc += __bfloat162float(a_hi) * (float)fp8_1;
#endif

            *(unsigned short*)&a_lo = (unsigned short)(a32_hi & 0xFFFF);
            *(unsigned short*)&a_hi = (unsigned short)(a32_hi >> 16);
#if defined(__SCALE__) || defined(__HIP_PLATFORM_AMD__)
            acc += __bfloat162float(a_lo) * scl_fp8((unsigned char)((w32 >> 16) & 0xFF));
            acc += __bfloat162float(a_hi) * scl_fp8((unsigned char)((w32 >> 24) & 0xFF));
#else
            acc += __bfloat162float(a_lo) * (float)fp8_2;
            acc += __bfloat162float(a_hi) * (float)fp8_3;
#endif
        }
    }

    // 2026-09-25: The per-row scale multiplies the finished partial sum once, not each weight.
    acc *= scale;


    const unsigned int warp_lane = threadIdx.x % WARP_SIZE;

    #pragma unroll
    for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1) {
        acc += __shfl_down_sync(0xFFFFFFFF, acc, offset);
    }

    // 2026-09-25: Two warps per output: add the warp partials through shared memory.
    __shared__ float smem[N_PER_BLOCK * 2];

    if (warp_lane == 0) {
        unsigned int smem_idx = local_out * 2 + (lane / WARP_SIZE);
        smem[smem_idx] = acc;
    }
    __syncthreads();


    if (lane == 0) {
        float result = smem[local_out * 2] + smem[local_out * 2 + 1];
        C[n] = __float2bfloat16(result);
    }
}
