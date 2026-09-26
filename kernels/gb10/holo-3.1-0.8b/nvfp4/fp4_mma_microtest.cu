// SPDX-License-Identifier: MIT OR Apache-2.0

// 2026-09-25: Hand-written SM120 block-scaled FP4 MMA test kernels (module
// `fp4_mma_microtest`): `fp4_microtest_pack` quantises BF16 rows to NVFP4 and
// `fp4_microtest_mma` multiplies them with one
//   mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3
// per 64-wide K step. crates/model-arch/examples/fp4_mma_microproof.rs launches them and
// compares the result with the CUTLASS NVFP4 GEMM (pass threshold: cos >= 0.999).
// Owner: gb10 kernels (holo-3.1-0.8b).
// Invariants: none beyond the preconditions: M % 16 == 0, N % 8 == 0 and K % 64 == 0 for
// the MMA kernel; K % 16 == 0 for the pack.
//
// The fragment layout the MMA kernel assumes. Per lane t of the warp, q = t % 4 and
// r = t / 4; k0 is the current K base:
//   A (4 regs, 8 e2m1 each, low nibble = lower k):
//     a0: m=r   , k = k0 +      q*8 + (0..7)
//     a1: m=r+8 , k = k0 +      q*8 + (0..7)
//     a2: m=r   , k = k0 + 32 + q*8 + (0..7)
//     a3: m=r+8 , k = k0 + 32 + q*8 + (0..7)
//   B (2 regs, 8 e2m1 each):
//     b0: n=r   , k = k0 +      q*8 + (0..7)
//     b1: n=r   , k = k0 + 32 + q*8 + (0..7)
//   SFA (4 ue4m3 bytes, byte j = scale of k-group k0/16 + j): row m = (t%2)*8 + (t/4)
//   SFB (4 ue4m3 bytes, byte j = scale of k-group k0/16 + j): col n = t/4
//
// The pack uses the same per-16-group scale (max_abs / 6) and e2m1 thresholds as
// metrale_cutlass_pack_bf16_act_nvfp4 (crates/gpu-runtime/cuda/cutlass_nvfp4_gemm.cu), but
// writes natural layouts: packed[rows][K/2] (low nibble = even k) and scales[rows][K/16].









#include <cuda_bf16.h>
#include <cuda_fp8.h>

// 2026-09-25: The e2m1 thresholds of float_to_e2m1 in cutlass_nvfp4_gemm.cu.
__device__ __forceinline__ unsigned char mt_float_to_e2m1(float x) {
    unsigned char sign = (x < 0.0f) ? 8u : 0u;
    float ax = fabsf(x);
    unsigned char mag;
    if (ax <= 0.25f)      mag = 0;
    else if (ax <= 0.75f) mag = 1;
    else if (ax <= 1.25f) mag = 2;
    else if (ax <= 1.75f) mag = 3;
    else if (ax <= 2.5f)  mag = 4;
    else if (ax <= 3.5f)  mag = 5;
    else if (ax <= 5.0f)  mag = 6;
    else                  mag = 7;
    return sign | mag;
}

// 2026-09-25: Encode a non-negative scale as the byte of __nv_fp8_e4m3(scale); the sign
// bit is 0, and that byte is the scale factor the MMA's ue4m3 operand reads.






__device__ __forceinline__ unsigned char mt_float_to_ue4m3(float scale) {







    __nv_fp8_e4m3 v = __nv_fp8_e4m3(scale);
    unsigned char b = *reinterpret_cast<unsigned char*>(&v);
    return b;
}

__device__ __forceinline__ float mt_ue4m3_to_float(unsigned char byte) {
    __nv_fp8_e4m3 v;
    *reinterpret_cast<unsigned char*>(&v) = byte;
    return static_cast<float>(v);
}

// 2026-09-25: One thread per (row, 16-wide K group): reads bf16 src[rows][K], writes
// packed[rows][K/2] (e2m1 nibbles) and scales[rows][K/16] (one scale byte per group).

extern "C" __global__ void fp4_microtest_pack(
    const __nv_bfloat16* __restrict__ src,
    unsigned char* __restrict__ packed,
    unsigned char* __restrict__ scales,
    int rows,
    int k) {
    int row = blockIdx.x;
    int group = blockIdx.y * blockDim.x + threadIdx.x;
    int groups = k / 16;
    if (row >= rows || group >= groups) return;

    int base = group * 16;
    float max_abs = 0.0f;
#pragma unroll
    for (int i = 0; i < 16; ++i) {
        float v = __bfloat162float(src[(unsigned long long)row * k + base + i]);
        max_abs = fmaxf(max_abs, fabsf(v));
    }
    float scale = max_abs > 0.0f ? max_abs / 6.0f : 1.0f;
    unsigned char sf = mt_float_to_ue4m3(scale);
    scales[(unsigned long long)row * groups + group] = sf;
    float decoded = mt_ue4m3_to_float(sf);
    float inv = decoded > 0.0f ? 1.0f / decoded : 0.0f;

#pragma unroll
    for (int i = 0; i < 16; i += 2) {
        float v0 = __bfloat162float(src[(unsigned long long)row * k + base + i])     * inv;
        float v1 = __bfloat162float(src[(unsigned long long)row * k + base + i + 1]) * inv;
        packed[(unsigned long long)row * (k / 2) + base / 2 + i / 2] =
            (unsigned char)(mt_float_to_e2m1(v0) | (mt_float_to_e2m1(v1) << 4));
    }
}

// 2026-09-25: The 8 e2m1 values packed[row][kk..kk+7] (kk even) as a u32; nibble j is
// element kk+j.

__device__ __forceinline__ unsigned int gather_a8(
    const unsigned char* __restrict__ packed, int row, int kk, int k) {
    const unsigned char* p = packed + (unsigned long long)row * (k / 2) + kk / 2;
    unsigned int r = 0;
#pragma unroll
    for (int j = 0; j < 8; j += 2) {
        unsigned char byte = p[j / 2];
        unsigned int lo = byte & 0xF;
        unsigned int hi = (byte >> 4) & 0xF;
        r |= lo << (4 * j);
        r |= hi << (4 * (j + 1));
    }
    return r;
}

// 2026-09-25: The 4 scale bytes of `row`'s K groups g0..g0+3 as a u32, byte j = group g0+j.
__device__ __forceinline__ unsigned int gather_sf4(
    const unsigned char* __restrict__ scales, int row, int g0, int k) {
    int groups = k / 16;
    const unsigned char* p = scales + (unsigned long long)row * groups + g0;
    unsigned int r = 0;
#pragma unroll
    for (int j = 0; j < 4; ++j) r |= ((unsigned int)p[j]) << (8 * j);
    return r;
}

// 2026-09-25: One warp per 16x8 output tile: grid (N/8, M/16), block 32.
// A: packed_a[M][K/2], scales_a[M][K/16]. B: packed_b[N][K/2], scales_b[N][K/16].
// out: bf16[M][N], row-major, out[m,n] = sum_k a[m,k] * b[n,k].

extern "C" __global__ void fp4_microtest_mma(
    const unsigned char* __restrict__ packed_a,
    const unsigned char* __restrict__ scales_a,
    const unsigned char* __restrict__ packed_b,
    const unsigned char* __restrict__ scales_b,
    __nv_bfloat16* __restrict__ out,
    int m, int n, int k) {
    int tile_n = blockIdx.x * 8;
    int tile_m = blockIdx.y * 16;
    int t = threadIdx.x;
    int q = t & 3;
    int r = t >> 2;

    float acc[4] = {0.f, 0.f, 0.f, 0.f};


    int sfa_m = (t & 1) * 8 + (t >> 2);
    int sfb_n = t >> 2;

    for (int k0 = 0; k0 < k; k0 += 64) {

        unsigned int a0 = gather_a8(packed_a, tile_m + r,     k0 +      q * 8, k);
        unsigned int a1 = gather_a8(packed_a, tile_m + r + 8, k0 +      q * 8, k);
        unsigned int a2 = gather_a8(packed_a, tile_m + r,     k0 + 32 + q * 8, k);
        unsigned int a3 = gather_a8(packed_a, tile_m + r + 8, k0 + 32 + q * 8, k);

        unsigned int b0 = gather_a8(packed_b, tile_n + r,     k0 +      q * 8, k);
        unsigned int b1 = gather_a8(packed_b, tile_n + r,     k0 + 32 + q * 8, k);

        unsigned int sfa = gather_sf4(scales_a, tile_m + sfa_m, k0 / 16, k);
        unsigned int sfb = gather_sf4(scales_b, tile_n + sfb_n, k0 / 16, k);

#if (__CUDA_ARCH__ >= 1200)
        unsigned short bidA = 0, tidA = 0, bidB = 0, tidB = 0;
        asm volatile(
            "mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3 "
            "{%0,  %1,  %2,  %3},"
            "{%4,  %5,  %6,  %7},"
            "{%8,  %9},"
            "{%10, %11, %12, %13},"
            "{%14},"
            "{%15, %16},"
            "{%17},"
            "{%18, %19};\n"
            : "=f"(acc[0]), "=f"(acc[1]), "=f"(acc[2]), "=f"(acc[3])
            : "r"(a0), "r"(a1), "r"(a2), "r"(a3),
              "r"(b0), "r"(b1),
              "f"(acc[0]), "f"(acc[1]), "f"(acc[2]), "f"(acc[3]),
              "r"(sfa), "h"(bidA), "h"(tidA),
              "r"(sfb), "h"(bidB), "h"(tidB));
#endif
    }

    // 2026-09-25: Accumulator layout: acc[0], acc[1] are row t/4 and acc[2], acc[3] row
    // t/4 + 8, each at columns 2*(t%4) and 2*(t%4) + 1.
    int crow0 = tile_m + (t >> 2);
    int crow1 = crow0 + 8;
    int ccol0 = tile_n + 2 * (t & 3);
    int ccol1 = ccol0 + 1;
    if (crow0 < m) {
        if (ccol0 < n) out[(unsigned long long)crow0 * n + ccol0] = __float2bfloat16(acc[0]);
        if (ccol1 < n) out[(unsigned long long)crow0 * n + ccol1] = __float2bfloat16(acc[1]);
    }
    if (crow1 < m) {
        if (ccol0 < n) out[(unsigned long long)crow1 * n + ccol0] = __float2bfloat16(acc[2]);
        if (ccol1 < n) out[(unsigned long long)crow1 * n + ccol1] = __float2bfloat16(acc[3]);
    }
}
