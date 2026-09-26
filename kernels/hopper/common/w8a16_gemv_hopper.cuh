// SPDX-License-Identifier: MIT OR Apache-2.0

// 2026-09-25: Inner loop and reduction of the Hopper W8A16 M=1 decode GEMVs, shared
// by w8a16_gemv.cu and w8a16_gemv_fused.cu in this directory.
//
// Owner: hopper kernels.
// Invariants:
// - For one output row, lane l of the 64 walks chunks l, l + 64, l + 128, ... in
//   that order whatever the unroll depth, and adds a chunk's 16 products in K
//   order; unrolling issues loads early and changes no addition.
// - The reduction is a 5-step `__shfl_down_sync` tree per warp, then the two
//   warps' partials added through shared memory by the group's first thread,
//   then one `__float2bfloat16` round.
//
// The accumulation order is gb10/common/w8a16_gemv.cu's. Two things differ:
// - E4M3 is decoded with `cvt.rn.f16x2.e4m3x2` (sm_89 and later), two bytes per
//   instruction, instead of one shared-memory table read per byte. FP16 holds
//   every finite E4M3 value exactly and FP16 -> FP32 is exact, so the FP32 result
//   equals `E4M3_LUT`'s for the 254 finite codes. The NaN codes 0x7F and 0xFF
//   decode to NaN here and to +0 / -0 in `E4M3_LUT`. Below sm_89 `E4M3_LUT` is used.
// - `HOPPER_GEMV_UNROLL` chunk loads and their scales are issued before the first
//   is used. `ops::w8a16_gemv` launches ceil(N/4) blocks, so at N = 1024 there are
//   256 blocks for the 132 SMs of kernels/hopper/HARDWARE.toml; the unroll raises
//   the loads in flight per warp without a larger grid.
// Each `acc += a * w` stays a separate multiply and add: hopper targets build with
// gb10/common/KERNEL.toml's --fmad=false (crates/kernels/build.rs merges the
// inherited tree's common KERNEL.toml flags).
//
// crates/model-arch/examples/native_fp8_gemv_hopper_microtest.rs compares
// `w8a16_gemv` and `w8a16_gemv_dual` with a host model of that order and fails
// unless every output is bit-equal. Its weight bytes exclude 0x7F and 0xFF.
//
// Measured 2026-09-11 with nsys on 1x H100 SXM5, Qwen/Qwen3.8-27B-FP8, C=1 decode,
// with the gb10 kernels: `w8a16_gemv` took 7.39 ms (224 launches) and
// `w8a16_gemv_dual` 5.79 ms (64 launches) of an 18.5 ms step, moving FP8 weights
// at 1.84 TB/s against the 3.35 TB/s of kernels/hopper/HARDWARE.toml.
















































































#ifndef METRALE_HOPPER_W8A16_GEMV_CUH
#define METRALE_HOPPER_W8A16_GEMV_CUH

#include <cuda_bf16.h>

#include "e4m3_lut.cuh"

#define BLOCK_SIZE 256
#define N_PER_BLOCK 4
#define WARP_SIZE 32
#define FP8_BLOCK 128
/// 2026-09-25: K values one lane consumes per chunk: one `uint4` of FP8 bytes.
#define K_PER_CHUNK 16
/// 2026-09-25: Chunks per 128-wide scale block; chunk k16 uses `scale_row[k16 / CHUNKS_PER_SCALE]`.
#define CHUNKS_PER_SCALE (FP8_BLOCK / K_PER_CHUNK)
/// 2026-09-25: Chunk loads `hopper_gemv_row` issues before using the first, for
/// `w8a16_gemv` and `w8a16_gemv_dual`; `w8a16_gemv_silu_input` uses 2 (see
/// w8a16_gemv_fused.cu).

#define HOPPER_GEMV_UNROLL 4

/// 2026-09-25: Two E4M3 bytes (`.x` from the low byte) to FP32; from sm_89 on, equal to
/// `E4M3_LUT` except for the NaN codes 0x7F and 0xFF.
__device__ __forceinline__ float2 hopper_e4m3x2_to_f32x2(unsigned short raw) {
#if !defined(__CUDA_ARCH__) || (__CUDA_ARCH__ >= 890)
    unsigned int h2;
    asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(h2) : "h"(raw));
    return __half22float2(*reinterpret_cast<const __half2*>(&h2));
#else
    return make_float2(E4M3_LUT[raw & 0xFFu], E4M3_LUT[raw >> 8]);
#endif
}


__device__ __forceinline__ float hopper_bf16_bits_to_f32(unsigned short bits) {
    __nv_bfloat16 h;
    *(unsigned short*)&h = bits;
    return __bfloat162float(h);
}


struct HopperActChunk {
    float v[K_PER_CHUNK];
};

/// 2026-09-25: The 8 BF16 values of a `uint4` to `out[0..8]` in memory order: the low
/// half of each 32-bit word before its high half.

__device__ __forceinline__ void hopper_unpack_bf16x8(uint4 d, float* out) {
    const unsigned int raw[4] = {d.x, d.y, d.z, d.w};
#pragma unroll
    for (int i = 0; i < 4; i++) {
        out[i * 2 + 0] = hopper_bf16_bits_to_f32((unsigned short)(raw[i] & 0xFFFFu));
        out[i * 2 + 1] = hopper_bf16_bits_to_f32((unsigned short)(raw[i] >> 16));
    }
}

/// 2026-09-25: Activation source: a BF16 row `A[1, K]`.
struct HopperActRow {
    const __nv_bfloat16* __restrict__ a;

    __device__ __forceinline__ void chunk(unsigned int k16, HopperActChunk& out) const {
        hopper_unpack_bf16x8(((const uint4*)a)[k16 * 2], &out.v[0]);
        hopper_unpack_bf16x8(((const uint4*)a)[k16 * 2 + 1], &out.v[8]);
    }
};

/// 2026-09-25: Adds one chunk's 16 products to `acc` in K order, each `a[i] * w_i` with
/// `w_i = decode(byte_i) * scale`.
template <typename Act>
__device__ __forceinline__ float hopper_gemv_chunk(
    float acc,
    uint4 b,
    float scale,
    const Act& act,
    unsigned int k16
) {
    HopperActChunk a;
    act.chunk(k16, a);
    const unsigned int b_raw[4] = {b.x, b.y, b.z, b.w};
#pragma unroll
    for (int i = 0; i < 4; i++) {
        float2 lo = hopper_e4m3x2_to_f32x2((unsigned short)(b_raw[i] & 0xFFFFu));
        float2 hi = hopper_e4m3x2_to_f32x2((unsigned short)(b_raw[i] >> 16));
        float w0 = lo.x * scale;
        float w1 = lo.y * scale;
        float w2 = hi.x * scale;
        float w3 = hi.y * scale;
        acc += a.v[i * 4 + 0] * w0;
        acc += a.v[i * 4 + 1] * w1;
        acc += a.v[i * 4 + 2] * w2;
        acc += a.v[i * 4 + 3] * w3;
    }
    return acc;
}

/// 2026-09-25: One output row's partial dot product for `lane` (0..63).
///
/// `b_row` is `B + n*K` and `scale_row` is `block_scale + n_block*k_blocks`. The
/// lane walks chunks `lane, lane+64, lane+128, ...`: `UNROLL` chunk loads and their
/// scales are issued before the first is used and then added in walk order, and a
/// tail loop takes the remaining chunks one at a time.
template <int UNROLL, typename Act>
__device__ __forceinline__ float hopper_gemv_row(
    const unsigned char* __restrict__ b_row,
    const float* __restrict__ scale_row,
    const Act& act,
    unsigned int k16_count,
    unsigned int lane
) {
    const unsigned int stride = BLOCK_SIZE / N_PER_BLOCK;
    const uint4* b4 = (const uint4*)b_row;
    float acc = 0.0f;
    unsigned int k16 = lane;


    for (; k16 + (UNROLL - 1) * stride < k16_count; k16 += UNROLL * stride) {
        uint4 b[UNROLL];
        float s[UNROLL];
#pragma unroll
        for (int u = 0; u < UNROLL; u++) {
            const unsigned int c = k16 + u * stride;
            b[u] = b4[c];
            s[u] = scale_row[c / CHUNKS_PER_SCALE];
        }
#pragma unroll
        for (int u = 0; u < UNROLL; u++) {
            acc = hopper_gemv_chunk(acc, b[u], s[u], act, k16 + u * stride);
        }
    }

    for (; k16 < k16_count; k16 += stride) {
        acc = hopper_gemv_chunk(acc, b4[k16], scale_row[k16 / CHUNKS_PER_SCALE], act, k16);
    }
    return acc;
}

/// 2026-09-25: A 5-step `__shfl_down_sync` tree per warp, then the group's first thread
/// adds the two warps' partials from `smem` and writes `c[n]` as BF16. Contains a
/// `__syncthreads()`.
__device__ __forceinline__ void hopper_gemv_reduce_store(
    float acc,
    float* smem,
    unsigned int local_out,
    unsigned int lane,
    __nv_bfloat16* __restrict__ c,
    unsigned int n
) {
#pragma unroll
    for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1) {
        acc += __shfl_down_sync(0xFFFFFFFF, acc, offset);
    }
    const unsigned int warp_in_out = lane / WARP_SIZE;
    if (lane % WARP_SIZE == 0) {
        smem[local_out * 2 + warp_in_out] = acc;
    }
    __syncthreads();
    if (lane == 0) {
        float result = smem[local_out * 2] + smem[local_out * 2 + 1];
        c[n] = __float2bfloat16(result);
    }
}

#endif
