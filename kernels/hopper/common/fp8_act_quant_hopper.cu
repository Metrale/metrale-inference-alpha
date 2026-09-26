// SPDX-License-Identifier: MIT OR Apache-2.0

// 2026-09-25: Hopper twin of `per_token_group_quant_fp8`: per-token, per-128-K-group
// FP8 E4M3 activation quantization for the W8A8 projections.
//
// Owner: hopper kernels.
// Invariants:
// - While M * K < 2^32 (the shared kernel indexes A in 32 bits), the FP8 bytes
//   and FP32 scales equal those of kernels/gb10/common/per_token_group_quant_fp8.cu: the same
//   `amax / 448.0f`, the same `1e-12f` floor, the same per-element division
//   by the scale (not a reciprocal multiply), the same clamp to +-448 and
//   the same `__nv_cvt_float_to_fp8(..., __NV_SATFINITE, __NV_E4M3)`. Only
//   the amax reduction tree differs, and it cannot change the result:
//   `fmaxf` is exact, drops a NaN operand, and both kernels seed the
//   combine with `0.0f`. `native_fp8_act_quant_hopper_microtest` compares
//   both outputs byte for byte, with a KNOWN_BAD control that must differ.
// - Launched with grid (M, Y, 1) and block (128, 1, 1). A CTA owns
//   ceil((K/128) / gridDim.y) groups, so any Y in 1..=K/128 covers the K/128
//   groups exactly once; `fp8_quant_grid` uses ceil(K/128 / 8). M is on
//   grid X, so it may exceed 65535.
//
// Why a twin: the shared kernel gives each 128-element group its own
// 128-thread CTA, one bf16 per thread, so a CTA loads 256 B. Here sixteen
// threads cover a group, eight groups share a CTA and each thread loads one
// `uint4` (8 bf16), so a CTA loads 2 KB. The values stay in registers, so A
// is read once (the shared kernel reads it twice, for the amax and to
// quantize), and the group max reduces through a 16-lane `__shfl_xor_sync`
// butterfly with no shared memory and no `__syncthreads`.
// Measurements: FP8-ACT-QUANT-ATTRIBUTION.md.















#include <cstdint>

#include <cuda_bf16.h>
#include <cuda_fp8.h>

#define FP8_GROUP_K 128
#define FP8_E4M3_MAX 448.0f

// 2026-09-25: 128 elements / 8 per thread = 16 threads per group; 128 threads / 16 = 8
// groups per CTA. Mirrored by `ops::FP8_QUANT_HOPPER_GROUPS_PER_CTA`.
#define HQ_LANES_PER_GROUP 16
#define HQ_GROUPS_PER_CTA 8
#define HQ_ELEMS_PER_THREAD 8

// 2026-09-25: One `uint4` (8 bf16) when `p` is 16-byte aligned, else eight scalar
// loads. Both arms load the same values in the same order.






__device__ __forceinline__ void hq_load8(const __nv_bfloat16* __restrict__ p, float* v) {
    if ((reinterpret_cast<uintptr_t>(p) & 15u) == 0) {
        const uint4 w = *reinterpret_cast<const uint4*>(p);
        const __nv_bfloat16* b = reinterpret_cast<const __nv_bfloat16*>(&w);
        #pragma unroll
        for (int i = 0; i < HQ_ELEMS_PER_THREAD; i++) v[i] = __bfloat162float(b[i]);
    } else {
        #pragma unroll
        for (int i = 0; i < HQ_ELEMS_PER_THREAD; i++) v[i] = __bfloat162float(p[i]);
    }
}

// 2026-09-25: The store side: one 8-byte `uint2` when `p` is 8-byte aligned.
__device__ __forceinline__ void hq_store8(unsigned char* __restrict__ p, const unsigned char* b) {
    if ((reinterpret_cast<uintptr_t>(p) & 7u) == 0) {
        uint2 w;
        w.x = (unsigned int)b[0] | ((unsigned int)b[1] << 8) | ((unsigned int)b[2] << 16)
              | ((unsigned int)b[3] << 24);
        w.y = (unsigned int)b[4] | ((unsigned int)b[5] << 8) | ((unsigned int)b[6] << 16)
              | ((unsigned int)b[7] << 24);
        *reinterpret_cast<uint2*>(p) = w;
    } else {
        #pragma unroll
        for (int i = 0; i < HQ_ELEMS_PER_THREAD; i++) p[i] = b[i];
    }
}

extern "C" __global__ void per_token_group_quant_fp8_hopper(
    const __nv_bfloat16* __restrict__ A,   // 2026-09-25: [M, K] BF16 activations
    unsigned char* __restrict__ A_fp8,     // 2026-09-25: [M, K] FP8 E4M3
    float* __restrict__ a_scale,           // 2026-09-25: [M, K/128] FP32 scale, row-major
    unsigned int M,
    unsigned int K
) {
    const unsigned int m = blockIdx.x;
    if (m >= M) return;

    // 2026-09-25: `K / 128`, the same group count as the shared kernel's grid.y. A K that
    // is not a multiple of 128 drops the same partial tail there as here.
    const unsigned int L = K / FP8_GROUP_K;
    if (L == 0) return;

    const unsigned int gpc = (L + gridDim.y - 1u) / gridDim.y;
    const unsigned int g0 = blockIdx.y * gpc;
    if (g0 >= L) return;
    const unsigned int g_end = (g0 + gpc < L) ? (g0 + gpc) : L;

    const unsigned int tid = threadIdx.x;
    const unsigned int sub = tid / HQ_LANES_PER_GROUP;
    const unsigned int lane = tid % HQ_LANES_PER_GROUP;

    const size_t row = (size_t)m * (size_t)K;
    // 2026-09-25: Uniform across the CTA, so every lane runs the same trip count and the
    // butterfly below is never entered by a divergent subset of the warp.
    const unsigned int iters = (gpc + HQ_GROUPS_PER_CTA - 1u) / HQ_GROUPS_PER_CTA;

    for (unsigned int it = 0; it < iters; ++it) {
        const unsigned int g = g0 + it * HQ_GROUPS_PER_CTA + sub;
        const bool live = (g < g_end);
        const size_t base =
            row + (size_t)g * FP8_GROUP_K + (size_t)lane * HQ_ELEMS_PER_THREAD;

        // 2026-09-25: 1. Load 8 elements per thread and take their abs-max.
        //
        // Seeded with 0.0f, as the shared kernel seeds its cross-warp combine.
        // Every input is an absolute value, so the seed is a no-op on the real
        // domain; on NaN it is what makes the two trees agree.
        float v[HQ_ELEMS_PER_THREAD];
        float amax = 0.0f;
        if (live) {
            hq_load8(A + base, v);
            #pragma unroll
            for (int i = 0; i < HQ_ELEMS_PER_THREAD; i++) amax = fmaxf(amax, fabsf(v[i]));
        }

        // 2026-09-25: 2. Reduce the max across the group's 16 lanes. XOR offsets 8, 4, 2, 1
        //    never cross a 16-lane boundary, so two groups share a warp without
        //    mixing and every lane ends holding its own group's amax.
        #pragma unroll
        for (int off = HQ_LANES_PER_GROUP / 2; off > 0; off >>= 1) {
            amax = fmaxf(amax, __shfl_xor_sync(0xFFFFFFFFu, amax, off));
        }

        // 2026-09-25: 3. The scale, the shared kernel's expression.
        float scale = amax / FP8_E4M3_MAX;
        if (scale < 1e-12f) scale = 1e-12f;
        if (live && lane == 0) a_scale[(size_t)m * (size_t)L + (size_t)g] = scale;
        if (!live) continue;

        // 2026-09-25: 4. Quantize from registers; A is not re-read.
        unsigned char out[HQ_ELEMS_PER_THREAD];
        #pragma unroll
        for (int i = 0; i < HQ_ELEMS_PER_THREAD; i++) {
            float q = v[i] / scale;
            q = fmaxf(fminf(q, FP8_E4M3_MAX), -FP8_E4M3_MAX);
            out[i] = (unsigned char)__nv_cvt_float_to_fp8(q, __NV_SATFINITE, __NV_E4M3);
        }
        hq_store8(A_fp8 + base, out);
    }
}
