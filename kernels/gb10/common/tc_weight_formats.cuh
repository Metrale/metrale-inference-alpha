// SPDX-License-Identifier: MIT OR Apache-2.0

// 2026-10-02: The weight-format policies of the tensor-core W*A16 kernels: how a 16-byte lane
// load of weights becomes m16n8k16 BF16 A fragments, and how block and tensor scales apply. One
// policy serves every kernel family that takes it (the grouped experts, moe_grouped_tc.cuh; the
// dense row tiles, tc_rows.cuh).
//
// Owner: gb10 kernels.
// Invariants:
// - A policy P supplies CHUNK_K (the K a lane-quad's 16-byte weight load spans; MMA j < CHUNK_K
//   / 16 takes, per lane, fragment slots 2t, 2t + 1 and 2t + 8, 2t + 9 from activation words
//   2j and 2j + 1, where activation word i holds the lane's K + 2i, 2i + 1), FOLDS, ACT_LIFT,
//   Mat, Tile, Sc, tile(), load(), scale(), frag(), fold_at(), fold_scale() and out().
// - FOLDS = true: each chunk sums into tmp, folded as tmp * fold_scale into acc where fold_at
//   holds; FOLDS = false: the MMAs accumulate into acc. out() scales the final sum.
// - ACT_LIFT multiplies BF16 activations before the MMAs (an exact power of two).

#pragma once

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>

// 2026-09-28: E4M3 bytes 0,1 (sel 0x1404) or 2,3 (sel 0x3424) of w as a BF16 pair, each
// E4M3 * 2^-120.
__device__ __forceinline__ unsigned int tc_e4m3_pair_bf16(unsigned int w, unsigned int sel) {
    const unsigned int x = __byte_perm(w, 0u, sel);
    return (x & 0x80008000u) | ((x >> 4) & 0x07F007F0u);
}

// 2026-10-02: The FP8 weight-format policy of gtc_warp: row-major [N, K] E4M3 with FP32 block
// scales [N / 128, K / 128]. A 16-byte lane load is 16 K of a 64-K chunk; word j of it feeds MMA j
// (bytes 0, 1 as slots 2t, 2t + 1, bytes 2, 3 as 2t + 8, 2t + 9). Weights enter as E4M3 * 2^-120
// and activations times 2^60, so products stay FP32 normals; each 128-K block's sum is scaled
// once by its block scale times 2^60.
struct Fp8Block128 {
    static constexpr int CHUNK_K = 64;
    static constexpr bool FOLDS = true;
    static constexpr float ACT_LIFT = 1152921504606846976.0f;
    struct Mat { const unsigned char* w; const float* s; };
    struct Tile { const unsigned char* wr[2]; const float* sr; };
    struct Sc {};
    static __device__ __forceinline__ Tile tile(const Mat& M, unsigned int col, unsigned int g, unsigned int t, unsigned int K) {
        Tile T;
        T.wr[0] = M.w + (unsigned long long)(col + g) * K + t * 16;
        T.wr[1] = M.w + (unsigned long long)(col + g + 8) * K + t * 16;
        T.sr = M.s + (col / 128) * (K / 128);
        return T;
    }
    static __device__ __forceinline__ uint4 load(const Tile& T, int h, unsigned int chunk) {
        return *(const uint4*)(T.wr[h] + chunk * 64);
    }
    static __device__ __forceinline__ Sc scale(const Tile&, int, unsigned int) { return Sc{}; }
    static __device__ __forceinline__ void frag(const uint4& lo, const uint4& hi, Sc, Sc, int j, unsigned int* a) {
        const unsigned int wg = (j == 0) ? lo.x : (j == 1) ? lo.y : (j == 2) ? lo.z : lo.w;
        const unsigned int wh = (j == 0) ? hi.x : (j == 1) ? hi.y : (j == 2) ? hi.z : hi.w;
        a[0] = tc_e4m3_pair_bf16(wg, 0x1404u);
        a[1] = tc_e4m3_pair_bf16(wh, 0x1404u);
        a[2] = tc_e4m3_pair_bf16(wg, 0x3424u);
        a[3] = tc_e4m3_pair_bf16(wh, 0x3424u);
    }
    // 2026-09-28: Chunks 2kb and 2kb + 1 make 128-K block kb: scale it once.
    static __device__ __forceinline__ bool fold_at(unsigned int chunk) { return chunk & 1; }
    static __device__ __forceinline__ float fold_scale(const Tile& T, unsigned int chunk) {
        return T.sr[chunk >> 1] * ACT_LIFT;
    }
    static __device__ __forceinline__ float out(const Tile&, float x) { return x; }
};

// 2026-10-02: Four E2M1 values (element e in bits 4e .. 4e + 3 of h; higher bits ignored) as two
// BF16 pairs: d[0] = elements 0 (low half), 1; d[1] = elements 2, 3. The magnitude picks the low
// and high BF16 bytes from two 8-entry byte tables; the sign bit moves to bit 15 or 31.
__device__ __forceinline__ void ntc_e2m1x4_bf16(unsigned int h, unsigned int* d) {
    // Low bytes of |v| for magnitudes 0..7 (0, 0.5, 1, 1.5 | 2, 3, 4, 6) and high bytes.
    const unsigned int L0 = 0xC0800000u, L1 = 0xC0804000u;
    const unsigned int H0 = 0x3F3F3F00u, H1 = 0x40404040u;
    const unsigned int s = h & 0x7777u;
    const unsigned int lo = __byte_perm(L0, L1, s), hi = __byte_perm(H0, H1, s);
    d[0] = __byte_perm(lo, hi, 0x5140u) | ((h << 12) & 0x8000u) | ((h << 24) & 0x80000000u);
    d[1] = __byte_perm(lo, hi, 0x7362u) | ((h << 4) & 0x8000u) | ((h << 16) & 0x80000000u);
}

// 2026-10-02: E4M3 byte b as a BF16 pair (b, b), exact.
__device__ __forceinline__ unsigned int ntc_e4m3_bf16x2(unsigned int b) {
    const __half_raw h = __nv_cvt_fp8_to_halfraw((__nv_fp8_storage_t)b, __NV_E4M3);
    const __nv_bfloat16 v = __float2bfloat16_rn(__half2float(__half(h)));
    const unsigned short u = *(const unsigned short*)&v;
    return (unsigned int)u | ((unsigned int)u << 16);
}

__device__ __forceinline__ unsigned int ntc_hmul2(unsigned int a, unsigned int s) {
    __nv_bfloat162 r = __hmul2(*(const __nv_bfloat162*)&a, *(const __nv_bfloat162*)&s);
    return *(unsigned int*)&r;
}

// 2026-10-02: The NVFP4 weight-format policy of gtc_warp. A 16-byte lane load is 32 K of a 128-K
// chunk; word j / 2 of it feeds MMA j (its low four elements for even j, its high four for odd
// j). Words 0, 1 lie in E4M3 block 2t of the chunk and words 2, 3 in block 2t + 1; one 2-byte
// load per row and chunk fetches both. A weight enters as BF16(E2M1 * E4M3) (exact) and s2
// multiplies the FP32 sum once.
struct Nvfp4G16 {
    static constexpr int CHUNK_K = 128;
    static constexpr bool FOLDS = false;
    static constexpr float ACT_LIFT = 1.0f;
    struct Mat { const unsigned char* w; const unsigned char* s; float s2; };
    struct Tile { const unsigned char* wr[2]; const unsigned char* sr[2]; float s2; };
    using Sc = unsigned int;
    static __device__ __forceinline__ Tile tile(const Mat& M, unsigned int col, unsigned int g, unsigned int t, unsigned int K) {
        Tile T;
        #pragma unroll
        for (int h = 0; h < 2; h++) {
            T.wr[h] = M.w + (unsigned long long)(col + g + 8 * h) * (K / 2) + t * 16;
            T.sr[h] = M.s + (unsigned long long)(col + g + 8 * h) * (K / 16) + 2 * t;
        }
        T.s2 = M.s2;
        return T;
    }
    static __device__ __forceinline__ uint4 load(const Tile& T, int h, unsigned int chunk) {
        return *(const uint4*)(T.wr[h] + chunk * 64);
    }
    static __device__ __forceinline__ Sc scale(const Tile& T, int h, unsigned int chunk) {
        return *(const unsigned short*)(T.sr[h] + chunk * 8);
    }
    static __device__ __forceinline__ void frag(const uint4& lo, const uint4& hi, Sc sg, Sc sh, int j, unsigned int* a) {
        const int wi = j >> 1, sh_bits = (j & 1) ? 16 : 0, sb = (j >= 4) ? 8 : 0;
        const unsigned int wg = ((wi == 0) ? lo.x : (wi == 1) ? lo.y : (wi == 2) ? lo.z : lo.w) >> sh_bits;
        const unsigned int wh = ((wi == 0) ? hi.x : (wi == 1) ? hi.y : (wi == 2) ? hi.z : hi.w) >> sh_bits;
        const unsigned int cg = ntc_e4m3_bf16x2((sg >> sb) & 0xFFu), ch = ntc_e4m3_bf16x2((sh >> sb) & 0xFFu);
        unsigned int dg[2], dh[2];
        ntc_e2m1x4_bf16(wg, dg);
        ntc_e2m1x4_bf16(wh, dh);
        a[0] = ntc_hmul2(dg[0], cg);
        a[1] = ntc_hmul2(dh[0], ch);
        a[2] = ntc_hmul2(dg[1], cg);
        a[3] = ntc_hmul2(dh[1], ch);
    }
    static __device__ __forceinline__ bool fold_at(unsigned int) { return false; }
    static __device__ __forceinline__ float fold_scale(const Tile&, unsigned int) { return 1.0f; }
    static __device__ __forceinline__ float out(const Tile& T, float x) { return x * T.s2; }
};

// 2026-10-02: The BF16 weight-format policy (W16A16): row-major [N, K] BF16 weights, no scales.
// A 16-byte lane load is 8 K of a 32-K chunk; words 2j and 2j + 1 of it feed MMA j (K + 4j + {0,1}
// and {2,3}), the activation words' K order.
struct Bf16Dense {
    static constexpr int CHUNK_K = 32;
    static constexpr bool FOLDS = false;
    static constexpr float ACT_LIFT = 1.0f;
    struct Mat { const unsigned char* w; };
    struct Tile { const unsigned char* wr[2]; };
    struct Sc {};
    static __device__ __forceinline__ Tile tile(const Mat& M, unsigned int col, unsigned int g, unsigned int t, unsigned int K) {
        Tile T;
        T.wr[0] = M.w + (unsigned long long)(col + g) * K * 2 + t * 16;
        T.wr[1] = M.w + (unsigned long long)(col + g + 8) * K * 2 + t * 16;
        return T;
    }
    static __device__ __forceinline__ uint4 load(const Tile& T, int h, unsigned int chunk) {
        return *(const uint4*)(T.wr[h] + chunk * 64);
    }
    static __device__ __forceinline__ Sc scale(const Tile&, int, unsigned int) { return Sc{}; }
    static __device__ __forceinline__ void frag(const uint4& lo, const uint4& hi, Sc, Sc, int j, unsigned int* a) {
        a[0] = j ? lo.z : lo.x;
        a[1] = j ? hi.z : hi.x;
        a[2] = j ? lo.w : lo.y;
        a[3] = j ? hi.w : hi.y;
    }
    static __device__ __forceinline__ bool fold_at(unsigned int) { return false; }
    static __device__ __forceinline__ float fold_scale(const Tile&, unsigned int) { return 1.0f; }
    static __device__ __forceinline__ float out(const Tile&, float x) { return x; }
};
