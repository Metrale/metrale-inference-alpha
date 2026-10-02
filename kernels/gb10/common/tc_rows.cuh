// SPDX-License-Identifier: MIT OR Apache-2.0

// 2026-10-02: The dense tensor-core row-tile projection, parameterized by a weight-format policy
// (tc_weight_formats.cuh): C[r, n] = sum_k A[r, k] * W[n, k] for 1..=8 * NT rows on mma.sync
// m16n8k16 BF16 tiles with the rows as the MMA's N columns. Instantiated for FP8 block-128
// (w8a16_tc_rows.cu) and NVFP4 g16 (w4a16_tc_rows.cu).
//
// Owner: gb10 kernels.
// Invariants:
// - A [M, lda] BF16 (the first K of each row read), C [M, ldc] BF16 (the first N of each row
//   written); the host guarantees 1 <= M <= 8 * NT, N a positive multiple of TR_COLS (and of
//   128 for FP8) unless RAGGED (any N; grid ceil(N / TR_COLS)), K a positive multiple of
//   P::CHUNK_K * G, lda a multiple of 8 and >= K, ldc >= N.
// - Activations are staged per load group in shared memory times P::ACT_LIFT; a row's sum
//   order is fixed by K alone and its MMA column reads only its own activations, so a row's
//   output bits do not depend on M, on the other rows, on NT or on G.
// - Grid (N / TR_COLS, 1, 1), block TR_THREADS, static shared memory only.

#pragma once

#include <cuda_bf16.h>

#include "tc_weight_formats.cuh"

#define TR_WARPS 4
#define TR_THREADS (TR_WARPS * 32)
#define TR_COLS (TR_WARPS * 16)

__device__ __forceinline__ void tr_mma_bf16(float* c, const unsigned int* a, unsigned int b0, unsigned int b1) {
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
        : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}

// 2026-09-28: One block: TR_COLS output columns, one m-tile (16 columns) per warp, NT row tiles
// of 8. The weights stream in load groups of G chunks of P::CHUNK_K (a weight row in 16 * G-byte
// lane runs), one group ahead; each group's activations are loaded one group ahead into
// registers and stored (times P::ACT_LIFT) into the other shared buffer after the current
// group's MMAs. G changes only how loads are batched, never the arithmetic.
template <class P, int NT, int G, bool RAGGED = false>
__device__ __forceinline__ void tr_block(
    const __nv_bfloat16* __restrict__ A, const typename P::Mat& W, __nv_bfloat16* __restrict__ C,
    unsigned int M, unsigned int N, unsigned int K, unsigned int lda, unsigned int ldc,
    unsigned int col_block
) {
    constexpr int GK = P::CHUNK_K * G;
    constexpr int MMAS = P::CHUNK_K / 16;
    constexpr int XU4 = P::CHUNK_K / 32;   // activation uint4 per lane per chunk
    // 2026-09-28: Shared row pitch in bytes: 16 bytes of padding keep the eight rows of one
    // 16-byte fragment load on distinct bank groups.
    constexpr int RS = GK * 2 + 16;
    constexpr int ROWS = 8 * NT;
    constexpr int U4 = ROWS * GK * 2 / 16;
    constexpr int PER = (U4 + TR_THREADS - 1) / TR_THREADS;
    __shared__ __align__(16) unsigned char xs[2][ROWS * RS];
    const unsigned int warp = threadIdx.x >> 5, lane = threadIdx.x & 31, g = lane >> 2, t = lane & 3;
    const unsigned int f0 = col_block * TR_COLS + warp * 16;
    const unsigned int ngroups = K / GK;
    const unsigned int nt_live = (M + 7) / 8;
    const __nv_bfloat162 liftx2 = __floats2bfloat162_rn(P::ACT_LIFT, P::ACT_LIFT);

    uint4 xr[PER];
    auto group_load = [&](unsigned int gi) {
        #pragma unroll
        for (int i = 0; i < PER; i++) {
            const unsigned int u = threadIdx.x + i * TR_THREADS;
            const unsigned int row = u / (GK / 8), col = u % (GK / 8);
            xr[i] = (u < U4 && row < M)
                ? *(const uint4*)(A + (unsigned long long)row * lda + gi * GK + col * 8)
                : make_uint4(0u, 0u, 0u, 0u);
        }
    };
    auto group_store = [&](unsigned int buf) {
        #pragma unroll
        for (int i = 0; i < PER; i++) {
            const unsigned int u = threadIdx.x + i * TR_THREADS;
            if (u >= U4) break;
            const unsigned int row = u / (GK / 8), col = u % (GK / 8);
            if (P::ACT_LIFT != 1.0f) {
                unsigned int* w = (unsigned int*)&xr[i];
                #pragma unroll
                for (int q = 0; q < 4; q++) {
                    __nv_bfloat162 v = *(__nv_bfloat162*)&w[q];
                    v = __hmul2(v, liftx2);
                    w[q] = *(unsigned int*)&v;
                }
            }
            *(uint4*)(&xs[buf][row * RS + col * 16]) = xr[i];
        }
    };

    const typename P::Tile T = P::tile(W, f0, g, t, K);
    // 2026-10-02: RAGGED: N need not be a multiple of TR_COLS; weight rows at or past N load as
    // zero and their columns are not stored.
#define wload(h, chunk) ((RAGGED && f0 + g + 8 * (h) >= N) ? make_uint4(0u, 0u, 0u, 0u) : P::load(T, (h), (chunk)))
#define sload(h, chunk) ((RAGGED && f0 + g + 8 * (h) >= N) ? typename P::Sc{} : P::scale(T, (h), (chunk)))
    float acc[NT][4], tmp[NT][4];
    #pragma unroll
    for (int n = 0; n < NT; n++)
        #pragma unroll
        for (int e = 0; e < 4; e++) acc[n][e] = tmp[n][e] = 0.f;

    group_load(0);
    group_store(0);
    uint4 wn[G][2];
    typename P::Sc sn[G][2];
    #pragma unroll
    for (int c = 0; c < G; c++) {
        wn[c][0] = wload(0, c); wn[c][1] = wload(1, c);
        sn[c][0] = sload(0, c); sn[c][1] = sload(1, c);
    }
    __syncthreads();
    for (unsigned int gi = 0; gi < ngroups; gi++) {
        const unsigned int buf = gi & 1;
        uint4 w[G][2];
        typename P::Sc s[G][2];
        #pragma unroll
        for (int c = 0; c < G; c++) { w[c][0] = wn[c][0]; w[c][1] = wn[c][1]; s[c][0] = sn[c][0]; s[c][1] = sn[c][1]; }
        if (gi + 1 < ngroups) {
            #pragma unroll
            for (int c = 0; c < G; c++) {
                wn[c][0] = wload(0, (gi + 1) * G + c);
                wn[c][1] = wload(1, (gi + 1) * G + c);
                sn[c][0] = sload(0, (gi + 1) * G + c);
                sn[c][1] = sload(1, (gi + 1) * G + c);
            }
            group_load(gi + 1);
        }
        #pragma unroll
        for (int c = 0; c < G; c++) {
            const unsigned int chunk = gi * G + c;
            unsigned int a[MMAS][4];
            #pragma unroll
            for (int j = 0; j < MMAS; j++) P::frag(w[c][0], w[c][1], s[c][0], s[c][1], j, a[j]);
            #pragma unroll
            for (int n = 0; n < NT; n++) {
                if (n >= (int)nt_live) break;
                const uint4* xp = (const uint4*)(&xs[buf][(n * 8 + g) * RS + c * (P::CHUNK_K * 2) + t * (P::CHUNK_K / 2)]);
                uint4 xv[XU4];
                #pragma unroll
                for (int u = 0; u < XU4; u++) xv[u] = xp[u];
                unsigned int xw[4 * XU4];
                #pragma unroll
                for (int u = 0; u < XU4; u++) { xw[4 * u] = xv[u].x; xw[4 * u + 1] = xv[u].y; xw[4 * u + 2] = xv[u].z; xw[4 * u + 3] = xv[u].w; }
                #pragma unroll
                for (int j = 0; j < MMAS; j++) tr_mma_bf16(P::FOLDS ? tmp[n] : acc[n], a[j], xw[2 * j], xw[2 * j + 1]);
            }
            // 2026-09-28: FP8: chunks 2kb and 2kb + 1 make 128-K block kb: scale it once.
            if (P::FOLDS && P::fold_at(chunk)) {
                const float sc = P::fold_scale(T, chunk);
                #pragma unroll
                for (int n = 0; n < NT; n++) {
                    if (n >= (int)nt_live) break;
                    #pragma unroll
                    for (int e = 0; e < 4; e++) { acc[n][e] += tmp[n][e] * sc; tmp[n][e] = 0.f; }
                }
            }
        }
        if (gi + 1 < ngroups) group_store(buf ^ 1);
        __syncthreads();
    }
    // 2026-09-28: acc[n][e]: column f0 + g (+ 8 for e >= 2) of row 8n + 2t + (e & 1).
    #pragma unroll
    for (int n = 0; n < NT; n++) {
        if (n >= (int)nt_live) break;
        #pragma unroll
        for (int e = 0; e < 4; e++) {
            const unsigned int r = n * 8 + 2 * t + (e & 1);
            const unsigned int col = f0 + g + ((e >> 1) ? 8 : 0);
            if (r < M && (!RAGGED || col < N)) C[(unsigned long long)r * ldc + col] = __float2bfloat16(P::out(T, acc[n][e]));
        }
    }
}
#undef wload
#undef sload
