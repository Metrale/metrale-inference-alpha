// SPDX-License-Identifier: MIT OR Apache-2.0

// 2026-10-02: The tensor-core grouped MoE decode warp, parameterized by a weight-format policy:
// mma.sync m16n8k16 BF16 tiles, FP32 accumulation, rows TC_ROWS at a time as the MMA's N
// columns. Instantiated for FP8 E4M3 with 128x128 block scales (moe_fp8_grouped_tc.cu) and for
// NVFP4 (packed E2M1, E4M3 block scales of 16, a per-tensor FP32 scale; moe_nvfp4_grouped_tc.cu).
//
// Owner: gb10 kernels.
// Invariants:
// - The weight-format policy P (tc_weight_formats.cuh: Fp8Block128, Nvfp4G16) fixes the chunk,
//   the fragment K order (shared with tc_rows.cuh), the block and tensor scaling and ACT_LIFT.
// - Column r of an MMA depends only on row r's activations, so a row's output bits do not
//   depend on which rows share its launch, its pass, its expert or RG (padding rows are zero).
// - gate+up rounds gate and up to BF16 (after P::out), forms the FP32 SiLU product a and stores
//   it as two BF16 terms of a * ACT_LIFT, hi then lo, in the act buffer's FP32 row space (row r:
//   N hi values then N lo values); down runs one MMA on each, hi first.

#pragma once

#include <cuda_bf16.h>

#include "moe_fp8_grouped_tc_rows.cuh"

__device__ __forceinline__ void gtc_mma_bf16(float* c, const unsigned int* a, unsigned int b0, unsigned int b1) {
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
        : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}

// 2026-10-02: One warp's MT output tiles (GATE_UP: MT gate tiles then MT up tiles of the same
// columns) over rows [begin, end). Input row r is X[row(r)], row(r) = sorted_token_ids[r]
// unless by_pos: BF16 [.., K] for gate+up, the hi|lo SiLU rows [.., 2K] for down. RG row groups
// of TC_ROWS share each decoded weight fragment; group q is its own MMA column block. G chunks
// make one load group; a warp keeps two groups in flight.
template <class P, bool GATE_UP, int MT, int RG, int G>
__device__ __forceinline__ void gtc_warp(
    const void* __restrict__ X, const int* __restrict__ sorted_token_ids, bool by_pos,
    unsigned int begin, unsigned int end, const typename P::Mat& M0, const typename P::Mat& M1,
    void* __restrict__ out, unsigned int N, unsigned int K, unsigned int f0
) {
    constexpr int TILES = GATE_UP ? 2 * MT : MT;
    constexpr unsigned int LANE_K = P::CHUNK_K / 4;   // K a lane holds per chunk
    constexpr unsigned int XU4 = LANE_K / 8;           // activation uint4 per lane per chunk
    constexpr unsigned int CU4 = P::CHUNK_K / 8;       // uint4 per chunk of an activation row
    constexpr int XW = LANE_K / 2;                     // activation words per lane per chunk
    const unsigned int lane = threadIdx.x & 31, g = lane >> 2, t = lane & 3;
    const unsigned int ngroups = (K / P::CHUNK_K) / G;
    const __nv_bfloat162 liftx2 = __floats2bfloat162_rn(P::ACT_LIFT, P::ACT_LIFT);
    typename P::Tile tile[TILES];
    #pragma unroll
    for (int m = 0; m < TILES; m++) {
        const bool second = GATE_UP && m >= MT;
        tile[m] = P::tile(second ? M1 : M0, f0 + 16 * (m % MT), g, t, K);
    }

    for (unsigned int row0 = begin; row0 < end; row0 += TC_ROWS * RG) {
        unsigned int cnt[RG];
        bool live[RG];
        const uint4* xp[RG];
        const uint4* xh[RG];
        const uint4* xlo[RG];
        #pragma unroll
        for (int q = 0; q < RG; q++) {
            const unsigned int r0 = row0 + q * TC_ROWS;
            cnt[q] = r0 < end ? min((unsigned int)TC_ROWS, end - r0) : 0u;
            live[q] = g < cnt[q];
            const unsigned int xrow = live[q] ? (by_pos ? r0 + g : (unsigned int)sorted_token_ids[r0 + g]) : 0u;
            xp[q] = (const uint4*)((const __nv_bfloat16*)X + (unsigned long long)xrow * K + t * LANE_K);
            // 2026-10-02: The down input's hi and lo rows (already times ACT_LIFT).
            xh[q] = (const uint4*)((const __nv_bfloat16*)X + (unsigned long long)xrow * 2 * K + t * LANE_K);
            xlo[q] = (const uint4*)((const __nv_bfloat16*)X + (unsigned long long)xrow * 2 * K + K + t * LANE_K);
        }
        float acc[RG][TILES][4], tmp[RG][TILES][4];
        #pragma unroll
        for (int q = 0; q < RG; q++)
            #pragma unroll
            for (int m = 0; m < TILES; m++)
                #pragma unroll
                for (int e = 0; e < 4; e++) acc[q][m][e] = tmp[q][m][e] = 0.f;
        uint4 wn[G][TILES][2];
        typename P::Sc sn[G][TILES][2];
        #pragma unroll
        for (int c = 0; c < G; c++)
            #pragma unroll
            for (int m = 0; m < TILES; m++)
                #pragma unroll
                for (int h = 0; h < 2; h++) {
                    wn[c][m][h] = P::load(tile[m], h, c);
                    sn[c][m][h] = P::scale(tile[m], h, c);
                }
        for (unsigned int gi = 0; gi < ngroups; gi++) {
            uint4 w[G][TILES][2];
            typename P::Sc s[G][TILES][2];
            #pragma unroll
            for (int c = 0; c < G; c++)
                #pragma unroll
                for (int m = 0; m < TILES; m++) {
                    w[c][m][0] = wn[c][m][0]; w[c][m][1] = wn[c][m][1];
                    s[c][m][0] = sn[c][m][0]; s[c][m][1] = sn[c][m][1];
                }
            if (gi + 1 < ngroups) {
                #pragma unroll
                for (int c = 0; c < G; c++)
                    #pragma unroll
                    for (int m = 0; m < TILES; m++) {
                        wn[c][m][0] = P::load(tile[m], 0, (gi + 1) * G + c);
                        wn[c][m][1] = P::load(tile[m], 1, (gi + 1) * G + c);
                        sn[c][m][0] = P::scale(tile[m], 0, (gi + 1) * G + c);
                        sn[c][m][1] = P::scale(tile[m], 1, (gi + 1) * G + c);
                    }
            }
            #pragma unroll
            for (int c = 0; c < G; c++) {
                const unsigned int chunk = gi * G + c;
                unsigned int xw[RG][XW], xl[RG][XW];
                #pragma unroll
                for (int q = 0; q < RG; q++) {
                    if (GATE_UP) {
                        uint4 xa[XU4];
                        #pragma unroll
                        for (int u = 0; u < (int)XU4; u++) xa[u] = make_uint4(0u, 0u, 0u, 0u);
                        if (live[q]) {
                            #pragma unroll
                            for (int u = 0; u < (int)XU4; u++) xa[u] = xp[q][chunk * CU4 + u];
                        }
                        #pragma unroll
                        for (int u = 0; u < (int)XU4; u++) {
                            const unsigned int raw[4] = {xa[u].x, xa[u].y, xa[u].z, xa[u].w};
                            #pragma unroll
                            for (int i = 0; i < 4; i++) {
                                __nv_bfloat162 v = *(const __nv_bfloat162*)&raw[i];
                                if (P::ACT_LIFT != 1.0f) v = __hmul2(v, liftx2);
                                xw[q][4 * u + i] = *(unsigned int*)&v;
                            }
                        }
                    } else {
                        uint4 ha[XU4], la[XU4];
                        #pragma unroll
                        for (int u = 0; u < (int)XU4; u++) ha[u] = la[u] = make_uint4(0u, 0u, 0u, 0u);
                        if (live[q]) {
                            #pragma unroll
                            for (int u = 0; u < (int)XU4; u++) ha[u] = xh[q][chunk * CU4 + u];
                            #pragma unroll
                            for (int u = 0; u < (int)XU4; u++) la[u] = xlo[q][chunk * CU4 + u];
                        }
                        #pragma unroll
                        for (int u = 0; u < (int)XU4; u++) {
                            xw[q][4 * u + 0] = ha[u].x; xw[q][4 * u + 1] = ha[u].y; xw[q][4 * u + 2] = ha[u].z; xw[q][4 * u + 3] = ha[u].w;
                            xl[q][4 * u + 0] = la[u].x; xl[q][4 * u + 1] = la[u].y; xl[q][4 * u + 2] = la[u].z; xl[q][4 * u + 3] = la[u].w;
                        }
                    }
                }
                #pragma unroll
                for (int j = 0; j < P::CHUNK_K / 16; j++)
                    #pragma unroll
                    for (int m = 0; m < TILES; m++) {
                        unsigned int a[4];
                        P::frag(w[c][m][0], w[c][m][1], s[c][m][0], s[c][m][1], j, a);
                        #pragma unroll
                        for (int q = 0; q < RG; q++) {
                            float* d = P::FOLDS ? tmp[q][m] : acc[q][m];
                            gtc_mma_bf16(d, a, xw[q][2 * j], xw[q][2 * j + 1]);
                            if (!GATE_UP) gtc_mma_bf16(d, a, xl[q][2 * j], xl[q][2 * j + 1]);
                        }
                    }
                if (P::FOLDS && P::fold_at(chunk)) {
                    #pragma unroll
                    for (int m = 0; m < TILES; m++) {
                        const float sc = P::fold_scale(tile[m], chunk);
                        #pragma unroll
                        for (int q = 0; q < RG; q++)
                            #pragma unroll
                            for (int e = 0; e < 4; e++) { acc[q][m][e] += tmp[q][m][e] * sc; tmp[q][m][e] = 0.f; }
                    }
                }
            }
        }
        // 2026-10-02: acc[q][m][e]: column f + g (+ 8 for e >= 2) of row 2t + (e & 1) of group q.
        #pragma unroll
        for (int q = 0; q < RG; q++)
            #pragma unroll
            for (int e = 0; e < 4; e++) {
                const unsigned int r = 2 * t + (e & 1);
                if (r >= cnt[q]) continue;
                const unsigned int row = row0 + q * TC_ROWS + r;
                const unsigned long long o = (unsigned long long)row * N + f0 + g + ((e >> 1) ? 8 : 0);
                #pragma unroll
                for (int m = 0; m < MT; m++) {
                    if (GATE_UP) {
                        const float gv = __bfloat162float(__float2bfloat16(P::out(tile[m], acc[q][m][e])));
                        const float uv = __bfloat162float(__float2bfloat16(P::out(tile[m + MT], acc[q][m + MT][e])));
                        const float hv = (gv / (1.0f + __expf(-gv))) * uv * P::ACT_LIFT;
                        const __nv_bfloat16 hi = __float2bfloat16(hv);
                        const unsigned long long ro = (unsigned long long)row * 2 * N + f0 + g + ((e >> 1) ? 8 : 0);
                        ((__nv_bfloat16*)out)[ro + 16 * m] = hi;
                        ((__nv_bfloat16*)out)[ro + N + 16 * m] = __float2bfloat16(hv - __bfloat162float(hi));
                    } else {
                        ((__nv_bfloat16*)out)[o + 16 * m] = __float2bfloat16(P::out(tile[m], acc[q][m][e]));
                    }
                }
            }
    }
}

// 2026-10-02: gtc_warp for a routed expert's rows: RG = 2 (one weight pass per 16 rows) when the
// expert has more than TC_ROWS rows, else RG = 1. The shared expert's block rows hold at most
// TC_ROWS rows and keep RG = 1.
template <class P, bool GATE_UP, int MT, int G>
__device__ __forceinline__ void gtc_warp_routed(
    const void* __restrict__ X, const int* __restrict__ sorted_token_ids, bool by_pos,
    unsigned int begin, unsigned int end, const typename P::Mat& M0, const typename P::Mat& M1,
    void* __restrict__ out, unsigned int N, unsigned int K, unsigned int f0
) {
    if (end - begin > TC_ROWS)
        gtc_warp<P, GATE_UP, MT, 2, G>(X, sorted_token_ids, by_pos, begin, end, M0, M1, out, N, K, f0);
    else
        gtc_warp<P, GATE_UP, MT, 1, G>(X, sorted_token_ids, by_pos, begin, end, M0, M1, out, N, K, f0);
}
