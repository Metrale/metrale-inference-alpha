// SPDX-License-Identifier: MIT OR Apache-2.0

// 2026-09-25: BF16-weight small-M GEMV on tensor cores: C[t, n] = sum_k A[t, k] * W[n, k].
//
// Owner: gb10 kernels.
// Invariants:
// - Arguments match dense_gemv_bf16_batchm: A is [M, K] contiguous, W is [N, K] row-major,
//   row t of C starts at C + t * out_stride.
// - Requires K % 64 == 0; the host route() declines any other K. Any N: a weight row past N
//   loads zeros and is never stored. A token row past M feeds zeros and is never stored.
// - Launch: grid (ceil(N / 16), 1, 1), block (DTC_WARPS * 32 = 256, 1, 1); NT = 1 in every
//   entry, so a CTA owns one 16-row weight tile.
// - The DTC_WARPS warps split K (interleaved 64-k blocks) and reduce through shared memory
//   in a fixed warp order with no atomics, so the output is deterministic, and a token's
//   partials never mix with another token's.
// - The result is not bit-identical to the CUDA-core GEMV: accumulation happens inside
//   the MMA.
//
// Caller: the MTP drafter (mtp_head), through ops/dense_gemv_tc.rs, for M = 2..=32.
//
// Operand placement: weights on the 16-row A side of mma.m16n8k16, tokens on the 8-wide
// B side, so one MMA covers a 16x16 weight block for up to 8 tokens and the raw BF16 words
// go into the fragments with no per-weight conversion. NB token tiles reuse each A
// fragment; a tile whose first row is at or past M is skipped (warp-uniform).
//
// K permutation (why there is no repack): thread (g = lane / 4, t = lane % 4) loads two
// 16-byte pieces of weight rows g and g + 8 of its tile, at k offsets 8t and 32 + 8t of a
// 64-k block, straight from the [N, K] layout. Its eight BF16x2 words w0..w7 (piece 0,
// then piece 1) feed four MMAs: MMA j takes w(2j) in k slots {2t, 2t+1} and w(2j+1) in
// {2t+8, 2t+9}. The B fragment of token g is built from the same 16 k of activation row g
// in the same order, so both operands carry the same permutation and the dot product is
// unchanged.
























#include <cuda_bf16.h>
#include <stdint.h>

#define DTC_WARPS 8
#define DTC_KB 64

__device__ __forceinline__ void dtc_mma(float (&d)[4], uint32_t a0, uint32_t a1,
                                        uint32_t a2, uint32_t a3, uint32_t b0,
                                        uint32_t b1) {
    asm(
        "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
        "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
        : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
        : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
}

// 2026-09-25: Streamed weight load: each weight is read once per launch, so it bypasses L1
// and leaves L1 to the activation rows every CTA re-reads.
__device__ __forceinline__ uint4 dtc_ld_stream(const void* p) {
    uint4 v;
    asm volatile("ld.global.nc.L1::no_allocate.v4.u32 {%0,%1,%2,%3}, [%4];\n"
                 : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w)
                 : "l"(p));
    return v;
}

__device__ __forceinline__ uint32_t dtc_word(const uint4& v, int i) {
    return i == 0 ? v.x : i == 1 ? v.y : i == 2 ? v.z : v.w;
}

template <int NB, int NT, int KU>
__device__ __forceinline__ void dense_gemv_bf16_tc_impl(
    const __nv_bfloat16* __restrict__ A,
    const __nv_bfloat16* __restrict__ W,
    __nv_bfloat16* __restrict__ C,
    unsigned int M, unsigned int N, unsigned int K, unsigned int out_stride)
{
    const unsigned int warp = threadIdx.x >> 5;
    const unsigned int lane = threadIdx.x & 31u;
    const unsigned int g = lane >> 2;
    const unsigned int t = lane & 3u;
    const unsigned int n0 = blockIdx.x * (16u * NT);
    const unsigned int num_kb = K / DTC_KB;

    // 2026-09-25: Weight rows g and g+8 of each 16-row tile; a row past N loads zeros.
    bool live[NT][2];
    const __nv_bfloat16* wrow[NT][2];
    #pragma unroll
    for (int i = 0; i < NT; i++) {
        #pragma unroll
        for (int h = 0; h < 2; h++) {
            const unsigned int n = n0 + (unsigned int)i * 16u + (unsigned int)h * 8u + g;
            live[i][h] = n < N;
            wrow[i][h] = W + (unsigned long long)(live[i][h] ? n : 0u) * K + t * 8u;
        }
    }
    // 2026-09-25: Token tile b is live iff its first row is < M (warp-uniform). Within a
    // live tile, a token row b*8+g past M feeds zeros.
    bool tile_on[NB];
    bool tok_live[NB];
    const __nv_bfloat16* arow[NB];
    #pragma unroll
    for (int b = 0; b < NB; b++) {
        const unsigned int tok = (unsigned int)b * 8u + g;
        tile_on[b] = (unsigned int)b * 8u < M;
        tok_live[b] = tok < M;
        arow[b] = A + (unsigned long long)(tok_live[b] ? tok : 0u) * K + t * 8u;
    }

    float acc[NT][NB][4];
    #pragma unroll
    for (int i = 0; i < NT; i++) {
        #pragma unroll
        for (int b = 0; b < NB; b++) {
            #pragma unroll
            for (int c = 0; c < 4; c++) acc[i][b][c] = 0.0f;
        }
    }

    // 2026-09-25: KU k-blocks per trip: all their weight loads are issued before any MMA,
    // so each thread has KU * NT * 2 rows x 32 B of weights in flight.
    for (unsigned int kb0 = warp; kb0 < num_kb; kb0 += DTC_WARPS * KU) {
        uint4 w[KU][NT][2][2];
        #pragma unroll
        for (int u = 0; u < KU; u++) {
            const unsigned int kb = kb0 + (unsigned int)u * DTC_WARPS;
            #pragma unroll
            for (int i = 0; i < NT; i++) {
                #pragma unroll
                for (int h = 0; h < 2; h++) {
                    if (live[i][h] && kb < num_kb) {
                        const __nv_bfloat16* p = wrow[i][h] + kb * DTC_KB;
                        w[u][i][h][0] = dtc_ld_stream(p);
                        w[u][i][h][1] = dtc_ld_stream(p + 32);
                    } else {
                        w[u][i][h][0] = make_uint4(0u, 0u, 0u, 0u);
                        w[u][i][h][1] = make_uint4(0u, 0u, 0u, 0u);
                    }
                }
            }
        }
        #pragma unroll
        for (int u = 0; u < KU; u++) {
            const unsigned int kb = kb0 + (unsigned int)u * DTC_WARPS;
            if (kb >= num_kb) break;
            #pragma unroll
            for (int b = 0; b < NB; b++) {
                if (!tile_on[b]) continue;
                uint4 av[2];
                if (tok_live[b]) {
                    const uint4* ap = (const uint4*)(arow[b] + kb * DTC_KB);
                    av[0] = ap[0];
                    av[1] = ap[4];   // 2026-09-25: +32 k, the same split as the weights
                } else {
                    av[0] = make_uint4(0u, 0u, 0u, 0u);
                    av[1] = make_uint4(0u, 0u, 0u, 0u);
                }
                #pragma unroll
                for (int j = 0; j < 4; j++) {
                    // 2026-09-25: Words 2j and 2j+1 of this thread's 16 k (j < 2: first piece).
                    const uint32_t b0 = dtc_word(av[j >> 1], (2 * j) & 3);
                    const uint32_t b1 = dtc_word(av[j >> 1], (2 * j + 1) & 3);
                    #pragma unroll
                    for (int i = 0; i < NT; i++) {
                        const uint32_t a0 = dtc_word(w[u][i][0][j >> 1], (2 * j) & 3);
                        const uint32_t a2 = dtc_word(w[u][i][0][j >> 1], (2 * j + 1) & 3);
                        const uint32_t a1 = dtc_word(w[u][i][1][j >> 1], (2 * j) & 3);
                        const uint32_t a3 = dtc_word(w[u][i][1][j >> 1], (2 * j + 1) & 3);
                        dtc_mma(acc[i][b], a0, a1, a2, a3, b0, b1);
                    }
                }
            }
        }
    }

    // 2026-09-25: Fixed-order split-K reduction across the CTA's warps.
    __shared__ float red[DTC_WARPS][NT * NB][4][32];
    #pragma unroll
    for (int i = 0; i < NT; i++) {
        #pragma unroll
        for (int b = 0; b < NB; b++) {
            #pragma unroll
            for (int c = 0; c < 4; c++) red[warp][i * NB + b][c][lane] = acc[i][b][c];
        }
    }
    __syncthreads();

    // 2026-09-25: D fragment: d0, d1 = (weight row g, tokens 2t, 2t+1); d2, d3 = row g+8.
    for (unsigned int f = warp; f < (unsigned int)(NT * NB); f += DTC_WARPS) {
        const unsigned int i = f / NB;
        const unsigned int b = f % NB;
        if (b * 8u >= M) continue;
        float r[4];
        #pragma unroll
        for (int c = 0; c < 4; c++) {
            float v = red[0][f][c][lane];
            #pragma unroll
            for (int ww = 1; ww < DTC_WARPS; ww++) v += red[ww][f][c][lane];
            r[c] = v;
        }
        const unsigned int tok0 = b * 8u + t * 2u;
        const unsigned int n_lo = n0 + i * 16u + g;
        const unsigned int n_hi = n_lo + 8u;
        #pragma unroll
        for (int c = 0; c < 4; c++) {
            const unsigned int tok = tok0 + (unsigned int)(c & 1);
            const unsigned int n = (c < 2) ? n_lo : n_hi;
            if (tok < M && n < N)
                C[(unsigned long long)tok * out_stride + n] = __float2bfloat16_rn(r[c]);
        }
    }
}

#define DTC_ENTRY(NAME, NB, NT, KU)                                                     \
    extern "C" __global__ __launch_bounds__(DTC_WARPS * 32) void NAME(                   \
        const __nv_bfloat16* __restrict__ A, const __nv_bfloat16* __restrict__ W,         \
        __nv_bfloat16* __restrict__ C, unsigned int M, unsigned int N, unsigned int K,    \
        unsigned int out_stride) {                                                        \
        dense_gemv_bf16_tc_impl<NB, NT, KU>(A, W, C, M, N, K, out_stride);               \
    }

// 2026-09-25: Entries per token-tile count NB (M <= 8 / 16 / 32); each covers every M up to
// its cap. Geometry measured 2026-09-23 on GB10, one 27B draft position (849 MB of cold
// BF16 weights): weight bytes in flight per thread moved the time, not occupancy. KU=8
// (512 B/thread, 1 CTA/SM) beat KU=4 (2 CTAs/SM) by 3.5-4.4%; KU=2 at 4 CTAs/SM and KU=3/4
// at 3 CTAs/SM were slower than KU=4; NT=2 was slower than KU=8. Against
// dense_gemv_bf16_batchm at M=2/4/8: 3554/3583/3583 us vs 3487/3564/3631 us, and
// 97/117/134 mJ vs 149/168/264 mJ per position. tc16 at KU=8: 3602 us (KU=4: 3737).
// tc32 at KU=4: 3803 us (KU=2: 3885; KU=8 does not fit).




DTC_ENTRY(dense_gemv_bf16_tc8, 1, 1, 8)
DTC_ENTRY(dense_gemv_bf16_tc16, 2, 1, 8)
DTC_ENTRY(dense_gemv_bf16_tc32, 4, 1, 4)
