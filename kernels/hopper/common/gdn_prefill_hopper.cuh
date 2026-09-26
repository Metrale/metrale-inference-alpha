// SPDX-License-Identifier: MIT OR Apache-2.0

// 2026-09-25: Shared device code for the Hopper GDN chunked-prefill twins,
// `gdn_fwd_o_hopper.cu` and `gdn_recompute_wu_hopper.cu`.
//
// Owner: hopper kernels.
// Invariants: none beyond the types; each helper states its contract.
// Measurements: GDN-PREFILL-ATTRIBUTION.md.




















#ifndef METRALE_GDN_PREFILL_HOPPER_CUH
#define METRALE_GDN_PREFILL_HOPPER_CUH

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#define GDNH_K_DIM 128
#define GDNH_V_DIM 128
#define GDNH_CHUNK 64

// 2026-09-25: Padded shared-memory row strides, in bf16 elements.
//
// A fragment read loads the 4-byte word (row * STRIDE/2 + ks/2 + q). With
// the parents' unpadded 128-element rows (`mma_gram` in
// kernels/gb10/common/gated_delta_rule_fla.cu) the row term is row*64 mod 32
// = 0, so the eight `grp` rows of a fragment hit the same banks. 136 gives
// row*68 mod 32 = row*4, 72 gives row*36 mod 32 = row*4 and 24 gives
// row*12 mod 32: eight distinct groups of four banks in each case.


#define GDNH_SW 136 // 2026-09-25: 128-column tiles: q/k/S panels
#define GDNH_SC 72  // 2026-09-25: 64-column tiles: Gram limbs, uc^T, L limbs
#define GDNH_SX 24  // 2026-09-25: 16-column tiles: the per-warp triangular-solve panel

// 2026-09-25: Per-stream prefill geometry, the computation of `GDN_GEOM` in
// gated_delta_rule_fla.cu and `TCF_GEOM` in gated_delta_rule_chunk_tc.cu:
// varlen reads cu_seqlens/cu_chunks, uniform reduces to b*seq_len.


struct GdnhGeom {
    unsigned int seqlen, nchunks, choff;
    unsigned long long tokoff;
};
#define GDNH_GEOM(g)                                                           \
    GdnhGeom g;                                                                \
    (void)cu_chunks;                                                           \
    if (is_varlen) {                                                           \
        unsigned int _s0 = (unsigned int)cu_seqlens[b];                        \
        g.seqlen = (unsigned int)cu_seqlens[b + 1] - _s0;                      \
        g.tokoff = (unsigned long long)_s0;                                    \
        unsigned int _co = 0;                                                  \
        for (unsigned int _i = 0; _i < b; _i++)                                \
            _co += ((unsigned int)(cu_seqlens[_i + 1] - cu_seqlens[_i])        \
                    + GDNH_CHUNK - 1)                                          \
                   / GDNH_CHUNK;                                               \
        g.choff = _co;                                                         \
        g.nchunks = (g.seqlen + GDNH_CHUNK - 1) / GDNH_CHUNK;                  \
    } else {                                                                   \
        g.seqlen = seq_len;                                                    \
        g.tokoff = (unsigned long long)b * seq_len;                            \
        g.choff = b * num_chunks;                                              \
        g.nchunks = num_chunks;                                                \
    }

// 2026-09-25: One warp's slab of C[m][n] += SUM_k A[m][k] * B[n][k] on tensor cores.
// A is [.][SA] bf16 row-major, B is [.][SB] bf16 row-major (the `.col`
// operand, indexed [n][k]), contraction extent KC (a multiple of 16). The
// warp owns m rows [m_base, m_base+16) and the NT n-tiles of 8 starting at
// n_base. `acc` is accumulated into and never zeroed here.
//
// The fragment addressing is that of `mma_gram` (gated_delta_rule_fla.cu)
// and `tcf_mma` (gated_delta_rule_chunk_tc.cu), with the strides as
// template parameters.

template <int NT, int KC, int SA, int SB>
__device__ __forceinline__ void gdnh_mma(const __nv_bfloat16* __restrict__ A,
                                         const __nv_bfloat16* __restrict__ B,
                                         unsigned int m_base, unsigned int n_base,
                                         unsigned int lane, float (&acc)[NT][4]) {
    const unsigned int grp = lane >> 2, q = lane & 3;
    const unsigned short* sA = (const unsigned short*)A;
    const unsigned short* sB = (const unsigned short*)B;
#pragma unroll
    for (int ks = 0; ks < KC; ks += 16) {
        const unsigned int fr0 = m_base + grp, fr1 = fr0 + 8;
        const unsigned int fc0 = ks + q * 2, fc1 = fc0 + 8;
        const unsigned int a0 = *(const unsigned int*)&sA[fr0 * SA + fc0];
        const unsigned int a1 = *(const unsigned int*)&sA[fr1 * SA + fc0];
        const unsigned int a2 = *(const unsigned int*)&sA[fr0 * SA + fc1];
        const unsigned int a3 = *(const unsigned int*)&sA[fr1 * SA + fc1];
#pragma unroll
        for (int nt = 0; nt < NT; nt++) {
            const unsigned int nc = n_base + nt * 8 + grp;
            const unsigned int k0 = ks + q * 2, k1 = k0 + 8;
            const unsigned int b0 =
                ((unsigned int)sB[nc * SB + k0 + 1] << 16) | (unsigned int)sB[nc * SB + k0];
            const unsigned int b1 =
                ((unsigned int)sB[nc * SB + k1 + 1] << 16) | (unsigned int)sB[nc * SB + k1];
            asm volatile(
                "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
                "{%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%10,%11,%12,%13};"
                : "=f"(acc[nt][0]), "=f"(acc[nt][1]), "=f"(acc[nt][2]), "=f"(acc[nt][3])
                : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1), "f"(acc[nt][0]),
                  "f"(acc[nt][1]), "f"(acc[nt][2]), "f"(acc[nt][3]));
        }
    }
}

// 2026-09-25: Split one f32 into two bf16 limbs: hi = bf16(x), lo = bf16(x - hi). The
// pair carries about 16 significant bits against one limb's 8.



__device__ __forceinline__ void gdnh_split(float x, __nv_bfloat16& hi, __nv_bfloat16& lo) {
    hi = __float2bfloat16(x);
    lo = __float2bfloat16(x - (float)hi);
}

#endif
