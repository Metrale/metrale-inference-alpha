// SPDX-License-Identifier: MIT OR Apache-2.0

// 2026-09-25: GDN chunked-prefill WY pass, the Hopper twin of
// `gated_delta_rule_recompute_wu` (kernels/gb10/common/gated_delta_rule_fla.cu).
//
// Owner: hopper kernels.
// Invariants:
// - The parent's arguments and grid (num_chunks, nv, batch); block 512
//   against 256 and shared memory WUH_SMEM = 79,104 B against 33,024 B.
// - K_DIM = V_DIM = 128 and CHUNK = 64 at compile time;
//   `gdn_hopper_remnant_reject` (crates/model-layers/src/layers/ops/
//   ssm_gdn_hopper_prefill.rs) keeps other shapes on the parent.
// - `gc_out` is computed as in the parent (logs in parallel, the additions
//   serially on one thread in index order), so it is bit-identical to it.
// - Only rows i < ce and columns below v_dim (U) or k_dim (W) are stored.
//
// The parent solves (I + L) U = beta V and (I + L) W = beta exp(gc) K by
// forward substitution, one thread per right-hand-side column. This twin
// runs a blocked solve over four blocks of 16 rows; for j = 0..3
//     X_j <- T_jj . B_j              T_jj = (I + L_jj)^-1, 16x16
//     B_i <- B_i - L_ij . X_j        for every i > j
// Both steps are `mma.sync.m16n8k16`. The 256 right-hand-side columns (128
// of U, 128 of W) are split 16 to a warp, so each of the 16 warps holds its
// [64 x 16] panel in C fragments (32 f32 registers) and the solve is
// warp-local (`__syncwarp` only). Per solve that is 4 diagonal applications
// and 6 off-diagonal updates of 16x16x128, 327,680 MACs, against the
// substitution's 258,048.
//
// The diagonal blocks are inverted, not substituted: T_jj is built once per
// (chunk, head) by an f32 forward substitution, one lane per column in warps
// 0-3 (680 MACs per block), and shared by both solves.
//
// K.K^T and the L build are fused: the Gram is symmetric, so the parent's
// kk[l][i] is this C fragment's own kk[i][l], and no f32 Gram buffer is
// needed.
//
// Numerics against the parent: the blocked solve with an explicit diagonal
// inverse is algebraically identical but rounds differently. Every MMA
// operand is two bf16 limbs and each product is Ah.Bh + Ah.Bl + Al.Bh
// (about 16 significant bits), because the solve subtracts L.X from B and
// so amplifies operand error by |L.X| / |B|. The k-sums are reassociated
// into the MMA's 16-wide tree. The device oracle is
// `native_gdn_prefill_remnants_microtest` (it needs a kernels/hopper
// build); ssm_gdn_remnants_tests.rs and ssm_gdn_remnants_numerics_tests.rs
// (crates/model-layers/src/layers/ops/) simulate the index maps and the
// arithmetic on the host.
//
// `__launch_bounds__(512, 1)`: measured 2026-09-11 with ptxas (CUDA 13.0,
// -arch=sm_90a, --fmad=false), (512, 1) takes 114 registers and no stack
// frame, while (512, 2) caps it at 64 registers and spills 84 B.
// Measurements: GDN-PREFILL-ATTRIBUTION.md.

















































#include "gdn_prefill_hopper.cuh"

// 2026-09-25: The parent's GATE_FLOOR. A zero gate would make gc -inf and
// exp(gc_i - gc_l) NaN.

#define GATE_FLOOR 1e-30f

// 2026-09-25: smem: sk[64][136] + Ld[64][24]f32 + Tf[64][24]f32 + Lh/Ll[64][72]
//       + Th/Tl[64][24] + Xh/Xl[16 warps][16][24] + gc[64]f32
//     = 17408 + 6144 + 6144 + 9216 + 9216 + 3072 + 3072 + 12288 + 12288 + 256
//     = 79 104 B. The launcher's copy is GDN_WU_HOPPER_SMEM in ssm_gdn_hopper_prefill.rs.
#define WUH_SMEM                                                                   \
    (GDNH_CHUNK * GDNH_SW * 2 + 2 * (GDNH_CHUNK * GDNH_SX * 4)                      \
     + 2 * (GDNH_CHUNK * GDNH_SC * 2) + 2 * (GDNH_CHUNK * GDNH_SX * 2)              \
     + 2 * (16 * 16 * GDNH_SX * 2) + GDNH_CHUNK * 4)

static_assert(WUH_SMEM == 79104, "WUH_SMEM must match ssm_gdn_a3.rs::GDN_WU_HOPPER_SMEM");

// 2026-09-25: Publish one 16x16 C fragment as the `.col` MMA operand `panel[col][row]`,
// in two bf16 limbs. The caller owns the `__syncwarp` on either side.
__device__ __forceinline__ void wuh_publish(const float (&p)[2][4], __nv_bfloat16* __restrict__ Xh,
                                            __nv_bfloat16* __restrict__ Xl, unsigned int grp,
                                            unsigned int q4) {
#pragma unroll
    for (int nt = 0; nt < 2; nt++) {
#pragma unroll
        for (int e = 0; e < 4; e++) {
            const unsigned int r = grp + (e >= 2 ? 8u : 0u);
            const unsigned int cl = nt * 8 + q4 * 2 + (e & 1);
            __nv_bfloat16 hi, lo;
            gdnh_split(p[nt][e], hi, lo);
            Xh[cl * GDNH_SX + r] = hi;
            Xl[cl * GDNH_SX + r] = lo;
        }
    }
}

extern "C" __global__ void __launch_bounds__(512, 1) gated_delta_rule_recompute_wu_hopper(
    const __nv_bfloat16* __restrict__ key, const __nv_bfloat16* __restrict__ value,
    const float* __restrict__ gate, const float* __restrict__ beta,
    __nv_bfloat16* __restrict__ W_out, __nv_bfloat16* __restrict__ U_out,
    float* __restrict__ gc_out, unsigned int batch_size, unsigned int seq_len,
    unsigned int num_chunks, unsigned int num_k_heads, unsigned int num_v_heads,
    unsigned int k_dim, unsigned int v_dim, unsigned int qk_stride, unsigned int v_stride,
    unsigned int gb_stride, const int* __restrict__ cu_seqlens,
    const int* __restrict__ cu_chunks, unsigned int is_varlen) {
    const unsigned int c = blockIdx.x;
    const unsigned int vh = blockIdx.y;
    const unsigned int b = blockIdx.z;
    if (vh >= num_v_heads || b >= batch_size) return;
    GDNH_GEOM(g);
    if (c >= g.nchunks) return;

    const unsigned int tid = threadIdx.x;
    const unsigned int warp = tid >> 5, lane = tid & 31;
    const unsigned int grp = lane >> 2, q4 = lane & 3;
    const unsigned int head_repeat = num_v_heads / num_k_heads;
    const unsigned int kh = vh / head_repeat;
    const unsigned int cs = c * GDNH_CHUNK;
    const unsigned int ce = (g.seqlen - cs) < GDNH_CHUNK ? (g.seqlen - cs) : GDNH_CHUNK;
    const unsigned long long bs = ((unsigned long long)(g.choff + c) * num_v_heads + vh);
    key += g.tokoff * qk_stride;
    value += g.tokoff * v_stride;
    gate += g.tokoff * gb_stride;
    beta += g.tokoff * gb_stride;

    extern __shared__ __align__(16) char wuh_smem[];
    __nv_bfloat16* sk = (__nv_bfloat16*)wuh_smem;                 // 2026-09-25: [CHUNK][SW]
    float* Ld = (float*)(sk + GDNH_CHUNK * GDNH_SW);              // 2026-09-25: [CHUNK][SX] diagonal blocks of L
    float* Tf = Ld + GDNH_CHUNK * GDNH_SX;                        // 2026-09-25: [CHUNK][SX] their inverses
    __nv_bfloat16* Lh = (__nv_bfloat16*)(Tf + GDNH_CHUNK * GDNH_SX); // 2026-09-25: [CHUNK][SC] hi(-L)
    __nv_bfloat16* Ll = Lh + GDNH_CHUNK * GDNH_SC;                // 2026-09-25: [CHUNK][SC] lo(-L)
    __nv_bfloat16* Th = Ll + GDNH_CHUNK * GDNH_SC;                // 2026-09-25: [CHUNK][SX] hi(T_jj)
    __nv_bfloat16* Tl = Th + GDNH_CHUNK * GDNH_SX;                // 2026-09-25: [CHUNK][SX] lo(T_jj)
    __nv_bfloat16* Xh = Tl + GDNH_CHUNK * GDNH_SX;                // 2026-09-25: [16][16][SX] hi panel
    __nv_bfloat16* Xl = Xh + 16 * 16 * GDNH_SX;                   // 2026-09-25: [16][16][SX] lo panel
    float* gc = (float*)(Xl + 16 * 16 * GDNH_SX);                 // 2026-09-25: [CHUNK]

    for (unsigned int idx = tid; idx < GDNH_CHUNK * GDNH_K_DIM; idx += 512) {
        const unsigned int i = idx / GDNH_K_DIM, j = idx % GDNH_K_DIM;
        sk[i * GDNH_SW + j] =
            (i < ce && j < k_dim)
                ? key[(unsigned long long)(cs + i) * qk_stride + kh * k_dim + j]
                : __float2bfloat16(0.0f);
    }
    // 2026-09-25: Order-stable gc scan, as in the parent: the 64 loads and logf calls in
    // parallel, the additions serially on one thread in index order. The
    // parent's comment records the measurement against a tree scan here.


    for (unsigned int idx = tid; idx < GDNH_CHUNK; idx += 512)
        gc[idx] = (idx < ce) ? logf(fmaxf(gate[(unsigned long long)(cs + idx) * gb_stride + vh],
                                          GATE_FLOOR))
                             : 0.0f;
    __syncthreads();
    if (tid == 0) {
        float a = 0.0f;
        for (unsigned int i = 0; i < ce; i++) {
            a += gc[i];
            gc[i] = a;
        }
    }
    __syncthreads();
    for (unsigned int idx = tid; idx < GDNH_CHUNK; idx += 512) {
        if (idx >= ce)
            gc[idx] = 0.0f;
        else
            gc_out[bs * GDNH_CHUNK + idx] = gc[idx];
    }
    __syncthreads();

    // 2026-09-25: (1) K.K^T and the L build, fused in the C fragment.
    // L[i][l] = beta_i * exp(gc_i - gc_l) * <k_l, k_i> for l < i. The Gram is
    // symmetric, so <k_l, k_i> is this fragment's own element. `-L` is stored,
    // because an MMA only accumulates and the update is a subtraction.

    {
        const unsigned int mg = (warp & 3u) * 16, ng = (warp >> 2) * 16;
        float kka[2][4] = {{0.0f, 0.0f, 0.0f, 0.0f}, {0.0f, 0.0f, 0.0f, 0.0f}};
        gdnh_mma<2, GDNH_K_DIM, GDNH_SW, GDNH_SW>(sk, sk, mg, ng, lane, kka);
        const unsigned int i0 = mg + grp, i1 = i0 + 8;
        const float b0 =
            (i0 < ce) ? beta[(unsigned long long)(cs + i0) * gb_stride + vh] : 0.0f;
        const float b1 =
            (i1 < ce) ? beta[(unsigned long long)(cs + i1) * gb_stride + vh] : 0.0f;
        const float g0 = gc[i0], g1 = gc[i1];
#pragma unroll
        for (int nt = 0; nt < 2; nt++) {
            const unsigned int l0 = ng + nt * 8 + q4 * 2, l1 = l0 + 1;
            const unsigned int ii[4] = {i0, i0, i1, i1};
            const unsigned int ll[4] = {l0, l1, l0, l1};
            const float bb[4] = {b0, b0, b1, b1};
            const float gg[4] = {g0, g0, g1, g1};
#pragma unroll
            for (int e = 0; e < 4; e++) {
                const float v = (ii[e] < ce && ll[e] < ii[e])
                                    ? bb[e] * expf(gg[e] - gc[ll[e]]) * kka[nt][e]
                                    : 0.0f;
                __nv_bfloat16 hi, lo;
                gdnh_split(-v, hi, lo);
                Lh[ii[e] * GDNH_SC + ll[e]] = hi;
                Ll[ii[e] * GDNH_SC + ll[e]] = lo;
                if ((ii[e] >> 4) == (ll[e] >> 4))
                    Ld[ii[e] * GDNH_SX + (ll[e] & 15u)] = v;
            }
        }
    }
    __syncthreads();

    // 2026-09-25: (2) T_jj = (I + L_jj)^-1, f32 forward substitution.
    // Warp j owns block j, lane c owns column c: every value it reads is one it
    // wrote, so the recurrence needs no barrier. Rows past `ce` have a zero L
    // row, which makes their T row the identity.

    if (warp < 4 && lane < 16) {
        const unsigned int r0 = warp * 16, cc = lane;
        for (unsigned int r = 0; r < 16; r++)
            Tf[(r0 + r) * GDNH_SX + cc] = (r == cc) ? 1.0f : 0.0f;
        for (unsigned int r = cc + 1; r < 16; r++) {
            float s = 0.0f;
            for (unsigned int m = cc; m < r; m++)
                s -= Ld[(r0 + r) * GDNH_SX + m] * Tf[(r0 + m) * GDNH_SX + cc];
            Tf[(r0 + r) * GDNH_SX + cc] = s;
        }
    }
    __syncthreads();
    for (unsigned int idx = tid; idx < GDNH_CHUNK * 16; idx += 512) {
        const unsigned int r = idx / 16, cc = idx % 16;
        __nv_bfloat16 hi, lo;
        gdnh_split(Tf[r * GDNH_SX + cc], hi, lo);
        Th[r * GDNH_SX + cc] = hi;
        Tl[r * GDNH_SX + cc] = lo;
    }
    __syncthreads();

    // 2026-09-25: (3) the blocked solve, warp-local.
    // Warp w solves columns [16*(w&7), +16) of U (w < 8) or W (w >= 8).
    const unsigned int solve = warp >> 3;
    const unsigned int nb = (warp & 7u) * 16;
    const unsigned int lim = (solve == 0) ? v_dim : k_dim;
    __nv_bfloat16* Xhw = Xh + warp * 16 * GDNH_SX;
    __nv_bfloat16* Xlw = Xl + warp * 16 * GDNH_SX;

    float acc[4][2][4];
#pragma unroll
    for (int mt = 0; mt < 4; mt++) {
#pragma unroll
        for (int nt = 0; nt < 2; nt++) {
#pragma unroll
            for (int e = 0; e < 4; e++) {
                const unsigned int i = mt * 16 + grp + (e >= 2 ? 8u : 0u);
                const unsigned int col = nb + nt * 8 + q4 * 2 + (e & 1);
                float v = 0.0f;
                if (i < ce && col < lim) {
                    const float bi = beta[(unsigned long long)(cs + i) * gb_stride + vh];
                    v = (solve == 0)
                            ? bi
                                  * (float)value[(unsigned long long)(cs + i) * v_stride
                                                 + vh * v_dim + col]
                            : bi * expf(gc[i]) * (float)sk[i * GDNH_SW + col];
                }
                acc[mt][nt][e] = v;
            }
        }
    }

#pragma unroll
    for (int j = 0; j < 4; j++) {
        wuh_publish(acc[j], Xhw, Xlw, grp, q4);
        __syncwarp();
        float x[2][4] = {{0.0f, 0.0f, 0.0f, 0.0f}, {0.0f, 0.0f, 0.0f, 0.0f}};
        const __nv_bfloat16* th = Th + (unsigned int)(j * 16) * GDNH_SX;
        const __nv_bfloat16* tl = Tl + (unsigned int)(j * 16) * GDNH_SX;
        gdnh_mma<2, 16, GDNH_SX, GDNH_SX>(th, Xhw, 0, 0, lane, x);
        gdnh_mma<2, 16, GDNH_SX, GDNH_SX>(th, Xlw, 0, 0, lane, x);
        gdnh_mma<2, 16, GDNH_SX, GDNH_SX>(tl, Xhw, 0, 0, lane, x);
#pragma unroll
        for (int nt = 0; nt < 2; nt++)
#pragma unroll
            for (int e = 0; e < 4; e++) acc[j][nt][e] = x[nt][e];
        __syncwarp();
        wuh_publish(acc[j], Xhw, Xlw, grp, q4);
        __syncwarp();
#pragma unroll
        for (int i = j + 1; i < 4; i++) {
            const __nv_bfloat16* ah = Lh + (unsigned int)(i * 16) * GDNH_SC + j * 16;
            const __nv_bfloat16* al = Ll + (unsigned int)(i * 16) * GDNH_SC + j * 16;
            gdnh_mma<2, 16, GDNH_SC, GDNH_SX>(ah, Xhw, 0, 0, lane, acc[i]);
            gdnh_mma<2, 16, GDNH_SC, GDNH_SX>(ah, Xlw, 0, 0, lane, acc[i]);
            gdnh_mma<2, 16, GDNH_SC, GDNH_SX>(al, Xhw, 0, 0, lane, acc[i]);
        }
        __syncwarp();
    }

#pragma unroll
    for (int mt = 0; mt < 4; mt++) {
#pragma unroll
        for (int nt = 0; nt < 2; nt++) {
#pragma unroll
            for (int e = 0; e < 4; e++) {
                const unsigned int i = mt * 16 + grp + (e >= 2 ? 8u : 0u);
                const unsigned int col = nb + nt * 8 + q4 * 2 + (e & 1);
                if (i >= ce || col >= lim) continue;
                const __nv_bfloat16 o = __float2bfloat16(acc[mt][nt][e]);
                if (solve == 0)
                    U_out[bs * GDNH_CHUNK * GDNH_V_DIM + i * v_dim + col] = o;
                else
                    W_out[bs * GDNH_CHUNK * GDNH_K_DIM + i * k_dim + col] = o;
            }
        }
    }
}
