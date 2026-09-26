// SPDX-License-Identifier: MIT OR Apache-2.0

// 2026-09-25: GDN chunked-prefill output pass, the Hopper twin of
// `gated_delta_rule_chunk_fwd_o` (kernels/gb10/common/gated_delta_rule_fla.cu).
//
// Owner: hopper kernels.
// Invariants:
// - The parent's arguments, grid (num_chunks, nv, batch) and block 512;
//   shared memory is FOH_SMEM = 97,536 B against the parent's 98,816 B.
// - K_DIM = V_DIM = 128 and CHUNK = 64 at compile time;
//   `gdn_hopper_remnant_reject` (crates/model-layers/src/layers/ops/
//   ssm_gdn_hopper_prefill.rs) keeps other shapes on the parent.
// - Only rows i < ce and columns v < v_dim of the chunk are stored.
//
// For token i and value column v:
//   O_i[v] = (exp(gc_i) <q_i, S_c[:, v]>
//             + SUM_{l <= i} exp(gc_i - gc_l) <q_i, k_l> uc_l[v]) * rsqrt(k_dim)
//
// Against the parent:
//  1. All 16 warps compute: each product is tiled as 4 m-tiles x 4
//     n-quarters. The parent runs its two `mma_gram` calls on warps 0-3 and
//     the triangular sum on the `tid < v_dim` threads.
//  2. The triangular sum is an MMA over the masked [64 x 64] kq square
//     (l > i and i >= ce set to zero): 64*64*128 = 524,288 MACs against the
//     triangle's 266,240, with no dependent chain. The decayed kq is split
//     into two bf16 limbs in the C fragment where it is produced, and `uc`
//     is staged transposed as the `.col` operand.
//  3. Fragment reads use the padded 136/72 strides of gdn_prefill_hopper.cuh.
//  4. <q_i, S_c> and the triangular sum accumulate into one f32 fragment,
//     so the first term is not rounded to bf16 before the combine; the
//     output is rounded once.
//
// Numerics against the parent: the kq limbs carry about 16 significant bits,
// the l-sum is reassociated into the MMA's 16-wide tree, and one bf16
// rounding of <q_i, S_c> is gone. `uc` is bf16 in memory on both paths. The
// device oracle is `native_gdn_prefill_remnants_microtest` (it needs a
// kernels/hopper build); ssm_gdn_remnants_tests.rs and
// ssm_gdn_remnants_numerics_tests.rs (crates/model-layers/src/layers/ops/)
// simulate the index maps and the arithmetic on the host.
//
// `__launch_bounds__(512, 2)`: two resident CTAs take 2 x 97,536 B of shared
// memory, within the 228 KB that ssm_gdn_remnants_tests.rs asserts.
// Measured 2026-09-11 with ptxas (CUDA 13.0, -arch=sm_90a, --fmad=false):
// 64 registers a thread, no spill.
// Measurements: GDN-PREFILL-ATTRIBUTION.md.



































#include "gdn_prefill_hopper.cuh"

// 2026-09-25: smem: sq[64][136] + sk[64][136] + Sb[128][136] + ucT[128][72] + kqh[64][72]
//       + gc[64] = 17408 + 17408 + 34816 + 18432 + 9216 + 256 = 97 536 B.
// The `kq` lo limb aliases `sk`, which is dead once the q.k^T Gram
// retires: 9216 <= 17408, and a `__syncthreads` separates the two uses.
// The launcher's copy is GDN_FWD_O_HOPPER_SMEM in ssm_gdn_hopper_prefill.rs.
#define FOH_SMEM                                                               \
    (GDNH_CHUNK * GDNH_SW * 2 + GDNH_CHUNK * GDNH_SW * 2 + GDNH_V_DIM * GDNH_SW * 2 \
     + GDNH_V_DIM * GDNH_SC * 2 + GDNH_CHUNK * GDNH_SC * 2 + GDNH_CHUNK * 4)

static_assert(FOH_SMEM == 97536, "FOH_SMEM must match ssm_gdn_a3.rs::GDN_FWD_O_HOPPER_SMEM");

extern "C" __global__ void __launch_bounds__(512, 2) gated_delta_rule_chunk_fwd_o_hopper(
    const __nv_bfloat16* __restrict__ query, const __nv_bfloat16* __restrict__ key,
    const float* __restrict__ gate, const float* __restrict__ gc_in,
    const __nv_bfloat16* __restrict__ S_in, const __nv_bfloat16* __restrict__ uc_in,
    __nv_bfloat16* __restrict__ output, unsigned int batch_size, unsigned int seq_len,
    unsigned int num_chunks, unsigned int num_k_heads, unsigned int num_v_heads,
    unsigned int k_dim, unsigned int v_dim, unsigned int qk_stride, unsigned int gb_stride,
    const int* __restrict__ cu_seqlens, const int* __restrict__ cu_chunks,
    unsigned int is_varlen) {
    (void)gate;
    (void)gb_stride;
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
    const float inv_sqrt_d = rsqrtf((float)k_dim);
    const unsigned int cs = c * GDNH_CHUNK;
    const unsigned int ce = (g.seqlen - cs) < GDNH_CHUNK ? (g.seqlen - cs) : GDNH_CHUNK;
    const unsigned long long base = ((unsigned long long)(g.choff + c) * num_v_heads + vh);
    const unsigned long long out_base = (g.tokoff * num_v_heads + vh) * v_dim;
    query += g.tokoff * qk_stride;
    key += g.tokoff * qk_stride;

    extern __shared__ __align__(16) char foh_smem[];
    __nv_bfloat16* sq = (__nv_bfloat16*)foh_smem;                    // 2026-09-25: [CHUNK][SW]
    __nv_bfloat16* sk = sq + GDNH_CHUNK * GDNH_SW;                   // 2026-09-25: [CHUNK][SW]
    __nv_bfloat16* kql = sk;                                         // 2026-09-25: [CHUNK][SC], aliases sk
    __nv_bfloat16* Sb = sk + GDNH_CHUNK * GDNH_SW;                   // 2026-09-25: [V_DIM][SW] = S_c^T
    __nv_bfloat16* ucT = Sb + GDNH_V_DIM * GDNH_SW;                  // 2026-09-25: [V_DIM][SC]
    __nv_bfloat16* kqh = ucT + GDNH_V_DIM * GDNH_SC;                 // 2026-09-25: [CHUNK][SC]
    float* gc = (float*)(kqh + GDNH_CHUNK * GDNH_SC);                // 2026-09-25: [CHUNK]

    // 2026-09-25: staging. The q and k tiles (64 x 128) and uc^T (all 128 v-columns) are
    // staged whole, zero-filled past `ce` and past k_dim / v_dim: an MMA reads
    // its whole fragment.

    for (unsigned int idx = tid; idx < GDNH_CHUNK * GDNH_K_DIM; idx += 512) {
        const unsigned int i = idx / GDNH_K_DIM, j = idx % GDNH_K_DIM;
        if (i < ce && j < k_dim) {
            const unsigned long long off =
                (unsigned long long)(cs + i) * qk_stride + kh * k_dim + j;
            sq[i * GDNH_SW + j] = query[off];
            sk[i * GDNH_SW + j] = key[off];
        } else {
            sq[i * GDNH_SW + j] = __float2bfloat16(0.0f);
            sk[i * GDNH_SW + j] = __float2bfloat16(0.0f);
        }
    }
    // 2026-09-25: S_c^T[v][k] = S_c[k][v], iterated in source order (k outer, v inner) so
    // the global read is coalesced and the transpose is paid in shared memory.

    for (unsigned int idx = tid; idx < GDNH_K_DIM * GDNH_V_DIM; idx += 512) {
        const unsigned int k = idx / GDNH_V_DIM, v = idx % GDNH_V_DIM;
        Sb[v * GDNH_SW + k] = S_in[base * GDNH_K_DIM * GDNH_V_DIM + idx];
    }
    for (unsigned int idx = tid; idx < GDNH_CHUNK * GDNH_V_DIM; idx += 512) {
        const unsigned int i = idx / GDNH_V_DIM, v = idx % GDNH_V_DIM;
        ucT[v * GDNH_SC + i] = (i < ce && v < v_dim)
                                   ? uc_in[base * GDNH_CHUNK * GDNH_V_DIM + i * v_dim + v]
                                   : __float2bfloat16(0.0f);
    }
    for (unsigned int i = tid; i < GDNH_CHUNK; i += 512)
        gc[i] = (i < ce) ? gc_in[base * GDNH_CHUNK + i] : 0.0f;
    __syncthreads();

    // 2026-09-25: (1) kq[i][l] = <q_i, k_l>, 16 warps: 4 m-tiles x 4 n-tiles of 16.
    const unsigned int m_base = (warp & 3u) * 16;
    float kqa[2][4] = {{0.0f, 0.0f, 0.0f, 0.0f}, {0.0f, 0.0f, 0.0f, 0.0f}};
    gdnh_mma<2, GDNH_K_DIM, GDNH_SW, GDNH_SW>(sq, sk, m_base, (warp >> 2) * 16, lane, kqa);
    __syncthreads(); // 2026-09-25: every warp is done reading `sk`; its bytes become kql

    // 2026-09-25: (2) fold the decay, apply the causal mask, split to two bf16 limbs. An
    // MMA has no loop bound, so the mask (l <= i, i < ce) is written into the
    // operand as zeros, and the decay is folded in the C fragment.


    {
        const unsigned int i0 = m_base + grp, i1 = i0 + 8;
        const float g0 = gc[i0], g1 = gc[i1];
#pragma unroll
        for (int nt = 0; nt < 2; nt++) {
            const unsigned int l0 = (warp >> 2) * 16 + nt * 8 + q4 * 2, l1 = l0 + 1;
            const unsigned int ii[4] = {i0, i0, i1, i1};
            const unsigned int ll[4] = {l0, l1, l0, l1};
            const float gi[4] = {g0, g0, g1, g1};
#pragma unroll
            for (int e = 0; e < 4; e++) {
                const float x = (ii[e] < ce && ll[e] <= ii[e])
                                    ? expf(gi[e] - gc[ll[e]]) * kqa[nt][e]
                                    : 0.0f;
                __nv_bfloat16 hi, lo;
                gdnh_split(x, hi, lo);
                kqh[ii[e] * GDNH_SC + ll[e]] = hi;
                kql[ii[e] * GDNH_SC + ll[e]] = lo;
            }
        }
    }
    __syncthreads();

    // 2026-09-25: (3) O_i = exp(gc_i) * <q_i, S_c[:,v]> + SUM_{l<=i} kq~[i][l] * uc[l][v]
    // Both products accumulate into one f32 fragment, so the epilogue rounds
    // once, at the bf16 output.
    const unsigned int n_base = (warp >> 2) * 32;
    float acc[4][4];
#pragma unroll
    for (int nt = 0; nt < 4; nt++) {
        acc[nt][0] = 0.0f;
        acc[nt][1] = 0.0f;
        acc[nt][2] = 0.0f;
        acc[nt][3] = 0.0f;
    }
    gdnh_mma<4, GDNH_K_DIM, GDNH_SW, GDNH_SW>(sq, Sb, m_base, n_base, lane, acc);
    {   // 2026-09-25: f32 scale by exp(gc_i) on the accumulator, per row
        const float e0 = expf(gc[m_base + grp]), e1 = expf(gc[m_base + grp + 8]);
#pragma unroll
        for (int nt = 0; nt < 4; nt++) {
            acc[nt][0] *= e0;
            acc[nt][1] *= e0;
            acc[nt][2] *= e1;
            acc[nt][3] *= e1;
        }
    }
    gdnh_mma<4, GDNH_CHUNK, GDNH_SC, GDNH_SC>(kqh, ucT, m_base, n_base, lane, acc);
    gdnh_mma<4, GDNH_CHUNK, GDNH_SC, GDNH_SC>(kql, ucT, m_base, n_base, lane, acc);

    const unsigned int i0 = m_base + grp, i1 = i0 + 8;
#pragma unroll
    for (int nt = 0; nt < 4; nt++) {
        const unsigned int v0 = n_base + nt * 8 + q4 * 2, v1 = v0 + 1;
        const unsigned int ii[4] = {i0, i0, i1, i1};
        const unsigned int vv[4] = {v0, v1, v0, v1};
#pragma unroll
        for (int e = 0; e < 4; e++) {
            if (ii[e] < ce && vv[e] < v_dim)
                output[out_base + (unsigned long long)(cs + ii[e]) * num_v_heads * v_dim
                       + vv[e]] = __float2bfloat16(acc[nt][e] * inv_sqrt_d);
        }
    }
}
