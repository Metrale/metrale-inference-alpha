// SPDX-License-Identifier: MIT OR Apache-2.0

// 2026-09-25: GDN chunked-prefill state spine on tensor cores, a drop-in for
// `gated_delta_rule_chunk_delta_h_vfused` (kernels/gb10/common/gated_delta_rule_fla.cu).
//
// Owner: hopper kernels.
// Invariants:
// - The same 21 arguments, grid (nv, batch) and block 256 as `..._vfused`;
//   S_out and uc_out are written in the layouts
//   `gated_delta_rule_chunk_fwd_o` reads. Shared memory is TCF_SMEM.
// - K_DIM = V_DIM = 128 and CHUNK = 64 at compile time; `gdn_tc_spine_reject`
//   (crates/model-layers/src/layers/ops/ssm_gdn_a3.rs) keeps other shapes,
//   and a qk_stride that is not a multiple of 8, off this kernel.
// - The recurrent state stays in f32 MMA accumulators (64 a thread) from
//   load to store, and the decay multiplies are f32.
//
// Per chunk, both products are `mma.sync.m16n8k16.row.col.f32.bf16.bf16.f32`:
//     Phase A   ws[i][v] = SUM_k W[i][k] * S_c[k][v]         (M=64, N=128, K=128)
//     Phase B   S_{c+1}  = edl*S_c + SUM_i K[i][k]*duc[i][v] (M=128, N=128, K=64)
// That is 128 MMAs per warp per chunk (1024 per CTA) on the one-limb entry.
//
// Numerics: S_c (Phase A's B operand) and duc (Phase B's) are rounded to
// bf16 as MMA operands; W, U and K are bf16 in memory. The k-reductions are
// reassociated into the MMA's 16-wide tree. `..._tcfuse_x2` adds a second
// bf16 limb of both S_c and duc. `GDN_TC_SPINE_ENTRY`
// (crates/model-layers/src/layers/ops/ssm_gdn_tc_route.rs) names `_x2` as
// the entry the `gdn_prefill_tc` lever launches. The oracle is
// `native_gdn_chunk_prefill_microtest`. Measurements: GDN-PREFILL-ATTRIBUTION.md.




























#include <cuda_bf16.h>
#include <cuda_runtime.h>

#define K_DIM 128
#define V_DIM 128
#define CHUNK 64
// 2026-09-25: Padded shared-memory row strides, in bf16. A fragment read loads the
// 4-byte word (row*STRIDE/2 + ks/2 + q); 136 and 72 give a row term of
// row*4 mod 32 (128 and 64 would give 0), so the 8 `grp` rows land on 8
// distinct bank groups.
#define TCF_SW 136   // 2026-09-25: W, U, St: 128 columns + 8 pad
#define TCF_SC 72    // 2026-09-25: Kt, ducT: 64 columns + 8 pad
// 2026-09-25: Stride of the x2 entry's `duc` residual limb. 68 instead of 72 because
// the limb aliases `Wp`, which is dead once Phase A has finished (a
// __syncthreads separates the two uses), and 128*68*2 = 17,408 B is Wp's
// size, so the limb needs no extra shared memory. 68 gives a row term of
// row*2 mod 32, a 2-way bank conflict on reads of this limb.
#define TCF_SCL 68
// 2026-09-25: SSOT for the launcher's `shared_mem`; mirrored as GDN_TC_SMEM in ssm_gdn_a3.rs.
#define TCF_SMEM (V_DIM * TCF_SW * 2 + 2 * (CHUNK * TCF_SW * 2) \
                  + V_DIM * TCF_SC * 2 + (CHUNK + 1) * 4)

// 2026-09-25: Per-stream prefill geometry, the computation of `GDN_GEOM` in
// gated_delta_rule_fla.cu: varlen reads cu_seqlens/cu_chunks, uniform reduces to b*seq_len.
struct TcfGeom { unsigned int seqlen, nchunks, choff; unsigned long long tokoff; };
#define TCF_GEOM(g)                                                            \
    TcfGeom g;                                                                 \
    (void)cu_chunks;                                                           \
    if (is_varlen) {                                                           \
        unsigned int _s0 = (unsigned int)cu_seqlens[b];                        \
        g.seqlen  = (unsigned int)cu_seqlens[b + 1] - _s0;                     \
        g.tokoff  = (unsigned long long)_s0;                                   \
        unsigned int _co = 0;                                                  \
        for (unsigned int _i = 0; _i < b; _i++)                                \
            _co += ((unsigned int)(cu_seqlens[_i + 1] - cu_seqlens[_i])        \
                    + CHUNK - 1) / CHUNK;                                      \
        g.choff   = _co;                                                       \
        g.nchunks = (g.seqlen + CHUNK - 1) / CHUNK;                            \
    } else {                                                                   \
        g.seqlen  = seq_len;                                                   \
        g.tokoff  = (unsigned long long)b * seq_len;                           \
        g.choff   = b * num_chunks;                                            \
        g.nchunks = num_chunks;                                                \
    }

__device__ __forceinline__ void tcf_cp_async16(void* dst_smem, const void* src_gmem) {
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" ::
                 "r"((unsigned int)__cvta_generic_to_shared(dst_smem)), "l"(src_gmem));
}
__device__ __forceinline__ void tcf_cp_commit() { asm volatile("cp.async.commit_group;\n" ::); }
__device__ __forceinline__ void tcf_cp_wait() { asm volatile("cp.async.wait_group 0;\n" ::); }

// 2026-09-25: One warp's slab of C = A * B^T on tensor cores. A is [M][SA] bf16
// row-major, B is [N][SB] bf16 row-major (the `.col` operand, already
// transposed), contraction extent KC. The warp owns m rows
// [m_base, m_base+16) and the NT n-tiles starting at n_base. `acc` is
// accumulated into, never zeroed here, so the caller decides between a
// fresh product and the edl-scaled state.
//
// The fragment addressing is that of `mma_gram` in gated_delta_rule_fla.cu,
// with the strides as template parameters.
template <int NT, int KC, int SA, int SB>
__device__ __forceinline__ void tcf_mma(
    const __nv_bfloat16* __restrict__ A, const __nv_bfloat16* __restrict__ B,
    unsigned int m_base, unsigned int n_base, unsigned int lane, float (&acc)[NT][4]
) {
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
                : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1),
                  "f"(acc[nt][0]), "f"(acc[nt][1]), "f"(acc[nt][2]), "f"(acc[nt][3]));
        }
    }
}

union TcfB8 { uint4 v; __nv_bfloat16 h[8]; };

// 2026-09-25: The spine. 256 threads = 8 warps. The f32 state S[k][v] lives in the
// Phase-B MMA accumulator: warp w owns k-rows [16w, 16w+16) and all 16
// n-tiles, i.e.
//   acc[nt][0..3] <-> S[16w+grp][8nt+2q], S[16w+grp][8nt+2q+1],
//                     S[16w+grp+8][8nt+2q], S[16w+grp+8][8nt+2q+1]
// with grp = lane>>2, q = lane&3: 64 f32 registers of state per thread.
template <bool X2>
__device__ __forceinline__ void tcf_core(
    float* __restrict__ h_state, const __nv_bfloat16* __restrict__ W_in,
    const __nv_bfloat16* __restrict__ U_in, const __nv_bfloat16* __restrict__ key,
    const float* __restrict__ gc_in,
    __nv_bfloat16* __restrict__ S_out, __nv_bfloat16* __restrict__ uc_out,
    unsigned int seq_len, unsigned int num_chunks,
    unsigned int num_k_heads, unsigned int num_v_heads, unsigned int k_dim,
    unsigned int v_dim, unsigned int qk_stride,
    unsigned int h_state_is_table,
    const int* __restrict__ cu_seqlens, const int* __restrict__ cu_chunks,
    unsigned int is_varlen
) {
    const unsigned int vh = blockIdx.x;
    const unsigned int b = blockIdx.y;
    if (vh >= num_v_heads) return;
    TCF_GEOM(g);

    const unsigned int tid = threadIdx.x;
    const unsigned int warp = tid >> 5, lane = tid & 31;
    const unsigned int grp = lane >> 2, q = lane & 3;
    const unsigned int head_repeat = num_v_heads / num_k_heads;
    const unsigned int kh = vh / head_repeat;

    extern __shared__ __align__(16) char tcf_smem[];
    __nv_bfloat16* St = (__nv_bfloat16*)tcf_smem;              // 2026-09-25: [V_DIM][TCF_SW]
    __nv_bfloat16* Kt = St;                                    // 2026-09-25: [K_DIM][TCF_SC], aliases St in Phase B
    __nv_bfloat16* Wp = St + V_DIM * TCF_SW;                   // 2026-09-25: [CHUNK][TCF_SW]
    __nv_bfloat16* Up = Wp + CHUNK * TCF_SW;                   // 2026-09-25: [CHUNK][TCF_SW]
    __nv_bfloat16* ducT = Up + CHUNK * TCF_SW;                 // 2026-09-25: [V_DIM][TCF_SC]
    float* dec = (float*)(ducT + V_DIM * TCF_SC);              // 2026-09-25: [CHUNK+1], [0] = exp(gc_last)

    float* H = h_state_is_table
        ? ((float* const*)h_state)[b] + (unsigned long long)vh * K_DIM * V_DIM
        : h_state + ((unsigned long long)(b * num_v_heads + vh) * K_DIM * V_DIM);

    // 2026-09-25: The Phase-B accumulator is the recurrent state. Load S_0 from the f32 pool.
    const unsigned int m0 = warp * 16 + grp, m1 = m0 + 8;
    float acc[16][4];
    #pragma unroll
    for (int nt = 0; nt < 16; nt++) {
        const unsigned int n0 = nt * 8 + q * 2;
        acc[nt][0] = H[m0 * V_DIM + n0];     acc[nt][1] = H[m0 * V_DIM + n0 + 1];
        acc[nt][2] = H[m1 * V_DIM + n0];     acc[nt][3] = H[m1 * V_DIM + n0 + 1];
    }

    const __nv_bfloat16* key_b = key + g.tokoff * qk_stride;
    // 2026-09-25: Phase-A warp split: 4 m-tiles (i) x 2 halves of the 16 n-tiles (v).
    const unsigned int a_m = (warp & 3u) * 16, a_n = (warp >> 2) * 64;
    // 2026-09-25: K staging map: thread owns token row `krow` and the 32 k-columns at `kcol`.
    const unsigned int krow = tid >> 2, kcol = (tid & 3u) * 32;

    for (unsigned int c = 0; c < g.nchunks; c++) {
        const unsigned int cs = c * CHUNK;
        const unsigned int ce = (g.seqlen - cs) < CHUNK ? (g.seqlen - cs) : CHUNK;
        const unsigned long long base = ((unsigned long long)(g.choff + c) * num_v_heads + vh);

        __syncthreads();   // 2026-09-25: the previous chunk's Phase B is done reading Kt/ducT
        // 2026-09-25: (1) stage W and U (cp.async, 16 B = 8 bf16 per issue) and the decay row.
        for (unsigned int idx = tid * 8; idx < CHUNK * K_DIM; idx += 256 * 8)
            tcf_cp_async16(&Wp[(idx / K_DIM) * TCF_SW + (idx % K_DIM)],
                           &W_in[base * CHUNK * K_DIM + idx]);
        for (unsigned int idx = tid * 8; idx < CHUNK * V_DIM; idx += 256 * 8)
            tcf_cp_async16(&Up[(idx / V_DIM) * TCF_SW + (idx % V_DIM)],
                           &U_in[base * CHUNK * V_DIM + idx]);
        tcf_cp_commit();
        {   // 2026-09-25: f32 decay, one thread per token.
            const float gl = gc_in[base * CHUNK + ce - 1];
            if (tid == 0) dec[0] = expf(gl);
            if (tid < CHUNK)
                dec[1 + tid] = (tid < ce) ? expf(gl - gc_in[base * CHUNK + tid]) : 0.0f;
        }
        // 2026-09-25: (2) entry state S_c -> S_out (bf16, read by chunk_fwd_o) and the
        //     bf16 snapshot St[v][k] that Phase A contracts against. The f32
        //     master in `acc` is only read here.
        #pragma unroll
        for (int nt = 0; nt < 16; nt++) {
            const unsigned int n0 = nt * 8 + q * 2, n1 = n0 + 1;
            const __nv_bfloat16 s00 = __float2bfloat16(acc[nt][0]);
            const __nv_bfloat16 s01 = __float2bfloat16(acc[nt][1]);
            const __nv_bfloat16 s10 = __float2bfloat16(acc[nt][2]);
            const __nv_bfloat16 s11 = __float2bfloat16(acc[nt][3]);
            S_out[base * K_DIM * V_DIM + m0 * V_DIM + n0] = s00;
            S_out[base * K_DIM * V_DIM + m0 * V_DIM + n1] = s01;
            S_out[base * K_DIM * V_DIM + m1 * V_DIM + n0] = s10;
            S_out[base * K_DIM * V_DIM + m1 * V_DIM + n1] = s11;
            St[n0 * TCF_SW + m0] = s00;  St[n1 * TCF_SW + m0] = s01;
            St[n0 * TCF_SW + m1] = s10;  St[n1 * TCF_SW + m1] = s11;
        }
        tcf_cp_wait();
        __syncthreads();
        // 2026-09-25: Rows past the sequence end hold whatever recompute_wu left there. The
        // MMA reads all 64 rows (row m of C depends only on row m of A), so they
        // are zeroed rather than relied on; `duc` is zeroed for them below too.
        if (ce < CHUNK)
            for (unsigned int e = tid; e < (CHUNK - ce) * TCF_SW; e += 256) {
                Wp[ce * TCF_SW + e] = __float2bfloat16(0.0f);
                Up[ce * TCF_SW + e] = __float2bfloat16(0.0f);
            }
        __syncthreads();

        // 2026-09-25: (3) K for this chunk into registers, issued before the Phase-A MMAs so
        //     its global latency hides behind them (Kt aliases St, which Phase A
        //     is still reading).
        TcfB8 kr[4];
        if (krow < ce) {
            const __nv_bfloat16* src =
                key_b + (unsigned long long)(cs + krow) * qk_stride + kh * k_dim + kcol;
            #pragma unroll
            for (int j = 0; j < 4; j++) kr[j].v = *(const uint4*)(src + j * 8);
        } else {
            #pragma unroll
            for (int j = 0; j < 4; j++) kr[j].v = make_uint4(0u, 0u, 0u, 0u);
        }

        // 2026-09-25: (4) Phase A (tensor core): ws[i][v] = <W_i, S_c[:,v]>.
        float wsa[8][4];
        #pragma unroll
        for (int nt = 0; nt < 8; nt++) {
            wsa[nt][0] = 0.0f; wsa[nt][1] = 0.0f; wsa[nt][2] = 0.0f; wsa[nt][3] = 0.0f;
        }
        tcf_mma<8, K_DIM, TCF_SW, TCF_SW>(Wp, St, a_m, a_n, lane, wsa);
        if (X2) {
            // 2026-09-25: Second bf16 limb of S_c (x2 entry). `uc = U - W.S_c` is a
            // difference, so S_c's bf16 rounding error reaches `uc` amplified by
            // |W.S_c| / |uc|. With S = hi + lo, hi = bf16(S) and lo = bf16(S - hi),
            // W.S is formed from about 16 significant bits of S. The lo limb
            // overwrites `St` in place: the f32 master is in `acc`, so the limb is
            // recomputed, not stored, and needs no extra shared memory.










            __syncthreads();   // 2026-09-25: every warp is done reading the hi limb
            #pragma unroll
            for (int nt = 0; nt < 16; nt++) {
                const unsigned int n0 = nt * 8 + q * 2, n1 = n0 + 1;
                const float a0 = acc[nt][0], a1 = acc[nt][1];
                const float a2 = acc[nt][2], a3 = acc[nt][3];
                St[n0 * TCF_SW + m0] = __float2bfloat16(a0 - (float)__float2bfloat16(a0));
                St[n1 * TCF_SW + m0] = __float2bfloat16(a1 - (float)__float2bfloat16(a1));
                St[n0 * TCF_SW + m1] = __float2bfloat16(a2 - (float)__float2bfloat16(a2));
                St[n1 * TCF_SW + m1] = __float2bfloat16(a3 - (float)__float2bfloat16(a3));
            }
            __syncthreads();
            tcf_mma<8, K_DIM, TCF_SW, TCF_SW>(Wp, St, a_m, a_n, lane, wsa);
        }

        // 2026-09-25: (5) uc = U - ws ; duc = exp(gc_last - gc_i) * uc, written transposed as
        //     Phase B's `.col` operand. Every (i, v) is covered exactly once by
        //     the 8 warps (4 m-tiles x 2 n-halves).
        if (X2) __syncthreads();   // 2026-09-25: Wp is dead; the duc residual limb takes it
        __nv_bfloat16* ducL = Wp;  // 2026-09-25: [V_DIM][TCF_SCL], x2 entry only
        const unsigned int i0 = a_m + grp, i1 = i0 + 8;
#define TCF_EMIT(ii, vv, a)                                                    \
            do {                                                                   \
                const float uci = (float)Up[(ii) * TCF_SW + (vv)] - (a);           \
                if ((ii) < ce)                                                     \
                    uc_out[base * CHUNK * V_DIM + (ii) * v_dim + (vv)] =            \
                        __float2bfloat16(uci);                                      \
                const float d = (ii) < ce ? dec[1 + (ii)] * uci : 0.0f;            \
                const __nv_bfloat16 dh = __float2bfloat16(d);                      \
                ducT[(vv) * TCF_SC + (ii)] = dh;                                    \
                if (X2)                                                             \
                    ducL[(vv) * TCF_SCL + (ii)] =                                   \
                        __float2bfloat16(d - (float)dh);                            \
            } while (0)
        #pragma unroll
        for (int nt = 0; nt < 8; nt++) {
            const unsigned int v0 = a_n + nt * 8 + q * 2, v1 = v0 + 1;
            TCF_EMIT(i0, v0, wsa[nt][0]);
            TCF_EMIT(i0, v1, wsa[nt][1]);
            TCF_EMIT(i1, v0, wsa[nt][2]);
            TCF_EMIT(i1, v1, wsa[nt][3]);
        }
#undef TCF_EMIT
        __syncthreads();   // 2026-09-25: St is dead past Phase A; Kt may overwrite it

        // 2026-09-25: (6) K^T into the freed St region: Kt[k][i] = K[i][k].
        #pragma unroll
        for (int j = 0; j < 4; j++)
            #pragma unroll
            for (int e = 0; e < 8; e++) Kt[(kcol + j * 8 + e) * TCF_SC + krow] = kr[j].h[e];
        __syncthreads();

        // 2026-09-25: (7) Phase B (tensor core): S_{c+1} = edl*S_c + K^T * duc. The edl scale
        //     is an f32 multiply on the accumulator; the MMA accumulates the
        //     correction into the same f32 registers.
        const float edl = dec[0];
        #pragma unroll
        for (int nt = 0; nt < 16; nt++) {
            acc[nt][0] *= edl; acc[nt][1] *= edl; acc[nt][2] *= edl; acc[nt][3] *= edl;
        }
        tcf_mma<16, CHUNK, TCF_SC, TCF_SC>(Kt, ducT, warp * 16, 0, lane, acc);
        if (X2) {
            // 2026-09-25: Second bf16 limb of `duc` (x2 entry), from the residual
            // d - bf16(d) written by TCF_EMIT.





            tcf_mma<16, CHUNK, TCF_SC, TCF_SCL>(Kt, ducL, warp * 16, 0, lane, acc);
        }
    }

    #pragma unroll
    for (int nt = 0; nt < 16; nt++) {
        const unsigned int n0 = nt * 8 + q * 2;
        H[m0 * V_DIM + n0] = acc[nt][0];     H[m0 * V_DIM + n0 + 1] = acc[nt][1];
        H[m1 * V_DIM + n0] = acc[nt][2];     H[m1 * V_DIM + n0 + 1] = acc[nt][3];
    }
}

// 2026-09-25: The two entry points. The same 21 arguments, grid [nv, batch], block 256
// and shared memory TCF_SMEM. `_x2` adds a second bf16 limb of S_c (Phase A,
// written over `St`) and of `duc` (Phase B, over the dead `Wp`), so it needs
// no extra shared memory and no extra global traffic.



#define TCF_ENTRY(NAME, SPLIT)                                                 \
    extern "C" __global__ void __launch_bounds__(256, 1) NAME(                 \
        float* __restrict__ h_state, const __nv_bfloat16* __restrict__ W_in,   \
        const __nv_bfloat16* __restrict__ U_in,                                \
        const __nv_bfloat16* __restrict__ key, const float* __restrict__ gate, \
        const float* __restrict__ gc_in, __nv_bfloat16* __restrict__ S_out,    \
        __nv_bfloat16* __restrict__ uc_out, unsigned int batch_size,           \
        unsigned int seq_len, unsigned int num_chunks, unsigned int num_k_heads,\
        unsigned int num_v_heads, unsigned int k_dim, unsigned int v_dim,      \
        unsigned int qk_stride, unsigned int gb_stride,                        \
        unsigned int h_state_is_table, const int* __restrict__ cu_seqlens,     \
        const int* __restrict__ cu_chunks, unsigned int is_varlen) {           \
        (void)gate;                                                            \
        (void)gb_stride;                                                       \
        (void)batch_size;                                                      \
        tcf_core<SPLIT>(h_state, W_in, U_in, key, gc_in, S_out, uc_out,        \
                        seq_len, num_chunks, num_k_heads, num_v_heads, k_dim,  \
                        v_dim, qk_stride, h_state_is_table, cu_seqlens,        \
                        cu_chunks, is_varlen);                                 \
    }

TCF_ENTRY(gated_delta_rule_chunk_delta_h_tcfuse, false)
TCF_ENTRY(gated_delta_rule_chunk_delta_h_tcfuse_x2, true)
