// SPDX-License-Identifier: MIT OR Apache-2.0

// 2026-09-25: Persistent, activation-staged W4A4 small-M GEMV (`w4a4_gemv_mx{16,32}_ps`),
// included by w4a4_gemv_mx.cu inside its guard, after `w4a4_mma` and `W4A4_WARPS`.
//
// Owner: gb10 kernels.
// Invariants:
// - Launch: grid (#SMs, 1, 1), block 256, the extra argument sst (k128 chunks each warp stages,
//   ops::w4a4_proj::ps_stripe_chunks gives ceil((K / 128) / 8)) and 8 * sst * MB * 576 + RJ * 4096
//   bytes of dynamic shared memory (ops::w4a4_proj::ps_smem_bytes on the host).
// - Shapes as in w4a4_gemv_mx.cu. Only C[m, n] with m < M and n < N is written, and the
//   module's tile counters are zero again when a launch ends.
// - Warp w accumulates chunks c = w (mod 8) of its tile in increasing order with the MMA fragments
//   of w4a4_gemv_mx_impl, and the 8 partials are summed in warp order, so each entry is
//   bit-identical to the one-tile kernel of its row range; only where a fragment is read from
//   differs. The model-arch example w4a4_gemv_nt_oracle compares them.
//
// The one-tile kernels read the activations from L2 once per 16-row weight tile. Here one CTA
// per SM stays resident for the launch and pulls 16-row tiles from a global counter, so the
// work unit stays 16 rows, and each warp stages up to sst chunks of its own stripe of the
// activations (c = warp mod 8) in shared memory once per launch. Weight loads are
// `ld.global.cs` (evict-first) so that streaming the weights should not evict the scale sectors
// that 4 warps share.
//
// Two launches of one entry must not overlap in time: they would share the tile counter, which
// the last CTA of each launch resets, as they would share ops::w4a4_proj's activation scratch.






__device__ __forceinline__ void w4a4_ps_cp16(void* dst, const void* src, bool live) {
    const unsigned int d = (unsigned int)__cvta_generic_to_shared(dst);
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;" ::"r"(d), "l"(src), "r"(live ? 16 : 0));
}

__device__ __forceinline__ void w4a4_ps_cp8(void* dst, const void* src, bool live) {
    const unsigned int d = (unsigned int)__cvta_generic_to_shared(dst);
    asm volatile("cp.async.ca.shared.global [%0], [%1], 8, %2;" ::"r"(d), "l"(src), "r"(live ? 8 : 0));
}

// 2026-09-25: MB 8-token column blocks, KU chunks per warp per step, RJ column blocks per
// reduction pass (the reduction buffer is RJ * 4 KiB).
template <int MB, int KU, int RJ>
__device__ __forceinline__ void w4a4_gemv_mx_ps_impl(
    const unsigned char* __restrict__ Aq,
    const unsigned char* __restrict__ As,
    const float* __restrict__ Ag,
    const unsigned char* __restrict__ Bq,
    const unsigned char* __restrict__ Bs,
    const float scale2,
    __nv_bfloat16* __restrict__ C,
    unsigned int M, unsigned int N, unsigned int K, unsigned int sst,
    unsigned int* __restrict__ ctr)
{
    extern __shared__ uint4 w4a4_ps_smem[];
    __shared__ unsigned int s_tile[2];
    const unsigned int warp = threadIdx.x >> 5;
    const unsigned int lane = threadIdx.x & 31u;
    const unsigned int g = lane >> 2;
    const unsigned int t = lane & 3u;
    const bool odd = (t & 1u) != 0u;
    const unsigned int half_K = K >> 1;
    const unsigned int groups = K >> 4;
    const unsigned int num_c = K >> 7;
    const unsigned int ntiles = (N + 15u) >> 4;

    // 2026-09-25: All warps' fragments ([warp][sst][MB][32 lanes] uint4), then all warps' scales
    // ([warp][sst][MB][8 g] uint2, read by the 4 lanes t of a g), then the reduction buffer.
    uint4* sq = w4a4_ps_smem + warp * sst * MB * 32u;
    uint2* ss = (uint2*)(w4a4_ps_smem + 8u * sst * MB * 32u) + warp * sst * MB * 8u;
    float* red = (float*)((uint2*)(w4a4_ps_smem + 8u * sst * MB * 32u) + 8u * sst * MB * 8u);

    bool tl[MB];
    const unsigned char* aq[MB];
    const unsigned char* as[MB];
    #pragma unroll
    for (int j = 0; j < MB; j++) {
        const unsigned int tok = (unsigned int)j * 8u + g;
        tl[j] = tok < M;
        aq[j] = Aq + (unsigned long long)tok * half_K + t * 16u;
        as[j] = As + (unsigned long long)tok * groups;
    }

    if (threadIdx.x == 0) s_tile[0] = atomicAdd(ctr, 1u);
    // 2026-09-25: Stage this warp's stripe asynchronously; it is waited on after the first weight
    // loads are issued. Tokens at or past M are zero-filled.
    for (unsigned int i = 0, c = warp; c < num_c && i < sst; i++, c += W4A4_WARPS) {
        #pragma unroll
        for (int j = 0; j < MB; j++) {
            w4a4_ps_cp16(sq + (i * MB + j) * 32u + lane, tl[j] ? (const void*)(aq[j] + c * 64u) : (const void*)Aq, tl[j]);
            if (t == 0u) {
                w4a4_ps_cp8(ss + (i * MB + j) * 8u + g, tl[j] ? (const void*)(as[j] + c * 8u) : (const void*)As, tl[j]);
            }
        }
    }
    asm volatile("cp.async.commit_group;");
    bool staged = sst == 0u;
    __syncthreads();

    unsigned int buf = 0;
    for (;;) {
        const unsigned int tile = s_tile[buf];
        if (tile >= ntiles) break;
        if (threadIdx.x == 0) s_tile[buf ^ 1u] = atomicAdd(ctr, 1u);
        const unsigned int n0 = tile * 16u;
        const unsigned int r0 = n0 + g, r1 = n0 + g + 8u, rs = n0 + g + (odd ? 8u : 0u);
        const bool l0 = r0 < N, l1 = r1 < N, ls = rs < N;
        const unsigned char* w0 = Bq + (unsigned long long)r0 * half_K + t * 16u;
        const unsigned char* w1 = Bq + (unsigned long long)r1 * half_K + t * 16u;
        const unsigned char* ws = Bs + (unsigned long long)rs * groups;

        float acc[MB][4];
        #pragma unroll
        for (int j = 0; j < MB; j++) {
            #pragma unroll
            for (int c = 0; c < 4; c++) acc[j][c] = 0.0f;
        }

        for (unsigned int c0 = warp, i0 = 0; c0 < num_c; c0 += W4A4_WARPS * KU, i0 += KU) {
            uint4 wl[KU], wh[KU], b[KU][MB];
            uint2 sw[KU], sb[KU][MB];
            #pragma unroll
            for (int u = 0; u < KU; u++) {
                const unsigned int c = c0 + (unsigned int)u * W4A4_WARPS;
                const bool live = c < num_c;
                const uint4 z4 = make_uint4(0u, 0u, 0u, 0u);
                const uint2 z2 = make_uint2(0u, 0u);
                wl[u] = (live && l0) ? __ldcs((const uint4*)(w0 + c * 64u)) : z4;
                wh[u] = (live && l1) ? __ldcs((const uint4*)(w1 + c * 64u)) : z4;
                sw[u] = (live && ls) ? *(const uint2*)(ws + c * 8u) : z2;
                if (i0 + (unsigned int)u >= sst) {
                    #pragma unroll
                    for (int j = 0; j < MB; j++) {
                        b[u][j] = (live && tl[j]) ? *(const uint4*)(aq[j] + c * 64u) : z4;
                        sb[u][j] = (live && tl[j]) ? *(const uint2*)(as[j] + c * 8u) : z2;
                    }
                }
            }
            if (!staged) {
                asm volatile("cp.async.wait_all;" ::: "memory");
                __syncwarp();
                staged = true;
            }
            #pragma unroll
            for (int u = 0; u < KU; u++) {
                if (c0 + (unsigned int)u * W4A4_WARPS >= num_c) break;
                if (i0 + (unsigned int)u < sst) {
                    #pragma unroll
                    for (int j = 0; j < MB; j++) {
                        b[u][j] = sq[((i0 + u) * MB + j) * 32u + lane];
                        sb[u][j] = ss[((i0 + u) * MB + j) * 8u + g];
                    }
                }
                // 2026-09-25: Fragment assembly and scale permutation as in w4a4_gemv_mx_impl.
                const uint32_t x0l = __shfl_xor_sync(0xFFFFFFFFu, odd ? wl[u].x : wl[u].y, 1);
                const uint32_t x0h = __shfl_xor_sync(0xFFFFFFFFu, odd ? wh[u].x : wh[u].y, 1);
                const uint32_t x1l = __shfl_xor_sync(0xFFFFFFFFu, odd ? wl[u].z : wl[u].w, 1);
                const uint32_t x1h = __shfl_xor_sync(0xFFFFFFFFu, odd ? wh[u].z : wh[u].w, 1);
                const uint32_t a00 = odd ? x0l : wl[u].x, a02 = odd ? wl[u].y : x0l;
                const uint32_t a01 = odd ? x0h : wh[u].x, a03 = odd ? wh[u].y : x0h;
                const uint32_t a10 = odd ? x1l : wl[u].z, a12 = odd ? wl[u].w : x1l;
                const uint32_t a11 = odd ? x1h : wh[u].z, a13 = odd ? wh[u].w : x1h;
                const uint32_t s0 = __byte_perm(sw[u].x, sw[u].y, 0x6240);
                const uint32_t s1 = __byte_perm(sw[u].x, sw[u].y, 0x7351);
                #pragma unroll
                for (int j = 0; j < MB; j++) {
                    w4a4_mma(acc[j], a00, a01, a02, a03, b[u][j].x, b[u][j].y, s0, sb[u][j].x);
                    w4a4_mma(acc[j], a10, a11, a12, a13, b[u][j].z, b[u][j].w, s1, sb[u][j].y);
                }
            }
        }

        // 2026-09-25: Warp-order reduction, RJ column blocks per pass.
        #pragma unroll
        for (int p = 0; p < MB / RJ; p++) {
            #pragma unroll
            for (int jj = 0; jj < RJ; jj++) {
                #pragma unroll
                for (int c = 0; c < 4; c++) red[((warp * RJ + jj) * 4 + c) * 32 + lane] = acc[p * RJ + jj][c];
            }
            __syncthreads();
            for (unsigned int jj = warp; jj < (unsigned int)RJ; jj += W4A4_WARPS) {
                const unsigned int j = (unsigned int)p * RJ + jj;
                float r[4];
                #pragma unroll
                for (int c = 0; c < 4; c++) {
                    float v = red[(jj * 4 + c) * 32 + lane];
                    #pragma unroll
                    for (int ww = 1; ww < W4A4_WARPS; ww++) v += red[((ww * RJ + jj) * 4 + c) * 32 + lane];
                    r[c] = v;
                }
                #pragma unroll
                for (int c = 0; c < 4; c++) {
                    const unsigned int n = (c < 2) ? r0 : r1;
                    const unsigned int tok = j * 8u + t * 2u + (unsigned int)(c & 1);
                    if (n < N && tok < M) {
                        C[(unsigned long long)tok * N + n] = __float2bfloat16_rn(r[c] * (Ag[tok] * scale2));
                    }
                }
            }
            __syncthreads();
        }
        buf ^= 1u;
    }
    // 2026-09-25: Every CTA fetched its last tile index before counting itself out, so the last
    // one out may reset both counters for the next launch.
    if (threadIdx.x == 0) {
        __threadfence();
        if (atomicAdd(ctr + 1, 1u) == gridDim.x - 1u) {
            atomicExch(ctr, 0u);
            atomicExch(ctr + 1, 0u);
        }
    }
}
