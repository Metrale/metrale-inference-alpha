// SPDX-License-Identifier: MIT OR Apache-2.0

// 2026-09-25: Mamba-2 SSD chunked prefill scan, the chunked formulation of "Transformers
// are SSMs" (Dao & Gu, 2024). nemotron_mamba2 prefill uses it when the kernels resolve, the
// shape checks and ssd_scan_fits pass and the ssd lever is on; otherwise it runs the
// sequential mamba2_ssm_prefill* kernels.
//
// Owner: gb10 kernels.
// Invariants:
// - SSD_L = 64 (chunk length) and SSD_PT = 64 (head_dim rows per scan block) must equal the
//   constants of the same names in crates/model-layers/src/layers/ops/ssm_ssd.rs.
// - mamba2_ssd_cumsum: grid (nchunks, heads, batch), block SSD_L, one thread per token.
//   mamba2_ssd_bmm: grid (nchunks, groups, batch), block 128. mamba2_ssd_scan: grid
//   (heads, head_dim / SSD_PT, batch), block 512, shared memory ssd_scan_smem(N).
// - head_dim is a multiple of SSD_PT and N a multiple of 32; nemotron_mamba2 checks both.
// - A token t >= seq_len has dt = 0 and reads x, B and C as zero, so it adds nothing to
//   the state or to other rows; its output row is not written.
//
// Per head h and chunk c, with t and s token indices inside the chunk:
//   dt_t   = clamp(softplus(dt_raw_t + dt_bias[h]), dt_min, dt_max)
//   a_t    = -exp(A_log[h]) * dt_t;   cs_t = inclusive cumsum of a over the chunk
//   y_t[p] = exp(cs_t) sum_n C_t[n] h0[p][n]
//          + sum_{s<=t} (C_t . B_s) exp(cs_t - cs_s) dt_s x_s[p]  +  D[h] x_t[p]
//   h0[p][n] <- exp(cs_last) h0[p][n] + sum_s exp(cs_last - cs_s) dt_s x_s[p] B_s[n]
// Every exponent is clamped to <= 0 before exp, so each factor lies in (0, 1]. Only h0
// crosses chunks: the scan keeps it in shared memory (sH) for the whole prefill, loading
// it from h_state first and storing it back last.



#include <cuda_bf16.h>

#define SSD_L 64




#define SSD_PT 64





extern "C" __global__ void mamba2_ssd_cumsum(
    const __nv_bfloat16* __restrict__ dt_raw,
    const float* __restrict__ A_log,
    const float* __restrict__ dt_bias,
    float* __restrict__ dt_out,
    float* __restrict__ dA_cs,
    unsigned int seq_len,
    unsigned int num_heads,
    unsigned int nchunks,
    unsigned int dt_stride,
    float dt_min,
    float dt_max
) {
    const unsigned int c = blockIdx.x;
    const unsigned int h = blockIdx.y;
    const unsigned int b = blockIdx.z;
    const unsigned int t = threadIdx.x;
    const unsigned int gt = c * SSD_L + t;

    float dtv = 0.0f;
    if (gt < seq_len) {
        dtv = (float)dt_raw[(unsigned long long)gt * dt_stride + b * num_heads + h]
            + dt_bias[h];
        dtv = (dtv > 20.0f) ? dtv : logf(1.0f + expf(dtv));
        dtv = fminf(fmaxf(dtv, dt_min), dt_max);
    }
    const float a = -expf(A_log[h]) * dtv;


    __shared__ float warp_tail[2];
    const unsigned int lane = t & 31u;
    const unsigned int warp = t >> 5;
    float v = a;
    #pragma unroll
    for (int off = 1; off < 32; off <<= 1) {
        float n = __shfl_up_sync(0xFFFFFFFFu, v, off);
        if (lane >= (unsigned int)off) v += n;
    }
    if (lane == 31u) warp_tail[warp] = v;
    __syncthreads();
    if (warp == 1u) v += warp_tail[0];

    const unsigned long long base =
        (((unsigned long long)b * num_heads + h) * nchunks + c) * SSD_L;
    dt_out[base + t] = dtv;
    dA_cs[base + t]  = v;
}


// 2026-09-25: CB[c][g][t][s] = sum_n C[c*L+t][g][n] * B[c*L+s][g][n], in FP32, with no
// dt or decay: those depend on the head and are applied by mamba2_ssd_scan.


extern "C" __global__ void mamba2_ssd_bmm(
    const __nv_bfloat16* __restrict__ B_in,
    const __nv_bfloat16* __restrict__ C_in,
    float* __restrict__ CB,
    unsigned int seq_len,
    unsigned int nchunks,
    unsigned int n_groups,
    unsigned int state_size,
    unsigned int bc_stride
) {
    const unsigned int c = blockIdx.x;
    const unsigned int g = blockIdx.y;
    const unsigned int b = blockIdx.z;
    const unsigned int N = state_size;

    extern __shared__ __nv_bfloat16 smem_bmm[];
    __nv_bfloat16* sC = smem_bmm;
    __nv_bfloat16* sB = sC + (unsigned long long)SSD_L * N;

    for (unsigned int i = threadIdx.x; i < SSD_L * N; i += blockDim.x) {
        const unsigned int t = i / N, n = i - t * N;
        const unsigned int gt = c * SSD_L + t;
        const unsigned long long off =
            (unsigned long long)gt * bc_stride + b * n_groups * N + g * N + n;
        const bool ok = gt < seq_len;
        sC[i] = ok ? C_in[off] : __float2bfloat16(0.0f);
        sB[i] = ok ? B_in[off] : __float2bfloat16(0.0f);
    }
    __syncthreads();

    // 2026-09-25: m16n8k16 with M = t, N = s and K = n: A = sC[t][n], and sB[s][n] is already
    // the N-by-K col operand.
    const unsigned int warp = threadIdx.x >> 5;
    const unsigned int lane = threadIdx.x & 31u;
    const unsigned int gid = lane >> 2;
    const unsigned int tid = lane & 3u;
    const unsigned int wm = warp * 16u;

    float acc[8][4];
    #pragma unroll
    for (int i = 0; i < 8; i++) { acc[i][0]=0.f; acc[i][1]=0.f; acc[i][2]=0.f; acc[i][3]=0.f; }

    const unsigned short* A16 = (const unsigned short*)sC;
    const unsigned short* B16 = (const unsigned short*)sB;

    for (unsigned int k = 0; k < N; k += 16u) {
        const unsigned int r0 = wm + gid, r1 = wm + gid + 8u;
        const unsigned int c0 = k + tid * 2u,  c1 = k + tid * 2u + 8u;
        if (c0 >= N) break;

        unsigned int a0 = ((unsigned int)A16[r0 * N + c0 + 1] << 16) | A16[r0 * N + c0];
        unsigned int a1 = ((unsigned int)A16[r1 * N + c0 + 1] << 16) | A16[r1 * N + c0];
        unsigned int a2 = ((unsigned int)A16[r0 * N + c1 + 1] << 16) | A16[r0 * N + c1];
        unsigned int a3 = ((unsigned int)A16[r1 * N + c1 + 1] << 16) | A16[r1 * N + c1];

        #pragma unroll
        for (int nt = 0; nt < 8; nt++) {
            const unsigned int s = nt * 8u + gid;
            unsigned int b0 = ((unsigned int)B16[s * N + c0 + 1] << 16) | B16[s * N + c0];
            unsigned int b1 = ((unsigned int)B16[s * N + c1 + 1] << 16) | B16[s * N + c1];
            asm volatile(
                "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
                "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%10,%11,%12,%13};"
                : "=f"(acc[nt][0]), "=f"(acc[nt][1]), "=f"(acc[nt][2]), "=f"(acc[nt][3])
                : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1),
                  "f"(acc[nt][0]), "f"(acc[nt][1]), "f"(acc[nt][2]), "f"(acc[nt][3]));
        }
    }

    float* out = CB + ((((unsigned long long)b * nchunks + c) * n_groups + g)
                       * SSD_L * SSD_L);
    #pragma unroll
    for (int nt = 0; nt < 8; nt++) {
        const unsigned int s0 = nt * 8u + tid * 2u;
        const unsigned int t0 = wm + gid, t1 = wm + gid + 8u;
        out[t0 * SSD_L + s0]      = acc[nt][0];
        out[t0 * SSD_L + s0 + 1]  = acc[nt][1];
        out[t1 * SSD_L + s0]      = acc[nt][2];
        out[t1 * SSD_L + s0 + 1]  = acc[nt][3];
    }
}

__device__ __forceinline__ void ssd_cpa16(void* d, const void* src, bool pred) {
    unsigned x = __cvta_generic_to_shared(d);
    int sz = pred ? 16 : 0;
    asm volatile("cp.async.ca.shared.global [%0],[%1],16,%2;\n" ::"r"(x), "l"(src), "r"(sz));
}
__device__ __forceinline__ void ssd_commit() { asm volatile("cp.async.commit_group;"); }
__device__ __forceinline__ void ssd_wait0() { asm volatile("cp.async.wait_group 0;"); }


// 2026-09-25: The streaming tiles (B, C, x, dA_cs and dt) are double-buffered: cp.async
// loads chunk c + 1 into the other slot while chunk c computes. ssd_cpa16 moves 16 bytes,
// so every x, B and C row start must be 16-byte aligned; a source size of 0 zero-fills the
// destination, which is how tokens past seq_len load as zeros.



extern "C" __global__ void mamba2_ssd_scan(
    float* __restrict__ h_state,
    const __nv_bfloat16* __restrict__ x,
    const __nv_bfloat16* __restrict__ B_in,
    const __nv_bfloat16* __restrict__ C_in,
    const float* __restrict__ D_param,
    const float* __restrict__ dt_f32,
    const float* __restrict__ dA_cs,
    const float* __restrict__ CB,
    __nv_bfloat16* __restrict__ output,
    unsigned int seq_len,
    unsigned int num_heads,
    unsigned int head_dim,
    unsigned int state_size,
    unsigned int n_groups,
    unsigned int nchunks,
    unsigned int x_stride,
    unsigned int bc_stride,
    unsigned int y_stride
) {
    const unsigned int h  = blockIdx.x;
    const unsigned int pt = blockIdx.y;
    const unsigned int b  = blockIdx.z;
    const unsigned int P  = head_dim;
    const unsigned int N  = state_size;
    const unsigned int g  = h / (num_heads / n_groups);
    const unsigned int p0 = pt * SSD_PT;

    const float D_val = D_param[h];









    extern __shared__ char smem_raw[];
    float*         sH   = (float*)smem_raw;
    __nv_bfloat16* sBd  = (__nv_bfloat16*)(sH + SSD_PT * (N + 1));
    __nv_bfloat16* sCMd = sBd + 2 * SSD_L * N;
    __nv_bfloat16* sXd  = sCMd + 2 * SSD_L * N;
    float*         sdAd = (float*)(sXd + 2 * SSD_L * SSD_PT);
    float*         sdtd = sdAd + 2 * SSD_L;

    const unsigned int warp = threadIdx.x >> 5;
    const unsigned int lane = threadIdx.x & 31u;
    const unsigned int gid  = lane >> 2;
    const unsigned int tid  = lane & 3u;

    float* Hg = h_state + ((unsigned long long)(b * num_heads + h) * P + p0) * N;


    for (unsigned int i = threadIdx.x; i < SSD_PT * N; i += blockDim.x) {
        const unsigned int p = i / N, n = i - p * N;
        sH[p * (N + 1) + n] = Hg[p * N + n];
    }
    __syncthreads();




    #define SSD_LOAD(sl, c) do { \
        const unsigned long long dbase_ = \
            (((unsigned long long)b * num_heads + h) * nchunks + (c)) * SSD_L; \
        for (unsigned int i = threadIdx.x; i < SSD_L / 4u; i += blockDim.x) { \
            ssd_cpa16(&sdAd[(sl) * SSD_L + i * 4u], dA_cs + dbase_ + i * 4u, true); \
            ssd_cpa16(&sdtd[(sl) * SSD_L + i * 4u], dt_f32 + dbase_ + i * 4u, true); \
        } \
        for (unsigned int i = threadIdx.x; i < SSD_L * SSD_PT / 8u; i += blockDim.x) { \
            const unsigned int t = i / (SSD_PT / 8u), pc = (i % (SSD_PT / 8u)) * 8u; \
            const unsigned int gt = (c) * SSD_L + t; \
            ssd_cpa16(&sXd[(sl) * SSD_L * SSD_PT + t * SSD_PT + pc], \
                x + (unsigned long long)gt * x_stride \
                  + (unsigned long long)(b * num_heads + h) * P + p0 + pc, \
                gt < seq_len); \
        } \
        for (unsigned int i = threadIdx.x; i < SSD_L * N / 8u; i += blockDim.x) { \
            const unsigned int t = i / (N / 8u), nc = (i % (N / 8u)) * 8u; \
            const unsigned int gt = (c) * SSD_L + t; \
            const unsigned long long off_ = \
                (unsigned long long)gt * bc_stride + b * n_groups * N + g * N + nc; \
            ssd_cpa16(&sBd[(sl) * SSD_L * N + t * N + nc],  B_in + off_, gt < seq_len); \
            ssd_cpa16(&sCMd[(sl) * SSD_L * N + t * N + nc], C_in + off_, gt < seq_len); \
        } \
    } while (0)

    SSD_LOAD(0, 0);
    ssd_commit();
    ssd_wait0();
    __syncthreads();

    unsigned int slot = 0;
    for (unsigned int c = 0; c < nchunks; c++) {

        if (c + 1 < nchunks) {
            SSD_LOAD(slot ^ 1u, c + 1);
            ssd_commit();
        }
        __nv_bfloat16* sB  = sBd  + slot * SSD_L * N;
        __nv_bfloat16* sCM = sCMd + slot * SSD_L * N;
        __nv_bfloat16* sX  = sXd  + slot * SSD_L * SSD_PT;
        float*         sdA = sdAd + slot * SSD_L;
        float*         sdt = sdtd + slot * SSD_L;

        const float cs_last = sdA[SSD_L - 1];


        for (unsigned int i = threadIdx.x; i < SSD_L * N; i += blockDim.x) {
            const unsigned int t = i / N;
            sCM[i] = __float2bfloat16((float)sCM[i] * __expf(fminf(sdA[t], 0.0f)));
        }
        __syncthreads();

        // 2026-09-25: (a) Y_off = Cd (L x N) @ h0^T (N x PT). 16 warps: warp >> 2 picks one of
        // four 16-row m-blocks and warp & 3 two adjacent 8-column n-tiles, which share the A
        // fragments.






        const unsigned int wm  = (warp >> 2) * 16u;
        const unsigned int wnb = (warp & 3u) * 2u;
        float acc[2][4];
        #pragma unroll
        for (int q = 0; q < 2; q++) { acc[q][0]=0.f; acc[q][1]=0.f; acc[q][2]=0.f; acc[q][3]=0.f; }
        {
            const unsigned short* A16 = (const unsigned short*)sCM;


            for (unsigned int k = 0; k < N; k += 16u) {
                const unsigned int r0 = wm + gid, r1 = wm + gid + 8u;
                const unsigned int c0 = k + tid * 2u, c1 = k + tid * 2u + 8u;
                if (c0 >= N) break;
                unsigned int a0 = ((unsigned int)A16[r0*N + c0+1] << 16) | A16[r0*N + c0];
                unsigned int a1 = ((unsigned int)A16[r1*N + c0+1] << 16) | A16[r1*N + c0];
                unsigned int a2 = ((unsigned int)A16[r0*N + c1+1] << 16) | A16[r0*N + c1];
                unsigned int a3 = ((unsigned int)A16[r1*N + c1+1] << 16) | A16[r1*N + c1];
                #pragma unroll
                for (unsigned int q = 0; q < 2u; q++) {
                    const unsigned int p = (wnb + q) * 8u + gid;
                    const float* hrow = sH + p * (N + 1);
                    __nv_bfloat162 h0p = __floats2bfloat162_rn(hrow[c0], hrow[c0 + 1]);
                    __nv_bfloat162 h1p = __floats2bfloat162_rn(hrow[c1], hrow[c1 + 1]);
                    unsigned int b0 = *(const unsigned int*)&h0p;
                    unsigned int b1 = *(const unsigned int*)&h1p;
                    asm volatile(
                        "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
                        "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%10,%11,%12,%13};"
                        : "=f"(acc[q][0]), "=f"(acc[q][1]), "=f"(acc[q][2]), "=f"(acc[q][3])
                        : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1),
                          "f"(acc[q][0]), "f"(acc[q][1]), "f"(acc[q][2]), "f"(acc[q][3]));
                }
            }
        }
        __syncthreads(); // 2026-09-25: sCM is reused below for M.

        // 2026-09-25: M[t][s] = CB[t][s] * exp(min(cs_t - cs_s, 0)) * dt_s for s <= t, else 0.
        {
            const float* cb = CB + ((((unsigned long long)b * nchunks + c) * n_groups + g)
                                    * SSD_L * SSD_L);
            for (unsigned int i = threadIdx.x; i < SSD_L * SSD_L; i += blockDim.x) {
                const unsigned int t = i / SSD_L, s = i - t * SSD_L;
                float v = 0.0f;
                if (s <= t) {
                    v = cb[i] * __expf(fminf(sdA[t] - sdA[s], 0.0f)) * sdt[s];
                }
                sCM[i] = __float2bfloat16(v);
            }
        }
        __syncthreads();

        // 2026-09-25: (b) Y_diag += M (L x L) @ X (L x PT).
        {
            const unsigned short* A16 = (const unsigned short*)sCM;
            const unsigned short* B16 = (const unsigned short*)sX;
            for (unsigned int k = 0; k < SSD_L; k += 16u) {
                const unsigned int r0 = wm + gid, r1 = wm + gid + 8u;
                const unsigned int c0 = k + tid * 2u, c1 = k + tid * 2u + 8u;
                unsigned int a0 = ((unsigned int)A16[r0*SSD_L + c0+1] << 16) | A16[r0*SSD_L + c0];
                unsigned int a1 = ((unsigned int)A16[r1*SSD_L + c0+1] << 16) | A16[r1*SSD_L + c0];
                unsigned int a2 = ((unsigned int)A16[r0*SSD_L + c1+1] << 16) | A16[r0*SSD_L + c1];
                unsigned int a3 = ((unsigned int)A16[r1*SSD_L + c1+1] << 16) | A16[r1*SSD_L + c1];
                #pragma unroll
                for (unsigned int q = 0; q < 2u; q++) {
                    const unsigned int p = (wnb + q) * 8u + gid;

                    unsigned int b0 = ((unsigned int)B16[(c0+1)*SSD_PT + p] << 16)
                                    | B16[c0*SSD_PT + p];
                    unsigned int b1 = ((unsigned int)B16[(c1+1)*SSD_PT + p] << 16)
                                    | B16[c1*SSD_PT + p];
                    asm volatile(
                        "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
                        "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%10,%11,%12,%13};"
                        : "=f"(acc[q][0]), "=f"(acc[q][1]), "=f"(acc[q][2]), "=f"(acc[q][3])
                        : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1),
                          "f"(acc[q][0]), "f"(acc[q][1]), "f"(acc[q][2]), "f"(acc[q][3]));
                }
            }
        }

        // 2026-09-25: (c) y = Y_off + Y_diag + D * x. The accumulator's row index is gid (and
        // gid + 8) and its column index tid * 2 (and + 1); the B operand instead takes its N
        // index from gid.


        #pragma unroll
        for (unsigned int q = 0; q < 2u; q++) {
            const unsigned int pa = (wnb + q) * 8u + tid * 2u;
            const unsigned int ta = wm + gid;
            const unsigned int tb = wm + gid + 8u;
            const unsigned int ts[4] = { ta, ta, tb, tb };
            const unsigned int ps[4] = { pa, pa + 1u, pa, pa + 1u };
            #pragma unroll
            for (int r = 0; r < 4; r++) {
                const unsigned int t  = ts[r];
                const unsigned int p  = ps[r];
                const unsigned int gt = c * SSD_L + t;
                if (gt >= seq_len || p >= SSD_PT) continue;
                const float xv = (float)sX[t * SSD_PT + p];
                output[(unsigned long long)gt * y_stride
                       + (unsigned long long)(b * num_heads + h) * P + p0 + p] =
                    __float2bfloat16(acc[q][r] + D_val * xv);
            }
        }
        __syncthreads();

        // 2026-09-25: (d) Bd[s][n] = B[s][n] * exp(min(cs_last - cs_s, 0)) * dt_s.
        for (unsigned int i = threadIdx.x; i < SSD_L * N; i += blockDim.x) {
            const unsigned int s = i / N;
            sB[i] = __float2bfloat16((float)sB[i]
                        * __expf(fminf(cs_last - sdA[s], 0.0f)) * sdt[s]);
        }
        __syncthreads();

        // 2026-09-25: (e) Hlocal[p][n] = sum_s x[s][p] * Bd[s][n]. Each warp takes m-tiles
        // warp >> 3 and (warp >> 3) + 2 and n-tiles warp & 7 and (warp & 7) + 8, so N / 8 must
        // be at most 16; ssd_scan_fits keeps N below that.
        const unsigned int NT = N / 8u;
        const unsigned int emt0 = (warp >> 3);
        const unsigned int enb  = warp & 7u;
        float hacc[2][2][4];
        #pragma unroll
        for (int u = 0; u < 2; u++)
            #pragma unroll
            for (int i = 0; i < 2; i++) {
                hacc[u][i][0]=0.f; hacc[u][i][1]=0.f; hacc[u][i][2]=0.f; hacc[u][i][3]=0.f;
            }
        {


            const unsigned short* X16 = (const unsigned short*)sX;
            const unsigned short* B16 = (const unsigned short*)sB;
            for (unsigned int k = 0; k < SSD_L; k += 16u) {
                const unsigned int c0 = k + tid * 2u, c1 = k + tid * 2u + 8u;
                #pragma unroll
                for (unsigned int u = 0; u < 2u; u++) {
                    const unsigned int emt = (emt0 + u * 2u) * 16u;
                    const unsigned int r0 = emt + gid, r1 = emt + gid + 8u;
                    unsigned int a0 = ((unsigned int)X16[(c0+1)*SSD_PT + r0] << 16) | X16[c0*SSD_PT + r0];
                    unsigned int a1 = ((unsigned int)X16[(c0+1)*SSD_PT + r1] << 16) | X16[c0*SSD_PT + r1];
                    unsigned int a2 = ((unsigned int)X16[(c1+1)*SSD_PT + r0] << 16) | X16[c1*SSD_PT + r0];
                    unsigned int a3 = ((unsigned int)X16[(c1+1)*SSD_PT + r1] << 16) | X16[c1*SSD_PT + r1];
                    #pragma unroll
                    for (unsigned int j = 0; j < 2u; j++) {
                        const unsigned int ntile = enb + j * 8u;
                        if (ntile >= NT) break;
                        const unsigned int n = ntile * 8u + gid;
                        unsigned int b0 = ((unsigned int)B16[(c0+1)*N + n] << 16) | B16[c0*N + n];
                        unsigned int b1 = ((unsigned int)B16[(c1+1)*N + n] << 16) | B16[c1*N + n];
                        asm volatile(
                            "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
                            "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%10,%11,%12,%13};"
                            : "=f"(hacc[u][j][0]), "=f"(hacc[u][j][1]),
                              "=f"(hacc[u][j][2]), "=f"(hacc[u][j][3])
                            : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1),
                              "f"(hacc[u][j][0]), "f"(hacc[u][j][1]),
                              "f"(hacc[u][j][2]), "f"(hacc[u][j][3]));
                    }
                }
            }
        }

        // 2026-09-25: (f) h0 <- exp(cs_last) * h0 + Hlocal.
        const float decay = __expf(fminf(cs_last, 0.0f));
        __syncthreads();
        for (unsigned int i = threadIdx.x; i < SSD_PT * N; i += blockDim.x) {
            const unsigned int p = i / N, n = i - p * N;
            sH[p * (N + 1) + n] *= decay;
        }
        __syncthreads();
        {
            #pragma unroll
            for (unsigned int u = 0; u < 2u; u++) {
                const unsigned int emt = (emt0 + u * 2u) * 16u;
                #pragma unroll
                for (unsigned int j = 0; j < 2u; j++) {
                    const unsigned int ntile = enb + j * 8u;
                    if (ntile >= NT) break;
                    const unsigned int n0 = ntile * 8u + tid * 2u;
                    const unsigned int pA = emt + gid, pB = emt + gid + 8u;
                    atomicAdd(&sH[pA * (N + 1) + n0],     hacc[u][j][0]);
                    atomicAdd(&sH[pA * (N + 1) + n0 + 1], hacc[u][j][1]);
                    atomicAdd(&sH[pB * (N + 1) + n0],     hacc[u][j][2]);
                    atomicAdd(&sH[pB * (N + 1) + n0 + 1], hacc[u][j][3]);
                }
            }
        }


        ssd_wait0();
        __syncthreads();
        slot ^= 1u;
    }
    #undef SSD_LOAD

    for (unsigned int i = threadIdx.x; i < SSD_PT * N; i += blockDim.x) {
        const unsigned int p = i / N, n = i - p * N;
        Hg[p * N + n] = sH[p * (N + 1) + n];
    }
}
