// SPDX-License-Identifier: MIT OR Apache-2.0

// 2026-09-25: Chunked GDN prefill (CHUNK = 64 tokens) as three kernels, launched in order by
// ops::gdn_prefill_fla:
//   1. gated_delta_rule_recompute_wu, one block per (chunk, head): solves (I + L) U = beta V and
//      (I + L) W = beta exp(gc) K by forward substitution, where gc is the in-chunk cumsum of
//      log(gate) and L[i][l] = beta_i exp(gc_i - gc_l) <k_l, k_i> for l < i.
//   2. a chunk_delta_h state spine, one block per head, serial over chunks: uc = U - W S_c and
//      S_c+1 = exp(gc_last) S_c + sum_i exp(gc_last - gc_i) k_i uc_i^T.
//   3. gated_delta_rule_chunk_fwd_o, one block per (chunk, head): the output from S_c and uc.
//
// Owner: gb10 kernels.
// Invariants:
// - K_DIM == V_DIM == 128 and CHUNK == 64 are compile-time, so the kernels require
//   k_dim == v_dim == 128.
// - The gate enters as log(max(gate, GATE_FLOOR)) and is not clamped above.
// - The recurrent state stays FP32; W, U, S_c and uc are stored as BF16.

#include <cuda_bf16.h>
#include <cuda.h>
#include <mma.h>


#define K_DIM 128
#define V_DIM 128
#define CHUNK 64
// 2026-09-25: Floor for the gate before the log: log(0) = -inf would make exp(gc_i - gc_l) NaN.
// log(1e-30) is about -69, a full decay, and the floor leaves every gate above 1e-30 unchanged.



#define GATE_FLOOR 1e-30f

// 2026-09-25: Per-sequence geometry. With is_varlen, sequence b spans tokens [cu_seqlens[b],
// cu_seqlens[b + 1]) and its chunks start after the earlier sequences' chunks; cu_chunks is not
// read. Otherwise sequence b starts at token b * seq_len and chunk b * num_chunks. Needs b,
// seq_len, num_chunks, cu_seqlens, cu_chunks and is_varlen in scope. With is_varlen the launch's
// num_chunks (grid x) is the maximum over sequences.

struct GdnGeom { unsigned int seqlen, nchunks, choff; unsigned long long tokoff; };
#define GDN_GEOM(g)                                                            \
    GdnGeom g;                                                                 \
    (void)cu_chunks;                                                          \
    if (is_varlen) {                                                           \
        unsigned int _s0 = (unsigned int)cu_seqlens[b];                       \
        g.seqlen  = (unsigned int)cu_seqlens[b + 1] - _s0;                    \
        g.tokoff  = (unsigned long long)_s0;                                  \
        unsigned int _co = 0;                                                  \
        for (unsigned int _i = 0; _i < b; _i++)                               \
            _co += ((unsigned int)(cu_seqlens[_i + 1] - cu_seqlens[_i])       \
                    + CHUNK - 1) / CHUNK;                                      \
        g.choff   = _co;                                                       \
        g.nchunks = (g.seqlen + CHUNK - 1) / CHUNK;                           \
    } else {                                                                  \
        g.seqlen  = seq_len;                                                   \
        g.tokoff  = (unsigned long long)b * seq_len;                          \
        g.choff   = b * num_chunks;                                            \
        g.nchunks = num_chunks;                                                \
    }













// 2026-09-25: TMA (cp.async.bulk.tensor) and mbarrier helpers. Per stage: mbar_init once, then
// per tile mbar_expect_tx(bytes), tma_load_2d(...), mbar_wait(parity). The barrier counts bytes:
// expect_tx declares the transaction size and the copy engine completes it. A tile that
// overhangs the tensor is zero-filled (TensorMap::tiled_2d_bf16). A `__grid_constant__ const
// CUtensorMap` is a 128-byte parameter passed by value (KernelLaunch::arg_tensormap).










__device__ __forceinline__ void mbar_init(uint64_t* bar, unsigned int count) {
    asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;\n" ::
                 "r"((unsigned int)__cvta_generic_to_shared(bar)), "r"(count));
}
__device__ __forceinline__ void mbar_expect_tx(uint64_t* bar, unsigned int bytes) {
    asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;\n" ::
                 "r"((unsigned int)__cvta_generic_to_shared(bar)), "r"(bytes) : "memory");
}


__device__ __forceinline__ void mbar_wait(uint64_t* bar, unsigned int parity) {
    asm volatile(
        "{\n"
        ".reg .pred P;\n"
        "WAIT_%=:\n"
        "mbarrier.try_wait.parity.shared::cta.b64 P, [%0], %1;\n"
        "@P bra DONE_%=;\n"
        "bra WAIT_%=;\n"
        "DONE_%=:\n"
        "}\n" ::
        "r"((unsigned int)__cvta_generic_to_shared(bar)), "r"(parity) : "memory");
}



__device__ __forceinline__ void tma_fence() {
    asm volatile("fence.proxy.async.shared::cta;\n" ::: "memory");
}
// 2026-09-25: One 2-D tile at (c0, c1): c0 is the column (fastest-varying), c1 the row, the
// order TensorMap::tiled_2d_bf16 encodes.

__device__ __forceinline__ void tma_load_2d(
    const CUtensorMap* desc, uint64_t* bar, void* dst_smem, int c0, int c1
) {
    asm volatile(
        "cp.async.bulk.tensor.2d.shared::cluster.global.tile.mbarrier::complete_tx::bytes"
        " [%0], [%1, {%3, %4}], [%2];\n" ::
        "r"((unsigned int)__cvta_generic_to_shared(dst_smem)),
        "l"((const void*)desc),
        "r"((unsigned int)__cvta_generic_to_shared(bar)),
        "r"(c0), "r"(c1) : "memory");
}

__device__ __forceinline__ void cp_async16(void* dst_smem, const void* src_gmem) {
    unsigned int s = (unsigned int)__cvta_generic_to_shared(dst_smem);
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" ::"r"(s), "l"(src_gmem));
}
__device__ __forceinline__ void cp_commit() { asm volatile("cp.async.commit_group;\n" ::); }
template <int N>
__device__ __forceinline__ void cp_wait() { asm volatile("cp.async.wait_group %0;\n" ::"n"(N)); }

// 2026-09-25: C[m][n] = sum_k A[m][k] B[n][k] (m < 64, k < K_DIM, n < NTC * 8) on mma.sync; A, B
// are BF16 in shared memory, row stride K_DIM; C row stride NSTRIDE. Warp w does rows 16w..16w+15.
template <int NTC, int NSTRIDE, bool OutBf16>
__device__ __forceinline__ void mma_gram(
    const __nv_bfloat16* __restrict__ A, const __nv_bfloat16* __restrict__ B, void* __restrict__ C
) {
    const unsigned warp = threadIdx.x >> 5;
    const unsigned lane = threadIdx.x & 31;
    const unsigned grp = lane >> 2;
    const unsigned q = lane & 3;
    const unsigned warp_m = warp * 16;
    const unsigned short* sA = (const unsigned short*)A;
    const unsigned short* sB = (const unsigned short*)B;
    float acc[NTC][4];
    #pragma unroll
    for (int nt = 0; nt < NTC; nt++) { acc[nt][0] = acc[nt][1] = acc[nt][2] = acc[nt][3] = 0.0f; }
    #pragma unroll
    for (unsigned ks = 0; ks < K_DIM; ks += 16) {
        unsigned fr0 = warp_m + grp, fr1 = fr0 + 8;
        unsigned fc0 = ks + q * 2, fc1 = fc0 + 8;
        unsigned a0 = *(const unsigned*)&sA[fr0 * K_DIM + fc0];
        unsigned a1 = *(const unsigned*)&sA[fr1 * K_DIM + fc0];
        unsigned a2 = *(const unsigned*)&sA[fr0 * K_DIM + fc1];
        unsigned a3 = *(const unsigned*)&sA[fr1 * K_DIM + fc1];
        #pragma unroll
        for (int nt = 0; nt < NTC; nt++) {
            unsigned nc = nt * 8 + grp;
            unsigned k0 = ks + q * 2, k1 = k0 + 8;
            unsigned b0 = ((unsigned)sB[nc * K_DIM + k0 + 1] << 16) | (unsigned)sB[nc * K_DIM + k0];
            unsigned b1 = ((unsigned)sB[nc * K_DIM + k1 + 1] << 16) | (unsigned)sB[nc * K_DIM + k1];
            asm volatile(
                "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%10,%11,%12,%13};"
                : "=f"(acc[nt][0]), "=f"(acc[nt][1]), "=f"(acc[nt][2]), "=f"(acc[nt][3])
                : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1),
                  "f"(acc[nt][0]), "f"(acc[nt][1]), "f"(acc[nt][2]), "f"(acc[nt][3]));
        }
    }
    #pragma unroll
    for (int nt = 0; nt < NTC; nt++) {
        unsigned n0 = nt * 8 + q * 2, n1 = n0 + 1;
        unsigned m0 = warp_m + grp, m1 = m0 + 8;
        if (OutBf16) {
            __nv_bfloat16* Cb = (__nv_bfloat16*)C;
            Cb[m0 * NSTRIDE + n0] = __float2bfloat16(acc[nt][0]);
            Cb[m0 * NSTRIDE + n1] = __float2bfloat16(acc[nt][1]);
            Cb[m1 * NSTRIDE + n0] = __float2bfloat16(acc[nt][2]);
            Cb[m1 * NSTRIDE + n1] = __float2bfloat16(acc[nt][3]);
        } else {
            float* Cf = (float*)C;
            Cf[m0 * NSTRIDE + n0] = acc[nt][0];
            Cf[m0 * NSTRIDE + n1] = acc[nt][1];
            Cf[m1 * NSTRIDE + n0] = acc[nt][2];
            Cf[m1 * NSTRIDE + n1] = acc[nt][3];
        }
    }
}

// 2026-09-25: Kernel 1, gated_delta_rule_recompute_wu: grid (num_chunks, num_v_heads,
// batch_size), 256 threads, one block per (chunk, head). Block row (choff + c) * num_v_heads + vh
// of U_out[CHUNK][V_DIM] = T (beta V) and W_out[CHUNK][K_DIM] = T (beta exp(gc) K), both BF16
// with T = (I + L)^-1, and gc_out (FP32) for the chunk's tokens. Dynamic shared memory:
// CHUNK * K_DIM BF16 plus CHUNK * CHUNK + CHUNK floats (33,024 B).


#define RL_BLK 16

// 2026-09-25: RL_BLK is the block width of the right-looking forward substitution below: each
// block of RL_BLK solved rows is held in registers (xb) and subtracted from every later row once.
// Measured 2026-08-22 with gdn_recompute_wu_gateb: 1.95x the left-looking form at nt=64. The same
// day, a shared-memory solve accumulator (64 KB per block) measured 0.92x at nt=64, so acc[] is
// not moved to shared memory.


















































// 2026-09-25: 256 threads: mma_gram runs on warps 0-3, then the U solve runs on threads 0-127
// while the W solve runs on threads 128-255.
extern "C" __global__ void __launch_bounds__(256, 1)
gated_delta_rule_recompute_wu(
    const __nv_bfloat16* __restrict__ key,
    const __nv_bfloat16* __restrict__ value,
    const float* __restrict__ gate,
    const float* __restrict__ beta,
    __nv_bfloat16* __restrict__ W_out,
    __nv_bfloat16* __restrict__ U_out,
    float* __restrict__ gc_out,
    unsigned int batch_size,
    unsigned int seq_len,
    unsigned int num_chunks,
    unsigned int num_k_heads,
    unsigned int num_v_heads,
    unsigned int k_dim,
    unsigned int v_dim,
    unsigned int qk_stride,
    unsigned int v_stride,
    unsigned int gb_stride,
    const int* __restrict__ cu_seqlens,
    const int* __restrict__ cu_chunks,
    unsigned int is_varlen
) {
    const unsigned int c = blockIdx.x;
    const unsigned int vh = blockIdx.y;
    const unsigned int b = blockIdx.z;
    if (vh >= num_v_heads || b >= batch_size) return;
    GDN_GEOM(g);
    if (c >= g.nchunks) return;

    const unsigned int tid = threadIdx.x;
    const unsigned int head_repeat = num_v_heads / num_k_heads;
    const unsigned int kh = vh / head_repeat;
    const unsigned int cs = c * CHUNK;
    const unsigned int ce = (g.seqlen - cs) < CHUNK ? (g.seqlen - cs) : CHUNK;
    const unsigned long long base = ((unsigned long long)(g.choff + c) * num_v_heads + vh);


    key   += g.tokoff * qk_stride;
    value += g.tokoff * v_stride;
    gate  += g.tokoff * gb_stride;
    beta  += g.tokoff * gb_stride;

    extern __shared__ char smem_raw[];
    __nv_bfloat16* sk = (__nv_bfloat16*)smem_raw;
    float* kk = (float*)(sk + CHUNK * K_DIM);
    // 2026-09-25: L aliases kk. The build below writes L only at (i, l) with l < i and reads kk only
    // at (l, i) with l < i, so no element is both written and read, and both solves read L only
    // below the diagonal.

    float* L = kk;
    float* gc = L + CHUNK * CHUNK;

    for (unsigned int idx = tid; idx < CHUNK * k_dim; idx += blockDim.x) {
        unsigned int i = idx / k_dim, j = idx % k_dim;
        sk[i * K_DIM + j] = (i < ce)
            ? key[(unsigned long long)(cs + i) * qk_stride + kh * k_dim + j]
            : __float2bfloat16(0.0f);
    }
    // 2026-09-25: The logs are computed in parallel; the additions run on one thread in index order,
    // so the cumsum's rounding does not depend on the thread count. Measured 2026-08-23: a tree scan
    // here, which reassociates the sum, dropped BFCL below its floors; gc feeds exp(gc_i - gc_l)
    // across the whole of L.















    for (unsigned int idx = tid; idx < CHUNK; idx += blockDim.x) {
        gc[idx] = (idx < ce)
            ? logf(fmaxf(gate[(unsigned long long)(cs + idx) * gb_stride + vh], GATE_FLOOR))
            : 0.0f;
    }
    __syncthreads();
    if (tid == 0) {
        float acc = 0.0f;
        for (unsigned int i = 0; i < ce; i++) {
            acc += gc[i];
            gc[i] = acc;
        }
    }
    __syncthreads();


    for (unsigned int idx = tid; idx < CHUNK; idx += blockDim.x) {
        if (idx >= ce) {
            gc[idx] = 0.0f;
        } else {
            gc_out[base * CHUNK + idx] = gc[idx];
        }
    }
    __syncthreads();

    if (tid < 128) mma_gram<8, CHUNK, false>(sk, sk, kk);
    __syncthreads();


    for (unsigned int p = tid; p < CHUNK * CHUNK; p += blockDim.x) {
        unsigned int i = p / CHUNK, l = p % CHUNK;



        if (i < ce && l < i) {
            float bi = beta[(unsigned long long)(cs + i) * gb_stride + vh];
            L[i * CHUNK + l] = bi * expf(gc[i] - gc[l]) * kk[l * CHUNK + i];
        }
    }
    __syncthreads();

    // 2026-09-25: The two solves read only L, gc, beta, value and sk and write disjoint outputs, so
    // they run concurrently. Pass 1 solves U, one thread per value column.



    if (tid < v_dim) {
        float acc[CHUNK];
        for (unsigned int i = 0; i < ce; i++) {
            float bi = beta[(unsigned long long)(cs + i) * gb_stride + vh];
            acc[i] = bi * (float)value[(unsigned long long)(cs + i) * v_stride + vh * v_dim + tid];
        }
        for (unsigned int jb = 0; jb < ce; jb += RL_BLK) {
            float xb[RL_BLK];
            // 2026-09-25: r must stay a compile-time loop bound: with a runtime bound xb is dynamically
            // indexed and goes to local memory.
            #pragma unroll
            for (unsigned int r = 0; r < RL_BLK; r++) {
                if (jb + r >= ce) continue;
                float x = acc[jb + r];
                #pragma unroll
                for (unsigned int q = 0; q < RL_BLK; q++) {
                    if (q < r) x -= L[(jb + r) * CHUNK + jb + q] * xb[q];
                }
                xb[r] = x;
                acc[jb + r] = x;
                U_out[base * CHUNK * V_DIM + (jb + r) * v_dim + tid] = __float2bfloat16(x);
            }

            for (unsigned int i = jb + RL_BLK; i < ce; i++) {
                float a = acc[i];
                #pragma unroll
                for (unsigned int q = 0; q < RL_BLK; q++) {
                    if (jb + q < ce) a -= L[i * CHUNK + jb + q] * xb[q];
                }
                acc[i] = a;
            }
        }
    }

    const unsigned int wtid = tid - 128u;
    if (tid >= 128u && wtid < k_dim) {
        float acc[CHUNK];
        for (unsigned int i = 0; i < ce; i++) {
            float bi = beta[(unsigned long long)(cs + i) * gb_stride + vh];
            acc[i] = bi * expf(gc[i]) * (float)sk[i * K_DIM + wtid];
        }
        for (unsigned int jb = 0; jb < ce; jb += RL_BLK) {
            float xb[RL_BLK];


            #pragma unroll
            for (unsigned int r = 0; r < RL_BLK; r++) {
                if (jb + r >= ce) continue;
                float x = acc[jb + r];
                #pragma unroll
                for (unsigned int q = 0; q < RL_BLK; q++) {
                    if (q < r) x -= L[(jb + r) * CHUNK + jb + q] * xb[q];
                }
                xb[r] = x;
                acc[jb + r] = x;
                W_out[base * CHUNK * K_DIM + (jb + r) * k_dim + wtid] = __float2bfloat16(x);
            }
            for (unsigned int i = jb + RL_BLK; i < ce; i++) {
                float a = acc[i];
                #pragma unroll
                for (unsigned int q = 0; q < RL_BLK; q++) {
                    if (jb + q < ce) a -= L[i * CHUNK + jb + q] * xb[q];
                }
                acc[i] = a;
            }
        }
    }
}



// 2026-09-25: One staging slot: W, K and U of one chunk, in BF16 elements (24,576 = 48 KB).
#define CDH_BUFSZ (CHUNK * (2 * K_DIM + V_DIM))

// 2026-09-25: Starts the cp.async copies of chunk c's W, U and K into slot p, fills slot p's gc
// and decay tables (decb[0] = exp(gc_last), decb[1 + i] = exp(gc_last - gc_i)) with plain
// stores, and commits one cp.async group. K rows i >= ce are not loaded, since they lie past the
// sequence; callers read only rows below ce.
__device__ __forceinline__ void cdh_prefetch(
    __nv_bfloat16* buf, float* gcb, float* decb, unsigned int p,
    const __nv_bfloat16* __restrict__ W_in, const __nv_bfloat16* __restrict__ U_in,
    const __nv_bfloat16* __restrict__ key, const float* __restrict__ gate,
    const float* __restrict__ gc_in,
    unsigned int c, unsigned int b, unsigned int vh, unsigned int seq_len,
    unsigned int num_chunks, unsigned int num_v_heads, unsigned int k_dim,
    unsigned int kh, unsigned int qk_stride, unsigned int gb_stride,
    const int* __restrict__ cu_seqlens, const int* __restrict__ cu_chunks,
    unsigned int is_varlen
) {
    const unsigned int tid = threadIdx.x;
    GDN_GEOM(g);
    const unsigned int cs = c * CHUNK;
    const unsigned int ce = (g.seqlen - cs) < CHUNK ? (g.seqlen - cs) : CHUNK;
    const unsigned long long base = ((unsigned long long)(g.choff + c) * num_v_heads + vh);
    key += g.tokoff * qk_stride;
    __nv_bfloat16* Wp = buf + (unsigned long long)p * CDH_BUFSZ;
    __nv_bfloat16* Kp = Wp + CHUNK * K_DIM;
    __nv_bfloat16* Up = Kp + CHUNK * K_DIM;
    const unsigned int nthr = blockDim.x;
    const __nv_bfloat16* Wsrc = W_in + base * CHUNK * K_DIM;
    for (unsigned int e = tid * 8; e < CHUNK * K_DIM; e += nthr * 8) cp_async16(&Wp[e], &Wsrc[e]);
    const __nv_bfloat16* Usrc = U_in + base * CHUNK * V_DIM;
    for (unsigned int e = tid * 8; e < CHUNK * V_DIM; e += nthr * 8) cp_async16(&Up[e], &Usrc[e]);
    for (unsigned int j = tid; j < CHUNK * 16; j += nthr) {
        unsigned int i = j >> 4, c16 = (j & 15) * 8;
        if (i < ce)
            cp_async16(&Kp[i * K_DIM + c16],
                       key + (unsigned long long)(cs + i) * qk_stride + kh * k_dim + c16);
    }




    {
        const float dl = gc_in[base * CHUNK + ce - 1];
        if (tid == 0) decb[p * (CHUNK + 1)] = expf(dl);
        for (unsigned int i = tid; i < ce; i += nthr) {
            const float gv = gc_in[base * CHUNK + i];
            gcb[p * CHUNK + i] = gv;
            decb[p * (CHUNK + 1) + 1 + i] = expf(dl - gv);
        }
    }
    cp_commit();
}

// 2026-09-25: gated_delta_rule_chunk_delta_h: the scalar state spine, grid (num_v_heads,
// batch_size), 128 threads. Thread v keeps state column S[:, v] in registers across all chunks,
// and chunk c + 1's W, K and U are copied (cdh_prefetch) while chunk c computes. Per chunk:
// S_out = BF16(S_c), uc = U - W S_c to uc_out, then the state update. Uniform batches only.
// Dynamic shared memory: two CDH_BUFSZ slots plus the gc and decay tables (99,336 B). No code
// in the repository launches it.















extern "C" __global__ void __launch_bounds__(128, 1)
gated_delta_rule_chunk_delta_h(
    float* __restrict__ h_state,
    const __nv_bfloat16* __restrict__ W_in,
    const __nv_bfloat16* __restrict__ U_in,
    const __nv_bfloat16* __restrict__ key,
    const float* __restrict__ gate,
    const float* __restrict__ gc_in,
    __nv_bfloat16* __restrict__ S_out,
    __nv_bfloat16* __restrict__ uc_out,
    unsigned int batch_size,
    unsigned int seq_len,
    unsigned int num_chunks,
    unsigned int num_k_heads,
    unsigned int num_v_heads,
    unsigned int k_dim,
    unsigned int v_dim,
    unsigned int qk_stride,
    unsigned int gb_stride
) {
    const unsigned int vh = blockIdx.x;
    const unsigned int b = blockIdx.y;
    if (vh >= num_v_heads || b >= batch_size) return;
    const unsigned int tid = threadIdx.x;
    const unsigned int head_repeat = num_v_heads / num_k_heads;
    const unsigned int kh = vh / head_repeat;

    extern __shared__ char smem_raw[];
    __nv_bfloat16* buf = (__nv_bfloat16*)smem_raw;
    float* gcb = (float*)(buf + 2 * CDH_BUFSZ);
    float* decb = gcb + 2 * CHUNK;


    float* H = h_state + ((unsigned long long)(b * num_v_heads + vh) * K_DIM * V_DIM);
    float Sreg[K_DIM];
    #pragma unroll
    for (unsigned int k = 0; k < K_DIM; k++) Sreg[k] = H[k * V_DIM + tid];


    cdh_prefetch(buf, gcb, decb, 0, W_in, U_in, key, gate, gc_in, 0, b, vh, seq_len,
                 num_chunks, num_v_heads, k_dim, kh, qk_stride, gb_stride,
                 nullptr, nullptr, 0);

    for (unsigned int c = 0; c < num_chunks; c++) {
        const unsigned int cur = c & 1u;
        const unsigned int cs = c * CHUNK;
        const unsigned int ce = (seq_len - cs) < CHUNK ? (seq_len - cs) : CHUNK;
        const unsigned long long base = ((unsigned long long)(b * num_chunks + c) * num_v_heads + vh);

        if (c + 1 < num_chunks) {
            cdh_prefetch(buf, gcb, decb, (c + 1) & 1u, W_in, U_in, key, gate, gc_in, c + 1, b, vh, seq_len,
                         num_chunks, num_v_heads, k_dim, kh, qk_stride, gb_stride,
                 nullptr, nullptr, 0);
            cp_wait<1>();
        } else {
            cp_wait<0>();
        }
        __syncthreads();

        __nv_bfloat16* Wp = buf + (unsigned long long)cur * CDH_BUFSZ;
        __nv_bfloat16* Kp = Wp + CHUNK * K_DIM;
        __nv_bfloat16* Up = Kp + CHUNK * K_DIM;
        const float* dec = decb + cur * (CHUNK + 1);

        // 2026-09-25: S_out is BF16 because chunk_fwd_o uses S_c only as a BF16 MMA operand.

        #pragma unroll
        for (unsigned int k = 0; k < K_DIM; k++)
            S_out[base * K_DIM * V_DIM + k * V_DIM + tid] = __float2bfloat16(Sreg[k]);


        float duc[CHUNK];
        const float edl = dec[0];
        for (unsigned int i = 0; i < ce; i++) {
            float ws = 0.0f;
            #pragma unroll
            for (unsigned int k = 0; k < K_DIM; k++)
                ws += (float)Wp[i * K_DIM + k] * Sreg[k];
            float uci = (float)Up[i * V_DIM + tid] - ws;
            uc_out[base * CHUNK * V_DIM + i * v_dim + tid] = __float2bfloat16(uci);
            duc[i] = dec[1 + i] * uci;
        }

        #pragma unroll
        for (unsigned int k = 0; k < K_DIM; k++) {
            float hv = edl * Sreg[k];
            for (unsigned int i = 0; i < ce; i++)
                hv += duc[i] * (float)Kp[i * K_DIM + k];
            Sreg[k] = hv;
        }
        __syncthreads();   // 2026-09-25: slot cur is refilled by the next iteration's prefetch.
    }

    #pragma unroll
    for (unsigned int k = 0; k < K_DIM; k++) H[k * V_DIM + tid] = Sreg[k];
}

// 2026-09-25: gated_delta_rule_chunk_delta_h_tc: the scalar spine with W S_c on tensor cores
// (mma_gram). Each chunk stages S_c^T as BF16 in shared memory as the MMA operand; the FP32
// state in registers is not rounded. Single-buffered, 128 threads; K reuses the S^T region.
// Dynamic shared memory: S^T, W, ws (FP32), U and gc (98,560 B). No code in the repository
// launches it.





extern "C" __global__ void __launch_bounds__(128, 1)
gated_delta_rule_chunk_delta_h_tc(
    float* __restrict__ h_state,
    const __nv_bfloat16* __restrict__ W_in,
    const __nv_bfloat16* __restrict__ U_in,
    const __nv_bfloat16* __restrict__ key,
    const float* __restrict__ gate,
    const float* __restrict__ gc_in,
    __nv_bfloat16* __restrict__ S_out,
    __nv_bfloat16* __restrict__ uc_out,
    unsigned int batch_size,
    unsigned int seq_len,
    unsigned int num_chunks,
    unsigned int num_k_heads,
    unsigned int num_v_heads,
    unsigned int k_dim,
    unsigned int v_dim,
    unsigned int qk_stride,
    unsigned int gb_stride
) {
    const unsigned int vh = blockIdx.x;
    const unsigned int b = blockIdx.y;
    if (vh >= num_v_heads || b >= batch_size) return;
    const unsigned int tid = threadIdx.x;
    const unsigned int head_repeat = num_v_heads / num_k_heads;
    const unsigned int kh = vh / head_repeat;

    extern __shared__ char smem_raw[];
    __nv_bfloat16* St = (__nv_bfloat16*)smem_raw;
    __nv_bfloat16* Wb = St + V_DIM * K_DIM;
    float* ws = (float*)(Wb + CHUNK * K_DIM);
    __nv_bfloat16* Ub = (__nv_bfloat16*)(ws + CHUNK * V_DIM);
    float* gc = (float*)(Ub + CHUNK * V_DIM);
    __nv_bfloat16* Kb = St;

    float* H = h_state + ((unsigned long long)(b * num_v_heads + vh) * K_DIM * V_DIM);
    float Sreg[K_DIM];
    #pragma unroll
    for (unsigned int k = 0; k < K_DIM; k++) Sreg[k] = H[k * V_DIM + tid];

    for (unsigned int c = 0; c < num_chunks; c++) {
        const unsigned int cs = c * CHUNK;
        const unsigned int ce = (seq_len - cs) < CHUNK ? (seq_len - cs) : CHUNK;
        const unsigned long long base = ((unsigned long long)(b * num_chunks + c) * num_v_heads + vh);


        #pragma unroll
        for (unsigned int k = 0; k < K_DIM; k++)
            S_out[base * K_DIM * V_DIM + k * V_DIM + tid] = __float2bfloat16(Sreg[k]);


        #pragma unroll
        for (unsigned int k = 0; k < K_DIM; k++) St[tid * K_DIM + k] = __float2bfloat16(Sreg[k]);
        for (unsigned int idx = tid; idx < CHUNK * k_dim; idx += 128) {
            unsigned int i = idx / k_dim, k = idx % k_dim;
            Wb[i * K_DIM + k] = (i < ce) ? W_in[base * CHUNK * K_DIM + i * k_dim + k] : __float2bfloat16(0.0f);
        }
        for (unsigned int i = tid; i < ce; i += 128) {
            gc[i] = gc_in[base * CHUNK + i];
        }
        __syncthreads();


        mma_gram<16, V_DIM, false>(Wb, St, ws);
        __syncthreads();


        float duc[CHUNK];
        const float dl = gc[ce - 1];
        const float edl = expf(dl);
        for (unsigned int idx = tid; idx < CHUNK * v_dim; idx += 128) {
            unsigned int i = idx / v_dim, v = idx % v_dim;
            Ub[i * V_DIM + v] = (i < ce) ? U_in[base * CHUNK * V_DIM + i * v_dim + v] : __float2bfloat16(0.0f);
        }
        __syncthreads();
        if (tid < v_dim) {
            for (unsigned int i = 0; i < ce; i++) {
                float uci = (float)Ub[i * V_DIM + tid] - ws[i * V_DIM + tid];
                uc_out[base * CHUNK * V_DIM + i * v_dim + tid] = __float2bfloat16(uci);
                duc[i] = expf(dl - gc[i]) * uci;
            }
        }
        __syncthreads();   // 2026-09-25: the S^T region is reused for K.


        for (unsigned int idx = tid; idx < CHUNK * k_dim; idx += 128) {
            unsigned int i = idx / k_dim, k = idx % k_dim;
            Kb[i * K_DIM + k] = (i < ce)
                ? key[(unsigned long long)(cs + i) * qk_stride + kh * k_dim + k]
                : __float2bfloat16(0.0f);
        }
        __syncthreads();
        #pragma unroll
        for (unsigned int k = 0; k < K_DIM; k++) {
            float hv = edl * Sreg[k];
            for (unsigned int i = 0; i < ce; i++)
                hv += duc[i] * (float)Kb[i * K_DIM + k];
            Sreg[k] = hv;
        }
        __syncthreads();   // 2026-09-25: St, Wb and ws are refilled next chunk.
    }

    #pragma unroll
    for (unsigned int k = 0; k < K_DIM; k++) H[k * V_DIM + tid] = Sreg[k];
}

// 2026-09-25: gated_delta_rule_chunk_delta_h_ksplit = cdh_ksplit_core<2>: the scalar spine with
// each state column split over SPLIT threads. Thread (v, sub) keeps rows sub * KH .. sub * KH +
// KH - 1 of S[:, v] in registers (KH = K_DIM / SPLIT), and a __shfl_xor butterfly over the SPLIT
// lanes completes each W S_c dot. Adds varlen and pointer-table h_state to the scalar spine;
// 256 threads, same dynamic shared memory (99,336 B). ops::gdn_prefill_fla launches it when
// neither the TC, TMA, fused nor tc_vblock spine is selected.




template <int SPLIT>
__device__ __forceinline__ void cdh_ksplit_core(
    float* __restrict__ h_state, const __nv_bfloat16* __restrict__ W_in,
    const __nv_bfloat16* __restrict__ U_in, const __nv_bfloat16* __restrict__ key,
    const float* __restrict__ gate, const float* __restrict__ gc_in,
    __nv_bfloat16* __restrict__ S_out, __nv_bfloat16* __restrict__ uc_out,
    unsigned int seq_len, unsigned int num_chunks, unsigned int num_k_heads,
    unsigned int num_v_heads, unsigned int k_dim, unsigned int v_dim,
    unsigned int qk_stride, unsigned int gb_stride, unsigned int h_state_is_table,
    const int* __restrict__ cu_seqlens, const int* __restrict__ cu_chunks,
    unsigned int is_varlen
) {
    constexpr int KH = K_DIM / SPLIT;
    const unsigned int vh = blockIdx.x;
    const unsigned int b = blockIdx.y;
    if (vh >= num_v_heads) return;
    GDN_GEOM(g);
    const unsigned int t = threadIdx.x;
    const unsigned int v = t / SPLIT;
    const unsigned int sub = t % SPLIT;
    const unsigned int k0 = sub * KH;
    const unsigned int head_repeat = num_v_heads / num_k_heads;
    const unsigned int kh = vh / head_repeat;




    extern __shared__ char smem_raw_dhc[];
    __nv_bfloat16* buf = (__nv_bfloat16*)smem_raw_dhc;
    float* gcb = (float*)(buf + 2 * CDH_BUFSZ);
    float* decb = gcb + 2 * CHUNK;

    // 2026-09-25: With h_state_is_table, h_state is a table of per-sequence pointers, each to a
    // [num_v_heads][K_DIM][V_DIM] state; otherwise a contiguous base indexed by b * num_v_heads + vh.

    float* H = h_state_is_table
        ? ((float* const*)h_state)[b] + (unsigned long long)vh * K_DIM * V_DIM
        : h_state + ((unsigned long long)(b * num_v_heads + vh) * K_DIM * V_DIM);
    float Sreg[KH];
    #pragma unroll
    for (int kk = 0; kk < KH; kk++) Sreg[kk] = H[(k0 + kk) * V_DIM + v];

    cdh_prefetch(buf, gcb, decb, 0, W_in, U_in, key, gate, gc_in, 0, b, vh, seq_len,
                 num_chunks, num_v_heads, k_dim, kh, qk_stride, gb_stride,
                 cu_seqlens, cu_chunks, is_varlen);

    for (unsigned int c = 0; c < g.nchunks; c++) {
        const unsigned int cur = c & 1u;
        const unsigned int cs = c * CHUNK;
        const unsigned int ce = (g.seqlen - cs) < CHUNK ? (g.seqlen - cs) : CHUNK;
        const unsigned long long base = ((unsigned long long)(g.choff + c) * num_v_heads + vh);

        if (c + 1 < g.nchunks) {
            cdh_prefetch(buf, gcb, decb, (c + 1) & 1u, W_in, U_in, key, gate, gc_in, c + 1, b, vh, seq_len,
                         num_chunks, num_v_heads, k_dim, kh, qk_stride, gb_stride,
                         cu_seqlens, cu_chunks, is_varlen);
            cp_wait<1>();
        } else {
            cp_wait<0>();
        }
        __syncthreads();

        __nv_bfloat16* Wp = buf + (unsigned long long)cur * CDH_BUFSZ;
        __nv_bfloat16* Kp = Wp + CHUNK * K_DIM;
        __nv_bfloat16* Up = Kp + CHUNK * K_DIM;
        const float* dec = decb + cur * (CHUNK + 1);

        #pragma unroll
        for (int kk = 0; kk < KH; kk++)
            S_out[base * K_DIM * V_DIM + (k0 + kk) * V_DIM + v] = __float2bfloat16(Sreg[kk]);

        const float edl = dec[0];
        float duc[CHUNK];
        for (unsigned int i = 0; i < ce; i++) {
            float wsp = 0.0f;
            #pragma unroll
            for (int kk = 0; kk < KH; kk++)
                wsp += (float)Wp[i * K_DIM + k0 + kk] * Sreg[kk];
            #pragma unroll
            for (int s = 1; s < SPLIT; s <<= 1) wsp += __shfl_xor_sync(0xffffffffu, wsp, s);
            float uci = (float)Up[i * V_DIM + v] - wsp;
            if (sub == 0) uc_out[base * CHUNK * V_DIM + i * v_dim + v] = __float2bfloat16(uci);
            duc[i] = dec[1 + i] * uci;
        }
        #pragma unroll
        for (int kk = 0; kk < KH; kk++) {
            float hv = edl * Sreg[kk];
            for (unsigned int i = 0; i < ce; i++)
                hv += duc[i] * (float)Kp[i * K_DIM + k0 + kk];
            Sreg[kk] = hv;
        }
        __syncthreads();
    }

    #pragma unroll
    for (int kk = 0; kk < KH; kk++) H[(k0 + kk) * V_DIM + v] = Sreg[kk];
}





extern "C" __global__ void __launch_bounds__(256, 1)
gated_delta_rule_chunk_delta_h_ksplit(
    float* __restrict__ h_state, const __nv_bfloat16* __restrict__ W_in,
    const __nv_bfloat16* __restrict__ U_in, const __nv_bfloat16* __restrict__ key,
    const float* __restrict__ gate, const float* __restrict__ gc_in,
    __nv_bfloat16* __restrict__ S_out, __nv_bfloat16* __restrict__ uc_out,
    unsigned int batch_size, unsigned int seq_len, unsigned int num_chunks,
    unsigned int num_k_heads, unsigned int num_v_heads, unsigned int k_dim,
    unsigned int v_dim, unsigned int qk_stride, unsigned int gb_stride,
    unsigned int h_state_is_table,
    const int* __restrict__ cu_seqlens, const int* __restrict__ cu_chunks,
    unsigned int is_varlen
) {
    cdh_ksplit_core<2>(h_state, W_in, U_in, key, gate, gc_in, S_out, uc_out, seq_len, num_chunks,
                       num_k_heads, num_v_heads, k_dim, v_dim, qk_stride, gb_stride, h_state_is_table,
                       cu_seqlens, cu_chunks, is_varlen);
}

// 2026-09-25: cdh_vtile_core<SPLIT, VT>: one pass per chunk. S_old feeds the W S dot for every
// token and is not written in the loop; each token's decayed correction is added to S_new at once,
// so no per-chunk duc array is kept. W, K and U are staged in shared memory, single-buffered, with
// plain loads (three 16 KB tiles and the decay row: 49,412 B). Each thread holds VT value columns
// and KH = K_DIM / SPLIT rows of each.































template <int SPLIT, int VT>
__device__ __forceinline__ void cdh_vtile_core(
    float* __restrict__ h_state, const __nv_bfloat16* __restrict__ W_in,
    const __nv_bfloat16* __restrict__ U_in, const __nv_bfloat16* __restrict__ key,
    const float* __restrict__ gate, const float* __restrict__ gc_in,
    __nv_bfloat16* __restrict__ S_out, __nv_bfloat16* __restrict__ uc_out,
    unsigned int seq_len, unsigned int num_chunks, unsigned int num_k_heads,
    unsigned int num_v_heads, unsigned int k_dim, unsigned int v_dim,
    unsigned int qk_stride, unsigned int gb_stride, unsigned int h_state_is_table,
    const int* __restrict__ cu_seqlens, const int* __restrict__ cu_chunks,
    unsigned int is_varlen
) {
    constexpr int KH = K_DIM / SPLIT;
    const unsigned int vh = blockIdx.x;
    const unsigned int b = blockIdx.y;
    if (vh >= num_v_heads) return;
    GDN_GEOM(g);
    const unsigned int t = threadIdx.x;
    const unsigned int v0 = (t / SPLIT) * VT;
    const unsigned int sub = t % SPLIT;
    const unsigned int k0 = sub * KH;
    const unsigned int head_repeat = num_v_heads / num_k_heads;
    const unsigned int kh = vh / head_repeat;

    extern __shared__ char smem_raw_vt[];
    __nv_bfloat16* Wp = (__nv_bfloat16*)smem_raw_vt;
    __nv_bfloat16* Kp = Wp + CHUNK * K_DIM;
    __nv_bfloat16* Up = Kp + CHUNK * K_DIM;
    float* decs = (float*)(Up + CHUNK * V_DIM);

    float* H = h_state_is_table
        ? ((float* const*)h_state)[b] + (unsigned long long)vh * K_DIM * V_DIM
        : h_state + ((unsigned long long)(b * num_v_heads + vh) * K_DIM * V_DIM);
    float Sold[KH][VT];
    #pragma unroll
    for (int kk = 0; kk < KH; kk++)
        #pragma unroll
        for (int vt = 0; vt < VT; vt++) Sold[kk][vt] = H[(k0 + kk) * V_DIM + v0 + vt];

    const __nv_bfloat16* key_b = key + g.tokoff * qk_stride;
    const unsigned int nthr = blockDim.x;

    for (unsigned int c = 0; c < g.nchunks; c++) {
        const unsigned int cs = c * CHUNK;
        const unsigned int ce = (g.seqlen - cs) < CHUNK ? (g.seqlen - cs) : CHUNK;
        const unsigned long long base = ((unsigned long long)(g.choff + c) * num_v_heads + vh);

        __syncthreads();
        const __nv_bfloat16* Wsrc = W_in + base * CHUNK * K_DIM;
        for (unsigned int e = t; e < CHUNK * K_DIM; e += nthr) Wp[e] = Wsrc[e];
        for (unsigned int e = t; e < CHUNK * K_DIM; e += nthr) {
            unsigned int i = e / K_DIM, kx = e % K_DIM;
            Kp[e] = (i < ce)
                ? key_b[(unsigned long long)(cs + i) * qk_stride + kh * k_dim + kx]
                : __float2bfloat16(0.0f);
        }
        {
            const __nv_bfloat16* Usrc0 = U_in + base * CHUNK * V_DIM;
            for (unsigned int e = t; e < CHUNK * V_DIM; e += nthr) Up[e] = Usrc0[e];
        }
        if (t == 0) {
            float dl = gc_in[base * CHUNK + ce - 1];
            decs[0] = expf(dl);
            for (unsigned int i = 0; i < ce; i++) decs[1 + i] = expf(dl - gc_in[base * CHUNK + i]);
        }
        __syncthreads();

        #pragma unroll
        for (int kk = 0; kk < KH; kk++)
            #pragma unroll
            for (int vt = 0; vt < VT; vt++)
                S_out[base * K_DIM * V_DIM + (k0 + kk) * V_DIM + v0 + vt] =
                    __float2bfloat16(Sold[kk][vt]);



        const float edl = decs[0];
        float Snew[KH][VT];
        #pragma unroll
        for (int kk = 0; kk < KH; kk++)
            #pragma unroll
            for (int vt = 0; vt < VT; vt++) Snew[kk][vt] = edl * Sold[kk][vt];

        for (unsigned int i = 0; i < ce; i++) {
            float wsp[VT];
            #pragma unroll
            for (int vt = 0; vt < VT; vt++) wsp[vt] = 0.0f;
            #pragma unroll
            for (int kk = 0; kk < KH; kk++) {
                const float w = (float)Wp[i * K_DIM + k0 + kk];
                #pragma unroll
                for (int vt = 0; vt < VT; vt++) wsp[vt] += w * Sold[kk][vt];
            }
            #pragma unroll
            for (int vt = 0; vt < VT; vt++)
                #pragma unroll
                for (int s = 1; s < SPLIT; s <<= 1)
                    wsp[vt] += __shfl_xor_sync(0xffffffffu, wsp[vt], s);
            const float dc = decs[1 + i];
            float d[VT];
            #pragma unroll
            for (int vt = 0; vt < VT; vt++) {
                const float uci = (float)Up[i * V_DIM + v0 + vt] - wsp[vt];
                if (sub == 0) uc_out[base * CHUNK * V_DIM + i * v_dim + v0 + vt] = __float2bfloat16(uci);
                d[vt] = dc * uci;
            }
            #pragma unroll
            for (int kk = 0; kk < KH; kk++) {
                const float kv = (float)Kp[i * K_DIM + k0 + kk];
                #pragma unroll
                for (int vt = 0; vt < VT; vt++) Snew[kk][vt] += d[vt] * kv;
            }
        }
        #pragma unroll
        for (int kk = 0; kk < KH; kk++)
            #pragma unroll
            for (int vt = 0; vt < VT; vt++) Sold[kk][vt] = Snew[kk][vt];
    }

    #pragma unroll
    for (int kk = 0; kk < KH; kk++)
        #pragma unroll
        for (int vt = 0; vt < VT; vt++) H[(k0 + kk) * V_DIM + v0 + vt] = Sold[kk][vt];
}

// 2026-09-25: gated_delta_rule_chunk_delta_h_vtile = cdh_vtile_core<4, 1>: 512 threads
// (128 columns x SPLIT 4, KH = 32), grid (num_v_heads, batch_size). Selected by
// METRALE_GDN_VTILE=1 (init_kernels::fused_spine_kernel), and launched with 512 threads.

extern "C" __global__ void __launch_bounds__(512, 1)
gated_delta_rule_chunk_delta_h_vtile(
    float* __restrict__ h_state, const __nv_bfloat16* __restrict__ W_in,
    const __nv_bfloat16* __restrict__ U_in, const __nv_bfloat16* __restrict__ key,
    const float* __restrict__ gate, const float* __restrict__ gc_in,
    __nv_bfloat16* __restrict__ S_out, __nv_bfloat16* __restrict__ uc_out,
    unsigned int batch_size, unsigned int seq_len, unsigned int num_chunks,
    unsigned int num_k_heads, unsigned int num_v_heads, unsigned int k_dim,
    unsigned int v_dim, unsigned int qk_stride, unsigned int gb_stride,
    unsigned int h_state_is_table,
    const int* __restrict__ cu_seqlens, const int* __restrict__ cu_chunks,
    unsigned int is_varlen
) {
    cdh_vtile_core<4, 1>(h_state, W_in, U_in, key, gate, gc_in, S_out, uc_out, seq_len,
                         num_chunks, num_k_heads, num_v_heads, k_dim, v_dim, qk_stride,
                         gb_stride, h_state_is_table, cu_seqlens, cu_chunks, is_varlen);
}

// 2026-09-25: gated_delta_rule_chunk_delta_h_vfused = cdh_vtile_core<2, 1>: 256 threads
// (128 columns x SPLIT 2), grid (num_v_heads, batch_size), 49,412 B of dynamic shared memory.
// init_kernels::fused_spine_kernel binds it unless METRALE_GDN_PIPE=1 or METRALE_GDN_VTILE=1.
























extern "C" __global__ void __launch_bounds__(256, 1)
gated_delta_rule_chunk_delta_h_vfused(
    float* __restrict__ h_state, const __nv_bfloat16* __restrict__ W_in,
    const __nv_bfloat16* __restrict__ U_in, const __nv_bfloat16* __restrict__ key,
    const float* __restrict__ gate, const float* __restrict__ gc_in,
    __nv_bfloat16* __restrict__ S_out, __nv_bfloat16* __restrict__ uc_out,
    unsigned int batch_size, unsigned int seq_len, unsigned int num_chunks,
    unsigned int num_k_heads, unsigned int num_v_heads, unsigned int k_dim,
    unsigned int v_dim, unsigned int qk_stride, unsigned int gb_stride,
    unsigned int h_state_is_table,
    const int* __restrict__ cu_seqlens, const int* __restrict__ cu_chunks,
    unsigned int is_varlen
) {
    cdh_vtile_core<2, 1>(h_state, W_in, U_in, key, gate, gc_in, S_out, uc_out, seq_len,
                         num_chunks, num_k_heads, num_v_heads, k_dim, v_dim, qk_stride,
                         gb_stride, h_state_is_table, cu_seqlens, cu_chunks, is_varlen);
}

// 2026-09-25: cdh_pipe_core: cdh_vtile_core's single pass per chunk, with W, K and U copied by
// cdh_prefetch into two slots so chunk c + 1 loads while chunk c computes. Dynamic shared
// memory: two CDH_BUFSZ slots plus the gc and decay tables (99,336 B, the launcher's smem_dh).






















template <int SPLIT, int VT>
__device__ __forceinline__ void cdh_pipe_core(
    float* __restrict__ h_state, const __nv_bfloat16* __restrict__ W_in,
    const __nv_bfloat16* __restrict__ U_in, const __nv_bfloat16* __restrict__ key,
    const float* __restrict__ gate, const float* __restrict__ gc_in,
    __nv_bfloat16* __restrict__ S_out, __nv_bfloat16* __restrict__ uc_out,
    unsigned int seq_len, unsigned int num_chunks, unsigned int num_k_heads,
    unsigned int num_v_heads, unsigned int k_dim, unsigned int v_dim,
    unsigned int qk_stride, unsigned int gb_stride, unsigned int h_state_is_table,
    const int* __restrict__ cu_seqlens, const int* __restrict__ cu_chunks,
    unsigned int is_varlen
) {
    constexpr int KH = K_DIM / SPLIT;
    const unsigned int vh = blockIdx.x;
    const unsigned int b = blockIdx.y;
    if (vh >= num_v_heads) return;
    GDN_GEOM(g);
    const unsigned int t = threadIdx.x;
    const unsigned int v0 = (t / SPLIT) * VT;
    const unsigned int sub = t % SPLIT;
    const unsigned int k0 = sub * KH;
    const unsigned int head_repeat = num_v_heads / num_k_heads;
    const unsigned int kh = vh / head_repeat;

    extern __shared__ char smem_raw_pipe[];
    __nv_bfloat16* bufs = (__nv_bfloat16*)smem_raw_pipe;
    float* gcb  = (float*)(bufs + 2 * CDH_BUFSZ);
    float* decb = gcb + 2 * CHUNK;

    float* H = h_state_is_table
        ? ((float* const*)h_state)[b] + (unsigned long long)vh * K_DIM * V_DIM
        : h_state + ((unsigned long long)(b * num_v_heads + vh) * K_DIM * V_DIM);
    float Sold[KH][VT];
    #pragma unroll
    for (int kk = 0; kk < KH; kk++)
        #pragma unroll
        for (int vt = 0; vt < VT; vt++) Sold[kk][vt] = H[(k0 + kk) * V_DIM + v0 + vt];

    if (g.nchunks == 0) return;


    cdh_prefetch(bufs, gcb, decb, 0, W_in, U_in, key, gate, gc_in, 0, b, vh, seq_len,
                 num_chunks, num_v_heads, k_dim, kh, qk_stride, gb_stride,
                 cu_seqlens, cu_chunks, is_varlen);

    for (unsigned int c = 0; c < g.nchunks; c++) {
        const unsigned int p = c & 1u;
        const unsigned int ce = (g.seqlen - c * CHUNK) < CHUNK ? (g.seqlen - c * CHUNK) : CHUNK;
        const unsigned long long base = ((unsigned long long)(g.choff + c) * num_v_heads + vh);

        if (c + 1 < g.nchunks) {
            cdh_prefetch(bufs, gcb, decb, p ^ 1u, W_in, U_in, key, gate, gc_in, c + 1, b, vh,
                         seq_len, num_chunks, num_v_heads, k_dim, kh, qk_stride, gb_stride,
                         cu_seqlens, cu_chunks, is_varlen);
            cp_wait<1>();
        } else {
            cp_wait<0>();
        }
        __syncthreads();

        const __nv_bfloat16* Wp = bufs + (unsigned long long)p * CDH_BUFSZ;
        const __nv_bfloat16* Kp = Wp + CHUNK * K_DIM;
        const __nv_bfloat16* Up = Kp + CHUNK * K_DIM;
        const float* decs = decb + p * (CHUNK + 1);

        #pragma unroll
        for (int kk = 0; kk < KH; kk++)
            #pragma unroll
            for (int vt = 0; vt < VT; vt++)
                S_out[base * K_DIM * V_DIM + (k0 + kk) * V_DIM + v0 + vt] =
                    __float2bfloat16(Sold[kk][vt]);

        const float edl = decs[0];
        float Snew[KH][VT];
        #pragma unroll
        for (int kk = 0; kk < KH; kk++)
            #pragma unroll
            for (int vt = 0; vt < VT; vt++) Snew[kk][vt] = edl * Sold[kk][vt];

        for (unsigned int i = 0; i < ce; i++) {
            float wsp[VT];
            #pragma unroll
            for (int vt = 0; vt < VT; vt++) wsp[vt] = 0.0f;
            #pragma unroll
            for (int kk = 0; kk < KH; kk++) {
                const float w = (float)Wp[i * K_DIM + k0 + kk];
                #pragma unroll
                for (int vt = 0; vt < VT; vt++) wsp[vt] += w * Sold[kk][vt];
            }
            #pragma unroll
            for (int vt = 0; vt < VT; vt++)
                #pragma unroll
                for (int sft = 1; sft < SPLIT; sft <<= 1)
                    wsp[vt] += __shfl_xor_sync(0xffffffffu, wsp[vt], sft);
            const float dc = decs[1 + i];
            float d[VT];
            #pragma unroll
            for (int vt = 0; vt < VT; vt++) {
                const float uci = (float)Up[i * V_DIM + v0 + vt] - wsp[vt];
                if (sub == 0) uc_out[base * CHUNK * V_DIM + i * v_dim + v0 + vt] = __float2bfloat16(uci);
                d[vt] = dc * uci;
            }
            #pragma unroll
            for (int kk = 0; kk < KH; kk++) {
                const float kv = (float)Kp[i * K_DIM + k0 + kk];
                #pragma unroll
                for (int vt = 0; vt < VT; vt++) Snew[kk][vt] += d[vt] * kv;
            }
        }
        #pragma unroll
        for (int kk = 0; kk < KH; kk++)
            #pragma unroll
            for (int vt = 0; vt < VT; vt++) Sold[kk][vt] = Snew[kk][vt];
        // 2026-09-25: The next iteration refills slot p; no thread may still be reading it.
        __syncthreads();
    }

    #pragma unroll
    for (int kk = 0; kk < KH; kk++)
        #pragma unroll
        for (int vt = 0; vt < VT; vt++) H[(k0 + kk) * V_DIM + v0 + vt] = Sold[kk][vt];
}

// 2026-09-25: gated_delta_rule_chunk_delta_h_pipe = cdh_pipe_core<2, 1>, 256 threads; METRALE_GDN_PIPE=1.
extern "C" __global__ void __launch_bounds__(256, 1)
gated_delta_rule_chunk_delta_h_pipe(
    float* __restrict__ h_state, const __nv_bfloat16* __restrict__ W_in,
    const __nv_bfloat16* __restrict__ U_in, const __nv_bfloat16* __restrict__ key,
    const float* __restrict__ gate, const float* __restrict__ gc_in,
    __nv_bfloat16* __restrict__ S_out, __nv_bfloat16* __restrict__ uc_out,
    unsigned int batch_size, unsigned int seq_len, unsigned int num_chunks,
    unsigned int num_k_heads, unsigned int num_v_heads, unsigned int k_dim,
    unsigned int v_dim, unsigned int qk_stride, unsigned int gb_stride,
    unsigned int h_state_is_table,
    const int* __restrict__ cu_seqlens, const int* __restrict__ cu_chunks,
    unsigned int is_varlen
) {
    cdh_pipe_core<2, 1>(h_state, W_in, U_in, key, gate, gc_in, S_out, uc_out, seq_len,
                        num_chunks, num_k_heads, num_v_heads, k_dim, v_dim, qk_stride,
                        gb_stride, h_state_is_table, cu_seqlens, cu_chunks, is_varlen);
}

// 2026-09-25: cdh_tma_core: cdh_pipe_core with the W, U and K tiles loaded by TMA into two
// stages. Thread 0 issues one mbar_expect_tx for the three tiles of a stage, then the three
// tma_load_2d; K is read as a 2-D tile at (column kh * K_DIM, row tokoff + c * CHUNK) of the
// packed key tensor. Stage p is reused every second chunk, so the parity to wait for is
// (c >> 1) & 1. Rows past a sequence's end are loaded but never read (the loops stop at ce).
// Requires k_dim == K_DIM, v_dim == V_DIM, CHUNK == 64, a uniform batch and qk_stride % 8 == 0,
// because the descriptors encode the compile-time tile; ops::gdn_prefill_fla checks these and
// falls back. Selected by METRALE_GDN_TMA=1 unless the tensor-core spine is active.


















template <int SPLIT, int VT>
__device__ __forceinline__ void cdh_tma_core(
    float* __restrict__ h_state, const CUtensorMap* w_desc, const CUtensorMap* u_desc,
    const CUtensorMap* k_desc, const float* __restrict__ gc_in,
    __nv_bfloat16* __restrict__ S_out, __nv_bfloat16* __restrict__ uc_out,
    unsigned int seq_len, unsigned int num_chunks, unsigned int num_k_heads,
    unsigned int num_v_heads, unsigned int v_dim, unsigned int h_state_is_table,
    const int* __restrict__ cu_seqlens, const int* __restrict__ cu_chunks,
    unsigned int is_varlen
) {
    constexpr int KH = K_DIM / SPLIT;
    constexpr unsigned int STAGE_BYTES =
        (unsigned int)(CHUNK * K_DIM * 2 + CHUNK * K_DIM * 2 + CHUNK * V_DIM * 2);
    const unsigned int vh = blockIdx.x;
    const unsigned int b = blockIdx.y;
    if (vh >= num_v_heads) return;
    GDN_GEOM(g);
    const unsigned int t = threadIdx.x;
    const unsigned int v0 = (t / SPLIT) * VT;
    const unsigned int sub = t % SPLIT;
    const unsigned int k0 = sub * KH;
    const unsigned int head_repeat = num_v_heads / num_k_heads;
    const unsigned int kh = vh / head_repeat;
    const unsigned int nthr = blockDim.x;


    extern __shared__ __align__(128) char smem_raw_tma[];
    __nv_bfloat16* bufs = (__nv_bfloat16*)smem_raw_tma;
    float* decb = (float*)(bufs + 2 * CDH_BUFSZ);
    __shared__ __align__(8) uint64_t bar[2];

    if (t == 0) {
        mbar_init(&bar[0], 1);
        mbar_init(&bar[1], 1);
    }
    __syncthreads();

    float* H = h_state_is_table
        ? ((float* const*)h_state)[b] + (unsigned long long)vh * K_DIM * V_DIM
        : h_state + ((unsigned long long)(b * num_v_heads + vh) * K_DIM * V_DIM);
    float Sold[KH][VT];
    #pragma unroll
    for (int kk = 0; kk < KH; kk++)
        #pragma unroll
        for (int vt = 0; vt < VT; vt++) Sold[kk][vt] = H[(k0 + kk) * V_DIM + v0 + vt];

    if (g.nchunks == 0) return;


    auto issue = [&](unsigned int c, unsigned int p) {
        const unsigned long long base =
            ((unsigned long long)(g.choff + c) * num_v_heads + vh);
        __nv_bfloat16* Wp = bufs + (unsigned long long)p * CDH_BUFSZ;
        __nv_bfloat16* Kp = Wp + CHUNK * K_DIM;
        __nv_bfloat16* Up = Kp + CHUNK * K_DIM;
        if (t == 0) {
            mbar_expect_tx(&bar[p], STAGE_BYTES);
            tma_fence();
            tma_load_2d(w_desc, &bar[p], Wp, 0, (int)(base * CHUNK));
            tma_load_2d(u_desc, &bar[p], Up, 0, (int)(base * CHUNK));
            tma_load_2d(k_desc, &bar[p], Kp, (int)(kh * K_DIM),
                        (int)(g.tokoff + (unsigned long long)c * CHUNK));
        }


        const unsigned int cs = c * CHUNK;
        const unsigned int ce = (g.seqlen - cs) < CHUNK ? (g.seqlen - cs) : CHUNK;
        const float dl = gc_in[base * CHUNK + ce - 1];
        if (t == 0) decb[p * (CHUNK + 1)] = expf(dl);
        for (unsigned int i = t; i < ce; i += nthr)
            decb[p * (CHUNK + 1) + 1 + i] = expf(dl - gc_in[base * CHUNK + i]);
    };

    issue(0, 0);

    for (unsigned int c = 0; c < g.nchunks; c++) {
        const unsigned int p = c & 1u;
        const unsigned int ce = (g.seqlen - c * CHUNK) < CHUNK ? (g.seqlen - c * CHUNK) : CHUNK;
        const unsigned long long base = ((unsigned long long)(g.choff + c) * num_v_heads + vh);

        if (c + 1 < g.nchunks) issue(c + 1, p ^ 1u);
        mbar_wait(&bar[p], (c >> 1) & 1u);
        __syncthreads();

        const __nv_bfloat16* Wp = bufs + (unsigned long long)p * CDH_BUFSZ;
        const __nv_bfloat16* Kp = Wp + CHUNK * K_DIM;
        const __nv_bfloat16* Up = Kp + CHUNK * K_DIM;
        const float* decs = decb + p * (CHUNK + 1);

        #pragma unroll
        for (int kk = 0; kk < KH; kk++)
            #pragma unroll
            for (int vt = 0; vt < VT; vt++)
                S_out[base * K_DIM * V_DIM + (k0 + kk) * V_DIM + v0 + vt] =
                    __float2bfloat16(Sold[kk][vt]);

        const float edl = decs[0];
        float Snew[KH][VT];
        #pragma unroll
        for (int kk = 0; kk < KH; kk++)
            #pragma unroll
            for (int vt = 0; vt < VT; vt++) Snew[kk][vt] = edl * Sold[kk][vt];

        for (unsigned int i = 0; i < ce; i++) {
            float wsp[VT];
            #pragma unroll
            for (int vt = 0; vt < VT; vt++) wsp[vt] = 0.0f;
            #pragma unroll
            for (int kk = 0; kk < KH; kk++) {
                const float w = (float)Wp[i * K_DIM + k0 + kk];
                #pragma unroll
                for (int vt = 0; vt < VT; vt++) wsp[vt] += w * Sold[kk][vt];
            }
            #pragma unroll
            for (int vt = 0; vt < VT; vt++)
                #pragma unroll
                for (int sft = 1; sft < SPLIT; sft <<= 1)
                    wsp[vt] += __shfl_xor_sync(0xffffffffu, wsp[vt], sft);
            const float dc = decs[1 + i];
            float d[VT];
            #pragma unroll
            for (int vt = 0; vt < VT; vt++) {
                const float uci = (float)Up[i * V_DIM + v0 + vt] - wsp[vt];
                if (sub == 0) uc_out[base * CHUNK * V_DIM + i * v_dim + v0 + vt] = __float2bfloat16(uci);
                d[vt] = dc * uci;
            }
            #pragma unroll
            for (int kk = 0; kk < KH; kk++) {
                const float kv = (float)Kp[i * K_DIM + k0 + kk];
                #pragma unroll
                for (int vt = 0; vt < VT; vt++) Snew[kk][vt] += d[vt] * kv;
            }
        }
        #pragma unroll
        for (int kk = 0; kk < KH; kk++)
            #pragma unroll
            for (int vt = 0; vt < VT; vt++) Sold[kk][vt] = Snew[kk][vt];
        __syncthreads();
    }

    #pragma unroll
    for (int kk = 0; kk < KH; kk++)
        #pragma unroll
        for (int vt = 0; vt < VT; vt++) H[(k0 + kk) * V_DIM + v0 + vt] = Sold[kk][vt];
}

extern "C" __global__ void __launch_bounds__(256, 1)
gated_delta_rule_chunk_delta_h_tma(
    float* __restrict__ h_state,
    const __grid_constant__ CUtensorMap w_desc,
    const __grid_constant__ CUtensorMap u_desc,
    const __grid_constant__ CUtensorMap k_desc,
    const float* __restrict__ gc_in,
    __nv_bfloat16* __restrict__ S_out, __nv_bfloat16* __restrict__ uc_out,
    unsigned int batch_size, unsigned int seq_len, unsigned int num_chunks,
    unsigned int num_k_heads, unsigned int num_v_heads, unsigned int v_dim,
    unsigned int h_state_is_table,
    const int* __restrict__ cu_seqlens, const int* __restrict__ cu_chunks,
    unsigned int is_varlen
) {
    cdh_tma_core<2, 1>(h_state, &w_desc, &u_desc, &k_desc, gc_in, S_out, uc_out, seq_len,
                       num_chunks, num_k_heads, num_v_heads, v_dim, h_state_is_table,
                       cu_seqlens, cu_chunks, is_varlen);
}

// 2026-09-25: cdh_dvsplit_core<SPLIT, DVB>: the ksplit spine with the value columns split into
// blocks of DVB, one per block (grid.y = batch_size * V_DIM / DVB, blockIdx.y = b * NDVB + dvb).
// A column block of S_c+1 depends only on the same columns of S_c, since W S contracts over k.
// W, K and the block's U columns are staged single-buffered with plain loads; every block loads
// the full W and K. Only the gdn_chunk_shapetest example launches it.

























template <int SPLIT, int DVB>
__device__ __forceinline__ void cdh_dvsplit_core(
    float* __restrict__ h_state, const __nv_bfloat16* __restrict__ W_in,
    const __nv_bfloat16* __restrict__ U_in, const __nv_bfloat16* __restrict__ key,
    const float* __restrict__ gate, const float* __restrict__ gc_in,
    __nv_bfloat16* __restrict__ S_out, __nv_bfloat16* __restrict__ uc_out,
    unsigned int seq_len, unsigned int num_chunks, unsigned int num_k_heads,
    unsigned int num_v_heads, unsigned int k_dim, unsigned int v_dim,
    unsigned int qk_stride, unsigned int gb_stride, unsigned int h_state_is_table,
    const int* __restrict__ cu_seqlens, const int* __restrict__ cu_chunks,
    unsigned int is_varlen
) {
    constexpr int KH = K_DIM / SPLIT;
    constexpr int NDVB = V_DIM / DVB;
    const unsigned int vh = blockIdx.x;
    if (vh >= num_v_heads) return;
    const unsigned int dvb = blockIdx.y % (unsigned int)NDVB;
    const unsigned int b = blockIdx.y / (unsigned int)NDVB;
    const unsigned int dv0 = dvb * (unsigned int)DVB;
    GDN_GEOM(g);
    const unsigned int t = threadIdx.x;
    const unsigned int vloc = t / SPLIT;
    const unsigned int v = dv0 + vloc;
    const unsigned int sub = t % SPLIT;
    const unsigned int k0 = sub * KH;
    const unsigned int head_repeat = num_v_heads / num_k_heads;
    const unsigned int kh = vh / head_repeat;

    extern __shared__ char smem_raw_dvs[];
    __nv_bfloat16* Wp = (__nv_bfloat16*)smem_raw_dvs;
    __nv_bfloat16* Kp = Wp + CHUNK * K_DIM;
    __nv_bfloat16* Up = Kp + CHUNK * K_DIM;
    float* decs = (float*)(Up + CHUNK * DVB);

    float* H = h_state_is_table
        ? ((float* const*)h_state)[b] + (unsigned long long)vh * K_DIM * V_DIM
        : h_state + ((unsigned long long)(b * num_v_heads + vh) * K_DIM * V_DIM);
    float Sreg[KH];
    #pragma unroll
    for (int kk = 0; kk < KH; kk++) Sreg[kk] = H[(k0 + kk) * V_DIM + v];

    const __nv_bfloat16* key_b = key + g.tokoff * qk_stride;
    const unsigned int nthr = blockDim.x;

    for (unsigned int c = 0; c < g.nchunks; c++) {
        const unsigned int cs = c * CHUNK;
        const unsigned int ce = (g.seqlen - cs) < CHUNK ? (g.seqlen - cs) : CHUNK;
        const unsigned long long base = ((unsigned long long)(g.choff + c) * num_v_heads + vh);

        __syncthreads();   // 2026-09-25: the previous chunk's readers are done with shared memory.
        const __nv_bfloat16* Wsrc = W_in + base * CHUNK * K_DIM;
        for (unsigned int e = t; e < CHUNK * K_DIM; e += nthr) Wp[e] = Wsrc[e];
        const __nv_bfloat16* Usrc = U_in + base * CHUNK * V_DIM;
        for (unsigned int e = t; e < CHUNK * DVB; e += nthr)
            Up[e] = Usrc[(e / DVB) * V_DIM + dv0 + (e % DVB)];
        for (unsigned int e = t; e < CHUNK * K_DIM; e += nthr) {
            unsigned int i = e / K_DIM, kx = e % K_DIM;
            Kp[e] = (i < ce)
                ? key_b[(unsigned long long)(cs + i) * qk_stride + kh * k_dim + kx]
                : __float2bfloat16(0.0f);
        }
        if (t == 0) {
            float dl = gc_in[base * CHUNK + ce - 1];
            decs[0] = expf(dl);
            for (unsigned int i = 0; i < ce; i++) decs[1 + i] = expf(dl - gc_in[base * CHUNK + i]);
        }
        __syncthreads();

        #pragma unroll
        for (int kk = 0; kk < KH; kk++)
            S_out[base * K_DIM * V_DIM + (k0 + kk) * V_DIM + v] = __float2bfloat16(Sreg[kk]);

        const float edl = decs[0];
        float duc[CHUNK];
        for (unsigned int i = 0; i < ce; i++) {
            float wsp = 0.0f;
            #pragma unroll
            for (int kk = 0; kk < KH; kk++)
                wsp += (float)Wp[i * K_DIM + k0 + kk] * Sreg[kk];
            #pragma unroll
            for (int s = 1; s < SPLIT; s <<= 1) wsp += __shfl_xor_sync(0xffffffffu, wsp, s);
            float uci = (float)Up[i * DVB + vloc] - wsp;
            if (sub == 0) uc_out[base * CHUNK * V_DIM + i * v_dim + v] = __float2bfloat16(uci);
            duc[i] = decs[1 + i] * uci;
        }
        #pragma unroll
        for (int kk = 0; kk < KH; kk++) {
            float hv = edl * Sreg[kk];
            for (unsigned int i = 0; i < ce; i++)
                hv += duc[i] * (float)Kp[i * K_DIM + k0 + kk];
            Sreg[kk] = hv;
        }
    }

    #pragma unroll
    for (int kk = 0; kk < KH; kk++) H[(k0 + kk) * V_DIM + v] = Sreg[kk];
}

// 2026-09-25: gated_delta_rule_chunk_delta_h_dvsplit = cdh_dvsplit_core<4, 64>: 256 threads
// (64 columns x SPLIT 4). __launch_bounds__(256, 2) limits it to 128 registers per thread
// (65,536 / (2 * 256)) so that two blocks can be resident on an SM.
extern "C" __global__ void __launch_bounds__(256, 2)
gated_delta_rule_chunk_delta_h_dvsplit(
    float* __restrict__ h_state, const __nv_bfloat16* __restrict__ W_in,
    const __nv_bfloat16* __restrict__ U_in, const __nv_bfloat16* __restrict__ key,
    const float* __restrict__ gate, const float* __restrict__ gc_in,
    __nv_bfloat16* __restrict__ S_out, __nv_bfloat16* __restrict__ uc_out,
    unsigned int batch_size, unsigned int seq_len, unsigned int num_chunks,
    unsigned int num_k_heads, unsigned int num_v_heads, unsigned int k_dim,
    unsigned int v_dim, unsigned int qk_stride, unsigned int gb_stride,
    unsigned int h_state_is_table,
    const int* __restrict__ cu_seqlens, const int* __restrict__ cu_chunks,
    unsigned int is_varlen
) {
    cdh_dvsplit_core<4, 64>(h_state, W_in, U_in, key, gate, gc_in, S_out, uc_out, seq_len,
                            num_chunks, num_k_heads, num_v_heads, k_dim, v_dim, qk_stride,
                            gb_stride, h_state_is_table, cu_seqlens, cu_chunks, is_varlen);
}

// 2026-09-25: gated_delta_rule_chunk_delta_h_tc_vblock: W S_c on tensor cores (mma_gram) with
// the value columns split into NUM_DV_BLK blocks of DV_BLK: grid (num_v_heads, batch_size *
// NUM_DV_BLK), blockIdx.y = dv_blk * batch_size + b. 256 threads = 64 columns x 4 strips of 32
// rows held in registers; mma_gram does the full K sum, so there is no shuffle. Warps 4-7 skip
// mma_gram, which covers 64 rows. W and the block's U columns are double-buffered by cp.async; K
// reuses the S^T region. The FP32 state stays in registers; S^T is a BF16 copy per chunk.
// Dynamic shared memory: 82,952 B. Selected by METRALE_GDN_TC_VBLOCK=1 when the fused spine is not.





















#define DV_BLK 64
#define NUM_DV_BLK (V_DIM / DV_BLK)

#define TCVB_SLOT (CHUNK * K_DIM + CHUNK * DV_BLK)

extern "C" __global__ void __launch_bounds__(256, 1)
gated_delta_rule_chunk_delta_h_tc_vblock(
    float* __restrict__ h_state,
    const __nv_bfloat16* __restrict__ W_in,
    const __nv_bfloat16* __restrict__ U_in,
    const __nv_bfloat16* __restrict__ key,
    const float* __restrict__ gate,
    const float* __restrict__ gc_in,
    __nv_bfloat16* __restrict__ S_out,
    __nv_bfloat16* __restrict__ uc_out,
    unsigned int batch_size, unsigned int seq_len, unsigned int num_chunks,
    unsigned int num_k_heads, unsigned int num_v_heads, unsigned int k_dim,
    unsigned int v_dim, unsigned int qk_stride, unsigned int gb_stride,
    unsigned int h_state_is_table,
    const int* __restrict__ cu_seqlens, const int* __restrict__ cu_chunks,
    unsigned int is_varlen
) {
    const unsigned int vh = blockIdx.x;
    if (vh >= num_v_heads) return;
    const unsigned int dv_blk = blockIdx.y / batch_size;
    const unsigned int b      = blockIdx.y % batch_size;
    const unsigned int dv_off = dv_blk * DV_BLK;
    GDN_GEOM(g);
    key += g.tokoff * qk_stride;



    const unsigned int tid = threadIdx.x;
    const unsigned int head_repeat = num_v_heads / num_k_heads;
    const unsigned int kh = vh / head_repeat;


    extern __shared__ char smem_raw[];
    __nv_bfloat16* St  = (__nv_bfloat16*)smem_raw;
    float*         ws  = (float*)(St + DV_BLK * K_DIM);
    __nv_bfloat16* buf = (__nv_bfloat16*)(ws + CHUNK * DV_BLK);
    float*         gcb = (float*)(buf + 2 * TCVB_SLOT);
    float*         decb = gcb + 2 * CHUNK;
    __nv_bfloat16* Kb  = St;





    constexpr int KSPLIT = 256 / DV_BLK;
    constexpr int KH     = K_DIM / KSPLIT;
    const unsigned int vloc = tid / KSPLIT;
    const unsigned int ksub = tid % KSPLIT;
    const unsigned int k0   = ksub * KH;
    const unsigned int v    = dv_off + vloc;

    float* H = h_state_is_table
        ? ((float* const*)h_state)[b] + (unsigned long long)vh * K_DIM * V_DIM
        : h_state + ((unsigned long long)(b * num_v_heads + vh) * K_DIM * V_DIM);
    float Sreg[KH];
    #pragma unroll
    for (int kk = 0; kk < KH; kk++) Sreg[kk] = H[(k0 + kk) * V_DIM + v];

    // 2026-09-25: A local prefetch instead of cdh_prefetch, because U is staged as this block's
    // DV_BLK columns only.

    auto prefetch = [&](unsigned int p, unsigned int c) {
        __nv_bfloat16* Wp = buf + (unsigned long long)p * TCVB_SLOT;
        __nv_bfloat16* Up = Wp + CHUNK * K_DIM;
        const unsigned int cs = c * CHUNK;
        const unsigned int ce = (g.seqlen - cs) < CHUNK ? (g.seqlen - cs) : CHUNK;
        const unsigned long long base = ((unsigned long long)(g.choff + c) * num_v_heads + vh);

        for (unsigned int idx = tid * 8; idx < CHUNK * K_DIM; idx += 256 * 8) {
            unsigned int i = idx / K_DIM, k = idx % K_DIM;
            if (i < ce) cp_async16(&Wp[i * K_DIM + k], &W_in[base * CHUNK * K_DIM + i * k_dim + k]);
        }

        for (unsigned int idx = tid * 8; idx < CHUNK * DV_BLK; idx += 256 * 8) {
            unsigned int i = idx / DV_BLK, vv = idx % DV_BLK;
            if (i < ce) cp_async16(&Up[i * DV_BLK + vv],
                                   &U_in[base * CHUNK * V_DIM + i * v_dim + (dv_off + vv)]);
        }
        cp_commit();
        if (tid == 0) {
            float* gcp = gcb + p * CHUNK; float* dp = decb + p * (CHUNK + 1);
            float gl = gc_in[base * CHUNK + (ce - 1)];
            dp[0] = expf(gl);
            for (unsigned int i = 0; i < ce; i++) {
                float gi = gc_in[base * CHUNK + i];
                gcp[i] = gi; dp[1 + i] = expf(gl - gi);
            }
        }
    };

    prefetch(0, 0);

    for (unsigned int c = 0; c < g.nchunks; c++) {
        const unsigned int cur = c & 1u;
        const unsigned int cs  = c * CHUNK;
        const unsigned int ce  = (g.seqlen - cs) < CHUNK ? (g.seqlen - cs) : CHUNK;
        const unsigned long long base = ((unsigned long long)(g.choff + c) * num_v_heads + vh);

        if (c + 1 < g.nchunks) { prefetch((c + 1) & 1u, c + 1); cp_wait<1>(); }
        else                   { cp_wait<0>(); }
        __syncthreads();

        __nv_bfloat16* Wp = buf + (unsigned long long)cur * TCVB_SLOT;
        __nv_bfloat16* Up = Wp + CHUNK * K_DIM;
        const float*   dec = decb + cur * (CHUNK + 1);
        const float    edl = dec[0];


        #pragma unroll
        for (int kk = 0; kk < KH; kk++)
            S_out[base * K_DIM * V_DIM + (k0 + kk) * V_DIM + v] = __float2bfloat16(Sreg[kk]);



        #pragma unroll
        for (int kk = 0; kk < KH; kk++)
            St[vloc * K_DIM + (k0 + kk)] = __float2bfloat16(Sreg[kk]);
        __syncthreads();

        // 2026-09-25: mma_gram has no barrier inside, so calling it from warps 0-3 only is safe.




        if ((tid >> 5) < 4)
            mma_gram<DV_BLK / 8, DV_BLK, false>(Wp, St, ws);
        __syncthreads();


        float duc[CHUNK];
        for (unsigned int i = 0; i < ce; i++) {
            float uci = (float)Up[i * DV_BLK + vloc] - ws[i * DV_BLK + vloc];
            if (ksub == 0)
                uc_out[base * CHUNK * V_DIM + i * v_dim + v] = __float2bfloat16(uci);
            duc[i] = dec[1 + i] * uci;
        }
        __syncthreads();   // 2026-09-25: the St region is reused for K.


        for (unsigned int idx = tid * 8; idx < CHUNK * K_DIM; idx += 256 * 8) {
            unsigned int i = idx / K_DIM, k = idx % K_DIM;
            #pragma unroll
            for (int j = 0; j < 8; j++)
                Kb[i * K_DIM + k + j] = (i < ce)
                    ? key[(unsigned long long)(cs + i) * qk_stride + kh * k_dim + (k + j)]
                    : __float2bfloat16(0.0f);
        }
        __syncthreads();


        #pragma unroll
        for (int kk = 0; kk < KH; kk++) {
            float hv = edl * Sreg[kk];
            for (unsigned int i = 0; i < ce; i++)
                hv += duc[i] * (float)Kb[i * K_DIM + (k0 + kk)];
            Sreg[kk] = hv;
        }
        __syncthreads();   // 2026-09-25: St, ws and buf are refilled next chunk.
    }

    #pragma unroll
    for (int kk = 0; kk < KH; kk++) H[(k0 + kk) * V_DIM + v] = Sreg[kk];
}

// 2026-09-25: cdh_ksplit_vblock_core<SPLIT, VTILES>: the ksplit spine with the value columns
// split into VTILES blocks, grid (num_v_heads, VTILES, batch_size). Each block still prefetches
// the whole W, U and K chunk and uses 1/VTILES of U. Measured 2026-06-25 with
// gdn_cdh_vblock_microtest on GB10: 0.71x, 0.65x and 0.34x of ksplit at batch 1 for VTILES 2, 4
// and 8. Only that microtest launches it.


























template <int SPLIT, int VTILES>
__device__ __forceinline__ void cdh_ksplit_vblock_core(
    float* __restrict__ h_state, const __nv_bfloat16* __restrict__ W_in,
    const __nv_bfloat16* __restrict__ U_in, const __nv_bfloat16* __restrict__ key,
    const float* __restrict__ gate, const float* __restrict__ gc_in,
    __nv_bfloat16* __restrict__ S_out, __nv_bfloat16* __restrict__ uc_out,
    unsigned int seq_len, unsigned int num_chunks, unsigned int num_k_heads,
    unsigned int num_v_heads, unsigned int k_dim, unsigned int v_dim,
    unsigned int qk_stride, unsigned int gb_stride, unsigned int h_state_is_table,
    const int* __restrict__ cu_seqlens, const int* __restrict__ cu_chunks,
    unsigned int is_varlen
) {
    constexpr int KH = K_DIM / SPLIT;
    constexpr int VW = V_DIM / VTILES;
    const unsigned int vh = blockIdx.x;
    const unsigned int vtile = blockIdx.y;
    const unsigned int b = blockIdx.z;
    if (vh >= num_v_heads) return;
    GDN_GEOM(g);
    const unsigned int t = threadIdx.x;
    const unsigned int v = vtile * VW + (t / SPLIT);
    const unsigned int sub = t % SPLIT;
    const unsigned int k0 = sub * KH;
    const unsigned int head_repeat = num_v_heads / num_k_heads;
    const unsigned int kh = vh / head_repeat;




    extern __shared__ char smem_raw_dhc[];
    __nv_bfloat16* buf = (__nv_bfloat16*)smem_raw_dhc;
    float* gcb = (float*)(buf + 2 * CDH_BUFSZ);
    float* decb = gcb + 2 * CHUNK;

    float* H = h_state_is_table
        ? ((float* const*)h_state)[b] + (unsigned long long)vh * K_DIM * V_DIM
        : h_state + ((unsigned long long)(b * num_v_heads + vh) * K_DIM * V_DIM);
    float Sreg[KH];
    #pragma unroll
    for (int kk = 0; kk < KH; kk++) Sreg[kk] = H[(k0 + kk) * V_DIM + v];

    cdh_prefetch(buf, gcb, decb, 0, W_in, U_in, key, gate, gc_in, 0, b, vh, seq_len,
                 num_chunks, num_v_heads, k_dim, kh, qk_stride, gb_stride,
                 cu_seqlens, cu_chunks, is_varlen);

    for (unsigned int c = 0; c < g.nchunks; c++) {
        const unsigned int cur = c & 1u;
        const unsigned int cs = c * CHUNK;
        const unsigned int ce = (g.seqlen - cs) < CHUNK ? (g.seqlen - cs) : CHUNK;
        const unsigned long long base = ((unsigned long long)(g.choff + c) * num_v_heads + vh);

        if (c + 1 < g.nchunks) {
            cdh_prefetch(buf, gcb, decb, (c + 1) & 1u, W_in, U_in, key, gate, gc_in, c + 1, b, vh, seq_len,
                         num_chunks, num_v_heads, k_dim, kh, qk_stride, gb_stride,
                         cu_seqlens, cu_chunks, is_varlen);
            cp_wait<1>();
        } else {
            cp_wait<0>();
        }
        __syncthreads();

        __nv_bfloat16* Wp = buf + (unsigned long long)cur * CDH_BUFSZ;
        __nv_bfloat16* Kp = Wp + CHUNK * K_DIM;
        __nv_bfloat16* Up = Kp + CHUNK * K_DIM;
        const float* dec = decb + cur * (CHUNK + 1);

        #pragma unroll
        for (int kk = 0; kk < KH; kk++)
            S_out[base * K_DIM * V_DIM + (k0 + kk) * V_DIM + v] = __float2bfloat16(Sreg[kk]);

        const float edl = dec[0];
        float duc[CHUNK];
        for (unsigned int i = 0; i < ce; i++) {
            float wsp = 0.0f;
            #pragma unroll
            for (int kk = 0; kk < KH; kk++)
                wsp += (float)Wp[i * K_DIM + k0 + kk] * Sreg[kk];
            #pragma unroll
            for (int s = 1; s < SPLIT; s <<= 1) wsp += __shfl_xor_sync(0xffffffffu, wsp, s);
            float uci = (float)Up[i * V_DIM + v] - wsp;
            if (sub == 0) uc_out[base * CHUNK * V_DIM + i * v_dim + v] = __float2bfloat16(uci);
            duc[i] = dec[1 + i] * uci;
        }
        #pragma unroll
        for (int kk = 0; kk < KH; kk++) {
            float hv = edl * Sreg[kk];
            for (unsigned int i = 0; i < ce; i++)
                hv += duc[i] * (float)Kp[i * K_DIM + k0 + kk];
            Sreg[kk] = hv;
        }
        __syncthreads();
    }

    #pragma unroll
    for (int kk = 0; kk < KH; kk++) H[(k0 + kk) * V_DIM + v] = Sreg[kk];
}

// 2026-09-25: Block = (V_DIM / VTILES) * SPLIT threads (128, 64, 32 for VTILES 2, 4, 8), grid
// (num_v_heads, VTILES, batch_size); dynamic shared memory as ksplit (99,336 B).
extern "C" __global__ void __launch_bounds__(128, 2)
gated_delta_rule_chunk_delta_h_ksplit_vblock2(
    float* __restrict__ h_state, const __nv_bfloat16* __restrict__ W_in,
    const __nv_bfloat16* __restrict__ U_in, const __nv_bfloat16* __restrict__ key,
    const float* __restrict__ gate, const float* __restrict__ gc_in,
    __nv_bfloat16* __restrict__ S_out, __nv_bfloat16* __restrict__ uc_out,
    unsigned int batch_size, unsigned int seq_len, unsigned int num_chunks,
    unsigned int num_k_heads, unsigned int num_v_heads, unsigned int k_dim,
    unsigned int v_dim, unsigned int qk_stride, unsigned int gb_stride,
    unsigned int h_state_is_table,
    const int* __restrict__ cu_seqlens, const int* __restrict__ cu_chunks,
    unsigned int is_varlen
) {
    cdh_ksplit_vblock_core<2, 2>(h_state, W_in, U_in, key, gate, gc_in, S_out, uc_out, seq_len,
        num_chunks, num_k_heads, num_v_heads, k_dim, v_dim, qk_stride, gb_stride,
        h_state_is_table, cu_seqlens, cu_chunks, is_varlen);
}
extern "C" __global__ void __launch_bounds__(64, 4)
gated_delta_rule_chunk_delta_h_ksplit_vblock4(
    float* __restrict__ h_state, const __nv_bfloat16* __restrict__ W_in,
    const __nv_bfloat16* __restrict__ U_in, const __nv_bfloat16* __restrict__ key,
    const float* __restrict__ gate, const float* __restrict__ gc_in,
    __nv_bfloat16* __restrict__ S_out, __nv_bfloat16* __restrict__ uc_out,
    unsigned int batch_size, unsigned int seq_len, unsigned int num_chunks,
    unsigned int num_k_heads, unsigned int num_v_heads, unsigned int k_dim,
    unsigned int v_dim, unsigned int qk_stride, unsigned int gb_stride,
    unsigned int h_state_is_table,
    const int* __restrict__ cu_seqlens, const int* __restrict__ cu_chunks,
    unsigned int is_varlen
) {
    cdh_ksplit_vblock_core<2, 4>(h_state, W_in, U_in, key, gate, gc_in, S_out, uc_out, seq_len,
        num_chunks, num_k_heads, num_v_heads, k_dim, v_dim, qk_stride, gb_stride,
        h_state_is_table, cu_seqlens, cu_chunks, is_varlen);
}
extern "C" __global__ void __launch_bounds__(32, 8)
gated_delta_rule_chunk_delta_h_ksplit_vblock8(
    float* __restrict__ h_state, const __nv_bfloat16* __restrict__ W_in,
    const __nv_bfloat16* __restrict__ U_in, const __nv_bfloat16* __restrict__ key,
    const float* __restrict__ gate, const float* __restrict__ gc_in,
    __nv_bfloat16* __restrict__ S_out, __nv_bfloat16* __restrict__ uc_out,
    unsigned int batch_size, unsigned int seq_len, unsigned int num_chunks,
    unsigned int num_k_heads, unsigned int num_v_heads, unsigned int k_dim,
    unsigned int v_dim, unsigned int qk_stride, unsigned int gb_stride,
    unsigned int h_state_is_table,
    const int* __restrict__ cu_seqlens, const int* __restrict__ cu_chunks,
    unsigned int is_varlen
) {
    cdh_ksplit_vblock_core<2, 8>(h_state, W_in, U_in, key, gate, gc_in, S_out, uc_out, seq_len,
        num_chunks, num_k_heads, num_v_heads, k_dim, v_dim, qk_stride, gb_stride,
        h_state_is_table, cu_seqlens, cu_chunks, is_varlen);
}

// 2026-09-25: Kernel 3, gated_delta_rule_chunk_fwd_o: grid (num_chunks, num_v_heads,
// batch_size), 512 threads, one block per (chunk, head). For token i and value column v:
//   O_i[v] = (exp(gc_i) <q_i, S_c[:, v]> + sum_{l <= i} exp(gc_i - gc_l) <q_i, k_l> uc_l[v]) * rsqrt(k_dim).
// Both inner products are mma_gram calls on warps 0-3: kq = Q K^T, then decayed for l <= i, and
// o1 = Q S_c into the sk region, which is free after the first call. All threads stage the
// inputs; threads 0-127 write the outputs. o1 is rounded to BF16 and reaches only the output.
// Token t of sequence b goes to output row (tokoff + t) * num_v_heads + vh, the layout of
// gated_delta_rule_prefill_persistent_wy4. Dynamic shared memory: 98,816 B.






extern "C" __global__ void __launch_bounds__(512, 1)
gated_delta_rule_chunk_fwd_o(
    const __nv_bfloat16* __restrict__ query,
    const __nv_bfloat16* __restrict__ key,
    const float* __restrict__ gate,
    const float* __restrict__ gc_in,
    const __nv_bfloat16* __restrict__ S_in,
    const __nv_bfloat16* __restrict__ uc_in,
    __nv_bfloat16* __restrict__ output,
    unsigned int batch_size,
    unsigned int seq_len,
    unsigned int num_chunks,
    unsigned int num_k_heads,
    unsigned int num_v_heads,
    unsigned int k_dim,
    unsigned int v_dim,
    unsigned int qk_stride,
    unsigned int gb_stride,
    const int* __restrict__ cu_seqlens,
    const int* __restrict__ cu_chunks,
    unsigned int is_varlen
) {
    const unsigned int c = blockIdx.x;
    const unsigned int vh = blockIdx.y;
    const unsigned int b = blockIdx.z;
    if (vh >= num_v_heads || b >= batch_size) return;
    GDN_GEOM(g);
    if (c >= g.nchunks) return;
    const unsigned int tid = threadIdx.x;
    const unsigned int head_repeat = num_v_heads / num_k_heads;
    const unsigned int kh = vh / head_repeat;
    const float inv_sqrt_d = rsqrtf((float)k_dim);
    const unsigned int cs = c * CHUNK;
    const unsigned int ce = (g.seqlen - cs) < CHUNK ? (g.seqlen - cs) : CHUNK;
    const unsigned long long base = ((unsigned long long)(g.choff + c) * num_v_heads + vh);
    const unsigned long long out_base = (g.tokoff * num_v_heads + vh) * v_dim;

    query += g.tokoff * qk_stride;
    key   += g.tokoff * qk_stride;

    extern __shared__ char smem_raw[];
    __nv_bfloat16* sq = (__nv_bfloat16*)smem_raw;
    __nv_bfloat16* sk = sq + CHUNK * K_DIM;
    float* kq = (float*)(sk + CHUNK * K_DIM);
    __nv_bfloat16* ucb = (__nv_bfloat16*)(kq + CHUNK * CHUNK);
    __nv_bfloat16* Sb = ucb + CHUNK * V_DIM;
    float* gc = (float*)(Sb + K_DIM * V_DIM);
    float* egc = gc + CHUNK;

    for (unsigned int idx = tid; idx < CHUNK * k_dim; idx += blockDim.x) {
        unsigned int i = idx / k_dim, j = idx % k_dim;
        if (i < ce) {
            unsigned long long off = (unsigned long long)(cs + i) * qk_stride + kh * k_dim + j;
            sq[i * K_DIM + j] = query[off];
            sk[i * K_DIM + j] = key[off];
        } else {
            sq[i * K_DIM + j] = __float2bfloat16(0.0f);
            sk[i * K_DIM + j] = __float2bfloat16(0.0f);
        }
    }
    for (unsigned int idx = tid; idx < CHUNK * v_dim; idx += blockDim.x) {
        unsigned int i = idx / v_dim, v = idx % v_dim;
        ucb[i * V_DIM + v] = (i < ce) ? uc_in[base * CHUNK * V_DIM + i * v_dim + v] : __float2bfloat16(0.0f);
    }
    // 2026-09-25: Sb holds S_c transposed, Sb[v][k] = S_c[k][v], so mma_gram(sq, Sb) gives <q_i, S_c[:, v]>.
    for (unsigned int idx = tid; idx < K_DIM * V_DIM; idx += blockDim.x) {
        unsigned int v = idx / K_DIM, k = idx % K_DIM;
        Sb[idx] = S_in[base * K_DIM * V_DIM + k * V_DIM + v];
    }
    for (unsigned int i = tid; i < ce; i += blockDim.x) {
        float g = gc_in[base * CHUNK + i];
        gc[i] = g;
        egc[i] = expf(g);
    }
    __syncthreads();

    if (tid < 128) mma_gram<8, CHUNK, false>(sq, sk, kq);
    __syncthreads();



    for (unsigned int p = tid; p < CHUNK * CHUNK; p += blockDim.x) {
        unsigned int i = p / CHUNK, l = p % CHUNK;
        if (i < ce && l <= i) kq[p] = expf(gc[i] - gc[l]) * kq[p];
    }
    __syncthreads();   // 2026-09-25: sk is reused for o1 below.


    __nv_bfloat16* o1 = sk;
    if (tid < 128) mma_gram<16, V_DIM, true>(sq, Sb, o1);
    __syncthreads();

    if (tid < v_dim) {
        for (unsigned int i = 0; i < ce; i++) {
            float t1 = egc[i] * (float)o1[i * V_DIM + tid];
            float t2 = 0.0f;
            for (unsigned int l = 0; l <= i; l++)
                t2 += kq[i * CHUNK + l] * (float)ucb[l * V_DIM + tid];
            output[out_base + (unsigned long long)(cs + i) * num_v_heads * v_dim + tid] =
                __float2bfloat16((t1 + t2) * inv_sqrt_d);
        }
    }
}
