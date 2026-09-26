// SPDX-License-Identifier: MIT OR Apache-2.0

// 2026-09-25: DSA (DeepSeek Sparse Attention) indexer and NoPE MLA kernels for GLM-5.3-Flash.
// forked-from: kernels/gb10/common/dsa_indexer.cu (2026-09-24; 70 of 627 lines differ, see kernels/FORKS.md)
//
// Pipeline, in launch order (crates/model-arch/src/glm5next_dsa/select/launch.rs):
//   dsa_kpool_compress    -> pool keys / pool token ids / pool validity
//   dsa_index_scores      -> per-(query, pool) score, with candidate masking
//   dsa_topk_pools        -> deterministic top-k over pools
//   dsa_expand_selection  -> pools -> raw token indices, tail appended, -1 padded
// dsa_topk_to_mask (per-query visibility mask) and dsa_mla_masked_attn (NoPE MLA over the
// selected tokens) are the test oracle's dense path; no serve path launches them
// (glm5next_dsa/mod.rs, MASKED_ATTN_MAX_KEYS).
//
// Owner: b300 kernels.
// Invariants:
// - -1 (DSA_INVALID) marks an invalid token or pool index. dsa_expand_selection
//   fills its whole row with -1 before a barrier and only then stores real
//   indices, so no slot of its output is left unwritten.
// - NoPE: `scale` is an argument, never derived here, and the query/key dim is
//   the full qk head dim; there is no rope section to skip or rotate.
// - dsa_kpool_compress handles KP <= 8 (its per-channel slot array is lg[8]).

#include <cuda_bf16.h>
#include <math_constants.h>
#include <float.h>
#include <limits.h>

#define DSA_INVALID (-1)

__device__ __forceinline__ float dsa_block_sum(float v, float* smem, unsigned tid, unsigned nthreads) {
    for (int off = 16; off > 0; off >>= 1) v += __shfl_down_sync(0xffffffff, v, off);
    if ((tid & 31u) == 0u) smem[tid >> 5] = v;
    __syncthreads();
    if (tid < 32u) {
        float x = (tid < ((nthreads + 31u) / 32u)) ? smem[tid] : 0.0f;
        for (int off = 16; off > 0; off >>= 1) x += __shfl_down_sync(0xffffffff, x, off);
        if (tid == 0u) smem[0] = x;
    }
    __syncthreads();
    return smem[0];
}

// 2026-09-25: 1. kpool compression. One block per pool; threads stride the channels. The
// softmax runs over the pool-slot axis, separately for each channel.
//
// Pools are formed from `first_key + p*KP + s`, so pooling starts at the first
// valid token and left padding is skipped. A pool counts only if every slot is
// valid, so a trailing partial pool never counts.



// 2026-09-25: Replay-safe geometry. Under CUDA-graph capture the scalar arguments are
// frozen, while S, the pool counts, the sort tile and select_k grow with the
// context. So each selector kernel takes `geom`, a 5-int device vector that
// dsa_write_geom writes every step from seq_len; with geom null, the scalar
// arguments are used. Blocks past the live pool count return, so the launcher
// can fix the grid at the context ceiling (glm5next_dsa/select/launch.rs,
// `DsaSelectLaunch::Ceiling`).






#define DSA_GEOM_S        0   // 2026-09-25: tokens resident in the indexer cache
#define DSA_GEOM_NPOOLS_F 1   // 2026-09-25: pools including the trailing partial one
#define DSA_GEOM_NPOOLS   2   // 2026-09-25: complete pools, the prefix everything downstream reads
#define DSA_GEOM_SELECT_K 3   // 2026-09-25: pools selected per query
#define DSA_GEOM_NP2      4   // 2026-09-25: tile width the top-k select walks the pool axis in

// 2026-09-25: One thread. Derives the selector geometry on device from `seq_len[0]`, so
// nothing about the pass is decided on the host.
extern "C" __global__ void dsa_write_geom(
    const int* __restrict__ seq_len,   // 2026-09-25: [1] tokens in the cache
    int* __restrict__ geom,            // 2026-09-25: [5] out
    unsigned int KP,
    unsigned int topk,
    unsigned int tile                  // 2026-09-25: dsa_topk_pools tile width (power of two, >= select_k)
) {
    if (threadIdx.x != 0 || blockIdx.x != 0) return;
    const int S = seq_len[0];
    const int np = S / (int)KP;
    // 2026-09-25: np2 is the top-k tile width: min(next power of two >= np, tile), at least 2.

    int np2 = 2;
    while (np2 < np && np2 < (int)tile) np2 <<= 1;
    const int cap = (int)(topk / KP);
    geom[DSA_GEOM_S] = S;
    geom[DSA_GEOM_NPOOLS_F] = (S + (int)KP - 1) / (int)KP;
    geom[DSA_GEOM_NPOOLS] = np;
    geom[DSA_GEOM_SELECT_K] = np < cap ? np : cap;
    geom[DSA_GEOM_NP2] = np2;
}

// 2026-09-25: Copies one staged indexer row into the cache at a device-side position
// (`pos[0]`) and marks it valid, so a replayed graph writes the current step's
// row.


extern "C" __global__ void dsa_indexer_store(
    const __nv_bfloat16* __restrict__ stage_k,     // 2026-09-25: [D]
    const __nv_bfloat16* __restrict__ stage_gate,  // 2026-09-25: [D]
    const int* __restrict__ pos,                   // 2026-09-25: [1] row to write
    __nv_bfloat16* __restrict__ k_normed,          // 2026-09-25: [capacity, D]
    __nv_bfloat16* __restrict__ gate,              // 2026-09-25: [capacity, D]
    unsigned char* __restrict__ valid,             // 2026-09-25: [capacity]
    unsigned int D
) {
    const size_t base = (size_t)pos[0] * D;
    for (unsigned int d = threadIdx.x; d < D; d += blockDim.x) {
        k_normed[base + d] = stage_k[d];
        gate[base + d] = stage_gate[d];
    }
    if (threadIdx.x == 0) valid[pos[0]] = 1;
}

extern "C" __global__ void dsa_kpool_compress(
    const __nv_bfloat16* __restrict__ k,      // 2026-09-25: [S, D] bf16
    const __nv_bfloat16* __restrict__ gate,   // 2026-09-25: [S, D] bf16
    const unsigned char* __restrict__ valid,  // 2026-09-25: [S]
    const float* __restrict__ ape,            // 2026-09-25: [KP, D] fp32
    float* __restrict__ pool_keys,            // 2026-09-25: [P, D] fp32
    int* __restrict__ pool_indices,           // 2026-09-25: [P, KP] int32, -1 where the slot is not real
    unsigned char* __restrict__ pool_valid,   // 2026-09-25: [P]
    unsigned int S,
    unsigned int D,
    unsigned int KP,
    int first_key,
    const int* __restrict__ geom              // 2026-09-25: [5] or null; see "Replay-safe geometry"
) {
    const unsigned int p = blockIdx.x;
    const unsigned int tid = threadIdx.x;
    if (geom) {
        S = (unsigned int)geom[DSA_GEOM_S];
        // 2026-09-25: The grid is fixed at the context ceiling under capture; this pool is not live yet.
        if (p >= (unsigned int)geom[DSA_GEOM_NPOOLS_F]) return;
    }

    // 2026-09-25: Slot bookkeeping is the same for every channel, so thread 0 owns the index/validity write.
    bool all_valid = true;
    for (unsigned int s = 0; s < KP; ++s) {
        long long raw = (long long)first_key + (long long)p * KP + s;
        bool in_range = raw >= 0 && raw < (long long)S;
        bool ok = in_range && valid[in_range ? (unsigned)raw : 0] != 0;
        all_valid &= ok;
        if (tid == 0) pool_indices[p * KP + s] = ok ? (int)raw : DSA_INVALID;
    }
    if (tid == 0) pool_valid[p] = all_valid ? 1 : 0;

    for (unsigned int d = tid; d < D; d += blockDim.x) {
        float mx = -CUDART_INF_F;
        float lg[8];
        for (unsigned int s = 0; s < KP && s < 8; ++s) {
            long long raw = (long long)first_key + (long long)p * KP + s;
            bool in_range = raw >= 0 && raw < (long long)S;
            bool ok = in_range && valid[in_range ? (unsigned)raw : 0] != 0;
            lg[s] = ok ? (__bfloat162float(gate[(size_t)raw * D + d]) + ape[s * D + d])
                       : -CUDART_INF_F;
            mx = fmaxf(mx, lg[s]);
        }
        float sum = 0.0f;
        for (unsigned int s = 0; s < KP && s < 8; ++s) {
            lg[s] = (lg[s] == -CUDART_INF_F) ? 0.0f : __expf(lg[s] - mx);
            sum += lg[s];
        }
        // 2026-09-25: A pool with no valid slot has sum 0 and gets zero keys.
        float inv = (sum > 0.0f) ? (1.0f / sum) : 0.0f;
        float acc = 0.0f;
        for (unsigned int s = 0; s < KP && s < 8; ++s) {
            long long raw = (long long)first_key + (long long)p * KP + s;
            bool in_range = raw >= 0 && raw < (long long)S;
            bool ok = in_range && valid[in_range ? (unsigned)raw : 0] != 0;
            if (ok) acc += lg[s] * inv * __bfloat162float(k[(size_t)raw * D + d]);
        }
        pool_keys[p * D + d] = acc;
    }
}

// 2026-09-25: 2. Per-(query, pool) index score. One block per (pool, query). `weights`
// must already carry the index_heads^-0.5 factor. ReLU is applied after the
// scale; relu(s*x) == s*relu(x) for s > 0, so the order matters only for a
// non-positive scale.
extern "C" __global__ void dsa_index_scores(
    const float* __restrict__ q,              // 2026-09-25: [Q, H, D] fp32
    const float* __restrict__ pool_keys,      // 2026-09-25: [P, D] fp32
    const float* __restrict__ weights,        // 2026-09-25: [Q, H] fp32, pre-scaled
    const int* __restrict__ pool_indices,     // 2026-09-25: [P, KP]
    const unsigned char* __restrict__ pool_valid,   // 2026-09-25: [P]
    const unsigned char* __restrict__ valid_keys,   // 2026-09-25: [S]
    const int* __restrict__ q_pos,            // 2026-09-25: [Q] absolute position of each query
    float* __restrict__ out,                  // 2026-09-25: [Q, P] fp32
    unsigned char* __restrict__ valid_cand,   // 2026-09-25: [Q, P]
    unsigned int Q,
    unsigned int P,
    unsigned int H,
    unsigned int D,
    unsigned int KP,
    unsigned int S,
    float scale,
    const int* __restrict__ geom              // 2026-09-25: [5] or null
) {
    const unsigned int p = blockIdx.x;
    const unsigned int r = blockIdx.y;
    const unsigned int tid = threadIdx.x;
    extern __shared__ float sh[];
    if (geom) {
        S = (unsigned int)geom[DSA_GEOM_S];
        P = (unsigned int)geom[DSA_GEOM_NPOOLS];
        // 2026-09-25: `P` is also the row stride of `out`/`valid_cand`. It may vary only
        // because the device-geometry path is decode-only (Q == 1), so every `r * P`
        // is 0; the launcher refuses Q > 1 there (glm5next_dsa/select/launch.rs).
        if (p >= P) return;
    }

    // 2026-09-25: A pool is a candidate only when it is complete and its last token is
    // visible to this query (causal, and not padding). The last token's index is clamped to [0, S-1].
    int end = pool_indices[p * KP + KP - 1];
    int end_c = end < 0 ? 0 : (end >= (int)S ? (int)S - 1 : end);
    bool vis = (end_c <= q_pos[r]) && (valid_keys[end_c] != 0);
    bool cand = (pool_valid[p] != 0) && vis;
    if (tid == 0) valid_cand[(size_t)r * P + p] = cand ? 1 : 0;
    if (!cand) {
        if (tid == 0) out[(size_t)r * P + p] = -FLT_MAX;
        return;
    }

    float acc = 0.0f;
    for (unsigned int h = 0; h < H; ++h) {
        float dot = 0.0f;
        for (unsigned int d = tid; d < D; d += blockDim.x)
            dot += q[((size_t)r * H + h) * D + d] * pool_keys[(size_t)p * D + d];
        dot = dsa_block_sum(dot, sh, tid, blockDim.x);
        if (tid == 0) acc += weights[(size_t)r * H + h] * fmaxf(scale * dot, 0.0f);
        __syncthreads();
    }
    if (tid == 0) out[(size_t)r * P + p] = acc;
}

// 2026-09-25: 3. Deterministic top-k over pools. One block per query. A tiled bitonic
// select: the pool axis is walked in fixed-size tiles and a running best-`T`
// list is kept in shared memory, so shared memory does not grow with the context.
//
// The tiebreak is part of the contract: score descending, then pool index
// ascending. That comparator is a total order over unique indices, so the
// top-`select_k` prefix does not depend on how the axis is tiled.
//
// Capacity: `NP2` is the tile width. The block holds two tiles (running best
// and candidate) of [f32, i32], i.e. `16 * NP2` bytes, so NP2 <= 2048 fits the
// 49,152 B default. `select_k` must be <= `NP2`; the launcher refuses otherwise
// (glm5next_dsa/select.rs).









extern "C" __global__ void dsa_topk_pools(
    const float* __restrict__ scores,   // 2026-09-25: [Q, P]
    int* __restrict__ selected,         // 2026-09-25: [Q, select_k]
    unsigned int Q,
    unsigned int P,
    unsigned int NP2,                   // 2026-09-25: tile width: min(next_pow2(P), the launcher's tile)
    unsigned int select_k,
    const int* __restrict__ geom        // 2026-09-25: [5] or null
) {
    if (geom) {
        P = (unsigned int)geom[DSA_GEOM_NPOOLS];
        NP2 = (unsigned int)geom[DSA_GEOM_NP2];
        select_k = (unsigned int)geom[DSA_GEOM_SELECT_K];
    }
    // 2026-09-25: Dynamic shared memory is sized at the ceiling under capture; the layout and
    // the walk still use this step's `NP2` and `P`.
    extern __shared__ char raw_sh[];
    const unsigned int T = NP2;
    float* sv = (float*)raw_sh;                                  // 2026-09-25: [2T] best | candidate
    int*   si = (int*)(raw_sh + (size_t)(2 * T) * sizeof(float)); // 2026-09-25: [2T]
    const unsigned int r = blockIdx.x;
    const unsigned int tid = threadIdx.x;

    // 2026-09-25: Running best-T, sorted descending. It starts empty: the pad sorts last on both keys.
    for (unsigned int i = tid; i < T; i += blockDim.x) {
        sv[i] = -FLT_MAX;
        si[i] = INT_MAX;
    }
    __syncthreads();

#define DSA_TOPK_GT(a, b) ((sv[(a)] > sv[(b)]) || (sv[(a)] == sv[(b)] && si[(a)] < si[(b)]))
#define DSA_TOPK_SWAP(a, b)                                                                  \
    do {                                                                                     \
        float tv_ = sv[(a)]; sv[(a)] = sv[(b)]; sv[(b)] = tv_;                               \
        int   ti_ = si[(a)]; si[(a)] = si[(b)]; si[(b)] = ti_;                               \
    } while (0)

    for (unsigned int base = 0; base < P; base += T) {
        // 2026-09-25: Candidate tile into [T, 2T). Short tiles pad with the same sorts-last sentinel.
        for (unsigned int i = tid; i < T; i += blockDim.x) {
            unsigned int idx = base + i;
            sv[T + i] = (idx < P) ? scores[(size_t)r * P + idx] : -FLT_MAX;
            si[T + i] = (idx < P) ? (int)idx : INT_MAX;
        }
        __syncthreads();

        // 2026-09-25: Sort the candidate tile descending (bitonic network, offset by T).
        for (unsigned int k = 2; k <= T; k <<= 1) {
            for (unsigned int j = k >> 1; j > 0; j >>= 1) {
                for (unsigned int i = tid; i < T; i += blockDim.x) {
                    unsigned int l = i ^ j;
                    if (l > i) {
                        bool gt = DSA_TOPK_GT(T + i, T + l);
                        bool want_desc = ((i & k) == 0);
                        if (want_desc != gt) DSA_TOPK_SWAP(T + i, T + l);
                    }
                }
                __syncthreads();
            }
        }

        // 2026-09-25: Batcher half-cleaner across the two descending runs: pairing best[i] with
        // cand[T-1-i] leaves the global top-T in [0, T), bitonic but not yet sorted.
        for (unsigned int i = tid; i < T; i += blockDim.x) {
            unsigned int a = i, b = T + (T - 1 - i);
            if (!DSA_TOPK_GT(a, b)) DSA_TOPK_SWAP(a, b);
        }
        __syncthreads();

        // 2026-09-25: Bitonic merge restores descending order over [0, T) for the next tile.
        for (unsigned int j = T >> 1; j > 0; j >>= 1) {
            for (unsigned int i = tid; i < T; i += blockDim.x) {
                unsigned int l = i ^ j;
                if (l > i && !DSA_TOPK_GT(i, l)) DSA_TOPK_SWAP(i, l);
            }
            __syncthreads();
        }
    }

#undef DSA_TOPK_GT
#undef DSA_TOPK_SWAP

    for (unsigned int i = tid; i < select_k; i += blockDim.x)
        selected[(size_t)r * select_k + i] = (si[i] == INT_MAX) ? DSA_INVALID : si[i];
}

// 2026-09-25: 4. Expand selected pools into raw token indices. The row is filled with
// -1 by every thread and __syncthreads()'d before any real index is written, so
// short rows, invalid pools and a missing tail all leave the sentinel.

extern "C" __global__ void dsa_expand_selection(
    const int* __restrict__ selected,               // 2026-09-25: [Q, select_k]
    const int* __restrict__ pool_indices,           // 2026-09-25: [P, KP]
    const unsigned char* __restrict__ valid_cand,   // 2026-09-25: [Q, P]
    const unsigned char* __restrict__ valid_keys,   // 2026-09-25: [S]
    const int* __restrict__ q_pos,                  // 2026-09-25: [Q]
    const unsigned char* __restrict__ q_mask,       // 2026-09-25: [Q]
    int* __restrict__ out,                          // 2026-09-25: [Q, width]
    unsigned int Q,
    unsigned int P,
    unsigned int KP,
    unsigned int S,
    unsigned int select_k,
    unsigned int width,
    int first_key,
    int always_tail,
    const int* __restrict__ geom                    // 2026-09-25: [5] or null
) {
    const unsigned int r = blockIdx.x;
    const unsigned int tid = threadIdx.x;
    if (geom) {
        S = (unsigned int)geom[DSA_GEOM_S];
        P = (unsigned int)geom[DSA_GEOM_NPOOLS];
        select_k = (unsigned int)geom[DSA_GEOM_SELECT_K];
    }
    int* row = out + (size_t)r * width;

    for (unsigned int i = tid; i < width; i += blockDim.x) row[i] = DSA_INVALID;
    __syncthreads();
    if (q_mask[r] == 0) return;   // 2026-09-25: a padded query selects nothing; the row stays all -1

    // 2026-09-25: Per-row `select_k`. The scalar is the pass's, planned from the cache
    // length the pass saw; a multi-row pass plans once from the group's final
    // length. Clamping to this row's own complete pool count, (q_pos[r]+1)/KP,
    // gives each row the select_k and tail slot (`base` below) that a single-row
    // pass for that row would use. `selected` keeps the pass stride; only the
    // count is per row.














    const unsigned int row_pools = (unsigned int)(q_pos[r] + 1) / KP;
    const unsigned int row_select_k = (row_pools < select_k) ? row_pools : select_k;

    for (unsigned int j = tid; j < row_select_k; j += blockDim.x) {
        int p = selected[(size_t)r * select_k + j];
        bool ok = (p >= 0) && (valid_cand[(size_t)r * P + p] != 0);
        for (unsigned int s = 0; s < KP; ++s) {
            unsigned int w = j * KP + s;
            if (w < width) row[w] = ok ? pool_indices[(size_t)p * KP + s] : DSA_INVALID;
        }
    }

    if (always_tail && tid == 0) {
        // 2026-09-25: The in-progress (incomplete) pool, as raw indices.
        int vis_count = 0;
        for (unsigned int t = 0; t < S; ++t)
            if ((int)t <= q_pos[r] && valid_keys[t] != 0) ++vis_count;
        int tail_count = vis_count % (int)KP;
        int tail_start = first_key + vis_count - tail_count;
        unsigned int base = row_select_k * KP;   // 2026-09-25: per row, see above
        for (unsigned int t = 0; t + 1 < KP; ++t) {
            long long idx = (long long)tail_start + t;
            bool ok = ((int)t < tail_count) && idx >= 0 && idx < (long long)S
                      && ((int)idx <= q_pos[r]) && valid_keys[(unsigned)idx] != 0;
            if (base + t < width) row[base + t] = ok ? (int)idx : DSA_INVALID;
        }
    }
}

// 2026-09-25: 5. Index row -> visibility mask (oracle). Duplicates collapse, so a repeated token
// is attended once. Out-of-range and -1 entries are dropped.

extern "C" __global__ void dsa_topk_to_mask(
    const int* __restrict__ topk,        // 2026-09-25: [Q, width]
    unsigned char* __restrict__ mask,    // 2026-09-25: [Q, S]
    unsigned int Q,
    unsigned int width,
    unsigned int S
) {
    const unsigned int r = blockIdx.x;
    const unsigned int tid = threadIdx.x;
    unsigned char* row = mask + (size_t)r * S;
    for (unsigned int i = tid; i < S; i += blockDim.x) row[i] = 0;
    __syncthreads();
    for (unsigned int j = tid; j < width; j += blockDim.x) {
        int i = topk[(size_t)r * width + j];
        if (i >= 0 && i < (int)S) row[i] = 1;   // 2026-09-25: idempotent: duplicates collapse
    }
}

// 2026-09-25: 6. NoPE MLA over the selected tokens (oracle). One block per (query, head).
//
// Parallelism is over keys for the scores and over dims for the accumulation,
// with the score row staged in shared memory in between. Parallelising over
// dims with a block reduction per key would cost one __syncthreads() per key.
//
// Capacity: the score row is `S` floats of dynamic shared memory, so at the
// 49,152 B default S must be <= 12,288 keys (glm5next_dsa MASKED_ATTN_MAX_KEYS).
//
// NoPE: `qd` is the full qk head dim, and `scale` is an argument, never derived here.




extern "C" __global__ void dsa_mla_masked_attn(
    const __nv_bfloat16* __restrict__ q,   // 2026-09-25: [Q, H, qd] bf16
    const __nv_bfloat16* __restrict__ k,   // 2026-09-25: [S, H, qd] bf16
    const __nv_bfloat16* __restrict__ v,   // 2026-09-25: [S, H, vd] bf16
    const unsigned char* __restrict__ mask,// 2026-09-25: [Q, S]
    float* __restrict__ out,               // 2026-09-25: [Q, H, vd] fp32
    unsigned int Q,
    unsigned int S,
    unsigned int H,
    unsigned int qd,
    unsigned int vd,
    float scale,
    // 2026-09-25: Nonzero rounds each pre-softmax score to BF16, to match a reference
    // that computes the scores in BF16.


    unsigned int round_scores_bf16
) {
    extern __shared__ float sc[];          // 2026-09-25: [S] score row
    __shared__ float red[32];
    __shared__ float s_m, s_l;
    const unsigned int r = blockIdx.x;
    const unsigned int h = blockIdx.y;
    const unsigned int tid = threadIdx.x;
    const __nv_bfloat16* qrow = q + ((size_t)r * H + h) * qd;
    const unsigned char* mrow = mask + (size_t)r * S;

    // 2026-09-25: Scores, parallel over keys.
    float local_max = -CUDART_INF_F;
    for (unsigned int t = tid; t < S; t += blockDim.x) {
        if (mrow[t] == 0) { sc[t] = -CUDART_INF_F; continue; }
        float dot = 0.0f;
        const __nv_bfloat16* krow = k + ((size_t)t * H + h) * qd;
        for (unsigned int d = 0; d < qd; ++d)
            dot += __bfloat162float(qrow[d]) * __bfloat162float(krow[d]);
        float sv = dot * scale;
        if (round_scores_bf16) sv = __bfloat162float(__float2bfloat16(sv));
        sc[t] = sv;
        local_max = fmaxf(local_max, sv);
    }
    for (int off = 16; off > 0; off >>= 1)
        local_max = fmaxf(local_max, __shfl_down_sync(0xffffffff, local_max, off));
    if ((tid & 31u) == 0u) red[tid >> 5] = local_max;
    __syncthreads();
    if (tid < 32u) {
        float x = (tid < ((blockDim.x + 31u) / 32u)) ? red[tid] : -CUDART_INF_F;
        for (int off = 16; off > 0; off >>= 1) x = fmaxf(x, __shfl_down_sync(0xffffffff, x, off));
        if (tid == 0u) s_m = x;
    }
    __syncthreads();

    // 2026-09-25: Exponentiate in place and sum.
    const float m = s_m;
    float local_sum = 0.0f;
    for (unsigned int t = tid; t < S; t += blockDim.x) {
        float e = (sc[t] == -CUDART_INF_F) ? 0.0f : __expf(sc[t] - m);
        sc[t] = e;
        local_sum += e;
    }
    local_sum = dsa_block_sum(local_sum, red, tid, blockDim.x);
    if (tid == 0) s_l = local_sum;
    __syncthreads();

    // 2026-09-25: Weighted value sum, parallel over dims.
    const float inv = (s_l > 0.0f) ? (1.0f / s_l) : 0.0f;
    for (unsigned int d = tid; d < vd; d += blockDim.x) {
        float acc = 0.0f;
        for (unsigned int t = 0; t < S; ++t) {
            float p = sc[t];
            if (p != 0.0f) acc += p * __bfloat162float(v[((size_t)t * H + h) * vd + d]);
        }
        out[((size_t)r * H + h) * vd + d] = acc * inv;
    }
}

// 2026-09-25: 1b. Pool-axis compaction: gathers the kept pools (`keep[p]` is the original
// pool id) into the compacted arrays, on device. The caller computes `keep`.







extern "C" __global__ void dsa_compact_pools(
    const float* __restrict__ keys_in,          // 2026-09-25: [P_full, D]
    const int* __restrict__ idx_in,             // 2026-09-25: [P_full, KP]
    const unsigned char* __restrict__ valid_in, // 2026-09-25: [P_full]
    const int* __restrict__ keep,               // 2026-09-25: [P_kept] original pool ids
    float* __restrict__ keys_out,               // 2026-09-25: [P_kept, D]
    int* __restrict__ idx_out,                  // 2026-09-25: [P_kept, KP]
    unsigned char* __restrict__ valid_out,      // 2026-09-25: [P_kept]
    unsigned int P_kept,
    unsigned int D,
    unsigned int KP
) {
    const unsigned int p = blockIdx.x;
    const unsigned int tid = threadIdx.x;
    const int src = keep[p];
    for (unsigned int d = tid; d < D; d += blockDim.x)
        keys_out[(size_t)p * D + d] = keys_in[(size_t)src * D + d];
    for (unsigned int s = tid; s < KP; s += blockDim.x)
        idx_out[(size_t)p * KP + s] = idx_in[(size_t)src * KP + s];
    if (tid == 0) valid_out[p] = valid_in[src];
}
