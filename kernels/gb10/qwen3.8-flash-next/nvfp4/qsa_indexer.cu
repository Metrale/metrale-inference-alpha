// SPDX-License-Identifier: MIT OR Apache-2.0

// 2026-09-25: Qwen3.8-Flash-Next QSA indexer: block pooling, scoring and selected-set attention.
// Reference: `Qwen4ExpTextQSAIndexer` in bench/qwen4_exp/ref/modeling_qwen4_exp.py.
// Per query, the visible prefix is grouped into `ratio`-token blocks; each
// block's key is the mean of its raw per-token indexer keys, then k_layernorm
// (`1 + weight` RMSNorm), then partial rope at the block's first token
// position. Scores are sum_h relu(q_h . k_b) / sqrt(head_dim); the top
// `block_topk` blocks plus the incomplete tail are the visible set.
// Launchers: crates/model-layers/src/layers/ops/qsa.rs.
//
// At decode, qsa_gather packs the selected tokens' K/V rows into a contiguous
// NHD scratch ([page, slot, kv_head, dim]) that the paged decode attention
// reads through an identity block table (layers/qsa.rs).
//
// Rope is computed inline in double precision, pairing (j, j + rot/2) with
// inv_freq_j = theta^(-2j/rot), rather than read from the attention rope tables.
//
// Owner: gb10 kernels (qwen3.8-flash-next).
// Invariants: none beyond the preconditions stated on each kernel.


#include <cuda_bf16.h>

__device__ __forceinline__ float qsa_block_reduce_sum(float v, float* red) {
    const unsigned int lane = threadIdx.x & 31u;
    const unsigned int warp = threadIdx.x >> 5;
    #pragma unroll
    for (int off = 16; off > 0; off >>= 1) {
        v += __shfl_down_sync(0xFFFFFFFFu, v, off);
    }
    if (lane == 0) red[warp] = v;
    __syncthreads();
    float tot = 0.0f;
    if (threadIdx.x == 0) {
        const unsigned int warps = (blockDim.x + 31) >> 5;
        for (unsigned int w = 0; w < warps; ++w) tot += red[w];
        red[0] = tot;
    }
    __syncthreads();
    return red[0];
}

// 2026-09-25: normed (in shared memory, length hd) -> rope at `pos` -> out (BF16).
// One thread per element d; rot must be even.
__device__ __forceinline__ void qsa_rope_store(
    const float* normed, __nv_bfloat16* out,
    unsigned int d, unsigned int rot, unsigned int pos, float theta
) {
    if (d < rot) {
        const unsigned int half = rot >> 1;
        const unsigned int j = (d < half) ? d : d - half;
        const double inv_freq = exp(-2.0 * (double)j / (double)rot * log((double)theta));
        double s, c;
        sincos((double)pos * inv_freq, &s, &c);
        const float x1 = normed[j];
        const float x2 = normed[j + half];
        const float v = (d < half) ? (x1 * (float)c - x2 * (float)s)
                                   : (x2 * (float)c + x1 * (float)s);
        out[d] = __float2bfloat16(v);
    } else {
        out[d] = __float2bfloat16(normed[d]);
    }
}

// 2026-09-25: qsa_block_pool: pool `n_new` complete blocks starting at `first_block`:
// mean(ratio raw keys) -> RMSNorm * (1 + w) -> rope at pos = block * ratio,
// written to block_keys[block]. Grid (n_new), block (hd); dynamic shared memory
// holds hd floats plus the reduction slots.
extern "C" __global__ void qsa_block_pool(
    const __nv_bfloat16* __restrict__ raw_keys,   // 2026-09-25: [S, hd]
    const __nv_bfloat16* __restrict__ k_norm_w,   // 2026-09-25: [hd]
    __nv_bfloat16* __restrict__ block_keys,       // 2026-09-25: [max_blocks, hd]
    const unsigned int first_block,
    const unsigned int ratio,
    const unsigned int hd,
    const unsigned int rot,
    const float theta,
    const float eps
) {
    const unsigned int b = first_block + blockIdx.x;
    const unsigned int d = threadIdx.x;

    extern __shared__ float smem[];
    float* stage = smem;
    float* red = smem + hd;

    float v = 0.0f;
    for (unsigned int r = 0; r < ratio; ++r) {
        v += (float)raw_keys[(size_t)(b * ratio + r) * hd + d];
    }
    v /= (float)ratio;

    const float sq = qsa_block_reduce_sum(v * v, red);
    const float rms = rsqrtf(sq / (float)hd + eps);
    stage[d] = v * rms * (1.0f + (float)k_norm_w[d]);
    __syncthreads();

    qsa_rope_store(stage, block_keys + (size_t)b * hd, d, rot, b * ratio, theta);
}

// 2026-09-25: qsa_qprep: one decode query: per head, RMSNorm * (1 + w) then rope
// at `pos`. q_in is the head-concatenated q part of the qk projection row.
// Grid (n_heads), block (hd). The output is FP32; qsa_score reads it.

extern "C" __global__ void qsa_qprep(
    const __nv_bfloat16* __restrict__ q_in,       // 2026-09-25: [n_heads, hd]
    const __nv_bfloat16* __restrict__ q_norm_w,   // 2026-09-25: [hd]
    float* __restrict__ q_out,                    // 2026-09-25: [n_heads, hd]
    const unsigned int hd,
    const unsigned int rot,
    const unsigned int pos,
    const float theta,
    const float eps
) {
    const unsigned int h = blockIdx.x;
    const unsigned int d = threadIdx.x;

    extern __shared__ float smem[];
    float* stage = smem;
    float* red = smem + hd;

    const float x = (float)q_in[(size_t)h * hd + d];
    const float sq = qsa_block_reduce_sum(x * x, red);
    const float rms = rsqrtf(sq / (float)hd + eps);
    stage[d] = x * rms * (1.0f + (float)q_norm_w[d]);
    __syncthreads();

    float* out = q_out + (size_t)h * hd;
    if (d < rot) {
        const unsigned int half = rot >> 1;
        const unsigned int j = (d < half) ? d : d - half;
        const double inv_freq = exp(-2.0 * (double)j / (double)rot * log((double)theta));
        double s, c;
        sincos((double)pos * inv_freq, &s, &c);
        const float x1 = stage[j];
        const float x2 = stage[j + half];
        out[d] = (d < half) ? (x1 * (float)c - x2 * (float)s)
                            : (x2 * (float)c + x1 * (float)s);
    } else {
        out[d] = stage[d];
    }
}

// 2026-09-25: qsa_score: scores[b] = sum_h relu(q_h . k_b) / sqrt(hd).
// Grid (n_blocks), block (hd).

extern "C" __global__ void qsa_score(
    const float* __restrict__ q,                  // 2026-09-25: [n_heads, hd]
    const __nv_bfloat16* __restrict__ block_keys, // 2026-09-25: [n_blocks, hd]
    float* __restrict__ scores,                   // 2026-09-25: [n_blocks]
    const unsigned int n_heads,
    const unsigned int hd
) {
    const unsigned int b = blockIdx.x;
    const unsigned int d = threadIdx.x;

    extern __shared__ float smem[];
    float* red = smem;

    const float k = (float)block_keys[(size_t)b * hd + d];
    float acc = 0.0f;
    for (unsigned int h = 0; h < n_heads; ++h) {
        const float dot = qsa_block_reduce_sum(q[(size_t)h * hd + d] * k, red);
        if (threadIdx.x == 0) acc += fmaxf(dot, 0.0f);
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        scores[b] = acc * rsqrtf((float)hd);
    }
}

// 2026-09-25: qsa_gather: copy the selected tokens' K/V rows (NHD paged layout)
// into contiguous scratch: dst slot i holds src position sel[i]. Through an
// identity block table the scratch reads as a paged cache.
// Grid (n_sel), block (256).

extern "C" __global__ void qsa_gather(
    const __nv_bfloat16* __restrict__ k_cache,    // 2026-09-25: [blocks, bs, nkv, hd]
    const __nv_bfloat16* __restrict__ v_cache,
    const int* __restrict__ block_table,          // 2026-09-25: logical page -> physical page
    const int* __restrict__ sel,                  // 2026-09-25: [n_sel] token positions
    __nv_bfloat16* __restrict__ k_out,            // 2026-09-25: [n_sel, nkv, hd]
    __nv_bfloat16* __restrict__ v_out,
    const unsigned int block_size,
    const unsigned int nkv,
    const unsigned int hd
) {
    const unsigned int i = blockIdx.x;
    const unsigned int pos = (unsigned int)sel[i];
    const unsigned int row = nkv * hd;
    const unsigned long long page_stride =
        (unsigned long long)block_size * row;
    const unsigned long long src_off =
        (unsigned long long)(unsigned int)block_table[pos / block_size] * page_stride
        + (unsigned long long)(pos % block_size) * row;
    const unsigned long long dst_off = (unsigned long long)i * row;
    for (unsigned int e = threadIdx.x; e < row; e += blockDim.x) {
        k_out[dst_off + e] = k_cache[src_off + e];
        v_out[dst_off + e] = v_cache[src_off + e];
    }
}


// 2026-09-25: Prefill selection. Each chunk row past the inert bound
// (`budget + ratio - 1` visible tokens, `inert_bound` in layers/qsa.rs) needs
// its own top-`block_topk` block set. Rows are processed as a contiguous range
// [first_pos, first_pos + n_rows): the row's scores are masked at its own
// complete-block count, a host top-k in layers/qsa_select.rs builds the block
// list, and qsa_prefill_attn overwrites that row's attention output with
// attention over the selected set, read from the paged KV cache.




// 2026-09-25: Per-row q prep: RMSNorm * (1 + w) and partial rope at pos = first_pos + row.
// qk rows are the indexer projection [rows, (n_heads + 1) * hd]; q is the
// head-concatenated prefix of each row. Grid (rows, n_heads), block (hd).
extern "C" __global__ void qsa_qprep_rows(
    const __nv_bfloat16* __restrict__ qk,       // 2026-09-25: [rows, qkw]
    const __nv_bfloat16* __restrict__ q_norm_w, // 2026-09-25: [hd]
    float* __restrict__ q_out,                  // 2026-09-25: [rows, n_heads, hd]
    const unsigned int first_pos,
    const unsigned int qkw,
    const unsigned int n_heads,
    const unsigned int hd,
    const unsigned int rot,
    const float theta,
    const float eps
) {
    const unsigned int r = blockIdx.x;
    const unsigned int hh = blockIdx.y;
    const unsigned int d = threadIdx.x;
    const unsigned int pos = first_pos + r;

    extern __shared__ float smem[];
    float* stage = smem;
    float* red = smem + hd;

    const float x = (float)qk[(size_t)r * qkw + (size_t)hh * hd + d];
    const float sq = qsa_block_reduce_sum(x * x, red);
    const float rms = rsqrtf(sq / (float)hd + eps);
    stage[d] = x * rms * (1.0f + (float)q_norm_w[d]);
    __syncthreads();

    float* out = q_out + ((size_t)r * n_heads + hh) * hd;
    if (d < rot) {
        const unsigned int half = rot >> 1;
        const unsigned int j = (d < half) ? d : d - half;
        const double inv_freq = exp(-2.0 * (double)j / (double)rot * log((double)theta));
        double s, c;
        sincos((double)pos * inv_freq, &s, &c);
        const float x1 = stage[j];
        const float x2 = stage[j + half];
        out[d] = (d < half) ? (x1 * (float)c - x2 * (float)s)
                            : (x2 * (float)c + x1 * (float)s);
    } else {
        out[d] = stage[d];
    }
}

// 2026-09-25: Per-row block scores: scores[r, b] = sum_h relu(q[r,h] . k_b) / sqrt(hd)
// for b < complete(row), and -1e30 otherwise, so the host top-k never picks it.
// Grid (rows, n_blocks_max), block (hd).
extern "C" __global__ void qsa_score_rows(
    const float* __restrict__ q,                // 2026-09-25: [rows, n_heads, hd]
    const __nv_bfloat16* __restrict__ block_keys,
    float* __restrict__ scores,                 // 2026-09-25: [rows, score_stride]
    const unsigned int first_pos,
    const unsigned int score_stride,
    const unsigned int ratio,
    const unsigned int n_heads,
    const unsigned int hd
) {
    const unsigned int r = blockIdx.x;
    const unsigned int b = blockIdx.y;
    const unsigned int d = threadIdx.x;
    const unsigned int complete = (first_pos + r + 1) / ratio;
    float* out = scores + (size_t)r * score_stride + b;
    if (b >= complete) {
        if (d == 0) *out = -1e30f;
        return;
    }

    extern __shared__ float smem[];
    float* red = smem;

    const float k = (float)block_keys[(size_t)b * hd + d];
    const float* qr = q + (size_t)r * n_heads * hd;
    float acc = 0.0f;
    for (unsigned int hh = 0; hh < n_heads; ++hh) {
        const float dot = qsa_block_reduce_sum(qr[(size_t)hh * hd + d] * k, red);
        if (d == 0) acc += fmaxf(dot, 0.0f);
        __syncthreads();
    }
    if (d == 0) *out = acc * rsqrtf((float)hd);
}

// 2026-09-25: qsa_prefill_attn (defined after qsa_score_rows_tc): attention over the
// selected set for one (row, q-head): the listed `topk` blocks (ratio tokens
// each) plus the incomplete tail [complete * ratio, pos]. K/V come from the
// paged cache, and the output replaces that row's context in attn_out.
// Grid (rows, nq), block (QSA_PA_WARPS * 32 = 256): warp-striped online softmax.
// hd must be a multiple of 32 and at most 256 (8 elements per lane).

#define QSA_PA_WARPS 8
// 2026-09-25: qsa_score_rows_tc: tensor-core qsa_score_rows. qsa_score_rows runs
// one block per (row, block) with one reduction per head and lane 0
// accumulating; here each head's scores are a matmul Q_h[rows, hd] x
// K^T[hd, blocks], with the ReLU applied per head before the head sum, so each
// head keeps its own accumulator:
//     scores[r,b] = rsqrt(hd) * SUM_h relu(q[r,h,:] . k[b,:])
//
// Precision: block_keys are already BF16, so they are exact as MMA operands;
// only q (FP32) would lose mantissa bits in a BF16 cast. q is split,
//     q ~ q_hi + q_lo,  q_hi = bf16(q),  q_lo = bf16(q - q_hi)
//     q.k = q_hi.k + q_lo.k        (two MMAs into the same FP32 accumulator)
// for twice the MMA work. The scores are therefore not bit-identical to
// qsa_score_rows; they feed a top-k.
//
// Needs n_heads == QSA_TC_H and hd == QSA_TC_HD (layers/qsa_select.rs checks
// both). Grid (ceil(rows/16), ceil(blocks/64)), block 256 = 8 warps, one warp
// per 8-block n-tile. It reads q and writes scores for whole 16-row tiles, so
// both buffers must hold rows rounded up to 16 (qsa_select.rs sizes them for
// 2048 rows).





#define QSA_TC_TR 16
#define QSA_TC_TB 64
#define QSA_TC_H 4
#define QSA_TC_HD 128
#define QSA_TC_APAD 8
#define QSA_TC_BPAD 8
extern "C" __global__ __launch_bounds__(256) void qsa_score_rows_tc(
    const float* __restrict__ q,                // 2026-09-25: [rows, H, HD] FP32
    const __nv_bfloat16* __restrict__ block_keys,
    float* __restrict__ scores,                 // 2026-09-25: [rows, score_stride]
    const unsigned int first_pos,
    const unsigned int score_stride,
    const unsigned int ratio,
    const unsigned int n_blocks
) {
    const int TR = QSA_TC_TR, TB = QSA_TC_TB, H = QSA_TC_H, HD = QSA_TC_HD;
    __shared__ __nv_bfloat16 sqh[QSA_TC_TR][QSA_TC_H][QSA_TC_HD + QSA_TC_APAD];
    __shared__ __nv_bfloat16 sql[QSA_TC_TR][QSA_TC_H][QSA_TC_HD + QSA_TC_APAD];
    __shared__ __nv_bfloat16 skT[QSA_TC_HD][QSA_TC_TB + QSA_TC_BPAD];

    const unsigned int r0 = blockIdx.x * TR, b0 = blockIdx.y * TB;
    const unsigned int tidx = threadIdx.x, NT = 256;

    for (unsigned int i = tidx; i < (unsigned int)(TR * H * HD); i += NT) {
        unsigned int rr = i / (H * HD), rem = i % (H * HD);
        float v = q[(size_t)(r0 + rr) * H * HD + rem];
        __nv_bfloat16 hi = __float2bfloat16(v);
        __nv_bfloat16 lo = __float2bfloat16(v - __bfloat162float(hi));
        sqh[rr][rem / HD][rem % HD] = hi;
        sql[rr][rem / HD][rem % HD] = lo;
    }
    for (unsigned int i = tidx; i < (unsigned int)(TB * HD); i += NT) {
        unsigned int bb = i / HD, dd = i % HD;
        skT[dd][bb] = (b0 + bb) < n_blocks ? block_keys[(size_t)(b0 + bb) * HD + dd]
                                           : __float2bfloat16(0.0f);
    }
    __syncthreads();

    const unsigned int warp = tidx >> 5, lane = tidx & 31u;
    const unsigned int gid = lane >> 2, tid = lane & 3u;
    float acc[QSA_TC_H][4];
#pragma unroll
    for (int h = 0; h < H; h++) { acc[h][0]=0.f; acc[h][1]=0.f; acc[h][2]=0.f; acc[h][3]=0.f; }

    const unsigned short* sB = (const unsigned short*)skT;
    const int b_stride = TB + QSA_TC_BPAD;
    const int a_row = H * (HD + QSA_TC_APAD);
#pragma unroll
    for (int h = 0; h < H; h++) {
        const unsigned short* sAh = (const unsigned short*)&sqh[0][h][0];
        const unsigned short* sAl = (const unsigned short*)&sql[0][h][0];
#pragma unroll
        for (int d0 = 0; d0 < HD; d0 += 16) {
            unsigned int fr0 = gid, fr1 = gid + 8;
            unsigned int fc0 = d0 + tid * 2, fc1 = fc0 + 8;
            unsigned int nc = warp * 8 + gid;
            unsigned int k0 = d0 + tid * 2, k1 = k0 + 8;
            unsigned int br0 = ((unsigned int)sB[(k0+1)*b_stride+nc]<<16) | (unsigned int)sB[k0*b_stride+nc];
            unsigned int br1 = ((unsigned int)sB[(k1+1)*b_stride+nc]<<16) | (unsigned int)sB[k1*b_stride+nc];
#pragma unroll
            for (int part = 0; part < 2; part++) {
                const unsigned short* sA = part ? sAl : sAh;
                unsigned int a0 = ((unsigned int)sA[fr0*a_row+fc0+1]<<16) | (unsigned int)sA[fr0*a_row+fc0];
                unsigned int a1 = ((unsigned int)sA[fr1*a_row+fc0+1]<<16) | (unsigned int)sA[fr1*a_row+fc0];
                unsigned int a2 = ((unsigned int)sA[fr0*a_row+fc1+1]<<16) | (unsigned int)sA[fr0*a_row+fc1];
                unsigned int a3 = ((unsigned int)sA[fr1*a_row+fc1+1]<<16) | (unsigned int)sA[fr1*a_row+fc1];
                asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%10,%11,%12,%13};"
                    :"=f"(acc[h][0]),"=f"(acc[h][1]),"=f"(acc[h][2]),"=f"(acc[h][3])
                    :"r"(a0),"r"(a1),"r"(a2),"r"(a3),"r"(br0),"r"(br1),
                     "f"(acc[h][0]),"f"(acc[h][1]),"f"(acc[h][2]),"f"(acc[h][3]));
            }
        }
    }

    const float inv = rsqrtf((float)HD);
#pragma unroll
    for (int part = 0; part < 2; part++) {
        unsigned int r = r0 + gid + part * 8;
        unsigned int complete = (first_pos + r + 1) / ratio;
#pragma unroll
        for (int cc = 0; cc < 2; cc++) {
            unsigned int b = b0 + warp * 8 + tid * 2 + cc;
            if (b >= n_blocks) continue;
            float sum = 0.f;
#pragma unroll
            for (int h = 0; h < H; h++) sum += fmaxf(acc[h][part*2+cc], 0.0f);
            scores[(size_t)r * score_stride + b] = (b >= complete) ? -1e30f : sum * inv;
        }
    }
}

extern "C" __global__ void qsa_prefill_attn(
    const __nv_bfloat16* __restrict__ q,        // 2026-09-25: [rows, nq, hd], roped
    const __nv_bfloat16* __restrict__ k_cache,  // 2026-09-25: paged NHD
    const __nv_bfloat16* __restrict__ v_cache,
    const int* __restrict__ block_table,
    const int* __restrict__ lists,              // 2026-09-25: [rows, topk] block ids
    __nv_bfloat16* __restrict__ attn_out,       // 2026-09-25: [rows, nq, hd]
    const unsigned int first_pos,
    const unsigned int topk,
    const unsigned int ratio,
    const unsigned int block_size,
    const unsigned int nq,
    const unsigned int nkv,
    const unsigned int hd,
    const float inv_sqrt_d
) {
    const unsigned int r = blockIdx.x;
    const unsigned int qh = blockIdx.y;
    const unsigned int lane = threadIdx.x & 31u;
    const unsigned int warp = threadIdx.x >> 5;
    const unsigned int pos = first_pos + r;
    const unsigned int complete = (pos + 1) / ratio;
    const unsigned int tail = (pos + 1) - complete * ratio;
    const unsigned int n_tok = topk * ratio + tail;
    const unsigned int kvh = qh / (nq / nkv);
    const unsigned int row_elems = nkv * hd;
    const unsigned long long page_stride = (unsigned long long)block_size * row_elems;
    const unsigned int vec = hd / 32;           // 2026-09-25: elements per lane

    extern __shared__ float smem[];
    // 2026-09-25: Per-warp partials: [warps][hd] acc, then [warps] m, [warps] l.
    float* acc_w = smem;
    float* m_w = smem + QSA_PA_WARPS * hd;
    float* l_w = m_w + QSA_PA_WARPS;


    const __nv_bfloat16* qrow = q + ((size_t)r * nq + qh) * hd;
    float qreg[8];
    #pragma unroll
    for (unsigned int e = 0; e < 8; ++e) {
        qreg[e] = (e < vec) ? (float)qrow[lane * vec + e] : 0.0f;
    }

    float m = -1e30f, l = 0.0f;
    float acc[8];
    #pragma unroll
    for (unsigned int e = 0; e < 8; ++e) acc[e] = 0.0f;

    const int* my_list = lists + (size_t)r * topk;
    for (unsigned int t = warp; t < n_tok; t += QSA_PA_WARPS) {
        unsigned int tok;
        if (t < topk * ratio) {
            tok = (unsigned int)my_list[t / ratio] * ratio + (t % ratio);
        } else {
            tok = complete * ratio + (t - topk * ratio);
        }
        const unsigned long long off =
            (unsigned long long)(unsigned int)block_table[tok / block_size] * page_stride
            + (unsigned long long)(tok % block_size) * row_elems
            + (unsigned long long)kvh * hd;
        const __nv_bfloat16* krow = k_cache + off;
        float dot = 0.0f;
        #pragma unroll
        for (unsigned int e = 0; e < 8; ++e) {
            if (e < vec) dot += qreg[e] * (float)krow[lane * vec + e];
        }
        #pragma unroll
        for (int o = 16; o > 0; o >>= 1) dot += __shfl_down_sync(0xFFFFFFFFu, dot, o);
        dot = __shfl_sync(0xFFFFFFFFu, dot, 0) * inv_sqrt_d;

        const float m_new = fmaxf(m, dot);
        const float scale = __expf(m - m_new);
        const float p = __expf(dot - m_new);
        l = l * scale + p;
        const __nv_bfloat16* vrow = v_cache + off;
        #pragma unroll
        for (unsigned int e = 0; e < 8; ++e) {
            if (e < vec) acc[e] = acc[e] * scale + p * (float)vrow[lane * vec + e];
        }
        m = m_new;
    }

    // 2026-09-25: Park the warp partials; warp 0 merges them.
    #pragma unroll
    for (unsigned int e = 0; e < 8; ++e) {
        if (e < vec) acc_w[warp * hd + lane * vec + e] = acc[e];
    }
    if (lane == 0) { m_w[warp] = m; l_w[warp] = l; }
    __syncthreads();

    if (warp == 0) {
        float m_tot = -1e30f;
        for (unsigned int w = 0; w < QSA_PA_WARPS; ++w) m_tot = fmaxf(m_tot, m_w[w]);
        float l_tot = 0.0f;
        float out[8];
        #pragma unroll
        for (unsigned int e = 0; e < 8; ++e) out[e] = 0.0f;
        for (unsigned int w = 0; w < QSA_PA_WARPS; ++w) {
            const float s = __expf(m_w[w] - m_tot);
            l_tot += l_w[w] * s;
            #pragma unroll
            for (unsigned int e = 0; e < 8; ++e) {
                if (e < vec) out[e] += acc_w[w * hd + lane * vec + e] * s;
            }
        }
        const float inv_l = (l_tot > 0.0f) ? 1.0f / l_tot : 0.0f;
        __nv_bfloat16* orow = attn_out + ((size_t)r * nq + qh) * hd;
        #pragma unroll
        for (unsigned int e = 0; e < 8; ++e) {
            if (e < vec) orow[lane * vec + e] = __float2bfloat16(out[e] * inv_l);
        }
    }
}
