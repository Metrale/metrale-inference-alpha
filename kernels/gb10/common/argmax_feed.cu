// SPDX-License-Identifier: MIT OR Apache-2.0

// 2026-09-25: The device token feed kernels: a batched argmax that writes its
// answers into the feed cells, and the resolve step that turns each row's
// source word into a token id. The model side is model-engine
// model/trait_impl/feed.rs; crates/sampling/src/feed_argmax.rs is the host
// specification of argmax_bf16_batch_feed.
//
// Owner: gb10 kernels.
// Invariants:
// - argmax_bf16_batch_feed reduces through 1024-entry shared arrays with a
//   halving tree, so blockDim.x must be a power of two no larger than 1024;
//   the host wrapper (model-layers layers/ops/sampling.rs) launches one
//   1024-thread block per row.
//
// For row r with mask ids (m0, m1), where 0xFFFFFFFF means no id:
//
//   p = the index argmax_bf16_batch returns for the row: each thread keeps
//       its first strict maximum above -1e30 and a tie between threads goes
//       to the lower thread, so a row whose maximum is at or below -1e30
//       answers 0;
//   q = the argmax of the row with m0 and m1 replaced by -inf, the higher
//       index winning ties. That is the host re-pick: PostCloseThinkMask
//       writes -inf to the two ids, and the temperature-0 sampler's `max_by`
//       returns the last of equal maxima;
//   cell[r] = (p == m0 || p == m1) ? q : p.
//
// A NaN never wins either reduction: both compare with `>`.










#include <cuda_bf16.h>
#include <math_constants.h>

#define FEED_NO_ID 0xFFFFFFFFu

extern "C" __global__ void argmax_bf16_batch_feed(
    const __nv_bfloat16* __restrict__ logits,
    const unsigned int* __restrict__ masks,   // 2026-09-25: [n_rows * 2], (m0, m1) per row
    unsigned int* __restrict__ cells,         // 2026-09-25: [n_rows], the feed cells
    unsigned int n,
    unsigned int row_stride
) {
    __shared__ float s_pval[1024];
    __shared__ unsigned int s_pidx[1024];
    __shared__ float s_qval[1024];
    __shared__ unsigned int s_qidx[1024];

    const unsigned int row = blockIdx.x;
    const __nv_bfloat16* __restrict__ row_logits =
        logits + (unsigned long long)row * (unsigned long long)row_stride;
    const unsigned int m0 = masks[row * 2];
    const unsigned int m1 = masks[row * 2 + 1];

    const unsigned int tid = threadIdx.x;
    const unsigned int stride = blockDim.x;

    // 2026-09-25: p: argmax_bf16_batch's scan.
    float p_max = -1e30f;
    unsigned int p_idx = 0;
    // 2026-09-25: q: the masked re-pick; the higher index wins ties.
    float q_max = -CUDART_INF_F;
    unsigned int q_idx = 0;
    for (unsigned int i = tid; i < n; i += stride) {
        const float v = __bfloat162float(row_logits[i]);
        if (v > p_max) {
            p_max = v;
            p_idx = i;
        }
        const float w = (i == m0 || i == m1) ? -CUDART_INF_F : v;
        if (w > q_max || (w == q_max && i > q_idx)) {
            q_max = w;
            q_idx = i;
        }
    }

    s_pval[tid] = p_max;
    s_pidx[tid] = p_idx;
    s_qval[tid] = q_max;
    s_qidx[tid] = q_idx;
    __syncthreads();

    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            // 2026-09-25: p: a tie goes to the lower thread, as in argmax_bf16_batch.
            if (s_pval[tid + s] > s_pval[tid]) {
                s_pval[tid] = s_pval[tid + s];
                s_pidx[tid] = s_pidx[tid + s];
            }
            // 2026-09-25: q: a tie goes to the higher index, the host's last-index rule.
            const float ov = s_qval[tid + s];
            if (ov > s_qval[tid] || (ov == s_qval[tid] && s_qidx[tid + s] > s_qidx[tid])) {
                s_qval[tid] = ov;
                s_qidx[tid] = s_qidx[tid + s];
            }
        }
        __syncthreads();
    }

    if (tid == 0) {
        const unsigned int p = s_pidx[0];
        cells[row] = (p == m0 || p == m1) ? s_qidx[0] : p;
    }
}

// 2026-09-25: The source word of a row: bit 31 set = an inline host id (low 31
// bits), clear = the index of the previous step's feed cell to read.
#define FEED_HOST_BIT 0x80000000u

extern "C" __global__ void feed_resolve(
    const unsigned int* __restrict__ sources,  // 2026-09-25: [n]
    const unsigned int* __restrict__ cells,
    unsigned int* __restrict__ ids_out,        // 2026-09-25: [n]
    unsigned int n
) {
    for (unsigned int i = blockIdx.x * blockDim.x + threadIdx.x; i < n;
         i += gridDim.x * blockDim.x) {
        const unsigned int s = sources[i];
        ids_out[i] = (s & FEED_HOST_BIT) ? (s & ~FEED_HOST_BIT) : cells[s];
    }
}
