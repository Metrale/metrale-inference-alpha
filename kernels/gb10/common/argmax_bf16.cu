// SPDX-License-Identifier: MIT OR Apache-2.0

// 2026-09-25: Argmax kernels over one row, or a batch of rows, of logits.
//
// Owner: gb10 kernels.
// Invariants:
// - Each block reduces through 1024-entry shared arrays with a halving tree,
//   so blockDim.x must be a power of two no larger than 1024; the host
//   wrappers (model-layers layers/ops/sampling.rs) launch 1024 threads.
// - A tie goes to the candidate held by the lower thread.

#include <cuda_bf16.h>

extern "C" __global__ void argmax_bf16(
    const __nv_bfloat16* __restrict__ logits,
    unsigned int* __restrict__ out,
    unsigned int n
) {
    __shared__ float s_val[1024];
    __shared__ unsigned int s_idx[1024];

    const unsigned int tid = threadIdx.x;
    const unsigned int stride = blockDim.x;


    float local_max = -1e30f;
    unsigned int local_idx = 0;

    for (unsigned int i = tid; i < n; i += stride) {
        float v = __bfloat162float(logits[i]);
        if (v > local_max) {
            local_max = v;
            local_idx = i;
        }
    }

    s_val[tid] = local_max;
    s_idx[tid] = local_idx;
    __syncthreads();


    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            if (s_val[tid + s] > s_val[tid]) {
                s_val[tid] = s_val[tid + s];
                s_idx[tid] = s_idx[tid + s];
            }
        }
        __syncthreads();
    }


    if (tid == 0) {
        out[0] = s_idx[0];
    }
}

// 2026-09-25: One block per row, running argmax_bf16's per-row body on
// `logits + row * row_stride`, so each row gets the index argmax_bf16 returns
// for that row, ties included. One launch replaces n single-block launches.







extern "C" __global__ void argmax_bf16_batch(
    const __nv_bfloat16* __restrict__ logits,
    unsigned int* __restrict__ out,
    unsigned int n,
    unsigned int row_stride
) {
    __shared__ float s_val[1024];
    __shared__ unsigned int s_idx[1024];

    const unsigned int row = blockIdx.x;
    const __nv_bfloat16* __restrict__ row_logits =
        logits + (unsigned long long)row * (unsigned long long)row_stride;

    const unsigned int tid = threadIdx.x;
    const unsigned int stride = blockDim.x;

    float local_max = -1e30f;
    unsigned int local_idx = 0;
    for (unsigned int i = tid; i < n; i += stride) {
        float v = __bfloat162float(row_logits[i]);
        if (v > local_max) {
            local_max = v;
            local_idx = i;
        }
    }

    s_val[tid] = local_max;
    s_idx[tid] = local_idx;
    __syncthreads();

    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            if (s_val[tid + s] > s_val[tid]) {
                s_val[tid] = s_val[tid + s];
                s_idx[tid] = s_idx[tid + s];
            }
        }
        __syncthreads();
    }

    if (tid == 0) {
        out[row] = s_idx[0];
    }
}

// 2026-09-25: argmax_bf16_batch plus `out_logprob[row] = log softmax(row)[argmax]`,
// computed by an online softmax in the same pass over the row.
//
// The caller is the MTP head's batched forward when D-Cut (`mtp_dcut`) wants
// per-position confidences (model-layers mtp_head/forward_batch/position.rs). It
// looks the kernel up with try_kernel and uses argmax_bf16_batch on a zero handle.
//
// log p_max = m - logsumexp = -log Σ exp(v_i - m).







extern "C" __global__ void argmax_bf16_batch_lp(
    const __nv_bfloat16* __restrict__ logits,
    unsigned int* __restrict__ out,
    float* __restrict__ out_logprob,
    unsigned int n,
    unsigned int row_stride
) {
    __shared__ float s_val[1024];
    __shared__ unsigned int s_idx[1024];
    __shared__ float s_sum[1024];

    const unsigned int row = blockIdx.x;
    const __nv_bfloat16* __restrict__ row_logits =
        logits + (unsigned long long)row * (unsigned long long)row_stride;

    const unsigned int tid = threadIdx.x;
    const unsigned int stride = blockDim.x;

    float local_max = -1e30f;
    unsigned int local_idx = 0;
    float local_sum = 0.0f;
    for (unsigned int i = tid; i < n; i += stride) {
        float v = __bfloat162float(row_logits[i]);
        if (v > local_max) {
            // 2026-09-25: Rescale the running sum to the new max, then add this element.
            local_sum = local_sum * __expf(local_max - v) + 1.0f;
            local_max = v;
            local_idx = i;
        } else {
            local_sum += __expf(v - local_max);
        }
    }

    s_val[tid] = local_max;
    s_idx[tid] = local_idx;
    s_sum[tid] = local_sum;
    __syncthreads();

    // 2026-09-25: Tree reduction over (max, index, sum). The max/index half is
    // argmax_bf16_batch's; the sum half is the online-softmax merge. A thread
    // that scanned nothing holds max=-1e30 and sum=0, which merges as a no-op.

    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            const float mine = s_val[tid];
            const float other = s_val[tid + s];
            if (other > mine) {
                s_sum[tid] = s_sum[tid] * __expf(mine - other) + s_sum[tid + s];
                s_val[tid] = other;
                s_idx[tid] = s_idx[tid + s];
            } else {
                s_sum[tid] = s_sum[tid] + s_sum[tid + s] * __expf(other - mine);
            }
        }
        __syncthreads();
    }

    if (tid == 0) {
        out[row] = s_idx[0];
        const float total = s_sum[0];
        // 2026-09-25: total >= 1 for a non-empty row: the max element contributes exp(0).
        out_logprob[row] = (total > 0.0f) ? -__logf(total) : 0.0f;
    }
}

// 2026-09-25: argmax_bf16 over FP32 logits, for an FP32 lm_head output (model-engine meta_argmax.rs).
extern "C" __global__ void argmax_fp32(
    const float* __restrict__ logits,
    unsigned int* __restrict__ out,
    unsigned int n
) {
    __shared__ float s_val[1024];
    __shared__ unsigned int s_idx[1024];

    const unsigned int tid = threadIdx.x;
    const unsigned int stride = blockDim.x;

    float local_max = -1e30f;
    unsigned int local_idx = 0;
    for (unsigned int i = tid; i < n; i += stride) {
        float v = logits[i];
        if (v > local_max) { local_max = v; local_idx = i; }
    }
    s_val[tid] = local_max;
    s_idx[tid] = local_idx;
    __syncthreads();

    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s && s_val[tid + s] > s_val[tid]) {
            s_val[tid] = s_val[tid + s];
            s_idx[tid] = s_idx[tid + s];
        }
        __syncthreads();
    }
    if (tid == 0) out[0] = s_idx[0];
}
