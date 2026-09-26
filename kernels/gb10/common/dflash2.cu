// SPDX-License-Identifier: MIT OR Apache-2.0
// provenance-id: 526f6e616c6420522e205374657369616b

// 2026-09-25: DFlash2 drafter kernels: grouped dynamic two-tap conv, per-row top-16, and
// the candidate-selector chain walk. Arithmetic in f32; activations stored as BF16.
//
// Owner: gb10 kernels.
// Invariants:
// - Rows are packed sequence-major in bands (gamma rows for the conv, `rows` rows for the
//   walk), and neither kernel reads a row of another band.


#include <cuda_bf16.h>

// 2026-09-25: Two-tap grouped dynamic causal convolution, one application (prepare or
// finish). For row t, channel c, group g = c / group_size:
//   coef_k(t, c) = base[k*h + c] + dyn[t*dyn_stride + app_off + k*groups + g]
//   out[t][c]    = coef_0 * x[t][c] + coef_1 * x[t-1][c]
// The second tap is dropped on the first row of each band of gamma rows (gamma == 0: only
// row 0), so no row convolves with another sequence's row. dyn rows hold
// [2 applications, kernel_size, groups]; app_off selects the application (0 = prepare,
// kernel_size * groups = finish). kernel_size is 2 (the host checks it). out must not
// alias x: block t reads row t - 1 of x. Launch: grid (rows, 1, 1), block (256, 1, 1).







extern "C" __global__ void dflash2_conv2(
    const __nv_bfloat16* __restrict__ x,
    const __nv_bfloat16* __restrict__ dyn,
    const __nv_bfloat16* __restrict__ base,
    __nv_bfloat16* __restrict__ out,
    unsigned int h,
    unsigned int group_size,
    unsigned int dyn_stride,
    unsigned int app_off,
    unsigned int gamma
) {
    const unsigned int t = blockIdx.x;
    const unsigned int groups = h / group_size;
    const __nv_bfloat16* xr = x + (size_t)t * h;





    const bool first_in_band = (gamma == 0u) ? (t == 0u) : (t % gamma == 0u);
    const __nv_bfloat16* xp = first_in_band ? nullptr : x + (size_t)(t - 1) * h;
    const __nv_bfloat16* dr = dyn + (size_t)t * dyn_stride + app_off;
    __nv_bfloat16* orow = out + (size_t)t * h;

    for (unsigned int c = threadIdx.x; c < h; c += blockDim.x) {
        const unsigned int g = c / group_size;
        const float c0 = __bfloat162float(base[c]) + __bfloat162float(dr[g]);
        float acc = c0 * __bfloat162float(xr[c]);
        if (xp != nullptr) {
            const float c1 =
                __bfloat162float(base[h + c]) + __bfloat162float(dr[groups + g]);
            acc += c1 * __bfloat162float(xp[c]);
        }
        orow[c] = __float2bfloat16(acc);
    }
}

// 2026-09-25: Per-row top-16 over BF16 logits, one block per row, 16 sequential
// block-argmax passes, so top_vals / top_idx come out in descending order. Destructive:
// each selected entry of `logits` is overwritten with -1e30. The tree reduction assumes
// blockDim.x is a power of two <= 1024 (the host launches 1024).

extern "C" __global__ void dflash2_topk16(
    __nv_bfloat16* __restrict__ logits,
    float* __restrict__ top_vals,
    unsigned int* __restrict__ top_idx,
    unsigned int n
) {
    __shared__ float s_val[1024];
    __shared__ unsigned int s_idx[1024];

    const unsigned int row = blockIdx.x;
    const unsigned int tid = threadIdx.x;
    const unsigned int stride = blockDim.x;
    __nv_bfloat16* lrow = logits + (size_t)row * n;

    for (unsigned int k = 0; k < 16; ++k) {
        float local_max = -1e30f;
        unsigned int local_idx = 0;
        for (unsigned int i = tid; i < n; i += stride) {
            const float v = __bfloat162float(lrow[i]);
            if (v > local_max) {
                local_max = v;
                local_idx = i;
            }
        }
        s_val[tid] = local_max;
        s_idx[tid] = local_idx;
        __syncthreads();
        for (unsigned int off = stride >> 1; off > 0; off >>= 1) {
            if (tid < off && s_val[tid + off] > s_val[tid]) {
                s_val[tid] = s_val[tid + off];
                s_idx[tid] = s_idx[tid + off];
            }
            __syncthreads();
        }
        if (tid == 0) {
            top_vals[(size_t)row * 16 + k] = s_val[0];
            top_idx[(size_t)row * 16 + k] = s_idx[0];
            lrow[s_idx[0]] = __float2bfloat16(-1e30f);
        }
        __syncthreads();
    }
}

// 2026-09-25: Candidate-selector chain walk, one block per sequence band:
//   S_t(a, b) = U_t(b) + sum_r pred[a][r] * hproj[t][r] * succ[b][r]
// U_t is top_vals and the candidates b are top_idx. Block i walks rows first_row..rows-1
// of band i (rows = rows per band). The predecessor a of the first walked row is
// tokens[i * rows]; each chosen candidate is written to tokens[row] and becomes the next
// row's predecessor. Launch: block (512, 1, 1), 16 warps, warp k scores candidate k.
// rank must be <= 256 (s_gate); the host checks only rank > 0.







extern "C" __global__ void dflash2_selector_walk(
    const float* __restrict__ top_vals,
    const unsigned int* __restrict__ top_idx,
    const __nv_bfloat16* __restrict__ hproj,
    const __nv_bfloat16* __restrict__ pred,
    const __nv_bfloat16* __restrict__ succ,
    unsigned int* __restrict__ tokens,
    unsigned int rows,
    unsigned int rank,
    unsigned int first_row
) {



    const size_t base_row = (size_t)blockIdx.x * rows;
    __shared__ float s_gate[256];
    __shared__ float s_score[16];
    __shared__ unsigned int s_prev;

    const unsigned int tid = threadIdx.x;
    const unsigned int warp = tid >> 5;
    const unsigned int lane = tid & 31;

    if (tid == 0) {
        s_prev = tokens[base_row];
    }
    __syncthreads();

    for (unsigned int t = first_row; t < rows; ++t) {
        const size_t row = base_row + t;

        const __nv_bfloat16* pr = pred + (size_t)s_prev * rank;
        const __nv_bfloat16* hr = hproj + row * rank;
        for (unsigned int r = tid; r < rank; r += blockDim.x) {
            s_gate[r] = __bfloat162float(pr[r]) * __bfloat162float(hr[r]);
        }
        __syncthreads();


        if (warp < 16) {
            const unsigned int cand = top_idx[row * 16 + warp];
            const __nv_bfloat16* sr = succ + (size_t)cand * rank;
            float acc = 0.0f;
            for (unsigned int r = lane; r < rank; r += 32) {
                acc += s_gate[r] * __bfloat162float(sr[r]);
            }
            #pragma unroll
            for (unsigned int off = 16; off > 0; off >>= 1) {
                acc += __shfl_down_sync(0xffffffff, acc, off);
            }
            if (lane == 0) {
                s_score[warp] = top_vals[row * 16 + warp] + acc;
            }
        }
        __syncthreads();

        if (tid == 0) {
            float best = s_score[0];
            unsigned int best_k = 0;
            #pragma unroll
            for (unsigned int k = 1; k < 16; ++k) {
                if (s_score[k] > best) {
                    best = s_score[k];
                    best_k = k;
                }
            }
            const unsigned int chosen = top_idx[row * 16 + best_k];
            tokens[row] = chosen;
            s_prev = chosen;
        }
        __syncthreads();
    }
}
