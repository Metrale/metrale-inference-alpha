// SPDX-License-Identifier: MIT OR Apache-2.0

// 2026-09-25: RMS norm with the plain weight, out = x * weight / sqrt(mean(x^2) + eps): BF16 in and
// out, FP32 math, two BF16 per 32-bit load. `rms_norm` in rms_norm.cu multiplies by 1 + weight
// instead; where a layer supports both, `ships_vanilla_norm_weights` (crates/model-layers/src/lib.rs)
// picks between them.
//
// Owner: gb10 kernels.
// Invariants:
// - Each row is read and written as 32-bit words from its start, so rows must be 4-byte aligned;
//   an odd hidden_size breaks that from row 1 on.




#include <cuda_bf16.h>

__device__ __forceinline__ void unpack_bf16x2(unsigned int packed, float& v0, float& v1) {
    v0 = __bfloat162float(__ushort_as_bfloat16((unsigned short)(packed & 0xFFFF)));
    v1 = __bfloat162float(__ushort_as_bfloat16((unsigned short)(packed >> 16)));
}

__device__ __forceinline__ unsigned int pack_bf16x2(float v0, float v1) {
    unsigned int lo = (unsigned int)__bfloat16_as_ushort(__float2bfloat16(v0));
    unsigned int hi = (unsigned int)__bfloat16_as_ushort(__float2bfloat16(v1));
    return lo | (hi << 16);
}

__device__ __forceinline__ float warp_reduce_sum(float val) {
    for (int offset = 16; offset > 0; offset >>= 1) {
        val += __shfl_xor_sync(0xFFFFFFFF, val, offset);
    }
    return val;
}

// 2026-09-25: One block per row. blockDim.x is at most 1024: warp_sums has one slot per warp.
// rms_norm in crates/model-layers/src/layers/ops/norm.rs launches min(hidden_size, 1024) threads.
extern "C" __global__ void rms_norm_vanilla(
    const __nv_bfloat16* __restrict__ input,
    const __nv_bfloat16* __restrict__ weight,
    __nv_bfloat16* __restrict__ output,
    unsigned int hidden_size,
    float eps
) {
    unsigned int token = blockIdx.x;
    unsigned int tid = threadIdx.x;

    const __nv_bfloat16* x = input + token * hidden_size;
    __nv_bfloat16* out = output + token * hidden_size;


    float sum_sq = 0.0f;
    const unsigned int half_size = hidden_size / 2;
    const unsigned int* x32 = (const unsigned int*)x;

    for (unsigned int i = tid; i < half_size; i += blockDim.x) {
        float v0, v1;
        unpack_bf16x2(x32[i], v0, v1);
        sum_sq += v0 * v0 + v1 * v1;
    }
    if ((hidden_size & 1) && tid == 0) {
        float val = __bfloat162float(x[hidden_size - 1]);
        sum_sq += val * val;
    }

    sum_sq = warp_reduce_sum(sum_sq);
    __shared__ float warp_sums[32];
    unsigned int warp_id = tid / 32;
    unsigned int lane_id = tid % 32;
    if (lane_id == 0) {
        warp_sums[warp_id] = sum_sq;
    }
    __syncthreads();
    if (warp_id == 0) {
        float val = (lane_id < (blockDim.x + 31) / 32) ? warp_sums[lane_id] : 0.0f;
        val = warp_reduce_sum(val);
        if (lane_id == 0) {
            warp_sums[0] = val;
        }
    }
    __syncthreads();

    float rms = rsqrtf(warp_sums[0] / (float)hidden_size + eps);

    const unsigned int* w32 = (const unsigned int*)weight;
    unsigned int* out32 = (unsigned int*)out;

    for (unsigned int i = tid; i < half_size; i += blockDim.x) {
        float xv0, xv1, wv0, wv1;
        unpack_bf16x2(x32[i], xv0, xv1);
        unpack_bf16x2(w32[i], wv0, wv1);

        out32[i] = pack_bf16x2(xv0 * rms * wv0, xv1 * rms * wv1);
    }
    if ((hidden_size & 1) && tid == 0) {
        float val = __bfloat162float(x[hidden_size - 1]);
        float w = __bfloat162float(weight[hidden_size - 1]);
        out[hidden_size - 1] = __float2bfloat16(val * rms * w);
    }
}











// 2026-09-25: One warp per row and RMSV_ROWS_PER_BLOCK rows per block, so the reduction is
// shuffles only, with no shared memory and no __syncthreads. It is for many short rows, the
// prefill per-head q/k norms: rms_norm_warp_row in crates/model-layers/src/layers/ops/norm.rs
// launches it with blockDim 32 * 8, and rms_norm_short_row_eligible there admits only an even
// hidden_size <= 256 with at least 1024 rows. Same formula as rms_norm_vanilla; the reduction
// order differs, so results are not bit-identical to it.
#define RMSV_ROWS_PER_BLOCK 8

extern "C" __global__ void rms_norm_vanilla_warp_row(
    const __nv_bfloat16* __restrict__ input,
    const __nv_bfloat16* __restrict__ weight,
    __nv_bfloat16* __restrict__ output,
    unsigned int num_rows,
    unsigned int hidden_size,
    float eps
) {
    const unsigned int lane = threadIdx.x & 31;
    const unsigned int row = blockIdx.x * RMSV_ROWS_PER_BLOCK + (threadIdx.x >> 5);
    if (row >= num_rows) return;

    const size_t base = (size_t)row * hidden_size;
    const unsigned int half_size = hidden_size >> 1;
    const unsigned int* x32 = (const unsigned int*)(input + base);
    const unsigned int* w32 = (const unsigned int*)weight;
    unsigned int* out32 = (unsigned int*)(output + base);

    float sum_sq = 0.0f;
    for (unsigned int i = lane; i < half_size; i += 32) {
        float v0, v1;
        unpack_bf16x2(x32[i], v0, v1);
        sum_sq += v0 * v0 + v1 * v1;
    }
    if ((hidden_size & 1) && lane == 0) {
        float val = __bfloat162float(input[base + hidden_size - 1]);
        sum_sq += val * val;
    }

    sum_sq = warp_reduce_sum(sum_sq);
    const float rms = rsqrtf(sum_sq / (float)hidden_size + eps);

    for (unsigned int i = lane; i < half_size; i += 32) {
        float xv0, xv1, wv0, wv1;
        unpack_bf16x2(x32[i], xv0, xv1);
        unpack_bf16x2(w32[i], wv0, wv1);
        out32[i] = pack_bf16x2(xv0 * rms * wv0, xv1 * rms * wv1);
    }
    if ((hidden_size & 1) && lane == 0) {
        float val = __bfloat162float(input[base + hidden_size - 1]);
        float w = __bfloat162float(weight[hidden_size - 1]);
        output[base + hidden_size - 1] = __float2bfloat16(val * rms * w);
    }
}
