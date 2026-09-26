// SPDX-License-Identifier: MIT OR Apache-2.0

// 2026-09-25: InnerQ state for the WHT-rotated (turbo) KV dtypes: per-channel scales for K
// (d_innerq_scale) and Q (d_innerq_scale_inv), and the calibration accumulators.
//
// Owner: gb10 kernels.
// Invariants: none beyond the types.
//
// The definitions are in tq_plus_innerq_apply.cu, with the only kernels that use them, and
// innerq_driver.rs (model-layers) resolves them in module tq_plus_innerq_apply. Each .cu is
// its own PTX module built without -rdc, so another file that includes this header does not
// share these globals with that module. The calibration and finalize steps are the driver's
// start and maybe_finalize.











#pragma once
#include <cuda_bf16.h>

namespace tq_plus {

// 2026-09-25: Channels per head; the apply kernels handle head_dim 128 only.

#define INNERQ_MAX_CHANNELS 128











extern __device__ float d_innerq_scale[INNERQ_MAX_CHANNELS];
extern __device__ float d_innerq_scale_inv[INNERQ_MAX_CHANNELS];
extern __device__ float d_innerq_sq_accum[INNERQ_MAX_CHANNELS];
extern __device__ int   d_innerq_count;
extern __device__ int   d_innerq_active;
extern __device__ int   d_innerq_calibrating;



__device__ __forceinline__ void apply_innerq_scale_inv_128(float vals[4], unsigned int lane) {
    if (d_innerq_active == 0) return;
    #pragma unroll
    for (int i = 0; i < 4; i++) {
        unsigned int ch = lane * 4 + i;
        vals[i] *= d_innerq_scale_inv[ch];
    }
}


__device__ __forceinline__ void apply_innerq_scale_128(float vals[4], unsigned int lane) {
    if (d_innerq_active == 0) return;
    #pragma unroll
    for (int i = 0; i < 4; i++) {
        unsigned int ch = lane * 4 + i;
        vals[i] *= d_innerq_scale[ch];
    }
}





__device__ __forceinline__ void accumulate_innerq_calibration_128(float vals[4], unsigned int lane) {
    if (d_innerq_calibrating == 0) return;
    #pragma unroll
    for (int i = 0; i < 4; i++) {
        unsigned int ch = lane * 4 + i;
        atomicAdd(&d_innerq_sq_accum[ch], vals[i] * vals[i]);
    }
    if (lane == 0 && threadIdx.y == 0) {
        atomicAdd(&d_innerq_count, 1);
    }
}

}
