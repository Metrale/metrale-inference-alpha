#pragma once
// SPDX-License-Identifier: MIT OR Apache-2.0

// 2026-09-25: CUDA mask-argument warp intrinsics for the Windows HIP build:
// `__shfl_sync`, `__shfl_up_sync`, `__shfl_down_sync`, `__shfl_xor_sync`,
// `__any_sync`, `__all_sync` and `__activemask`, each mapped onto the base
// HIP intrinsic (`__shfl*`, `__any`, `__all`, `__ballot`) with the mask
// argument ignored. Each shuffle has overloads with and without `width`.
//
// Owner: HIP compat headers (kernels crate).
// Invariants: none beyond the declarations.
//
// `HipTarget::compile` (`build_target.rs`) force-includes this header only
// when the build host is Windows; nothing else includes it.




#if defined(__HIP_PLATFORM_AMD__) || defined(__HIP__)

template <typename T>
__device__ __forceinline__ T __shfl_sync(unsigned long long, T v, int src_lane) {
    return __shfl(v, src_lane);
}
template <typename T>
__device__ __forceinline__ T __shfl_sync(unsigned long long, T v, int src_lane, int width) {
    return __shfl(v, src_lane, width);
}

template <typename T>
__device__ __forceinline__ T __shfl_up_sync(unsigned long long, T v, unsigned int delta) {
    return __shfl_up(v, delta);
}
template <typename T>
__device__ __forceinline__ T __shfl_up_sync(unsigned long long, T v, unsigned int delta, int width) {
    return __shfl_up(v, delta, width);
}

template <typename T>
__device__ __forceinline__ T __shfl_down_sync(unsigned long long, T v, unsigned int delta) {
    return __shfl_down(v, delta);
}
template <typename T>
__device__ __forceinline__ T __shfl_down_sync(unsigned long long, T v, unsigned int delta, int width) {
    return __shfl_down(v, delta, width);
}

template <typename T>
__device__ __forceinline__ T __shfl_xor_sync(unsigned long long, T v, int lane_mask) {
    return __shfl_xor(v, lane_mask);
}
template <typename T>
__device__ __forceinline__ T __shfl_xor_sync(unsigned long long, T v, int lane_mask, int width) {
    return __shfl_xor(v, lane_mask, width);
}

__device__ __forceinline__ int __any_sync(unsigned long long, int pred) { return __any(pred); }
__device__ __forceinline__ int __all_sync(unsigned long long, int pred) { return __all(pred); }



__device__ __forceinline__ unsigned long long __activemask() { return __ballot(1); }

#endif
