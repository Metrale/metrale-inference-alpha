// SPDX-License-Identifier: MIT OR Apache-2.0

// 2026-09-25: CUDA runtime API shim for the HIP target: `cudaMalloc`,
// `cudaFree`, `cudaHostAlloc`, `cudaMemcpy`, `cudaMemcpy2DAsync` and
// `cudaDeviceSynchronize`, each forwarding to the matching HIP call. Built as
// `libcudart.so` (Linux) or into the HIP `cuda.dll` (Windows) so `-lcudart`
// resolves.
//
// Owner: HIP shims (kernels crate).
// Invariants: none beyond the declarations.
#include <hip/hip_runtime.h>

extern "C" {

int cudaMalloc(void** ptr, size_t size) { return hipMalloc(ptr, size); }
int cudaFree(void* ptr) { return hipFree(ptr); }

int cudaHostAlloc(void** ptr, size_t size, unsigned int flags) {
    return hipHostMalloc(ptr, size, flags);
}

int cudaMemcpy(void* dst, const void* src, size_t count, int kind) {
    return hipMemcpy(dst, src, count, (hipMemcpyKind)kind);
}

int cudaMemcpy2DAsync(void* dst, size_t dpitch, const void* src, size_t spitch,
                      size_t width, size_t height, int kind, void* stream) {
    return hipMemcpy2DAsync(dst, dpitch, src, spitch, width, height,
                            (hipMemcpyKind)kind, (hipStream_t)stream);
}

int cudaDeviceSynchronize() { return hipDeviceSynchronize(); }

}
