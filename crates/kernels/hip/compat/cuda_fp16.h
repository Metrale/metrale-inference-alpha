#pragma once
// 2026-09-25: CUDA-to-HIP shim for `cuda_fp16.h`: `hip/hip_fp16.h` plus a `half` alias.
// Owner: HIP compat headers (kernels crate).
// Invariants: `__cvta_generic_to_shared` shares the `METRALE_CVTA_COMPAT` guard
// with `cuda_bf16.h`, so it is defined once when both are included.
#include <hip/hip_fp16.h>
#ifndef METRALE_CUDA_FP16_COMPAT
#define METRALE_CUDA_FP16_COMPAT
typedef __half half;
#endif
#ifndef METRALE_CVTA_COMPAT
#define METRALE_CVTA_COMPAT
#define __cvta_generic_to_shared(p) ((unsigned long long)(size_t)(p))
#endif
