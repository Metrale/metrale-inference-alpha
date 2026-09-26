// SPDX-License-Identifier: MIT OR Apache-2.0

// 2026-09-25: Logit softcap in place: logits[i] = cap * tanh(logits[i] * inv_cap) over N BF16 logits. The engine
// loads it when the config sets final_logit_softcapping and launches it with inv_cap = 1 / cap, 256 threads per
// block.
// Owner: gb10 kernels (gemma-4-31b).
// Invariants: none beyond the types; a thread with idx >= N returns without touching memory.


#include <cuda_bf16.h>

extern "C" __global__ void logit_softcap_bf16(
    __nv_bfloat16* __restrict__ logits,
    unsigned int N,
    float inv_cap,
    float cap
) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N) return;

    float x = __bfloat162float(logits[idx]);
    float y = cap * tanhf(x * inv_cap);
    logits[idx] = __float2bfloat16(y);
}

// 2026-09-25: FP32 in/out twin of logit_softcap_bf16. The engine never loads it: logit_softcap_fp32_kernel is
// always KernelHandle(0) (crates/model-engine/src/model/impl_a1.rs).



extern "C" __global__ void logit_softcap_fp32(
    float* __restrict__ logits,
    unsigned int N,
    float inv_cap,
    float cap
) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N) return;
    float x = logits[idx];
    logits[idx] = cap * tanhf(x * inv_cap);
}
