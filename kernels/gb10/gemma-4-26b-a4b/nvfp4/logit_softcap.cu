// SPDX-License-Identifier: MIT OR Apache-2.0
// forked-from: kernels/gb10/gemma-4-31b/nvfp4/logit_softcap.cu (2026-09-24; 17 of 41 lines differ, see kernels/FORKS.md)

// 2026-09-25: Logit softcap in place: logits[i] = cap * tanh(logits[i] * inv_cap) over N BF16 logits. The engine
// loads it when the config sets final_logit_softcapping and launches it with inv_cap = 1 / cap, 256 threads per
// block.
// Owner: gb10 kernels (gemma-4-26b-a4b).
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
