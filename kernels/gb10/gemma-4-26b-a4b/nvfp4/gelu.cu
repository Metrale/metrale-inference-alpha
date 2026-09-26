// SPDX-License-Identifier: MIT OR Apache-2.0

// 2026-09-25: Tanh-approximation GELU kernels, BF16 in and out, FP32 math:
//   gelu(x) = 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3))).
// gelu_mul writes gelu(gate[i]) * up[i]; the dense FFN (FfnActivation::GeLU) and the MoE experts
// (set_gelu_activation) load it as gelu::gelu_mul. Nothing in crates/ loads gelu_tanh.
//
// Owner: gb10 kernels (gemma-4-26b-a4b, and gemma-4-31b through `[sources] use`).
// Invariants: none beyond the types; a thread with idx >= N returns without touching memory.





#include <cuda_bf16.h>





extern "C" __global__ void gelu_tanh(
    const __nv_bfloat16* __restrict__ input,
    __nv_bfloat16* __restrict__ output,
    unsigned int N
) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N) return;

    float x = __bfloat162float(input[idx]);
    // 2026-09-25: 0.7978845608 = sqrt(2/pi).
    float inner = 0.7978845608f * (x + 0.044715f * x * x * x);
    float gelu = 0.5f * x * (1.0f + tanhf(inner));
    output[idx] = __float2bfloat16(gelu);
}








extern "C" __global__ void gelu_mul(
    const __nv_bfloat16* __restrict__ gate,
    const __nv_bfloat16* __restrict__ up,
    __nv_bfloat16* __restrict__ output,
    unsigned int N
) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N) return;

    float g = __bfloat162float(gate[idx]);
    float u = __bfloat162float(up[idx]);


    float inner = 0.7978845608f * (g + 0.044715f * g * g * g);
    float gelu_g = 0.5f * g * (1.0f + tanhf(inner));

    output[idx] = __float2bfloat16(gelu_g * u);
}
