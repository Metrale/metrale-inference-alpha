// SPDX-License-Identifier: MIT OR Apache-2.0
// forked-from: kernels/gb10/common/moe_silu_mul.cu (2026-09-24; 179 of 169 lines differ, see kernels/FORKS.md)

// 2026-09-25: MoE SiLU(gate) * up with DeepSeek-V4's SwiGLU clamp, for the deepseek-v4-flash tree
// (deepseek-v4.1-flash builds the same tree through its MODEL.toml kernel_source).
//
// Owner: gb10 kernels (deepseek-v4-flash).
// Invariants:
// - output[i] = silu(min(gate[i], 10)) * clamp(up[i], -10, 10), one thread per element;
//   ops::moe_silu_mul launches grid ceil(total_elements / 256), block 256.
// - The clamp is the checkpoint's inference/model.py: gate is bounded above only, up on both sides.
// - The file defines no silu_mul_quant_fp8 (KERNEL.toml [shadow_exempt]); the MoE prefill checks
//   the handle in fused_silu_quant_ok and runs the unfused pair when it is absent.
























#include <cuda_bf16.h>

extern "C" __global__ void moe_silu_mul(
    const __nv_bfloat16* __restrict__ gate,
    const __nv_bfloat16* __restrict__ up,
    __nv_bfloat16* __restrict__ output,
    unsigned int total_elements
) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total_elements) return;

    float g = __bfloat162float(gate[idx]);
    float u = __bfloat162float(up[idx]);
    // 2026-09-25: The checkpoint's config.json swiglu_limit (10.0). ModelConfig::swiglu_limit holds it,
    // but ops::moe_silu_mul passes no limit argument, so the kernel fixes the value here.
    const float SWIGLU_LIMIT = 10.0f;
    g = fminf(g, SWIGLU_LIMIT);
    u = fminf(fmaxf(u, -SWIGLU_LIMIT), SWIGLU_LIMIT);
    float sigmoid_g = 1.0f / (1.0f + __expf(-g));
    float result = g * sigmoid_g * u;
    output[idx] = __float2bfloat16(result);
}
