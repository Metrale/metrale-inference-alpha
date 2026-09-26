// SPDX-License-Identifier: MIT OR Apache-2.0
// forked-from: kernels/gb10/common/moe_silu_mul.cu (2026-09-24; 170 of 169 lines differ, see kernels/FORKS.md)

// 2026-09-25: moe_silu_mul for step3p7-flash: output = silu(gate) * up over BF16 values, with gate
// clamped to at most SWIGLU_LIMIT and up to [-SWIGLU_LIMIT, SWIGLU_LIMIT] first. gb10/common's
// moe_silu_mul does not clamp, and every moe_silu_mul launch of this target uses this one.
//
// Owner: gb10 kernels (step3p7-flash).
// Invariants: none beyond the types.
//
// SWIGLU_LIMIT is a constant rather than the checkpoint's per-layer limits: crates/config/src/parsers/
// step3p7.rs removes the swiglu_limits and swiglu_limits_shared arrays before deserialisation, and
// ModelConfig holds only a scalar swiglu_limit. This file defines no silu_mul_quant_fp8; KERNEL.toml declares that
// in [shadow_exempt].
//
// Grid (ceil(total_elements / 256)), block 256 (ops/activations.rs silu_mul, ops/moe_grouped_a.rs moe_silu_mul).












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


    const float SWIGLU_LIMIT = 10.0f;
    g = fminf(g, SWIGLU_LIMIT);
    u = fminf(fmaxf(u, -SWIGLU_LIMIT), SWIGLU_LIMIT);
    float sigmoid_g = 1.0f / (1.0f + __expf(-g));
    float result = g * sigmoid_g * u;
    output[idx] = __float2bfloat16(result);
}
