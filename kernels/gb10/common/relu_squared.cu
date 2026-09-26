// SPDX-License-Identifier: MIT OR Apache-2.0

// 2026-09-25: ReLU-squared, max(0, x)^2, and small elementwise helpers for the Nemotron MoE path.
// One thread per element with a bounds check; FP32 math. crates/model-arch/src/nemotron_moe.rs
// resolves relu_squared_inplace and moe_weighted_sum_scale; the other kernels have no launcher.
//
// Owner: gb10 kernels.
// Invariants: none beyond the types.

#include <cuda_bf16.h>

extern "C" __global__ void relu_squared(
    const __nv_bfloat16* __restrict__ input,
    __nv_bfloat16* __restrict__ output,
    unsigned int total_elements
) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total_elements) return;

    float x = __bfloat162float(input[idx]);
    float r = fmaxf(x, 0.0f);
    output[idx] = __float2bfloat16(r * r);
}


extern "C" __global__ void relu_squared_inplace(
    __nv_bfloat16* __restrict__ data,
    unsigned int total_elements
) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total_elements) return;

    float x = __bfloat162float(data[idx]);
    float r = fmaxf(x, 0.0f);
    data[idx] = __float2bfloat16(r * r);
}






extern "C" __global__ void bias_add_bf16_f32(
    __nv_bfloat16* __restrict__ data,
    const float* __restrict__ bias,
    unsigned int N
) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N) return;
    float val = __bfloat162float(data[idx]) + bias[idx];
    data[idx] = __float2bfloat16(val);
}




extern "C" __global__ void moe_weighted_sum_scale(
    __nv_bfloat16* __restrict__ output,
    const __nv_bfloat16* __restrict__ expert_down,
    const float* __restrict__ weights,
    const __nv_bfloat16* __restrict__ shared_down,
    unsigned int H,
    unsigned int top_k,
    float routed_scaling_factor
) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= H) return;

    float routed_sum = 0.0f;
    for (unsigned int k = 0; k < top_k; k++) {
        routed_sum += weights[k] * __bfloat162float(expert_down[k * H + idx]);
    }
    float shared_val = __bfloat162float(shared_down[idx]);
    output[idx] = __float2bfloat16(routed_scaling_factor * routed_sum + shared_val);
}




extern "C" __global__ void convert_f32_to_bf16(
    const float* __restrict__ input,
    __nv_bfloat16* __restrict__ output,
    unsigned int N
) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N) return;
    output[idx] = __float2bfloat16(input[idx]);
}
