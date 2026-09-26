// SPDX-License-Identifier: MIT OR Apache-2.0

// 2026-09-25: Elementwise BF16 kernels: residual and scaled adds, sigmoid and softplus gates,
// blends, a BF16 -> FP32 copy and a concat. One thread per element with a bounds check; FP32 math.
//
// Owner: gb10 kernels.
// Invariants: none beyond the types.
#include <cuda_bf16.h>

extern "C" __global__ void bf16_residual_add(
    __nv_bfloat16* __restrict__ residual,
    const __nv_bfloat16* __restrict__ src,
    unsigned int n
) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        float r = __bfloat162float(residual[i]);
        float s = __bfloat162float(src[i]);
        residual[i] = __float2bfloat16(r + s);
    }
}



extern "C" __global__ void bf16_to_f32(
    const __nv_bfloat16* __restrict__ src,
    float* __restrict__ dst,
    unsigned int n
) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        dst[i] = __bfloat162float(src[i]);
    }
}






extern "C" __global__ void silu_mul_separate(
    const __nv_bfloat16* __restrict__ gate,
    const __nv_bfloat16* __restrict__ up,
    __nv_bfloat16* __restrict__ output,
    unsigned int n
) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        float g = __bfloat162float(gate[i]);
        float u = __bfloat162float(up[i]);

        float silu_g = g / (1.0f + expf(-g));
        output[i] = __float2bfloat16(silu_g * u);
    }
}




extern "C" __global__ void bf16_scaled_add(
    __nv_bfloat16* __restrict__ output,
    const __nv_bfloat16* __restrict__ src,
    float scale,
    unsigned int n
) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        float o = __bfloat162float(output[i]);
        float s = __bfloat162float(src[i]);
        output[i] = __float2bfloat16(o + scale * s);
    }
}



// 2026-09-25: output += sigmoid_gate * src. No sigmoid is applied here: sigmoid_gate is the
// gate value the caller has already computed.
extern "C" __global__ void bf16_sigmoid_blend(
    __nv_bfloat16* __restrict__ output,
    const __nv_bfloat16* __restrict__ src,
    float sigmoid_gate,
    unsigned int n
) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        float o = __bfloat162float(output[i]);
        float s = __bfloat162float(src[i]);
        output[i] = __float2bfloat16(o + sigmoid_gate * s);
    }
}





extern "C" __global__ void sigmoid_gate_mul(
    const __nv_bfloat16* __restrict__ input,
    const __nv_bfloat16* __restrict__ gate,
    __nv_bfloat16* __restrict__ output,
    unsigned int n
) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        float x = __bfloat162float(input[i]);
        float g = __bfloat162float(gate[i]);
        float sigmoid_g = 1.0f / (1.0f + expf(-g));
        output[i] = __float2bfloat16(x * sigmoid_g);
    }
}







// 2026-09-25: input and output are contiguous [num_tokens, dim]; gate rows are gate_stride
// elements apart, so the gate can be a slice of a wider projection.
extern "C" __global__ void sigmoid_gate_mul_batched(
    const __nv_bfloat16* __restrict__ input,
    const __nv_bfloat16* __restrict__ gate,
    __nv_bfloat16* __restrict__ output,
    unsigned int dim,
    unsigned int gate_stride,
    unsigned int total_elements
) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < total_elements) {
        unsigned int t = i / dim;
        unsigned int d = i % dim;
        float x = __bfloat162float(input[i]);
        float g = __bfloat162float(gate[t * gate_stride + d]);
        float sigmoid_g = 1.0f / (1.0f + expf(-g));
        output[i] = __float2bfloat16(x * sigmoid_g);
    }
}





extern "C" __global__ void bf16_concat(
    const __nv_bfloat16* __restrict__ a,
    const __nv_bfloat16* __restrict__ b,
    __nv_bfloat16* __restrict__ out,
    unsigned int N
) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) {
        out[i] = a[i];
        out[N + i] = b[i];
    }
}















// 2026-09-25: input and output are [num_tokens, nq * hd] and gate is [num_tokens, nq]: one gate
// per head, output[t, h, d] = input[t, h, d] * sigmoid(gate[t, h]).
extern "C" __global__ void sigmoid_gate_mul_head_broadcast(
    const __nv_bfloat16* __restrict__ input,
    const __nv_bfloat16* __restrict__ gate,
    __nv_bfloat16* __restrict__ output,
    unsigned int nq,
    unsigned int hd,
    unsigned int total_elements
) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < total_elements) {
        unsigned int dim = nq * hd;
        unsigned int t = i / dim;
        unsigned int within_token = i % dim;
        unsigned int head_idx = within_token / hd;
        float x = __bfloat162float(input[i]);
        float g = __bfloat162float(gate[t * nq + head_idx]);
        float sigmoid_g = 1.0f / (1.0f + expf(-g));
        output[i] = __float2bfloat16(x * sigmoid_g);
    }
}
// 2026-09-25: As above with softplus(g) = max(g, 0) + log1p(exp(-|g|)) as the gate; exp cannot overflow.
extern "C" __global__ void softplus_gate_mul_head_broadcast(
    const __nv_bfloat16* __restrict__ input,
    const __nv_bfloat16* __restrict__ gate,
    __nv_bfloat16* __restrict__ output,
    unsigned int nq,
    unsigned int hd,
    unsigned int total_elements
) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < total_elements) {
        unsigned int dim = nq * hd;
        unsigned int t = i / dim;
        unsigned int head_idx = (i % dim) / hd;
        float x = __bfloat162float(input[i]);
        float g = __bfloat162float(gate[t * nq + head_idx]);
        float softplus_g = fmaxf(g, 0.0f) + log1pf(expf(-fabsf(g)));
        output[i] = __float2bfloat16(x * softplus_g);
    }
}

extern "C" __global__ void bf16_sigmoid_blend_device(
    __nv_bfloat16* __restrict__ output,
    const __nv_bfloat16* __restrict__ src,
    const __nv_bfloat16* __restrict__ gate_ptr,
    unsigned int n
) {
    __shared__ float sigmoid_val;
    if (threadIdx.x == 0) {
        float g = __bfloat162float(*gate_ptr);
        sigmoid_val = 1.0f / (1.0f + expf(-g));
    }
    __syncthreads();

    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        float o = __bfloat162float(output[i]);
        float s = __bfloat162float(src[i]);
        output[i] = __float2bfloat16(o + sigmoid_val * s);
    }
}
