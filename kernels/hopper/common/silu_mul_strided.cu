// SPDX-License-Identifier: MIT OR Apache-2.0

// 2026-09-25: SiLU(gate) * up over row-strided operands, the consumer of the fused
// dense-FFN gate+up GEMM.
//
// Owner: hopper kernels.
// Invariants:
// - For r < rows and c < cols, output[r * out_stride + c] =
//   silu(g) * u with g = gate[r * in_stride + c], u = up[r * in_stride + c]:
//   the expression of `moe_silu_mul` (kernels/gb10/common/moe_silu_mul.cu).
//   Nothing outside those elements is written.
// - Launched with grid (ceil(cols / 256), rows, 1) and block (256, 1, 1)
//   (`ops::silu_mul_strided`).
//
// The fused GEMM writes [m, 2 * inter] with gate in columns [0, inter) and up
// in [inter, 2 * inter). `moe_silu_mul` indexes all three tensors with one
// flat index, so it needs them contiguous; this kernel takes row strides
// instead (`w8a8_gate_up_fused` passes in_stride = 2 * inter and
// out_stride = inter).
//
// The source is in kernels/hopper/common because kernels/hopper/HARDWARE.toml
// declares `ffn_gateup_fused = true`; gb10, b200 and b300 declare it false.



















#include <cuda_bf16.h>

extern "C" __global__ void silu_mul_strided(
    const __nv_bfloat16* __restrict__ gate,
    const __nv_bfloat16* __restrict__ up,
    __nv_bfloat16* __restrict__ output,
    unsigned int rows,
    unsigned int cols,
    unsigned int in_stride,
    unsigned int out_stride
) {
    unsigned int c = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int r = blockIdx.y;
    if (c >= cols || r >= rows) return;

    const size_t in_base = (size_t)r * (size_t)in_stride + (size_t)c;
    float g = __bfloat162float(gate[in_base]);
    float u = __bfloat162float(up[in_base]);
    float sigmoid_g = 1.0f / (1.0f + __expf(-g));
    float result = g * sigmoid_g * u;
    output[(size_t)r * (size_t)out_stride + (size_t)c] = __float2bfloat16(result);
}
