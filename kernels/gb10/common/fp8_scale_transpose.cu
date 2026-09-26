// SPDX-License-Identifier: MIT OR Apache-2.0

// 2026-09-25: Activation-scale layout adapter for the cuBLASLt block-scaled FP8 GEMM:
// transposes per_token_group_quant_fp8's row-major a_scale [M, L] (L = K / 128) into the
// [L, M_pad] layout that the cuBLASLt arm passes as its VEC128 scale.
//
// Owner: gb10 kernels.
// Invariants:
// - dst[l * M_pad + m] = src[m * L + l] for m < M, and 0.0f for the pad rows M..M_pad, so a
//   pad row never reads stale scratch. The quantizer's own buffer is not modified;
//   fp8_gemm_t_blockscaled still reads that one directly.
// - Launch: grid (ceil(M_pad / 256), L, 1), block (256, 1, 1); one thread per dst element.
// - The index math mirrors metrale_gpu_runtime::cublaslt::scale_layout (vec128_b_index,
//   act_scale_rowmajor_to_kmajor), which holds the CPU reference and the layout's sources.




















extern "C" __global__ void fp8_act_scale_to_kmajor(
    const float* __restrict__ src,
    float* __restrict__ dst,
    unsigned int M,
    unsigned int M_pad,
    unsigned int L
) {
    const unsigned int m = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int l = blockIdx.y;
    if (m >= M_pad || l >= L) return;
    dst[(size_t)l * M_pad + m] = (m < M) ? src[(size_t)m * L + l] : 0.0f;
}
