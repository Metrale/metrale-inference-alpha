// SPDX-License-Identifier: MIT OR Apache-2.0

// 2026-09-25: KDA bounded forget gate:
//   gate[t,h,d] = lower_bound * sigmoid(exp(A_log[h]) * (g_raw[t,h,d] + dt_bias[h,d])).
//
// Owner: gb10 kernels.
// Invariants:
// - One block per (token, head) row, with blockIdx.x = t * H + h; threads stride over D,
//   so any block size is correct. The host launches t * H blocks of 128 threads.
// - A_log is FP32 [H], one value per head; dt_bias is FP32 [H, D], one value per channel;
//   g_raw and gate_out are [T, H, D], and gate_out is FP32.
// - lower_bound is an argument, not a constant; the host passes the checkpoint's
//   linear_attn_config.gate_lower_bound.
//
// compute_gdn_gates (ssm_preprocess.cu) is not reused: it takes one dt_bias per head,
// writes [T, heads], and applies exp(-exp(A_log) * softplus(a + dt_bias)), so it cannot
// express a per-channel gate or this law.












































#include <cuda_bf16.h>
#include <math.h>

// 2026-09-25: Both entry points call this, so their arithmetic is the same.

__device__ __forceinline__ float kda_gate_scalar(float g_raw, float dt_bias,
                                                 float decay, float lower_bound) {
    return lower_bound * (1.0f / (1.0f + expf(-(decay * (g_raw + dt_bias)))));
}

// 2026-09-25: The entry point glm5next_kda resolves; g_raw is the BF16 output of f_b.

extern "C" __global__ void kda_gate_bf16(
    const __nv_bfloat16* __restrict__ g_raw,
    const float* __restrict__ dt_bias,
    const float* __restrict__ A_log,
    float* __restrict__ gate_out,
    unsigned int num_tokens,
    unsigned int H,
    unsigned int D,
    float lower_bound
) {
    unsigned int row = blockIdx.x;
    if (row >= num_tokens * H) return;
    unsigned int h = row % H;

    const float decay = expf(A_log[h]);
    const __nv_bfloat16* g_row = g_raw + (size_t)row * D;
    const float* b_row = dt_bias + (size_t)h * D;
    float* o_row = gate_out + (size_t)row * D;

    for (unsigned int d = threadIdx.x; d < D; d += blockDim.x) {
        o_row[d] = kda_gate_scalar(__bfloat162float(g_row[d]), b_row[d], decay, lower_bound);
    }
}

// 2026-09-25: FP32-input twin of kda_gate_bf16, used by the kda_gate_microtest example,
// so a comparison against a reference measures this kernel's error without the BF16
// rounding of its input.
extern "C" __global__ void kda_gate_f32(
    const float* __restrict__ g_raw,
    const float* __restrict__ dt_bias,
    const float* __restrict__ A_log,
    float* __restrict__ gate_out,
    unsigned int num_tokens,
    unsigned int H,
    unsigned int D,
    float lower_bound
) {
    unsigned int row = blockIdx.x;
    if (row >= num_tokens * H) return;
    unsigned int h = row % H;

    const float decay = expf(A_log[h]);
    const float* g_row = g_raw + (size_t)row * D;
    const float* b_row = dt_bias + (size_t)h * D;
    float* o_row = gate_out + (size_t)row * D;

    for (unsigned int d = threadIdx.x; d < D; d += blockDim.x) {
        o_row[d] = kda_gate_scalar(g_row[d], b_row[d], decay, lower_bound);
    }
}
