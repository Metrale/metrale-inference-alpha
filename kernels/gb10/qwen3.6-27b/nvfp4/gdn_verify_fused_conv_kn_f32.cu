// SPDX-License-Identifier: MIT OR Apache-2.0

// 2026-09-25: Kernel `gdn_verify_fused_conv_kn_f32`: for num_tokens consecutive positions of one sequence, the conv1d
// update + SiLU (with L2 norm per head_dim group in blocks that start below qk_channels) in one launch. It writes each
// position's FP32 output and a snapshot of every channel's window after that position to conv_state_inter, then the
// final window to conv_state. conv_state and weight are [dim, d_conv]; new_input rows are input_stride BF16 elements
// apart, output rows output_stride FP32 elements, snapshots inter_stride FP32 elements.
//
// It is `gdn_verify_fused_conv_kn` (gb10/common/gdn_verify_fused_conv_kn.cu) with a float output and no BF16
// rounding. Per position the arithmetic is that of `causal_conv1d_update_l2norm_f32` (gb10/common/causal_conv1d.cu)
// with a null bias, which is what ops::conv1d_update_l2norm passes; examples/verify_exact_microtest compares the two
// byte for byte.
//
// Only kernels/gb10/qwen3.6-27b/nvfp4 defines it (qwen3.8-27b compiles this tree through kernel_source). When the
// handle is 0, or the conv intermediates are not contiguous, the caller runs the per-token conv and copies the conv
// state (qwen3_ssm/trait_decode_batched_conv_gdn_exact.rs). Grid (ceil(dim / 256), 1, 1), Block (256, 1, 1).
//
// Owner: gb10 kernels (qwen3.6-27b).
// Invariants: none beyond the types. d_conv <= 8 (register window), and the L2 reduction assumes head_dim == 128 in
// 256-thread blocks; nothing checks either.










#include <cuda_bf16.h>

extern "C" __global__ void gdn_verify_fused_conv_kn_f32(
    float* __restrict__ conv_state,
    const __nv_bfloat16* __restrict__ new_input,
    const __nv_bfloat16* __restrict__ weight,
    float* __restrict__ output,
    float* __restrict__ conv_state_inter,
    unsigned int num_tokens,
    unsigned int dim,
    unsigned int d_conv,
    unsigned int qk_channels,
    unsigned int head_dim,
    unsigned int input_stride,
    unsigned int output_stride,
    unsigned int inter_stride,
    float l2_eps
) {
    const unsigned int ch = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int tid = threadIdx.x;
    const unsigned int block_start = blockIdx.x * blockDim.x;
    const bool block_needs_l2 = (block_start < qk_channels);
    const bool valid = (ch < dim);


    float win[8];
    if (valid) {
        const float* state = conv_state + ch * d_conv;
        for (unsigned int i = 0; i < d_conv; i++) win[i] = state[i];
    }

    const __nv_bfloat16* w = valid ? (weight + ch * d_conv) : nullptr;
    float wcoef[8];
    if (valid) {
        for (unsigned int k = 0; k < d_conv; k++) wcoef[k] = (float)w[k];
    }

    __shared__ float warp_sums[8];

    // 2026-09-25: Positions run in order; the L2 norm needs barriers inside each position, as in the per-token kernel.


    for (unsigned int t = 0; t < num_tokens; t++) {
        float silu = 0.0f;
        if (valid) {

            for (unsigned int i = 0; i < d_conv - 1; i++) win[i] = win[i + 1];
            win[d_conv - 1] = (float)new_input[t * input_stride + ch];
            // 2026-09-25: No bias term: ops::conv1d_update_l2norm always passes a null bias.
            float acc = 0.0f;
            for (unsigned int k = 0; k < d_conv; k++) acc += win[k] * wcoef[k];
            float sigmoid_acc = 1.0f / (1.0f + __expf(-acc));
            silu = acc * sigmoid_acc;
        }

        if (block_needs_l2) {
            float sq = valid ? (silu * silu) : 0.0f;
            const unsigned int warp_id = tid / 32;
            const unsigned int lane = tid % 32;
            for (int offset = 16; offset >= 1; offset >>= 1)
                sq += __shfl_down_sync(0xFFFFFFFF, sq, offset);
            if (lane == 0) warp_sums[warp_id] = sq;
            __syncthreads();
            const unsigned int head_in_block = tid / head_dim;
            const unsigned int base_warp = head_in_block * (head_dim / 32);
            if (tid == 0 || tid == head_dim) {
                float total = warp_sums[base_warp] + warp_sums[base_warp + 1]
                            + warp_sums[base_warp + 2] + warp_sums[base_warp + 3];
                warp_sums[base_warp] = rsqrtf(total + l2_eps);
            }
            __syncthreads();
            if (valid) silu *= warp_sums[base_warp];
            // 2026-09-25: The next position's lane-0 write to warp_sums must not overtake this read.

            __syncthreads();
        }

        if (valid) output[t * output_stride + ch] = silu;

        // 2026-09-25: Snapshot t: the window after position t.
        if (valid) {
            float* snap = conv_state_inter + t * inter_stride + ch * d_conv;
            for (unsigned int i = 0; i < d_conv; i++) snap[i] = win[i];
        }
    }


    if (valid) {
        float* state = conv_state + ch * d_conv;
        for (unsigned int i = 0; i < d_conv; i++) state[i] = win[i];
    }
}
