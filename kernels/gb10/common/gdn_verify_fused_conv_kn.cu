// SPDX-License-Identifier: MIT OR Apache-2.0

// 2026-09-25: Fused conv1d + SiLU + L2-norm over every position of a GDN verify, one launch.
//
// Owner: gb10 kernels.
// Invariants:
// - Per position, the arithmetic and its order are causal_conv1d_update_l2norm's
//   (causal_conv1d.cu) with a null bias, which is what ops::conv1d_update_l2norm passes.
//   The gb10 tree builds with --fmad=false (kernels/gb10/common/KERNEL.toml), and
//   gdn_conv_kn_microtest (model-arch examples) checks outputs, snapshots and the committed
//   state byte for byte against per-token causal_conv1d_update_l2norm calls.
// - After position t the window is written to conv_state_inter + t * inter_stride, the
//   rollback state for accepting t + 1 positions. The last snapshot equals the committed
//   window left in conv_state, so no copy is needed afterwards.
// - A third __syncthreads after the L2 apply orders each position's read of
//   warp_sums[base_warp] before the next position's lane-0 write to the same slot.
//
// Contract: d_conv <= 8 (the window arrays hold 8 floats). As in the per-token kernel, the
// L2 grouping assumes blockDim 256 and head_dim 128 (two heads per block, four warps each),
// and qk_channels a multiple of 256, so no block mixes normalized and plain channels.

















#include <cuda_bf16.h>






// 2026-09-25: Grid (ceil(dim / 256), 1, 1), block 256. Thread ch owns channel ch and keeps
// its d_conv window in registers across positions 0..num_tokens-1.
// conv_state and each snapshot are [dim, d_conv] FP32, weight is [dim, d_conv] BF16.
// Position t reads new_input at t * input_stride and writes output at t * output_stride
// (BF16). Channels below qk_channels are L2-normalized in groups of head_dim.
extern "C" __global__ void gdn_verify_fused_conv_kn(
    float* __restrict__ conv_state,
    const __nv_bfloat16* __restrict__ new_input,
    const __nv_bfloat16* __restrict__ weight,
    __nv_bfloat16* __restrict__ output,
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




    for (unsigned int t = 0; t < num_tokens; t++) {
        float silu = 0.0f;
        if (valid) {

            for (unsigned int i = 0; i < d_conv - 1; i++) win[i] = win[i + 1];
            win[d_conv - 1] = (float)new_input[t * input_stride + ch];

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

            // 2026-09-25: Keeps the next position's lane-0 write to warp_sums behind this read.
            __syncthreads();
        }

        if (valid) output[t * output_stride + ch] = __float2bfloat16(silu);


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










// 2026-09-25: Batched form: gdn_verify_fused_conv_kn with the sequence on gridDim.y.
// Sequence blockIdx.y offsets conv_state, new_input, output and conv_state_inter by its
// *_seq_stride argument. Windows are independent and the per-sequence loop is unchanged,
// so each sequence's results equal a separate gdn_verify_fused_conv_kn launch.
// Grid (ceil(dim / 256), n_seq, 1), block 256.
extern "C" __global__ void gdn_verify_fused_conv_kn_batched(
    float* __restrict__ conv_state,
    const __nv_bfloat16* __restrict__ new_input,
    const __nv_bfloat16* __restrict__ weight,
    __nv_bfloat16* __restrict__ output,
    float* __restrict__ conv_state_inter,
    unsigned int num_tokens,
    unsigned int dim,
    unsigned int d_conv,
    unsigned int qk_channels,
    unsigned int head_dim,
    unsigned int input_stride,
    unsigned int output_stride,
    unsigned int inter_stride,
    float l2_eps,





    unsigned int conv_state_seq_stride,
    unsigned int input_seq_stride,
    unsigned int output_seq_stride,
    unsigned int inter_seq_stride
) {
    const unsigned int seq = blockIdx.y;
    conv_state       += (size_t) seq * conv_state_seq_stride;
    new_input        += (size_t) seq * input_seq_stride;
    output           += (size_t) seq * output_seq_stride;
    conv_state_inter += (size_t) seq * inter_seq_stride;

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




    for (unsigned int t = 0; t < num_tokens; t++) {
        float silu = 0.0f;
        if (valid) {

            for (unsigned int i = 0; i < d_conv - 1; i++) win[i] = win[i + 1];
            win[d_conv - 1] = (float)new_input[t * input_stride + ch];

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

            // 2026-09-25: Keeps the next position's lane-0 write to warp_sums behind this read.
            __syncthreads();
        }

        if (valid) output[t * output_stride + ch] = __float2bfloat16(silu);


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
