// SPDX-License-Identifier: MIT OR Apache-2.0

// 2026-09-25: Single-token causal depthwise 1-D convolution, one thread per
// channel: output[c] = sum_i weights[c, i] * window[i], where the window is
// conv_state[c] followed by new_input[c]. conv_state then shifts left by one
// and takes new_input[c] as its last entry. No activation is applied.
//
// Layout:
//   weights     : bfloat [num_channels, kernel_size]
//   new_input   : bfloat [num_channels]
//   conv_state  : bfloat [num_channels, kernel_size - 1]    (in/out)
//   output      : bfloat [num_channels]
//
// Owner: metal kernels.
// Invariants: a kernel_size outside 1..MAX_K returns without writing output
// or conv_state.






#include <metal_stdlib>
using namespace metal;

constant uint MAX_K = 8;

kernel void causal_conv1d_decode(
    constant uint &num_channels [[buffer(0)]],
    constant uint &kernel_size  [[buffer(1)]],
    device const bfloat *weights    [[buffer(2)]],
    device const bfloat *new_input  [[buffer(3)]],
    device bfloat       *conv_state [[buffer(4)]],
    device bfloat       *output     [[buffer(5)]],
    uint c [[thread_position_in_grid]])
{
    if (c >= num_channels) {
        return;
    }
    if (kernel_size < 1u || kernel_size > MAX_K) {
        return;
    }

    uint k = kernel_size;
    uint state_len = k - 1u;




    float past[MAX_K];
    for (uint i = 0; i < state_len; ++i) {
        past[i] = float(conv_state[c * state_len + i]);
    }
    past[state_len] = float(new_input[c]);


    float acc = 0.0f;
    for (uint i = 0; i < k; ++i) {
        acc += float(weights[c * k + i]) * past[i];
    }
    output[c] = bfloat(acc);


    for (uint i = 0; i < state_len; ++i) {
        conv_state[c * state_len + i] = bfloat(past[i + 1u]);
    }
}
