// SPDX-License-Identifier: MIT OR Apache-2.0

// 2026-09-25: 3-D convolution with stride equal to the kernel size (patch
// embedding), plus bias, one thread per output cell:
//
//   weight: bfloat [out_channels, kT, kH, kW, in_channels]
//   bias:   bfloat [out_channels]
//   input:  bfloat [in_channels, t_out * kT, h_out * kH, w_out * kW]
//   output: bfloat [out_channels, t_out, h_out, w_out]
//
// Grid: (out_channels, t_out * h_out * w_out, 1).
//
// Owner: metal kernels.
// Invariants: none beyond the types.






#include <metal_stdlib>
using namespace metal;

kernel void conv3d_patch_embed(
    constant uint &out_channels [[buffer(0)]],
    constant uint &in_channels  [[buffer(1)]],
    constant uint &kt           [[buffer(2)]],
    constant uint &kh           [[buffer(3)]],
    constant uint &kw           [[buffer(4)]],
    constant uint &t_out        [[buffer(5)]],
    constant uint &h_out        [[buffer(6)]],
    constant uint &w_out        [[buffer(7)]],
    device const bfloat *input  [[buffer(8)]],
    device const bfloat *weight [[buffer(9)]],
    device const bfloat *bias   [[buffer(10)]],
    device bfloat       *output [[buffer(11)]],
    uint2 gid [[thread_position_in_grid]])
{
    uint c_out = gid.x;
    uint flat  = gid.y;
    if (c_out >= out_channels || flat >= t_out * h_out * w_out) {
        return;
    }
    uint w_o = flat % w_out;
    uint h_o = (flat / w_out) % h_out;
    uint t_o = flat / (h_out * w_out);



    uint t_in = t_out * kt;
    uint h_in = h_out * kh;
    uint w_in = w_out * kw;


    float acc = float(bias[c_out]);
    for (uint dt = 0; dt < kt; ++dt) {
        uint t_idx = t_o * kt + dt;
        for (uint dh = 0; dh < kh; ++dh) {
            uint h_idx = h_o * kh + dh;
            for (uint dw = 0; dw < kw; ++dw) {
                uint w_idx = w_o * kw + dw;
                for (uint ic = 0; ic < in_channels; ++ic) {

                    uint w_off = (((c_out * kt + dt) * kh + dh) * kw + dw)
                                  * in_channels + ic;

                    uint i_off = ((ic * t_in + t_idx) * h_in + h_idx) * w_in + w_idx;
                    acc += float(weight[w_off]) * float(input[i_off]);
                }
            }
        }
    }
    uint out_idx = ((c_out * t_out + t_o) * h_out + h_o) * w_out + w_o;
    output[out_idx] = bfloat(acc);
}
