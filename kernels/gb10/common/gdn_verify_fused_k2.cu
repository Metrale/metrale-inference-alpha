// SPDX-License-Identifier: MIT OR Apache-2.0

// 2026-09-25: Fused K=2 GDN verify epilogue: conv1d + SiLU + L2-norm for both positions in
// one launch, and the gated RMS norm for both positions in one launch.
//
// Owner: gb10 kernels.
// Invariants:
// - gdn_verify_fused_conv_k2 keeps each channel's d_conv window in registers across both
//   positions, writes both conv outputs, snapshots the window after position 0 to
//   conv_state_inter, and leaves the window after position 1 in conv_state.
// - Per position, the conv arithmetic and its order are causal_conv1d_update_l2norm's
//   (causal_conv1d.cu) with a null bias, which is what ops::conv1d_update_l2norm passes,
//   and the norm arithmetic is gated_rms_norm's (rms_norm.cu). The gb10 tree builds with
//   --fmad=false (KERNEL.toml).
// gdn_verify_fused_microtest (model-arch examples) checks cos >= 0.99999 against the
// per-token path. Qwen3SsmLayer::fused_verify_k2_enabled (qwen3_ssm/
// trait_decode_batched_conv_gdn.rs) runs these kernels only with METRALE_GDN_FUSED_VERIFY=1
// and both handles linked.
// Contract: d_conv <= 8 (the window arrays hold 8 floats); the conv L2 grouping assumes
// blockDim 256, head_dim 128 and qk_channels a multiple of 256, as the per-token kernel does.









#include <cuda_bf16.h>

// 2026-09-25: Copies of unpack_bf16x2, pack_bf16x2 and warp_reduce_sum from rms_norm.cu,
// with the same instructions, so the norm rounds as gated_rms_norm does.

__device__ __forceinline__ void fused_unpack_bf16x2(unsigned int packed, float& v0, float& v1) {
    v0 = __bfloat162float(__ushort_as_bfloat16((unsigned short)(packed & 0xFFFF)));
    v1 = __bfloat162float(__ushort_as_bfloat16((unsigned short)(packed >> 16)));
}
__device__ __forceinline__ unsigned int fused_pack_bf16x2(float v0, float v1) {
    unsigned int lo = (unsigned int)__bfloat16_as_ushort(__float2bfloat16(v0));
    unsigned int hi = (unsigned int)__bfloat16_as_ushort(__float2bfloat16(v1));
    return lo | (hi << 16);
}
__device__ __forceinline__ float fused_warp_reduce_sum(float val) {
    for (int offset = 16; offset > 0; offset >>= 1)
        val += __shfl_xor_sync(0xFFFFFFFF, val, offset);
    return val;
}






// 2026-09-25: Grid (ceil(dim / 256), 1, 1), block 256. Thread ch owns channel ch and keeps
// its d_conv window in registers across positions 0 and 1. conv_state and conv_state_inter
// are [dim, d_conv] FP32, weight is [dim, d_conv] BF16. Position t reads new_input at
// t * input_stride and writes output at t * output_stride (BF16). Channels below
// qk_channels are L2-normalized in groups of head_dim.
extern "C" __global__ void gdn_verify_fused_conv_k2(
    float* __restrict__ conv_state,
    const __nv_bfloat16* __restrict__ new_input,
    const __nv_bfloat16* __restrict__ weight,
    __nv_bfloat16* __restrict__ output,
    float* __restrict__ conv_state_inter,
    unsigned int dim,
    unsigned int d_conv,
    unsigned int qk_channels,
    unsigned int head_dim,
    unsigned int input_stride,
    unsigned int output_stride,
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



    for (unsigned int t = 0; t < 2; t++) {
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
        }

        if (valid) output[t * output_stride + ch] = __float2bfloat16(silu);


        if (valid && t == 0) {
            float* snap = conv_state_inter + ch * d_conv;
            for (unsigned int i = 0; i < d_conv; i++) snap[i] = win[i];
        }
    }


    if (valid) {
        float* state = conv_state + ch * d_conv;
        for (unsigned int i = 0; i < d_conv; i++) state[i] = win[i];
    }
}




// 2026-09-25: Grid (num_v_heads, 2, 1). Block (head, t) normalizes one hidden_size group as
// one row of gated_rms_norm does: x at gdn_out + t * out_stride + head * hidden_size, the
// gate at deint + t * deint_stride + z_offset + head * hidden_size, the result at
// output + t * out_stride + head * hidden_size. hidden_size must be a multiple of 4 and at
// most 16 * blockDim (x_cache holds 16 floats per thread).
extern "C" __global__ void gdn_verify_fused_norm_k2(
    const __nv_bfloat16* __restrict__ gdn_out,
    const __nv_bfloat16* __restrict__ deint,
    const __nv_bfloat16* __restrict__ weight,
    __nv_bfloat16* __restrict__ output,
    unsigned int hidden_size,
    float eps,
    unsigned int deint_stride,
    unsigned int z_offset,
    unsigned int out_stride
) {
    const unsigned int head = blockIdx.x;
    const unsigned int t = blockIdx.y;
    const unsigned int tid = threadIdx.x;



    const __nv_bfloat16* x = gdn_out + t * out_stride + head * hidden_size;


    const __nv_bfloat16* g = deint + t * deint_stride + z_offset + head * hidden_size;
    __nv_bfloat16* out = output + t * out_stride + head * hidden_size;



    const unsigned int quad_size = hidden_size / 4;
    const unsigned long long* x64 = (const unsigned long long*)x;

    float x_cache[16];
    float sum_sq = 0.0f;
    unsigned int n_cached = 0;

    for (unsigned int i = tid; i < quad_size; i += blockDim.x) {
        unsigned long long v = x64[i];
        float f0, f1, f2, f3;
        fused_unpack_bf16x2((unsigned int)v, f0, f1);
        fused_unpack_bf16x2((unsigned int)(v >> 32), f2, f3);
        x_cache[n_cached]     = f0;
        x_cache[n_cached + 1] = f1;
        x_cache[n_cached + 2] = f2;
        x_cache[n_cached + 3] = f3;
        n_cached += 4;
        sum_sq += f0 * f0 + f1 * f1 + f2 * f2 + f3 * f3;
    }

    sum_sq = fused_warp_reduce_sum(sum_sq);

    __shared__ float warp_sums[32];
    const unsigned int warp_id = tid / 32;
    const unsigned int lane_id = tid % 32;
    if (lane_id == 0) warp_sums[warp_id] = sum_sq;
    __syncthreads();
    if (warp_id == 0) {
        float val = (lane_id < (blockDim.x + 31) / 32) ? warp_sums[lane_id] : 0.0f;
        val = fused_warp_reduce_sum(val);
        if (lane_id == 0) warp_sums[0] = val;
    }
    __syncthreads();

    float rms = rsqrtf(warp_sums[0] / (float)hidden_size + eps);


    const unsigned long long* g64 = (const unsigned long long*)g;
    const unsigned long long* w64 = (const unsigned long long*)weight;
    unsigned long long* out64 = (unsigned long long*)out;

    unsigned int ci = 0;
    for (unsigned int i = tid; i < quad_size; i += blockDim.x) {
        float f0 = x_cache[ci];
        float f1 = x_cache[ci + 1];
        float f2 = x_cache[ci + 2];
        float f3 = x_cache[ci + 3];
        ci += 4;

        unsigned long long wv = w64[i];
        float w0, w1, w2, w3;
        fused_unpack_bf16x2((unsigned int)wv, w0, w1);
        fused_unpack_bf16x2((unsigned int)(wv >> 32), w2, w3);

        unsigned long long gv = g64[i];
        float g0, g1, g2, g3;
        fused_unpack_bf16x2((unsigned int)gv, g0, g1);
        fused_unpack_bf16x2((unsigned int)(gv >> 32), g2, g3);

        float s0 = g0 / (1.0f + expf(-g0));
        float s1 = g1 / (1.0f + expf(-g1));
        float s2 = g2 / (1.0f + expf(-g2));
        float s3 = g3 / (1.0f + expf(-g3));

        unsigned int lo = fused_pack_bf16x2(f0 * rms * w0 * s0, f1 * rms * w1 * s1);
        unsigned int hi = fused_pack_bf16x2(f2 * rms * w2 * s2, f3 * rms * w3 * s3);
        out64[i] = ((unsigned long long)hi << 32) | (unsigned long long)lo;
    }
}
