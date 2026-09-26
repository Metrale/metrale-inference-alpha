// SPDX-License-Identifier: MIT OR Apache-2.0

// 2026-09-25: Hopper twin of `dense_gemm_ba_gates_prefill`
// (kernels/gb10/common/ssm_preprocess.cu): the SSM BA projection and the GDN
// gate transforms, one CTA per token.
//
// Owner: hopper kernels.
// Invariants:
// - The outputs are bit-identical to the parent's, because the parent's
//   reduction order is kept:
//   1. per lane, `for (kv = lane; kv < K/8; kv += 64)` over uint4 vectors,
//      and inside each uint4 the eight bf16 in index order (low, high half
//      of words 0..3);
//   2. the `__shfl_down_sync` butterfly at offsets 16, 8, 4, 2, 1;
//   3. the cross-warp sum as `warp_even + warp_odd`;
//   4. the same sigmoid / softplus / exp transforms and output indices.
//   `__bfloat162float` is an exact widening, so converting A once per uint4
//   instead of once per output changes no operand.
//   `native_ssm_ba_gates_hopper_microtest` checks byte equality.
//
// The parent runs grid (ceil(N/4), M) and gives each output 64 lanes that
// sweep all of K, so every output re-reads and re-converts the token's
// activation row. Here one CTA covers all N outputs of a token and a thread
// keeps its lane while accumulating BAH_GROUPS output groups, so one read
// and one conversion of the row serve BAH_GROUPS outputs.
// Measurements: SSM-BA-GATES-ATTRIBUTION.md.





































#include <cuda_bf16.h>
#include <math.h>

// 2026-09-25: The parent's block shape: 256 threads split into BAH_OUTS outputs of
// BAH_LANES lanes each. Changing any of the three changes the reduction
// order and breaks the bit-identity above.
#define BAH_BLOCK 256
#define BAH_LANES 64
#define BAH_OUTS (BAH_BLOCK / BAH_LANES)
#define BAH_WARPS (BAH_BLOCK / 32)

// 2026-09-25: Output groups (of BAH_OUTS outputs) a thread accumulates at once. One
// A fetch serves this many groups; each group costs a thread one
// accumulator and one B row pointer.












#define BAH_GROUPS 8








/// 2026-09-25: One CTA per token; all N BA outputs, the parent's arithmetic, in order.
///
/// Output layout is the parent's: `[gate(nv), beta(nv)]` per token, stride
/// `gate_stride`.
///
///   gate_out[token * gate_stride + vh]      = gate  (alpha -> exp transform)
///   gate_out[token * gate_stride + nv + vh] = beta  (sigmoid)
///
/// Grid: (M_tokens, 1, 1)  Block: (256, 1, 1)  Shared: 256 B, static.
///
/// `ssm_ba_gates_hopper_reject` (crates/model-layers/src/layers/ops/
/// ssm_ba_gates_hopper.rs) sends a launch to the parent unless K % 8 == 0
/// (the uint4 sweep), K_stride >= K and M >= `ba_gates_min_tokens`.


extern "C" __global__ __launch_bounds__(BAH_BLOCK) void dense_gemm_ba_gates_prefill_hopper(
    const __nv_bfloat16* __restrict__ A,  // 2026-09-25: [M, K_stride] activations
    const __nv_bfloat16* __restrict__ B,  // 2026-09-25: [N, K] BA weight (row-major)
    const float* __restrict__ A_log,      // 2026-09-25: [nv] learned A_log parameter
    const float* __restrict__ dt_bias,    // 2026-09-25: [nv] learned dt_bias parameter
    float* __restrict__ gate_out,         // 2026-09-25: [M, gate_stride] FP32 output
    unsigned int M,                       // 2026-09-25: num_tokens
    unsigned int N,                       // 2026-09-25: ssm_ba_size (2 * nv)
    unsigned int K,                       // 2026-09-25: hidden_size
    unsigned int K_stride,                // 2026-09-25: BF16 elements per token in A
    unsigned int gate_stride,             // 2026-09-25: FP32 elements per token in gate_out
    unsigned int nv,                      // 2026-09-25: num_v_heads
    unsigned int vheads_per_group
) {
    const unsigned int token = blockIdx.x;
    if (token >= M) return;

    // 2026-09-25: Same decomposition as the parent, so the lane -> k mapping is the same.
    const unsigned int local_out = threadIdx.x / BAH_LANES;
    const unsigned int lane = threadIdx.x % BAH_LANES;
    // 2026-09-25: The parent writes `smem[local_out * 2 + (lane / 32)]`, which is the
    // block-wide warp index; spelled that way here.
    const unsigned int warp = threadIdx.x / 32;

    const unsigned int K_VEC = K / 8;
    const uint4* __restrict__ A_vec = (const uint4*)(A + (unsigned long long)token * K_stride);

    // 2026-09-25: BAH_GROUPS groups x BAH_WARPS warp partials. Reused every tile, which
    // is why the tile loop syncs before writing as well as after.
    __shared__ float red[BAH_GROUPS * BAH_WARPS];

    const unsigned int n_groups = (N + BAH_OUTS - 1) / BAH_OUTS;

    for (unsigned int g0 = 0; g0 < n_groups; g0 += BAH_GROUPS) {
        float acc[BAH_GROUPS];
        const uint4* __restrict__ B_vec[BAH_GROUPS];
        #pragma unroll
        for (int g = 0; g < BAH_GROUPS; g++) {
            acc[g] = 0.0f;
            unsigned int n = (g0 + (unsigned int)g) * BAH_OUTS + local_out;
            // 2026-09-25: Inactive outputs (N not a multiple of BAH_GROUPS*BAH_OUTS) read
            // row 0 so the address is in bounds; their acc is never written.
            unsigned int row = (n < N) ? n : 0u;
            B_vec[g] = (const uint4*)(B + (unsigned long long)row * K);
        }

        // 2026-09-25: The parent's K sweep, with the A conversion hoisted out of the
        // output loop. Element index is kv*8 + (2*i) for the low half of word
        // i and kv*8 + (2*i+1) for the high half, the parent's order.
        for (unsigned int kv = lane; kv < K_VEC; kv += BAH_LANES) {
            uint4 a_data = A_vec[kv];
            const unsigned int a_raw[4] = {a_data.x, a_data.y, a_data.z, a_data.w};
            float af[8];
            #pragma unroll
            for (int i = 0; i < 4; i++) {
                __nv_bfloat16 a_lo, a_hi;
                *(unsigned short*)&a_lo = (unsigned short)(a_raw[i] & 0xFFFF);
                *(unsigned short*)&a_hi = (unsigned short)(a_raw[i] >> 16);
                af[2 * i] = __bfloat162float(a_lo);
                af[2 * i + 1] = __bfloat162float(a_hi);
            }
            #pragma unroll
            for (int g = 0; g < BAH_GROUPS; g++) {
                uint4 b_data = B_vec[g][kv];
                const unsigned int b_raw[4] = {b_data.x, b_data.y, b_data.z, b_data.w};
                #pragma unroll
                for (int i = 0; i < 4; i++) {
                    __nv_bfloat16 b_lo, b_hi;
                    *(unsigned short*)&b_lo = (unsigned short)(b_raw[i] & 0xFFFF);
                    *(unsigned short*)&b_hi = (unsigned short)(b_raw[i] >> 16);
                    acc[g] += af[2 * i] * __bfloat162float(b_lo);
                    acc[g] += af[2 * i + 1] * __bfloat162float(b_hi);
                }
            }
        }

        // 2026-09-25: Warp shuffle reduction: the parent's offsets, in the parent's order.
        #pragma unroll
        for (int g = 0; g < BAH_GROUPS; g++) {
            #pragma unroll
            for (int offset = 16; offset > 0; offset >>= 1) {
                acc[g] += __shfl_down_sync(0xFFFFFFFF, acc[g], offset);
            }
        }

        // 2026-09-25: The previous tile's partials must be fully consumed before this one
        // overwrites them; `red` is the only state carried across tiles.
        __syncthreads();
        if ((threadIdx.x % 32) == 0) {
            #pragma unroll
            for (int g = 0; g < BAH_GROUPS; g++) {
                red[g * BAH_WARPS + warp] = acc[g];
            }
        }
        __syncthreads();

        // 2026-09-25: Cross-warp sum + transforms. One thread per (group, output), and the
        // sum is `even + odd` as the parent's
        // `smem[local_out*2] + smem[local_out*2 + 1]`.
        if (threadIdx.x < BAH_GROUPS * BAH_OUTS) {
            const unsigned int g = threadIdx.x / BAH_OUTS;
            const unsigned int lo = threadIdx.x % BAH_OUTS;
            const unsigned int n = (g0 + g) * BAH_OUTS + lo;
            if (n < N) {
                float result = red[g * BAH_WARPS + lo * 2] + red[g * BAH_WARPS + lo * 2 + 1];
                unsigned int group_dim_ba = 2 * vheads_per_group;
                unsigned int within_group = n % group_dim_ba;
                unsigned int group = n / group_dim_ba;

                float* gate_tok = gate_out + (unsigned long long)token * gate_stride;

                if (within_group < vheads_per_group) {
                    // 2026-09-25: Beta element: sigmoid(b_raw), stored at offset nv.
                    unsigned int vh = group * vheads_per_group + within_group;
                    gate_tok[nv + vh] = 1.0f / (1.0f + __expf(-result));
                } else {
                    // 2026-09-25: Alpha (gate): exp(-exp(A_log) * softplus(alpha + dt_bias)).
                    unsigned int vh = group * vheads_per_group + (within_group - vheads_per_group);
                    float a_log_val = A_log[vh];
                    float dt_b = dt_bias[vh];
                    float A_val = __expf(fminf(a_log_val, 20.0f));
                    float dt = __logf(1.0f + __expf(fminf(result + dt_b, 20.0f)));
                    gate_tok[vh] = __expf(-A_val * dt);
                }
            }
        }
    }
}
