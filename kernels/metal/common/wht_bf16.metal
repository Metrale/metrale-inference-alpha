// SPDX-License-Identifier: MIT OR Apache-2.0

// 2026-09-25: In-place Walsh-Hadamard transform of each head of a BF16
// [num_heads, head_dim] tensor, normalized by 1/sqrt(n), one threadgroup of
// 32 threads (one simdgroup) per head. wht_bf16_inplace is the forward
// transform and wht_bf16_inplace_inv its inverse.
//
// With TQ_PLUS_SIGNS defined, the forward transform is S2·H·S1 with
// Rademacher sign tables S1, S2 and the inverse is S1·H·S2; without it both
// are the plain self-inverse H. The sign tables hold the same values as
// kernels/gb10/common/tq_plus_signs.cuh.
//
// Owner: metal kernels.
// Invariants: assumes head_dim is 128 or 256; a head_dim >= 256 transforms
// the first 256 elements and a smaller one transforms 128.


#include <metal_stdlib>
using namespace metal;

#ifdef TQ_PLUS_SIGNS


constant float TQP_SIGNS1_128[128] = {
    -1, 1, 1, -1, -1, 1, -1, 1, -1, -1, 1, 1, 1, 1, 1, 1,
    1, -1, 1, -1, 1, -1, -1, 1, 1, 1, -1, 1, 1, -1, -1, -1,
    -1, 1, 1, -1, 1, 1, -1, 1, -1, 1, 1, -1, -1, 1, -1, 1,
    1, 1, 1, -1, -1, -1, -1, -1, 1, -1, 1, 1, 1, 1, -1, 1,
    -1, -1, 1, -1, -1, -1, 1, -1, -1, -1, 1, -1, -1, -1, 1, 1,
    1, -1, -1, 1, 1, 1, -1, -1, 1, 1, -1, 1, 1, -1, 1, -1,
    -1, 1, 1, -1, 1, -1, 1, -1, 1, 1, 1, 1, -1, 1, -1, 1,
    1, -1, 1, 1, -1, -1, -1, -1, -1, 1, 1, -1, 1, 1, -1, 1
};
constant float TQP_SIGNS2_128[128] = {
    1, 1, 1, 1, -1, 1, 1, -1, 1, -1, -1, -1, 1, -1, -1, -1,
    1, 1, -1, -1, 1, -1, 1, -1, 1, -1, -1, 1, -1, 1, 1, 1,
    1, 1, -1, -1, -1, 1, -1, -1, -1, -1, -1, -1, 1, 1, 1, -1,
    1, -1, 1, 1, 1, -1, -1, 1, -1, -1, -1, -1, -1, -1, 1, 1,
    1, -1, 1, -1, -1, -1, -1, 1, -1, 1, -1, 1, -1, -1, 1, 1,
    -1, 1, -1, 1, 1, -1, 1, -1, -1, -1, -1, 1, -1, -1, 1, -1,
    1, -1, 1, 1, 1, -1, -1, 1, -1, 1, -1, 1, 1, -1, -1, 1,
    -1, 1, -1, 1, 1, -1, 1, -1, 1, -1, -1, -1, -1, -1, 1, -1
};
constant float TQP_SIGNS1_256[256] = {
    -1, 1, -1, -1, -1, 1, -1, -1, -1, 1, -1, -1, -1, -1, 1, -1,
    1, 1, 1, -1, 1, -1, 1, 1, 1, 1, 1, 1, 1, 1, -1, -1,
    1, 1, 1, -1, 1, -1, -1, -1, -1, -1, 1, 1, 1, 1, 1, -1,
    1, 1, -1, 1, -1, 1, -1, 1, 1, -1, -1, -1, -1, -1, -1, -1,
    -1, 1, 1, -1, 1, 1, 1, 1, -1, 1, -1, 1, 1, 1, -1, 1,
    -1, 1, -1, 1, -1, -1, 1, -1, 1, 1, 1, 1, 1, 1, 1, 1,
    1, 1, 1, -1, -1, 1, 1, 1, 1, 1, 1, 1, 1, -1, 1, -1,
    1, 1, -1, 1, -1, 1, 1, -1, 1, -1, 1, -1, -1, 1, 1, -1,
    1, 1, 1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 1, -1, 1, 1,
    1, -1, -1, -1, -1, 1, -1, -1, -1, -1, -1, 1, -1, 1, -1, 1,
    -1, -1, 1, 1, 1, -1, 1, -1, -1, 1, 1, -1, -1, 1, 1, 1,
    -1, -1, -1, -1, -1, -1, 1, -1, -1, -1, 1, -1, -1, 1, -1, -1,
    -1, -1, -1, 1, 1, 1, -1, -1, -1, 1, -1, -1, 1, -1, 1, 1,
    1, -1, -1, -1, -1, -1, -1, -1, 1, -1, 1, -1, -1, -1, 1, 1,
    1, 1, -1, 1, -1, -1, 1, 1, 1, 1, 1, 1, 1, 1, -1, 1,
    1, -1, 1, -1, -1, 1, -1, -1, -1, -1, 1, -1, 1, -1, -1, -1
};
constant float TQP_SIGNS2_256[256] = {
    -1, 1, 1, -1, -1, 1, -1, -1, -1, 1, 1, 1, -1, -1, 1, 1,
    1, 1, -1, 1, -1, 1, -1, 1, 1, 1, 1, -1, 1, -1, -1, -1,
    -1, 1, -1, -1, -1, 1, 1, 1, 1, -1, -1, 1, -1, -1, -1, 1,
    1, -1, 1, 1, 1, 1, 1, -1, 1, -1, -1, 1, -1, -1, -1, 1,
    1, -1, 1, 1, 1, 1, -1, -1, 1, 1, -1, -1, 1, -1, 1, 1,
    -1, -1, 1, 1, -1, 1, -1, 1, -1, -1, -1, 1, 1, -1, 1, -1,
    -1, 1, 1, -1, 1, 1, -1, -1, 1, -1, -1, 1, -1, -1, 1, 1,
    -1, -1, 1, 1, 1, 1, -1, 1, 1, -1, 1, -1, 1, 1, 1, -1,
    -1, 1, -1, -1, -1, -1, -1, -1, 1, 1, -1, 1, 1, 1, 1, -1,
    1, 1, -1, -1, 1, 1, 1, 1, 1, 1, -1, 1, 1, -1, -1, -1,
    -1, 1, 1, 1, 1, 1, 1, -1, 1, -1, -1, 1, -1, 1, -1, 1,
    -1, 1, 1, 1, 1, 1, -1, -1, -1, 1, -1, 1, 1, -1, -1, 1,
    -1, 1, 1, 1, 1, 1, -1, -1, 1, 1, -1, -1, 1, -1, 1, -1,
    1, -1, 1, -1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1,
    1, 1, -1, 1, -1, 1, -1, 1, -1, -1, 1, -1, -1, 1, -1, -1,
    -1, 1, -1, 1, -1, -1, -1, 1, -1, -1, 1, -1, 1, 1, -1, 1
};
#endif

// 2026-09-25: Butterfly network over the VPT values each of the 32 lanes holds:
// strides 1..VPT/2 within a thread, then masks 1..16 across lanes through
// simd_shuffle_xor, then a 1/sqrt(32 * VPT) normalization.

template <int VPT>
static inline void wht_simdgroup(thread float (&vals)[VPT], uint lane) {
    for (int stride = 1; stride <= VPT / 2; stride <<= 1) {
        for (int i = 0; i < VPT; i += stride * 2) {
            for (int j = 0; j < stride; j++) {
                float a = vals[i + j];
                float b = vals[i + j + stride];
                vals[i + j] = a + b;
                vals[i + j + stride] = a - b;
            }
        }
    }
    for (int xor_mask = 1; xor_mask <= 16; xor_mask <<= 1) {
        for (int i = 0; i < VPT; i++) {
            float other = simd_shuffle_xor(vals[i], (ushort)xor_mask);
            vals[i] = (lane & xor_mask) ? (other - vals[i]) : (vals[i] + other);
        }
    }
    float norm = 1.0f / sqrt(float(32 * VPT));
    for (int i = 0; i < VPT; i++) vals[i] *= norm;
}




template <int VPT, int DIRECTION>
static inline void wht_head(device bfloat *head_data, uint lane
#ifdef TQ_PLUS_SIGNS
                            , constant float *signs1, constant float *signs2
#endif
) {
    float vals[VPT];
    for (int i = 0; i < VPT; i++) {
        vals[i] = float(head_data[lane * VPT + i]);
    }
#ifdef TQ_PLUS_SIGNS
    constant float *pre = (DIRECTION == 0) ? signs1 : signs2;
    for (int i = 0; i < VPT; i++) vals[i] *= pre[lane * VPT + i];
#endif
    wht_simdgroup<VPT>(vals, lane);
#ifdef TQ_PLUS_SIGNS
    constant float *post = (DIRECTION == 0) ? signs2 : signs1;
    for (int i = 0; i < VPT; i++) vals[i] *= post[lane * VPT + i];
#endif
    for (int i = 0; i < VPT; i++) {
        head_data[lane * VPT + i] = bfloat(vals[i]);
    }
}

#ifdef TQ_PLUS_SIGNS
#define WHT_DISPATCH(VPT, DIR, S1, S2) \
    wht_head<VPT, DIR>(head_data, lane, S1, S2)
#else
#define WHT_DISPATCH(VPT, DIR, S1, S2) wht_head<VPT, DIR>(head_data, lane)
#endif

kernel void wht_bf16_inplace(
    constant uint &head_dim   [[buffer(0)]],
    device bfloat *data       [[buffer(1)]],
    uint head [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_threadgroup]]
) {
    if (lane >= 32) return;
    device bfloat *head_data = data + (ulong)head * head_dim;
    if (head_dim >= 256) {
        WHT_DISPATCH(8, 0, TQP_SIGNS1_256, TQP_SIGNS2_256);
    } else {
        WHT_DISPATCH(4, 0, TQP_SIGNS1_128, TQP_SIGNS2_128);
    }
}

kernel void wht_bf16_inplace_inv(
    constant uint &head_dim   [[buffer(0)]],
    device bfloat *data       [[buffer(1)]],
    uint head [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_threadgroup]]
) {
    if (lane >= 32) return;
    device bfloat *head_data = data + (ulong)head * head_dim;
    if (head_dim >= 256) {
        WHT_DISPATCH(8, 1, TQP_SIGNS1_256, TQP_SIGNS2_256);
    } else {
        WHT_DISPATCH(4, 1, TQP_SIGNS1_128, TQP_SIGNS2_128);
    }
}
