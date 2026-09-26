// SPDX-License-Identifier: MIT OR Apache-2.0

// 2026-09-25: Kernels `wht_bf16_inplace` and `wht_bf16_inplace_inv`: in-place normalised Walsh-Hadamard transform
// of each head of a BF16 [num_heads, head_dim] buffer, and its inverse. Grid (num_heads, 1, 1), Block (32, 1, 1).
// With TQ_PLUS_SIGNS the forward applies signs1, the butterfly, then signs2 (tq_plus_signs.cuh); the inverse reverses.
// Owner: gb10 kernels.
// Invariants: none beyond the types. head_dim must be 128, 256 or 512; nothing checks it.

#include <cuda_bf16.h>

#ifdef TQ_PLUS_SIGNS
#include "tq_plus_signs.cuh"
#endif

// 2026-09-25: 256-point transform across a warp, 8 values per lane: butterflies at strides 1, 2 and 4 within a lane,
// then xor-shuffles at masks 1 to 16 across lanes, then a scale by 1/16.
__device__ __forceinline__ void wht256_warp_bf16(float vals[8], unsigned int lane) {

    #pragma unroll
    for (int stride = 1; stride <= 4; stride <<= 1) {
        #pragma unroll
        for (int i = 0; i < 8; i += stride * 2) {
            for (int j = 0; j < stride; j++) {
                float a = vals[i + j];
                float b = vals[i + j + stride];
                vals[i + j] = a + b;
                vals[i + j + stride] = a - b;
            }
        }
    }

    #pragma unroll
    for (int xor_mask = 1; xor_mask <= 16; xor_mask <<= 1) {
        #pragma unroll
        for (int i = 0; i < 8; i++) {
            float other = __shfl_xor_sync(0xFFFFFFFF, vals[i], xor_mask);
            if (lane & xor_mask)
                vals[i] = other - vals[i];
            else
                vals[i] = vals[i] + other;
        }
    }

    #pragma unroll
    for (int i = 0; i < 8; i++) vals[i] *= 0.0625f;
}

// 2026-09-25: head_dim >= 512 transforms the first 512 values, >= 256 the first 256, and anything smaller takes the
// 128-point path.

extern "C" __global__ void wht_bf16_inplace(
    __nv_bfloat16* __restrict__ data,
    const unsigned int head_dim
) {
    const unsigned int head = blockIdx.x;
    const unsigned int lane = threadIdx.x;
    if (lane >= 32) return;

    __nv_bfloat16* head_data = data + (unsigned long long)head * head_dim;

    if (head_dim >= 512) {

        float vals[16];
        #pragma unroll
        for (int i = 0; i < 16; i++)
            vals[i] = __bfloat162float(head_data[lane * 16 + i]);

#ifdef TQ_PLUS_SIGNS

        tq_plus::apply_signs_512_pre(vals, lane);
#endif

        for (int stride = 1; stride <= 8; stride <<= 1) {
            for (int i = 0; i < 16; i += stride * 2) {
                for (int j = 0; j < stride; j++) {
                    float a = vals[i + j];
                    float b = vals[i + j + stride];
                    vals[i + j] = a + b;
                    vals[i + j + stride] = a - b;
                }
            }
        }

        for (int xor_mask = 1; xor_mask <= 16; xor_mask <<= 1) {
            for (int i = 0; i < 16; i++) {
                float other = __shfl_xor_sync(0xFFFFFFFF, vals[i], xor_mask);
                if (lane & xor_mask) vals[i] = other - vals[i];
                else vals[i] = vals[i] + other;
            }
        }

        float norm = 1.0f / sqrtf(512.0f);
        for (int i = 0; i < 16; i++) vals[i] *= norm;
#ifdef TQ_PLUS_SIGNS

        tq_plus::apply_signs_512_post(vals, lane);
#endif

        #pragma unroll
        for (int i = 0; i < 16; i++)
            head_data[lane * 16 + i] = __float2bfloat16(vals[i]);
    } else if (head_dim >= 256) {

        float vals[8];
        #pragma unroll
        for (int i = 0; i < 8; i++)
            vals[i] = __bfloat162float(head_data[lane * 8 + i]);
#ifdef TQ_PLUS_SIGNS

        tq_plus::apply_signs_256_pre(vals, lane);
#endif
        wht256_warp_bf16(vals, lane);
#ifdef TQ_PLUS_SIGNS

        tq_plus::apply_signs_256_post(vals, lane);
#endif
        #pragma unroll
        for (int i = 0; i < 8; i++)
            head_data[lane * 8 + i] = __float2bfloat16(vals[i]);
    } else {

        float vals[4];
        unsigned int elems_per_thread = head_dim / 32;
        #pragma unroll
        for (unsigned int i = 0; i < 4; i++) {
            unsigned int idx = lane * elems_per_thread + i;
            vals[i] = (idx < head_dim) ? __bfloat162float(head_data[idx]) : 0.0f;
        }

#ifdef TQ_PLUS_SIGNS


        if (head_dim == 128) tq_plus::apply_signs_128_pre(vals, lane);
#endif



        for (int stride = 1; stride <= 2; stride <<= 1) {
            for (int i = 0; i < 4; i += stride * 2) {
                for (int j = 0; j < stride; j++) {
                    float a = vals[i + j];
                    float b = vals[i + j + stride];
                    vals[i + j] = a + b;
                    vals[i + j + stride] = a - b;
                }
            }
        }

        for (int xor_mask = 1; xor_mask <= 16; xor_mask <<= 1) {
            for (int i = 0; i < 4; i++) {
                float other = __shfl_xor_sync(0xFFFFFFFF, vals[i], xor_mask);
                if (lane & xor_mask) vals[i] = other - vals[i];
                else vals[i] = vals[i] + other;
            }
        }

        float norm = 1.0f / sqrtf((float)head_dim);
        for (int i = 0; i < 4; i++) vals[i] *= norm;

#ifdef TQ_PLUS_SIGNS

        if (head_dim == 128) tq_plus::apply_signs_128_post(vals, lane);
#endif

        #pragma unroll
        for (unsigned int i = 0; i < 4; i++) {
            unsigned int idx = lane * elems_per_thread + i;
            if (idx < head_dim) head_data[idx] = __float2bfloat16(vals[i]);
        }
    }
}

// 2026-09-25: Inverse of `wht_bf16_inplace`. Without TQ_PLUS_SIGNS it is the same transform: the normalised WHT is its
// own inverse. With it the signs run in reverse order, because the forward S2*H*S1 applied twice is not the identity
// when S1 != S2.







extern "C" __global__ void wht_bf16_inplace_inv(
    __nv_bfloat16* __restrict__ data,
    const unsigned int head_dim
) {
    const unsigned int head = blockIdx.x;
    const unsigned int lane = threadIdx.x;
    if (lane >= 32) return;

    __nv_bfloat16* head_data = data + (unsigned long long)head * head_dim;

    if (head_dim >= 512) {


        float vals[16];
        #pragma unroll
        for (int i = 0; i < 16; i++)
            vals[i] = __bfloat162float(head_data[lane * 16 + i]);

#ifdef TQ_PLUS_SIGNS

        tq_plus::apply_signs_512_post(vals, lane);
#endif
                for (int stride = 1; stride <= 8; stride <<= 1) {
            for (int i = 0; i < 16; i += stride * 2) {
                for (int j = 0; j < stride; j++) {
                    float a = vals[i + j];
                    float b = vals[i + j + stride];
                    vals[i + j] = a + b;
                    vals[i + j + stride] = a - b;
                }
            }
        }
        for (int xor_mask = 1; xor_mask <= 16; xor_mask <<= 1) {
            for (int i = 0; i < 16; i++) {
                float other = __shfl_xor_sync(0xFFFFFFFF, vals[i], xor_mask);
                if (lane & xor_mask) vals[i] = other - vals[i];
                else vals[i] = vals[i] + other;
            }
        }
        float norm = 1.0f / sqrtf(512.0f);
        for (int i = 0; i < 16; i++) vals[i] *= norm;
#ifdef TQ_PLUS_SIGNS

        tq_plus::apply_signs_512_pre(vals, lane);
#endif
        #pragma unroll
        for (int i = 0; i < 16; i++)
            head_data[lane * 16 + i] = __float2bfloat16(vals[i]);
    } else if (head_dim >= 256) {

        float vals[8];
        #pragma unroll
        for (int i = 0; i < 8; i++)
            vals[i] = __bfloat162float(head_data[lane * 8 + i]);
#ifdef TQ_PLUS_SIGNS

        tq_plus::apply_signs_256_post(vals, lane);
#endif
        wht256_warp_bf16(vals, lane);
#ifdef TQ_PLUS_SIGNS

        tq_plus::apply_signs_256_pre(vals, lane);
#endif
        #pragma unroll
        for (int i = 0; i < 8; i++)
            head_data[lane * 8 + i] = __float2bfloat16(vals[i]);
    } else {

        float vals[4];
        unsigned int elems_per_thread = head_dim / 32;
        #pragma unroll
        for (unsigned int i = 0; i < 4; i++) {
            unsigned int idx = lane * elems_per_thread + i;
            vals[i] = (idx < head_dim) ? __bfloat162float(head_data[idx]) : 0.0f;
        }

#ifdef TQ_PLUS_SIGNS

        if (head_dim == 128) tq_plus::apply_signs_128_post(vals, lane);
#endif

        for (int stride = 1; stride <= 2; stride <<= 1) {
            for (int i = 0; i < 4; i += stride * 2) {
                for (int j = 0; j < stride; j++) {
                    float a = vals[i + j];
                    float b = vals[i + j + stride];
                    vals[i + j] = a + b;
                    vals[i + j + stride] = a - b;
                }
            }
        }
        for (int xor_mask = 1; xor_mask <= 16; xor_mask <<= 1) {
            for (int i = 0; i < 4; i++) {
                float other = __shfl_xor_sync(0xFFFFFFFF, vals[i], xor_mask);
                if (lane & xor_mask) vals[i] = other - vals[i];
                else vals[i] = vals[i] + other;
            }
        }
        float norm = 1.0f / sqrtf((float)head_dim);
        for (int i = 0; i < 4; i++) vals[i] *= norm;

#ifdef TQ_PLUS_SIGNS

        if (head_dim == 128) tq_plus::apply_signs_128_pre(vals, lane);
#endif

        #pragma unroll
        for (unsigned int i = 0; i < 4; i++) {
            unsigned int idx = lane * elems_per_thread + i;
            if (idx < head_dim) head_data[idx] = __float2bfloat16(vals[i]);
        }
    }
}
