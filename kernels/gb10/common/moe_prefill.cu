// SPDX-License-Identifier: MIT OR Apache-2.0

// 2026-09-25: NVFP4 MoE GEMVs for num_tokens tokens that run the routed experts and the
// shared expert in one launch per stage: gate+up, SiLU(gate) * up then down, and the
// weighted sum blended with the sigmoid-gated shared expert.
//
// Owner: gb10 kernels.
// Invariants:
// - In the gate+up and down kernels, blockIdx.y below num_tokens * top_k is routed slot y
//   (token y / top_k, slot y % top_k); a larger y is the shared expert of token
//   y - num_tokens * top_k.
// - Weights are NVFP4: a row-major [N, K / 2] byte matrix, low nibble first, with one
//   FP8 E4M3 scale per 16 inputs ([N, K / 16]) and a per-tensor scale2.
// - K must be a multiple of 8: inputs past the last multiple of 8 are not read.
// - A null routed-expert weight pointer (an expert this rank does not hold) writes zeros.




#include <cuda_bf16.h>
#include <cuda_fp8.h>

#define BLOCK_SIZE 128
#define N_PER_BLOCK 4
#define WARP_SIZE 32
#define GROUP_SIZE 16

__device__ __constant__ float E2M1_LUT_PREFILL[16] = {
    0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f,
    -0.0f, -0.5f, -1.0f, -1.5f, -2.0f, -3.0f, -4.0f, -6.0f
};

// 2026-09-25: Decode an FP8 E4M3 scale byte. The SCALE/HIP branch decodes in software and
// maps the NaN encoding (exponent 15, mantissa 7) to 0.

#if defined(__SCALE__) || defined(__HIP_PLATFORM_AMD__)
__device__ __forceinline__ float metrale_dec_e4m3(unsigned char b) {
    unsigned int s = (b >> 7) & 1u, e = (b >> 3) & 0xFu, m = b & 0x7u; float v;
    if (e == 0u)               v = (float)m * 0.001953125f;
    else if (e == 15u && m == 7u) v = 0.0f;
    else                       v = __uint_as_float(((e + 120u) << 23) | (m << 20));
    return s ? -v : v;
}
#else
__device__ __forceinline__ float metrale_dec_e4m3(unsigned char b) {
    __nv_fp8_e4m3 f; *(unsigned char*)&f = b; return (float)f;
}
#endif

// 2026-09-25: Gate+up. blockIdx.z selects gate (0) or up (1). Each warp computes two
// adjacent outputs, so a block of BLOCK_SIZE (128) threads covers 8.
// Launch: grid (ceil(N / 8), num_tokens * (top_k + 1), 2), block 128.
// Routed results go to row token * top_k + slot of gate_out/up_out, shared results to
// row token of sh_gate_out/sh_up_out.




extern "C" __global__ void moe_expert_gate_up_shared_prefill(
    const __nv_bfloat16* __restrict__ A,

    const unsigned long long* __restrict__ gate_packed_ptrs,
    const unsigned long long* __restrict__ gate_scale_ptrs,
    const float* __restrict__ gate_scale2_vals,
    __nv_bfloat16* __restrict__ gate_out,
    const unsigned long long* __restrict__ up_packed_ptrs,
    const unsigned long long* __restrict__ up_scale_ptrs,
    const float* __restrict__ up_scale2_vals,
    __nv_bfloat16* __restrict__ up_out,
    const unsigned int* __restrict__ expert_indices,

    const unsigned char* __restrict__ sh_gate_packed,
    const unsigned char* __restrict__ sh_gate_scale,
    float sh_gate_s2,
    __nv_bfloat16* __restrict__ sh_gate_out,
    const unsigned char* __restrict__ sh_up_packed,
    const unsigned char* __restrict__ sh_up_scale,
    float sh_up_s2,
    __nv_bfloat16* __restrict__ sh_up_out,
    unsigned int N,
    unsigned int K,
    unsigned int top_k,
    unsigned int num_tokens
) {
    const unsigned int total_routed = num_tokens * top_k;
    const unsigned int y = blockIdx.y;
    const unsigned int proj = blockIdx.z;
    const bool is_shared = (y >= total_routed);


    unsigned int token, expert_slot;
    if (is_shared) {
        token = y - total_routed;
        expert_slot = 0;
    } else {
        token = y / top_k;
        expert_slot = y % top_k;
    }


    const __nv_bfloat16* A_token = A + (unsigned long long)token * K;

    const unsigned char* B_packed;
    const unsigned char* B_scale;
    float s2;
    __nv_bfloat16* C;

    if (is_shared) {
        if (proj == 0) {
            B_packed = sh_gate_packed; B_scale = sh_gate_scale;
            s2 = sh_gate_s2; C = sh_gate_out + (unsigned long long)token * N;
        } else {
            B_packed = sh_up_packed; B_scale = sh_up_scale;
            s2 = sh_up_s2; C = sh_up_out + (unsigned long long)token * N;
        }
    } else {
        const unsigned int expert_id = expert_indices[token * top_k + expert_slot];
        if (proj == 0) {
            B_packed = (const unsigned char*)gate_packed_ptrs[expert_id];
            B_scale = (const unsigned char*)gate_scale_ptrs[expert_id];
            s2 = gate_scale2_vals[expert_id];
            C = gate_out + (unsigned long long)(token * top_k + expert_slot) * N;
        } else {
            B_packed = (const unsigned char*)up_packed_ptrs[expert_id];
            B_scale = (const unsigned char*)up_scale_ptrs[expert_id];
            s2 = up_scale2_vals[expert_id];
            C = up_out + (unsigned long long)(token * top_k + expert_slot) * N;
        }

        if (B_packed == 0) {
            const unsigned int n_base = blockIdx.x * (N_PER_BLOCK * 2);
            for (unsigned int i = threadIdx.x; i < N_PER_BLOCK * 2 && n_base + i < N; i += BLOCK_SIZE) {
                C[n_base + i] = __float2bfloat16(0.0f);
            }
            return;
        }
    }


    const unsigned int threads_per_out = BLOCK_SIZE / N_PER_BLOCK;
    const unsigned int local_out = threadIdx.x / threads_per_out;
    const unsigned int lane = threadIdx.x % threads_per_out;

    const unsigned int n1 = blockIdx.x * (N_PER_BLOCK * 2) + local_out * 2;
    const unsigned int n2 = n1 + 1;
    if (n1 >= N) return;
    const bool have_n2 = (n2 < N);

    const unsigned int half_K = K / 2;
    const unsigned int num_groups = K / GROUP_SIZE;
    const unsigned int K8 = K / 8;

    __shared__ float s_lut[16];
    if (threadIdx.x < 16) s_lut[threadIdx.x] = E2M1_LUT_PREFILL[threadIdx.x];
    __syncthreads();

    float acc1 = 0.0f, acc2 = 0.0f;

    for (unsigned int k8 = lane; k8 < K8; k8 += threads_per_out) {
        uint4 a_data = ((const uint4*)A_token)[k8];
        const unsigned int a_raw[4] = {a_data.x, a_data.y, a_data.z, a_data.w};
        const unsigned int base_k = k8 * 8;

        unsigned int packed4_1 = *(const unsigned int*)(B_packed + (unsigned long long)n1 * half_K + k8 * 4);
        unsigned int sg = base_k / GROUP_SIZE;
        unsigned char sb1 = B_scale[(unsigned long long)n1 * num_groups + sg];
        float sc1 = metrale_dec_e4m3(sb1) * s2;

        unsigned int packed4_2 = have_n2 ?
            *(const unsigned int*)(B_packed + (unsigned long long)n2 * half_K + k8 * 4) : 0;
        unsigned char sb2 = have_n2 ? B_scale[(unsigned long long)n2 * num_groups + sg] : 0;
        float sc2 = have_n2 ? metrale_dec_e4m3(sb2) * s2 : 0.0f;

        #pragma unroll
        for (int b = 0; b < 4; b++) {
            unsigned char bv1 = (packed4_1 >> (b * 8)) & 0xFF;
            float w1l = s_lut[bv1 & 0xF] * sc1, w1h = s_lut[bv1 >> 4] * sc1;
            unsigned char bv2 = (packed4_2 >> (b * 8)) & 0xFF;
            float w2l = s_lut[bv2 & 0xF] * sc2, w2h = s_lut[bv2 >> 4] * sc2;
            __nv_bfloat16 al, ah;
            *(unsigned short*)&al = (unsigned short)(a_raw[b] & 0xFFFF);
            *(unsigned short*)&ah = (unsigned short)(a_raw[b] >> 16);
            float afl = __bfloat162float(al), afh = __bfloat162float(ah);
            acc1 += afl * w1l + afh * w1h;
            acc2 += afl * w2l + afh * w2h;
        }
    }

    #pragma unroll
    for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1)
        acc1 += __shfl_down_sync(0xFFFFFFFF, acc1, offset);
    if (lane == 0) C[n1] = __float2bfloat16(acc1);

    if (have_n2) {
        #pragma unroll
        for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1)
            acc2 += __shfl_down_sync(0xFFFFFFFF, acc2, offset);
        if (lane == 0) C[n2] = __float2bfloat16(acc2);
    }
}

// 2026-09-25: SiLU(gate) * up, then the down GEMV. The block first writes its row's
// activation into shared memory (s_act); then each warp computes two adjacent outputs.
// Launch: grid (ceil(N / 8), num_tokens * (top_k + 1), 1), block 128.






extern "C" __global__ void moe_expert_silu_down_shared_prefill(
    const __nv_bfloat16* __restrict__ gate_out,
    const __nv_bfloat16* __restrict__ up_out,
    const unsigned long long* __restrict__ packed_ptrs,
    const unsigned long long* __restrict__ scale_ptrs,
    const float* __restrict__ scale2_vals,
    __nv_bfloat16* __restrict__ C,
    const unsigned int* __restrict__ expert_indices,

    const __nv_bfloat16* __restrict__ sh_gate_in,
    const __nv_bfloat16* __restrict__ sh_up_in,
    const unsigned char* __restrict__ sh_down_packed,
    const unsigned char* __restrict__ sh_down_scale,
    float sh_down_s2,
    __nv_bfloat16* __restrict__ sh_down_out,
    unsigned int N,
    unsigned int K,
    unsigned int top_k,
    unsigned int num_tokens
) {
    const unsigned int total_routed = num_tokens * top_k;
    const unsigned int y = blockIdx.y;
    const bool is_shared = (y >= total_routed);

    unsigned int token, expert_slot;
    if (is_shared) {
        token = y - total_routed;
        expert_slot = 0;
    } else {
        token = y / top_k;
        expert_slot = y % top_k;
    }

    const unsigned char* B_packed;
    const unsigned char* B_scale;
    float s2;

    const __nv_bfloat16* g_ptr;
    const __nv_bfloat16* u_ptr;

    if (is_shared) {
        B_packed = sh_down_packed; B_scale = sh_down_scale; s2 = sh_down_s2;
        g_ptr = sh_gate_in + (unsigned long long)token * K;
        u_ptr = sh_up_in + (unsigned long long)token * K;
    } else {
        const unsigned int expert_id = expert_indices[token * top_k + expert_slot];
        B_packed = (const unsigned char*)packed_ptrs[expert_id];
        B_scale = (const unsigned char*)scale_ptrs[expert_id];
        s2 = scale2_vals[expert_id];
        g_ptr = gate_out + (unsigned long long)(token * top_k + expert_slot) * K;
        u_ptr = up_out + (unsigned long long)(token * top_k + expert_slot) * K;

        if (B_packed == 0) {
            __nv_bfloat16* out = C + (unsigned long long)(token * top_k + expert_slot) * N;
            const unsigned int n_base = blockIdx.x * (N_PER_BLOCK * 2);
            for (unsigned int i = threadIdx.x; i < N_PER_BLOCK * 2 && n_base + i < N; i += BLOCK_SIZE) {
                out[n_base + i] = __float2bfloat16(0.0f);
            }
            return;
        }
    }

    const unsigned int threads_per_out = BLOCK_SIZE / N_PER_BLOCK;
    const unsigned int local_out = threadIdx.x / threads_per_out;
    const unsigned int lane = threadIdx.x % threads_per_out;

    const unsigned int n1 = blockIdx.x * (N_PER_BLOCK * 2) + local_out * 2;
    const unsigned int n2 = n1 + 1;
    if (n1 >= N) return;
    const bool have_n2 = (n2 < N);

    const unsigned int half_K = K / 2;
    const unsigned int num_groups = K / GROUP_SIZE;
    const unsigned int K8 = K / 8;


    __shared__ float s_lut[16];
    __shared__ float s_act[1024]; // 2026-09-25: K must be at most 1024; the kernel does not check it.

    if (threadIdx.x < 16) s_lut[threadIdx.x] = E2M1_LUT_PREFILL[threadIdx.x];


    for (unsigned int i = threadIdx.x; i < K; i += BLOCK_SIZE) {
        float gf = __bfloat162float(g_ptr[i]);
        float uf = __bfloat162float(u_ptr[i]);
        s_act[i] = (gf / (1.0f + __expf(-gf))) * uf;
    }
    __syncthreads();


    float acc1 = 0.0f, acc2 = 0.0f;

    for (unsigned int k8 = lane; k8 < K8; k8 += threads_per_out) {
        const unsigned int base_k = k8 * 8;

        unsigned int packed4_1 = *(const unsigned int*)(B_packed + (unsigned long long)n1 * half_K + k8 * 4);
        unsigned int sg = base_k / GROUP_SIZE;
        unsigned char sb1 = B_scale[(unsigned long long)n1 * num_groups + sg];
        float sc1 = metrale_dec_e4m3(sb1) * s2;

        unsigned int packed4_2 = have_n2 ?
            *(const unsigned int*)(B_packed + (unsigned long long)n2 * half_K + k8 * 4) : 0;
        unsigned char sb2 = have_n2 ? B_scale[(unsigned long long)n2 * num_groups + sg] : 0;
        float sc2 = have_n2 ? metrale_dec_e4m3(sb2) * s2 : 0.0f;

        #pragma unroll
        for (int b = 0; b < 4; b++) {
            float al = s_act[base_k + b * 2];
            float ah = s_act[base_k + b * 2 + 1];

            unsigned char bv1 = (packed4_1 >> (b * 8)) & 0xFF;
            float w1l = s_lut[bv1 & 0xF] * sc1, w1h = s_lut[bv1 >> 4] * sc1;
            unsigned char bv2 = (packed4_2 >> (b * 8)) & 0xFF;
            float w2l = s_lut[bv2 & 0xF] * sc2, w2h = s_lut[bv2 >> 4] * sc2;

            acc1 += al * w1l + ah * w1h;
            acc2 += al * w2l + ah * w2h;
        }
    }


    __nv_bfloat16* out = is_shared ?
        (sh_down_out + (unsigned long long)token * N) :
        (C + (unsigned long long)(token * top_k + expert_slot) * N);

    #pragma unroll
    for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1)
        acc1 += __shfl_down_sync(0xFFFFFFFF, acc1, offset);
    if (lane == 0) out[n1] = __float2bfloat16(acc1);

    if (have_n2) {
        #pragma unroll
        for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1)
            acc2 += __shfl_down_sync(0xFFFFFFFF, acc2, offset);
        if (lane == 0) out[n2] = __float2bfloat16(acc2);
    }
}

// 2026-09-25: output[t, j] = sum over e of expert_weights[t * top_k + e] *
// expert_out[t * top_k + e, j] + sigmoid(dot(input[t], gate_weight)) * shared_out[t, j].
// gate_weight must be non-null. Every block recomputes its token's gate dot product.
// Launch: grid (ceil(hidden / 256), num_tokens, 1), block 256. The kernel assumes 256
// threads: the dot loop strides by 256 and sums 8 warp partials.






extern "C" __global__ void moe_weighted_sum_blend_prefill(
    __nv_bfloat16* __restrict__ output,
    const __nv_bfloat16* __restrict__ expert_out,
    const float* __restrict__ expert_weights,
    const __nv_bfloat16* __restrict__ shared_out,
    const __nv_bfloat16* __restrict__ input,
    const __nv_bfloat16* __restrict__ gate_weight,
    unsigned int hidden,
    unsigned int top_k,
    unsigned int K,
    unsigned int num_tokens
) {
    const unsigned int token = blockIdx.y;
    const unsigned int tid = threadIdx.x;


    const __nv_bfloat16* input_t = input + (unsigned long long)token * K;

    __shared__ float s_warp_sums[8];
    __shared__ float sigmoid_val;

    float dot_acc = 0.0f;
    unsigned int K8 = K / 8;
    for (unsigned int k8 = tid; k8 < K8; k8 += 256) {
        uint4 a_data = ((const uint4*)input_t)[k8];
        uint4 w_data = ((const uint4*)gate_weight)[k8];
        const unsigned int a_raw[4] = {a_data.x, a_data.y, a_data.z, a_data.w};
        const unsigned int w_raw[4] = {w_data.x, w_data.y, w_data.z, w_data.w};

        #pragma unroll
        for (int b = 0; b < 4; b++) {
            __nv_bfloat16 a_lo, a_hi, w_lo, w_hi;
            *(unsigned short*)&a_lo = (unsigned short)(a_raw[b] & 0xFFFF);
            *(unsigned short*)&a_hi = (unsigned short)(a_raw[b] >> 16);
            *(unsigned short*)&w_lo = (unsigned short)(w_raw[b] & 0xFFFF);
            *(unsigned short*)&w_hi = (unsigned short)(w_raw[b] >> 16);
            dot_acc += __bfloat162float(a_lo) * __bfloat162float(w_lo);
            dot_acc += __bfloat162float(a_hi) * __bfloat162float(w_hi);
        }
    }


    unsigned int warp_id = tid / WARP_SIZE;
    unsigned int lane = tid % WARP_SIZE;
    #pragma unroll
    for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1) {
        dot_acc += __shfl_down_sync(0xFFFFFFFF, dot_acc, offset);
    }
    if (lane == 0) s_warp_sums[warp_id] = dot_acc;
    __syncthreads();

    if (tid == 0) {
        float gate_scalar = 0.0f;
        #pragma unroll
        for (int w = 0; w < 8; w++) gate_scalar += s_warp_sums[w];
        sigmoid_val = 1.0f / (1.0f + __expf(-gate_scalar));
    }
    __syncthreads();


    unsigned int j = blockIdx.x * blockDim.x + tid;
    if (j >= hidden) return;

    float acc = 0.0f;
    for (unsigned int e = 0; e < top_k; e++) {
        float w = expert_weights[token * top_k + e];
        acc += w * __bfloat162float(expert_out[(unsigned long long)(token * top_k + e) * hidden + j]);
    }
    acc += sigmoid_val * __bfloat162float(shared_out[(unsigned long long)token * hidden + j]);
    output[(unsigned long long)token * hidden + j] = __float2bfloat16(acc);
}
