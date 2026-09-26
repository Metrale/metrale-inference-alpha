// SPDX-License-Identifier: MIT OR Apache-2.0

// 2026-09-25: NVFP4 (W4A16) expert GEMVs for decode: moe_expert_gemv_gate_up* compute the
// gate and up projections in one launch, and moe_expert_gemv_silu_down* fold
// silu(g) * u = g / (1 + exp(-g)) * u into the down projection's activation.
// Owner: gb10 kernels.
// Invariants:
// - Block 128. blockIdx.y is the slot, whose expert is expert_indices[slot]; in the gate_up
//   kernels blockIdx.z selects gate (0) or up (1). Outputs per block: 4 for the base
//   kernels (one warp each), 8 for _2x (two per warp) and 16 for _wide (8 threads each),
//   so grid.x is ceil(N / 4), ceil(N / 8) or ceil(N / 16).
// - Expert e's weights: packed [N, K / 2] bytes of E2M1 pairs (low nibble first), E4M3
//   group scales [N, K / 16], and w = E2M1 * scale * scale2[e]. Outputs are [top_k, N].
// - A slot whose expert has a NULL packed pointer gets zero outputs.
// - The gate_up kernels read one activation row A[K] for every slot; the silu_down kernels
//   read row slot of gate_out and up_out ([top_k, K]).
// - K must be a multiple of 8: each step reads 8 activations, and there is no tail loop.
#include <cuda_bf16.h>
#include <cuda_fp8.h>

#define BLOCK_SIZE 128
#define N_PER_BLOCK 4
#define WARP_SIZE 32
#define GROUP_SIZE 16

__device__ __constant__ float E2M1_LUT_FUSED[16] = {
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






extern "C" __global__ void moe_expert_gemv_gate_up(
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
    unsigned int N,
    unsigned int K,
    unsigned int top_k
) {
    const unsigned int expert_slot = blockIdx.y;
    if (expert_slot >= top_k) return;

    const unsigned int proj = blockIdx.z;

    const unsigned int expert_id = expert_indices[expert_slot];


    const unsigned char* B_packed;
    const unsigned char* B_scale;
    float scale2;
    __nv_bfloat16* C;

    if (proj == 0) {
        B_packed = (const unsigned char*)gate_packed_ptrs[expert_id];
        B_scale = (const unsigned char*)gate_scale_ptrs[expert_id];
        scale2 = gate_scale2_vals[expert_id];
        C = gate_out;
    } else {
        B_packed = (const unsigned char*)up_packed_ptrs[expert_id];
        B_scale = (const unsigned char*)up_scale_ptrs[expert_id];
        scale2 = up_scale2_vals[expert_id];
        C = up_out;
    }


    if (B_packed == 0) {
        const unsigned int n_base = blockIdx.x * N_PER_BLOCK;
        for (unsigned int i = threadIdx.x; i < N_PER_BLOCK && n_base + i < N; i += BLOCK_SIZE) {
            C[expert_slot * N + n_base + i] = __float2bfloat16(0.0f);
        }
        return;
    }

    const unsigned int threads_per_out = BLOCK_SIZE / N_PER_BLOCK;
    const unsigned int local_out = threadIdx.x / threads_per_out;
    const unsigned int lane = threadIdx.x % threads_per_out;

    const unsigned int n = blockIdx.x * N_PER_BLOCK + local_out;
    if (n >= N) return;

    const unsigned int half_K = K / 2;
    const unsigned int num_groups = K / GROUP_SIZE;
    const unsigned int K8 = K / 8;

    __shared__ float s_lut[16];
    if (threadIdx.x < 16) s_lut[threadIdx.x] = E2M1_LUT_FUSED[threadIdx.x];
    __syncthreads();

    float acc = 0.0f;

    for (unsigned int k8 = lane; k8 < K8; k8 += threads_per_out) {
        const unsigned int base_k = k8 * 8;

        uint4 a_data = ((const uint4*)A)[k8];
        const unsigned int a_raw[4] = {a_data.x, a_data.y, a_data.z, a_data.w};

        unsigned int packed4 = *(const unsigned int*)(B_packed + (unsigned long long)n * half_K + k8 * 4);

        unsigned int scale_group = base_k / GROUP_SIZE;
        unsigned char scale_byte = B_scale[(unsigned long long)n * num_groups + scale_group];
        float scale = metrale_dec_e4m3(scale_byte) * scale2;

        #pragma unroll
        for (int b = 0; b < 4; b++) {
            unsigned char byte_val = (packed4 >> (b * 8)) & 0xFF;
            float w_lo = s_lut[byte_val & 0xF] * scale;
            float w_hi = s_lut[byte_val >> 4] * scale;

            __nv_bfloat16 a_lo, a_hi;
            *(unsigned short*)&a_lo = (unsigned short)(a_raw[b] & 0xFFFF);
            *(unsigned short*)&a_hi = (unsigned short)(a_raw[b] >> 16);
            acc += __bfloat162float(a_lo) * w_lo;
            acc += __bfloat162float(a_hi) * w_hi;
        }
    }

    #pragma unroll
    for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1) {
        acc += __shfl_down_sync(0xFFFFFFFF, acc, offset);
    }

    if (lane == 0) {
        C[(unsigned long long)expert_slot * N + n] = __float2bfloat16(acc);
    }
}









extern "C" __global__ void moe_expert_gemv_gate_up_2x(
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
    unsigned int N,
    unsigned int K,
    unsigned int top_k
) {
    const unsigned int expert_slot = blockIdx.y;
    if (expert_slot >= top_k) return;

    const unsigned int proj = blockIdx.z;
    const unsigned int expert_id = expert_indices[expert_slot];

    const unsigned char* B_packed;
    const unsigned char* B_scale;
    float s2;
    __nv_bfloat16* C;

    if (proj == 0) {
        B_packed = (const unsigned char*)gate_packed_ptrs[expert_id];
        B_scale = (const unsigned char*)gate_scale_ptrs[expert_id];
        s2 = gate_scale2_vals[expert_id];
        C = gate_out;
    } else {
        B_packed = (const unsigned char*)up_packed_ptrs[expert_id];
        B_scale = (const unsigned char*)up_scale_ptrs[expert_id];
        s2 = up_scale2_vals[expert_id];
        C = up_out;
    }


    if (B_packed == 0) {
        const unsigned int n_base = blockIdx.x * (N_PER_BLOCK * 2);
        for (unsigned int i = threadIdx.x; i < N_PER_BLOCK * 2 && n_base + i < N; i += BLOCK_SIZE) {
            C[expert_slot * N + n_base + i] = __float2bfloat16(0.0f);
        }
        return;
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
    if (threadIdx.x < 16) s_lut[threadIdx.x] = E2M1_LUT_FUSED[threadIdx.x];
    __syncthreads();

    float acc1 = 0.0f;
    float acc2 = 0.0f;

    for (unsigned int k8 = lane; k8 < K8; k8 += threads_per_out) {
        const unsigned int base_k = k8 * 8;


        uint4 a_data = ((const uint4*)A)[k8];
        const unsigned int a_raw[4] = {a_data.x, a_data.y, a_data.z, a_data.w};


        unsigned int packed4_1 = *(const unsigned int*)(B_packed + (unsigned long long)n1 * half_K + k8 * 4);
        unsigned int scale_group = base_k / GROUP_SIZE;
        unsigned char sb1 = B_scale[(unsigned long long)n1 * num_groups + scale_group];
        float scale_1 = metrale_dec_e4m3(sb1) * s2;


        unsigned int packed4_2 = have_n2 ?
            *(const unsigned int*)(B_packed + (unsigned long long)n2 * half_K + k8 * 4) : 0;
        unsigned char sb2 = have_n2 ?
            B_scale[(unsigned long long)n2 * num_groups + scale_group] : 0;
        float scale_2 = have_n2 ? metrale_dec_e4m3(sb2) * s2 : 0.0f;

        #pragma unroll
        for (int b = 0; b < 4; b++) {

            unsigned char bv1 = (packed4_1 >> (b * 8)) & 0xFF;
            float w1_lo = s_lut[bv1 & 0xF] * scale_1;
            float w1_hi = s_lut[bv1 >> 4] * scale_1;


            unsigned char bv2 = (packed4_2 >> (b * 8)) & 0xFF;
            float w2_lo = s_lut[bv2 & 0xF] * scale_2;
            float w2_hi = s_lut[bv2 >> 4] * scale_2;


            __nv_bfloat16 a_lo, a_hi;
            *(unsigned short*)&a_lo = (unsigned short)(a_raw[b] & 0xFFFF);
            *(unsigned short*)&a_hi = (unsigned short)(a_raw[b] >> 16);
            float af_lo = __bfloat162float(a_lo);
            float af_hi = __bfloat162float(a_hi);

            acc1 += af_lo * w1_lo + af_hi * w1_hi;
            acc2 += af_lo * w2_lo + af_hi * w2_hi;
        }
    }


    #pragma unroll
    for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1) {
        acc1 += __shfl_down_sync(0xFFFFFFFF, acc1, offset);
    }
    if (lane == 0) {
        C[(unsigned long long)expert_slot * N + n1] = __float2bfloat16(acc1);
    }


    if (have_n2) {
        #pragma unroll
        for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1) {
            acc2 += __shfl_down_sync(0xFFFFFFFF, acc2, offset);
        }
        if (lane == 0) {
            C[(unsigned long long)expert_slot * N + n2] = __float2bfloat16(acc2);
        }
    }
}








extern "C" __global__ void moe_expert_gemv_silu_down(
    const __nv_bfloat16* __restrict__ gate_out,
    const __nv_bfloat16* __restrict__ up_out,
    const unsigned long long* __restrict__ packed_ptrs,
    const unsigned long long* __restrict__ scale_ptrs,
    const float* __restrict__ scale2_vals,
    __nv_bfloat16* __restrict__ C,
    const unsigned int* __restrict__ expert_indices,
    unsigned int N,
    unsigned int K,
    unsigned int top_k
) {
    const unsigned int expert_slot = blockIdx.y;
    if (expert_slot >= top_k) return;

    const unsigned int expert_id = expert_indices[expert_slot];

    const unsigned char* B_packed = (const unsigned char*)packed_ptrs[expert_id];
    const unsigned char* B_scale = (const unsigned char*)scale_ptrs[expert_id];
    const float scale2 = scale2_vals[expert_id];


    if (B_packed == 0) {
        const unsigned int n_base = blockIdx.x * N_PER_BLOCK;
        for (unsigned int i = threadIdx.x; i < N_PER_BLOCK && n_base + i < N; i += BLOCK_SIZE) {
            C[expert_slot * N + n_base + i] = __float2bfloat16(0.0f);
        }
        return;
    }


    const __nv_bfloat16* g_ptr = gate_out + (unsigned long long)expert_slot * K;
    const __nv_bfloat16* u_ptr = up_out + (unsigned long long)expert_slot * K;

    const unsigned int threads_per_out = BLOCK_SIZE / N_PER_BLOCK;
    const unsigned int local_out = threadIdx.x / threads_per_out;
    const unsigned int lane = threadIdx.x % threads_per_out;

    const unsigned int n = blockIdx.x * N_PER_BLOCK + local_out;
    if (n >= N) return;

    const unsigned int half_K = K / 2;
    const unsigned int num_groups = K / GROUP_SIZE;
    const unsigned int K8 = K / 8;

    __shared__ float s_lut[16];
    if (threadIdx.x < 16) s_lut[threadIdx.x] = E2M1_LUT_FUSED[threadIdx.x];
    __syncthreads();

    float acc = 0.0f;

    for (unsigned int k8 = lane; k8 < K8; k8 += threads_per_out) {
        const unsigned int base_k = k8 * 8;


        uint4 g_data = ((const uint4*)g_ptr)[k8];
        uint4 u_data = ((const uint4*)u_ptr)[k8];

        unsigned int packed4 = *(const unsigned int*)(B_packed + (unsigned long long)n * half_K + k8 * 4);

        unsigned int scale_group = base_k / GROUP_SIZE;
        unsigned char scale_byte = B_scale[(unsigned long long)n * num_groups + scale_group];
        float scale = metrale_dec_e4m3(scale_byte) * scale2;


        const unsigned int g_raw[4] = {g_data.x, g_data.y, g_data.z, g_data.w};
        const unsigned int u_raw[4] = {u_data.x, u_data.y, u_data.z, u_data.w};

        #pragma unroll
        for (int b = 0; b < 4; b++) {
            unsigned char byte_val = (packed4 >> (b * 8)) & 0xFF;
            float w_lo = s_lut[byte_val & 0xF] * scale;
            float w_hi = s_lut[byte_val >> 4] * scale;


            __nv_bfloat16 g_lo, g_hi;
            *(unsigned short*)&g_lo = (unsigned short)(g_raw[b] & 0xFFFF);
            *(unsigned short*)&g_hi = (unsigned short)(g_raw[b] >> 16);
            float gf_lo = __bfloat162float(g_lo);
            float gf_hi = __bfloat162float(g_hi);


            __nv_bfloat16 u_lo, u_hi;
            *(unsigned short*)&u_lo = (unsigned short)(u_raw[b] & 0xFFFF);
            *(unsigned short*)&u_hi = (unsigned short)(u_raw[b] >> 16);


            float a_lo = (gf_lo / (1.0f + __expf(-gf_lo))) * __bfloat162float(u_lo);
            float a_hi = (gf_hi / (1.0f + __expf(-gf_hi))) * __bfloat162float(u_hi);

            acc += a_lo * w_lo;
            acc += a_hi * w_hi;
        }
    }

    #pragma unroll
    for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1) {
        acc += __shfl_down_sync(0xFFFFFFFF, acc, offset);
    }

    if (lane == 0) {
        C[(unsigned long long)expert_slot * N + n] = __float2bfloat16(acc);
    }
}










extern "C" __global__ void moe_expert_gemv_silu_down_2x(
    const __nv_bfloat16* __restrict__ gate_out,
    const __nv_bfloat16* __restrict__ up_out,
    const unsigned long long* __restrict__ packed_ptrs,
    const unsigned long long* __restrict__ scale_ptrs,
    const float* __restrict__ scale2_vals,
    __nv_bfloat16* __restrict__ C,
    const unsigned int* __restrict__ expert_indices,
    unsigned int N,
    unsigned int K,
    unsigned int top_k
) {
    const unsigned int expert_slot = blockIdx.y;
    if (expert_slot >= top_k) return;

    const unsigned int expert_id = expert_indices[expert_slot];

    const unsigned char* B_packed = (const unsigned char*)packed_ptrs[expert_id];
    const unsigned char* B_scale = (const unsigned char*)scale_ptrs[expert_id];
    const float scale2 = scale2_vals[expert_id];


    if (B_packed == 0) {
        const unsigned int n_base = blockIdx.x * (N_PER_BLOCK * 2);
        for (unsigned int i = threadIdx.x; i < N_PER_BLOCK * 2 && n_base + i < N; i += BLOCK_SIZE) {
            C[expert_slot * N + n_base + i] = __float2bfloat16(0.0f);
        }
        return;
    }

    const __nv_bfloat16* g_ptr = gate_out + (unsigned long long)expert_slot * K;
    const __nv_bfloat16* u_ptr = up_out + (unsigned long long)expert_slot * K;


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
    if (threadIdx.x < 16) s_lut[threadIdx.x] = E2M1_LUT_FUSED[threadIdx.x];
    __syncthreads();

    float acc1 = 0.0f;
    float acc2 = 0.0f;

    for (unsigned int k8 = lane; k8 < K8; k8 += threads_per_out) {
        const unsigned int base_k = k8 * 8;


        uint4 g_data = ((const uint4*)g_ptr)[k8];
        uint4 u_data = ((const uint4*)u_ptr)[k8];


        unsigned int packed4_1 = *(const unsigned int*)(B_packed + (unsigned long long)n1 * half_K + k8 * 4);
        unsigned int scale_group = base_k / GROUP_SIZE;
        unsigned char sb1 = B_scale[(unsigned long long)n1 * num_groups + scale_group];
        float scale_1 = metrale_dec_e4m3(sb1) * scale2;


        unsigned int packed4_2 = have_n2 ?
            *(const unsigned int*)(B_packed + (unsigned long long)n2 * half_K + k8 * 4) : 0;
        unsigned char sb2 = have_n2 ?
            B_scale[(unsigned long long)n2 * num_groups + scale_group] : 0;
        float scale_2 = have_n2 ? metrale_dec_e4m3(sb2) * scale2 : 0.0f;

        const unsigned int g_raw[4] = {g_data.x, g_data.y, g_data.z, g_data.w};
        const unsigned int u_raw[4] = {u_data.x, u_data.y, u_data.z, u_data.w};

        #pragma unroll
        for (int b = 0; b < 4; b++) {

            unsigned char bv1 = (packed4_1 >> (b * 8)) & 0xFF;
            float w1_lo = s_lut[bv1 & 0xF] * scale_1;
            float w1_hi = s_lut[bv1 >> 4] * scale_1;

            unsigned char bv2 = (packed4_2 >> (b * 8)) & 0xFF;
            float w2_lo = s_lut[bv2 & 0xF] * scale_2;
            float w2_hi = s_lut[bv2 >> 4] * scale_2;


            __nv_bfloat16 g_lo, g_hi;
            *(unsigned short*)&g_lo = (unsigned short)(g_raw[b] & 0xFFFF);
            *(unsigned short*)&g_hi = (unsigned short)(g_raw[b] >> 16);
            float gf_lo = __bfloat162float(g_lo);
            float gf_hi = __bfloat162float(g_hi);

            __nv_bfloat16 u_lo, u_hi;
            *(unsigned short*)&u_lo = (unsigned short)(u_raw[b] & 0xFFFF);
            *(unsigned short*)&u_hi = (unsigned short)(u_raw[b] >> 16);

            float a_lo = (gf_lo / (1.0f + __expf(-gf_lo))) * __bfloat162float(u_lo);
            float a_hi = (gf_hi / (1.0f + __expf(-gf_hi))) * __bfloat162float(u_hi);

            acc1 += a_lo * w1_lo + a_hi * w1_hi;
            acc2 += a_lo * w2_lo + a_hi * w2_hi;
        }
    }


    #pragma unroll
    for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1) {
        acc1 += __shfl_down_sync(0xFFFFFFFF, acc1, offset);
    }
    if (lane == 0) {
        C[(unsigned long long)expert_slot * N + n1] = __float2bfloat16(acc1);
    }


    if (have_n2) {
        #pragma unroll
        for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1) {
            acc2 += __shfl_down_sync(0xFFFFFFFF, acc2, offset);
        }
        if (lane == 0) {
            C[(unsigned long long)expert_slot * N + n2] = __float2bfloat16(acc2);
        }
    }
}








extern "C" __global__ void moe_expert_gemv_silu_down_wide(
    const __nv_bfloat16* __restrict__ gate_out,
    const __nv_bfloat16* __restrict__ up_out,
    const unsigned long long* __restrict__ packed_ptrs,
    const unsigned long long* __restrict__ scale_ptrs,
    const float* __restrict__ scale2_vals,
    __nv_bfloat16* __restrict__ C,
    const unsigned int* __restrict__ expert_indices,
    unsigned int N,
    unsigned int K,
    unsigned int top_k
) {
    const unsigned int expert_slot = blockIdx.y;
    if (expert_slot >= top_k) return;

    const unsigned int expert_id = expert_indices[expert_slot];

    const unsigned char* B_packed = (const unsigned char*)packed_ptrs[expert_id];
    const unsigned char* B_scale = (const unsigned char*)scale_ptrs[expert_id];
    const float scale2 = scale2_vals[expert_id];


    if (B_packed == 0) {
        const unsigned int N_WIDE = 16;
        const unsigned int n_base = blockIdx.x * N_WIDE;
        for (unsigned int i = threadIdx.x; i < N_WIDE && n_base + i < N; i += BLOCK_SIZE) {
            C[expert_slot * N + n_base + i] = __float2bfloat16(0.0f);
        }
        return;
    }

    const __nv_bfloat16* g_ptr = gate_out + (unsigned long long)expert_slot * K;
    const __nv_bfloat16* u_ptr = up_out + (unsigned long long)expert_slot * K;


    const unsigned int N_WIDE = 16;
    const unsigned int tpo = BLOCK_SIZE / N_WIDE;
    const unsigned int local_out = threadIdx.x / tpo;
    const unsigned int lane = threadIdx.x % tpo;

    const unsigned int n = blockIdx.x * N_WIDE + local_out;
    if (n >= N) return;

    const unsigned int half_K = K / 2;
    const unsigned int num_groups = K / GROUP_SIZE;
    const unsigned int K8 = K / 8;

    __shared__ float s_lut[16];
    if (threadIdx.x < 16) s_lut[threadIdx.x] = E2M1_LUT_FUSED[threadIdx.x];
    __syncthreads();

    float acc = 0.0f;

    for (unsigned int k8 = lane; k8 < K8; k8 += tpo) {
        const unsigned int base_k = k8 * 8;

        uint4 g_data = ((const uint4*)g_ptr)[k8];
        uint4 u_data = ((const uint4*)u_ptr)[k8];

        unsigned int packed4 = *(const unsigned int*)(B_packed + (unsigned long long)n * half_K + k8 * 4);

        unsigned int scale_group = base_k / GROUP_SIZE;
        unsigned char scale_byte = B_scale[(unsigned long long)n * num_groups + scale_group];
        float scale = metrale_dec_e4m3(scale_byte) * scale2;

        const unsigned int g_raw[4] = {g_data.x, g_data.y, g_data.z, g_data.w};
        const unsigned int u_raw[4] = {u_data.x, u_data.y, u_data.z, u_data.w};

        #pragma unroll
        for (int b = 0; b < 4; b++) {
            unsigned char byte_val = (packed4 >> (b * 8)) & 0xFF;
            float w_lo = s_lut[byte_val & 0xF] * scale;
            float w_hi = s_lut[byte_val >> 4] * scale;

            __nv_bfloat16 g_lo, g_hi;
            *(unsigned short*)&g_lo = (unsigned short)(g_raw[b] & 0xFFFF);
            *(unsigned short*)&g_hi = (unsigned short)(g_raw[b] >> 16);
            float gf_lo = __bfloat162float(g_lo);
            float gf_hi = __bfloat162float(g_hi);

            __nv_bfloat16 u_lo, u_hi;
            *(unsigned short*)&u_lo = (unsigned short)(u_raw[b] & 0xFFFF);
            *(unsigned short*)&u_hi = (unsigned short)(u_raw[b] >> 16);

            float a_lo = (gf_lo / (1.0f + __expf(-gf_lo))) * __bfloat162float(u_lo);
            float a_hi = (gf_hi / (1.0f + __expf(-gf_hi))) * __bfloat162float(u_hi);

            acc += a_lo * w_lo;
            acc += a_hi * w_hi;
        }
    }


    #pragma unroll
    for (int offset = tpo / 2; offset > 0; offset >>= 1) {
        acc += __shfl_down_sync(0xFFFFFFFF, acc, offset, tpo);
    }

    if (lane == 0) {
        C[(unsigned long long)expert_slot * N + n] = __float2bfloat16(acc);
    }
}
