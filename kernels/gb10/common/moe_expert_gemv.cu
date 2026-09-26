// SPDX-License-Identifier: MIT OR Apache-2.0

// 2026-09-25: Decode MoE kernels over the top-k routed experts: moe_expert_gemv, an NVFP4
// (W4A16) GEMV per expert slot, and the weighted sums moe_weighted_sum and
// moe_weighted_sum_blend.
//
// Owner: gb10 kernels.
// Invariants:
// - moe_expert_gemv: grid (ceil(N / 4), top_k), block 128. blockIdx.y is the slot, whose
//   expert is expert_indices[slot]; each output n gets one warp. C is [top_k, N] BF16.
// - Expert e's weights: packed_ptrs[e] is [N, K / 2] bytes of E2M1 pairs (low nibble
//   first) and scale_ptrs[e] is [N, K / 16] E4M3 group scales; w = E2M1 * scale *
//   scale2_vals[e].
// - A slot whose expert has a NULL packed pointer gets a zero C row.
// - input_stride 0 makes every slot read the same activation row; otherwise slot s reads
//   A + s * input_stride.
// - K must be a multiple of 8: each step reads 8 activations and 4 weight bytes, and there
//   is no tail loop.



#include <cuda_bf16.h>
#include <cuda_fp8.h>

// 2026-09-25: Software E4M3 decode for the SCALE and HIP builds, which use it instead of the
// __nv_fp8_e4m3 conversion. It follows the standard E4M3 layout (bias 7, subnormals
// m * 2^-9), except that NaN (e = 15, m = 7) decodes to 0.


#if defined(__SCALE__) || defined(__HIP_PLATFORM_AMD__)
__device__ __forceinline__ float scl_fp8(unsigned char b) {
    unsigned int s = (b >> 7) & 1u, e = (b >> 3) & 0xFu, m = b & 0x7u; float v;
    if (e == 0u)               v = (float)m * 0.001953125f;
    else if (e == 15u && m == 7u) v = 0.0f;
    else                       v = __uint_as_float(((e + 120u) << 23) | (m << 20));
    return s ? -v : v;
}
#endif

#define BLOCK_SIZE 128
#define N_PER_BLOCK 4
#define WARP_SIZE 32
#define GROUP_SIZE 16

__device__ __constant__ float E2M1_LUT_EXP[16] = {
    0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f,
    -0.0f, -0.5f, -1.0f, -1.5f, -2.0f, -3.0f, -4.0f, -6.0f
};








extern "C" __global__ void moe_expert_gemv(
    const __nv_bfloat16* __restrict__ A,
    const unsigned long long* __restrict__ packed_ptrs,
    const unsigned long long* __restrict__ scale_ptrs,
    const float* __restrict__ scale2_vals,
    __nv_bfloat16* __restrict__ C,
    const unsigned int* __restrict__ expert_indices,
    unsigned int N,
    unsigned int K,
    unsigned int top_k,
    unsigned int input_stride
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


    const __nv_bfloat16* input = A + (input_stride > 0 ? (unsigned long long)expert_slot * input_stride : 0);


    const unsigned int threads_per_out = BLOCK_SIZE / N_PER_BLOCK;
    const unsigned int local_out = threadIdx.x / threads_per_out;
    const unsigned int lane = threadIdx.x % threads_per_out;

    const unsigned int n = blockIdx.x * N_PER_BLOCK + local_out;
    if (n >= N) return;

    const unsigned int half_K = K / 2;
    const unsigned int num_groups = K / GROUP_SIZE;
    const unsigned int K8 = K / 8;


    __shared__ float s_lut[16];
    if (threadIdx.x < 16) s_lut[threadIdx.x] = E2M1_LUT_EXP[threadIdx.x];
    __syncthreads();

    float acc = 0.0f;


    for (unsigned int k8 = lane; k8 < K8; k8 += threads_per_out) {
        const unsigned int base_k = k8 * 8;


        uint4 a_data = ((const uint4*)input)[k8];
        const unsigned int a_raw[4] = {a_data.x, a_data.y, a_data.z, a_data.w};


        unsigned int packed4 = *(const unsigned int*)(B_packed + (unsigned long long)n * half_K + k8 * 4);


        unsigned int scale_group = base_k / GROUP_SIZE;
        unsigned char scale_byte = B_scale[(unsigned long long)n * num_groups + scale_group];
        __nv_fp8_e4m3 fp8;
        *(unsigned char*)&fp8 = scale_byte;
#if defined(__SCALE__) || defined(__HIP_PLATFORM_AMD__)
        float scale = scl_fp8(scale_byte) * scale2;
#else
        float scale = (float)fp8 * scale2;
#endif


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

// 2026-09-25: output[j] = sum_e expert_weights[e] * expert_out[e][j], one thread per j.




extern "C" __global__ void moe_weighted_sum(
    __nv_bfloat16* __restrict__ output,
    const __nv_bfloat16* __restrict__ expert_out,
    const float* __restrict__ expert_weights,
    unsigned int hidden,
    unsigned int top_k
) {
    unsigned int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j >= hidden) return;

    float acc = 0.0f;
    for (unsigned int e = 0; e < top_k; e++) {
        float val = __bfloat162float(expert_out[(unsigned long long)e * hidden + j]);
        acc += expert_weights[e] * val;
    }
    output[j] = __float2bfloat16(acc);
}

// 2026-09-25: moe_weighted_sum plus a gated shared expert:
//   output[j] = sum_e weights[e] * expert_out[e][j] + sigmoid(dot(input, gate_weight)) * shared_out[j]
// with the sigmoid taken as 1 when gate_weight is NULL. Every block computes the dot itself.
// The block must be 256 threads (eight warp sums into s_warp_sums[8]) and K a multiple of 8.









extern "C" __global__ void moe_weighted_sum_blend(
    __nv_bfloat16* __restrict__ output,
    const __nv_bfloat16* __restrict__ expert_out,
    const float* __restrict__ expert_weights,
    const __nv_bfloat16* __restrict__ shared_out,
    const __nv_bfloat16* __restrict__ input,
    const __nv_bfloat16* __restrict__ gate_weight,
    unsigned int hidden,
    unsigned int top_k,
    unsigned int K
) {
    const unsigned int tid = threadIdx.x;
    const unsigned int warp_id = tid / WARP_SIZE;
    const unsigned int lane = tid % WARP_SIZE;






    __shared__ float s_warp_sums[8];
    __shared__ float sigmoid_val;

    if (gate_weight == 0) {

        if (tid == 0) sigmoid_val = 1.0f;
        __syncthreads();
    } else {

    float dot_acc = 0.0f;
    unsigned int K8 = K / 8;
    for (unsigned int k8 = tid; k8 < K8; k8 += 256) {
        uint4 a_data = ((const uint4*)input)[k8];
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


    #pragma unroll
    for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1) {
        dot_acc += __shfl_down_sync(0xFFFFFFFF, dot_acc, offset);
    }
    if (lane == 0) {
        s_warp_sums[warp_id] = dot_acc;
    }
    __syncthreads();


    if (tid == 0) {
        float gate_scalar = 0.0f;
        #pragma unroll
        for (int w = 0; w < 8; w++) {
            gate_scalar += s_warp_sums[w];
        }
        sigmoid_val = 1.0f / (1.0f + __expf(-gate_scalar));
    }
    __syncthreads();

    }


    unsigned int j = blockIdx.x * blockDim.x + tid;
    if (j >= hidden) return;

    float acc = 0.0f;
    for (unsigned int e = 0; e < top_k; e++) {
        acc += expert_weights[e] * __bfloat162float(expert_out[(unsigned long long)e * hidden + j]);
    }
    acc += sigmoid_val * __bfloat162float(shared_out[j]);
    output[j] = __float2bfloat16(acc);
}
