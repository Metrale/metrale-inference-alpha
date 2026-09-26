// SPDX-License-Identifier: MIT OR Apache-2.0

// 2026-09-25: Kernels `w8a16_gemv_dual` and `w8a16_gemv_silu_input`: M = 1 GEMVs over block-scaled FP8 E4M3 weights
// for a gated FFN.
//
// Weights B [N, K], one E4M3 byte per value, decoded through a 256-entry LUT; block_scale [ceil(N/128), ceil(K/128)]
// FP32 row-major, one scale per 128 x 128 block. Grid (ceil(N / 4), 1, z), Block (256, 1, 1): 4 outputs per CTA,
// 64 threads (2 warps) per output, 16 values of K per iteration; values past the last multiple of 16 are not read.
//
// `w8a16_gemv_dual` (z = 2): blockIdx.z 0 computes C1 from B1, 1 computes C2 from B2, both from the same A [1, K].
// Per output it performs the same FP32 operations in the same order as `w8a16_gemv` (w8a16_gemv.cu).
// `w8a16_gemv_silu_input` (z = 1): the activation is silu(gate_out) * up_out, computed in FP32 and not rounded to
// BF16 before the multiply, so it can differ from a separate BF16 silu-mul followed by `w8a16_gemv`.
//
// Owner: gb10 kernels.
// Invariants: none beyond the types.


















#include <cuda_bf16.h>

#define BLOCK_SIZE 256
#define N_PER_BLOCK 4
#define WARP_SIZE 32
#define FP8_BLOCK 128

// 2026-09-25: FP8 E4M3 byte to f32: the same 256 values as `E4M3_LUT` in w8a16_gemv.cu (E4M3, bias 7,
// range [-448, 448], NaN bytes 0x7F and 0xFF mapped to 0.0 and -0.0).





__device__ __constant__ float E4M3_LUT_FUSED_W8[256] = {

    0.0f, 0.001953125f, 0.00390625f, 0.005859375f,
    0.0078125f, 0.009765625f, 0.01171875f, 0.013671875f,
    0.015625f, 0.017578125f, 0.01953125f, 0.021484375f,
    0.0234375f, 0.025390625f, 0.02734375f, 0.029296875f,
    0.03125f, 0.03515625f, 0.0390625f, 0.04296875f,
    0.046875f, 0.05078125f, 0.0546875f, 0.05859375f,
    0.0625f, 0.0703125f, 0.078125f, 0.0859375f,
    0.09375f, 0.1015625f, 0.109375f, 0.1171875f,
    0.125f, 0.140625f, 0.15625f, 0.171875f,
    0.1875f, 0.203125f, 0.21875f, 0.234375f,
    0.25f, 0.28125f, 0.3125f, 0.34375f,
    0.375f, 0.40625f, 0.4375f, 0.46875f,
    0.5f, 0.5625f, 0.625f, 0.6875f,
    0.75f, 0.8125f, 0.875f, 0.9375f,
    1.0f, 1.125f, 1.25f, 1.375f,
    1.5f, 1.625f, 1.75f, 1.875f,
    2.0f, 2.25f, 2.5f, 2.75f,
    3.0f, 3.25f, 3.5f, 3.75f,
    4.0f, 4.5f, 5.0f, 5.5f,
    6.0f, 6.5f, 7.0f, 7.5f,
    8.0f, 9.0f, 10.0f, 11.0f,
    12.0f, 13.0f, 14.0f, 15.0f,
    16.0f, 18.0f, 20.0f, 22.0f,
    24.0f, 26.0f, 28.0f, 30.0f,
    32.0f, 36.0f, 40.0f, 44.0f,
    48.0f, 52.0f, 56.0f, 60.0f,
    64.0f, 72.0f, 80.0f, 88.0f,
    96.0f, 104.0f, 112.0f, 120.0f,
    128.0f, 144.0f, 160.0f, 176.0f,
    192.0f, 208.0f, 224.0f, 240.0f,
    256.0f, 288.0f, 320.0f, 352.0f,
    384.0f, 416.0f, 448.0f, 0.0f,

    -0.0f, -0.001953125f, -0.00390625f, -0.005859375f,
    -0.0078125f, -0.009765625f, -0.01171875f, -0.013671875f,
    -0.015625f, -0.017578125f, -0.01953125f, -0.021484375f,
    -0.0234375f, -0.025390625f, -0.02734375f, -0.029296875f,
    -0.03125f, -0.03515625f, -0.0390625f, -0.04296875f,
    -0.046875f, -0.05078125f, -0.0546875f, -0.05859375f,
    -0.0625f, -0.0703125f, -0.078125f, -0.0859375f,
    -0.09375f, -0.1015625f, -0.109375f, -0.1171875f,
    -0.125f, -0.140625f, -0.15625f, -0.171875f,
    -0.1875f, -0.203125f, -0.21875f, -0.234375f,
    -0.25f, -0.28125f, -0.3125f, -0.34375f,
    -0.375f, -0.40625f, -0.4375f, -0.46875f,
    -0.5f, -0.5625f, -0.625f, -0.6875f,
    -0.75f, -0.8125f, -0.875f, -0.9375f,
    -1.0f, -1.125f, -1.25f, -1.375f,
    -1.5f, -1.625f, -1.75f, -1.875f,
    -2.0f, -2.25f, -2.5f, -2.75f,
    -3.0f, -3.25f, -3.5f, -3.75f,
    -4.0f, -4.5f, -5.0f, -5.5f,
    -6.0f, -6.5f, -7.0f, -7.5f,
    -8.0f, -9.0f, -10.0f, -11.0f,
    -12.0f, -13.0f, -14.0f, -15.0f,
    -16.0f, -18.0f, -20.0f, -22.0f,
    -24.0f, -26.0f, -28.0f, -30.0f,
    -32.0f, -36.0f, -40.0f, -44.0f,
    -48.0f, -52.0f, -56.0f, -60.0f,
    -64.0f, -72.0f, -80.0f, -88.0f,
    -96.0f, -104.0f, -112.0f, -120.0f,
    -128.0f, -144.0f, -160.0f, -176.0f,
    -192.0f, -208.0f, -224.0f, -240.0f,
    -256.0f, -288.0f, -320.0f, -352.0f,
    -384.0f, -416.0f, -448.0f, -0.0f,
};






extern "C" __global__ void w8a16_gemv_dual(
    const __nv_bfloat16* __restrict__ A,
    const unsigned char* __restrict__ B1,
    const float* __restrict__ B1_scale,
    __nv_bfloat16* __restrict__ C1,
    const unsigned char* __restrict__ B2,
    const float* __restrict__ B2_scale,
    __nv_bfloat16* __restrict__ C2,
    unsigned int N,
    unsigned int K
) {
    const unsigned int proj = blockIdx.z;
    const unsigned char* B = proj == 0 ? B1 : B2;
    const float* block_scale = proj == 0 ? B1_scale : B2_scale;
    __nv_bfloat16* C = proj == 0 ? C1 : C2;

    const unsigned int threads_per_out = BLOCK_SIZE / N_PER_BLOCK;
    const unsigned int local_out = threadIdx.x / threads_per_out;
    const unsigned int lane = threadIdx.x % threads_per_out;

    const unsigned int n = blockIdx.x * N_PER_BLOCK + local_out;
    if (n >= N) return;

    const unsigned int K16 = K / 16;
    const unsigned int k_blocks = (K + FP8_BLOCK - 1) / FP8_BLOCK;
    const unsigned int n_block = n / FP8_BLOCK;




    __shared__ float s_lut[256];
    __shared__ float smem[N_PER_BLOCK * 2];
    s_lut[threadIdx.x] = E4M3_LUT_FUSED_W8[threadIdx.x];
    __syncthreads();

    float acc = 0.0f;


    for (unsigned int k16 = lane; k16 < K16; k16 += threads_per_out) {
        const unsigned int base_k = k16 * 16;

        const unsigned int k_block = base_k / FP8_BLOCK;
        float scale = block_scale[n_block * k_blocks + k_block];


        uint4 b_data = ((const uint4*)(B + (unsigned long long)n * K))[k16];

        uint4 a_data0 = ((const uint4*)A)[k16 * 2];
        uint4 a_data1 = ((const uint4*)A)[k16 * 2 + 1];

        const unsigned int b_raw0[2] = {b_data.x, b_data.y};
        const unsigned int a_raw0[4] = {a_data0.x, a_data0.y, a_data0.z, a_data0.w};

        #pragma unroll
        for (int i = 0; i < 2; i++) {
            unsigned int w32 = b_raw0[i];
            unsigned int a32_lo = a_raw0[i * 2];
            unsigned int a32_hi = a_raw0[i * 2 + 1];

            float w0 = s_lut[(w32      ) & 0xFF] * scale;
            float w1 = s_lut[(w32 >>  8) & 0xFF] * scale;
            float w2 = s_lut[(w32 >> 16) & 0xFF] * scale;
            float w3 = s_lut[(w32 >> 24) & 0xFF] * scale;

            __nv_bfloat16 a0, a1, a2, a3;
            *(unsigned short*)&a0 = (unsigned short)(a32_lo & 0xFFFF);
            *(unsigned short*)&a1 = (unsigned short)(a32_lo >> 16);
            *(unsigned short*)&a2 = (unsigned short)(a32_hi & 0xFFFF);
            *(unsigned short*)&a3 = (unsigned short)(a32_hi >> 16);

            acc += __bfloat162float(a0) * w0;
            acc += __bfloat162float(a1) * w1;
            acc += __bfloat162float(a2) * w2;
            acc += __bfloat162float(a3) * w3;
        }

        const unsigned int b_raw1[2] = {b_data.z, b_data.w};
        const unsigned int a_raw1[4] = {a_data1.x, a_data1.y, a_data1.z, a_data1.w};

        #pragma unroll
        for (int i = 0; i < 2; i++) {
            unsigned int w32 = b_raw1[i];
            unsigned int a32_lo = a_raw1[i * 2];
            unsigned int a32_hi = a_raw1[i * 2 + 1];

            float w0 = s_lut[(w32      ) & 0xFF] * scale;
            float w1 = s_lut[(w32 >>  8) & 0xFF] * scale;
            float w2 = s_lut[(w32 >> 16) & 0xFF] * scale;
            float w3 = s_lut[(w32 >> 24) & 0xFF] * scale;

            __nv_bfloat16 a0, a1, a2, a3;
            *(unsigned short*)&a0 = (unsigned short)(a32_lo & 0xFFFF);
            *(unsigned short*)&a1 = (unsigned short)(a32_lo >> 16);
            *(unsigned short*)&a2 = (unsigned short)(a32_hi & 0xFFFF);
            *(unsigned short*)&a3 = (unsigned short)(a32_hi >> 16);

            acc += __bfloat162float(a0) * w0;
            acc += __bfloat162float(a1) * w1;
            acc += __bfloat162float(a2) * w2;
            acc += __bfloat162float(a3) * w3;
        }
    }

    // 2026-09-25: Shuffle-reduce within each warp, then sum the output's two warps through shared memory.
    #pragma unroll
    for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1) {
        acc += __shfl_down_sync(0xFFFFFFFF, acc, offset);
    }

    unsigned int warp_in_out = lane / WARP_SIZE;
    if (lane % WARP_SIZE == 0) {
        smem[local_out * 2 + warp_in_out] = acc;
    }
    __syncthreads();

    if (lane == 0) {
        float result = smem[local_out * 2] + smem[local_out * 2 + 1];
        C[n] = __float2bfloat16(result);
    }
}






extern "C" __global__ void w8a16_gemv_silu_input(
    const __nv_bfloat16* __restrict__ gate_out,
    const __nv_bfloat16* __restrict__ up_out,
    const unsigned char* __restrict__ B,
    const float* __restrict__ block_scale,
    __nv_bfloat16* __restrict__ C,
    unsigned int N,
    unsigned int K
) {
    const unsigned int threads_per_out = BLOCK_SIZE / N_PER_BLOCK;
    const unsigned int local_out = threadIdx.x / threads_per_out;
    const unsigned int lane = threadIdx.x % threads_per_out;

    const unsigned int n = blockIdx.x * N_PER_BLOCK + local_out;
    if (n >= N) return;

    const unsigned int K16 = K / 16;
    const unsigned int k_blocks = (K + FP8_BLOCK - 1) / FP8_BLOCK;
    const unsigned int n_block = n / FP8_BLOCK;

    __shared__ float s_lut[256];
    __shared__ float smem[N_PER_BLOCK * 2];
    s_lut[threadIdx.x] = E4M3_LUT_FUSED_W8[threadIdx.x];
    __syncthreads();

    float acc = 0.0f;

    for (unsigned int k16 = lane; k16 < K16; k16 += threads_per_out) {
        const unsigned int base_k = k16 * 16;

        const unsigned int k_block = base_k / FP8_BLOCK;
        float scale = block_scale[n_block * k_blocks + k_block];


        uint4 b_data = ((const uint4*)(B + (unsigned long long)n * K))[k16];

        uint4 g_data0 = ((const uint4*)gate_out)[k16 * 2];
        uint4 g_data1 = ((const uint4*)gate_out)[k16 * 2 + 1];
        uint4 u_data0 = ((const uint4*)up_out)[k16 * 2];
        uint4 u_data1 = ((const uint4*)up_out)[k16 * 2 + 1];

        const unsigned int b_raw0[2] = {b_data.x, b_data.y};
        const unsigned int g_raw0[4] = {g_data0.x, g_data0.y, g_data0.z, g_data0.w};
        const unsigned int u_raw0[4] = {u_data0.x, u_data0.y, u_data0.z, u_data0.w};

        #pragma unroll
        for (int i = 0; i < 2; i++) {
            unsigned int w32 = b_raw0[i];
            unsigned int g32_lo = g_raw0[i * 2];
            unsigned int g32_hi = g_raw0[i * 2 + 1];
            unsigned int u32_lo = u_raw0[i * 2];
            unsigned int u32_hi = u_raw0[i * 2 + 1];

            float w0 = s_lut[(w32      ) & 0xFF] * scale;
            float w1 = s_lut[(w32 >>  8) & 0xFF] * scale;
            float w2 = s_lut[(w32 >> 16) & 0xFF] * scale;
            float w3 = s_lut[(w32 >> 24) & 0xFF] * scale;

            __nv_bfloat16 g0, g1, g2, g3, u0, u1, u2, u3;
            *(unsigned short*)&g0 = (unsigned short)(g32_lo & 0xFFFF);
            *(unsigned short*)&g1 = (unsigned short)(g32_lo >> 16);
            *(unsigned short*)&g2 = (unsigned short)(g32_hi & 0xFFFF);
            *(unsigned short*)&g3 = (unsigned short)(g32_hi >> 16);
            *(unsigned short*)&u0 = (unsigned short)(u32_lo & 0xFFFF);
            *(unsigned short*)&u1 = (unsigned short)(u32_lo >> 16);
            *(unsigned short*)&u2 = (unsigned short)(u32_hi & 0xFFFF);
            *(unsigned short*)&u3 = (unsigned short)(u32_hi >> 16);

            float gf0 = __bfloat162float(g0), gf1 = __bfloat162float(g1);
            float gf2 = __bfloat162float(g2), gf3 = __bfloat162float(g3);


            float a0 = (gf0 / (1.0f + __expf(-gf0))) * __bfloat162float(u0);
            float a1 = (gf1 / (1.0f + __expf(-gf1))) * __bfloat162float(u1);
            float a2 = (gf2 / (1.0f + __expf(-gf2))) * __bfloat162float(u2);
            float a3 = (gf3 / (1.0f + __expf(-gf3))) * __bfloat162float(u3);

            acc += a0 * w0;
            acc += a1 * w1;
            acc += a2 * w2;
            acc += a3 * w3;
        }

        const unsigned int b_raw1[2] = {b_data.z, b_data.w};
        const unsigned int g_raw1[4] = {g_data1.x, g_data1.y, g_data1.z, g_data1.w};
        const unsigned int u_raw1[4] = {u_data1.x, u_data1.y, u_data1.z, u_data1.w};

        #pragma unroll
        for (int i = 0; i < 2; i++) {
            unsigned int w32 = b_raw1[i];
            unsigned int g32_lo = g_raw1[i * 2];
            unsigned int g32_hi = g_raw1[i * 2 + 1];
            unsigned int u32_lo = u_raw1[i * 2];
            unsigned int u32_hi = u_raw1[i * 2 + 1];

            float w0 = s_lut[(w32      ) & 0xFF] * scale;
            float w1 = s_lut[(w32 >>  8) & 0xFF] * scale;
            float w2 = s_lut[(w32 >> 16) & 0xFF] * scale;
            float w3 = s_lut[(w32 >> 24) & 0xFF] * scale;

            __nv_bfloat16 g0, g1, g2, g3, u0, u1, u2, u3;
            *(unsigned short*)&g0 = (unsigned short)(g32_lo & 0xFFFF);
            *(unsigned short*)&g1 = (unsigned short)(g32_lo >> 16);
            *(unsigned short*)&g2 = (unsigned short)(g32_hi & 0xFFFF);
            *(unsigned short*)&g3 = (unsigned short)(g32_hi >> 16);
            *(unsigned short*)&u0 = (unsigned short)(u32_lo & 0xFFFF);
            *(unsigned short*)&u1 = (unsigned short)(u32_lo >> 16);
            *(unsigned short*)&u2 = (unsigned short)(u32_hi & 0xFFFF);
            *(unsigned short*)&u3 = (unsigned short)(u32_hi >> 16);

            float gf0 = __bfloat162float(g0), gf1 = __bfloat162float(g1);
            float gf2 = __bfloat162float(g2), gf3 = __bfloat162float(g3);

            float a0 = (gf0 / (1.0f + __expf(-gf0))) * __bfloat162float(u0);
            float a1 = (gf1 / (1.0f + __expf(-gf1))) * __bfloat162float(u1);
            float a2 = (gf2 / (1.0f + __expf(-gf2))) * __bfloat162float(u2);
            float a3 = (gf3 / (1.0f + __expf(-gf3))) * __bfloat162float(u3);

            acc += a0 * w0;
            acc += a1 * w1;
            acc += a2 * w2;
            acc += a3 * w3;
        }
    }

    #pragma unroll
    for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1) {
        acc += __shfl_down_sync(0xFFFFFFFF, acc, offset);
    }

    unsigned int warp_in_out = lane / WARP_SIZE;
    if (lane % WARP_SIZE == 0) {
        smem[local_out * 2 + warp_in_out] = acc;
    }
    __syncthreads();

    if (lane == 0) {
        float result = smem[local_out * 2] + smem[local_out * 2 + 1];
        C[n] = __float2bfloat16(result);
    }
}
