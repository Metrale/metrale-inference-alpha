// SPDX-License-Identifier: MIT OR Apache-2.0

// 2026-09-25: NVFP4 W4A16 decode GEMVs for M=1: two projections of one BF16 input in one
// launch (`w4a16_gemv_dual`), and a GEMV whose input is silu(gate) * up computed per element
// (`w4a16_gemv_silu_input`), each with a single-warp-per-output `_sw` twin.
//
// Owner: gb10 kernels.
// Invariants:
// - Weights are E2M1 pairs [N, K / 2] with one E4M3 scale per 16 K values [N, K / 16], times
//   the per-tensor scale2. Activations and outputs are BF16.
// - Each output C[n], n < N, is written once, by one thread; nothing else in global memory is
//   written. The dual kernels read K / 16 16-value chunks and the silu kernels K / 8 8-value
//   chunks; values past the last full chunk are not read.




#include <cuda_bf16.h>
#include <cuda_fp8.h>

// 2026-09-25: E4M3 (1 sign, 4 exponent, 3 mantissa bits, bias 7) to float in integer
// arithmetic, compiled only for SCALE and HIP builds; NVIDIA builds convert through
// __nv_fp8_e4m3. Exponent 0 is subnormal (m * 2^-9); the NaN codes 0x7F and 0xFF give +0 and -0.


#if defined(__SCALE__) || defined(__HIP_PLATFORM_AMD__)
__device__ __forceinline__ float scl_fp8(unsigned char b) {
    unsigned int s = (b >> 7) & 1u, e = (b >> 3) & 0xFu, m = b & 0x7u; float v;
    if (e == 0u)               v = (float)m * 0.001953125f;
    else if (e == 15u && m == 7u) v = 0.0f;
    else                       v = __uint_as_float(((e + 120u) << 23) | (m << 20));
    return s ? -v : v;
}
#endif

#define BLOCK_SIZE 256
#define N_PER_BLOCK 4
#define WARP_SIZE 32
#define GROUP_SIZE 16

__device__ __constant__ float E2M1_LUT_FUSED_W4[16] = {
    0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f,
    -0.0f, -0.5f, -1.0f, -1.5f, -2.0f, -3.0f, -4.0f, -6.0f
};
// 2026-09-25: METRALE_WARP_LUT_STAGED selects whether the `_sw` kernels stage the E2M1 table
// per warp (1) or index the __constant__ table directly (0). It is 0 under __SCALE__ and
// __HIP_PLATFORM_AMD__, so those builds compile no __syncwarp(). The strix and strix-hip trees
// compile this file through `[sources] use` in their common/KERNEL.toml. The base kernels
// stage the table block-wide with __syncthreads() on every target.














#if defined(__SCALE__) || defined(__HIP_PLATFORM_AMD__)
#define METRALE_WARP_LUT_STAGED 0
#else
#define METRALE_WARP_LUT_STAGED 1
#endif


// 2026-09-25: Copy the E2M1 table to shared memory before the decode loop. The index is a
// data-dependent weight nibble: constant memory serializes a warp's distinct addresses, while
// shared memory serves the 16 entries from 16 banks. The copy holds the same FP32 values, so
// results do not change.
//
// Warp-scoped, for the `_sw` kernels: they return early when n >= N, which is uniform per warp
// but not per block, so a __syncthreads() here would be a divergent barrier. The `_sw` kernels
// contain no block barrier.


__device__ __forceinline__ void stage_e2m1_lut_fused_warp(float* s_lut, unsigned int lane) {
#if METRALE_WARP_LUT_STAGED
    if (lane < 16u) s_lut[lane] = E2M1_LUT_FUSED_W4[lane];
    __syncwarp();
#else
    (void)s_lut; (void)lane;
#endif
}

// 2026-09-25: One lane's partial over the 16-value K chunks 2 * orig_lane and 2 * orig_lane + 1,
// then the pairs 128, 256, ... chunks on, in two accumulators added at the end. Both dual kernels call it.
__device__ __forceinline__ float w4a16_dual_partial(
    const __nv_bfloat16* __restrict__ A,
    const unsigned char* __restrict__ B_packed,
    const unsigned char* __restrict__ B_scale,
    const float scale2,
    unsigned int n, unsigned int half_K, unsigned int num_groups,
    unsigned int K16, unsigned int orig_lane,
    const float* __restrict__ lut)
{
    float acc0 = 0.0f, acc1 = 0.0f;
    const unsigned int stride2 = 128u;
    for (unsigned int k16 = orig_lane * 2u; k16 < K16 + 1u; k16 += stride2) {
        #pragma unroll
        for (int c = 0; c < 2; c++) {
            const unsigned int kk = k16 + (unsigned int)c;
            if (kk >= K16) break;

            uint4 a_lo4 = ((const uint4*)A)[kk * 2];
            uint4 a_hi4 = ((const uint4*)A)[kk * 2 + 1];
            const unsigned int a_raw[8] = {a_lo4.x, a_lo4.y, a_lo4.z, a_lo4.w,
                                            a_hi4.x, a_hi4.y, a_hi4.z, a_hi4.w};
            unsigned long long packed8 = *(const unsigned long long*)(
                B_packed + (unsigned long long)n * half_K + kk * 8);
            unsigned char scale_byte = B_scale[
                (unsigned long long)n * num_groups + kk];
            __nv_fp8_e4m3 fp8;
            *(unsigned char*)&fp8 = scale_byte;
#if defined(__SCALE__) || defined(__HIP_PLATFORM_AMD__)
            float scale = scl_fp8(scale_byte) * scale2;
#else
            float scale = (float)fp8 * scale2;
#endif
            float part = 0.0f;
            #pragma unroll
            for (int b = 0; b < 8; b++) {
                unsigned char byte_val = (unsigned char)(packed8 >> (b * 8));
                float2 af = __bfloat1622float2(*(const __nv_bfloat162*)&a_raw[b]);
                part = fmaf(af.x, lut[byte_val & 0xF], part);
                part = fmaf(af.y, lut[byte_val >> 4], part);
            }
            if (c == 0) acc0 = fmaf(scale, part, acc0);
            else        acc1 = fmaf(scale, part, acc1);
        }
    }
    return acc0 + acc1;
}

// 2026-09-25: blockIdx.z = 0 computes C1 from (B1_packed, B1_scale, scale2_1) and 1 computes C2
// from the second set; both read the same A. A is [1, K], B*_packed [N, K / 2], B*_scale
// [N, K / 16], C1 and C2 [1, N]. Grid (ceil(N / 4), 1, 2), block 256: 64 threads per output,
// whose two warps are summed through shared memory.

extern "C" __global__ void w4a16_gemv_dual(
    const __nv_bfloat16* __restrict__ A,
    const unsigned char* __restrict__ B1_packed,
    const unsigned char* __restrict__ B1_scale,
    const float scale2_1,
    __nv_bfloat16* __restrict__ C1,
    const unsigned char* __restrict__ B2_packed,
    const unsigned char* __restrict__ B2_scale,
    const float scale2_2,
    __nv_bfloat16* __restrict__ C2,
    unsigned int N,
    unsigned int K
) {
    const unsigned int proj = blockIdx.z;
    const unsigned char* B_packed = proj == 0 ? B1_packed : B2_packed;
    const unsigned char* B_scale = proj == 0 ? B1_scale : B2_scale;
    float scale2 = proj == 0 ? scale2_1 : scale2_2;
    __nv_bfloat16* C = proj == 0 ? C1 : C2;

    const unsigned int threads_per_out = BLOCK_SIZE / N_PER_BLOCK;
    const unsigned int local_out = threadIdx.x / threads_per_out;
    const unsigned int lane = threadIdx.x % threads_per_out;

    const unsigned int n = blockIdx.x * N_PER_BLOCK + local_out;
    const unsigned int half_K = K / 2;
    const unsigned int num_groups = K / GROUP_SIZE;
    const unsigned int K16 = K / 16;

    __shared__ float s_lut[16];
    __shared__ float smem[N_PER_BLOCK * 2];
    // 2026-09-25: No early return: every thread reaches both __syncthreads(), including threads with n >= N.
    if (threadIdx.x < 16) s_lut[threadIdx.x] = E2M1_LUT_FUSED_W4[threadIdx.x];
    __syncthreads();


    float acc = 0.0f;
    if (n < N) {
        acc = w4a16_dual_partial(A, B_packed, B_scale, scale2, n, half_K, num_groups, K16, lane, s_lut);
    }

    const unsigned int warp_lane = threadIdx.x % WARP_SIZE;

    #pragma unroll
    for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1) {
        acc += __shfl_down_sync(0xFFFFFFFF, acc, offset);
    }

    if (warp_lane == 0) {
        unsigned int smem_idx = local_out * 2 + (lane / WARP_SIZE);
        smem[smem_idx] = acc;
    }
    __syncthreads();

    if (lane == 0 && n < N) {
        float result = smem[local_out * 2] + smem[local_out * 2 + 1];
        C[n] = __float2bfloat16(result);
    }
}

// 2026-09-25: C = (silu(gate_out) * up_out) times the down weights, the activation computed per
// element as gate / (1 + exp(-gate)) * up. gate_out and up_out are [1, K], B_packed [N, K / 2],
// B_scale [N, K / 16], C [1, N]. Grid (ceil(N / 4), 1, 1), block 256, 64 threads per output.
// The return for n >= N comes before two __syncthreads(), so N must be a multiple of 4.


extern "C" __global__ void w4a16_gemv_silu_input(
    const __nv_bfloat16* __restrict__ gate_out,
    const __nv_bfloat16* __restrict__ up_out,
    const unsigned char* __restrict__ B_packed,
    const unsigned char* __restrict__ B_scale,
    const float scale2,
    __nv_bfloat16* __restrict__ C,
    unsigned int N,
    unsigned int K
) {
    const unsigned int threads_per_out = BLOCK_SIZE / N_PER_BLOCK;
    const unsigned int local_out = threadIdx.x / threads_per_out;
    const unsigned int lane = threadIdx.x % threads_per_out;

    const unsigned int n = blockIdx.x * N_PER_BLOCK + local_out;
    if (n >= N) return;

    const unsigned int half_K = K / 2;
    const unsigned int num_groups = K / GROUP_SIZE;
    const unsigned int K8 = K / 8;

    __shared__ float s_lut[16];
    __shared__ float smem[N_PER_BLOCK * 2];
    if (threadIdx.x < 16) s_lut[threadIdx.x] = E2M1_LUT_FUSED_W4[threadIdx.x];
    __syncthreads();

    float acc = 0.0f;

    for (unsigned int k8 = lane; k8 < K8; k8 += threads_per_out) {
        const unsigned int base_k = k8 * 8;

        uint4 g_data = ((const uint4*)gate_out)[k8];
        uint4 u_data = ((const uint4*)up_out)[k8];

        unsigned int packed4 = *(const unsigned int*)(
            B_packed + (unsigned long long)n * half_K + k8 * 4);

        unsigned int scale_group = base_k / GROUP_SIZE;
        unsigned char scale_byte = B_scale[
            (unsigned long long)n * num_groups + scale_group];
        __nv_fp8_e4m3 fp8;
        *(unsigned char*)&fp8 = scale_byte;
#if defined(__SCALE__) || defined(__HIP_PLATFORM_AMD__)
        float scale = scl_fp8(scale_byte) * scale2;
#else
        float scale = (float)fp8 * scale2;
#endif

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

    const unsigned int warp_lane = threadIdx.x % WARP_SIZE;

    #pragma unroll
    for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1) {
        acc += __shfl_down_sync(0xFFFFFFFF, acc, offset);
    }

    if (warp_lane == 0) {
        unsigned int smem_idx = local_out * 2 + (lane / WARP_SIZE);
        smem[smem_idx] = acc;
    }
    __syncthreads();

    if (lane == 0) {
        float result = smem[local_out * 2] + smem[local_out * 2 + 1];
        C[n] = __float2bfloat16(result);
    }
}

// 2026-09-25: Single-warp-per-output twins: N_PER_BLOCK_SW outputs per 256-thread block, grid
// (ceil(N / 8), 1, z). Lane l computes what lanes l and l + 32 of the base kernel's 64-thread
// group compute, and the two sums are reduced in the same order, so the outputs equal the base
// kernels' bit for bit (the model-arch example w4a16_gemv_sw_microtest checks this). dense_ffn
// uses them unless METRALE_NO_GEMV_SW=1 or the kernel handle is missing.



#define N_PER_BLOCK_SW 8

extern "C" __global__ void w4a16_gemv_dual_sw(
    const __nv_bfloat16* __restrict__ A,
    const unsigned char* __restrict__ B1_packed,
    const unsigned char* __restrict__ B1_scale,
    const float scale2_1,
    __nv_bfloat16* __restrict__ C1,
    const unsigned char* __restrict__ B2_packed,
    const unsigned char* __restrict__ B2_scale,
    const float scale2_2,
    __nv_bfloat16* __restrict__ C2,
    unsigned int N,
    unsigned int K
) {
    const unsigned int proj = blockIdx.z;
    const unsigned char* B_packed = proj == 0 ? B1_packed : B2_packed;
    const unsigned char* B_scale = proj == 0 ? B1_scale : B2_scale;
    float scale2 = proj == 0 ? scale2_1 : scale2_2;
    __nv_bfloat16* C = proj == 0 ? C1 : C2;

    const unsigned int local_out = threadIdx.x / WARP_SIZE;
    const unsigned int lane = threadIdx.x % WARP_SIZE;
    const unsigned int n = blockIdx.x * N_PER_BLOCK_SW + local_out;
    if (n >= N) return;

    const unsigned int half_K = K / 2;
    const unsigned int num_groups = K / GROUP_SIZE;
    const unsigned int K16 = K / 16;

    // 2026-09-25: One 16-float row per warp (512 B per block at N_PER_BLOCK_SW = 8); no block barrier.
    __shared__ float s_lut[N_PER_BLOCK_SW][16];
    stage_e2m1_lut_fused_warp(s_lut[local_out], lane);
#if METRALE_WARP_LUT_STAGED
    const float* __restrict__ warp_lut = s_lut[local_out];
#else
    const float* __restrict__ warp_lut = E2M1_LUT_FUSED_W4;
#endif

    float acc_a = w4a16_dual_partial(A, B_packed, B_scale, scale2, n, half_K, num_groups, K16, lane, warp_lut);
    float acc_b = w4a16_dual_partial(A, B_packed, B_scale, scale2, n, half_K, num_groups, K16, lane + 32u, warp_lut);

    #pragma unroll
    for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1) {
        acc_a += __shfl_down_sync(0xFFFFFFFF, acc_a, offset);
        acc_b += __shfl_down_sync(0xFFFFFFFF, acc_b, offset);
    }
    if (lane == 0) {
        C[n] = __float2bfloat16(acc_a + acc_b);
    }
}

// 2026-09-25: One lane's partial for the silu `_sw` kernel: 8-value K chunks from start_chunk, stride 64.
__device__ __forceinline__ float w4a16_silu_partial(
    const __nv_bfloat16* __restrict__ gate_out,
    const __nv_bfloat16* __restrict__ up_out,
    const unsigned char* __restrict__ B_packed,
    const unsigned char* __restrict__ B_scale,
    const float scale2,
    unsigned int n, unsigned int half_K, unsigned int num_groups,
    unsigned int K8, unsigned int start_chunk,
    const float* __restrict__ lut)
{
    float acc = 0.0f;
    for (unsigned int k8 = start_chunk; k8 < K8; k8 += 64u) {
        const unsigned int base_k = k8 * 8;
        uint4 g_data = ((const uint4*)gate_out)[k8];
        uint4 u_data = ((const uint4*)up_out)[k8];
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
        const unsigned int g_raw[4] = {g_data.x, g_data.y, g_data.z, g_data.w};
        const unsigned int u_raw[4] = {u_data.x, u_data.y, u_data.z, u_data.w};
        #pragma unroll
        for (int b = 0; b < 4; b++) {
            unsigned char byte_val = (packed4 >> (b * 8)) & 0xFF;
            float w_lo = lut[byte_val & 0xF] * scale;
            float w_hi = lut[byte_val >> 4] * scale;
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
    return acc;
}

extern "C" __global__ void w4a16_gemv_silu_input_sw(
    const __nv_bfloat16* __restrict__ gate_out,
    const __nv_bfloat16* __restrict__ up_out,
    const unsigned char* __restrict__ B_packed,
    const unsigned char* __restrict__ B_scale,
    const float scale2,
    __nv_bfloat16* __restrict__ C,
    unsigned int N,
    unsigned int K
) {
    const unsigned int local_out = threadIdx.x / WARP_SIZE;
    const unsigned int lane = threadIdx.x % WARP_SIZE;
    const unsigned int n = blockIdx.x * N_PER_BLOCK_SW + local_out;
    if (n >= N) return;

    const unsigned int half_K = K / 2;
    const unsigned int num_groups = K / GROUP_SIZE;
    const unsigned int K8 = K / 8;

    // 2026-09-25: One 16-float row per warp (512 B per block at N_PER_BLOCK_SW = 8); no block barrier.
    __shared__ float s_lut[N_PER_BLOCK_SW][16];
    stage_e2m1_lut_fused_warp(s_lut[local_out], lane);
#if METRALE_WARP_LUT_STAGED
    const float* __restrict__ warp_lut = s_lut[local_out];
#else
    const float* __restrict__ warp_lut = E2M1_LUT_FUSED_W4;
#endif

    float acc_a = w4a16_silu_partial(gate_out, up_out, B_packed, B_scale, scale2, n, half_K, num_groups, K8, lane, warp_lut);
    float acc_b = w4a16_silu_partial(gate_out, up_out, B_packed, B_scale, scale2, n, half_K, num_groups, K8, lane + 32u, warp_lut);

    #pragma unroll
    for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1) {
        acc_a += __shfl_down_sync(0xFFFFFFFF, acc_a, offset);
        acc_b += __shfl_down_sync(0xFFFFFFFF, acc_b, offset);
    }
    if (lane == 0) {
        C[n] = __float2bfloat16(acc_a + acc_b);
    }
}
