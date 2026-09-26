// SPDX-License-Identifier: MIT OR Apache-2.0
// forked-from: kernels/gb10/common/w4a16_gemm.cu (2026-09-24; 145 of 327 lines differ, see kernels/FORKS.md)

// 2026-09-25: W4A16 GEMM for HIP on gfx1151: NVFP4 weights dequantised to BF16
// in shared memory, then a BF16 WMMA 16x16x16 GEMM, plus a standalone dequant.
//
// Owner: strix-hip kernels.
// Invariants: none beyond the types.
//
// C[M,N] = A[M,K] (BF16) x dequant(B), where
//   B_packed holds two E2M1 weights per byte, the even K index in the low
//            nibble: [N, K/2] for w4a16_gemm and w4a16_dequant, [K/2, ldb] for
//            w4a16_gemm_t;
//   B_scale  holds one E4M3 scale per 16 K values, in the same orientation;
//   scale2   is one FP32 scale per tensor;
//   value = E2M1_LUT[nibble] * scl_fp8(scale byte) * scale2.
// Both strix-hip model targets replace this file ([shadow] in their
// KERNEL.toml), so neither of them compiles it.
//
// WMMA store mapping: lane l, element e (0..7) -> row 2*e + (l>>4) of the
// warp's 16 rows, column l&15 of a 16-column tile.


#include <cuda_bf16.h>

typedef __bf16 v16bf __attribute__((ext_vector_type(16)));
typedef float  v8f   __attribute__((ext_vector_type(8)));

__device__ __forceinline__ float scl_fp8(unsigned char b) {
    unsigned int s = (b >> 7) & 1u, e = (b >> 3) & 0xFu, m = b & 0x7u; float v;
    if (e == 0u)               v = (float)m * 0.001953125f;
    else if (e == 15u && m == 7u) v = 0.0f;
    else                       v = __uint_as_float(((e + 120u) << 23) | (m << 20));
    return s ? -v : v;
}
__device__ __forceinline__ unsigned char scl_enc_fp8(float v) {
    if (v != v) return 0x7F;
    unsigned int bb = __float_as_uint(v); unsigned int sign = (bb >> 31) & 1u;
    int e = (int)((bb >> 23) & 0xFF) - 127; unsigned int man = bb & 0x7FFFFFu;
    int ee = e + 7; unsigned int em;
    if (ee < 1) { ee = 0; em = 0; if (e >= -10) { float a = v < 0 ? -v : v; em = (unsigned int)(a / 0.001953125f + 0.5f); if (em > 7u) em = 7u; } }
    else if (ee > 15) { ee = 15; em = 6; }
    else { em = (man + (1u << 19)) >> 20; if (em > 7u) { em = 0; ee++; if (ee > 15) { ee = 15; em = 6; } } }
    return (unsigned char)((sign << 7) | ((unsigned)ee << 3) | em);
}

#include <cuda_fp8.h>

#define M_TILE 64
#define N_TILE 64
#define K_STEP 16
#define PAD 2
#define GROUP_SIZE 16

// 2026-09-25: E2M1 decode table, indexed by the 4-bit code.
__device__ __constant__ float E2M1_LUT[16] = {
    0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f,
    -0.0f, -0.5f, -1.0f, -1.5f, -2.0f, -3.0f, -4.0f, -6.0f
};

// 2026-09-25: WMMA over smem_A[M_TILE][K_STEP+PAD] and smem_B[K_STEP][N_TILE+PAD].
// acc[4] holds 4 16-column tiles (64 N); warp_m_offset selects the warp's
// 16 rows.
__device__ __forceinline__ void w4a16_wmma_compute(
    __nv_bfloat16 smem_A[][K_STEP + PAD],
    __nv_bfloat16 smem_B[][N_TILE + PAD],
    v8f acc[4],
    unsigned int warp_m_offset, unsigned int lane
) {
    v16bf a;
    #pragma unroll
    for (int i = 0; i < 16; i++) a[i] = (__bf16)(float)smem_A[warp_m_offset + (lane & 15)][i];
    #pragma unroll
    for (int nb = 0; nb < 4; nb++) {
        v16bf b;
        #pragma unroll
        for (int k = 0; k < 16; k++) b[k] = (__bf16)(float)smem_B[k][nb * 16 + (lane & 15)];
        acc[nb] = __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32(a, b, acc[nb]);
    }
}

// 2026-09-25: Store epilogue shared by w4a16_gemm and w4a16_gemm_t.
__device__ __forceinline__ void w4a16_wmma_store(
    __nv_bfloat16* __restrict__ C, v8f acc[4],
    unsigned int cta_m, unsigned int cta_n, unsigned int warp_m_offset,
    unsigned int lane, unsigned int M, unsigned int N
) {
    #pragma unroll
    for (int nb = 0; nb < 4; nb++) {
        #pragma unroll
        for (int e = 0; e < 8; e++) {
            unsigned int row = cta_m + warp_m_offset + 2 * e + (lane >> 4);
            unsigned int col = cta_n + nb * 16 + (lane & 15);
            if (row < M && col < N) C[row * N + col] = __float2bfloat16(acc[nb][e]);
        }
    }
}

/// 2026-09-25: B_packed [N, K/2], B_scale [N, K/GROUP_SIZE].
extern "C" __global__ void w4a16_gemm(
    const __nv_bfloat16* __restrict__ A,
    const unsigned char* __restrict__ B_packed,
    const unsigned char* __restrict__ B_scale,
    const float scale2,
    __nv_bfloat16* __restrict__ C,
    unsigned int M,
    unsigned int N,
    unsigned int K
) {
    const unsigned int cta_m = blockIdx.y * M_TILE;
    const unsigned int cta_n = blockIdx.x * N_TILE;
    const unsigned int warp_id = threadIdx.x / 32;
    const unsigned int lane_id = threadIdx.x % 32;
    const unsigned int warp_m_offset = warp_id * 16;

    __shared__ __nv_bfloat16 smem_A[M_TILE][K_STEP + PAD];
    __shared__ __nv_bfloat16 smem_B[K_STEP][N_TILE + PAD];

    v8f acc[4];
    #pragma unroll
    for (int i = 0; i < 4; i++) acc[i] = v8f{0, 0, 0, 0, 0, 0, 0, 0};

    const unsigned int half_K = K / 2;
    const unsigned int num_groups = K / GROUP_SIZE;

    for (unsigned int k_base = 0; k_base < K; k_base += K_STEP) {

        {
            const unsigned int elems_per_thread = (M_TILE * K_STEP) / 128;
            #pragma unroll
            for (unsigned int i = 0; i < elems_per_thread; i++) {
                unsigned int idx = threadIdx.x * elems_per_thread + i;
                unsigned int row = idx / K_STEP;
                unsigned int col = idx % K_STEP;
                unsigned int gr = cta_m + row;
                unsigned int gc = k_base + col;
                smem_A[row][col] = (gr < M && gc < K) ? A[gr * K + gc] : __float2bfloat16(0.0f);
            }
        }


        {
            #pragma unroll
            for (unsigned int i = 0; i < 8; i++) {
                unsigned int idx = threadIdx.x * 8 + i;
                unsigned int k = idx / N_TILE;
                unsigned int n = idx % N_TILE;
                unsigned int gk = k_base + k;
                unsigned int gn = cta_n + n;

                if (gk < K && gn < N) {
                    unsigned int k_pair = gk / 2;
                    unsigned char packed_byte = B_packed[(unsigned long long)gn * half_K + k_pair];
                    unsigned int nibble = (gk & 1) ? (packed_byte >> 4) : (packed_byte & 0xF);

                    unsigned int scale_group = gk / GROUP_SIZE;
                    unsigned char scale_byte = B_scale[(unsigned long long)gn * num_groups + scale_group];
                    float dequant_val = E2M1_LUT[nibble] * scl_fp8(scale_byte) * scale2;
                    smem_B[k][n] = __float2bfloat16(dequant_val);
                } else {
                    smem_B[k][n] = __float2bfloat16(0.0f);
                }
            }
        }

        __syncthreads();
        w4a16_wmma_compute(smem_A, smem_B, acc, warp_m_offset, lane_id);
        __syncthreads();
    }

    w4a16_wmma_store(C, acc, cta_m, cta_n, warp_m_offset, lane_id, M, N);
}

/// 2026-09-25: Transposed B: B_packed [K/2, ldb], B_scale [K/GROUP_SIZE, ldb].
/// 8 consecutive threads read the 64 columns of one K row of the tile.
// 2026-09-25: `ldb` is the row stride of the transposed B and may exceed N.
// The loads use it; the `gn < N` guard and the stores stay bounded by N. The
// caller must pass ldb >= N.








extern "C" __global__ void w4a16_gemm_t(
    const __nv_bfloat16* __restrict__ A,
    const unsigned char* __restrict__ B_packed,
    const unsigned char* __restrict__ B_scale,
    const float scale2,
    __nv_bfloat16* __restrict__ C,
    unsigned int M,
    unsigned int N,
    unsigned int K,
    unsigned int ldb
) {
    const unsigned int LDB = ldb;
    const unsigned int cta_m = blockIdx.y * M_TILE;
    const unsigned int cta_n = blockIdx.x * N_TILE;
    const unsigned int warp_id = threadIdx.x / 32;
    const unsigned int lane_id = threadIdx.x % 32;
    const unsigned int warp_m_offset = warp_id * 16;

    __shared__ __nv_bfloat16 smem_A[M_TILE][K_STEP + PAD];
    __shared__ __nv_bfloat16 smem_B[K_STEP][N_TILE + PAD];

    v8f acc[4];
    #pragma unroll
    for (int i = 0; i < 4; i++) acc[i] = v8f{0, 0, 0, 0, 0, 0, 0, 0};

    for (unsigned int k_base = 0; k_base < K; k_base += K_STEP) {

        {
            const unsigned int elems_per_thread = (M_TILE * K_STEP) / 128;
            #pragma unroll
            for (unsigned int i = 0; i < elems_per_thread; i++) {
                unsigned int idx = threadIdx.x * elems_per_thread + i;
                unsigned int row = idx / K_STEP;
                unsigned int col = idx % K_STEP;
                unsigned int gr = cta_m + row;
                unsigned int gc = k_base + col;
                smem_A[row][col] = (gr < M && gc < K) ? A[gr * K + gc] : __float2bfloat16(0.0f);
            }
        }


        {
            #pragma unroll
            for (unsigned int i = 0; i < 8; i++) {
                unsigned int idx = threadIdx.x * 8 + i;
                unsigned int k = idx / N_TILE;
                unsigned int n = idx % N_TILE;
                unsigned int gk = k_base + k;
                unsigned int gn = cta_n + n;

                if (gk < K && gn < N) {
                    unsigned int k_pair = gk / 2;
                    unsigned char packed_byte = B_packed[(unsigned long long)k_pair * LDB + gn];
                    unsigned int nibble = (gk & 1) ? (packed_byte >> 4) : (packed_byte & 0xF);

                    unsigned int scale_group = gk / GROUP_SIZE;
                    unsigned char scale_byte = B_scale[(unsigned long long)scale_group * LDB + gn];
                    float dequant_val = E2M1_LUT[nibble] * scl_fp8(scale_byte) * scale2;
                    smem_B[k][n] = __float2bfloat16(dequant_val);
                } else {
                    smem_B[k][n] = __float2bfloat16(0.0f);
                }
            }
        }

        __syncthreads();
        w4a16_wmma_compute(smem_A, smem_B, acc, warp_m_offset, lane_id);
        __syncthreads();
    }

    w4a16_wmma_store(C, acc, cta_m, cta_n, warp_m_offset, lane_id, M, N);
}

/// 2026-09-25: Standalone dequant of B_packed [N, K/2] to BF16 [N, K], no WMMA.
/// Each thread converts one packed byte into 2 consecutive K values.

extern "C" __global__ void w4a16_dequant(
    const unsigned char* __restrict__ B_packed,
    const unsigned char* __restrict__ B_scale,
    const float scale2,
    __nv_bfloat16* __restrict__ B_bf16,
    unsigned int K,
    unsigned int N
) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int total_bytes = N * (K / 2);
    if (idx >= total_bytes) return;

    unsigned int n = idx / (K / 2);
    unsigned int k_pair = idx % (K / 2);
    unsigned int k0 = k_pair * 2;
    unsigned int k1 = k0 + 1;

    unsigned char packed = B_packed[idx];
    unsigned int nib0 = packed & 0xF;
    unsigned int nib1 = packed >> 4;

    unsigned int num_groups = K / GROUP_SIZE;
    unsigned int scale_group = k0 / GROUP_SIZE;
    // 2026-09-25: k0 is even and GROUP_SIZE is 16, so k0 and k1 share a scale group.
    unsigned char s_byte = B_scale[(unsigned long long)n * num_groups + scale_group];
    float s = scl_fp8(s_byte) * scale2;

    float v0 = E2M1_LUT[nib0] * s;
    float v1 = E2M1_LUT[nib1] * s;

    B_bf16[n * K + k0] = __float2bfloat16(v0);
    if (k1 < K) B_bf16[n * K + k1] = __float2bfloat16(v1);
}
