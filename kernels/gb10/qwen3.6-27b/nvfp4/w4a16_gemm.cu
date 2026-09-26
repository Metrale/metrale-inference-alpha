// SPDX-License-Identifier: MIT OR Apache-2.0
// forked-from: kernels/gb10/common/w4a16_gemm.cu (2026-09-24; 7317 of 7349 lines differ, see kernels/FORKS.md)

// 2026-09-25: NVFP4-weight GEMMs for the qwen3.6-27b tree: W4A16 tile GEMMs with E4M3 or BF16 MMAs, FP8
// GEMMs over pre-converted E4M3 weights, int8 GEMMs over requantized weights, and their conversion kernels.
//
// Owner: gb10 kernels (qwen3.6-27b, and the targets that list this file in `[sources] use`).
// Invariants: an NVFP4 weight is E2M1 codes, two per byte with the low nibble first, one E4M3 scale per
// 16 values and a per-tensor FP32 scale2; a weight value is E2M1 code x E4M3 scale x scale2.


#include <cuda_bf16.h>
#include <cuda_fp8.h>

// 2026-09-25: Software E4M3 (1-4-3, bias 7) decode and encode, for the SCALE and HIP builds. The encoder maps
// NaN to 0x7F and |v| >= 496 to +-448, but |v| in [464, 496) rounds to mantissa 7 at the top exponent, the
// NaN code. The decoder maps the NaN codes to 0.





#if defined(__SCALE__) || defined(__HIP_PLATFORM_AMD__)
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
#endif

// 2026-09-25: Two floats to packed E4M3: the high byte is e4m3(a_hi), the low byte e4m3(b_lo).
// SCALE and HIP builds use scl_enc_fp8 above; other builds use cvt.rn.satfinite.e4m3x2.f32, which saturates.






__device__ __forceinline__ unsigned short metrale_cvt_e4m3x2_f32(float a_hi, float b_lo) {
#if defined(__SCALE__)
    unsigned a8 = (unsigned)scl_enc_fp8(a_hi);
    unsigned b8 = (unsigned)scl_enc_fp8(b_lo);
    return (unsigned short)((a8 << 8) | (b8 & 0xFFu));
#elif defined(__HIP_PLATFORM_AMD__)


    unsigned a8 = (unsigned)scl_enc_fp8(a_hi);
    unsigned b8 = (unsigned)scl_enc_fp8(b_lo);
    return (unsigned short)((a8 << 8) | (b8 & 0xFFu));
#else
    unsigned short d;
    asm volatile("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(d) : "f"(a_hi), "f"(b_lo));
    return d;
#endif
}

// 2026-09-25: metrale_mma_e4m3 is one m16n8k32 E4M3 MMA with FP32 accumulation. Under __SCALE__ it shuffles the E4M3
// fragments into BF16 and issues two m16n8k16 BF16 MMAs, for K 0..15 and 16..31.


















#if defined(__SCALE__)
__device__ __forceinline__ float metrale_e4m3_to_f32(unsigned char b) {
    return scl_fp8(b);
}
__device__ __forceinline__ unsigned metrale_bf2(float lo, float hi) {
    unsigned short l = __bfloat16_as_ushort(__float2bfloat16(lo));
    unsigned short h = __bfloat16_as_ushort(__float2bfloat16(hi));
    return ((unsigned)h << 16) | l;
}
#endif
__device__ __forceinline__ void metrale_mma_e4m3(float* acc,
    unsigned a0, unsigned a1, unsigned a2, unsigned a3,
    unsigned b0, unsigned b1) {
#if defined(__SCALE__)
    unsigned lane = threadIdx.x & 31u, tig = lane & 3u, base = lane & ~3u;
    #pragma unroll
    for (int half = 0; half < 2; half++) {
        unsigned A_g = half ? a2 : a0, A_g8 = half ? a3 : a1, B_g = half ? b1 : b0;
        #define METRALE_GA(reg, j) metrale_e4m3_to_f32((unsigned char)( \
            __shfl_sync(0xffffffffu, (reg), base + ((unsigned)(j) >> 2)) \
            >> (8 * ((j) & 3))))
        int j0 = 2 * (int)tig, j1 = 8 + 2 * (int)tig;
        unsigned A0 = metrale_bf2(METRALE_GA(A_g, j0),  METRALE_GA(A_g, j0 + 1));
        unsigned A1 = metrale_bf2(METRALE_GA(A_g8, j0), METRALE_GA(A_g8, j0 + 1));
        unsigned A2 = metrale_bf2(METRALE_GA(A_g, j1),  METRALE_GA(A_g, j1 + 1));
        unsigned A3 = metrale_bf2(METRALE_GA(A_g8, j1), METRALE_GA(A_g8, j1 + 1));
        unsigned B0 = metrale_bf2(METRALE_GA(B_g, j0),  METRALE_GA(B_g, j0 + 1));
        unsigned B1 = metrale_bf2(METRALE_GA(B_g, j1),  METRALE_GA(B_g, j1 + 1));
        #undef METRALE_GA
        asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
            "{%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%10,%11,%12,%13};"
            : "=f"(acc[0]), "=f"(acc[1]), "=f"(acc[2]), "=f"(acc[3])
            : "r"(A0), "r"(A1), "r"(A2), "r"(A3), "r"(B0), "r"(B1),
              "f"(acc[0]), "f"(acc[1]), "f"(acc[2]), "f"(acc[3]));
    }
#else
    asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 "
        "{%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%10,%11,%12,%13};"
        : "=f"(acc[0]), "=f"(acc[1]), "=f"(acc[2]), "=f"(acc[3])
        : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1),
          "f"(acc[0]), "f"(acc[1]), "f"(acc[2]), "f"(acc[3]));
#endif
}

#define M_TILE 64
#define N_TILE_SM 64
#define N_TILE_LG 128
#define K_STEP 16
#define K_STEP_T 32
#define PAD 2
#define PAD_T 8        // 2026-09-25: (32 + 8) BF16 = 80-byte rows keep cp.async destinations 16-byte aligned.
#define BP_PAD 16      // 2026-09-25: keeps smem_Bp/smem_Bs rows (N tile + 16 bytes) 16-byte aligned for cp.async.
#define B_PAD 2
#define GROUP_SIZE 16

__device__ __constant__ float E2M1_LUT[16] = {
    0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f,
    -0.0f, -0.5f, -1.0f, -1.5f, -2.0f, -3.0f, -4.0f, -6.0f
};

// 2026-09-25: C[M, N] = A[M, K] x W[N, K]^T, W as B_packed [N, K/2] and B_scale [N, K/16]; 128 threads, 64 x 64 tile.
extern "C" __global__ void w4a16_gemm(
    const __nv_bfloat16* __restrict__ A,
    const unsigned char* __restrict__ B_packed,
    const unsigned char* __restrict__ B_scale,
    const float scale2,
    __nv_bfloat16* __restrict__ C,
    unsigned int M, unsigned int N, unsigned int K
) {
    const unsigned int cta_m = blockIdx.y * M_TILE;
    const unsigned int cta_n = blockIdx.x * N_TILE_SM;
    const unsigned int warp_id = threadIdx.x / 32;
    const unsigned int lane_id = threadIdx.x % 32;
    const unsigned int warp_m_offset = warp_id * 16;
    const unsigned int group_id = lane_id >> 2;
    const unsigned int tid = lane_id & 3;

    __shared__ __nv_bfloat16 smem_A[M_TILE][K_STEP + PAD];
    __shared__ __nv_bfloat16 smem_B[K_STEP][N_TILE_SM + PAD];

    float acc[8][4];
    #pragma unroll
    for (int i = 0; i < 8; i++) {
        acc[i][0] = 0.0f; acc[i][1] = 0.0f;
        acc[i][2] = 0.0f; acc[i][3] = 0.0f;
    }

    const unsigned int half_K = K / 2;
    const unsigned int num_groups = K / GROUP_SIZE;
    const unsigned int a_stride = K_STEP + PAD;
    const unsigned int b_stride = N_TILE_SM + PAD;

    for (unsigned int k_base = 0; k_base < K; k_base += K_STEP) {
        {
            const unsigned int ept = (M_TILE * K_STEP) / 128;
            #pragma unroll
            for (unsigned int i = 0; i < ept; i++) {
                unsigned int idx = threadIdx.x * ept + i;
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
                unsigned int k = idx / N_TILE_SM;
                unsigned int n = idx % N_TILE_SM;
                unsigned int gk = k_base + k;
                unsigned int gn = cta_n + n;
                if (gk < K && gn < N) {
                    unsigned int k_pair = gk / 2;
                    unsigned char packed_byte = B_packed[(unsigned long long)gn * half_K + k_pair];
                    unsigned int nibble = (gk & 1) ? (packed_byte >> 4) : (packed_byte & 0xF);
                    unsigned int sg = gk / GROUP_SIZE;
                    unsigned char sb = B_scale[(unsigned long long)gn * num_groups + sg];
                    __nv_fp8_e4m3 fp8; *(unsigned char*)&fp8 = sb;
#if defined(__SCALE__)
                    smem_B[k][n] = __float2bfloat16(E2M1_LUT[nibble] * scl_fp8(sb) * scale2);
#else
                    smem_B[k][n] = __float2bfloat16(E2M1_LUT[nibble] * (float)fp8 * scale2);
#endif
                } else {
                    smem_B[k][n] = __float2bfloat16(0.0f);
                }
            }
        }
        __syncthreads();

        const unsigned short* sA = (const unsigned short*)smem_A;
        const unsigned short* sB = (const unsigned short*)smem_B;
        unsigned int fr0 = warp_m_offset + group_id;
        unsigned int fr1 = fr0 + 8;
        unsigned int fc0 = tid * 2, fc1 = fc0 + 8;
        unsigned int a0 = *(const unsigned int*)&sA[fr0 * a_stride + fc0];
        unsigned int a1 = *(const unsigned int*)&sA[fr1 * a_stride + fc0];
        unsigned int a2 = *(const unsigned int*)&sA[fr0 * a_stride + fc1];
        unsigned int a3 = *(const unsigned int*)&sA[fr1 * a_stride + fc1];
        #pragma unroll
        for (int nt = 0; nt < 8; nt++) {
            unsigned int nc = nt * 8 + group_id;
            unsigned int k0 = tid * 2, k1 = k0 + 8;
            unsigned int b0 = ((unsigned int)sB[(k0+1)*b_stride+nc]<<16) | (unsigned int)sB[k0*b_stride+nc];
            unsigned int b1 = ((unsigned int)sB[(k1+1)*b_stride+nc]<<16) | (unsigned int)sB[k1*b_stride+nc];
            asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%10,%11,%12,%13};"
                :"=f"(acc[nt][0]),"=f"(acc[nt][1]),"=f"(acc[nt][2]),"=f"(acc[nt][3])
                :"r"(a0),"r"(a1),"r"(a2),"r"(a3),"r"(b0),"r"(b1),
                 "f"(acc[nt][0]),"f"(acc[nt][1]),"f"(acc[nt][2]),"f"(acc[nt][3]));
        }
        __syncthreads();
    }

    #pragma unroll
    for (int nt = 0; nt < 8; nt++) {
        unsigned int c0 = cta_n + nt*8 + tid*2;
        unsigned int c1 = c0 + 1;
        unsigned int r0 = cta_m + warp_m_offset + group_id;
        unsigned int r1 = r0 + 8;
        if (r0 < M && c0 < N) C[r0*N+c0] = __float2bfloat16(acc[nt][0]);
        if (r0 < M && c1 < N) C[r0*N+c1] = __float2bfloat16(acc[nt][1]);
        if (r1 < M && c0 < N) C[r1*N+c0] = __float2bfloat16(acc[nt][2]);
        if (r1 < M && c1 < N) C[r1*N+c1] = __float2bfloat16(acc[nt][3]);
    }
}


















__device__ __forceinline__ void cp_async_pred_16(void* dst_smem, const void* src_gmem, bool pred) {
    unsigned int dst = __cvta_generic_to_shared(dst_smem);
    unsigned int src_bytes = pred ? 16 : 0;
    asm volatile("cp.async.ca.shared.global [%0], [%1], 16, %2;"
                 :: "r"(dst), "l"(src_gmem), "r"(src_bytes));
}

__device__ __forceinline__ void cp_async_commit() {
    asm volatile("cp.async.commit_group;");
}

__device__ __forceinline__ void cp_async_wait_group1() {
    // 2026-09-25: Waits until at most one cp.async group is outstanding, so the latest tile's loads stay
    // in flight; cp_async_wait_all drains every group.

    asm volatile("cp.async.wait_group 1;\n" ::);
}

__device__ __forceinline__ void cp_async_wait_all() {
    asm volatile("cp.async.wait_group 0;");
}

// 2026-09-25: cp_async_wait_group1 for any N; N is an immediate operand, hence the template parameter.


template<int N>
__device__ __forceinline__ void cp_async_wait_group() {
    asm volatile("cp.async.wait_group %0;" :: "n"(N));
}

__device__ __forceinline__ unsigned int pack_bf16_pair(float lo, float hi) {
    unsigned int result;
    asm("prmt.b32 %0, %1, %2, 0x7632;" : "=r"(result)
        : "r"(__float_as_uint(lo)), "r"(__float_as_uint(hi)));
    return result;
}














// 2026-09-25: Four BF16 values in shared memory to four E4M3 bytes (metrale_cvt_e4m3x2_f32), the first in the low byte.
__device__ __forceinline__ unsigned int bf16x4_to_e4m3x4(const unsigned short* src) {
    unsigned int p0 = *(const unsigned int*)src;
    unsigned int p1 = *(const unsigned int*)(src + 2);
    unsigned short bf0 = (unsigned short)(p0 & 0xFFFFu);
    unsigned short bf1 = (unsigned short)(p0 >> 16);
    unsigned short bf2 = (unsigned short)(p1 & 0xFFFFu);
    unsigned short bf3 = (unsigned short)(p1 >> 16);
    float f0, f1, f2, f3;
    asm volatile("cvt.f32.bf16 %0, %1;" : "=f"(f0) : "h"(bf0));
    asm volatile("cvt.f32.bf16 %0, %1;" : "=f"(f1) : "h"(bf1));
    asm volatile("cvt.f32.bf16 %0, %1;" : "=f"(f2) : "h"(bf2));
    asm volatile("cvt.f32.bf16 %0, %1;" : "=f"(f3) : "h"(bf3));
    unsigned short h0, h1;
    h0 = metrale_cvt_e4m3x2_f32(f1, f0);
    h1 = metrale_cvt_e4m3x2_f32(f3, f2);
    return ((unsigned int)h1 << 16) | (unsigned int)h0;
}

// 2026-09-25: C[M, N] = A[M, K] x W^T with W stored transposed: B_packed [K/2, ldb] (row k/2 holds K rows
// k and k + 1, low nibble first) and B_scale [K/16, ldb]. 128 threads, 64 x 128 tile, K step 32, A and B
// double-buffered with cp.async. W is dequantized to E4M3 and A converted to E4M3 (saturating, unscaled)
// for m16n8k32 E4M3 MMAs; under __SCALE__ W is dequantized to BF16 and A stays BF16.
//
// ldb is B's row stride in elements and may exceed N. B moves in 16-byte cp.async chunks, which need
// 16-byte aligned sources, so ldb must be a multiple of 16: w4a16_gemm_n128 passes ldb = N, and the lm_head
// passes its padded vocab stride. Loads are bounded by ldb, because a bound of N would drop the real columns
// of a final partial 16-wide chunk; stores are bounded by N.







extern "C" __global__ void w4a16_gemm_t(
    const __nv_bfloat16* __restrict__ A,
    const unsigned char* __restrict__ B_packed,
    const unsigned char* __restrict__ B_scale,
    const float scale2,
    __nv_bfloat16* __restrict__ C,
    unsigned int M, unsigned int N, unsigned int K,
    unsigned int ldb
) {
    const unsigned int LDB = ldb;
    const unsigned int cta_n = blockIdx.x * N_TILE_LG;
    const unsigned int cta_m = blockIdx.y * M_TILE;
    const unsigned int warp_id = threadIdx.x / 32;
    const unsigned int lane_id = threadIdx.x % 32;
    const unsigned int warp_m_offset = warp_id * 16;
    const unsigned int group_id = lane_id >> 2;
    const unsigned int tid = lane_id & 3;

    __shared__ __nv_bfloat16 smem_A[2][M_TILE][K_STEP_T + PAD_T];
    __shared__ unsigned char smem_Bp[2][K_STEP_T / 2][N_TILE_LG + BP_PAD];
    __shared__ unsigned char smem_Bs[2][K_STEP_T / GROUP_SIZE][N_TILE_LG + BP_PAD];
#if defined(__SCALE__)
    __shared__ __nv_bfloat16 smem_B_bf16[N_TILE_LG][K_STEP_T];
#else
    __shared__ unsigned char smem_B_fp8[N_TILE_LG][K_STEP_T];
#endif
    __shared__ float smem_LUT[16];

    if (threadIdx.x < 16) smem_LUT[threadIdx.x] = E2M1_LUT[threadIdx.x];

    float acc[16][4];
    #pragma unroll
    for (int i = 0; i < 16; i++) {
        acc[i][0] = 0.0f; acc[i][1] = 0.0f;
        acc[i][2] = 0.0f; acc[i][3] = 0.0f;
    }

    const unsigned int a_stride = K_STEP_T + PAD_T;

    #define ISSUE_LOADS(buf, kb) do { \
        { \
            unsigned int a_row_base = threadIdx.x >> 2; \
            unsigned int a_col = (threadIdx.x & 3) << 3; \
            unsigned int gc = (kb) + a_col; \
            _Pragma("unroll") \
            for (int rnd = 0; rnd < 2; rnd++) { \
                unsigned int row = rnd * 32 + a_row_base; \
                unsigned int gr = cta_m + row; \
                cp_async_pred_16(&smem_A[(buf)][row][a_col], \
                    &A[gr * K + gc], (gr < M) && (gc + 7 < K)); \
            } \
        } \
        { \
            unsigned int kp = threadIdx.x >> 3; \
            unsigned int ns = (threadIdx.x & 7) << 4; \
            unsigned int gke = (kb) + (kp << 1); \
            unsigned int gns = cta_n + ns; \
            cp_async_pred_16(&smem_Bp[(buf)][kp][ns], \
                &B_packed[(unsigned long long)(gke >> 1) * LDB + gns], \
                (gke + 1 <= K) && (gns + 15 < LDB)); \
            if (kp < K_STEP_T / GROUP_SIZE) { \
                unsigned int sg = (kb) / GROUP_SIZE + kp; \
                cp_async_pred_16(&smem_Bs[(buf)][kp][ns], \
                    &B_scale[(unsigned long long)sg * LDB + gns], \
                    (gns + 15 < LDB)); \
            } \
        } \
    } while(0)

#if defined(__SCALE__)
    // 2026-09-25: Under __SCALE__, W is dequantized to BF16 and multiplied in two m16n8k16 BF16 MMAs per K step.


    #define DEQUANT_T(buf) do { \
        unsigned int my_n = threadIdx.x; \
        unsigned char sb0 = smem_Bs[(buf)][0][my_n]; \
        unsigned char sb1 = smem_Bs[(buf)][1][my_n]; \
        __nv_fp8_e4m3 f0, f1; \
        *(unsigned char*)&f0 = sb0; *(unsigned char*)&f1 = sb1; \
        float sv0 = scl_fp8(*(const unsigned char*)&f0) * scale2, sv1 = scl_fp8(*(const unsigned char*)&f1) * scale2; \
        _Pragma("unroll") \
        for (int kp = 0; kp < 8; kp++) { \
            unsigned char packed = smem_Bp[(buf)][kp][my_n]; \
            smem_B_bf16[my_n][kp * 2]     = __float2bfloat16(smem_LUT[packed & 0xF] * sv0); \
            smem_B_bf16[my_n][kp * 2 + 1] = __float2bfloat16(smem_LUT[packed >> 4] * sv0); \
        } \
        _Pragma("unroll") \
        for (int kp = 8; kp < 16; kp++) { \
            unsigned char packed = smem_Bp[(buf)][kp][my_n]; \
            smem_B_bf16[my_n][kp * 2]     = __float2bfloat16(smem_LUT[packed & 0xF] * sv1); \
            smem_B_bf16[my_n][kp * 2 + 1] = __float2bfloat16(smem_LUT[packed >> 4] * sv1); \
        } \
    } while(0)


    #define COMPUTE_MMA(a_buf) do { \
        const __nv_bfloat16* sA = (const __nv_bfloat16*)smem_A[(a_buf)]; \
        unsigned int fr0 = warp_m_offset + group_id, fr1 = fr0 + 8; \
        _Pragma("unroll") \
        for (int h = 0; h < 2; h++) { \
            unsigned int fc0 = h * 16 + tid * 2, fc1 = fc0 + 8; \
            unsigned int a0 = *(const unsigned int*)&sA[fr0 * a_stride + fc0]; \
            unsigned int a1 = *(const unsigned int*)&sA[fr1 * a_stride + fc0]; \
            unsigned int a2 = *(const unsigned int*)&sA[fr0 * a_stride + fc1]; \
            unsigned int a3 = *(const unsigned int*)&sA[fr1 * a_stride + fc1]; \
            _Pragma("unroll") \
            for (int nt = 0; nt < 16; nt++) { \
                unsigned int nc = nt * 8 + group_id; \
                const __nv_bfloat16* sb = &smem_B_bf16[nc][0]; \
                unsigned int b0 = *(const unsigned int*)&sb[fc0]; \
                unsigned int b1 = *(const unsigned int*)&sb[fc1]; \
                asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 " \
                    "{%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%10,%11,%12,%13};" \
                    : "=f"(acc[nt][0]), "=f"(acc[nt][1]), "=f"(acc[nt][2]), "=f"(acc[nt][3]) \
                    : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1), \
                      "f"(acc[nt][0]), "f"(acc[nt][1]), "f"(acc[nt][2]), "f"(acc[nt][3])); \
            } \
        } \
    } while(0)
#else

    #define DEQUANT_T(buf) do { \
        unsigned int my_n = threadIdx.x; \
        unsigned char sb0 = smem_Bs[(buf)][0][my_n]; \
        unsigned char sb1 = smem_Bs[(buf)][1][my_n]; \
        __nv_fp8_e4m3 f0, f1; \
        *(unsigned char*)&f0 = sb0; *(unsigned char*)&f1 = sb1; \
        float sv0 = (float)f0 * scale2, sv1 = (float)f1 * scale2; \
        _Pragma("unroll") \
        for (int kp = 0; kp < 8; kp++) { \
            unsigned char packed = smem_Bp[(buf)][kp][my_n]; \
            float lo = smem_LUT[packed & 0xF] * sv0; \
            float hi = smem_LUT[packed >> 4] * sv0; \
            unsigned short fp8_pair; \
            asm volatile("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" \
                         : "=h"(fp8_pair) : "f"(hi), "f"(lo)); \
            *(unsigned short*)&smem_B_fp8[my_n][kp * 2] = fp8_pair; \
        } \
        _Pragma("unroll") \
        for (int kp = 8; kp < 16; kp++) { \
            unsigned char packed = smem_Bp[(buf)][kp][my_n]; \
            float lo = smem_LUT[packed & 0xF] * sv1; \
            float hi = smem_LUT[packed >> 4] * sv1; \
            unsigned short fp8_pair; \
            asm volatile("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" \
                         : "=h"(fp8_pair) : "f"(hi), "f"(lo)); \
            *(unsigned short*)&smem_B_fp8[my_n][kp * 2] = fp8_pair; \
        } \
    } while(0)


    #define COMPUTE_MMA(a_buf) do { \
        const unsigned short* sA = (const unsigned short*)smem_A[(a_buf)]; \
        unsigned int fr0 = warp_m_offset + group_id, fr1 = fr0 + 8; \
        unsigned int a0 = bf16x4_to_e4m3x4(&sA[fr0 * a_stride + tid * 4]); \
        unsigned int a1 = bf16x4_to_e4m3x4(&sA[fr1 * a_stride + tid * 4]); \
        unsigned int a2 = bf16x4_to_e4m3x4(&sA[fr0 * a_stride + 16 + tid * 4]); \
        unsigned int a3 = bf16x4_to_e4m3x4(&sA[fr1 * a_stride + 16 + tid * 4]); \
        _Pragma("unroll") \
        for (int nt = 0; nt < 16; nt++) { \
            unsigned int nc = nt * 8 + group_id; \
            unsigned int b0 = *(const unsigned int*)&smem_B_fp8[nc][4 * tid]; \
            unsigned int b1 = *(const unsigned int*)&smem_B_fp8[nc][16 + 4 * tid]; \
            asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 " \
                "{%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%10,%11,%12,%13};" \
                :"=f"(acc[nt][0]),"=f"(acc[nt][1]),"=f"(acc[nt][2]),"=f"(acc[nt][3]) \
                :"r"(a0),"r"(a1),"r"(a2),"r"(a3),"r"(b0),"r"(b1), \
                 "f"(acc[nt][0]),"f"(acc[nt][1]),"f"(acc[nt][2]),"f"(acc[nt][3])); \
        } \
    } while(0)
#endif

    ISSUE_LOADS(0, 0);
    cp_async_commit();
    cp_async_wait_all();
    __syncthreads();
    DEQUANT_T(0);
    __syncthreads();

    int cur = 0;
    for (unsigned int k_base = K_STEP_T; k_base < K; k_base += K_STEP_T) {
        int nxt = 1 - cur;
        ISSUE_LOADS(nxt, k_base);
        cp_async_commit();
        COMPUTE_MMA(cur);
        cp_async_wait_all();
        __syncthreads();
        DEQUANT_T(nxt);
        __syncthreads();
        cur = nxt;
    }

    COMPUTE_MMA(cur);

    #undef ISSUE_LOADS
    #undef DEQUANT_T
    #undef COMPUTE_MMA

    #pragma unroll
    for (int nt = 0; nt < 16; nt++) {
        unsigned int c0 = cta_n + nt*8 + tid*2;
        unsigned int c1 = c0 + 1;
        unsigned int r0 = cta_m + warp_m_offset + group_id;
        unsigned int r1 = r0 + 8;
        if (r0 < M && c0 < N) C[r0*N+c0] = __float2bfloat16(acc[nt][0]);
        if (r0 < M && c1 < N) C[r0*N+c1] = __float2bfloat16(acc[nt][1]);
        if (r1 < M && c0 < N) C[r1*N+c0] = __float2bfloat16(acc[nt][2]);
        if (r1 < M && c1 < N) C[r1*N+c1] = __float2bfloat16(acc[nt][3]);
    }
}

// 2026-09-25: w4a16_gemm_t with the weight tiles triple-buffered, so step i+2's weight loads stay in flight
// across the dequant of step i+1; A stays double-buffered. It walks K / 32 steps, so K must be a multiple
// of 32; then its MMAs and their order match w4a16_gemm_t. tgemm_kernel (model-layers layers/mod.rs)
// prefers it unless METRALE_NO_TGEMM_PIPELINE3 is set.










extern "C" __global__ void w4a16_gemm_t_p3(
    const __nv_bfloat16* __restrict__ A,
    const unsigned char* __restrict__ B_packed,
    const unsigned char* __restrict__ B_scale,
    const float scale2,
    __nv_bfloat16* __restrict__ C,
    unsigned int M, unsigned int N, unsigned int K,
    unsigned int ldb
) {
    const unsigned int LDB = ldb;
    const unsigned int cta_n = blockIdx.x * N_TILE_LG;
    const unsigned int cta_m = blockIdx.y * M_TILE;
    const unsigned int warp_id = threadIdx.x / 32;
    const unsigned int lane_id = threadIdx.x % 32;
    const unsigned int warp_m_offset = warp_id * 16;
    const unsigned int group_id = lane_id >> 2;
    const unsigned int tid = lane_id & 3;

    __shared__ __nv_bfloat16 smem_A_p3[2][M_TILE][K_STEP_T + PAD_T];
    __shared__ unsigned char smem_Bp_p3[3][K_STEP_T / 2][N_TILE_LG + BP_PAD];
    __shared__ unsigned char smem_Bs_p3[3][K_STEP_T / GROUP_SIZE][N_TILE_LG + BP_PAD];
#if defined(__SCALE__)
    __shared__ __nv_bfloat16 smem_B_bf16_p3[N_TILE_LG][K_STEP_T];
#else
    __shared__ unsigned char smem_B_fp8_p3[N_TILE_LG][K_STEP_T];
#endif
    __shared__ float smem_LUT_p3[16];

    if (threadIdx.x < 16) smem_LUT_p3[threadIdx.x] = E2M1_LUT[threadIdx.x];

    float acc[16][4];
    #pragma unroll
    for (int i = 0; i < 16; i++) {
        acc[i][0] = 0.0f; acc[i][1] = 0.0f;
        acc[i][2] = 0.0f; acc[i][3] = 0.0f;
    }

    const unsigned int a_stride = K_STEP_T + PAD_T;

    #define P3_ISSUE_LOADS(abuf, bbuf, kb) do { \
        { \
            unsigned int a_row_base = threadIdx.x >> 2; \
            unsigned int a_col = (threadIdx.x & 3) << 3; \
            unsigned int gc = (kb) + a_col; \
            _Pragma("unroll") \
            for (int rnd = 0; rnd < 2; rnd++) { \
                unsigned int row = rnd * 32 + a_row_base; \
                unsigned int gr = cta_m + row; \
                cp_async_pred_16(&smem_A_p3[(abuf)][row][a_col], \
                    &A[gr * K + gc], (gr < M) && (gc + 7 < K)); \
            } \
        } \
        { \
            unsigned int kp = threadIdx.x >> 3; \
            unsigned int ns = (threadIdx.x & 7) << 4; \
            unsigned int gke = (kb) + (kp << 1); \
            unsigned int gns = cta_n + ns; \
            cp_async_pred_16(&smem_Bp_p3[(bbuf)][kp][ns], \
                &B_packed[(unsigned long long)(gke >> 1) * LDB + gns], \
                (gke + 1 <= K) && (gns + 15 < LDB)); \
            if (kp < K_STEP_T / GROUP_SIZE) { \
                unsigned int sg = (kb) / GROUP_SIZE + kp; \
                cp_async_pred_16(&smem_Bs_p3[(bbuf)][kp][ns], \
                    &B_scale[(unsigned long long)sg * LDB + gns], \
                    (gns + 15 < LDB)); \
            } \
        } \
    } while(0)

#if defined(__SCALE__)



    #define P3_DEQUANT_T(buf) do { \
        unsigned int my_n = threadIdx.x; \
        unsigned char sb0 = smem_Bs_p3[(buf)][0][my_n]; \
        unsigned char sb1 = smem_Bs_p3[(buf)][1][my_n]; \
        __nv_fp8_e4m3 f0, f1; \
        *(unsigned char*)&f0 = sb0; *(unsigned char*)&f1 = sb1; \
        float sv0 = scl_fp8(*(const unsigned char*)&f0) * scale2, sv1 = scl_fp8(*(const unsigned char*)&f1) * scale2; \
        _Pragma("unroll") \
        for (int kp = 0; kp < 8; kp++) { \
            unsigned char packed = smem_Bp_p3[(buf)][kp][my_n]; \
            smem_B_bf16_p3[my_n][kp * 2]     = __float2bfloat16(smem_LUT_p3[packed & 0xF] * sv0); \
            smem_B_bf16_p3[my_n][kp * 2 + 1] = __float2bfloat16(smem_LUT_p3[packed >> 4] * sv0); \
        } \
        _Pragma("unroll") \
        for (int kp = 8; kp < 16; kp++) { \
            unsigned char packed = smem_Bp_p3[(buf)][kp][my_n]; \
            smem_B_bf16_p3[my_n][kp * 2]     = __float2bfloat16(smem_LUT_p3[packed & 0xF] * sv1); \
            smem_B_bf16_p3[my_n][kp * 2 + 1] = __float2bfloat16(smem_LUT_p3[packed >> 4] * sv1); \
        } \
    } while(0)


    #define P3_COMPUTE_MMA(a_buf) do { \
        const __nv_bfloat16* sA = (const __nv_bfloat16*)smem_A_p3[(a_buf)]; \
        unsigned int fr0 = warp_m_offset + group_id, fr1 = fr0 + 8; \
        _Pragma("unroll") \
        for (int h = 0; h < 2; h++) { \
            unsigned int fc0 = h * 16 + tid * 2, fc1 = fc0 + 8; \
            unsigned int a0 = *(const unsigned int*)&sA[fr0 * a_stride + fc0]; \
            unsigned int a1 = *(const unsigned int*)&sA[fr1 * a_stride + fc0]; \
            unsigned int a2 = *(const unsigned int*)&sA[fr0 * a_stride + fc1]; \
            unsigned int a3 = *(const unsigned int*)&sA[fr1 * a_stride + fc1]; \
            _Pragma("unroll") \
            for (int nt = 0; nt < 16; nt++) { \
                unsigned int nc = nt * 8 + group_id; \
                const __nv_bfloat16* sb = &smem_B_bf16_p3[nc][0]; \
                unsigned int b0 = *(const unsigned int*)&sb[fc0]; \
                unsigned int b1 = *(const unsigned int*)&sb[fc1]; \
                asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 " \
                    "{%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%10,%11,%12,%13};" \
                    : "=f"(acc[nt][0]), "=f"(acc[nt][1]), "=f"(acc[nt][2]), "=f"(acc[nt][3]) \
                    : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1), \
                      "f"(acc[nt][0]), "f"(acc[nt][1]), "f"(acc[nt][2]), "f"(acc[nt][3])); \
            } \
        } \
    } while(0)
#else

    #define P3_DEQUANT_T(buf) do { \
        unsigned int my_n = threadIdx.x; \
        unsigned char sb0 = smem_Bs_p3[(buf)][0][my_n]; \
        unsigned char sb1 = smem_Bs_p3[(buf)][1][my_n]; \
        __nv_fp8_e4m3 f0, f1; \
        *(unsigned char*)&f0 = sb0; *(unsigned char*)&f1 = sb1; \
        float sv0 = (float)f0 * scale2, sv1 = (float)f1 * scale2; \
        _Pragma("unroll") \
        for (int kp = 0; kp < 8; kp++) { \
            unsigned char packed = smem_Bp_p3[(buf)][kp][my_n]; \
            float lo = smem_LUT_p3[packed & 0xF] * sv0; \
            float hi = smem_LUT_p3[packed >> 4] * sv0; \
            unsigned short fp8_pair; \
            asm volatile("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" \
                         : "=h"(fp8_pair) : "f"(hi), "f"(lo)); \
            *(unsigned short*)&smem_B_fp8_p3[my_n][kp * 2] = fp8_pair; \
        } \
        _Pragma("unroll") \
        for (int kp = 8; kp < 16; kp++) { \
            unsigned char packed = smem_Bp_p3[(buf)][kp][my_n]; \
            float lo = smem_LUT_p3[packed & 0xF] * sv1; \
            float hi = smem_LUT_p3[packed >> 4] * sv1; \
            unsigned short fp8_pair; \
            asm volatile("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" \
                         : "=h"(fp8_pair) : "f"(hi), "f"(lo)); \
            *(unsigned short*)&smem_B_fp8_p3[my_n][kp * 2] = fp8_pair; \
        } \
    } while(0)


    #define P3_COMPUTE_MMA(a_buf) do { \
        const unsigned short* sA = (const unsigned short*)smem_A_p3[(a_buf)]; \
        unsigned int fr0 = warp_m_offset + group_id, fr1 = fr0 + 8; \
        unsigned int a0 = bf16x4_to_e4m3x4(&sA[fr0 * a_stride + tid * 4]); \
        unsigned int a1 = bf16x4_to_e4m3x4(&sA[fr1 * a_stride + tid * 4]); \
        unsigned int a2 = bf16x4_to_e4m3x4(&sA[fr0 * a_stride + 16 + tid * 4]); \
        unsigned int a3 = bf16x4_to_e4m3x4(&sA[fr1 * a_stride + 16 + tid * 4]); \
        _Pragma("unroll") \
        for (int nt = 0; nt < 16; nt++) { \
            unsigned int nc = nt * 8 + group_id; \
            unsigned int b0 = *(const unsigned int*)&smem_B_fp8_p3[nc][4 * tid]; \
            unsigned int b1 = *(const unsigned int*)&smem_B_fp8_p3[nc][16 + 4 * tid]; \
            asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 " \
                "{%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%10,%11,%12,%13};" \
                :"=f"(acc[nt][0]),"=f"(acc[nt][1]),"=f"(acc[nt][2]),"=f"(acc[nt][3]) \
                :"r"(a0),"r"(a1),"r"(a2),"r"(a3),"r"(b0),"r"(b1), \
                 "f"(acc[nt][0]),"f"(acc[nt][1]),"f"(acc[nt][2]),"f"(acc[nt][3])); \
        } \
    } while(0)
#endif










    const unsigned int nsteps_p3 = K / K_STEP_T;
    P3_ISSUE_LOADS(0, 0, 0);
    cp_async_commit();
    if (nsteps_p3 > 1) { P3_ISSUE_LOADS(1, 1, K_STEP_T); }
    cp_async_commit();
    cp_async_wait_group1();
    __syncthreads();
    P3_DEQUANT_T(0);
    __syncthreads();

    for (unsigned int i = 0; i + 1 < nsteps_p3; i++) {
        P3_COMPUTE_MMA(i & 1);
        __syncthreads();
        if (i + 2 < nsteps_p3) {
            P3_ISSUE_LOADS((i + 2) & 1, (i + 2) % 3, (i + 2) * K_STEP_T);
        }
        cp_async_commit();
        cp_async_wait_group1();
        __syncthreads();
        P3_DEQUANT_T((i + 1) % 3);
        __syncthreads();
    }

    P3_COMPUTE_MMA((nsteps_p3 - 1) & 1);

    #undef P3_ISSUE_LOADS
    #undef P3_DEQUANT_T
    #undef P3_COMPUTE_MMA

    #pragma unroll
    for (int nt = 0; nt < 16; nt++) {
        unsigned int c0 = cta_n + nt*8 + tid*2;
        unsigned int c1 = c0 + 1;
        unsigned int r0 = cta_m + warp_m_offset + group_id;
        unsigned int r1 = r0 + 8;
        if (r0 < M && c0 < N) C[r0*N+c0] = __float2bfloat16(acc[nt][0]);
        if (r0 < M && c1 < N) C[r0*N+c1] = __float2bfloat16(acc[nt][1]);
        if (r1 < M && c0 < N) C[r1*N+c0] = __float2bfloat16(acc[nt][2]);
        if (r1 < M && c1 < N) C[r1*N+c1] = __float2bfloat16(acc[nt][3]);
    }
}

// 2026-09-25: C[M, N] = A[M, K] x B[N, K]^T with B already E4M3 (predequant_nvfp4_to_fp8) and A converted
// BF16 -> E4M3 in registers (metrale_cvt_e4m3x2_f32, unscaled); E4M3 MMAs with FP32 accumulation, BF16 out. 128 threads,
// 64 x 128 tile, K step 32, A and B double-buffered with cp.async.









extern "C" __global__ void fp8_gemm_t(
    const __nv_bfloat16* __restrict__ A,
    const unsigned char* __restrict__ B_fp8,
    __nv_bfloat16* __restrict__ C,
    unsigned int M, unsigned int N, unsigned int K
) {
    const unsigned int cta_n = blockIdx.x * N_TILE_LG;
    const unsigned int cta_m = blockIdx.y * M_TILE;
    const unsigned int warp_id = threadIdx.x / 32;
    const unsigned int lane_id = threadIdx.x % 32;
    const unsigned int warp_m_offset = warp_id * 16;
    const unsigned int group_id = lane_id >> 2;
    const unsigned int tid = lane_id & 3;

    __shared__ __nv_bfloat16 smem_A[2][M_TILE][K_STEP_T + PAD_T];
    __shared__ unsigned char smem_B[2][N_TILE_LG][K_STEP_T];

    float acc[16][4];
    #pragma unroll
    for (int i = 0; i < 16; i++) {
        acc[i][0] = 0.0f; acc[i][1] = 0.0f;
        acc[i][2] = 0.0f; acc[i][3] = 0.0f;
    }

    const unsigned int a_stride = K_STEP_T + PAD_T;


    #define FP8_LOADS(buf, kb) do { \
        { \
            unsigned int a_row_base = threadIdx.x >> 2; \
            unsigned int a_col = (threadIdx.x & 3) << 3; \
            unsigned int gc = (kb) + a_col; \
            _Pragma("unroll") \
            for (int rnd = 0; rnd < 2; rnd++) { \
                unsigned int row = rnd * 32 + a_row_base; \
                unsigned int gr = cta_m + row; \
                cp_async_pred_16(&smem_A[(buf)][row][a_col], \
                    &A[(unsigned long long)gr * K + gc], \
                    (gr < M) && (gc + 7 < K)); \
            } \
        } \
        { \
            unsigned int my_n = threadIdx.x; \
            unsigned int gn = cta_n + my_n; \
            bool valid = (gn < N) && ((kb) + 31 < K); \
            cp_async_pred_16(&smem_B[(buf)][my_n][0], \
                &B_fp8[(unsigned long long)gn * K + (kb)], valid); \
            cp_async_pred_16(&smem_B[(buf)][my_n][16], \
                &B_fp8[(unsigned long long)gn * K + (kb) + 16], valid); \
        } \
    } while(0)


    #define FP8_COMPUTE(a_buf, b_buf) do { \
        const unsigned short* sA = (const unsigned short*)smem_A[(a_buf)]; \
        unsigned int fr0 = warp_m_offset + group_id, fr1 = fr0 + 8; \
        unsigned int a0 = bf16x4_to_e4m3x4(&sA[fr0 * a_stride + tid * 4]); \
        unsigned int a1 = bf16x4_to_e4m3x4(&sA[fr1 * a_stride + tid * 4]); \
        unsigned int a2 = bf16x4_to_e4m3x4(&sA[fr0 * a_stride + 16 + tid * 4]); \
        unsigned int a3 = bf16x4_to_e4m3x4(&sA[fr1 * a_stride + 16 + tid * 4]); \
        _Pragma("unroll") \
        for (int nt = 0; nt < 16; nt++) { \
            unsigned int nc = nt * 8 + group_id; \
            unsigned int b0 = *(const unsigned int*)&smem_B[(b_buf)][nc][4 * tid]; \
            unsigned int b1 = *(const unsigned int*)&smem_B[(b_buf)][nc][16 + 4 * tid]; \
            metrale_mma_e4m3(acc[nt], a0,a1,a2,a3, b0, b1); \
        } \
    } while(0)


    FP8_LOADS(0, 0);
    cp_async_commit();
    cp_async_wait_all();
    __syncthreads();


    int cur = 0;
    for (unsigned int k_base = K_STEP_T; k_base < K; k_base += K_STEP_T) {
        int nxt = 1 - cur;
        FP8_LOADS(nxt, k_base);
        cp_async_commit();
        FP8_COMPUTE(cur, cur);
        cp_async_wait_all();
        __syncthreads();
        cur = nxt;
    }
    FP8_COMPUTE(cur, cur);

    #undef FP8_LOADS
    #undef FP8_COMPUTE

    #pragma unroll
    for (int nt = 0; nt < 16; nt++) {
        unsigned int c0 = cta_n + nt*8 + tid*2;
        unsigned int c1 = c0 + 1;
        unsigned int r0 = cta_m + warp_m_offset + group_id;
        unsigned int r1 = r0 + 8;
        if (r0 < M && c0 < N) C[r0*N+c0] = __float2bfloat16(acc[nt][0]);
        if (r0 < M && c1 < N) C[r0*N+c1] = __float2bfloat16(acc[nt][1]);
        if (r1 < M && c0 < N) C[r1*N+c0] = __float2bfloat16(acc[nt][2]);
        if (r1 < M && c1 < N) C[r1*N+c1] = __float2bfloat16(acc[nt][3]);
    }
}

// 2026-09-25: B_packed [N, K/2] and B_scale [N, K/16] (NVFP4) -> B_fp8 [N, K] E4M3 through metrale_cvt_e4m3x2_f32; one packed byte
// (two values) per thread over N * K / 2 threads.






extern "C" __global__ void predequant_nvfp4_to_fp8(
    const unsigned char* __restrict__ B_packed,
    const unsigned char* __restrict__ B_scale,
    float scale2,
    unsigned char* __restrict__ B_fp8,
    unsigned int N, unsigned int K
) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int half_K = K / 2;
    unsigned int total = N * half_K;
    if (idx >= total) return;

    unsigned int n = idx / half_K;
    unsigned int k_pair = idx % half_K;
    unsigned int k_even = k_pair * 2;

    unsigned char packed = B_packed[(unsigned long long)n * half_K + k_pair];
    unsigned int group = k_even / GROUP_SIZE;
    unsigned char sb = B_scale[(unsigned long long)n * (K / GROUP_SIZE) + group];
    __nv_fp8_e4m3 fp8_scale;
    *(unsigned char*)&fp8_scale = sb;
#if defined(__SCALE__)
    float sv = scl_fp8(sb) * scale2;
#else
    float sv = (float)fp8_scale * scale2;
#endif

    float val_lo = E2M1_LUT[packed & 0xF] * sv;
    float val_hi = E2M1_LUT[packed >> 4] * sv;

    unsigned short fp8_pair;
    fp8_pair = metrale_cvt_e4m3x2_f32(val_hi, val_lo);

    *(unsigned short*)&B_fp8[(unsigned long long)n * K + k_even] = fp8_pair;
}

// 2026-09-25: BF16 -> E4M3 through metrale_cvt_e4m3x2_f32, unscaled; two values per thread over total_elements / 2 threads,
// so total_elements must be even.




extern "C" __global__ void bf16_to_fp8(
    const __nv_bfloat16* __restrict__ src,
    unsigned char* __restrict__ dst,
    unsigned int total_elements
) {
    unsigned int idx = (blockIdx.x * blockDim.x + threadIdx.x) * 2;
    if (idx >= total_elements) return;

    unsigned int p = *(const unsigned int*)&src[idx];
    unsigned short bf0 = (unsigned short)(p & 0xFFFFu);
    unsigned short bf1 = (unsigned short)(p >> 16);
    float f0, f1;
    asm volatile("cvt.f32.bf16 %0, %1;" : "=f"(f0) : "h"(bf0));
    asm volatile("cvt.f32.bf16 %0, %1;" : "=f"(f1) : "h"(bf1));
    unsigned short fp8_pair;
    fp8_pair = metrale_cvt_e4m3x2_f32(f1, f0);
    *(unsigned short*)&dst[idx] = fp8_pair;
}

// 2026-09-25: C[M, N] = A[M, K] x B[N, K]^T with A and B both E4M3 (bf16_to_fp8, predequant_nvfp4_to_fp8);
// FP32 accumulation, BF16 out. Same tiling as fp8_gemm_t: 128 threads, 64 x 128 tile, K step 32.







#define A_FP8_STRIDE 32

extern "C" __global__ void fp8_fp8_gemm_t(
    const unsigned char* __restrict__ A_fp8,
    const unsigned char* __restrict__ B_fp8,
    __nv_bfloat16* __restrict__ C,
    unsigned int M, unsigned int N, unsigned int K
) {
    const unsigned int cta_n = blockIdx.x * N_TILE_LG;
    const unsigned int cta_m = blockIdx.y * M_TILE;
    const unsigned int warp_id = threadIdx.x / 32;
    const unsigned int lane_id = threadIdx.x % 32;
    const unsigned int warp_m_offset = warp_id * 16;
    const unsigned int group_id = lane_id >> 2;
    const unsigned int tid = lane_id & 3;


    __shared__ unsigned char smem_Af[2][M_TILE][A_FP8_STRIDE];
    __shared__ unsigned char smem_Bf[2][N_TILE_LG][K_STEP_T];

    float acc[16][4];
    #pragma unroll
    for (int i = 0; i < 16; i++) {
        acc[i][0] = 0.0f; acc[i][1] = 0.0f;
        acc[i][2] = 0.0f; acc[i][3] = 0.0f;
    }


    #define FF_LOADS(buf, kb) do { \
        { \
            /* 2026-09-25: 128 threads load 64 rows x 32 bytes, 16 bytes each */ \
            unsigned int a_row_base = threadIdx.x >> 1; \
            unsigned int a_col = (threadIdx.x & 1) << 4; \
            unsigned int gc = (kb) + a_col; \
            unsigned int row = a_row_base; \
            unsigned int gr = cta_m + row; \
            cp_async_pred_16(&smem_Af[(buf)][row][a_col], \
                &A_fp8[(unsigned long long)gr * K + gc], \
                (gr < M) && (gc + 15 < K)); \
        } \
        { \
            unsigned int my_n = threadIdx.x; \
            unsigned int gn = cta_n + my_n; \
            bool valid = (gn < N) && ((kb) + 31 < K); \
            cp_async_pred_16(&smem_Bf[(buf)][my_n][0], \
                &B_fp8[(unsigned long long)gn * K + (kb)], valid); \
            cp_async_pred_16(&smem_Bf[(buf)][my_n][16], \
                &B_fp8[(unsigned long long)gn * K + (kb) + 16], valid); \
        } \
    } while(0)


    #define FF_COMPUTE(a_buf, b_buf) do { \
        unsigned int fr0 = warp_m_offset + group_id, fr1 = fr0 + 8; \
        /* 2026-09-25: four A registers, each four E4M3 bytes (m16 x k32) */ \
        unsigned int a0 = *(const unsigned int*)&smem_Af[(a_buf)][fr0][4 * tid]; \
        unsigned int a1 = *(const unsigned int*)&smem_Af[(a_buf)][fr1][4 * tid]; \
        unsigned int a2 = *(const unsigned int*)&smem_Af[(a_buf)][fr0][16 + 4 * tid]; \
        unsigned int a3 = *(const unsigned int*)&smem_Af[(a_buf)][fr1][16 + 4 * tid]; \
        _Pragma("unroll") \
        for (int nt = 0; nt < 16; nt++) { \
            unsigned int nc = nt * 8 + group_id; \
            unsigned int b0 = *(const unsigned int*)&smem_Bf[(b_buf)][nc][4 * tid]; \
            unsigned int b1 = *(const unsigned int*)&smem_Bf[(b_buf)][nc][16 + 4 * tid]; \
            metrale_mma_e4m3(acc[nt], a0,a1,a2,a3, b0, b1); \
        } \
    } while(0)


    FF_LOADS(0, 0);
    cp_async_commit();
    cp_async_wait_all();
    __syncthreads();


    int cur = 0;
    for (unsigned int k_base = K_STEP_T; k_base < K; k_base += K_STEP_T) {
        int nxt = 1 - cur;
        FF_LOADS(nxt, k_base);
        cp_async_commit();
        FF_COMPUTE(cur, cur);
        cp_async_wait_all();
        __syncthreads();
        cur = nxt;
    }
    FF_COMPUTE(cur, cur);

    #undef FF_LOADS
    #undef FF_COMPUTE


    #pragma unroll
    for (int nt = 0; nt < 16; nt++) {
        unsigned int c0 = cta_n + nt*8 + tid*2;
        unsigned int c1 = c0 + 1;
        unsigned int r0 = cta_m + warp_m_offset + group_id;
        unsigned int r1 = r0 + 8;
        if (r0 < M && c0 < N) C[r0*N+c0] = __float2bfloat16(acc[nt][0]);
        if (r0 < M && c1 < N) C[r0*N+c1] = __float2bfloat16(acc[nt][1]);
        if (r1 < M && c0 < N) C[r1*N+c0] = __float2bfloat16(acc[nt][2]);
        if (r1 < M && c1 < N) C[r1*N+c1] = __float2bfloat16(acc[nt][3]);
    }
}

// 2026-09-25: w4a16_gemm_t with B's row stride fixed at N (no ldb) and a K step of 64: two m16n8k32 E4M3 MMAs
// per 8-column N tile and step. K must be a multiple of 64.










#define K_STEP_T64 64
#define PAD_T64    8   // 2026-09-25: (64 + 8) BF16 = 144-byte rows keep cp.async destinations 16-byte aligned.

extern "C" __global__ void w4a16_gemm_t_k64(
    const __nv_bfloat16* __restrict__ A,
    const unsigned char* __restrict__ B_packed,
    const unsigned char* __restrict__ B_scale,
    const float scale2,
    __nv_bfloat16* __restrict__ C,
    unsigned int M, unsigned int N, unsigned int K
) {
    const unsigned int cta_n = blockIdx.x * N_TILE_LG;
    const unsigned int cta_m = blockIdx.y * M_TILE;
    const unsigned int warp_id = threadIdx.x / 32;
    const unsigned int lane_id = threadIdx.x % 32;
    const unsigned int warp_m_offset = warp_id * 16;
    const unsigned int group_id = lane_id >> 2;
    const unsigned int tid = lane_id & 3;

    // 2026-09-25: B_fp8 rows are K64 + 16 = 80 bytes, so a warp's 4-byte fragment reads hit 32 distinct banks.
    __shared__ __nv_bfloat16 smem_A_k64[2][M_TILE][K_STEP_T64 + PAD_T64];
    __shared__ unsigned char smem_Bp_k64[2][K_STEP_T64 / 2][N_TILE_LG + BP_PAD];
    __shared__ unsigned char smem_Bs_k64[2][K_STEP_T64 / GROUP_SIZE][N_TILE_LG + BP_PAD];
    __shared__ unsigned char smem_B_fp8_k64[N_TILE_LG][K_STEP_T64 + 16];
    __shared__ float smem_LUT_k64[16];

    if (threadIdx.x < 16) smem_LUT_k64[threadIdx.x] = E2M1_LUT[threadIdx.x];

    float acc[16][4];
    #pragma unroll
    for (int i = 0; i < 16; i++) {
        acc[i][0] = 0.0f; acc[i][1] = 0.0f;
        acc[i][2] = 0.0f; acc[i][3] = 0.0f;
    }

    const unsigned int ast64 = K_STEP_T64 + PAD_T64;




    #define K64_ISSUE_LOADS(buf, kb) do { \
        { \
            unsigned int a_row_base = threadIdx.x >> 3; \
            unsigned int a_col = (threadIdx.x & 7) << 3; \
            unsigned int gc = (kb) + a_col; \
            _Pragma("unroll") \
            for (int rnd = 0; rnd < 4; rnd++) { \
                unsigned int row = rnd * 16 + a_row_base; \
                unsigned int gr = cta_m + row; \
                cp_async_pred_16(&smem_A_k64[(buf)][row][a_col], \
                    &A[(unsigned long long)gr * K + gc], \
                    (gr < M) && (gc + 7 < K)); \
            } \
        } \
        { \
            unsigned int kp = threadIdx.x >> 3; \
            unsigned int ns = (threadIdx.x & 7) << 4; \
            unsigned int gns = cta_n + ns; \
            _Pragma("unroll") \
            for (int rnd = 0; rnd < 2; rnd++) { \
                unsigned int kp_cur = rnd * 16 + kp; \
                unsigned int gke = (kb) + (kp_cur << 1); \
                cp_async_pred_16(&smem_Bp_k64[(buf)][kp_cur][ns], \
                    &B_packed[(unsigned long long)(gke >> 1) * N + gns], \
                    (gke + 1 <= K) && (gns + 15 < N)); \
                if (kp_cur < K_STEP_T64 / GROUP_SIZE) { \
                    unsigned int sg = (kb) / GROUP_SIZE + kp_cur; \
                    cp_async_pred_16(&smem_Bs_k64[(buf)][kp_cur][ns], \
                        &B_scale[(unsigned long long)sg * N + gns], \
                        (gns + 15 < N)); \
                } \
            } \
        } \
    } while(0)



#if defined(__SCALE__)
    #define K64_DEQUANT(buf) do { \
        unsigned int my_n = threadIdx.x; \
        __nv_fp8_e4m3 f0, f1, f2, f3; \
        *(unsigned char*)&f0 = smem_Bs_k64[(buf)][0][my_n]; \
        *(unsigned char*)&f1 = smem_Bs_k64[(buf)][1][my_n]; \
        *(unsigned char*)&f2 = smem_Bs_k64[(buf)][2][my_n]; \
        *(unsigned char*)&f3 = smem_Bs_k64[(buf)][3][my_n]; \
        float sv0 = scl_fp8(*(const unsigned char*)&f0) * scale2, sv1 = scl_fp8(*(const unsigned char*)&f1) * scale2; \
        float sv2 = scl_fp8(*(const unsigned char*)&f2) * scale2, sv3 = scl_fp8(*(const unsigned char*)&f3) * scale2; \
        _Pragma("unroll") \
        for (int kp = 0; kp < 8; kp++) { \
            unsigned char packed = smem_Bp_k64[(buf)][kp][my_n]; \
            float lo = smem_LUT_k64[packed & 0xF] * sv0; \
            float hi = smem_LUT_k64[packed >> 4] * sv0; \
            unsigned short fp8_pair; \
            fp8_pair = metrale_cvt_e4m3x2_f32(hi, lo); \
            *(unsigned short*)&smem_B_fp8_k64[my_n][kp * 2] = fp8_pair; \
        } \
        _Pragma("unroll") \
        for (int kp = 8; kp < 16; kp++) { \
            unsigned char packed = smem_Bp_k64[(buf)][kp][my_n]; \
            float lo = smem_LUT_k64[packed & 0xF] * sv1; \
            float hi = smem_LUT_k64[packed >> 4] * sv1; \
            unsigned short fp8_pair; \
            fp8_pair = metrale_cvt_e4m3x2_f32(hi, lo); \
            *(unsigned short*)&smem_B_fp8_k64[my_n][kp * 2] = fp8_pair; \
        } \
        _Pragma("unroll") \
        for (int kp = 16; kp < 24; kp++) { \
            unsigned char packed = smem_Bp_k64[(buf)][kp][my_n]; \
            float lo = smem_LUT_k64[packed & 0xF] * sv2; \
            float hi = smem_LUT_k64[packed >> 4] * sv2; \
            unsigned short fp8_pair; \
            fp8_pair = metrale_cvt_e4m3x2_f32(hi, lo); \
            *(unsigned short*)&smem_B_fp8_k64[my_n][kp * 2] = fp8_pair; \
        } \
        _Pragma("unroll") \
        for (int kp = 24; kp < 32; kp++) { \
            unsigned char packed = smem_Bp_k64[(buf)][kp][my_n]; \
            float lo = smem_LUT_k64[packed & 0xF] * sv3; \
            float hi = smem_LUT_k64[packed >> 4] * sv3; \
            unsigned short fp8_pair; \
            fp8_pair = metrale_cvt_e4m3x2_f32(hi, lo); \
            *(unsigned short*)&smem_B_fp8_k64[my_n][kp * 2] = fp8_pair; \
        } \
    } while(0)
#else
    #define K64_DEQUANT(buf) do { \
        unsigned int my_n = threadIdx.x; \
        __nv_fp8_e4m3 f0, f1, f2, f3; \
        *(unsigned char*)&f0 = smem_Bs_k64[(buf)][0][my_n]; \
        *(unsigned char*)&f1 = smem_Bs_k64[(buf)][1][my_n]; \
        *(unsigned char*)&f2 = smem_Bs_k64[(buf)][2][my_n]; \
        *(unsigned char*)&f3 = smem_Bs_k64[(buf)][3][my_n]; \
        float sv0 = (float)f0 * scale2, sv1 = (float)f1 * scale2; \
        float sv2 = (float)f2 * scale2, sv3 = (float)f3 * scale2; \
        _Pragma("unroll") \
        for (int kp = 0; kp < 8; kp++) { \
            unsigned char packed = smem_Bp_k64[(buf)][kp][my_n]; \
            float lo = smem_LUT_k64[packed & 0xF] * sv0; \
            float hi = smem_LUT_k64[packed >> 4] * sv0; \
            unsigned short fp8_pair; \
            asm volatile("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" \
                         : "=h"(fp8_pair) : "f"(hi), "f"(lo)); \
            *(unsigned short*)&smem_B_fp8_k64[my_n][kp * 2] = fp8_pair; \
        } \
        _Pragma("unroll") \
        for (int kp = 8; kp < 16; kp++) { \
            unsigned char packed = smem_Bp_k64[(buf)][kp][my_n]; \
            float lo = smem_LUT_k64[packed & 0xF] * sv1; \
            float hi = smem_LUT_k64[packed >> 4] * sv1; \
            unsigned short fp8_pair; \
            asm volatile("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" \
                         : "=h"(fp8_pair) : "f"(hi), "f"(lo)); \
            *(unsigned short*)&smem_B_fp8_k64[my_n][kp * 2] = fp8_pair; \
        } \
        _Pragma("unroll") \
        for (int kp = 16; kp < 24; kp++) { \
            unsigned char packed = smem_Bp_k64[(buf)][kp][my_n]; \
            float lo = smem_LUT_k64[packed & 0xF] * sv2; \
            float hi = smem_LUT_k64[packed >> 4] * sv2; \
            unsigned short fp8_pair; \
            asm volatile("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" \
                         : "=h"(fp8_pair) : "f"(hi), "f"(lo)); \
            *(unsigned short*)&smem_B_fp8_k64[my_n][kp * 2] = fp8_pair; \
        } \
        _Pragma("unroll") \
        for (int kp = 24; kp < 32; kp++) { \
            unsigned char packed = smem_Bp_k64[(buf)][kp][my_n]; \
            float lo = smem_LUT_k64[packed & 0xF] * sv3; \
            float hi = smem_LUT_k64[packed >> 4] * sv3; \
            unsigned short fp8_pair; \
            asm volatile("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" \
                         : "=h"(fp8_pair) : "f"(hi), "f"(lo)); \
            *(unsigned short*)&smem_B_fp8_k64[my_n][kp * 2] = fp8_pair; \
        } \
    } while(0)
#endif


    #define K64_COMPUTE_MMA(a_buf) do { \
        const unsigned short* sA = (const unsigned short*)smem_A_k64[(a_buf)]; \
        unsigned int fr0 = warp_m_offset + group_id, fr1 = fr0 + 8; \
        unsigned int a0 = bf16x4_to_e4m3x4(&sA[fr0 * ast64 + tid * 4]); \
        unsigned int a1 = bf16x4_to_e4m3x4(&sA[fr1 * ast64 + tid * 4]); \
        unsigned int a2 = bf16x4_to_e4m3x4(&sA[fr0 * ast64 + 16 + tid * 4]); \
        unsigned int a3 = bf16x4_to_e4m3x4(&sA[fr1 * ast64 + 16 + tid * 4]); \
        _Pragma("unroll") \
        for (int nt = 0; nt < 16; nt++) { \
            unsigned int nc = nt * 8 + group_id; \
            unsigned int b0 = *(const unsigned int*)&smem_B_fp8_k64[nc][4 * tid]; \
            unsigned int b1 = *(const unsigned int*)&smem_B_fp8_k64[nc][16 + 4 * tid]; \
            metrale_mma_e4m3(acc[nt], a0,a1,a2,a3, b0, b1); \
        } \
        unsigned int a4 = bf16x4_to_e4m3x4(&sA[fr0 * ast64 + 32 + tid * 4]); \
        unsigned int a5 = bf16x4_to_e4m3x4(&sA[fr1 * ast64 + 32 + tid * 4]); \
        unsigned int a6 = bf16x4_to_e4m3x4(&sA[fr0 * ast64 + 48 + tid * 4]); \
        unsigned int a7 = bf16x4_to_e4m3x4(&sA[fr1 * ast64 + 48 + tid * 4]); \
        _Pragma("unroll") \
        for (int nt = 0; nt < 16; nt++) { \
            unsigned int nc = nt * 8 + group_id; \
            unsigned int b0 = *(const unsigned int*)&smem_B_fp8_k64[nc][32 + 4 * tid]; \
            unsigned int b1 = *(const unsigned int*)&smem_B_fp8_k64[nc][48 + 4 * tid]; \
            metrale_mma_e4m3(acc[nt], a4,a5,a6,a7, b0, b1); \
        } \
    } while(0)

    K64_ISSUE_LOADS(0, 0);
    cp_async_commit();
    cp_async_wait_all();
    __syncthreads();
    K64_DEQUANT(0);
    __syncthreads();

    int cur = 0;
    for (unsigned int k_base = K_STEP_T64; k_base < K; k_base += K_STEP_T64) {
        int nxt = 1 - cur;
        K64_ISSUE_LOADS(nxt, k_base);
        cp_async_commit();
        K64_COMPUTE_MMA(cur);
        cp_async_wait_all();
        __syncthreads();
        K64_DEQUANT(nxt);
        __syncthreads();
        cur = nxt;
    }
    K64_COMPUTE_MMA(cur);

    #undef K64_ISSUE_LOADS
    #undef K64_DEQUANT
    #undef K64_COMPUTE_MMA
    #undef K_STEP_T64
    #undef PAD_T64

    #pragma unroll
    for (int nt = 0; nt < 16; nt++) {
        unsigned int c0 = cta_n + nt*8 + tid*2;
        unsigned int c1 = c0 + 1;
        unsigned int r0 = cta_m + warp_m_offset + group_id;
        unsigned int r1 = r0 + 8;
        if (r0 < M && c0 < N) C[r0*N+c0] = __float2bfloat16(acc[nt][0]);
        if (r0 < M && c1 < N) C[r0*N+c1] = __float2bfloat16(acc[nt][1]);
        if (r1 < M && c0 < N) C[r1*N+c0] = __float2bfloat16(acc[nt][2]);
        if (r1 < M && c1 < N) C[r1*N+c1] = __float2bfloat16(acc[nt][3]);
    }
}

// 2026-09-25: w4a16_gemm_t_k64 with the weight tiles triple-buffered (schedule note in the body). k64_kernel
// (model-layers layers/mod.rs) prefers it unless METRALE_NO_K64_PIPELINE3 is set.


#define K_STEP_T64 64
#define PAD_T64    8
extern "C" __global__ void w4a16_gemm_t_k64_p3(
    const __nv_bfloat16* __restrict__ A,
    const unsigned char* __restrict__ B_packed,
    const unsigned char* __restrict__ B_scale,
    const float scale2,
    __nv_bfloat16* __restrict__ C,
    unsigned int M, unsigned int N, unsigned int K
) {
    const unsigned int cta_n = blockIdx.x * N_TILE_LG;
    const unsigned int cta_m = blockIdx.y * M_TILE;
    const unsigned int warp_id = threadIdx.x / 32;
    const unsigned int lane_id = threadIdx.x % 32;
    const unsigned int warp_m_offset = warp_id * 16;
    const unsigned int group_id = lane_id >> 2;
    const unsigned int tid = lane_id & 3;

    // 2026-09-25: B_fp8 rows are K64 + 16 = 80 bytes, so a warp's 4-byte fragment reads hit 32 distinct banks.
    __shared__ __nv_bfloat16 smem_A_k64p3[2][M_TILE][K_STEP_T64 + PAD_T64];
    __shared__ unsigned char smem_Bp_k64p3[3][K_STEP_T64 / 2][N_TILE_LG + BP_PAD];
    __shared__ unsigned char smem_Bs_k64p3[3][K_STEP_T64 / GROUP_SIZE][N_TILE_LG + BP_PAD];
    __shared__ unsigned char smem_B_fp8_k64p3[N_TILE_LG][K_STEP_T64 + 16];
    __shared__ float smem_LUT_k64p3[16];

    if (threadIdx.x < 16) smem_LUT_k64p3[threadIdx.x] = E2M1_LUT[threadIdx.x];

    float acc[16][4];
    #pragma unroll
    for (int i = 0; i < 16; i++) {
        acc[i][0] = 0.0f; acc[i][1] = 0.0f;
        acc[i][2] = 0.0f; acc[i][3] = 0.0f;
    }

    const unsigned int ast64 = K_STEP_T64 + PAD_T64;




    #define K64P3_ISSUE_LOADS(abuf, bbuf, kb) do { \
        { \
            unsigned int a_row_base = threadIdx.x >> 3; \
            unsigned int a_col = (threadIdx.x & 7) << 3; \
            unsigned int gc = (kb) + a_col; \
            _Pragma("unroll") \
            for (int rnd = 0; rnd < 4; rnd++) { \
                unsigned int row = rnd * 16 + a_row_base; \
                unsigned int gr = cta_m + row; \
                cp_async_pred_16(&smem_A_k64p3[(abuf)][row][a_col], \
                    &A[(unsigned long long)gr * K + gc], \
                    (gr < M) && (gc + 7 < K)); \
            } \
        } \
        { \
            unsigned int kp = threadIdx.x >> 3; \
            unsigned int ns = (threadIdx.x & 7) << 4; \
            unsigned int gns = cta_n + ns; \
            _Pragma("unroll") \
            for (int rnd = 0; rnd < 2; rnd++) { \
                unsigned int kp_cur = rnd * 16 + kp; \
                unsigned int gke = (kb) + (kp_cur << 1); \
                cp_async_pred_16(&smem_Bp_k64p3[(bbuf)][kp_cur][ns], \
                    &B_packed[(unsigned long long)(gke >> 1) * N + gns], \
                    (gke + 1 <= K) && (gns + 15 < N)); \
                if (kp_cur < K_STEP_T64 / GROUP_SIZE) { \
                    unsigned int sg = (kb) / GROUP_SIZE + kp_cur; \
                    cp_async_pred_16(&smem_Bs_k64p3[(bbuf)][kp_cur][ns], \
                        &B_scale[(unsigned long long)sg * N + gns], \
                        (gns + 15 < N)); \
                } \
            } \
        } \
    } while(0)



#if defined(__SCALE__)
    #define K64P3_DEQUANT(buf) do { \
        unsigned int my_n = threadIdx.x; \
        __nv_fp8_e4m3 f0, f1, f2, f3; \
        *(unsigned char*)&f0 = smem_Bs_k64p3[(buf)][0][my_n]; \
        *(unsigned char*)&f1 = smem_Bs_k64p3[(buf)][1][my_n]; \
        *(unsigned char*)&f2 = smem_Bs_k64p3[(buf)][2][my_n]; \
        *(unsigned char*)&f3 = smem_Bs_k64p3[(buf)][3][my_n]; \
        float sv0 = scl_fp8(*(const unsigned char*)&f0) * scale2, sv1 = scl_fp8(*(const unsigned char*)&f1) * scale2; \
        float sv2 = scl_fp8(*(const unsigned char*)&f2) * scale2, sv3 = scl_fp8(*(const unsigned char*)&f3) * scale2; \
        _Pragma("unroll") \
        for (int kp = 0; kp < 8; kp++) { \
            unsigned char packed = smem_Bp_k64p3[(buf)][kp][my_n]; \
            float lo = smem_LUT_k64p3[packed & 0xF] * sv0; \
            float hi = smem_LUT_k64p3[packed >> 4] * sv0; \
            unsigned short fp8_pair; \
            fp8_pair = metrale_cvt_e4m3x2_f32(hi, lo); \
            *(unsigned short*)&smem_B_fp8_k64p3[my_n][kp * 2] = fp8_pair; \
        } \
        _Pragma("unroll") \
        for (int kp = 8; kp < 16; kp++) { \
            unsigned char packed = smem_Bp_k64p3[(buf)][kp][my_n]; \
            float lo = smem_LUT_k64p3[packed & 0xF] * sv1; \
            float hi = smem_LUT_k64p3[packed >> 4] * sv1; \
            unsigned short fp8_pair; \
            fp8_pair = metrale_cvt_e4m3x2_f32(hi, lo); \
            *(unsigned short*)&smem_B_fp8_k64p3[my_n][kp * 2] = fp8_pair; \
        } \
        _Pragma("unroll") \
        for (int kp = 16; kp < 24; kp++) { \
            unsigned char packed = smem_Bp_k64p3[(buf)][kp][my_n]; \
            float lo = smem_LUT_k64p3[packed & 0xF] * sv2; \
            float hi = smem_LUT_k64p3[packed >> 4] * sv2; \
            unsigned short fp8_pair; \
            fp8_pair = metrale_cvt_e4m3x2_f32(hi, lo); \
            *(unsigned short*)&smem_B_fp8_k64p3[my_n][kp * 2] = fp8_pair; \
        } \
        _Pragma("unroll") \
        for (int kp = 24; kp < 32; kp++) { \
            unsigned char packed = smem_Bp_k64p3[(buf)][kp][my_n]; \
            float lo = smem_LUT_k64p3[packed & 0xF] * sv3; \
            float hi = smem_LUT_k64p3[packed >> 4] * sv3; \
            unsigned short fp8_pair; \
            fp8_pair = metrale_cvt_e4m3x2_f32(hi, lo); \
            *(unsigned short*)&smem_B_fp8_k64p3[my_n][kp * 2] = fp8_pair; \
        } \
    } while(0)
#else
    #define K64P3_DEQUANT(buf) do { \
        unsigned int my_n = threadIdx.x; \
        __nv_fp8_e4m3 f0, f1, f2, f3; \
        *(unsigned char*)&f0 = smem_Bs_k64p3[(buf)][0][my_n]; \
        *(unsigned char*)&f1 = smem_Bs_k64p3[(buf)][1][my_n]; \
        *(unsigned char*)&f2 = smem_Bs_k64p3[(buf)][2][my_n]; \
        *(unsigned char*)&f3 = smem_Bs_k64p3[(buf)][3][my_n]; \
        float sv0 = (float)f0 * scale2, sv1 = (float)f1 * scale2; \
        float sv2 = (float)f2 * scale2, sv3 = (float)f3 * scale2; \
        _Pragma("unroll") \
        for (int kp = 0; kp < 8; kp++) { \
            unsigned char packed = smem_Bp_k64p3[(buf)][kp][my_n]; \
            float lo = smem_LUT_k64p3[packed & 0xF] * sv0; \
            float hi = smem_LUT_k64p3[packed >> 4] * sv0; \
            unsigned short fp8_pair; \
            asm volatile("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" \
                         : "=h"(fp8_pair) : "f"(hi), "f"(lo)); \
            *(unsigned short*)&smem_B_fp8_k64p3[my_n][kp * 2] = fp8_pair; \
        } \
        _Pragma("unroll") \
        for (int kp = 8; kp < 16; kp++) { \
            unsigned char packed = smem_Bp_k64p3[(buf)][kp][my_n]; \
            float lo = smem_LUT_k64p3[packed & 0xF] * sv1; \
            float hi = smem_LUT_k64p3[packed >> 4] * sv1; \
            unsigned short fp8_pair; \
            asm volatile("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" \
                         : "=h"(fp8_pair) : "f"(hi), "f"(lo)); \
            *(unsigned short*)&smem_B_fp8_k64p3[my_n][kp * 2] = fp8_pair; \
        } \
        _Pragma("unroll") \
        for (int kp = 16; kp < 24; kp++) { \
            unsigned char packed = smem_Bp_k64p3[(buf)][kp][my_n]; \
            float lo = smem_LUT_k64p3[packed & 0xF] * sv2; \
            float hi = smem_LUT_k64p3[packed >> 4] * sv2; \
            unsigned short fp8_pair; \
            asm volatile("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" \
                         : "=h"(fp8_pair) : "f"(hi), "f"(lo)); \
            *(unsigned short*)&smem_B_fp8_k64p3[my_n][kp * 2] = fp8_pair; \
        } \
        _Pragma("unroll") \
        for (int kp = 24; kp < 32; kp++) { \
            unsigned char packed = smem_Bp_k64p3[(buf)][kp][my_n]; \
            float lo = smem_LUT_k64p3[packed & 0xF] * sv3; \
            float hi = smem_LUT_k64p3[packed >> 4] * sv3; \
            unsigned short fp8_pair; \
            asm volatile("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" \
                         : "=h"(fp8_pair) : "f"(hi), "f"(lo)); \
            *(unsigned short*)&smem_B_fp8_k64p3[my_n][kp * 2] = fp8_pair; \
        } \
    } while(0)
#endif


    #define K64P3_COMPUTE_MMA(a_buf) do { \
        const unsigned short* sA = (const unsigned short*)smem_A_k64p3[(a_buf)]; \
        unsigned int fr0 = warp_m_offset + group_id, fr1 = fr0 + 8; \
        unsigned int a0 = bf16x4_to_e4m3x4(&sA[fr0 * ast64 + tid * 4]); \
        unsigned int a1 = bf16x4_to_e4m3x4(&sA[fr1 * ast64 + tid * 4]); \
        unsigned int a2 = bf16x4_to_e4m3x4(&sA[fr0 * ast64 + 16 + tid * 4]); \
        unsigned int a3 = bf16x4_to_e4m3x4(&sA[fr1 * ast64 + 16 + tid * 4]); \
        _Pragma("unroll") \
        for (int nt = 0; nt < 16; nt++) { \
            unsigned int nc = nt * 8 + group_id; \
            unsigned int b0 = *(const unsigned int*)&smem_B_fp8_k64p3[nc][4 * tid]; \
            unsigned int b1 = *(const unsigned int*)&smem_B_fp8_k64p3[nc][16 + 4 * tid]; \
            metrale_mma_e4m3(acc[nt], a0,a1,a2,a3, b0, b1); \
        } \
        unsigned int a4 = bf16x4_to_e4m3x4(&sA[fr0 * ast64 + 32 + tid * 4]); \
        unsigned int a5 = bf16x4_to_e4m3x4(&sA[fr1 * ast64 + 32 + tid * 4]); \
        unsigned int a6 = bf16x4_to_e4m3x4(&sA[fr0 * ast64 + 48 + tid * 4]); \
        unsigned int a7 = bf16x4_to_e4m3x4(&sA[fr1 * ast64 + 48 + tid * 4]); \
        _Pragma("unroll") \
        for (int nt = 0; nt < 16; nt++) { \
            unsigned int nc = nt * 8 + group_id; \
            unsigned int b0 = *(const unsigned int*)&smem_B_fp8_k64p3[nc][32 + 4 * tid]; \
            unsigned int b1 = *(const unsigned int*)&smem_B_fp8_k64p3[nc][48 + 4 * tid]; \
            metrale_mma_e4m3(acc[nt], a4,a5,a6,a7, b0, b1); \
        } \
    } while(0)

    // 2026-09-25: The loads for step i+2 are issued before step i+1 is dequantized and stay in flight across
    // it; w4a16_gemm_t_k64 drains its only group (cp_async_wait_all) before each dequant. A stays
    // double-buffered, because A[i&1] is free once MMA(i) passes its barrier, and only the weight tiles are
    // tripled: static shared memory is 44,288 bytes, under the 48 KiB static limit. It walks K / 64 steps;
    // with K a multiple of 64 its MMAs and their order match w4a16_gemm_t_k64.










    const unsigned int nsteps = K / K_STEP_T64;
    K64P3_ISSUE_LOADS(0, 0, 0);
    cp_async_commit();
    if (nsteps > 1) { K64P3_ISSUE_LOADS(1, 1, K_STEP_T64); }
    cp_async_commit();
    cp_async_wait_group1();
    __syncthreads();
    K64P3_DEQUANT(0);
    __syncthreads();

    for (unsigned int i = 0; i + 1 < nsteps; i++) {
        K64P3_COMPUTE_MMA(i & 1);
        __syncthreads();                       // 2026-09-25: MMA(i) is done with A[i&1] and B_fp8
        if (i + 2 < nsteps) {
            K64P3_ISSUE_LOADS((i + 2) & 1, (i + 2) % 3, (i + 2) * K_STEP_T64);
        }
        cp_async_commit();                     // 2026-09-25: commit even when empty, so wait_group 1 counts groups exactly
        cp_async_wait_group1();                // 2026-09-25: step i+1 has landed; step i+2 stays in flight
        __syncthreads();
        K64P3_DEQUANT((i + 1) % 3);
        __syncthreads();
    }
    K64P3_COMPUTE_MMA((nsteps - 1) & 1);

    #undef K64P3_ISSUE_LOADS
    #undef K64P3_DEQUANT
    #undef K64P3_COMPUTE_MMA
    #undef K_STEP_T64
    #undef PAD_T64

    #pragma unroll
    for (int nt = 0; nt < 16; nt++) {
        unsigned int c0 = cta_n + nt*8 + tid*2;
        unsigned int c1 = c0 + 1;
        unsigned int r0 = cta_m + warp_m_offset + group_id;
        unsigned int r1 = r0 + 8;
        if (r0 < M && c0 < N) C[r0*N+c0] = __float2bfloat16(acc[nt][0]);
        if (r0 < M && c1 < N) C[r0*N+c1] = __float2bfloat16(acc[nt][1]);
        if (r1 < M && c0 < N) C[r1*N+c0] = __float2bfloat16(acc[nt][2]);
        if (r1 < M && c1 < N) C[r1*N+c1] = __float2bfloat16(acc[nt][3]);
    }
}

// 2026-09-25: w4a16_gemm_t_k64_p3 with a 64-wide N tile, which doubles the grid. Each output element gets the
// same dequantized E4M3 bytes, K order and MMA operands as in w4a16_gemm_t_k64_p3. The w4a16_gemm launcher
// (model-layers ops/gemm_dense.rs) launches it with the same arguments.














#define K_STEP_T64 64
#define PAD_T64    8
extern "C" __global__ void w4a16_gemm_t_k64_n64_p3(
    const __nv_bfloat16* __restrict__ A,
    const unsigned char* __restrict__ B_packed,
    const unsigned char* __restrict__ B_scale,
    const float scale2,
    __nv_bfloat16* __restrict__ C,
    unsigned int M, unsigned int N, unsigned int K
) {
    const unsigned int cta_n = blockIdx.x * N_TILE_SM;
    const unsigned int cta_m = blockIdx.y * M_TILE;
    const unsigned int warp_id = threadIdx.x / 32;
    const unsigned int lane_id = threadIdx.x % 32;
    const unsigned int warp_m_offset = warp_id * 16;
    const unsigned int group_id = lane_id >> 2;
    const unsigned int tid = lane_id & 3;

    __shared__ __nv_bfloat16 smem_A_n64[2][M_TILE][K_STEP_T64 + PAD_T64];
    __shared__ unsigned char smem_Bp_n64[3][K_STEP_T64 / 2][N_TILE_SM + BP_PAD];
    __shared__ unsigned char smem_Bs_n64[3][K_STEP_T64 / GROUP_SIZE][N_TILE_SM + BP_PAD];
    __shared__ unsigned char smem_B_fp8_n64[N_TILE_SM][K_STEP_T64 + 16];
    __shared__ float smem_LUT_n64[16];

    if (threadIdx.x < 16) smem_LUT_n64[threadIdx.x] = E2M1_LUT[threadIdx.x];

    float acc[8][4];
    #pragma unroll
    for (int i = 0; i < 8; i++) { acc[i][0]=0.f; acc[i][1]=0.f; acc[i][2]=0.f; acc[i][3]=0.f; }

    const unsigned int ast64 = K_STEP_T64 + PAD_T64;




    #define N64_ISSUE_LOADS(abuf, bbuf, kb) do { \
        { \
            unsigned int a_row_base = threadIdx.x >> 3; \
            unsigned int a_col = (threadIdx.x & 7) << 3; \
            unsigned int gc = (kb) + a_col; \
            _Pragma("unroll") \
            for (int rnd = 0; rnd < 4; rnd++) { \
                unsigned int row = rnd * 16 + a_row_base; \
                unsigned int gr = cta_m + row; \
                cp_async_pred_16(&smem_A_n64[(abuf)][row][a_col], \
                    &A[(unsigned long long)gr * K + gc], \
                    (gr < M) && (gc + 7 < K)); \
            } \
        } \
        { \
            unsigned int kp_cur = threadIdx.x >> 2; \
            unsigned int ns = (threadIdx.x & 3) << 4; \
            unsigned int gns = cta_n + ns; \
            unsigned int gke = (kb) + (kp_cur << 1); \
            cp_async_pred_16(&smem_Bp_n64[(bbuf)][kp_cur][ns], \
                &B_packed[(unsigned long long)(gke >> 1) * N + gns], \
                (gke + 1 <= K) && (gns + 15 < N)); \
            if (kp_cur < K_STEP_T64 / GROUP_SIZE) { \
                unsigned int sg = (kb) / GROUP_SIZE + kp_cur; \
                cp_async_pred_16(&smem_Bs_n64[(bbuf)][kp_cur][ns], \
                    &B_scale[(unsigned long long)sg * N + gns], \
                    (gns + 15 < N)); \
            } \
        } \
    } while(0)




#if defined(__SCALE__) || defined(__HIP_PLATFORM_AMD__)
    #define N64_DEQUANT(buf) do { \
        unsigned int my_n = threadIdx.x & 63; \
        unsigned int half = threadIdx.x >> 6; \
        __nv_fp8_e4m3 fa, fb; \
        *(unsigned char*)&fa = smem_Bs_n64[(buf)][half * 2 + 0][my_n]; \
        *(unsigned char*)&fb = smem_Bs_n64[(buf)][half * 2 + 1][my_n]; \
        float sva = scl_fp8(*(const unsigned char*)&fa) * scale2, svb = scl_fp8(*(const unsigned char*)&fb) * scale2; \
        unsigned int kp0 = half * 16; \
        _Pragma("unroll") \
        for (int j = 0; j < 8; j++) { \
            unsigned int kp = kp0 + j; \
            unsigned char packed = smem_Bp_n64[(buf)][kp][my_n]; \
            float lo = smem_LUT_n64[packed & 0xF] * sva; \
            float hi = smem_LUT_n64[packed >> 4] * sva; \
            unsigned short fp8_pair; \
            fp8_pair = metrale_cvt_e4m3x2_f32(hi, lo); \
            *(unsigned short*)&smem_B_fp8_n64[my_n][kp * 2] = fp8_pair; \
        } \
        _Pragma("unroll") \
        for (int j = 8; j < 16; j++) { \
            unsigned int kp = kp0 + j; \
            unsigned char packed = smem_Bp_n64[(buf)][kp][my_n]; \
            float lo = smem_LUT_n64[packed & 0xF] * svb; \
            float hi = smem_LUT_n64[packed >> 4] * svb; \
            unsigned short fp8_pair; \
            fp8_pair = metrale_cvt_e4m3x2_f32(hi, lo); \
            *(unsigned short*)&smem_B_fp8_n64[my_n][kp * 2] = fp8_pair; \
        } \
    } while(0)

#else
    #define N64_DEQUANT(buf) do { \
        unsigned int my_n = threadIdx.x & 63; \
        unsigned int half = threadIdx.x >> 6; \
        __nv_fp8_e4m3 fa, fb; \
        *(unsigned char*)&fa = smem_Bs_n64[(buf)][half * 2 + 0][my_n]; \
        *(unsigned char*)&fb = smem_Bs_n64[(buf)][half * 2 + 1][my_n]; \
        float sva = (float)fa * scale2, svb = (float)fb * scale2; \
        unsigned int kp0 = half * 16; \
        _Pragma("unroll") \
        for (int j = 0; j < 8; j++) { \
            unsigned int kp = kp0 + j; \
            unsigned char packed = smem_Bp_n64[(buf)][kp][my_n]; \
            float lo = smem_LUT_n64[packed & 0xF] * sva; \
            float hi = smem_LUT_n64[packed >> 4] * sva; \
            unsigned short fp8_pair; \
            asm volatile("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(fp8_pair) : "f"(hi), "f"(lo)); \
            *(unsigned short*)&smem_B_fp8_n64[my_n][kp * 2] = fp8_pair; \
        } \
        _Pragma("unroll") \
        for (int j = 8; j < 16; j++) { \
            unsigned int kp = kp0 + j; \
            unsigned char packed = smem_Bp_n64[(buf)][kp][my_n]; \
            float lo = smem_LUT_n64[packed & 0xF] * svb; \
            float hi = smem_LUT_n64[packed >> 4] * svb; \
            unsigned short fp8_pair; \
            asm volatile("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(fp8_pair) : "f"(hi), "f"(lo)); \
            *(unsigned short*)&smem_B_fp8_n64[my_n][kp * 2] = fp8_pair; \
        } \
    } while(0)

#endif

    #define N64_COMPUTE_MMA(a_buf) do { \
        const unsigned short* sA = (const unsigned short*)smem_A_n64[(a_buf)]; \
        unsigned int fr0 = warp_m_offset + group_id, fr1 = fr0 + 8; \
        unsigned int a0 = bf16x4_to_e4m3x4(&sA[fr0 * ast64 + tid * 4]); \
        unsigned int a1 = bf16x4_to_e4m3x4(&sA[fr1 * ast64 + tid * 4]); \
        unsigned int a2 = bf16x4_to_e4m3x4(&sA[fr0 * ast64 + 16 + tid * 4]); \
        unsigned int a3 = bf16x4_to_e4m3x4(&sA[fr1 * ast64 + 16 + tid * 4]); \
        _Pragma("unroll") \
        for (int nt = 0; nt < 8; nt++) { \
            unsigned int nc = nt * 8 + group_id; \
            unsigned int b0 = *(const unsigned int*)&smem_B_fp8_n64[nc][4 * tid]; \
            unsigned int b1 = *(const unsigned int*)&smem_B_fp8_n64[nc][16 + 4 * tid]; \
            metrale_mma_e4m3(acc[nt], a0,a1,a2,a3, b0, b1); \
        } \
        unsigned int a4 = bf16x4_to_e4m3x4(&sA[fr0 * ast64 + 32 + tid * 4]); \
        unsigned int a5 = bf16x4_to_e4m3x4(&sA[fr1 * ast64 + 32 + tid * 4]); \
        unsigned int a6 = bf16x4_to_e4m3x4(&sA[fr0 * ast64 + 48 + tid * 4]); \
        unsigned int a7 = bf16x4_to_e4m3x4(&sA[fr1 * ast64 + 48 + tid * 4]); \
        _Pragma("unroll") \
        for (int nt = 0; nt < 8; nt++) { \
            unsigned int nc = nt * 8 + group_id; \
            unsigned int b0 = *(const unsigned int*)&smem_B_fp8_n64[nc][32 + 4 * tid]; \
            unsigned int b1 = *(const unsigned int*)&smem_B_fp8_n64[nc][48 + 4 * tid]; \
            metrale_mma_e4m3(acc[nt], a4,a5,a6,a7, b0, b1); \
        } \
    } while(0)

    const unsigned int nsteps = K / K_STEP_T64;
    N64_ISSUE_LOADS(0, 0, 0);
    cp_async_commit();
    if (nsteps > 1) { N64_ISSUE_LOADS(1, 1, K_STEP_T64); }
    cp_async_commit();
    cp_async_wait_group1();
    __syncthreads();
    N64_DEQUANT(0);
    __syncthreads();

    for (unsigned int i = 0; i + 1 < nsteps; i++) {
        N64_COMPUTE_MMA(i & 1);
        __syncthreads();
        if (i + 2 < nsteps) { N64_ISSUE_LOADS((i + 2) & 1, (i + 2) % 3, (i + 2) * K_STEP_T64); }
        cp_async_commit();
        cp_async_wait_group1();
        __syncthreads();
        N64_DEQUANT((i + 1) % 3);
        __syncthreads();
    }
    N64_COMPUTE_MMA((nsteps - 1) & 1);

    #undef N64_ISSUE_LOADS
    #undef N64_DEQUANT
    #undef N64_COMPUTE_MMA
    #undef K_STEP_T64
    #undef PAD_T64

    #pragma unroll
    for (int nt = 0; nt < 8; nt++) {
        unsigned int c0 = cta_n + nt*8 + tid*2;
        unsigned int c1 = c0 + 1;
        unsigned int r0 = cta_m + warp_m_offset + group_id;
        unsigned int r1 = r0 + 8;
        if (r0 < M && c0 < N) C[r0*N+c0] = __float2bfloat16(acc[nt][0]);
        if (r0 < M && c1 < N) C[r0*N+c1] = __float2bfloat16(acc[nt][1]);
        if (r1 < M && c0 < N) C[r1*N+c0] = __float2bfloat16(acc[nt][2]);
        if (r1 < M && c1 < N) C[r1*N+c1] = __float2bfloat16(acc[nt][3]);
    }
}

// 2026-09-25: w4a16_gemm_t with B's row stride fixed at N (no ldb) and a 128-row M tile: each CTA runs two
// 64-row chunks against one dequantized B tile. Grid (ceil(N/128), ceil(M/128)), 128 threads.











extern "C" __global__
__launch_bounds__(128, 3)
void w4a16_gemm_t_m128(
    const __nv_bfloat16* __restrict__ A,
    const unsigned char* __restrict__ B_packed,
    const unsigned char* __restrict__ B_scale,
    const float scale2,
    __nv_bfloat16* __restrict__ C,
    unsigned int M, unsigned int N, unsigned int K
) {
    const unsigned int cta_n  = blockIdx.x * N_TILE_LG;
    const unsigned int cta_m  = blockIdx.y * (2 * M_TILE);
    if (cta_m >= M) return;

    const unsigned int warp_id = threadIdx.x / 32;
    const unsigned int lane_id = threadIdx.x % 32;
    const unsigned int warp_m_offset = warp_id * 16;
    const unsigned int group_id = lane_id >> 2;
    const unsigned int tid = lane_id & 3;


    __shared__ __nv_bfloat16 smem_A[2][2 * M_TILE][K_STEP_T + PAD_T];
    __shared__ unsigned char smem_Bp[2][K_STEP_T / 2][N_TILE_LG + BP_PAD];
    __shared__ unsigned char smem_Bs[2][K_STEP_T / GROUP_SIZE][N_TILE_LG + BP_PAD];
#if defined(__SCALE__)
    __shared__ __nv_bfloat16 smem_B_bf16[N_TILE_LG][K_STEP_T];
#else
    __shared__ unsigned char smem_B_fp8[N_TILE_LG][K_STEP_T + 4];
#endif
    __shared__ float smem_LUT[16];


    if (threadIdx.x < 16) smem_LUT[threadIdx.x] = E2M1_LUT[threadIdx.x];



    float acc0[16][4], acc1[16][4];
    #pragma unroll
    for (int i = 0; i < 16; i++) {
        acc0[i][0] = 0.f; acc0[i][1] = 0.f; acc0[i][2] = 0.f; acc0[i][3] = 0.f;
        acc1[i][0] = 0.f; acc1[i][1] = 0.f; acc1[i][2] = 0.f; acc1[i][3] = 0.f;
    }

    const unsigned int a_stride = K_STEP_T + PAD_T;


    #define M128_LOADS(buf, kb) do { \
        { \
            unsigned int a_row_base = threadIdx.x >> 2; \
            unsigned int a_col      = (threadIdx.x & 3) << 3; \
            unsigned int gc = (kb) + a_col; \
            _Pragma("unroll") \
            for (int rnd = 0; rnd < 4; rnd++) { \
                unsigned int row = (unsigned int)(rnd * 32) + a_row_base; \
                unsigned int gr  = cta_m + row; \
                cp_async_pred_16(&smem_A[(buf)][row][a_col], \
                    &A[(unsigned long long)gr * K + gc], \
                    (gr < M) && (gc + 7 < K)); \
            } \
        } \
        { \
            unsigned int kp  = threadIdx.x >> 3; \
            unsigned int ns  = (threadIdx.x & 7) << 4; \
            unsigned int gke = (kb) + (kp << 1); \
            unsigned int gns = cta_n + ns; \
            cp_async_pred_16(&smem_Bp[(buf)][kp][ns], \
                &B_packed[(unsigned long long)(gke >> 1) * N + gns], \
                (gke + 1 <= K) && (gns + 15 < N)); \
            if (kp < K_STEP_T / GROUP_SIZE) { \
                unsigned int sg = (kb) / GROUP_SIZE + kp; \
                cp_async_pred_16(&smem_Bs[(buf)][kp][ns], \
                    &B_scale[(unsigned long long)sg * N + gns], \
                    (gns + 15 < N)); \
            } \
        } \
    } while(0)

#if defined(__SCALE__)

    #define M128_DEQUANT(buf) do { \
        unsigned int my_n = threadIdx.x; \
        unsigned char sb0 = smem_Bs[(buf)][0][my_n]; \
        unsigned char sb1 = smem_Bs[(buf)][1][my_n]; \
        __nv_fp8_e4m3 f0, f1; \
        *(unsigned char*)&f0 = sb0; *(unsigned char*)&f1 = sb1; \
        float sv0 = scl_fp8(*(const unsigned char*)&f0) * scale2, sv1 = scl_fp8(*(const unsigned char*)&f1) * scale2; \
        _Pragma("unroll") \
        for (int kp = 0; kp < 8; kp++) { \
            unsigned char packed = smem_Bp[(buf)][kp][my_n]; \
            smem_B_bf16[my_n][kp * 2]     = __float2bfloat16(smem_LUT[packed & 0xF] * sv0); \
            smem_B_bf16[my_n][kp * 2 + 1] = __float2bfloat16(smem_LUT[packed >> 4]  * sv0); \
        } \
        _Pragma("unroll") \
        for (int kp = 8; kp < 16; kp++) { \
            unsigned char packed = smem_Bp[(buf)][kp][my_n]; \
            smem_B_bf16[my_n][kp * 2]     = __float2bfloat16(smem_LUT[packed & 0xF] * sv1); \
            smem_B_bf16[my_n][kp * 2 + 1] = __float2bfloat16(smem_LUT[packed >> 4]  * sv1); \
        } \
    } while(0)


    #define M128_COMPUTE(a_buf) do { \
        const __nv_bfloat16* sA = (const __nv_bfloat16*)smem_A[(a_buf)]; \
        _Pragma("unroll") \
        for (int ch = 0; ch < 2; ch++) { \
            unsigned int fr0 = ch * M_TILE + warp_m_offset + group_id; \
            unsigned int fr1 = fr0 + 8; \
            _Pragma("unroll") \
            for (int h = 0; h < 2; h++) { \
                unsigned int fc0 = h * 16 + tid * 2, fc1 = fc0 + 8; \
                unsigned int a0 = *(const unsigned int*)&sA[fr0 * a_stride + fc0]; \
                unsigned int a1 = *(const unsigned int*)&sA[fr1 * a_stride + fc0]; \
                unsigned int a2 = *(const unsigned int*)&sA[fr0 * a_stride + fc1]; \
                unsigned int a3 = *(const unsigned int*)&sA[fr1 * a_stride + fc1]; \
                _Pragma("unroll") \
                for (int nt = 0; nt < 16; nt++) { \
                    unsigned int nc = nt * 8 + group_id; \
                    const __nv_bfloat16* sb = &smem_B_bf16[nc][0]; \
                    unsigned int b0 = *(const unsigned int*)&sb[fc0]; \
                    unsigned int b1 = *(const unsigned int*)&sb[fc1]; \
                    float* acc = ch ? acc1[nt] : acc0[nt]; \
                    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 " \
                        "{%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%10,%11,%12,%13};" \
                        : "=f"(acc[0]), "=f"(acc[1]), "=f"(acc[2]), "=f"(acc[3]) \
                        : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1), \
                          "f"(acc[0]), "f"(acc[1]), "f"(acc[2]), "f"(acc[3])); \
                } \
            } \
        } \
    } while(0)
#else

    #define M128_DEQUANT(buf) do { \
        unsigned int my_n = threadIdx.x; \
        unsigned char sb0 = smem_Bs[(buf)][0][my_n]; \
        unsigned char sb1 = smem_Bs[(buf)][1][my_n]; \
        __nv_fp8_e4m3 f0, f1; \
        *(unsigned char*)&f0 = sb0; *(unsigned char*)&f1 = sb1; \
        float sv0 = (float)f0 * scale2, sv1 = (float)f1 * scale2; \
        _Pragma("unroll") \
        for (int kp = 0; kp < 8; kp++) { \
            unsigned char packed = smem_Bp[(buf)][kp][my_n]; \
            float lo = smem_LUT[packed & 0xF] * sv0; \
            float hi = smem_LUT[packed >> 4]  * sv0; \
            unsigned short fp8_pair; \
            asm volatile("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" \
                         : "=h"(fp8_pair) : "f"(hi), "f"(lo)); \
            *(unsigned short*)&smem_B_fp8[my_n][kp * 2] = fp8_pair; \
        } \
        _Pragma("unroll") \
        for (int kp = 8; kp < 16; kp++) { \
            unsigned char packed = smem_Bp[(buf)][kp][my_n]; \
            float lo = smem_LUT[packed & 0xF] * sv1; \
            float hi = smem_LUT[packed >> 4]  * sv1; \
            unsigned short fp8_pair; \
            asm volatile("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" \
                         : "=h"(fp8_pair) : "f"(hi), "f"(lo)); \
            *(unsigned short*)&smem_B_fp8[my_n][kp * 2] = fp8_pair; \
        } \
    } while(0)


    #define M128_COMPUTE(a_buf) do { \
        const unsigned short* sA = (const unsigned short*)smem_A[(a_buf)]; \
        unsigned int fr0, fr1, a0, a1, a2, a3; \
        \
        fr0 = warp_m_offset + group_id; \
        fr1 = fr0 + 8; \
        a0 = bf16x4_to_e4m3x4(&sA[fr0 * a_stride + tid * 4]); \
        a1 = bf16x4_to_e4m3x4(&sA[fr1 * a_stride + tid * 4]); \
        a2 = bf16x4_to_e4m3x4(&sA[fr0 * a_stride + 16 + tid * 4]); \
        a3 = bf16x4_to_e4m3x4(&sA[fr1 * a_stride + 16 + tid * 4]); \
        _Pragma("unroll") \
        for (int nt = 0; nt < 16; nt++) { \
            unsigned int nc = nt * 8 + group_id; \
            unsigned int b0 = *(const unsigned int*)&smem_B_fp8[nc][4 * tid]; \
            unsigned int b1 = *(const unsigned int*)&smem_B_fp8[nc][16 + 4 * tid]; \
            asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 " \
                "{%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%10,%11,%12,%13};" \
                :"=f"(acc0[nt][0]),"=f"(acc0[nt][1]),"=f"(acc0[nt][2]),"=f"(acc0[nt][3]) \
                :"r"(a0),"r"(a1),"r"(a2),"r"(a3),"r"(b0),"r"(b1), \
                 "f"(acc0[nt][0]),"f"(acc0[nt][1]),"f"(acc0[nt][2]),"f"(acc0[nt][3])); \
        } \
        \
        fr0 = M_TILE + warp_m_offset + group_id; \
        fr1 = fr0 + 8; \
        a0 = bf16x4_to_e4m3x4(&sA[fr0 * a_stride + tid * 4]); \
        a1 = bf16x4_to_e4m3x4(&sA[fr1 * a_stride + tid * 4]); \
        a2 = bf16x4_to_e4m3x4(&sA[fr0 * a_stride + 16 + tid * 4]); \
        a3 = bf16x4_to_e4m3x4(&sA[fr1 * a_stride + 16 + tid * 4]); \
        _Pragma("unroll") \
        for (int nt = 0; nt < 16; nt++) { \
            unsigned int nc = nt * 8 + group_id; \
            unsigned int b0 = *(const unsigned int*)&smem_B_fp8[nc][4 * tid]; \
            unsigned int b1 = *(const unsigned int*)&smem_B_fp8[nc][16 + 4 * tid]; \
            asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 " \
                "{%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%10,%11,%12,%13};" \
                :"=f"(acc1[nt][0]),"=f"(acc1[nt][1]),"=f"(acc1[nt][2]),"=f"(acc1[nt][3]) \
                :"r"(a0),"r"(a1),"r"(a2),"r"(a3),"r"(b0),"r"(b1), \
                 "f"(acc1[nt][0]),"f"(acc1[nt][1]),"f"(acc1[nt][2]),"f"(acc1[nt][3])); \
        } \
    } while(0)
#endif


    M128_LOADS(0, 0);
    cp_async_commit();
    cp_async_wait_all();
    __syncthreads();
    M128_DEQUANT(0);
    __syncthreads();

    int cur = 0;
    for (unsigned int k_base = K_STEP_T; k_base < K; k_base += K_STEP_T) {
        int nxt = 1 - cur;
        M128_LOADS(nxt, k_base);
        cp_async_commit();
        M128_COMPUTE(cur);
        cp_async_wait_all();
        __syncthreads();
        M128_DEQUANT(nxt);
        __syncthreads();
        cur = nxt;
    }
    M128_COMPUTE(cur);

    #undef M128_LOADS
    #undef M128_DEQUANT
    #undef M128_COMPUTE


    #pragma unroll
    for (int nt = 0; nt < 16; nt++) {
        unsigned int c0 = cta_n + nt * 8 + tid * 2;
        unsigned int c1 = c0 + 1;
        unsigned int r0 = cta_m + warp_m_offset + group_id;
        unsigned int r1 = r0 + 8;
        if (r0 < M && c0 < N) C[r0 * N + c0] = __float2bfloat16(acc0[nt][0]);
        if (r0 < M && c1 < N) C[r0 * N + c1] = __float2bfloat16(acc0[nt][1]);
        if (r1 < M && c0 < N) C[r1 * N + c0] = __float2bfloat16(acc0[nt][2]);
        if (r1 < M && c1 < N) C[r1 * N + c1] = __float2bfloat16(acc0[nt][3]);
    }

    #pragma unroll
    for (int nt = 0; nt < 16; nt++) {
        unsigned int c0 = cta_n + nt * 8 + tid * 2;
        unsigned int c1 = c0 + 1;
        unsigned int r0 = cta_m + M_TILE + warp_m_offset + group_id;
        unsigned int r1 = r0 + 8;
        if (r0 < M && c0 < N) C[r0 * N + c0] = __float2bfloat16(acc1[nt][0]);
        if (r0 < M && c1 < N) C[r0 * N + c1] = __float2bfloat16(acc1[nt][1]);
        if (r1 < M && c0 < N) C[r1 * N + c0] = __float2bfloat16(acc1[nt][2]);
        if (r1 < M && c1 < N) C[r1 * N + c1] = __float2bfloat16(acc1[nt][3]);
    }
}

// 2026-09-25: w4a16_gemm_t_m128 with W dequantized to BF16 and multiplied in BF16 m16n8k16 MMAs on every build,
// so neither W nor A is rounded to E4M3. The block scale is decoded as (float)__nv_fp8_e4m3 on every build.
// Grid (ceil(N/128), ceil(M/128)), 128 threads.





















extern "C" __global__
__launch_bounds__(128, 3)
void w4a16_gemm_t_m128_bf16(
    const __nv_bfloat16* __restrict__ A,
    const unsigned char* __restrict__ B_packed,
    const unsigned char* __restrict__ B_scale,
    const float scale2,
    __nv_bfloat16* __restrict__ C,
    unsigned int M, unsigned int N, unsigned int K
) {
    const unsigned int cta_n  = blockIdx.x * N_TILE_LG;
    const unsigned int cta_m  = blockIdx.y * (2 * M_TILE);
    if (cta_m >= M) return;

    const unsigned int warp_id = threadIdx.x / 32;
    const unsigned int lane_id = threadIdx.x % 32;
    const unsigned int warp_m_offset = warp_id * 16;
    const unsigned int group_id = lane_id >> 2;
    const unsigned int tid = lane_id & 3;


    __shared__ __nv_bfloat16 smem_A[2][2 * M_TILE][K_STEP_T + PAD_T];
    __shared__ unsigned char smem_Bp[2][K_STEP_T / 2][N_TILE_LG + BP_PAD];
    __shared__ unsigned char smem_Bs[2][K_STEP_T / GROUP_SIZE][N_TILE_LG + BP_PAD];
    __shared__ __nv_bfloat16 smem_B_bf16[N_TILE_LG][K_STEP_T];
    __shared__ float smem_LUT[16];

    if (threadIdx.x < 16) smem_LUT[threadIdx.x] = E2M1_LUT[threadIdx.x];



    float acc0[16][4], acc1[16][4];
    #pragma unroll
    for (int i = 0; i < 16; i++) {
        acc0[i][0] = 0.f; acc0[i][1] = 0.f; acc0[i][2] = 0.f; acc0[i][3] = 0.f;
        acc1[i][0] = 0.f; acc1[i][1] = 0.f; acc1[i][2] = 0.f; acc1[i][3] = 0.f;
    }

    const unsigned int a_stride = K_STEP_T + PAD_T;


    #define M128B_LOADS(buf, kb) do { \
        { \
            unsigned int a_row_base = threadIdx.x >> 2; \
            unsigned int a_col      = (threadIdx.x & 3) << 3; \
            unsigned int gc = (kb) + a_col; \
            _Pragma("unroll") \
            for (int rnd = 0; rnd < 4; rnd++) { \
                unsigned int row = (unsigned int)(rnd * 32) + a_row_base; \
                unsigned int gr  = cta_m + row; \
                cp_async_pred_16(&smem_A[(buf)][row][a_col], \
                    &A[(unsigned long long)gr * K + gc], \
                    (gr < M) && (gc + 7 < K)); \
            } \
        } \
        { \
            unsigned int kp  = threadIdx.x >> 3; \
            unsigned int ns  = (threadIdx.x & 7) << 4; \
            unsigned int gke = (kb) + (kp << 1); \
            unsigned int gns = cta_n + ns; \
            cp_async_pred_16(&smem_Bp[(buf)][kp][ns], \
                &B_packed[(unsigned long long)(gke >> 1) * N + gns], \
                (gke + 1 <= K) && (gns + 15 < N)); \
            if (kp < K_STEP_T / GROUP_SIZE) { \
                unsigned int sg = (kb) / GROUP_SIZE + kp; \
                cp_async_pred_16(&smem_Bs[(buf)][kp][ns], \
                    &B_scale[(unsigned long long)sg * N + gns], \
                    (gns + 15 < N)); \
            } \
        } \
    } while(0)




    #define M128B_DEQUANT(buf) do { \
        unsigned int my_n = threadIdx.x; \
        unsigned char sb0 = smem_Bs[(buf)][0][my_n]; \
        unsigned char sb1 = smem_Bs[(buf)][1][my_n]; \
        __nv_fp8_e4m3 f0, f1; \
        *(unsigned char*)&f0 = sb0; *(unsigned char*)&f1 = sb1; \
        float sv0 = (float)f0 * scale2, sv1 = (float)f1 * scale2; \
        _Pragma("unroll") \
        for (int kp = 0; kp < 8; kp++) { \
            unsigned char packed = smem_Bp[(buf)][kp][my_n]; \
            smem_B_bf16[my_n][kp * 2]     = __float2bfloat16(smem_LUT[packed & 0xF] * sv0); \
            smem_B_bf16[my_n][kp * 2 + 1] = __float2bfloat16(smem_LUT[packed >> 4]  * sv0); \
        } \
        _Pragma("unroll") \
        for (int kp = 8; kp < 16; kp++) { \
            unsigned char packed = smem_Bp[(buf)][kp][my_n]; \
            smem_B_bf16[my_n][kp * 2]     = __float2bfloat16(smem_LUT[packed & 0xF] * sv1); \
            smem_B_bf16[my_n][kp * 2 + 1] = __float2bfloat16(smem_LUT[packed >> 4]  * sv1); \
        } \
    } while(0)



    #define M128B_COMPUTE(a_buf) do { \
        const __nv_bfloat16* sA = (const __nv_bfloat16*)smem_A[(a_buf)]; \
        _Pragma("unroll") \
        for (int ch = 0; ch < 2; ch++) { \
            unsigned int fr0 = ch * M_TILE + warp_m_offset + group_id; \
            unsigned int fr1 = fr0 + 8; \
            _Pragma("unroll") \
            for (int h = 0; h < 2; h++) { \
                unsigned int fc0 = h * 16 + tid * 2, fc1 = fc0 + 8; \
                unsigned int a0 = *(const unsigned int*)&sA[fr0 * a_stride + fc0]; \
                unsigned int a1 = *(const unsigned int*)&sA[fr1 * a_stride + fc0]; \
                unsigned int a2 = *(const unsigned int*)&sA[fr0 * a_stride + fc1]; \
                unsigned int a3 = *(const unsigned int*)&sA[fr1 * a_stride + fc1]; \
                _Pragma("unroll") \
                for (int nt = 0; nt < 16; nt++) { \
                    unsigned int nc = nt * 8 + group_id; \
                    const __nv_bfloat16* sb = &smem_B_bf16[nc][0]; \
                    unsigned int b0 = *(const unsigned int*)&sb[fc0]; \
                    unsigned int b1 = *(const unsigned int*)&sb[fc1]; \
                    float* acc = ch ? acc1[nt] : acc0[nt]; \
                    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 " \
                        "{%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%10,%11,%12,%13};" \
                        : "=f"(acc[0]), "=f"(acc[1]), "=f"(acc[2]), "=f"(acc[3]) \
                        : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1), \
                          "f"(acc[0]), "f"(acc[1]), "f"(acc[2]), "f"(acc[3])); \
                } \
            } \
        } \
    } while(0)


    M128B_LOADS(0, 0);
    cp_async_commit();
    cp_async_wait_all();
    __syncthreads();
    M128B_DEQUANT(0);
    __syncthreads();

    int cur = 0;
    for (unsigned int k_base = K_STEP_T; k_base < K; k_base += K_STEP_T) {
        int nxt = 1 - cur;
        M128B_LOADS(nxt, k_base);
        cp_async_commit();
        M128B_COMPUTE(cur);
        cp_async_wait_all();
        __syncthreads();
        M128B_DEQUANT(nxt);
        __syncthreads();
        cur = nxt;
    }
    M128B_COMPUTE(cur);

    #undef M128B_LOADS
    #undef M128B_DEQUANT
    #undef M128B_COMPUTE


    #pragma unroll
    for (int nt = 0; nt < 16; nt++) {
        unsigned int c0 = cta_n + nt * 8 + tid * 2;
        unsigned int c1 = c0 + 1;
        unsigned int r0 = cta_m + warp_m_offset + group_id;
        unsigned int r1 = r0 + 8;
        if (r0 < M && c0 < N) C[r0 * N + c0] = __float2bfloat16(acc0[nt][0]);
        if (r0 < M && c1 < N) C[r0 * N + c1] = __float2bfloat16(acc0[nt][1]);
        if (r1 < M && c0 < N) C[r1 * N + c0] = __float2bfloat16(acc0[nt][2]);
        if (r1 < M && c1 < N) C[r1 * N + c1] = __float2bfloat16(acc0[nt][3]);
    }

    #pragma unroll
    for (int nt = 0; nt < 16; nt++) {
        unsigned int c0 = cta_n + nt * 8 + tid * 2;
        unsigned int c1 = c0 + 1;
        unsigned int r0 = cta_m + M_TILE + warp_m_offset + group_id;
        unsigned int r1 = r0 + 8;
        if (r0 < M && c0 < N) C[r0 * N + c0] = __float2bfloat16(acc1[nt][0]);
        if (r0 < M && c1 < N) C[r0 * N + c1] = __float2bfloat16(acc1[nt][1]);
        if (r1 < M && c0 < N) C[r1 * N + c0] = __float2bfloat16(acc1[nt][2]);
        if (r1 < M && c1 < N) C[r1 * N + c1] = __float2bfloat16(acc1[nt][3]);
    }
}

// 2026-09-25: w4a16_gemm_t_m128_bf16 with an ldb argument (as in w4a16_gemm_t) and a smaller shared-memory
// layout: A rows have no pad and are XOR-swizzled (A_SWZ_V2), and smem_B_bf16 rows get PADB_V2 = 2 BF16 of
// pad, making them 17 words, so a warp's row-wise dequant stores hit 32 distinct banks. Static shared
// memory is 30,336 bytes. The values each MMA reads, the MMA order and the store are those of
// w4a16_gemm_t_m128_bf16, so for ldb = N the output is identical.










































#define M128B_STAGES 2
#define PAD_T_V2 0
#define PADB_V2 2
// 2026-09-25: A-tile XOR swizzle at 16-byte (8 BF16) granularity: physical chunk = logical chunk ^ ((row >> 1) & 3).
// cp.async moves whole 16-byte chunks, so permuting chunks within a row keeps it legal, and a warp's A-fragment
// reads hit 32 distinct banks instead of 8. The value each lane reads is unchanged.


#define A_SWZ_V2(col, row) ((((((col) >> 3) ^ (((row) >> 1) & 3u)) << 3) | ((col) & 7u)))
extern "C" __global__
__launch_bounds__(128, 3)
void w4a16_gemm_t_m128_bf16_v2(
    const __nv_bfloat16* __restrict__ A,
    const unsigned char* __restrict__ B_packed,
    const unsigned char* __restrict__ B_scale,
    const float scale2,
    __nv_bfloat16* __restrict__ C,
    unsigned int M, unsigned int N, unsigned int K,
    unsigned int ldb          // 2026-09-25: transposed-B row stride; may exceed N (see w4a16_gemm_t)
) {
    const unsigned int LDB = ldb;
    const unsigned int cta_n  = blockIdx.x * N_TILE_LG;
    const unsigned int cta_m  = blockIdx.y * (2 * M_TILE);
    if (cta_m >= M) return;

    const unsigned int warp_id = threadIdx.x / 32;
    const unsigned int lane_id = threadIdx.x % 32;
    const unsigned int warp_m_offset = warp_id * 16;
    const unsigned int group_id = lane_id >> 2;
    const unsigned int tid = lane_id & 3;




    __shared__ __nv_bfloat16 smem_A[M128B_STAGES][2 * M_TILE][K_STEP_T + PAD_T_V2];
    __shared__ unsigned char smem_Bp[M128B_STAGES][K_STEP_T / 2][N_TILE_LG + BP_PAD];
    __shared__ unsigned char smem_Bs[M128B_STAGES][K_STEP_T / GROUP_SIZE][N_TILE_LG + BP_PAD];
    __shared__ __nv_bfloat16 smem_B_bf16[N_TILE_LG][K_STEP_T + PADB_V2];
    __shared__ float smem_LUT[16];

    if (threadIdx.x < 16) smem_LUT[threadIdx.x] = E2M1_LUT[threadIdx.x];

    float acc0[16][4], acc1[16][4];
    #pragma unroll
    for (int i = 0; i < 16; i++) {
        acc0[i][0] = 0.f; acc0[i][1] = 0.f; acc0[i][2] = 0.f; acc0[i][3] = 0.f;
        acc1[i][0] = 0.f; acc1[i][1] = 0.f; acc1[i][2] = 0.f; acc1[i][3] = 0.f;
    }

    const unsigned int a_stride = K_STEP_T + PAD_T_V2;



    #define V2_LOADS(buf, kb) do { \
        { \
            unsigned int a_row_base = threadIdx.x >> 2; \
            unsigned int a_col      = (threadIdx.x & 3) << 3; \
            unsigned int gc = (kb) + a_col; \
            _Pragma("unroll") \
            for (int rnd = 0; rnd < 4; rnd++) { \
                unsigned int row = (unsigned int)(rnd * 32) + a_row_base; \
                unsigned int gr  = cta_m + row; \
                cp_async_pred_16(&smem_A[(buf)][row][A_SWZ_V2(a_col, row)], \
                    &A[(unsigned long long)gr * K + gc], \
                    (gr < M) && (gc + 7 < K)); \
            } \
        } \
        { \
            unsigned int kp  = threadIdx.x >> 3; \
            unsigned int ns  = (threadIdx.x & 7) << 4; \
            unsigned int gke = (kb) + (kp << 1); \
            unsigned int gns = cta_n + ns; \
            cp_async_pred_16(&smem_Bp[(buf)][kp][ns], \
                &B_packed[(unsigned long long)(gke >> 1) * LDB + gns], \
                (gke + 1 <= K) && (gns + 15 < LDB)); \
            if (kp < K_STEP_T / GROUP_SIZE) { \
                unsigned int sg = (kb) / GROUP_SIZE + kp; \
                cp_async_pred_16(&smem_Bs[(buf)][kp][ns], \
                    &B_scale[(unsigned long long)sg * LDB + gns], \
                    (gns + 15 < LDB)); \
            } \
        } \
    } while(0)



    #define V2_DEQUANT(buf) do { \
        unsigned int my_n = threadIdx.x; \
        unsigned char sb0 = smem_Bs[(buf)][0][my_n]; \
        unsigned char sb1 = smem_Bs[(buf)][1][my_n]; \
        __nv_fp8_e4m3 f0, f1; \
        *(unsigned char*)&f0 = sb0; *(unsigned char*)&f1 = sb1; \
        float sv0 = (float)f0 * scale2, sv1 = (float)f1 * scale2; \
        _Pragma("unroll") \
        for (int kp = 0; kp < 8; kp++) { \
            unsigned char packed = smem_Bp[(buf)][kp][my_n]; \
            smem_B_bf16[my_n][kp * 2]     = __float2bfloat16(smem_LUT[packed & 0xF] * sv0); \
            smem_B_bf16[my_n][kp * 2 + 1] = __float2bfloat16(smem_LUT[packed >> 4]  * sv0); \
        } \
        _Pragma("unroll") \
        for (int kp = 8; kp < 16; kp++) { \
            unsigned char packed = smem_Bp[(buf)][kp][my_n]; \
            smem_B_bf16[my_n][kp * 2]     = __float2bfloat16(smem_LUT[packed & 0xF] * sv1); \
            smem_B_bf16[my_n][kp * 2 + 1] = __float2bfloat16(smem_LUT[packed >> 4]  * sv1); \
        } \
    } while(0)



    #define V2_COMPUTE(a_buf) do { \
        const __nv_bfloat16* sA = (const __nv_bfloat16*)smem_A[(a_buf)]; \
        _Pragma("unroll") \
        for (int ch = 0; ch < 2; ch++) { \
            unsigned int fr0 = ch * M_TILE + warp_m_offset + group_id; \
            unsigned int fr1 = fr0 + 8; \
            _Pragma("unroll") \
            for (int hh = 0; hh < 2; hh++) { \
                unsigned int fc0 = hh * 16 + tid * 2, fc1 = fc0 + 8; \
                unsigned int a0 = *(const unsigned int*)&sA[fr0 * a_stride + A_SWZ_V2(fc0, fr0)]; \
                unsigned int a1 = *(const unsigned int*)&sA[fr1 * a_stride + A_SWZ_V2(fc0, fr1)]; \
                unsigned int a2 = *(const unsigned int*)&sA[fr0 * a_stride + A_SWZ_V2(fc1, fr0)]; \
                unsigned int a3 = *(const unsigned int*)&sA[fr1 * a_stride + A_SWZ_V2(fc1, fr1)]; \
                _Pragma("unroll") \
                for (int nt = 0; nt < 16; nt++) { \
                    unsigned int nc = nt * 8 + group_id; \
                    const __nv_bfloat16* sb = &smem_B_bf16[nc][0]; \
                    unsigned int b0 = *(const unsigned int*)&sb[fc0]; \
                    unsigned int b1 = *(const unsigned int*)&sb[fc1]; \
                    float* acc = ch ? acc1[nt] : acc0[nt]; \
                    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 " \
                        "{%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%10,%11,%12,%13};" \
                        : "=f"(acc[0]), "=f"(acc[1]), "=f"(acc[2]), "=f"(acc[3]) \
                        : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1), \
                          "f"(acc[0]), "f"(acc[1]), "f"(acc[2]), "f"(acc[3])); \
                } \
            } \
        } \
    } while(0)




    V2_LOADS(0, 0);
    cp_async_commit();
    cp_async_wait_all();
    __syncthreads();
    V2_DEQUANT(0);
    __syncthreads();

    int cur = 0;
    for (unsigned int k_base = K_STEP_T; k_base < K; k_base += K_STEP_T) {
        int nxt = 1 - cur;
        V2_LOADS(nxt, k_base);
        cp_async_commit();
        V2_COMPUTE(cur);
        cp_async_wait_all();
        __syncthreads();
        V2_DEQUANT(nxt);
        __syncthreads();
        cur = nxt;
    }
    V2_COMPUTE(cur);

    #undef V2_LOADS
    #undef V2_DEQUANT
    #undef V2_COMPUTE


    #pragma unroll
    for (int nt = 0; nt < 16; nt++) {
        unsigned int c0 = cta_n + nt * 8 + tid * 2;
        unsigned int c1 = c0 + 1;
        unsigned int r0 = cta_m + warp_m_offset + group_id;
        unsigned int r1 = r0 + 8;
        if (r0 < M && c0 < N) C[r0 * N + c0] = __float2bfloat16(acc0[nt][0]);
        if (r0 < M && c1 < N) C[r0 * N + c1] = __float2bfloat16(acc0[nt][1]);
        if (r1 < M && c0 < N) C[r1 * N + c0] = __float2bfloat16(acc0[nt][2]);
        if (r1 < M && c1 < N) C[r1 * N + c1] = __float2bfloat16(acc0[nt][3]);
    }

    #pragma unroll
    for (int nt = 0; nt < 16; nt++) {
        unsigned int c0 = cta_n + nt * 8 + tid * 2;
        unsigned int c1 = c0 + 1;
        unsigned int r0 = cta_m + M_TILE + warp_m_offset + group_id;
        unsigned int r1 = r0 + 8;
        if (r0 < M && c0 < N) C[r0 * N + c0] = __float2bfloat16(acc1[nt][0]);
        if (r0 < M && c1 < N) C[r0 * N + c1] = __float2bfloat16(acc1[nt][1]);
        if (r1 < M && c0 < N) C[r1 * N + c0] = __float2bfloat16(acc1[nt][2]);
        if (r1 < M && c1 < N) C[r1 * N + c1] = __float2bfloat16(acc1[nt][3]);
    }
}
#undef M128B_STAGES

// 2026-09-25: w4a16_gemm_t_m128_bf16 with one 64-row M chunk per CTA and unpadded A rows (M64B_PAD = 0); B's
// row stride is N. Each output element gets the same dequantized values, MMA sequence and order as chunk 0 of
// w4a16_gemm_t_m128_bf16. Grid (ceil(N/128), ceil(M/64)), 128 threads.



















#define M64B_PAD 0
extern "C" __global__
__launch_bounds__(128, 4)
void w4a16_gemm_t_m64_bf16(
    const __nv_bfloat16* __restrict__ A,
    const unsigned char* __restrict__ B_packed,
    const unsigned char* __restrict__ B_scale,
    const float scale2,
    __nv_bfloat16* __restrict__ C,
    unsigned int M, unsigned int N, unsigned int K
) {
    const unsigned int cta_n  = blockIdx.x * N_TILE_LG;
    const unsigned int cta_m  = blockIdx.y * M_TILE;
    if (cta_m >= M) return;

    const unsigned int warp_id = threadIdx.x / 32;
    const unsigned int lane_id = threadIdx.x % 32;
    const unsigned int warp_m_offset = warp_id * 16;
    const unsigned int group_id = lane_id >> 2;
    const unsigned int tid = lane_id & 3;

    __shared__ __nv_bfloat16 smem_A[2][M_TILE][K_STEP_T + M64B_PAD];
    __shared__ unsigned char smem_Bp[2][K_STEP_T / 2][N_TILE_LG + BP_PAD];
    __shared__ unsigned char smem_Bs[2][K_STEP_T / GROUP_SIZE][N_TILE_LG + BP_PAD];
    __shared__ __nv_bfloat16 smem_B_bf16[N_TILE_LG][K_STEP_T];
    __shared__ float smem_LUT[16];

    if (threadIdx.x < 16) smem_LUT[threadIdx.x] = E2M1_LUT[threadIdx.x];

    float acc[16][4];
    #pragma unroll
    for (int i = 0; i < 16; i++) { acc[i][0]=0.f; acc[i][1]=0.f; acc[i][2]=0.f; acc[i][3]=0.f; }

    const unsigned int a_stride = K_STEP_T + M64B_PAD;

    #define M64_LOADS(buf, kb) do { \
        { \
            unsigned int a_row_base = threadIdx.x >> 2; \
            unsigned int a_col      = (threadIdx.x & 3) << 3; \
            unsigned int gc = (kb) + a_col; \
            _Pragma("unroll") \
            for (int rnd = 0; rnd < 2; rnd++) { \
                unsigned int row = (unsigned int)(rnd * 32) + a_row_base; \
                unsigned int gr  = cta_m + row; \
                cp_async_pred_16(&smem_A[(buf)][row][a_col], \
                    &A[(unsigned long long)gr * K + gc], \
                    (gr < M) && (gc + 7 < K)); \
            } \
        } \
        { \
            unsigned int kp  = threadIdx.x >> 3; \
            unsigned int ns  = (threadIdx.x & 7) << 4; \
            unsigned int gke = (kb) + (kp << 1); \
            unsigned int gns = cta_n + ns; \
            cp_async_pred_16(&smem_Bp[(buf)][kp][ns], \
                &B_packed[(unsigned long long)(gke >> 1) * N + gns], \
                (gke + 1 <= K) && (gns + 15 < N)); \
            if (kp < K_STEP_T / GROUP_SIZE) { \
                unsigned int sg = (kb) / GROUP_SIZE + kp; \
                cp_async_pred_16(&smem_Bs[(buf)][kp][ns], \
                    &B_scale[(unsigned long long)sg * N + gns], \
                    (gns + 15 < N)); \
            } \
        } \
    } while(0)

    #define M64_DEQUANT(buf) do { \
        unsigned int my_n = threadIdx.x; \
        unsigned char sb0 = smem_Bs[(buf)][0][my_n]; \
        unsigned char sb1 = smem_Bs[(buf)][1][my_n]; \
        __nv_fp8_e4m3 f0, f1; \
        *(unsigned char*)&f0 = sb0; *(unsigned char*)&f1 = sb1; \
        float sv0 = (float)f0 * scale2, sv1 = (float)f1 * scale2; \
        _Pragma("unroll") \
        for (int kp = 0; kp < 8; kp++) { \
            unsigned char packed = smem_Bp[(buf)][kp][my_n]; \
            smem_B_bf16[my_n][kp * 2]     = __float2bfloat16(smem_LUT[packed & 0xF] * sv0); \
            smem_B_bf16[my_n][kp * 2 + 1] = __float2bfloat16(smem_LUT[packed >> 4]  * sv0); \
        } \
        _Pragma("unroll") \
        for (int kp = 8; kp < 16; kp++) { \
            unsigned char packed = smem_Bp[(buf)][kp][my_n]; \
            smem_B_bf16[my_n][kp * 2]     = __float2bfloat16(smem_LUT[packed & 0xF] * sv1); \
            smem_B_bf16[my_n][kp * 2 + 1] = __float2bfloat16(smem_LUT[packed >> 4]  * sv1); \
        } \
    } while(0)

    #define M64_COMPUTE(a_buf) do { \
        const __nv_bfloat16* sA = (const __nv_bfloat16*)smem_A[(a_buf)]; \
        unsigned int fr0 = warp_m_offset + group_id; \
        unsigned int fr1 = fr0 + 8; \
        _Pragma("unroll") \
        for (int h = 0; h < 2; h++) { \
            unsigned int fc0 = h * 16 + tid * 2, fc1 = fc0 + 8; \
            unsigned int a0 = *(const unsigned int*)&sA[fr0 * a_stride + fc0]; \
            unsigned int a1 = *(const unsigned int*)&sA[fr1 * a_stride + fc0]; \
            unsigned int a2 = *(const unsigned int*)&sA[fr0 * a_stride + fc1]; \
            unsigned int a3 = *(const unsigned int*)&sA[fr1 * a_stride + fc1]; \
            _Pragma("unroll") \
            for (int nt = 0; nt < 16; nt++) { \
                unsigned int nc = nt * 8 + group_id; \
                const __nv_bfloat16* sb = &smem_B_bf16[nc][0]; \
                unsigned int b0 = *(const unsigned int*)&sb[fc0]; \
                unsigned int b1 = *(const unsigned int*)&sb[fc1]; \
                float* ac = acc[nt]; \
                asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 " \
                    "{%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%10,%11,%12,%13};" \
                    : "=f"(ac[0]), "=f"(ac[1]), "=f"(ac[2]), "=f"(ac[3]) \
                    : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1), \
                      "f"(ac[0]), "f"(ac[1]), "f"(ac[2]), "f"(ac[3])); \
            } \
        } \
    } while(0)

    M64_LOADS(0, 0);
    cp_async_commit();
    cp_async_wait_all();
    __syncthreads();
    M64_DEQUANT(0);
    __syncthreads();

    int cur = 0;
    for (unsigned int k_base = K_STEP_T; k_base < K; k_base += K_STEP_T) {
        int nxt = 1 - cur;
        M64_LOADS(nxt, k_base);
        cp_async_commit();
        M64_COMPUTE(cur);
        cp_async_wait_all();
        __syncthreads();
        M64_DEQUANT(nxt);
        __syncthreads();
        cur = nxt;
    }
    M64_COMPUTE(cur);

    #undef M64_LOADS
    #undef M64_DEQUANT
    #undef M64_COMPUTE

    #pragma unroll
    for (int nt = 0; nt < 16; nt++) {
        unsigned int c0 = cta_n + nt * 8 + tid * 2;
        unsigned int c1 = c0 + 1;
        unsigned int r0 = cta_m + warp_m_offset + group_id;
        unsigned int r1 = r0 + 8;
        if (r0 < M && c0 < N) C[r0 * N + c0] = __float2bfloat16(acc[nt][0]);
        if (r0 < M && c1 < N) C[r0 * N + c1] = __float2bfloat16(acc[nt][1]);
        if (r1 < M && c0 < N) C[r1 * N + c0] = __float2bfloat16(acc[nt][2]);
        if (r1 < M && c1 < N) C[r1 * N + c1] = __float2bfloat16(acc[nt][3]);
    }
}
#undef M64B_PAD

// 2026-09-25: fp8_gemm_t with a 128-row M tile: each CTA runs two 64-row chunks against one E4M3 B tile.
// Grid (ceil(N/128), ceil(M/128)), 128 threads.







extern "C" __global__
__launch_bounds__(128, 3)
void fp8_gemm_t_m128(
    const __nv_bfloat16* __restrict__ A,
    const unsigned char* __restrict__ B_fp8,
    __nv_bfloat16* __restrict__ C,
    unsigned int M, unsigned int N, unsigned int K
) {
    const unsigned int cta_n = blockIdx.x * N_TILE_LG;
    const unsigned int cta_m = blockIdx.y * (2 * M_TILE);
    if (cta_m >= M) return;

    const unsigned int warp_id = threadIdx.x / 32;
    const unsigned int lane_id = threadIdx.x % 32;
    const unsigned int warp_m_offset = warp_id * 16;
    const unsigned int group_id = lane_id >> 2;
    const unsigned int tid = lane_id & 3;

    __shared__ __nv_bfloat16 smem_A[2][2 * M_TILE][K_STEP_T + PAD_T];
    __shared__ unsigned char  smem_B[2][N_TILE_LG][K_STEP_T];

    float acc0[16][4], acc1[16][4];
    #pragma unroll
    for (int i = 0; i < 16; i++) {
        acc0[i][0] = 0.f; acc0[i][1] = 0.f; acc0[i][2] = 0.f; acc0[i][3] = 0.f;
        acc1[i][0] = 0.f; acc1[i][1] = 0.f; acc1[i][2] = 0.f; acc1[i][3] = 0.f;
    }

    const unsigned int a_stride = K_STEP_T + PAD_T;


    #define FGM128_LOADS(buf, kb) do { \
        { \
            unsigned int a_row_base = threadIdx.x >> 2; \
            unsigned int a_col = (threadIdx.x & 3) << 3; \
            unsigned int gc = (kb) + a_col; \
            _Pragma("unroll") \
            for (int rnd = 0; rnd < 4; rnd++) { \
                unsigned int row = (unsigned int)(rnd * 32) + a_row_base; \
                unsigned int gr  = cta_m + row; \
                cp_async_pred_16(&smem_A[(buf)][row][a_col], \
                    &A[(unsigned long long)gr * K + gc], \
                    (gr < M) && (gc + 7 < K)); \
            } \
        } \
        { \
            unsigned int my_n = threadIdx.x; \
            unsigned int gn = cta_n + my_n; \
            bool valid = (gn < N) && ((kb) + 31 < K); \
            cp_async_pred_16(&smem_B[(buf)][my_n][0], \
                &B_fp8[(unsigned long long)gn * K + (kb)], valid); \
            cp_async_pred_16(&smem_B[(buf)][my_n][16], \
                &B_fp8[(unsigned long long)gn * K + (kb) + 16], valid); \
        } \
    } while(0)


    #define FGM128_COMPUTE(a_buf, b_buf) do { \
        const unsigned short* sA = (const unsigned short*)smem_A[(a_buf)]; \
        unsigned int fr0, fr1, a0, a1, a2, a3; \
        \
        fr0 = warp_m_offset + group_id; \
        fr1 = fr0 + 8; \
        a0 = bf16x4_to_e4m3x4(&sA[fr0 * a_stride + tid * 4]); \
        a1 = bf16x4_to_e4m3x4(&sA[fr1 * a_stride + tid * 4]); \
        a2 = bf16x4_to_e4m3x4(&sA[fr0 * a_stride + 16 + tid * 4]); \
        a3 = bf16x4_to_e4m3x4(&sA[fr1 * a_stride + 16 + tid * 4]); \
        _Pragma("unroll") \
        for (int nt = 0; nt < 16; nt++) { \
            unsigned int nc = nt * 8 + group_id; \
            unsigned int b0 = *(const unsigned int*)&smem_B[(b_buf)][nc][4 * tid]; \
            unsigned int b1 = *(const unsigned int*)&smem_B[(b_buf)][nc][16 + 4 * tid]; \
            metrale_mma_e4m3(acc0[nt], a0,a1,a2,a3, b0, b1); \
        } \
        \
        fr0 = M_TILE + warp_m_offset + group_id; \
        fr1 = fr0 + 8; \
        a0 = bf16x4_to_e4m3x4(&sA[fr0 * a_stride + tid * 4]); \
        a1 = bf16x4_to_e4m3x4(&sA[fr1 * a_stride + tid * 4]); \
        a2 = bf16x4_to_e4m3x4(&sA[fr0 * a_stride + 16 + tid * 4]); \
        a3 = bf16x4_to_e4m3x4(&sA[fr1 * a_stride + 16 + tid * 4]); \
        _Pragma("unroll") \
        for (int nt = 0; nt < 16; nt++) { \
            unsigned int nc = nt * 8 + group_id; \
            unsigned int b0 = *(const unsigned int*)&smem_B[(b_buf)][nc][4 * tid]; \
            unsigned int b1 = *(const unsigned int*)&smem_B[(b_buf)][nc][16 + 4 * tid]; \
            metrale_mma_e4m3(acc1[nt], a0,a1,a2,a3, b0, b1); \
        } \
    } while(0)

    FGM128_LOADS(0, 0);
    cp_async_commit();
    cp_async_wait_all();
    __syncthreads();

    int cur = 0;
    for (unsigned int k_base = K_STEP_T; k_base < K; k_base += K_STEP_T) {
        int nxt = 1 - cur;
        FGM128_LOADS(nxt, k_base);
        cp_async_commit();
        FGM128_COMPUTE(cur, cur);
        cp_async_wait_all();
        __syncthreads();
        cur = nxt;
    }
    FGM128_COMPUTE(cur, cur);

    #undef FGM128_LOADS
    #undef FGM128_COMPUTE


    #pragma unroll
    for (int nt = 0; nt < 16; nt++) {
        unsigned int c0 = cta_n + nt * 8 + tid * 2;
        unsigned int c1 = c0 + 1;
        unsigned int r0 = cta_m + warp_m_offset + group_id;
        unsigned int r1 = r0 + 8;
        if (r0 < M && c0 < N) C[r0 * N + c0] = __float2bfloat16(acc0[nt][0]);
        if (r0 < M && c1 < N) C[r0 * N + c1] = __float2bfloat16(acc0[nt][1]);
        if (r1 < M && c0 < N) C[r1 * N + c0] = __float2bfloat16(acc0[nt][2]);
        if (r1 < M && c1 < N) C[r1 * N + c1] = __float2bfloat16(acc0[nt][3]);
    }

    #pragma unroll
    for (int nt = 0; nt < 16; nt++) {
        unsigned int c0 = cta_n + nt * 8 + tid * 2;
        unsigned int c1 = c0 + 1;
        unsigned int r0 = cta_m + M_TILE + warp_m_offset + group_id;
        unsigned int r1 = r0 + 8;
        if (r0 < M && c0 < N) C[r0 * N + c0] = __float2bfloat16(acc1[nt][0]);
        if (r0 < M && c1 < N) C[r0 * N + c1] = __float2bfloat16(acc1[nt][1]);
        if (r1 < M && c0 < N) C[r1 * N + c0] = __float2bfloat16(acc1[nt][2]);
        if (r1 < M && c1 < N) C[r1 * N + c1] = __float2bfloat16(acc1[nt][3]);
    }
}

// 2026-09-25: fp8_fp8_gemm_t with a 128-row M tile: each CTA runs two 64-row chunks against one E4M3 B tile.
// Grid (ceil(N/128), ceil(M/128)), 128 threads.








extern "C" __global__
__launch_bounds__(128, 3)
void fp8_fp8_gemm_t_m128(
    const unsigned char* __restrict__ A_fp8,
    const unsigned char* __restrict__ B_fp8,
    __nv_bfloat16* __restrict__ C,
    unsigned int M, unsigned int N, unsigned int K
) {
    const unsigned int cta_n = blockIdx.x * N_TILE_LG;
    const unsigned int cta_m = blockIdx.y * (2 * M_TILE);
    if (cta_m >= M) return;

    const unsigned int warp_id = threadIdx.x / 32;
    const unsigned int lane_id = threadIdx.x % 32;
    const unsigned int warp_m_offset = warp_id * 16;
    const unsigned int group_id = lane_id >> 2;
    const unsigned int tid = lane_id & 3;

    __shared__ unsigned char smem_Af[2][2 * M_TILE][A_FP8_STRIDE];
    __shared__ unsigned char smem_Bf[2][N_TILE_LG][K_STEP_T];

    float acc0[16][4], acc1[16][4];
    #pragma unroll
    for (int i = 0; i < 16; i++) {
        acc0[i][0] = 0.f; acc0[i][1] = 0.f; acc0[i][2] = 0.f; acc0[i][3] = 0.f;
        acc1[i][0] = 0.f; acc1[i][1] = 0.f; acc1[i][2] = 0.f; acc1[i][3] = 0.f;
    }


    #define FFM128_LOADS(buf, kb) do { \
        { \
            unsigned int a_row_base = threadIdx.x >> 1; \
            unsigned int a_col = (threadIdx.x & 1) << 4; \
            unsigned int gc = (kb) + a_col; \
            _Pragma("unroll") \
            for (int rnd = 0; rnd < 2; rnd++) { \
                unsigned int row = (unsigned int)(rnd * 64) + a_row_base; \
                unsigned int gr  = cta_m + row; \
                cp_async_pred_16(&smem_Af[(buf)][row][a_col], \
                    &A_fp8[(unsigned long long)gr * K + gc], \
                    (gr < M) && (gc + 15 < K)); \
            } \
        } \
        { \
            unsigned int my_n = threadIdx.x; \
            unsigned int gn = cta_n + my_n; \
            bool valid = (gn < N) && ((kb) + 31 < K); \
            cp_async_pred_16(&smem_Bf[(buf)][my_n][0], \
                &B_fp8[(unsigned long long)gn * K + (kb)], valid); \
            cp_async_pred_16(&smem_Bf[(buf)][my_n][16], \
                &B_fp8[(unsigned long long)gn * K + (kb) + 16], valid); \
        } \
    } while(0)


    #define FFM128_COMPUTE(a_buf, b_buf) do { \
        unsigned int fr0, fr1, a0, a1, a2, a3; \
        \
        fr0 = warp_m_offset + group_id; \
        fr1 = fr0 + 8; \
        a0 = *(const unsigned int*)&smem_Af[(a_buf)][fr0][4 * tid]; \
        a1 = *(const unsigned int*)&smem_Af[(a_buf)][fr1][4 * tid]; \
        a2 = *(const unsigned int*)&smem_Af[(a_buf)][fr0][16 + 4 * tid]; \
        a3 = *(const unsigned int*)&smem_Af[(a_buf)][fr1][16 + 4 * tid]; \
        _Pragma("unroll") \
        for (int nt = 0; nt < 16; nt++) { \
            unsigned int nc = nt * 8 + group_id; \
            unsigned int b0 = *(const unsigned int*)&smem_Bf[(b_buf)][nc][4 * tid]; \
            unsigned int b1 = *(const unsigned int*)&smem_Bf[(b_buf)][nc][16 + 4 * tid]; \
            metrale_mma_e4m3(acc0[nt], a0,a1,a2,a3, b0, b1); \
        } \
        \
        fr0 = M_TILE + warp_m_offset + group_id; \
        fr1 = fr0 + 8; \
        a0 = *(const unsigned int*)&smem_Af[(a_buf)][fr0][4 * tid]; \
        a1 = *(const unsigned int*)&smem_Af[(a_buf)][fr1][4 * tid]; \
        a2 = *(const unsigned int*)&smem_Af[(a_buf)][fr0][16 + 4 * tid]; \
        a3 = *(const unsigned int*)&smem_Af[(a_buf)][fr1][16 + 4 * tid]; \
        _Pragma("unroll") \
        for (int nt = 0; nt < 16; nt++) { \
            unsigned int nc = nt * 8 + group_id; \
            unsigned int b0 = *(const unsigned int*)&smem_Bf[(b_buf)][nc][4 * tid]; \
            unsigned int b1 = *(const unsigned int*)&smem_Bf[(b_buf)][nc][16 + 4 * tid]; \
            metrale_mma_e4m3(acc1[nt], a0,a1,a2,a3, b0, b1); \
        } \
    } while(0)

    FFM128_LOADS(0, 0);
    cp_async_commit();
    cp_async_wait_all();
    __syncthreads();

    int cur = 0;
    for (unsigned int k_base = K_STEP_T; k_base < K; k_base += K_STEP_T) {
        int nxt = 1 - cur;
        FFM128_LOADS(nxt, k_base);
        cp_async_commit();
        FFM128_COMPUTE(cur, cur);
        cp_async_wait_all();
        __syncthreads();
        cur = nxt;
    }
    FFM128_COMPUTE(cur, cur);

    #undef FFM128_LOADS
    #undef FFM128_COMPUTE


    #pragma unroll
    for (int nt = 0; nt < 16; nt++) {
        unsigned int c0 = cta_n + nt * 8 + tid * 2;
        unsigned int c1 = c0 + 1;
        unsigned int r0 = cta_m + warp_m_offset + group_id;
        unsigned int r1 = r0 + 8;
        if (r0 < M && c0 < N) C[r0 * N + c0] = __float2bfloat16(acc0[nt][0]);
        if (r0 < M && c1 < N) C[r0 * N + c1] = __float2bfloat16(acc0[nt][1]);
        if (r1 < M && c0 < N) C[r1 * N + c0] = __float2bfloat16(acc0[nt][2]);
        if (r1 < M && c1 < N) C[r1 * N + c1] = __float2bfloat16(acc0[nt][3]);
    }

    #pragma unroll
    for (int nt = 0; nt < 16; nt++) {
        unsigned int c0 = cta_n + nt * 8 + tid * 2;
        unsigned int c1 = c0 + 1;
        unsigned int r0 = cta_m + M_TILE + warp_m_offset + group_id;
        unsigned int r1 = r0 + 8;
        if (r0 < M && c0 < N) C[r0 * N + c0] = __float2bfloat16(acc1[nt][0]);
        if (r0 < M && c1 < N) C[r0 * N + c1] = __float2bfloat16(acc1[nt][1]);
        if (r1 < M && c0 < N) C[r1 * N + c0] = __float2bfloat16(acc1[nt][2]);
        if (r1 < M && c1 < N) C[r1 * N + c1] = __float2bfloat16(acc1[nt][3]);
    }
}

// 2026-09-25: C[m, n] = sum over 32-wide K blocks of dot(A_i8[m, blk], B_i8[n, blk]) x A_scale[m, blk] x
// B_scale[n, blk]: int8 A [M, K] and B [N, K], FP32 scales [M, K/32] and [N, K/32], one m16n8k32 s8 MMA per
// block into int32, FP32 accumulation, BF16 out. 128-row M tile (two 64-row chunks), 128 threads, grid
// (ceil(N/128), ceil(M/128)). K must be a multiple of 32.








#define METRALE_MMA_S8(d, a0,a1,a2,a3, b0,b1) \
    asm volatile("mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 " \
        "{%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%10,%11,%12,%13};" \
        : "=r"((d)[0]), "=r"((d)[1]), "=r"((d)[2]), "=r"((d)[3]) \
        : "r"(a0),"r"(a1),"r"(a2),"r"(a3),"r"(b0),"r"(b1), \
          "r"((d)[0]),"r"((d)[1]),"r"((d)[2]),"r"((d)[3]))

extern "C" __global__
__launch_bounds__(128, 3)
void int8_gemm_t_m128(
    const signed char* __restrict__ A_i8,
    const signed char* __restrict__ B_i8,
    const float* __restrict__ A_scale,
    const float* __restrict__ B_scale,
    __nv_bfloat16* __restrict__ C,
    unsigned int M, unsigned int N, unsigned int K
) {
    const unsigned int cta_n = blockIdx.x * N_TILE_LG;
    const unsigned int cta_m = blockIdx.y * (2 * M_TILE);
    if (cta_m >= M) return;
    const unsigned int warp_id = threadIdx.x / 32;
    const unsigned int lane_id = threadIdx.x % 32;
    const unsigned int warp_m_offset = warp_id * 16;
    const unsigned int group_id = lane_id >> 2;
    const unsigned int tid = lane_id & 3;
    const unsigned int nb = K >> 5;

    __shared__ signed char smem_Ai[2][2 * M_TILE][32];
    __shared__ signed char smem_Bi[2][N_TILE_LG][32];
    __shared__ float smem_As[2][2 * M_TILE];
    __shared__ float smem_Bs[2][N_TILE_LG];

    float acc0[16][4], acc1[16][4];
    #pragma unroll
    for (int i = 0; i < 16; i++) {
        acc0[i][0]=0.f; acc0[i][1]=0.f; acc0[i][2]=0.f; acc0[i][3]=0.f;
        acc1[i][0]=0.f; acc1[i][1]=0.f; acc1[i][2]=0.f; acc1[i][3]=0.f;
    }

    #define I8_LOADS(buf, kb) do { \
        { unsigned ar = threadIdx.x >> 1; unsigned ac = (threadIdx.x & 1) << 4; unsigned gc = (kb) + ac; \
          _Pragma("unroll") for (int rnd = 0; rnd < 2; rnd++) { \
            unsigned row = (unsigned)(rnd * 64) + ar; unsigned gr = cta_m + row; \
            cp_async_pred_16(&smem_Ai[(buf)][row][ac], &A_i8[(unsigned long long)gr * K + gc], (gr < M) && (gc + 15 < K)); } } \
        { unsigned my_n = threadIdx.x; unsigned gn = cta_n + my_n; bool v = (gn < N) && ((kb) + 31 < K); \
          cp_async_pred_16(&smem_Bi[(buf)][my_n][0],  &B_i8[(unsigned long long)gn * K + (kb)],      v); \
          cp_async_pred_16(&smem_Bi[(buf)][my_n][16], &B_i8[(unsigned long long)gn * K + (kb) + 16], v); } \
        { unsigned blk = (kb) >> 5; unsigned gr = cta_m + threadIdx.x; unsigned gn = cta_n + threadIdx.x; \
          smem_As[(buf)][threadIdx.x] = (gr < M) ? A_scale[(unsigned long long)gr * nb + blk] : 0.f; \
          smem_Bs[(buf)][threadIdx.x] = (gn < N) ? B_scale[(unsigned long long)gn * nb + blk] : 0.f; } \
    } while(0)

    #define I8_COMPUTE(buf, kb) do { \
        float as00 = smem_As[(buf)][warp_m_offset + group_id]; \
        float as01 = smem_As[(buf)][warp_m_offset + group_id + 8]; \
        float as10 = smem_As[(buf)][M_TILE + warp_m_offset + group_id]; \
        float as11 = smem_As[(buf)][M_TILE + warp_m_offset + group_id + 8]; \
        unsigned fr00 = warp_m_offset + group_id, fr01 = fr00 + 8; \
        unsigned a0c0 = *(const unsigned*)&smem_Ai[(buf)][fr00][4*tid]; \
        unsigned a1c0 = *(const unsigned*)&smem_Ai[(buf)][fr01][4*tid]; \
        unsigned a2c0 = *(const unsigned*)&smem_Ai[(buf)][fr00][16+4*tid]; \
        unsigned a3c0 = *(const unsigned*)&smem_Ai[(buf)][fr01][16+4*tid]; \
        unsigned fr10 = M_TILE + warp_m_offset + group_id, fr11 = fr10 + 8; \
        unsigned a0c1 = *(const unsigned*)&smem_Ai[(buf)][fr10][4*tid]; \
        unsigned a1c1 = *(const unsigned*)&smem_Ai[(buf)][fr11][4*tid]; \
        unsigned a2c1 = *(const unsigned*)&smem_Ai[(buf)][fr10][16+4*tid]; \
        unsigned a3c1 = *(const unsigned*)&smem_Ai[(buf)][fr11][16+4*tid]; \
        _Pragma("unroll") for (int nt = 0; nt < 16; nt++) { \
            unsigned nc = nt * 8 + group_id; \
            unsigned b0 = *(const unsigned*)&smem_Bi[(buf)][nc][4*tid]; \
            unsigned b1 = *(const unsigned*)&smem_Bi[(buf)][nc][16+4*tid]; \
            float bs0 = smem_Bs[(buf)][nt*8 + tid*2]; \
            float bs1 = smem_Bs[(buf)][nt*8 + tid*2 + 1]; \
            int s0[4] = {0,0,0,0}, s1[4] = {0,0,0,0}; \
            METRALE_MMA_S8(s0, a0c0,a1c0,a2c0,a3c0, b0,b1); \
            METRALE_MMA_S8(s1, a0c1,a1c1,a2c1,a3c1, b0,b1); \
            acc0[nt][0] += (float)s0[0]*as00*bs0; acc0[nt][1] += (float)s0[1]*as00*bs1; \
            acc0[nt][2] += (float)s0[2]*as01*bs0; acc0[nt][3] += (float)s0[3]*as01*bs1; \
            acc1[nt][0] += (float)s1[0]*as10*bs0; acc1[nt][1] += (float)s1[1]*as10*bs1; \
            acc1[nt][2] += (float)s1[2]*as11*bs0; acc1[nt][3] += (float)s1[3]*as11*bs1; \
        } \
    } while(0)

    I8_LOADS(0, 0);
    cp_async_commit();
    cp_async_wait_all();
    __syncthreads();

    int cur = 0;
    for (unsigned int k_base = 32; k_base < K; k_base += 32) {
        int nxt = 1 - cur;
        I8_LOADS(nxt, k_base);
        cp_async_commit();
        I8_COMPUTE(cur, k_base - 32);
        cp_async_wait_all();
        __syncthreads();
        cur = nxt;
    }
    I8_COMPUTE(cur, K - 32);

    #undef I8_LOADS
    #undef I8_COMPUTE

    #pragma unroll
    for (int nt = 0; nt < 16; nt++) {
        unsigned c0 = cta_n + nt*8 + tid*2, c1 = c0 + 1;
        unsigned r0 = cta_m + warp_m_offset + group_id, r1 = r0 + 8;
        if (r0 < M && c0 < N) C[r0*N+c0] = __float2bfloat16(acc0[nt][0]);
        if (r0 < M && c1 < N) C[r0*N+c1] = __float2bfloat16(acc0[nt][1]);
        if (r1 < M && c0 < N) C[r1*N+c0] = __float2bfloat16(acc0[nt][2]);
        if (r1 < M && c1 < N) C[r1*N+c1] = __float2bfloat16(acc0[nt][3]);
    }
    #pragma unroll
    for (int nt = 0; nt < 16; nt++) {
        unsigned c0 = cta_n + nt*8 + tid*2, c1 = c0 + 1;
        unsigned r0 = cta_m + M_TILE + warp_m_offset + group_id, r1 = r0 + 8;
        if (r0 < M && c0 < N) C[r0*N+c0] = __float2bfloat16(acc1[nt][0]);
        if (r0 < M && c1 < N) C[r0*N+c1] = __float2bfloat16(acc1[nt][1]);
        if (r1 < M && c0 < N) C[r1*N+c0] = __float2bfloat16(acc1[nt][2]);
        if (r1 < M && c1 < N) C[r1*N+c1] = __float2bfloat16(acc1[nt][3]);
    }
}
#undef METRALE_MMA_S8

// 2026-09-25: int8_gemm_t_m128 with one 64-row M chunk per CTA; grid (ceil(N/128), ceil(M/64)), 128 threads.





#define METRALE_MMA_S8(d, a0,a1,a2,a3, b0,b1) \
    asm volatile("mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 " \
        "{%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%10,%11,%12,%13};" \
        : "=r"((d)[0]), "=r"((d)[1]), "=r"((d)[2]), "=r"((d)[3]) \
        : "r"(a0),"r"(a1),"r"(a2),"r"(a3),"r"(b0),"r"(b1), \
          "r"((d)[0]),"r"((d)[1]),"r"((d)[2]),"r"((d)[3]))

extern "C" __global__
__launch_bounds__(128, 4)
void int8_gemm_t_m64(
    const signed char* __restrict__ A_i8,
    const signed char* __restrict__ B_i8,
    const float* __restrict__ A_scale,
    const float* __restrict__ B_scale,
    __nv_bfloat16* __restrict__ C,
    unsigned int M, unsigned int N, unsigned int K
) {
    const unsigned int cta_n = blockIdx.x * N_TILE_LG;
    const unsigned int cta_m = blockIdx.y * M_TILE;
    if (cta_m >= M) return;
    const unsigned int warp_id = threadIdx.x / 32;
    const unsigned int lane_id = threadIdx.x % 32;
    const unsigned int warp_m_offset = warp_id * 16;
    const unsigned int group_id = lane_id >> 2;
    const unsigned int tid = lane_id & 3;
    const unsigned int nb = K >> 5;

    __shared__ signed char smem_Ai[2][M_TILE][32];
    __shared__ signed char smem_Bi[2][N_TILE_LG][32];
    __shared__ float smem_As[2][M_TILE];
    __shared__ float smem_Bs[2][N_TILE_LG];

    float acc[16][4];
    #pragma unroll
    for (int i = 0; i < 16; i++) { acc[i][0]=0.f; acc[i][1]=0.f; acc[i][2]=0.f; acc[i][3]=0.f; }

    #define I8M64_LOADS(buf, kb) do { \
        { unsigned ar = threadIdx.x >> 1; unsigned ac = (threadIdx.x & 1) << 4; unsigned gc = (kb) + ac; \
          unsigned gr = cta_m + ar; \
          cp_async_pred_16(&smem_Ai[(buf)][ar][ac], &A_i8[(unsigned long long)gr*K+gc], (gr<M)&&(gc+15<K)); } \
        { unsigned my_n = threadIdx.x; unsigned gn = cta_n + my_n; bool v = (gn<N)&&((kb)+31<K); \
          cp_async_pred_16(&smem_Bi[(buf)][my_n][0],  &B_i8[(unsigned long long)gn*K+(kb)],    v); \
          cp_async_pred_16(&smem_Bi[(buf)][my_n][16], &B_i8[(unsigned long long)gn*K+(kb)+16], v); } \
        { unsigned blk=(kb)>>5; \
          if (threadIdx.x < M_TILE) { unsigned gr=cta_m+threadIdx.x; smem_As[(buf)][threadIdx.x]=(gr<M)?A_scale[(unsigned long long)gr*nb+blk]:0.f; } \
          unsigned gn=cta_n+threadIdx.x; smem_Bs[(buf)][threadIdx.x]=(gn<N)?B_scale[(unsigned long long)gn*nb+blk]:0.f; } \
    } while(0)

    #define I8M64_COMPUTE(buf) do { \
        float as0 = smem_As[(buf)][warp_m_offset + group_id]; \
        float as1 = smem_As[(buf)][warp_m_offset + group_id + 8]; \
        unsigned fr0 = warp_m_offset + group_id, fr1 = fr0 + 8; \
        unsigned a0 = *(const unsigned*)&smem_Ai[(buf)][fr0][4*tid]; \
        unsigned a1 = *(const unsigned*)&smem_Ai[(buf)][fr1][4*tid]; \
        unsigned a2 = *(const unsigned*)&smem_Ai[(buf)][fr0][16+4*tid]; \
        unsigned a3 = *(const unsigned*)&smem_Ai[(buf)][fr1][16+4*tid]; \
        _Pragma("unroll") for (int nt = 0; nt < 16; nt++) { \
            unsigned nc = nt*8 + group_id; \
            unsigned b0 = *(const unsigned*)&smem_Bi[(buf)][nc][4*tid]; \
            unsigned b1 = *(const unsigned*)&smem_Bi[(buf)][nc][16+4*tid]; \
            float bs0 = smem_Bs[(buf)][nt*8+tid*2]; \
            float bs1 = smem_Bs[(buf)][nt*8+tid*2+1]; \
            int s[4] = {0,0,0,0}; \
            METRALE_MMA_S8(s, a0,a1,a2,a3, b0,b1); \
            acc[nt][0]+=(float)s[0]*as0*bs0; acc[nt][1]+=(float)s[1]*as0*bs1; \
            acc[nt][2]+=(float)s[2]*as1*bs0; acc[nt][3]+=(float)s[3]*as1*bs1; \
        } \
    } while(0)

    I8M64_LOADS(0, 0); cp_async_commit(); cp_async_wait_all(); __syncthreads();
    int cur = 0;
    for (unsigned int k_base = 32; k_base < K; k_base += 32) {
        int nxt = 1 - cur;
        I8M64_LOADS(nxt, k_base); cp_async_commit();
        I8M64_COMPUTE(cur);
        cp_async_wait_all(); __syncthreads();
        cur = nxt;
    }
    I8M64_COMPUTE(cur);
    #undef I8M64_LOADS
    #undef I8M64_COMPUTE

    #pragma unroll
    for (int nt = 0; nt < 16; nt++) {
        unsigned c0 = cta_n + nt*8 + tid*2, c1 = c0 + 1;
        unsigned r0 = cta_m + warp_m_offset + group_id, r1 = r0 + 8;
        if (r0<M&&c0<N) C[r0*N+c0]=__float2bfloat16(acc[nt][0]);
        if (r0<M&&c1<N) C[r0*N+c1]=__float2bfloat16(acc[nt][1]);
        if (r1<M&&c0<N) C[r1*N+c0]=__float2bfloat16(acc[nt][2]);
        if (r1<M&&c1<N) C[r1*N+c1]=__float2bfloat16(acc[nt][3]);
    }
}
#undef METRALE_MMA_S8

// 2026-09-25: int8_gemm_t_m128 split over K: CTA z of ksplits (gridDim.z) covers K / ksplits and writes FP32
// partials to Cp [ksplits, M, N]; int8_splitk_reduce sums them into C. K must be a multiple of 32 * ksplits.










#define METRALE_MMA_S8(d, a0,a1,a2,a3, b0,b1) \
    asm volatile("mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 " \
        "{%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%10,%11,%12,%13};" \
        : "=r"((d)[0]), "=r"((d)[1]), "=r"((d)[2]), "=r"((d)[3]) \
        : "r"(a0),"r"(a1),"r"(a2),"r"(a3),"r"(b0),"r"(b1), \
          "r"((d)[0]),"r"((d)[1]),"r"((d)[2]),"r"((d)[3]))

extern "C" __global__
__launch_bounds__(128, 3)
void int8_gemm_splitk(
    const signed char* __restrict__ A_i8,
    const signed char* __restrict__ B_i8,
    const float* __restrict__ A_scale,
    const float* __restrict__ B_scale,
    float* __restrict__ Cp,
    unsigned int M, unsigned int N, unsigned int K, unsigned int ksplits
) {
    const unsigned int cta_n = blockIdx.x * N_TILE_LG;
    const unsigned int cta_m = blockIdx.y * (2 * M_TILE);
    const unsigned int z = blockIdx.z;
    if (cta_m >= M) return;
    const unsigned int warp_id = threadIdx.x / 32;
    const unsigned int lane_id = threadIdx.x % 32;
    const unsigned int warp_m_offset = warp_id * 16;
    const unsigned int group_id = lane_id >> 2;
    const unsigned int tid = lane_id & 3;
    const unsigned int nb = K >> 5;
    const unsigned int k_per = K / ksplits;
    const unsigned int k_lo = z * k_per;
    const unsigned int k_hi = k_lo + k_per;

    __shared__ signed char smem_Ai[2][2 * M_TILE][32];
    __shared__ signed char smem_Bi[2][N_TILE_LG][32];
    __shared__ float smem_As[2][2 * M_TILE];
    __shared__ float smem_Bs[2][N_TILE_LG];

    float acc0[16][4], acc1[16][4];
    #pragma unroll
    for (int i = 0; i < 16; i++) {
        acc0[i][0]=0.f; acc0[i][1]=0.f; acc0[i][2]=0.f; acc0[i][3]=0.f;
        acc1[i][0]=0.f; acc1[i][1]=0.f; acc1[i][2]=0.f; acc1[i][3]=0.f;
    }

    #define SK_LOADS(buf, kb) do { \
        { unsigned ar = threadIdx.x >> 1; unsigned ac = (threadIdx.x & 1) << 4; unsigned gc = (kb) + ac; \
          _Pragma("unroll") for (int rnd = 0; rnd < 2; rnd++) { \
            unsigned row = (unsigned)(rnd * 64) + ar; unsigned gr = cta_m + row; \
            cp_async_pred_16(&smem_Ai[(buf)][row][ac], &A_i8[(unsigned long long)gr * K + gc], (gr < M) && (gc + 15 < K)); } } \
        { unsigned my_n = threadIdx.x; unsigned gn = cta_n + my_n; bool v = (gn < N) && ((kb) + 31 < K); \
          cp_async_pred_16(&smem_Bi[(buf)][my_n][0],  &B_i8[(unsigned long long)gn * K + (kb)],      v); \
          cp_async_pred_16(&smem_Bi[(buf)][my_n][16], &B_i8[(unsigned long long)gn * K + (kb) + 16], v); } \
        { unsigned blk = (kb) >> 5; unsigned gr = cta_m + threadIdx.x; unsigned gn = cta_n + threadIdx.x; \
          smem_As[(buf)][threadIdx.x] = (gr < M) ? A_scale[(unsigned long long)gr * nb + blk] : 0.f; \
          smem_Bs[(buf)][threadIdx.x] = (gn < N) ? B_scale[(unsigned long long)gn * nb + blk] : 0.f; } \
    } while(0)

    #define SK_COMPUTE(buf) do { \
        float as00 = smem_As[(buf)][warp_m_offset + group_id]; \
        float as01 = smem_As[(buf)][warp_m_offset + group_id + 8]; \
        float as10 = smem_As[(buf)][M_TILE + warp_m_offset + group_id]; \
        float as11 = smem_As[(buf)][M_TILE + warp_m_offset + group_id + 8]; \
        unsigned fr00 = warp_m_offset + group_id, fr01 = fr00 + 8; \
        unsigned a0c0 = *(const unsigned*)&smem_Ai[(buf)][fr00][4*tid]; \
        unsigned a1c0 = *(const unsigned*)&smem_Ai[(buf)][fr01][4*tid]; \
        unsigned a2c0 = *(const unsigned*)&smem_Ai[(buf)][fr00][16+4*tid]; \
        unsigned a3c0 = *(const unsigned*)&smem_Ai[(buf)][fr01][16+4*tid]; \
        unsigned fr10 = M_TILE + warp_m_offset + group_id, fr11 = fr10 + 8; \
        unsigned a0c1 = *(const unsigned*)&smem_Ai[(buf)][fr10][4*tid]; \
        unsigned a1c1 = *(const unsigned*)&smem_Ai[(buf)][fr11][4*tid]; \
        unsigned a2c1 = *(const unsigned*)&smem_Ai[(buf)][fr10][16+4*tid]; \
        unsigned a3c1 = *(const unsigned*)&smem_Ai[(buf)][fr11][16+4*tid]; \
        _Pragma("unroll") for (int nt = 0; nt < 16; nt++) { \
            unsigned nc = nt * 8 + group_id; \
            unsigned b0 = *(const unsigned*)&smem_Bi[(buf)][nc][4*tid]; \
            unsigned b1 = *(const unsigned*)&smem_Bi[(buf)][nc][16+4*tid]; \
            float bs0 = smem_Bs[(buf)][nt*8 + tid*2]; \
            float bs1 = smem_Bs[(buf)][nt*8 + tid*2 + 1]; \
            int s0[4] = {0,0,0,0}, s1[4] = {0,0,0,0}; \
            METRALE_MMA_S8(s0, a0c0,a1c0,a2c0,a3c0, b0,b1); \
            METRALE_MMA_S8(s1, a0c1,a1c1,a2c1,a3c1, b0,b1); \
            acc0[nt][0] += (float)s0[0]*as00*bs0; acc0[nt][1] += (float)s0[1]*as00*bs1; \
            acc0[nt][2] += (float)s0[2]*as01*bs0; acc0[nt][3] += (float)s0[3]*as01*bs1; \
            acc1[nt][0] += (float)s1[0]*as10*bs0; acc1[nt][1] += (float)s1[1]*as10*bs1; \
            acc1[nt][2] += (float)s1[2]*as11*bs0; acc1[nt][3] += (float)s1[3]*as11*bs1; \
        } \
    } while(0)

    SK_LOADS(0, k_lo);
    cp_async_commit();
    cp_async_wait_all();
    __syncthreads();
    int cur = 0;
    for (unsigned int kb = k_lo + 32; kb < k_hi; kb += 32) {
        int nxt = 1 - cur;
        SK_LOADS(nxt, kb);
        cp_async_commit();
        SK_COMPUTE(cur);
        cp_async_wait_all();
        __syncthreads();
        cur = nxt;
    }
    SK_COMPUTE(cur);
    #undef SK_LOADS
    #undef SK_COMPUTE

    unsigned long long zoff = (unsigned long long)z * M * N;
    #pragma unroll
    for (int nt = 0; nt < 16; nt++) {
        unsigned c0 = cta_n + nt*8 + tid*2, c1 = c0 + 1;
        unsigned r0 = cta_m + warp_m_offset + group_id, r1 = r0 + 8;
        if (r0 < M && c0 < N) Cp[zoff + (unsigned long long)r0*N+c0] = acc0[nt][0];
        if (r0 < M && c1 < N) Cp[zoff + (unsigned long long)r0*N+c1] = acc0[nt][1];
        if (r1 < M && c0 < N) Cp[zoff + (unsigned long long)r1*N+c0] = acc0[nt][2];
        if (r1 < M && c1 < N) Cp[zoff + (unsigned long long)r1*N+c1] = acc0[nt][3];
    }
    #pragma unroll
    for (int nt = 0; nt < 16; nt++) {
        unsigned c0 = cta_n + nt*8 + tid*2, c1 = c0 + 1;
        unsigned r0 = cta_m + M_TILE + warp_m_offset + group_id, r1 = r0 + 8;
        if (r0 < M && c0 < N) Cp[zoff + (unsigned long long)r0*N+c0] = acc1[nt][0];
        if (r0 < M && c1 < N) Cp[zoff + (unsigned long long)r0*N+c1] = acc1[nt][1];
        if (r1 < M && c0 < N) Cp[zoff + (unsigned long long)r1*N+c0] = acc1[nt][2];
        if (r1 < M && c1 < N) Cp[zoff + (unsigned long long)r1*N+c1] = acc1[nt][3];
    }
}
#undef METRALE_MMA_S8

// 2026-09-25: C [M, N] BF16 = sum over z of Cp[z], Cp [ksplits, M, N] FP32; one element per thread.
extern "C" __global__ void int8_splitk_reduce(
    const float* __restrict__ Cp, __nv_bfloat16* __restrict__ C,
    unsigned int M, unsigned int N, unsigned int ksplits
) {
    unsigned long long idx = (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;
    unsigned long long MN = (unsigned long long)M * N;
    if (idx >= MN) return;
    float s = 0.f;
    for (unsigned z = 0; z < ksplits; z++) s += Cp[(unsigned long long)z * MN + idx];
    C[idx] = __float2bfloat16(s);
}

// 2026-09-25: int8_gemm_t_m128 with a K step of 64: two 32-wide sub-blocks per step, each with its own scales.
// K must be a multiple of 64.







#define METRALE_MMA_S8(d, a0,a1,a2,a3, b0,b1) \
    asm volatile("mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 " \
        "{%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%10,%11,%12,%13};" \
        : "=r"((d)[0]), "=r"((d)[1]), "=r"((d)[2]), "=r"((d)[3]) \
        : "r"(a0),"r"(a1),"r"(a2),"r"(a3),"r"(b0),"r"(b1), \
          "r"((d)[0]),"r"((d)[1]),"r"((d)[2]),"r"((d)[3]))

extern "C" __global__
__launch_bounds__(128, 3)
void int8_gemm_t_m128_k64(
    const signed char* __restrict__ A_i8,
    const signed char* __restrict__ B_i8,
    const float* __restrict__ A_scale,
    const float* __restrict__ B_scale,
    __nv_bfloat16* __restrict__ C,
    unsigned int M, unsigned int N, unsigned int K
) {
    const unsigned int cta_n = blockIdx.x * N_TILE_LG;
    const unsigned int cta_m = blockIdx.y * (2 * M_TILE);
    if (cta_m >= M) return;
    const unsigned int warp_id = threadIdx.x / 32;
    const unsigned int lane_id = threadIdx.x % 32;
    const unsigned int warp_m_offset = warp_id * 16;
    const unsigned int group_id = lane_id >> 2;
    const unsigned int tid = lane_id & 3;
    const unsigned int nb = K >> 5;

    __shared__ signed char smem_Ai[2][2 * M_TILE][64];
    __shared__ signed char smem_Bi[2][N_TILE_LG][64];
    __shared__ float smem_As[2][2 * M_TILE][2];
    __shared__ float smem_Bs[2][N_TILE_LG][2];

    float acc0[16][4], acc1[16][4];
    #pragma unroll
    for (int i = 0; i < 16; i++) {
        acc0[i][0]=0.f; acc0[i][1]=0.f; acc0[i][2]=0.f; acc0[i][3]=0.f;
        acc1[i][0]=0.f; acc1[i][1]=0.f; acc1[i][2]=0.f; acc1[i][3]=0.f;
    }

    #define K64_LOADS(buf, kb) do { \
        { unsigned ar = threadIdx.x >> 2; unsigned ac = (threadIdx.x & 3) << 4; unsigned gc = (kb) + ac; \
          _Pragma("unroll") for (int rnd = 0; rnd < 4; rnd++) { \
            unsigned row = (unsigned)(rnd * 32) + ar; unsigned gr = cta_m + row; \
            cp_async_pred_16(&smem_Ai[(buf)][row][ac], &A_i8[(unsigned long long)gr * K + gc], (gr < M) && (gc + 15 < K)); } } \
        { unsigned my_n = threadIdx.x; unsigned gn = cta_n + my_n; bool v = (gn < N) && ((kb) + 63 < K); \
          _Pragma("unroll") for (int c = 0; c < 4; c++) \
            cp_async_pred_16(&smem_Bi[(buf)][my_n][c*16], &B_i8[(unsigned long long)gn * K + (kb) + c*16], v); } \
        { unsigned blk = (kb) >> 5; unsigned gr = cta_m + threadIdx.x; unsigned gn = cta_n + threadIdx.x; \
          smem_As[(buf)][threadIdx.x][0] = (gr < M) ? A_scale[(unsigned long long)gr * nb + blk]     : 0.f; \
          smem_As[(buf)][threadIdx.x][1] = (gr < M) ? A_scale[(unsigned long long)gr * nb + blk + 1] : 0.f; \
          smem_Bs[(buf)][threadIdx.x][0] = (gn < N) ? B_scale[(unsigned long long)gn * nb + blk]     : 0.f; \
          smem_Bs[(buf)][threadIdx.x][1] = (gn < N) ? B_scale[(unsigned long long)gn * nb + blk + 1] : 0.f; } \
    } while(0)


    #define K64_SUB(buf, sb) do { \
        float as00 = smem_As[(buf)][warp_m_offset + group_id][sb]; \
        float as01 = smem_As[(buf)][warp_m_offset + group_id + 8][sb]; \
        float as10 = smem_As[(buf)][M_TILE + warp_m_offset + group_id][sb]; \
        float as11 = smem_As[(buf)][M_TILE + warp_m_offset + group_id + 8][sb]; \
        unsigned off = (sb) * 32; \
        unsigned fr00 = warp_m_offset + group_id, fr01 = fr00 + 8; \
        unsigned a0c0 = *(const unsigned*)&smem_Ai[(buf)][fr00][off+4*tid]; \
        unsigned a1c0 = *(const unsigned*)&smem_Ai[(buf)][fr01][off+4*tid]; \
        unsigned a2c0 = *(const unsigned*)&smem_Ai[(buf)][fr00][off+16+4*tid]; \
        unsigned a3c0 = *(const unsigned*)&smem_Ai[(buf)][fr01][off+16+4*tid]; \
        unsigned fr10 = M_TILE + warp_m_offset + group_id, fr11 = fr10 + 8; \
        unsigned a0c1 = *(const unsigned*)&smem_Ai[(buf)][fr10][off+4*tid]; \
        unsigned a1c1 = *(const unsigned*)&smem_Ai[(buf)][fr11][off+4*tid]; \
        unsigned a2c1 = *(const unsigned*)&smem_Ai[(buf)][fr10][off+16+4*tid]; \
        unsigned a3c1 = *(const unsigned*)&smem_Ai[(buf)][fr11][off+16+4*tid]; \
        _Pragma("unroll") for (int nt = 0; nt < 16; nt++) { \
            unsigned nc = nt * 8 + group_id; \
            unsigned b0 = *(const unsigned*)&smem_Bi[(buf)][nc][off+4*tid]; \
            unsigned b1 = *(const unsigned*)&smem_Bi[(buf)][nc][off+16+4*tid]; \
            float bs0 = smem_Bs[(buf)][nt*8 + tid*2][sb]; \
            float bs1 = smem_Bs[(buf)][nt*8 + tid*2 + 1][sb]; \
            int s0[4] = {0,0,0,0}, s1[4] = {0,0,0,0}; \
            METRALE_MMA_S8(s0, a0c0,a1c0,a2c0,a3c0, b0,b1); \
            METRALE_MMA_S8(s1, a0c1,a1c1,a2c1,a3c1, b0,b1); \
            acc0[nt][0] += (float)s0[0]*as00*bs0; acc0[nt][1] += (float)s0[1]*as00*bs1; \
            acc0[nt][2] += (float)s0[2]*as01*bs0; acc0[nt][3] += (float)s0[3]*as01*bs1; \
            acc1[nt][0] += (float)s1[0]*as10*bs0; acc1[nt][1] += (float)s1[1]*as10*bs1; \
            acc1[nt][2] += (float)s1[2]*as11*bs0; acc1[nt][3] += (float)s1[3]*as11*bs1; \
        } \
    } while(0)

    K64_LOADS(0, 0);
    cp_async_commit();
    cp_async_wait_all();
    __syncthreads();
    int cur = 0;
    for (unsigned int kb = 64; kb < K; kb += 64) {
        int nxt = 1 - cur;
        K64_LOADS(nxt, kb);
        cp_async_commit();
        K64_SUB(cur, 0);
        K64_SUB(cur, 1);
        cp_async_wait_all();
        __syncthreads();
        cur = nxt;
    }
    K64_SUB(cur, 0);
    K64_SUB(cur, 1);
    #undef K64_LOADS
    #undef K64_SUB

    #pragma unroll
    for (int nt = 0; nt < 16; nt++) {
        unsigned c0 = cta_n + nt*8 + tid*2, c1 = c0 + 1;
        unsigned r0 = cta_m + warp_m_offset + group_id, r1 = r0 + 8;
        if (r0 < M && c0 < N) C[r0*N+c0] = __float2bfloat16(acc0[nt][0]);
        if (r0 < M && c1 < N) C[r0*N+c1] = __float2bfloat16(acc0[nt][1]);
        if (r1 < M && c0 < N) C[r1*N+c0] = __float2bfloat16(acc0[nt][2]);
        if (r1 < M && c1 < N) C[r1*N+c1] = __float2bfloat16(acc0[nt][3]);
    }
    #pragma unroll
    for (int nt = 0; nt < 16; nt++) {
        unsigned c0 = cta_n + nt*8 + tid*2, c1 = c0 + 1;
        unsigned r0 = cta_m + M_TILE + warp_m_offset + group_id, r1 = r0 + 8;
        if (r0 < M && c0 < N) C[r0*N+c0] = __float2bfloat16(acc1[nt][0]);
        if (r0 < M && c1 < N) C[r0*N+c1] = __float2bfloat16(acc1[nt][1]);
        if (r1 < M && c0 < N) C[r1*N+c0] = __float2bfloat16(acc1[nt][2]);
        if (r1 < M && c1 < N) C[r1*N+c1] = __float2bfloat16(acc1[nt][3]);
    }
}
#undef METRALE_MMA_S8

// 2026-09-25: int8_gemm_t_m128's math on 256 threads: eight warps, each owning 16 rows of a 128 x 128 tile;
// grid (ceil(N/128), ceil(M/128)).








#define METRALE_MMA_S8(d, a0,a1,a2,a3, b0,b1) \
    asm volatile("mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 " \
        "{%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%10,%11,%12,%13};" \
        : "=r"((d)[0]), "=r"((d)[1]), "=r"((d)[2]), "=r"((d)[3]) \
        : "r"(a0),"r"(a1),"r"(a2),"r"(a3),"r"(b0),"r"(b1), \
          "r"((d)[0]),"r"((d)[1]),"r"((d)[2]),"r"((d)[3]))

extern "C" __global__
__launch_bounds__(256, 2)
void int8_gemm_8w(
    const signed char* __restrict__ A_i8,
    const signed char* __restrict__ B_i8,
    const float* __restrict__ A_scale,
    const float* __restrict__ B_scale,
    __nv_bfloat16* __restrict__ C,
    unsigned int M, unsigned int N, unsigned int K
) {
    const unsigned int cta_m = blockIdx.y * 128;
    const unsigned int cta_n = blockIdx.x * 128;
    if (cta_m >= M) return;
    const unsigned int t = threadIdx.x;
    const unsigned int warp_id = t >> 5;
    const unsigned int lane = t & 31;
    const unsigned int group_id = lane >> 2;
    const unsigned int t4 = lane & 3;
    const unsigned int nb = K >> 5;
    const unsigned int wrow = warp_id * 16;

    __shared__ signed char smem_Ai[2][128][32];
    __shared__ signed char smem_Bi[2][128][32];
    __shared__ float smem_As[2][128];
    __shared__ float smem_Bs[2][128];

    float acc[16][4];
    #pragma unroll
    for (int i = 0; i < 16; i++) { acc[i][0]=0.f; acc[i][1]=0.f; acc[i][2]=0.f; acc[i][3]=0.f; }

    #define W8_LOADS(buf, kb) do { \
        { unsigned ar = t >> 1; unsigned ac = (t & 1) << 4; unsigned gc = (kb) + ac; unsigned gr = cta_m + ar; \
          cp_async_pred_16(&smem_Ai[(buf)][ar][ac], &A_i8[(unsigned long long)gr*K+gc], (gr<M)&&(gc+15<K)); } \
        { unsigned an = t >> 1; unsigned ac = (t & 1) << 4; unsigned gc = (kb) + ac; unsigned gn = cta_n + an; \
          cp_async_pred_16(&smem_Bi[(buf)][an][ac], &B_i8[(unsigned long long)gn*K+gc], (gn<N)&&(gc+15<K)); } \
        if (t < 128) { unsigned blk=(kb)>>5; unsigned gr=cta_m+t; unsigned gn=cta_n+t; \
          smem_As[(buf)][t] = (gr<M)?A_scale[(unsigned long long)gr*nb+blk]:0.f; \
          smem_Bs[(buf)][t] = (gn<N)?B_scale[(unsigned long long)gn*nb+blk]:0.f; } \
    } while(0)

    #define W8_COMPUTE(buf) do { \
        float as0 = smem_As[(buf)][wrow + group_id]; \
        float as1 = smem_As[(buf)][wrow + group_id + 8]; \
        unsigned fr0 = wrow + group_id, fr1 = fr0 + 8; \
        unsigned a0 = *(const unsigned*)&smem_Ai[(buf)][fr0][4*t4]; \
        unsigned a1 = *(const unsigned*)&smem_Ai[(buf)][fr1][4*t4]; \
        unsigned a2 = *(const unsigned*)&smem_Ai[(buf)][fr0][16+4*t4]; \
        unsigned a3 = *(const unsigned*)&smem_Ai[(buf)][fr1][16+4*t4]; \
        _Pragma("unroll") for (int nt = 0; nt < 16; nt++) { \
            unsigned nc = nt*8 + group_id; \
            unsigned b0 = *(const unsigned*)&smem_Bi[(buf)][nc][4*t4]; \
            unsigned b1 = *(const unsigned*)&smem_Bi[(buf)][nc][16+4*t4]; \
            float bs0 = smem_Bs[(buf)][nt*8 + t4*2]; \
            float bs1 = smem_Bs[(buf)][nt*8 + t4*2 + 1]; \
            int s[4] = {0,0,0,0}; \
            METRALE_MMA_S8(s, a0,a1,a2,a3, b0,b1); \
            acc[nt][0] += (float)s[0]*as0*bs0; acc[nt][1] += (float)s[1]*as0*bs1; \
            acc[nt][2] += (float)s[2]*as1*bs0; acc[nt][3] += (float)s[3]*as1*bs1; \
        } \
    } while(0)

    W8_LOADS(0, 0); cp_async_commit(); cp_async_wait_all(); __syncthreads();
    int cur = 0;
    for (unsigned int kb = 32; kb < K; kb += 32) {
        int nxt = 1 - cur;
        W8_LOADS(nxt, kb); cp_async_commit();
        W8_COMPUTE(cur);
        cp_async_wait_all(); __syncthreads();
        cur = nxt;
    }
    W8_COMPUTE(cur);
    #undef W8_LOADS
    #undef W8_COMPUTE

    #pragma unroll
    for (int nt = 0; nt < 16; nt++) {
        unsigned c0 = cta_n + nt*8 + t4*2, c1 = c0 + 1;
        unsigned r0 = cta_m + wrow + group_id, r1 = r0 + 8;
        if (r0<M&&c0<N) C[r0*N+c0]=__float2bfloat16(acc[nt][0]);
        if (r0<M&&c1<N) C[r0*N+c1]=__float2bfloat16(acc[nt][1]);
        if (r1<M&&c0<N) C[r1*N+c0]=__float2bfloat16(acc[nt][2]);
        if (r1<M&&c1<N) C[r1*N+c1]=__float2bfloat16(acc[nt][3]);
    }
}
#undef METRALE_MMA_S8

// 2026-09-25: int8_gemm_8w with three stages: up to two cp.async groups in flight, and between steps it waits
// with cp.async.wait_group 1 instead of draining every group.







#define METRALE_MMA_S8(d, a0,a1,a2,a3, b0,b1) \
    asm volatile("mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 " \
        "{%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%10,%11,%12,%13};" \
        : "=r"((d)[0]), "=r"((d)[1]), "=r"((d)[2]), "=r"((d)[3]) \
        : "r"(a0),"r"(a1),"r"(a2),"r"(a3),"r"(b0),"r"(b1), \
          "r"((d)[0]),"r"((d)[1]),"r"((d)[2]),"r"((d)[3]))
// 2026-09-25: wait until at most N cp.async groups are in flight; N must be a compile-time immediate
#define CP_WAIT_GROUP(N) asm volatile("cp.async.wait_group %0;" :: "n"(N))

extern "C" __global__
__launch_bounds__(256, 2)
void int8_gemm_8w3(
    const signed char* __restrict__ A_i8,
    const signed char* __restrict__ B_i8,
    const float* __restrict__ A_scale,
    const float* __restrict__ B_scale,
    __nv_bfloat16* __restrict__ C,
    unsigned int M, unsigned int N, unsigned int K
) {
    const unsigned int cta_m = blockIdx.y * 128;
    const unsigned int cta_n = blockIdx.x * 128;
    if (cta_m >= M) return;
    const unsigned int t = threadIdx.x;
    const unsigned int warp_id = t >> 5;
    const unsigned int lane = t & 31;
    const unsigned int group_id = lane >> 2;
    const unsigned int t4 = lane & 3;
    const unsigned int nb = K >> 5;
    const unsigned int wrow = warp_id * 16;
    const unsigned int nk = K >> 5;

    __shared__ signed char smem_Ai[3][128][32];
    __shared__ signed char smem_Bi[3][128][32];
    __shared__ float smem_As[3][128];
    __shared__ float smem_Bs[3][128];

    float acc[16][4];
    #pragma unroll
    for (int i = 0; i < 16; i++) { acc[i][0]=0.f; acc[i][1]=0.f; acc[i][2]=0.f; acc[i][3]=0.f; }

    #define W83_LOADS(buf, kb) do { \
        { unsigned ar = t >> 1; unsigned ac = (t & 1) << 4; unsigned gc = (kb) + ac; unsigned gr = cta_m + ar; \
          cp_async_pred_16(&smem_Ai[(buf)][ar][ac], &A_i8[(unsigned long long)gr*K+gc], (gr<M)&&(gc+15<K)); } \
        { unsigned an = t >> 1; unsigned ac = (t & 1) << 4; unsigned gc = (kb) + ac; unsigned gn = cta_n + an; \
          cp_async_pred_16(&smem_Bi[(buf)][an][ac], &B_i8[(unsigned long long)gn*K+gc], (gn<N)&&(gc+15<K)); } \
        if (t < 128) { unsigned blk=(kb)>>5; unsigned gr=cta_m+t; unsigned gn=cta_n+t; \
          smem_As[(buf)][t] = (gr<M)?A_scale[(unsigned long long)gr*nb+blk]:0.f; \
          smem_Bs[(buf)][t] = (gn<N)?B_scale[(unsigned long long)gn*nb+blk]:0.f; } \
    } while(0)

    #define W83_COMPUTE(buf) do { \
        float as0 = smem_As[(buf)][wrow + group_id]; \
        float as1 = smem_As[(buf)][wrow + group_id + 8]; \
        unsigned fr0 = wrow + group_id, fr1 = fr0 + 8; \
        unsigned a0 = *(const unsigned*)&smem_Ai[(buf)][fr0][4*t4]; \
        unsigned a1 = *(const unsigned*)&smem_Ai[(buf)][fr1][4*t4]; \
        unsigned a2 = *(const unsigned*)&smem_Ai[(buf)][fr0][16+4*t4]; \
        unsigned a3 = *(const unsigned*)&smem_Ai[(buf)][fr1][16+4*t4]; \
        _Pragma("unroll") for (int nt = 0; nt < 16; nt++) { \
            unsigned nc = nt*8 + group_id; \
            unsigned b0 = *(const unsigned*)&smem_Bi[(buf)][nc][4*t4]; \
            unsigned b1 = *(const unsigned*)&smem_Bi[(buf)][nc][16+4*t4]; \
            float bs0 = smem_Bs[(buf)][nt*8 + t4*2]; \
            float bs1 = smem_Bs[(buf)][nt*8 + t4*2 + 1]; \
            int s[4] = {0,0,0,0}; \
            METRALE_MMA_S8(s, a0,a1,a2,a3, b0,b1); \
            acc[nt][0] += (float)s[0]*as0*bs0; acc[nt][1] += (float)s[1]*as0*bs1; \
            acc[nt][2] += (float)s[2]*as1*bs0; acc[nt][3] += (float)s[3]*as1*bs1; \
        } \
    } while(0)


    W83_LOADS(0, 0);  cp_async_commit();
    if (nk > 1) { W83_LOADS(1, 32); cp_async_commit(); }
    CP_WAIT_GROUP(1);
    __syncthreads();

    int cur = 0;
    for (unsigned int ki = 0; ki < nk; ki++) {

        unsigned kn = ki + 2;
        if (kn < nk) { int b = kn % 3; W83_LOADS(b, kn*32); cp_async_commit(); }
        W83_COMPUTE(cur);

        if (ki + 1 < nk) { CP_WAIT_GROUP(1); __syncthreads(); }
        cur = (cur + 1) % 3;
    }
    #undef W83_LOADS
    #undef W83_COMPUTE

    #pragma unroll
    for (int nt = 0; nt < 16; nt++) {
        unsigned c0 = cta_n + nt*8 + t4*2, c1 = c0 + 1;
        unsigned r0 = cta_m + wrow + group_id, r1 = r0 + 8;
        if (r0<M&&c0<N) C[r0*N+c0]=__float2bfloat16(acc[nt][0]);
        if (r0<M&&c1<N) C[r0*N+c1]=__float2bfloat16(acc[nt][1]);
        if (r1<M&&c0<N) C[r1*N+c0]=__float2bfloat16(acc[nt][2]);
        if (r1<M&&c1<N) C[r1*N+c1]=__float2bfloat16(acc[nt][3]);
    }
}
#undef METRALE_MMA_S8
#undef CP_WAIT_GROUP

// 2026-09-25: int8_gemm_8w with the A fragment loaded by one ldmatrix.sync.aligned.m8n8.x4.b16 per step.









#define METRALE_MMA_S8(d, a0,a1,a2,a3, b0,b1) \
    asm volatile("mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 " \
        "{%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%10,%11,%12,%13};" \
        : "=r"((d)[0]), "=r"((d)[1]), "=r"((d)[2]), "=r"((d)[3]) \
        : "r"(a0),"r"(a1),"r"(a2),"r"(a3),"r"(b0),"r"(b1), \
          "r"((d)[0]),"r"((d)[1]),"r"((d)[2]),"r"((d)[3]))

extern "C" __global__
__launch_bounds__(256, 2)
void int8_gemm_8w_ldm(
    const signed char* __restrict__ A_i8,
    const signed char* __restrict__ B_i8,
    const float* __restrict__ A_scale,
    const float* __restrict__ B_scale,
    __nv_bfloat16* __restrict__ C,
    unsigned int M, unsigned int N, unsigned int K
) {
    const unsigned int cta_m = blockIdx.y * 128;
    const unsigned int cta_n = blockIdx.x * 128;
    if (cta_m >= M) return;
    const unsigned int t = threadIdx.x;
    const unsigned int warp_id = t >> 5;
    const unsigned int lane = t & 31;
    const unsigned int group_id = lane >> 2;
    const unsigned int t4 = lane & 3;
    const unsigned int nb = K >> 5;
    const unsigned int wrow = warp_id * 16;

    __shared__ signed char smem_Ai[2][128][32];
    __shared__ signed char smem_Bi[2][128][32];
    __shared__ float smem_As[2][128];
    __shared__ float smem_Bs[2][128];

    float acc[16][4];
    #pragma unroll
    for (int i = 0; i < 16; i++) { acc[i][0]=0.f; acc[i][1]=0.f; acc[i][2]=0.f; acc[i][3]=0.f; }

    #define L_LOADS(buf, kb) do { \
        { unsigned ar = t >> 1; unsigned ac = (t & 1) << 4; unsigned gc = (kb) + ac; unsigned gr = cta_m + ar; \
          cp_async_pred_16(&smem_Ai[(buf)][ar][ac], &A_i8[(unsigned long long)gr*K+gc], (gr<M)&&(gc+15<K)); } \
        { unsigned an = t >> 1; unsigned ac = (t & 1) << 4; unsigned gc = (kb) + ac; unsigned gn = cta_n + an; \
          cp_async_pred_16(&smem_Bi[(buf)][an][ac], &B_i8[(unsigned long long)gn*K+gc], (gn<N)&&(gc+15<K)); } \
        if (t < 128) { unsigned blk=(kb)>>5; unsigned gr=cta_m+t; unsigned gn=cta_n+t; \
          smem_As[(buf)][t] = (gr<M)?A_scale[(unsigned long long)gr*nb+blk]:0.f; \
          smem_Bs[(buf)][t] = (gn<N)?B_scale[(unsigned long long)gn*nb+blk]:0.f; } \
    } while(0)

    #define L_COMPUTE(buf) do { \
        float as0 = smem_As[(buf)][wrow + group_id]; \
        float as1 = smem_As[(buf)][wrow + group_id + 8]; \
        unsigned a0,a1,a2,a3; \
        const int* xs = (const int*)&smem_Ai[(buf)][wrow][0] + (lane % 16)*8 + (lane / 16)*4; \
        asm volatile("ldmatrix.sync.aligned.m8n8.x4.b16 {%0,%1,%2,%3},[%4];" \
            : "=r"(a0),"=r"(a1),"=r"(a2),"=r"(a3) : "l"(xs)); \
        _Pragma("unroll") for (int nt = 0; nt < 16; nt++) { \
            unsigned nc = nt*8 + group_id; \
            unsigned b0 = *(const unsigned*)&smem_Bi[(buf)][nc][4*t4]; \
            unsigned b1 = *(const unsigned*)&smem_Bi[(buf)][nc][16+4*t4]; \
            float bs0 = smem_Bs[(buf)][nt*8 + t4*2]; \
            float bs1 = smem_Bs[(buf)][nt*8 + t4*2 + 1]; \
            int s[4] = {0,0,0,0}; \
            METRALE_MMA_S8(s, a0,a1,a2,a3, b0,b1); \
            acc[nt][0] += (float)s[0]*as0*bs0; acc[nt][1] += (float)s[1]*as0*bs1; \
            acc[nt][2] += (float)s[2]*as1*bs0; acc[nt][3] += (float)s[3]*as1*bs1; \
        } \
    } while(0)

    L_LOADS(0, 0); cp_async_commit(); cp_async_wait_all(); __syncthreads();
    int cur = 0;
    for (unsigned int kb = 32; kb < K; kb += 32) {
        int nxt = 1 - cur;
        L_LOADS(nxt, kb); cp_async_commit();
        L_COMPUTE(cur);
        cp_async_wait_all(); __syncthreads();
        cur = nxt;
    }
    L_COMPUTE(cur);
    #undef L_LOADS
    #undef L_COMPUTE

    #pragma unroll
    for (int nt = 0; nt < 16; nt++) {
        unsigned c0 = cta_n + nt*8 + t4*2, c1 = c0 + 1;
        unsigned r0 = cta_m + wrow + group_id, r1 = r0 + 8;
        if (r0<M&&c0<N) C[r0*N+c0]=__float2bfloat16(acc[nt][0]);
        if (r0<M&&c1<N) C[r0*N+c1]=__float2bfloat16(acc[nt][1]);
        if (r1<M&&c0<N) C[r1*N+c0]=__float2bfloat16(acc[nt][2]);
        if (r1<M&&c1<N) C[r1*N+c1]=__float2bfloat16(acc[nt][3]);
    }
}

// 2026-09-25: int8_gemm_8w_ldm with the B fragments loaded by ldmatrix.x4 as well, two 8-column N tiles per
// load, and two MMAs per load that share the A fragment.










extern "C" __global__
__launch_bounds__(256, 2)
void int8_gemm_8w_ldmab(
    const signed char* __restrict__ A_i8,
    const signed char* __restrict__ B_i8,
    const float* __restrict__ A_scale,
    const float* __restrict__ B_scale,
    __nv_bfloat16* __restrict__ C,
    unsigned int M, unsigned int N, unsigned int K
) {
    const unsigned int cta_m = blockIdx.y * 128;
    const unsigned int cta_n = blockIdx.x * 128;
    if (cta_m >= M) return;
    const unsigned int t = threadIdx.x;
    const unsigned int warp_id = t >> 5;
    const unsigned int lane = t & 31;
    const unsigned int group_id = lane >> 2;
    const unsigned int t4 = lane & 3;
    const unsigned int nb = K >> 5;
    const unsigned int wrow = warp_id * 16;

    __shared__ signed char smem_Ai[2][128][32];
    __shared__ signed char smem_Bi[2][128][32];
    __shared__ float smem_As[2][128];
    __shared__ float smem_Bs[2][128];

    float acc[16][4];
    #pragma unroll
    for (int i = 0; i < 16; i++) { acc[i][0]=0.f; acc[i][1]=0.f; acc[i][2]=0.f; acc[i][3]=0.f; }

    #define LAB_LOADS(buf, kb) do { \
        { unsigned ar = t >> 1; unsigned ac = (t & 1) << 4; unsigned gc = (kb) + ac; unsigned gr = cta_m + ar; \
          cp_async_pred_16(&smem_Ai[(buf)][ar][ac], &A_i8[(unsigned long long)gr*K+gc], (gr<M)&&(gc+15<K)); } \
        { unsigned an = t >> 1; unsigned ac = (t & 1) << 4; unsigned gc = (kb) + ac; unsigned gn = cta_n + an; \
          cp_async_pred_16(&smem_Bi[(buf)][an][ac], &B_i8[(unsigned long long)gn*K+gc], (gn<N)&&(gc+15<K)); } \
        if (t < 128) { unsigned blk=(kb)>>5; unsigned gr=cta_m+t; unsigned gn=cta_n+t; \
          smem_As[(buf)][t] = (gr<M)?A_scale[(unsigned long long)gr*nb+blk]:0.f; \
          smem_Bs[(buf)][t] = (gn<N)?B_scale[(unsigned long long)gn*nb+blk]:0.f; } \
    } while(0)

    #define LAB_COMPUTE(buf) do { \
        float as0 = smem_As[(buf)][wrow + group_id]; \
        float as1 = smem_As[(buf)][wrow + group_id + 8]; \
        unsigned a0,a1,a2,a3; \
        const int* xs = (const int*)&smem_Ai[(buf)][wrow][0] + (lane % 16)*8 + (lane / 16)*4; \
        asm volatile("ldmatrix.sync.aligned.m8n8.x4.b16 {%0,%1,%2,%3},[%4];" \
            : "=r"(a0),"=r"(a1),"=r"(a2),"=r"(a3) : "l"(xs)); \
        _Pragma("unroll") for (int p = 0; p < 8; p++) { \
            unsigned nt0 = 2*p, nt1 = 2*p+1; \
            unsigned brow = ((lane<16)?nt0:nt1)*8 + (lane&7); \
            const void* bxs = &smem_Bi[(buf)][brow][((lane>>3)&1)*16]; \
            unsigned q0,q1,q2,q3; \
            asm volatile("ldmatrix.sync.aligned.m8n8.x4.b16 {%0,%1,%2,%3},[%4];" \
                : "=r"(q0),"=r"(q1),"=r"(q2),"=r"(q3) : "l"(bxs)); \
            float b00 = smem_Bs[(buf)][nt0*8 + t4*2]; float b10 = smem_Bs[(buf)][nt0*8 + t4*2 + 1]; \
            float b01 = smem_Bs[(buf)][nt1*8 + t4*2]; float b11 = smem_Bs[(buf)][nt1*8 + t4*2 + 1]; \
            int s0[4]={0,0,0,0}, s1[4]={0,0,0,0}; \
            METRALE_MMA_S8(s0, a0,a1,a2,a3, q0,q1); \
            METRALE_MMA_S8(s1, a0,a1,a2,a3, q2,q3); \
            acc[nt0][0]+=(float)s0[0]*as0*b00; acc[nt0][1]+=(float)s0[1]*as0*b10; \
            acc[nt0][2]+=(float)s0[2]*as1*b00; acc[nt0][3]+=(float)s0[3]*as1*b10; \
            acc[nt1][0]+=(float)s1[0]*as0*b01; acc[nt1][1]+=(float)s1[1]*as0*b11; \
            acc[nt1][2]+=(float)s1[2]*as1*b01; acc[nt1][3]+=(float)s1[3]*as1*b11; \
        } \
    } while(0)

    LAB_LOADS(0, 0); cp_async_commit(); cp_async_wait_all(); __syncthreads();
    int cur = 0;
    for (unsigned int kb = 32; kb < K; kb += 32) {
        int nxt = 1 - cur;
        LAB_LOADS(nxt, kb); cp_async_commit();
        LAB_COMPUTE(cur);
        cp_async_wait_all(); __syncthreads();
        cur = nxt;
    }
    LAB_COMPUTE(cur);
    #undef LAB_LOADS
    #undef LAB_COMPUTE

    #pragma unroll
    for (int nt = 0; nt < 16; nt++) {
        unsigned c0 = cta_n + nt*8 + t4*2, c1 = c0 + 1;
        unsigned r0 = cta_m + wrow + group_id, r1 = r0 + 8;
        if (r0<M&&c0<N) C[r0*N+c0]=__float2bfloat16(acc[nt][0]);
        if (r0<M&&c1<N) C[r0*N+c1]=__float2bfloat16(acc[nt][1]);
        if (r1<M&&c0<N) C[r1*N+c0]=__float2bfloat16(acc[nt][2]);
        if (r1<M&&c1<N) C[r1*N+c1]=__float2bfloat16(acc[nt][3]);
    }
}
#undef METRALE_MMA_S8

// 2026-09-25: int8_gemm_8w_ldm with all 16 MMAs of a step issued before any scale is applied (int32 results
// in sv[16][4]), and __launch_bounds__(256, 1).








#define METRALE_MMA_S8(d, a0,a1,a2,a3, b0,b1) \
    asm volatile("mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 " \
        "{%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%10,%11,%12,%13};" \
        : "=r"((d)[0]), "=r"((d)[1]), "=r"((d)[2]), "=r"((d)[3]) \
        : "r"(a0),"r"(a1),"r"(a2),"r"(a3),"r"(b0),"r"(b1), \
          "r"((d)[0]),"r"((d)[1]),"r"((d)[2]),"r"((d)[3]))

extern "C" __global__
__launch_bounds__(256, 1)
void int8_gemm_8w_ilp(
    const signed char* __restrict__ A_i8,
    const signed char* __restrict__ B_i8,
    const float* __restrict__ A_scale,
    const float* __restrict__ B_scale,
    __nv_bfloat16* __restrict__ C,
    unsigned int M, unsigned int N, unsigned int K
) {
    const unsigned int cta_m = blockIdx.y * 128;
    const unsigned int cta_n = blockIdx.x * 128;
    if (cta_m >= M) return;
    const unsigned int t = threadIdx.x;
    const unsigned int warp_id = t >> 5;
    const unsigned int lane = t & 31;
    const unsigned int group_id = lane >> 2;
    const unsigned int t4 = lane & 3;
    const unsigned int nb = K >> 5;
    const unsigned int wrow = warp_id * 16;

    __shared__ signed char smem_Ai[2][128][32];
    __shared__ signed char smem_Bi[2][128][32];
    __shared__ float smem_As[2][128];
    __shared__ float smem_Bs[2][128];

    float acc[16][4];
    #pragma unroll
    for (int i = 0; i < 16; i++) { acc[i][0]=0.f; acc[i][1]=0.f; acc[i][2]=0.f; acc[i][3]=0.f; }

    #define L_LOADS(buf, kb) do { \
        { unsigned ar = t >> 1; unsigned ac = (t & 1) << 4; unsigned gc = (kb) + ac; unsigned gr = cta_m + ar; \
          cp_async_pred_16(&smem_Ai[(buf)][ar][ac], &A_i8[(unsigned long long)gr*K+gc], (gr<M)&&(gc+15<K)); } \
        { unsigned an = t >> 1; unsigned ac = (t & 1) << 4; unsigned gc = (kb) + ac; unsigned gn = cta_n + an; \
          cp_async_pred_16(&smem_Bi[(buf)][an][ac], &B_i8[(unsigned long long)gn*K+gc], (gn<N)&&(gc+15<K)); } \
        if (t < 128) { unsigned blk=(kb)>>5; unsigned gr=cta_m+t; unsigned gn=cta_n+t; \
          smem_As[(buf)][t] = (gr<M)?A_scale[(unsigned long long)gr*nb+blk]:0.f; \
          smem_Bs[(buf)][t] = (gn<N)?B_scale[(unsigned long long)gn*nb+blk]:0.f; } \
    } while(0)

    #define L_COMPUTE(buf) do { \
        float as0 = smem_As[(buf)][wrow + group_id]; \
        float as1 = smem_As[(buf)][wrow + group_id + 8]; \
        unsigned a0,a1,a2,a3; \
        const int* xs = (const int*)&smem_Ai[(buf)][wrow][0] + (lane % 16)*8 + (lane / 16)*4; \
        asm volatile("ldmatrix.sync.aligned.m8n8.x4.b16 {%0,%1,%2,%3},[%4];" \
            : "=r"(a0),"=r"(a1),"=r"(a2),"=r"(a3) : "l"(xs)); \
        int sv[16][4]; \
        _Pragma("unroll") for (int nt = 0; nt < 16; nt++) { \
            unsigned nc = nt*8 + group_id; \
            unsigned b0 = *(const unsigned*)&smem_Bi[(buf)][nc][4*t4]; \
            unsigned b1 = *(const unsigned*)&smem_Bi[(buf)][nc][16+4*t4]; \
            sv[nt][0]=0; sv[nt][1]=0; sv[nt][2]=0; sv[nt][3]=0; \
            METRALE_MMA_S8(sv[nt], a0,a1,a2,a3, b0,b1); \
        } \
        _Pragma("unroll") for (int nt = 0; nt < 16; nt++) { \
            float bs0 = smem_Bs[(buf)][nt*8 + t4*2]; \
            float bs1 = smem_Bs[(buf)][nt*8 + t4*2 + 1]; \
            acc[nt][0] += (float)sv[nt][0]*as0*bs0; acc[nt][1] += (float)sv[nt][1]*as0*bs1; \
            acc[nt][2] += (float)sv[nt][2]*as1*bs0; acc[nt][3] += (float)sv[nt][3]*as1*bs1; \
        } \
    } while(0)

    L_LOADS(0, 0); cp_async_commit(); cp_async_wait_all(); __syncthreads();
    int cur = 0;
    for (unsigned int kb = 32; kb < K; kb += 32) {
        int nxt = 1 - cur;
        L_LOADS(nxt, kb); cp_async_commit();
        L_COMPUTE(cur);
        cp_async_wait_all(); __syncthreads();
        cur = nxt;
    }
    L_COMPUTE(cur);
    #undef L_LOADS
    #undef L_COMPUTE

    #pragma unroll
    for (int nt = 0; nt < 16; nt++) {
        unsigned c0 = cta_n + nt*8 + t4*2, c1 = c0 + 1;
        unsigned r0 = cta_m + wrow + group_id, r1 = r0 + 8;
        if (r0<M&&c0<N) C[r0*N+c0]=__float2bfloat16(acc[nt][0]);
        if (r0<M&&c1<N) C[r0*N+c1]=__float2bfloat16(acc[nt][1]);
        if (r1<M&&c0<N) C[r1*N+c0]=__float2bfloat16(acc[nt][2]);
        if (r1<M&&c1<N) C[r1*N+c1]=__float2bfloat16(acc[nt][3]);
    }
}
#undef METRALE_MMA_S8

// 2026-09-25: int8_gemm_8w_ldm with a 128-wide K tile loaded once per outer step, its four 32-wide sub-blocks
// multiplied from shared memory with no barrier between them. K must be a multiple of 128.









#define MMQ_BK 128
#define METRALE_MMA_S8(d, a0,a1,a2,a3, b0,b1) \
    asm volatile("mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 " \
        "{%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%10,%11,%12,%13};" \
        : "=r"((d)[0]), "=r"((d)[1]), "=r"((d)[2]), "=r"((d)[3]) \
        : "r"(a0),"r"(a1),"r"(a2),"r"(a3),"r"(b0),"r"(b1), \
          "r"((d)[0]),"r"((d)[1]),"r"((d)[2]),"r"((d)[3]))

extern "C" __global__
__launch_bounds__(256, 2)
void int8_gemm_mmq(
    const signed char* __restrict__ A_i8,
    const signed char* __restrict__ B_i8,
    const float* __restrict__ A_scale,
    const float* __restrict__ B_scale,
    __nv_bfloat16* __restrict__ C,
    unsigned int M, unsigned int N, unsigned int K
) {
    const unsigned int cta_m = blockIdx.y * 128;
    const unsigned int cta_n = blockIdx.x * 128;
    if (cta_m >= M) return;
    const unsigned int t = threadIdx.x;
    const unsigned int warp_id = t >> 5;
    const unsigned int lane = t & 31;
    const unsigned int group_id = lane >> 2;
    const unsigned int t4 = lane & 3;
    const unsigned int nb = K >> 5;
    const unsigned int wrow = warp_id * 16;

    __shared__ signed char sA[128][MMQ_BK];
    __shared__ signed char sB[128][MMQ_BK];
    __shared__ float sAs[128][MMQ_BK/32];
    __shared__ float sBs[128][MMQ_BK/32];

    float acc[16][4];
    #pragma unroll
    for (int i = 0; i < 16; i++) { acc[i][0]=0.f; acc[i][1]=0.f; acc[i][2]=0.f; acc[i][3]=0.f; }

    for (unsigned int kb0 = 0; kb0 < K; kb0 += MMQ_BK) {

        #pragma unroll
        for (int c = 0; c < 4; c++) {
            unsigned lin = c*256 + t;
            unsigned row = lin >> 3;
            unsigned col = (lin & 7) << 4;
            unsigned gcA = kb0 + col, gcB = kb0 + col;
            cp_async_pred_16(&sA[row][col], &A_i8[(unsigned long long)(cta_m+row)*K + gcA], (cta_m+row<M)&&(gcA+15<K));
            cp_async_pred_16(&sB[row][col], &B_i8[(unsigned long long)(cta_n+row)*K + gcB], (cta_n+row<N)&&(gcB+15<K));
        }
        if (t < 128) {
            unsigned blk0 = kb0 >> 5;
            #pragma unroll
            for (int b = 0; b < MMQ_BK/32; b++) {
                unsigned gr = cta_m + t, gn = cta_n + t;
                sAs[t][b] = (gr<M)?A_scale[(unsigned long long)gr*nb + blk0 + b]:0.f;
                sBs[t][b] = (gn<N)?B_scale[(unsigned long long)gn*nb + blk0 + b]:0.f;
            }
        }
        cp_async_commit();
        cp_async_wait_all();
        __syncthreads();


        #pragma unroll
        for (int sb = 0; sb < MMQ_BK/32; sb++) {
            float as0 = sAs[wrow + group_id][sb];
            float as1 = sAs[wrow + group_id + 8][sb];
            unsigned a0,a1,a2,a3;
            const int* xs = (const int*)&sA[wrow][0] + (lane % 16)*(MMQ_BK/4) + sb*8 + (lane / 16)*4;
            asm volatile("ldmatrix.sync.aligned.m8n8.x4.b16 {%0,%1,%2,%3},[%4];"
                : "=r"(a0),"=r"(a1),"=r"(a2),"=r"(a3) : "l"(xs));
            #pragma unroll
            for (int nt = 0; nt < 16; nt++) {
                unsigned nc = nt*8 + group_id;
                unsigned b0 = *(const unsigned*)&sB[nc][sb*32 + 4*t4];
                unsigned b1 = *(const unsigned*)&sB[nc][sb*32 + 16 + 4*t4];
                float bs0 = sBs[nt*8 + t4*2][sb];
                float bs1 = sBs[nt*8 + t4*2 + 1][sb];
                int s[4] = {0,0,0,0};
                METRALE_MMA_S8(s, a0,a1,a2,a3, b0,b1);
                acc[nt][0] += (float)s[0]*as0*bs0; acc[nt][1] += (float)s[1]*as0*bs1;
                acc[nt][2] += (float)s[2]*as1*bs0; acc[nt][3] += (float)s[3]*as1*bs1;
            }
        }
        __syncthreads();  // 2026-09-25: the next tile's loads overwrite sA and sB
    }

    #pragma unroll
    for (int nt = 0; nt < 16; nt++) {
        unsigned c0 = cta_n + nt*8 + t4*2, c1 = c0 + 1;
        unsigned r0 = cta_m + wrow + group_id, r1 = r0 + 8;
        if (r0<M&&c0<N) C[r0*N+c0]=__float2bfloat16(acc[nt][0]);
        if (r0<M&&c1<N) C[r0*N+c1]=__float2bfloat16(acc[nt][1]);
        if (r1<M&&c0<N) C[r1*N+c0]=__float2bfloat16(acc[nt][2]);
        if (r1<M&&c1<N) C[r1*N+c1]=__float2bfloat16(acc[nt][3]);
    }
}
#undef METRALE_MMA_S8
#undef MMQ_BK

// 2026-09-25: int8_gemm_8w_ldmab on 512 threads: sixteen warps, each owning 16 rows by 64 columns
// (warp_id & 7 picks the rows, warp_id >> 3 the column half).







#define METRALE_MMA_S8(d, a0,a1,a2,a3, b0,b1) \
    asm volatile("mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 " \
        "{%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%10,%11,%12,%13};" \
        : "=r"((d)[0]), "=r"((d)[1]), "=r"((d)[2]), "=r"((d)[3]) \
        : "r"(a0),"r"(a1),"r"(a2),"r"(a3),"r"(b0),"r"(b1), \
          "r"((d)[0]),"r"((d)[1]),"r"((d)[2]),"r"((d)[3]))

extern "C" __global__
__launch_bounds__(512, 2)
void int8_gemm_8w_pipe(
    const signed char* __restrict__ A_i8,
    const signed char* __restrict__ B_i8,
    const float* __restrict__ A_scale,
    const float* __restrict__ B_scale,
    __nv_bfloat16* __restrict__ C,
    unsigned int M, unsigned int N, unsigned int K
) {
    const unsigned int cta_m = blockIdx.y * 128;
    const unsigned int cta_n = blockIdx.x * 128;
    if (cta_m >= M) return;
    const unsigned int t = threadIdx.x;
    const unsigned int warp_id = t >> 5;
    const unsigned int lane = t & 31;
    const unsigned int group_id = lane >> 2;
    const unsigned int t4 = lane & 3;
    const unsigned int nb = K >> 5;
    const unsigned int wm = warp_id & 7;
    const unsigned int wn = warp_id >> 3;
    const unsigned int wrow = wm * 16;
    const unsigned int ncol0 = wn * 64;

    __shared__ signed char smem_Ai[2][128][32];
    __shared__ signed char smem_Bi[2][128][32];
    __shared__ float smem_As[2][128];
    __shared__ float smem_Bs[2][128];

    float acc[8][4];
    #pragma unroll
    for (int i = 0; i < 8; i++) { acc[i][0]=0.f; acc[i][1]=0.f; acc[i][2]=0.f; acc[i][3]=0.f; }


    #define P_LOADS(buf, kb) do { \
        if (t < 256) { unsigned ar = t >> 1; unsigned ac = (t & 1) << 4; unsigned gc = (kb) + ac; unsigned gr = cta_m + ar; \
          cp_async_pred_16(&smem_Ai[(buf)][ar][ac], &A_i8[(unsigned long long)gr*K+gc], (gr<M)&&(gc+15<K)); } \
        else { unsigned u = t - 256; unsigned an = u >> 1; unsigned ac = (u & 1) << 4; unsigned gc = (kb) + ac; unsigned gn = cta_n + an; \
          cp_async_pred_16(&smem_Bi[(buf)][an][ac], &B_i8[(unsigned long long)gn*K+gc], (gn<N)&&(gc+15<K)); } \
        if (t < 128) { unsigned blk=(kb)>>5; unsigned gr=cta_m+t; smem_As[(buf)][t] = (gr<M)?A_scale[(unsigned long long)gr*nb+blk]:0.f; } \
        else if (t < 256) { unsigned blk=(kb)>>5; unsigned gn=cta_n+(t-128); smem_Bs[(buf)][t-128] = (gn<N)?B_scale[(unsigned long long)gn*nb+blk]:0.f; } \
    } while(0)

    #define P_COMPUTE(buf) do { \
        float as0 = smem_As[(buf)][wrow + group_id]; \
        float as1 = smem_As[(buf)][wrow + group_id + 8]; \
        unsigned a0,a1,a2,a3; \
        const int* xs = (const int*)&smem_Ai[(buf)][wrow][0] + (lane % 16)*8 + (lane / 16)*4; \
        asm volatile("ldmatrix.sync.aligned.m8n8.x4.b16 {%0,%1,%2,%3},[%4];" \
            : "=r"(a0),"=r"(a1),"=r"(a2),"=r"(a3) : "l"(xs)); \
        _Pragma("unroll") for (int p = 0; p < 4; p++) { \
            unsigned nt0 = 2*p, nt1 = 2*p+1; \
            unsigned brow = ncol0 + ((lane<16)?nt0:nt1)*8 + (lane&7); \
            const void* bxs = &smem_Bi[(buf)][brow][((lane>>3)&1)*16]; \
            unsigned q0,q1,q2,q3; \
            asm volatile("ldmatrix.sync.aligned.m8n8.x4.b16 {%0,%1,%2,%3},[%4];" \
                : "=r"(q0),"=r"(q1),"=r"(q2),"=r"(q3) : "l"(bxs)); \
            float b00 = smem_Bs[(buf)][ncol0 + nt0*8 + t4*2]; float b10 = smem_Bs[(buf)][ncol0 + nt0*8 + t4*2 + 1]; \
            float b01 = smem_Bs[(buf)][ncol0 + nt1*8 + t4*2]; float b11 = smem_Bs[(buf)][ncol0 + nt1*8 + t4*2 + 1]; \
            int s0[4]={0,0,0,0}, s1[4]={0,0,0,0}; \
            METRALE_MMA_S8(s0, a0,a1,a2,a3, q0,q1); \
            METRALE_MMA_S8(s1, a0,a1,a2,a3, q2,q3); \
            acc[nt0][0]+=(float)s0[0]*as0*b00; acc[nt0][1]+=(float)s0[1]*as0*b10; \
            acc[nt0][2]+=(float)s0[2]*as1*b00; acc[nt0][3]+=(float)s0[3]*as1*b10; \
            acc[nt1][0]+=(float)s1[0]*as0*b01; acc[nt1][1]+=(float)s1[1]*as0*b11; \
            acc[nt1][2]+=(float)s1[2]*as1*b01; acc[nt1][3]+=(float)s1[3]*as1*b11; \
        } \
    } while(0)

    P_LOADS(0, 0); cp_async_commit(); cp_async_wait_all(); __syncthreads();
    int cur = 0;
    for (unsigned int kb = 32; kb < K; kb += 32) {
        int nxt = 1 - cur;
        P_LOADS(nxt, kb); cp_async_commit();
        P_COMPUTE(cur);
        cp_async_wait_all(); __syncthreads();
        cur = nxt;
    }
    P_COMPUTE(cur);
    #undef P_LOADS
    #undef P_COMPUTE

    #pragma unroll
    for (int nt = 0; nt < 8; nt++) {
        unsigned c0 = cta_n + ncol0 + nt*8 + t4*2, c1 = c0 + 1;
        unsigned r0 = cta_m + wrow + group_id, r1 = r0 + 8;
        if (r0<M&&c0<N) C[r0*N+c0]=__float2bfloat16(acc[nt][0]);
        if (r0<M&&c1<N) C[r0*N+c1]=__float2bfloat16(acc[nt][1]);
        if (r1<M&&c0<N) C[r1*N+c0]=__float2bfloat16(acc[nt][2]);
        if (r1<M&&c1<N) C[r1*N+c1]=__float2bfloat16(acc[nt][3]);
    }
}
#undef METRALE_MMA_S8

// 2026-09-25: int8_gemm_8w_ldm with A rows padded to PADI = 12 int32 (48 bytes, three 16-byte units), so the
// eight row addresses of an ldmatrix 8x8 matrix fall in distinct 16-byte bank groups (r * 3 mod 8 is a
// permutation). Otherwise the same as int8_gemm_8w_ldm.







#define PADI 12

#define METRALE_MMA_S8(d, a0,a1,a2,a3, b0,b1) \
    asm volatile("mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 " \
        "{%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%10,%11,%12,%13};" \
        : "=r"((d)[0]), "=r"((d)[1]), "=r"((d)[2]), "=r"((d)[3]) \
        : "r"(a0),"r"(a1),"r"(a2),"r"(a3),"r"(b0),"r"(b1), \
          "r"((d)[0]),"r"((d)[1]),"r"((d)[2]),"r"((d)[3]))

extern "C" __global__
__launch_bounds__(256, 2)
void int8_gemm_padA(
    const signed char* __restrict__ A_i8,
    const signed char* __restrict__ B_i8,
    const float* __restrict__ A_scale,
    const float* __restrict__ B_scale,
    __nv_bfloat16* __restrict__ C,
    unsigned int M, unsigned int N, unsigned int K
) {
    const unsigned int cta_m = blockIdx.y * 128;
    const unsigned int cta_n = blockIdx.x * 128;
    if (cta_m >= M) return;
    const unsigned int t = threadIdx.x;
    const unsigned int warp_id = t >> 5;
    const unsigned int lane = t & 31;
    const unsigned int group_id = lane >> 2;
    const unsigned int t4 = lane & 3;
    const unsigned int nb = K >> 5;
    const unsigned int wrow = warp_id * 16;

    __shared__ int   smem_Ai[2][128][PADI];
    __shared__ signed char smem_Bi[2][128][32];
    __shared__ float smem_As[2][128];
    __shared__ float smem_Bs[2][128];

    float acc[16][4];
    #pragma unroll
    for (int i = 0; i < 16; i++) { acc[i][0]=0.f; acc[i][1]=0.f; acc[i][2]=0.f; acc[i][3]=0.f; }

    #define PA_LOADS(buf, kb) do { \
        { unsigned ar = t >> 1; unsigned ac = (t & 1) << 4; unsigned gc = (kb) + ac; unsigned gr = cta_m + ar; \
          cp_async_pred_16(((signed char*)&smem_Ai[(buf)][ar][0]) + ac, &A_i8[(unsigned long long)gr*K+gc], (gr<M)&&(gc+15<K)); } \
        { unsigned an = t >> 1; unsigned ac = (t & 1) << 4; unsigned gc = (kb) + ac; unsigned gn = cta_n + an; \
          cp_async_pred_16(&smem_Bi[(buf)][an][ac], &B_i8[(unsigned long long)gn*K+gc], (gn<N)&&(gc+15<K)); } \
        if (t < 128) { unsigned blk=(kb)>>5; unsigned gr=cta_m+t; unsigned gn=cta_n+t; \
          smem_As[(buf)][t] = (gr<M)?A_scale[(unsigned long long)gr*nb+blk]:0.f; \
          smem_Bs[(buf)][t] = (gn<N)?B_scale[(unsigned long long)gn*nb+blk]:0.f; } \
    } while(0)

    #define PA_COMPUTE(buf) do { \
        float as0 = smem_As[(buf)][wrow + group_id]; \
        float as1 = smem_As[(buf)][wrow + group_id + 8]; \
        unsigned a0,a1,a2,a3; \
        const int* xs = &smem_Ai[(buf)][wrow][0] + (lane % 16)*PADI + (lane / 16)*4; \
        asm volatile("ldmatrix.sync.aligned.m8n8.x4.b16 {%0,%1,%2,%3},[%4];" \
            : "=r"(a0),"=r"(a1),"=r"(a2),"=r"(a3) : "l"(xs)); \
        _Pragma("unroll") for (int nt = 0; nt < 16; nt++) { \
            unsigned nc = nt*8 + group_id; \
            unsigned b0 = *(const unsigned*)&smem_Bi[(buf)][nc][4*t4]; \
            unsigned b1 = *(const unsigned*)&smem_Bi[(buf)][nc][16+4*t4]; \
            float bs0 = smem_Bs[(buf)][nt*8 + t4*2]; \
            float bs1 = smem_Bs[(buf)][nt*8 + t4*2 + 1]; \
            int s[4] = {0,0,0,0}; \
            METRALE_MMA_S8(s, a0,a1,a2,a3, b0,b1); \
            acc[nt][0] += (float)s[0]*as0*bs0; acc[nt][1] += (float)s[1]*as0*bs1; \
            acc[nt][2] += (float)s[2]*as1*bs0; acc[nt][3] += (float)s[3]*as1*bs1; \
        } \
    } while(0)

    PA_LOADS(0, 0); cp_async_commit(); cp_async_wait_all(); __syncthreads();
    int cur = 0;
    for (unsigned int kb = 32; kb < K; kb += 32) {
        int nxt = 1 - cur;
        PA_LOADS(nxt, kb); cp_async_commit();
        PA_COMPUTE(cur);
        cp_async_wait_all(); __syncthreads();
        cur = nxt;
    }
    PA_COMPUTE(cur);
    #undef PA_LOADS
    #undef PA_COMPUTE

    #pragma unroll
    for (int nt = 0; nt < 16; nt++) {
        unsigned c0 = cta_n + nt*8 + t4*2, c1 = c0 + 1;
        unsigned r0 = cta_m + wrow + group_id, r1 = r0 + 8;
        if (r0<M&&c0<N) C[r0*N+c0]=__float2bfloat16(acc[nt][0]);
        if (r0<M&&c1<N) C[r0*N+c1]=__float2bfloat16(acc[nt][1]);
        if (r1<M&&c0<N) C[r1*N+c0]=__float2bfloat16(acc[nt][2]);
        if (r1<M&&c1<N) C[r1*N+c1]=__float2bfloat16(acc[nt][3]);
    }
}
#undef METRALE_MMA_S8
#undef PADI

// 2026-09-25: int8 GEMM with the weight as the ldmatrix (row) operand and the activation as the scalar-loaded
// (column) operand, so the MMA tile is transposed: each warp owns 32 weight rows by 64 tokens. K tile 64 (two
// 32-wide sub-blocks) loaded once per step, and a step's weight fragments and scales are held in registers
// before its MMAs. Per-block scales as in int8_gemm_t_m128; grid (ceil(N/128), ceil(M/128)), 256 threads.
// K must be a multiple of 64.








#define FK_TILE 64
#define FK_SB   (FK_TILE/32)
#define FW_STRIDE 36
#define METRALE_MMA_S8(d, a0,a1,a2,a3, b0,b1) \
    asm volatile("mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 " \
        "{%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%10,%11,%12,%13};" \
        : "=r"((d)[0]), "=r"((d)[1]), "=r"((d)[2]), "=r"((d)[3]) \
        : "r"(a0),"r"(a1),"r"(a2),"r"(a3),"r"(b0),"r"(b1), \
          "r"((d)[0]),"r"((d)[1]),"r"((d)[2]),"r"((d)[3]))

extern "C" __global__
__launch_bounds__(256, 1)
void int8_gemm_faith(
    const signed char* __restrict__ A_i8,
    const signed char* __restrict__ B_i8,
    const float* __restrict__ A_scale,
    const float* __restrict__ B_scale,
    __nv_bfloat16* __restrict__ C,
    unsigned int M, unsigned int N, unsigned int K
) {
    const unsigned int cta_n = blockIdx.x * 128;
    const unsigned int cta_m = blockIdx.y * 128;
    if (cta_m >= M || cta_n >= N) return;
    const unsigned int t = threadIdx.x;
    const unsigned int warp_id = t >> 5;
    const unsigned int lane = t & 31;
    const unsigned int ng = warp_id >> 1;
    const unsigned int mh = warp_id & 1;
    const unsigned int nb = K >> 5;

    // 2026-09-25: Rows of 20 int32 (16 data + 4 pad): r * 5 mod 8 is a permutation, so the eight rows of an
    // ldmatrix 8x8 matrix fall in distinct 16-byte bank groups.

    __shared__ int  sW[128][20];
    __shared__ int  sA[128][20];
    __shared__ float sWs[128][FK_SB];
    __shared__ float sAs[128][FK_SB];

    float acc[2][8][4];
    #pragma unroll
    for (int n=0;n<2;n++) for(int j=0;j<8;j++){acc[n][j][0]=0;acc[n][j][1]=0;acc[n][j][2]=0;acc[n][j][3]=0;}

    for (unsigned int kb = 0; kb < K; kb += FK_TILE) {

        #pragma unroll
        for (int c = 0; c < 2; c++) {
            unsigned lin = c*256 + t;
            unsigned row = lin >> 2;
            unsigned col = (lin & 3) << 4;
            unsigned gk = kb + col;
            signed char* wdst = ((signed char*)&sW[row][0]) + col;
            signed char* adst = ((signed char*)&sA[row][0]) + col;
            cp_async_pred_16(wdst, &B_i8[(unsigned long long)(cta_n+row)*K + gk], (cta_n+row<N)&&(gk+15<K));
            cp_async_pred_16(adst, &A_i8[(unsigned long long)(cta_m+row)*K + gk], (cta_m+row<M)&&(gk+15<K));
        }
        if (t < 128) {
            unsigned blk = kb >> 5;
            #pragma unroll
            for (int s=0;s<FK_SB;s++){
                sWs[t][s] = (cta_n+t<N)?B_scale[(unsigned long long)(cta_n+t)*nb + blk + s]:0.f;
                sAs[t][s] = (cta_m+t<M)?A_scale[(unsigned long long)(cta_m+t)*nb + blk + s]:0.f;
            }
        }
        cp_async_commit(); cp_async_wait_all(); __syncthreads();


        unsigned WA[2][FK_SB][4];
        float wsc[2][2][FK_SB];
        #pragma unroll
        for (int n=0;n<2;n++){
            unsigned wbase_row = ng*32 + n*16;
            #pragma unroll
            for (int sb=0; sb<FK_SB; sb++){
                const int* xs = &sW[wbase_row][0] + (lane%16)*20 + sb*8 + (lane/16)*4;
                asm volatile("ldmatrix.sync.aligned.m8n8.x4.b16 {%0,%1,%2,%3},[%4];"
                    : "=r"(WA[n][sb][0]),"=r"(WA[n][sb][1]),"=r"(WA[n][sb][2]),"=r"(WA[n][sb][3]) : "l"(xs));
                wsc[n][0][sb] = sWs[wbase_row + lane/4][sb];
                wsc[n][1][sb] = sWs[wbase_row + 8 + lane/4][sb];
            }
        }

        #pragma unroll
        for (int j=0;j<8;j++){
            unsigned mcol0 = mh*64 + j*8;
            float asc[2][FK_SB];
            #pragma unroll
            for (int sb=0;sb<FK_SB;sb++){
                asc[0][sb] = sAs[mcol0 + (lane%4)*2][sb];
                asc[1][sb] = sAs[mcol0 + (lane%4)*2 + 1][sb];
            }
            #pragma unroll
            for (int sb=0;sb<FK_SB;sb++){

                const int* abase = &sA[mcol0 + lane/4][0] + sb*8;
                unsigned b0 = abase[lane%4];
                unsigned b1 = abase[lane%4 + 4];
                #pragma unroll
                for (int n=0;n<2;n++){
                    int s[4]={0,0,0,0};
                    METRALE_MMA_S8(s, WA[n][sb][0],WA[n][sb][1],WA[n][sb][2],WA[n][sb][3], b0,b1);
                    acc[n][j][0]+=(float)s[0]*wsc[n][0][sb]*asc[0][sb];
                    acc[n][j][1]+=(float)s[1]*wsc[n][0][sb]*asc[1][sb];
                    acc[n][j][2]+=(float)s[2]*wsc[n][1][sb]*asc[0][sb];
                    acc[n][j][3]+=(float)s[3]*wsc[n][1][sb]*asc[1][sb];
                }
            }
        }
        __syncthreads();
    }

    // 2026-09-25: The MMA tile is transposed (rows are N features, columns M tokens); store to C[m, n].
    #pragma unroll
    for (int n=0;n<2;n++){
        unsigned nrow0 = cta_n + ng*32 + n*16 + lane/4;
        #pragma unroll
        for (int j=0;j<8;j++){
            unsigned mcol = cta_m + mh*64 + j*8 + (lane%4)*2;
            unsigned cN0=nrow0, cN1=nrow0+8;


            if (mcol<M   && cN0<N) C[(unsigned long long)mcol*N + cN0]     = __float2bfloat16(acc[n][j][0]);
            if (mcol+1<M && cN0<N) C[(unsigned long long)(mcol+1)*N + cN0] = __float2bfloat16(acc[n][j][1]);
            if (mcol<M   && cN1<N) C[(unsigned long long)mcol*N + cN1]     = __float2bfloat16(acc[n][j][2]);
            if (mcol+1<M && cN1<N) C[(unsigned long long)(mcol+1)*N + cN1] = __float2bfloat16(acc[n][j][3]);
        }
    }
}
#undef METRALE_MMA_S8
#undef FK_TILE
#undef FK_SB
#undef FW_STRIDE

// 2026-09-25: int8_gemm_faith with a K tile of 128 (four 32-wide sub-blocks) and the sub-block loop outside the
// token loop, so only one sub-block's weight fragments are held in registers at a time. Rows are F2W = 36 int32
// (32 data + 4 pad): r * 9 mod 8 is a permutation, so the eight rows of an ldmatrix 8x8 matrix fall in distinct
// 16-byte bank groups. K must be a multiple of 128. The dense-FFN prefill runs it when METRALE_INT8_PREFILL
// is set (model-layers dense_ffn.rs), after requant_w_nvfp4_int8 and requant_a_bf16_int8.










#define F2_TILE 128
#define F2_SB   (F2_TILE/32)
#define F2W     36
#define METRALE_MMA_S8(d, a0,a1,a2,a3, b0,b1) \
    asm volatile("mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 " \
        "{%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%10,%11,%12,%13};" \
        : "=r"((d)[0]), "=r"((d)[1]), "=r"((d)[2]), "=r"((d)[3]) \
        : "r"(a0),"r"(a1),"r"(a2),"r"(a3),"r"(b0),"r"(b1), \
          "r"((d)[0]),"r"((d)[1]),"r"((d)[2]),"r"((d)[3]))

extern "C" __global__
__launch_bounds__(256, 1)
void int8_gemm_faith2(
    const signed char* __restrict__ A_i8,
    const signed char* __restrict__ B_i8,
    const float* __restrict__ A_scale,
    const float* __restrict__ B_scale,
    __nv_bfloat16* __restrict__ C,
    unsigned int M, unsigned int N, unsigned int K
) {
    const unsigned int cta_n = blockIdx.x * 128;
    const unsigned int cta_m = blockIdx.y * 128;
    if (cta_m >= M || cta_n >= N) return;
    const unsigned int t = threadIdx.x;
    const unsigned int warp_id = t >> 5;
    const unsigned int lane = t & 31;
    const unsigned int ng = warp_id >> 1;
    const unsigned int mh = warp_id & 1;
    const unsigned int nb = K >> 5;

    __shared__ int   sW[128][F2W];
    __shared__ int   sA[128][F2W];
    __shared__ float sWs[128][F2_SB];
    __shared__ float sAs[128][F2_SB];

    float acc[2][8][4];
    #pragma unroll
    for (int n=0;n<2;n++) for(int j=0;j<8;j++){acc[n][j][0]=0;acc[n][j][1]=0;acc[n][j][2]=0;acc[n][j][3]=0;}

    for (unsigned int kb = 0; kb < K; kb += F2_TILE) {


        const unsigned F2_CPR = F2_TILE/16;
        #pragma unroll
        for (int c = 0; c < F2_TILE/32; c++) {
            unsigned lin = c*256 + t;
            unsigned row = lin / F2_CPR;
            unsigned col = (lin % F2_CPR) << 4;
            unsigned gk = kb + col;
            signed char* wdst = ((signed char*)&sW[row][0]) + col;
            signed char* adst = ((signed char*)&sA[row][0]) + col;
            cp_async_pred_16(wdst, &B_i8[(unsigned long long)(cta_n+row)*K + gk], (cta_n+row<N)&&(gk+15<K));
            cp_async_pred_16(adst, &A_i8[(unsigned long long)(cta_m+row)*K + gk], (cta_m+row<M)&&(gk+15<K));
        }
        if (t < 128) {
            unsigned blk = kb >> 5;
            #pragma unroll
            for (int s=0;s<F2_SB;s++){
                sWs[t][s] = (cta_n+t<N)?B_scale[(unsigned long long)(cta_n+t)*nb + blk + s]:0.f;
                sAs[t][s] = (cta_m+t<M)?A_scale[(unsigned long long)(cta_m+t)*nb + blk + s]:0.f;
            }
        }
        cp_async_commit(); cp_async_wait_all(); __syncthreads();


        #pragma unroll
        for (int sb=0; sb<F2_SB; sb++){
            unsigned WA[2][4];
            float    wsc[2][2];
            #pragma unroll
            for (int n=0;n<2;n++){
                unsigned wbase_row = ng*32 + n*16;
                const int* xs = &sW[wbase_row][0] + (lane%16)*F2W + sb*8 + (lane/16)*4;
                asm volatile("ldmatrix.sync.aligned.m8n8.x4.b16 {%0,%1,%2,%3},[%4];"
                    : "=r"(WA[n][0]),"=r"(WA[n][1]),"=r"(WA[n][2]),"=r"(WA[n][3]) : "l"(xs));
                wsc[n][0] = sWs[wbase_row + lane/4][sb];
                wsc[n][1] = sWs[wbase_row + 8 + lane/4][sb];
            }
            #pragma unroll
            for (int j=0;j<8;j++){
                unsigned mcol0 = mh*64 + j*8;
                float asc0 = sAs[mcol0 + (lane%4)*2][sb];
                float asc1 = sAs[mcol0 + (lane%4)*2 + 1][sb];
                const int* abase = &sA[mcol0 + lane/4][0] + sb*8;
                unsigned b0 = abase[lane%4];
                unsigned b1 = abase[lane%4 + 4];
                #pragma unroll
                for (int n=0;n<2;n++){
                    int s[4]={0,0,0,0};
                    METRALE_MMA_S8(s, WA[n][0],WA[n][1],WA[n][2],WA[n][3], b0,b1);
                    acc[n][j][0]+=(float)s[0]*wsc[n][0]*asc0;
                    acc[n][j][1]+=(float)s[1]*wsc[n][0]*asc1;
                    acc[n][j][2]+=(float)s[2]*wsc[n][1]*asc0;
                    acc[n][j][3]+=(float)s[3]*wsc[n][1]*asc1;
                }
            }
        }
        __syncthreads();
    }

    #pragma unroll
    for (int n=0;n<2;n++){
        unsigned nrow0 = cta_n + ng*32 + n*16 + lane/4;
        #pragma unroll
        for (int j=0;j<8;j++){
            unsigned mcol = cta_m + mh*64 + j*8 + (lane%4)*2;
            unsigned cN0=nrow0, cN1=nrow0+8;
            if (mcol<M   && cN0<N) C[(unsigned long long)mcol*N + cN0]     = __float2bfloat16(acc[n][j][0]);
            if (mcol+1<M && cN0<N) C[(unsigned long long)(mcol+1)*N + cN0] = __float2bfloat16(acc[n][j][1]);
            if (mcol<M   && cN1<N) C[(unsigned long long)mcol*N + cN1]     = __float2bfloat16(acc[n][j][2]);
            if (mcol+1<M && cN1<N) C[(unsigned long long)(mcol+1)*N + cN1] = __float2bfloat16(acc[n][j][3]);
        }
    }
}
#undef METRALE_MMA_S8
#undef F2_TILE
#undef F2_SB
#undef F2W

// 2026-09-25: int8_gemm_faith2 with the scale step taken out of the MMA loop: a sub-block's 16 int32 MMA results
// are kept in iacc and scaled into the FP32 accumulator facc after all of them are issued.

















#define I32A_TILE 128
#define I32A_SB   (I32A_TILE/32)
#define I32AW     36
#define METRALE_MMA_S8_I32A(d, a0,a1,a2,a3, b0,b1) \
    asm volatile("mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 " \
        "{%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%10,%11,%12,%13};" \
        : "=r"((d)[0]), "=r"((d)[1]), "=r"((d)[2]), "=r"((d)[3]) \
        : "r"(a0),"r"(a1),"r"(a2),"r"(a3),"r"(b0),"r"(b1), \
          "r"((d)[0]),"r"((d)[1]),"r"((d)[2]),"r"((d)[3]))

extern "C" __global__
__launch_bounds__(256, 1)
void int8_gemm_i32acc(
    const signed char* __restrict__ A_i8,
    const signed char* __restrict__ B_i8,
    const float* __restrict__ A_scale,
    const float* __restrict__ B_scale,
    __nv_bfloat16* __restrict__ C,
    unsigned int M, unsigned int N, unsigned int K
) {
    const unsigned int cta_n = blockIdx.x * 128;
    const unsigned int cta_m = blockIdx.y * 128;
    if (cta_m >= M || cta_n >= N) return;
    const unsigned int t = threadIdx.x;
    const unsigned int warp_id = t >> 5;
    const unsigned int lane = t & 31;
    const unsigned int ng = warp_id >> 1;
    const unsigned int mh = warp_id & 1;
    const unsigned int nb = K >> 5;

    __shared__ int   sW[128][I32AW];
    __shared__ int   sA[128][I32AW];
    __shared__ float sWs[128][I32A_SB];
    __shared__ float sAs[128][I32A_SB];


    float facc[2][8][4];
    #pragma unroll
    for (int n=0;n<2;n++) for(int j=0;j<8;j++){facc[n][j][0]=0;facc[n][j][1]=0;facc[n][j][2]=0;facc[n][j][3]=0;}

    for (unsigned int kb = 0; kb < K; kb += I32A_TILE) {

        const unsigned I32A_CPR = I32A_TILE/16;
        #pragma unroll
        for (int c = 0; c < I32A_TILE/32; c++) {
            unsigned lin = c*256 + t;
            unsigned row = lin / I32A_CPR;
            unsigned col = (lin % I32A_CPR) << 4;
            unsigned gk = kb + col;
            signed char* wdst = ((signed char*)&sW[row][0]) + col;
            signed char* adst = ((signed char*)&sA[row][0]) + col;
            cp_async_pred_16(wdst, &B_i8[(unsigned long long)(cta_n+row)*K + gk], (cta_n+row<N)&&(gk+15<K));
            cp_async_pred_16(adst, &A_i8[(unsigned long long)(cta_m+row)*K + gk], (cta_m+row<M)&&(gk+15<K));
        }
        if (t < 128) {
            unsigned blk = kb >> 5;
            #pragma unroll
            for (int s=0;s<I32A_SB;s++){
                sWs[t][s] = (cta_n+t<N)?B_scale[(unsigned long long)(cta_n+t)*nb + blk + s]:0.f;
                sAs[t][s] = (cta_m+t<M)?A_scale[(unsigned long long)(cta_m+t)*nb + blk + s]:0.f;
            }
        }
        cp_async_commit(); cp_async_wait_all(); __syncthreads();


        #pragma unroll
        for (int sb=0; sb<I32A_SB; sb++){
            unsigned WA[2][4];
            float    wsc[2][2];
            #pragma unroll
            for (int n=0;n<2;n++){
                unsigned wbase_row = ng*32 + n*16;
                const int* xs = &sW[wbase_row][0] + (lane%16)*I32AW + sb*8 + (lane/16)*4;
                asm volatile("ldmatrix.sync.aligned.m8n8.x4.b16 {%0,%1,%2,%3},[%4];"
                    : "=r"(WA[n][0]),"=r"(WA[n][1]),"=r"(WA[n][2]),"=r"(WA[n][3]) : "l"(xs));
                wsc[n][0] = sWs[wbase_row + lane/4][sb];
                wsc[n][1] = sWs[wbase_row + 8 + lane/4][sb];
            }


            int iacc[2][8][4];
            #pragma unroll
            for (int n=0;n<2;n++) for(int j=0;j<8;j++){iacc[n][j][0]=0;iacc[n][j][1]=0;iacc[n][j][2]=0;iacc[n][j][3]=0;}

            float ascs[8][2];



            #pragma unroll
            for (int j=0;j<8;j++){
                unsigned mcol0 = mh*64 + j*8;
                ascs[j][0] = sAs[mcol0 + (lane%4)*2][sb];
                ascs[j][1] = sAs[mcol0 + (lane%4)*2 + 1][sb];
                const int* abase = &sA[mcol0 + lane/4][0] + sb*8;
                unsigned b0 = abase[lane%4];
                unsigned b1 = abase[lane%4 + 4];
                #pragma unroll
                for (int n=0;n<2;n++){
                    int s[4]={0,0,0,0};
                    METRALE_MMA_S8_I32A(s, WA[n][0],WA[n][1],WA[n][2],WA[n][3], b0,b1);

                    iacc[n][j][0] += s[0];
                    iacc[n][j][1] += s[1];
                    iacc[n][j][2] += s[2];
                    iacc[n][j][3] += s[3];
                }
            }



            #pragma unroll
            for (int n=0;n<2;n++){
                #pragma unroll
                for (int j=0;j<8;j++){
                    float sc0 = wsc[n][0] * ascs[j][0];
                    float sc1 = wsc[n][0] * ascs[j][1];
                    float sc2 = wsc[n][1] * ascs[j][0];
                    float sc3 = wsc[n][1] * ascs[j][1];
                    facc[n][j][0] += (float)iacc[n][j][0] * sc0;
                    facc[n][j][1] += (float)iacc[n][j][1] * sc1;
                    facc[n][j][2] += (float)iacc[n][j][2] * sc2;
                    facc[n][j][3] += (float)iacc[n][j][3] * sc3;
                }
            }
        }
        __syncthreads();
    }


    #pragma unroll
    for (int n=0;n<2;n++){
        unsigned nrow0 = cta_n + ng*32 + n*16 + lane/4;
        #pragma unroll
        for (int j=0;j<8;j++){
            unsigned mcol = cta_m + mh*64 + j*8 + (lane%4)*2;
            unsigned cN0=nrow0, cN1=nrow0+8;
            if (mcol<M   && cN0<N) C[(unsigned long long)mcol*N + cN0]     = __float2bfloat16(facc[n][j][0]);
            if (mcol+1<M && cN0<N) C[(unsigned long long)(mcol+1)*N + cN0] = __float2bfloat16(facc[n][j][1]);
            if (mcol<M   && cN1<N) C[(unsigned long long)mcol*N + cN1]     = __float2bfloat16(facc[n][j][2]);
            if (mcol+1<M && cN1<N) C[(unsigned long long)(mcol+1)*N + cN1] = __float2bfloat16(facc[n][j][3]);
        }
    }
}
#undef METRALE_MMA_S8_I32A
#undef I32A_TILE
#undef I32A_SB
#undef I32AW
// 2026-09-25: int8_gemm_faith2 with a sub-block's eight activation fragments and scales (bb, aa) all loaded
// before its MMAs.






#define F3_TILE 128
#define F3_SB   (F3_TILE/32)
#define F3W     36
#define METRALE_MMA_S8(d, a0,a1,a2,a3, b0,b1) \
    asm volatile("mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 " \
        "{%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%10,%11,%12,%13};" \
        : "=r"((d)[0]), "=r"((d)[1]), "=r"((d)[2]), "=r"((d)[3]) \
        : "r"(a0),"r"(a1),"r"(a2),"r"(a3),"r"(b0),"r"(b1), \
          "r"((d)[0]),"r"((d)[1]),"r"((d)[2]),"r"((d)[3]))

extern "C" __global__
__launch_bounds__(256, 1)
void int8_gemm_faith3(
    const signed char* __restrict__ A_i8,
    const signed char* __restrict__ B_i8,
    const float* __restrict__ A_scale,
    const float* __restrict__ B_scale,
    __nv_bfloat16* __restrict__ C,
    unsigned int M, unsigned int N, unsigned int K
) {
    const unsigned int cta_n = blockIdx.x * 128;
    const unsigned int cta_m = blockIdx.y * 128;
    if (cta_m >= M || cta_n >= N) return;
    const unsigned int t = threadIdx.x;
    const unsigned int warp_id = t >> 5;
    const unsigned int lane = t & 31;
    const unsigned int ng = warp_id >> 1;
    const unsigned int mh = warp_id & 1;
    const unsigned int nb = K >> 5;

    __shared__ int   sW[128][F3W];
    __shared__ int   sA[128][F3W];
    __shared__ float sWs[128][F3_SB];
    __shared__ float sAs[128][F3_SB];

    float acc[2][8][4];
    #pragma unroll
    for (int n=0;n<2;n++) for(int j=0;j<8;j++){acc[n][j][0]=0;acc[n][j][1]=0;acc[n][j][2]=0;acc[n][j][3]=0;}

    for (unsigned int kb = 0; kb < K; kb += F3_TILE) {
        const unsigned F3_CPR = F3_TILE/16;
        #pragma unroll
        for (int c = 0; c < F3_TILE/32; c++) {
            unsigned lin = c*256 + t;
            unsigned row = lin / F3_CPR;
            unsigned col = (lin % F3_CPR) << 4;
            unsigned gk = kb + col;
            signed char* wdst = ((signed char*)&sW[row][0]) + col;
            signed char* adst = ((signed char*)&sA[row][0]) + col;
            cp_async_pred_16(wdst, &B_i8[(unsigned long long)(cta_n+row)*K + gk], (cta_n+row<N)&&(gk+15<K));
            cp_async_pred_16(adst, &A_i8[(unsigned long long)(cta_m+row)*K + gk], (cta_m+row<M)&&(gk+15<K));
        }
        if (t < 128) {
            unsigned blk = kb >> 5;
            #pragma unroll
            for (int s=0;s<F3_SB;s++){
                sWs[t][s] = (cta_n+t<N)?B_scale[(unsigned long long)(cta_n+t)*nb + blk + s]:0.f;
                sAs[t][s] = (cta_m+t<M)?A_scale[(unsigned long long)(cta_m+t)*nb + blk + s]:0.f;
            }
        }
        cp_async_commit(); cp_async_wait_all(); __syncthreads();

        #pragma unroll
        for (int sb=0; sb<F3_SB; sb++){
            unsigned WA[2][4];
            float    wsc[2][2];
            #pragma unroll
            for (int n=0;n<2;n++){
                unsigned wbase_row = ng*32 + n*16;
                const int* xs = &sW[wbase_row][0] + (lane%16)*F3W + sb*8 + (lane/16)*4;
                asm volatile("ldmatrix.sync.aligned.m8n8.x4.b16 {%0,%1,%2,%3},[%4];"
                    : "=r"(WA[n][0]),"=r"(WA[n][1]),"=r"(WA[n][2]),"=r"(WA[n][3]) : "l"(xs));
                wsc[n][0] = sWs[wbase_row + lane/4][sb];
                wsc[n][1] = sWs[wbase_row + 8 + lane/4][sb];
            }

            unsigned bb[8][2];
            float    aa[8][2];
            #pragma unroll
            for (int j=0;j<8;j++){
                unsigned mcol0 = mh*64 + j*8;
                const int* abase = &sA[mcol0 + lane/4][0] + sb*8;
                bb[j][0] = abase[lane%4];
                bb[j][1] = abase[lane%4 + 4];
                aa[j][0] = sAs[mcol0 + (lane%4)*2][sb];
                aa[j][1] = sAs[mcol0 + (lane%4)*2 + 1][sb];
            }
            #pragma unroll
            for (int j=0;j<8;j++){
                #pragma unroll
                for (int n=0;n<2;n++){
                    int s[4]={0,0,0,0};
                    METRALE_MMA_S8(s, WA[n][0],WA[n][1],WA[n][2],WA[n][3], bb[j][0],bb[j][1]);
                    acc[n][j][0]+=(float)s[0]*wsc[n][0]*aa[j][0];
                    acc[n][j][1]+=(float)s[1]*wsc[n][0]*aa[j][1];
                    acc[n][j][2]+=(float)s[2]*wsc[n][1]*aa[j][0];
                    acc[n][j][3]+=(float)s[3]*wsc[n][1]*aa[j][1];
                }
            }
        }
        __syncthreads();
    }

    #pragma unroll
    for (int n=0;n<2;n++){
        unsigned nrow0 = cta_n + ng*32 + n*16 + lane/4;
        #pragma unroll
        for (int j=0;j<8;j++){
            unsigned mcol = cta_m + mh*64 + j*8 + (lane%4)*2;
            unsigned cN0=nrow0, cN1=nrow0+8;
            if (mcol<M   && cN0<N) C[(unsigned long long)mcol*N + cN0]     = __float2bfloat16(acc[n][j][0]);
            if (mcol+1<M && cN0<N) C[(unsigned long long)(mcol+1)*N + cN0] = __float2bfloat16(acc[n][j][1]);
            if (mcol<M   && cN1<N) C[(unsigned long long)mcol*N + cN1]     = __float2bfloat16(acc[n][j][2]);
            if (mcol+1<M && cN1<N) C[(unsigned long long)(mcol+1)*N + cN1] = __float2bfloat16(acc[n][j][3]);
        }
    }
}
#undef METRALE_MMA_S8
#undef F3_TILE
#undef F3_SB
#undef F3W

// 2026-09-25: int8_gemm_faith2 on 512 threads: sixteen warps in a 4 x 4 grid, each owning 32 weight rows by 32
// tokens (acc[2][4][4]).









#define F4_TILE 128
#define F4_SB   (F4_TILE/32)
#define F4W     36
#define METRALE_MMA_S8(d, a0,a1,a2,a3, b0,b1) \
    asm volatile("mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 " \
        "{%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%10,%11,%12,%13};" \
        : "=r"((d)[0]), "=r"((d)[1]), "=r"((d)[2]), "=r"((d)[3]) \
        : "r"(a0),"r"(a1),"r"(a2),"r"(a3),"r"(b0),"r"(b1), \
          "r"((d)[0]),"r"((d)[1]),"r"((d)[2]),"r"((d)[3]))

extern "C" __global__
__launch_bounds__(512, 1)
void int8_gemm_faith4(
    const signed char* __restrict__ A_i8,
    const signed char* __restrict__ B_i8,
    const float* __restrict__ A_scale,
    const float* __restrict__ B_scale,
    __nv_bfloat16* __restrict__ C,
    unsigned int M, unsigned int N, unsigned int K
) {
    const unsigned int cta_n = blockIdx.x * 128;
    const unsigned int cta_m = blockIdx.y * 128;
    if (cta_m >= M || cta_n >= N) return;
    const unsigned int t = threadIdx.x;
    const unsigned int warp_id = t >> 5;
    const unsigned int lane = t & 31;
    const unsigned int ng = warp_id & 3;
    const unsigned int mg = warp_id >> 2;
    const unsigned int nb = K >> 5;

    __shared__ int   sW[128][F4W];
    __shared__ int   sA[128][F4W];
    __shared__ float sWs[128][F4_SB];
    __shared__ float sAs[128][F4_SB];

    float acc[2][4][4];
    #pragma unroll
    for (int n=0;n<2;n++) for(int j=0;j<4;j++){acc[n][j][0]=0;acc[n][j][1]=0;acc[n][j][2]=0;acc[n][j][3]=0;}

    for (unsigned int kb = 0; kb < K; kb += F4_TILE) {
        const unsigned F4_CPR = F4_TILE/16;

        #pragma unroll
        for (int c = 0; c < (128*F4_TILE/16)/512; c++) {
            unsigned lin = c*512 + t;
            unsigned row = lin / F4_CPR;
            unsigned col = (lin % F4_CPR) << 4;
            unsigned gk = kb + col;
            signed char* wdst = ((signed char*)&sW[row][0]) + col;
            signed char* adst = ((signed char*)&sA[row][0]) + col;
            cp_async_pred_16(wdst, &B_i8[(unsigned long long)(cta_n+row)*K + gk], (cta_n+row<N)&&(gk+15<K));
            cp_async_pred_16(adst, &A_i8[(unsigned long long)(cta_m+row)*K + gk], (cta_m+row<M)&&(gk+15<K));
        }
        if (t < 128) {
            unsigned blk = kb >> 5;
            #pragma unroll
            for (int s=0;s<F4_SB;s++){
                sWs[t][s] = (cta_n+t<N)?B_scale[(unsigned long long)(cta_n+t)*nb + blk + s]:0.f;
                sAs[t][s] = (cta_m+t<M)?A_scale[(unsigned long long)(cta_m+t)*nb + blk + s]:0.f;
            }
        }
        cp_async_commit(); cp_async_wait_all(); __syncthreads();

        #pragma unroll
        for (int sb=0; sb<F4_SB; sb++){
            unsigned WA[2][4];
            float    wsc[2][2];
            #pragma unroll
            for (int n=0;n<2;n++){
                unsigned wbase_row = ng*32 + n*16;
                const int* xs = &sW[wbase_row][0] + (lane%16)*F4W + sb*8 + (lane/16)*4;
                asm volatile("ldmatrix.sync.aligned.m8n8.x4.b16 {%0,%1,%2,%3},[%4];"
                    : "=r"(WA[n][0]),"=r"(WA[n][1]),"=r"(WA[n][2]),"=r"(WA[n][3]) : "l"(xs));
                wsc[n][0] = sWs[wbase_row + lane/4][sb];
                wsc[n][1] = sWs[wbase_row + 8 + lane/4][sb];
            }
            #pragma unroll
            for (int j=0;j<4;j++){
                unsigned mcol0 = mg*32 + j*8;
                float asc0 = sAs[mcol0 + (lane%4)*2][sb];
                float asc1 = sAs[mcol0 + (lane%4)*2 + 1][sb];
                const int* abase = &sA[mcol0 + lane/4][0] + sb*8;
                unsigned b0 = abase[lane%4];
                unsigned b1 = abase[lane%4 + 4];
                #pragma unroll
                for (int n=0;n<2;n++){
                    int s[4]={0,0,0,0};
                    METRALE_MMA_S8(s, WA[n][0],WA[n][1],WA[n][2],WA[n][3], b0,b1);
                    acc[n][j][0]+=(float)s[0]*wsc[n][0]*asc0;
                    acc[n][j][1]+=(float)s[1]*wsc[n][0]*asc1;
                    acc[n][j][2]+=(float)s[2]*wsc[n][1]*asc0;
                    acc[n][j][3]+=(float)s[3]*wsc[n][1]*asc1;
                }
            }
        }
        __syncthreads();
    }

    #pragma unroll
    for (int n=0;n<2;n++){
        unsigned nrow0 = cta_n + ng*32 + n*16 + lane/4;
        #pragma unroll
        for (int j=0;j<4;j++){
            unsigned mcol = cta_m + mg*32 + j*8 + (lane%4)*2;
            unsigned cN0=nrow0, cN1=nrow0+8;
            if (mcol<M   && cN0<N) C[(unsigned long long)mcol*N + cN0]     = __float2bfloat16(acc[n][j][0]);
            if (mcol+1<M && cN0<N) C[(unsigned long long)(mcol+1)*N + cN0] = __float2bfloat16(acc[n][j][1]);
            if (mcol<M   && cN1<N) C[(unsigned long long)mcol*N + cN1]     = __float2bfloat16(acc[n][j][2]);
            if (mcol+1<M && cN1<N) C[(unsigned long long)(mcol+1)*N + cN1] = __float2bfloat16(acc[n][j][3]);
        }
    }
}
#undef METRALE_MMA_S8
#undef F4_TILE
#undef F4_SB
#undef F4W

// 2026-09-25: int8_gemm_faith2 with sW, sA and their scales double-buffered: tile i+1's loads are issued before
// tile i is multiplied and stay in flight (cp_async_wait_group<1>). Not built under __SCALE__.









#define M2_TILE 128
#define M2_SB   (M2_TILE/32)
#define M2W     36
#define METRALE_MMA_S8(d, a0,a1,a2,a3, b0,b1) \
    asm volatile("mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 " \
        "{%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%10,%11,%12,%13};" \
        : "=r"((d)[0]), "=r"((d)[1]), "=r"((d)[2]), "=r"((d)[3]) \
        : "r"(a0),"r"(a1),"r"(a2),"r"(a3),"r"(b0),"r"(b1), \
          "r"((d)[0]),"r"((d)[1]),"r"((d)[2]),"r"((d)[3]))







#if !defined(__SCALE__)
extern "C" __global__
__launch_bounds__(256, 1)
void int8_gemm_mmq2(
    const signed char* __restrict__ A_i8,
    const signed char* __restrict__ B_i8,
    const float* __restrict__ A_scale,
    const float* __restrict__ B_scale,
    __nv_bfloat16* __restrict__ C,
    unsigned int M, unsigned int N, unsigned int K
) {
    const unsigned int cta_n = blockIdx.x * 128;
    const unsigned int cta_m = blockIdx.y * 128;
    if (cta_m >= M || cta_n >= N) return;
    const unsigned int t = threadIdx.x;
    const unsigned int warp_id = t >> 5;
    const unsigned int lane = t & 31;
    const unsigned int ng = warp_id >> 1;
    const unsigned int mh = warp_id & 1;
    const unsigned int nb = K >> 5;

    __shared__ int   sW[2][128][M2W];
    __shared__ int   sA[2][128][M2W];
    __shared__ float sWs[2][128][M2_SB];
    __shared__ float sAs[2][128][M2_SB];

    float acc[2][8][4];
    #pragma unroll
    for (int n=0;n<2;n++) for(int j=0;j<8;j++){acc[n][j][0]=0;acc[n][j][1]=0;acc[n][j][2]=0;acc[n][j][3]=0;}

    const unsigned int ntiles = (K + M2_TILE - 1) / M2_TILE;
    const unsigned int M2_CPR = M2_TILE/16;

    #define M2_LOAD_TILE(buf, kb_) { \
        _Pragma("unroll") \
        for (int c = 0; c < M2_TILE/32; c++) { \
            unsigned lin = c*256 + t; \
            unsigned row = lin / M2_CPR; \
            unsigned col = (lin % M2_CPR) << 4; \
            unsigned gk  = (kb_) + col; \
            signed char* wdst = ((signed char*)&sW[buf][row][0]) + col; \
            signed char* adst = ((signed char*)&sA[buf][row][0]) + col; \
            cp_async_pred_16(wdst, &B_i8[(unsigned long long)(cta_n+row)*K + gk], (cta_n+row<N)&&(gk+15<K)); \
            cp_async_pred_16(adst, &A_i8[(unsigned long long)(cta_m+row)*K + gk], (cta_m+row<M)&&(gk+15<K)); \
        } \
        if (t < 128) { \
            unsigned blk = (kb_) >> 5; \
            _Pragma("unroll") \
            for (int s=0;s<M2_SB;s++){ \
                sWs[buf][t][s] = (cta_n+t<N)?B_scale[(unsigned long long)(cta_n+t)*nb + blk + s]:0.f; \
                sAs[buf][t][s] = (cta_m+t<M)?A_scale[(unsigned long long)(cta_m+t)*nb + blk + s]:0.f; \
            } \
        } \
    }


    M2_LOAD_TILE(0, 0); cp_async_commit();

    for (unsigned int i = 0; i < ntiles; i++) {
        unsigned int cur = i & 1, nxt = (i + 1) & 1;

        if (i + 1 < ntiles) { M2_LOAD_TILE(nxt, (i+1)*M2_TILE); cp_async_commit(); }

        if (i + 1 < ntiles) cp_async_wait_group<1>(); else cp_async_wait_all();
        __syncthreads();

        #pragma unroll
        for (int sb=0; sb<M2_SB; sb++){
            unsigned WA[2][4];
            float    wsc[2][2];
            #pragma unroll
            for (int n=0;n<2;n++){
                unsigned wbase_row = ng*32 + n*16;
                const int* xs = &sW[cur][wbase_row][0] + (lane%16)*M2W + sb*8 + (lane/16)*4;
                asm volatile("ldmatrix.sync.aligned.m8n8.x4.b16 {%0,%1,%2,%3},[%4];"
                    : "=r"(WA[n][0]),"=r"(WA[n][1]),"=r"(WA[n][2]),"=r"(WA[n][3]) : "l"(xs));
                wsc[n][0] = sWs[cur][wbase_row + lane/4][sb];
                wsc[n][1] = sWs[cur][wbase_row + 8 + lane/4][sb];
            }
            #pragma unroll
            for (int j=0;j<8;j++){
                unsigned mcol0 = mh*64 + j*8;
                float asc0 = sAs[cur][mcol0 + (lane%4)*2][sb];
                float asc1 = sAs[cur][mcol0 + (lane%4)*2 + 1][sb];
                const int* abase = &sA[cur][mcol0 + lane/4][0] + sb*8;
                unsigned b0 = abase[lane%4];
                unsigned b1 = abase[lane%4 + 4];
                #pragma unroll
                for (int n=0;n<2;n++){
                    int s[4]={0,0,0,0};
                    METRALE_MMA_S8(s, WA[n][0],WA[n][1],WA[n][2],WA[n][3], b0,b1);
                    acc[n][j][0]+=(float)s[0]*wsc[n][0]*asc0;
                    acc[n][j][1]+=(float)s[1]*wsc[n][0]*asc1;
                    acc[n][j][2]+=(float)s[2]*wsc[n][1]*asc0;
                    acc[n][j][3]+=(float)s[3]*wsc[n][1]*asc1;
                }
            }
        }
        __syncthreads();  // 2026-09-25: buffer cur is reloaded at step i + 2
    }
    #undef M2_LOAD_TILE

    #pragma unroll
    for (int n=0;n<2;n++){
        unsigned nrow0 = cta_n + ng*32 + n*16 + lane/4;
        #pragma unroll
        for (int j=0;j<8;j++){
            unsigned mcol = cta_m + mh*64 + j*8 + (lane%4)*2;
            unsigned cN0=nrow0, cN1=nrow0+8;
            if (mcol<M   && cN0<N) C[(unsigned long long)mcol*N + cN0]     = __float2bfloat16(acc[n][j][0]);
            if (mcol+1<M && cN0<N) C[(unsigned long long)(mcol+1)*N + cN0] = __float2bfloat16(acc[n][j][1]);
            if (mcol<M   && cN1<N) C[(unsigned long long)mcol*N + cN1]     = __float2bfloat16(acc[n][j][2]);
            if (mcol+1<M && cN1<N) C[(unsigned long long)(mcol+1)*N + cN1] = __float2bfloat16(acc[n][j][3]);
        }
    }
}
#endif
#undef METRALE_MMA_S8
#undef M2_TILE
#undef M2_SB
#undef M2W

// 2026-09-25: int8_gemm_faith2 for activations stored K-interleaved as [0,4,1,5,2,6,3,7] within each 8 int32
// (requant_a_bf16_int8_il): the two B-fragment int32 are adjacent and load as one int2.












#define F2_TILE 128
#define F2_SB   (F2_TILE/32)
#define F2W     36
#define METRALE_MMA_S8(d, a0,a1,a2,a3, b0,b1) \
    asm volatile("mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 " \
        "{%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%10,%11,%12,%13};" \
        : "=r"((d)[0]), "=r"((d)[1]), "=r"((d)[2]), "=r"((d)[3]) \
        : "r"(a0),"r"(a1),"r"(a2),"r"(a3),"r"(b0),"r"(b1), \
          "r"((d)[0]),"r"((d)[1]),"r"((d)[2]),"r"((d)[3]))
extern "C" __global__
__launch_bounds__(256, 1)
void int8_gemm_faith5(
    const signed char* __restrict__ A_i8,
    const signed char* __restrict__ B_i8,
    const float* __restrict__ A_scale,
    const float* __restrict__ B_scale,
    __nv_bfloat16* __restrict__ C,
    unsigned int M, unsigned int N, unsigned int K
) {
    const unsigned int cta_n = blockIdx.x * 128;
    const unsigned int cta_m = blockIdx.y * 128;
    if (cta_m >= M || cta_n >= N) return;
    const unsigned int t = threadIdx.x;
    const unsigned int warp_id = t >> 5;
    const unsigned int lane = t & 31;
    const unsigned int ng = warp_id >> 1;
    const unsigned int mh = warp_id & 1;
    const unsigned int nb = K >> 5;

    __shared__ int   sW[128][F2W];
    __shared__ int   sA[128][F2W];
    __shared__ float sWs[128][F2_SB];
    __shared__ float sAs[128][F2_SB];

    float acc[2][8][4];
    #pragma unroll
    for (int n=0;n<2;n++) for(int j=0;j<8;j++){acc[n][j][0]=0;acc[n][j][1]=0;acc[n][j][2]=0;acc[n][j][3]=0;}

    for (unsigned int kb = 0; kb < K; kb += F2_TILE) {
        const unsigned F2_CPR = F2_TILE/16;
        #pragma unroll
        for (int c = 0; c < F2_TILE/32; c++) {
            unsigned lin = c*256 + t;
            unsigned row = lin / F2_CPR;
            unsigned col = (lin % F2_CPR) << 4;
            unsigned gk = kb + col;
            signed char* wdst = ((signed char*)&sW[row][0]) + col;
            signed char* adst = ((signed char*)&sA[row][0]) + col;
            cp_async_pred_16(wdst, &B_i8[(unsigned long long)(cta_n+row)*K + gk], (cta_n+row<N)&&(gk+15<K));
            cp_async_pred_16(adst, &A_i8[(unsigned long long)(cta_m+row)*K + gk], (cta_m+row<M)&&(gk+15<K));
        }
        if (t < 128) {
            unsigned blk = kb >> 5;
            #pragma unroll
            for (int s=0;s<F2_SB;s++){
                sWs[t][s] = (cta_n+t<N)?B_scale[(unsigned long long)(cta_n+t)*nb + blk + s]:0.f;
                sAs[t][s] = (cta_m+t<M)?A_scale[(unsigned long long)(cta_m+t)*nb + blk + s]:0.f;
            }
        }
        cp_async_commit(); cp_async_wait_all(); __syncthreads();

        #pragma unroll
        for (int sb=0; sb<F2_SB; sb++){
            unsigned WA[2][4];
            float    wsc[2][2];
            #pragma unroll
            for (int n=0;n<2;n++){
                unsigned wbase_row = ng*32 + n*16;
                const int* xs = &sW[wbase_row][0] + (lane%16)*F2W + sb*8 + (lane/16)*4;
                asm volatile("ldmatrix.sync.aligned.m8n8.x4.b16 {%0,%1,%2,%3},[%4];"
                    : "=r"(WA[n][0]),"=r"(WA[n][1]),"=r"(WA[n][2]),"=r"(WA[n][3]) : "l"(xs));
                wsc[n][0] = sWs[wbase_row + lane/4][sb];
                wsc[n][1] = sWs[wbase_row + 8 + lane/4][sb];
            }
            #pragma unroll
            for (int j=0;j<8;j++){
                unsigned mcol0 = mh*64 + j*8;
                float asc0 = sAs[mcol0 + (lane%4)*2][sb];
                float asc1 = sAs[mcol0 + (lane%4)*2 + 1][sb];



                const int* abase = &sA[mcol0 + lane/4][0] + sb*8;
                int2 bb = *(const int2*)(abase + (lane%4)*2);
                unsigned b0 = (unsigned)bb.x;
                unsigned b1 = (unsigned)bb.y;
                #pragma unroll
                for (int n=0;n<2;n++){
                    int s[4]={0,0,0,0};
                    METRALE_MMA_S8(s, WA[n][0],WA[n][1],WA[n][2],WA[n][3], b0,b1);
                    acc[n][j][0]+=(float)s[0]*wsc[n][0]*asc0;
                    acc[n][j][1]+=(float)s[1]*wsc[n][0]*asc1;
                    acc[n][j][2]+=(float)s[2]*wsc[n][1]*asc0;
                    acc[n][j][3]+=(float)s[3]*wsc[n][1]*asc1;
                }
            }
        }
        __syncthreads();
    }

    #pragma unroll
    for (int n=0;n<2;n++){
        unsigned nrow0 = cta_n + ng*32 + n*16 + lane/4;
        #pragma unroll
        for (int j=0;j<8;j++){
            unsigned mcol = cta_m + mh*64 + j*8 + (lane%4)*2;
            unsigned cN0=nrow0, cN1=nrow0+8;
            if (mcol<M   && cN0<N) C[(unsigned long long)mcol*N + cN0]     = __float2bfloat16(acc[n][j][0]);
            if (mcol+1<M && cN0<N) C[(unsigned long long)(mcol+1)*N + cN0] = __float2bfloat16(acc[n][j][1]);
            if (mcol<M   && cN1<N) C[(unsigned long long)mcol*N + cN1]     = __float2bfloat16(acc[n][j][2]);
            if (mcol+1<M && cN1<N) C[(unsigned long long)(mcol+1)*N + cN1] = __float2bfloat16(acc[n][j][3]);
        }
    }
}
#undef METRALE_MMA_S8
#undef F2_TILE
#undef F2_SB
#undef F2W

// 2026-09-25: int8_gemm_faith2 with each sub-block in three phases: load all eight activation fragments and
// scales, issue all 16 MMAs into separate int32 results si[8][2][4], then apply the scales.























#define F2_TILE 128
#define F2_SB   (F2_TILE/32)
#define F2W     36
#define METRALE_MMA_S8(d, a0,a1,a2,a3, b0,b1) \
    asm volatile("mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 " \
        "{%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%10,%11,%12,%13};" \
        : "=r"((d)[0]), "=r"((d)[1]), "=r"((d)[2]), "=r"((d)[3]) \
        : "r"(a0),"r"(a1),"r"(a2),"r"(a3),"r"(b0),"r"(b1), \
          "r"((d)[0]),"r"((d)[1]),"r"((d)[2]),"r"((d)[3]))
extern "C" __global__
__launch_bounds__(256, 1)
void int8_gemm_faith6(
    const signed char* __restrict__ A_i8,
    const signed char* __restrict__ B_i8,
    const float* __restrict__ A_scale,
    const float* __restrict__ B_scale,
    __nv_bfloat16* __restrict__ C,
    unsigned int M, unsigned int N, unsigned int K
) {
    const unsigned int cta_n = blockIdx.x * 128;
    const unsigned int cta_m = blockIdx.y * 128;
    if (cta_m >= M || cta_n >= N) return;
    const unsigned int t = threadIdx.x;
    const unsigned int warp_id = t >> 5;
    const unsigned int lane = t & 31;
    const unsigned int ng = warp_id >> 1;
    const unsigned int mh = warp_id & 1;
    const unsigned int nb = K >> 5;

    __shared__ int   sW[128][F2W];
    __shared__ int   sA[128][F2W];
    __shared__ float sWs[128][F2_SB];
    __shared__ float sAs[128][F2_SB];

    float acc[2][8][4];
    #pragma unroll
    for (int n=0;n<2;n++) for(int j=0;j<8;j++){acc[n][j][0]=0;acc[n][j][1]=0;acc[n][j][2]=0;acc[n][j][3]=0;}

    for (unsigned int kb = 0; kb < K; kb += F2_TILE) {
        const unsigned F2_CPR = F2_TILE/16;
        #pragma unroll
        for (int c = 0; c < F2_TILE/32; c++) {
            unsigned lin = c*256 + t;
            unsigned row = lin / F2_CPR;
            unsigned col = (lin % F2_CPR) << 4;
            unsigned gk = kb + col;
            signed char* wdst = ((signed char*)&sW[row][0]) + col;
            signed char* adst = ((signed char*)&sA[row][0]) + col;
            cp_async_pred_16(wdst, &B_i8[(unsigned long long)(cta_n+row)*K + gk], (cta_n+row<N)&&(gk+15<K));
            cp_async_pred_16(adst, &A_i8[(unsigned long long)(cta_m+row)*K + gk], (cta_m+row<M)&&(gk+15<K));
        }
        if (t < 128) {
            unsigned blk = kb >> 5;
            #pragma unroll
            for (int s=0;s<F2_SB;s++){
                sWs[t][s] = (cta_n+t<N)?B_scale[(unsigned long long)(cta_n+t)*nb + blk + s]:0.f;
                sAs[t][s] = (cta_m+t<M)?A_scale[(unsigned long long)(cta_m+t)*nb + blk + s]:0.f;
            }
        }
        cp_async_commit(); cp_async_wait_all(); __syncthreads();

        #pragma unroll
        for (int sb=0; sb<F2_SB; sb++){
            unsigned WA[2][4];
            float    wsc[2][2];
            #pragma unroll
            for (int n=0;n<2;n++){
                unsigned wbase_row = ng*32 + n*16;
                const int* xs = &sW[wbase_row][0] + (lane%16)*F2W + sb*8 + (lane/16)*4;
                asm volatile("ldmatrix.sync.aligned.m8n8.x4.b16 {%0,%1,%2,%3},[%4];"
                    : "=r"(WA[n][0]),"=r"(WA[n][1]),"=r"(WA[n][2]),"=r"(WA[n][3]) : "l"(xs));
                wsc[n][0] = sWs[wbase_row + lane/4][sb];
                wsc[n][1] = sWs[wbase_row + 8 + lane/4][sb];
            }

            unsigned bb[8][2];
            float    asc[8][2];
            #pragma unroll
            for (int j=0;j<8;j++){
                unsigned mcol0 = mh*64 + j*8;
                asc[j][0] = sAs[mcol0 + (lane%4)*2][sb];
                asc[j][1] = sAs[mcol0 + (lane%4)*2 + 1][sb];
                const int* abase = &sA[mcol0 + lane/4][0] + sb*8;
                bb[j][0] = abase[lane%4];
                bb[j][1] = abase[lane%4 + 4];
            }

            int si[8][2][4];
            #pragma unroll
            for (int j=0;j<8;j++){
                #pragma unroll
                for (int n=0;n<2;n++){
                    si[j][n][0]=0; si[j][n][1]=0; si[j][n][2]=0; si[j][n][3]=0;
                    METRALE_MMA_S8(si[j][n], WA[n][0],WA[n][1],WA[n][2],WA[n][3], bb[j][0],bb[j][1]);
                }
            }

            #pragma unroll
            for (int j=0;j<8;j++){
                #pragma unroll
                for (int n=0;n<2;n++){
                    acc[n][j][0]+=(float)si[j][n][0]*wsc[n][0]*asc[j][0];
                    acc[n][j][1]+=(float)si[j][n][1]*wsc[n][0]*asc[j][1];
                    acc[n][j][2]+=(float)si[j][n][2]*wsc[n][1]*asc[j][0];
                    acc[n][j][3]+=(float)si[j][n][3]*wsc[n][1]*asc[j][1];
                }
            }
        }
        __syncthreads();
    }

    #pragma unroll
    for (int n=0;n<2;n++){
        unsigned nrow0 = cta_n + ng*32 + n*16 + lane/4;
        #pragma unroll
        for (int j=0;j<8;j++){
            unsigned mcol = cta_m + mh*64 + j*8 + (lane%4)*2;
            unsigned cN0=nrow0, cN1=nrow0+8;
            if (mcol<M   && cN0<N) C[(unsigned long long)mcol*N + cN0]     = __float2bfloat16(acc[n][j][0]);
            if (mcol+1<M && cN0<N) C[(unsigned long long)(mcol+1)*N + cN0] = __float2bfloat16(acc[n][j][1]);
            if (mcol<M   && cN1<N) C[(unsigned long long)mcol*N + cN1]     = __float2bfloat16(acc[n][j][2]);
            if (mcol+1<M && cN1<N) C[(unsigned long long)(mcol+1)*N + cN1] = __float2bfloat16(acc[n][j][3]);
        }
    }
}
#undef METRALE_MMA_S8
#undef F2_TILE
#undef F2_SB
#undef F2W

// 2026-09-25: int8_gemm_faith2 with the warp tile transposed to 64 weight rows by 32 tokens (four 16-row weight
// tiles by four 8-token chunks), so each loaded fragment feeds four MMAs.











#define F2_TILE 128
#define F2_SB   (F2_TILE/32)
#define F2W     36
#define METRALE_MMA_S8(d, a0,a1,a2,a3, b0,b1) \
    asm volatile("mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 " \
        "{%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%10,%11,%12,%13};" \
        : "=r"((d)[0]), "=r"((d)[1]), "=r"((d)[2]), "=r"((d)[3]) \
        : "r"(a0),"r"(a1),"r"(a2),"r"(a3),"r"(b0),"r"(b1), \
          "r"((d)[0]),"r"((d)[1]),"r"((d)[2]),"r"((d)[3]))
extern "C" __global__
__launch_bounds__(256, 1)
void int8_gemm_faith7(
    const signed char* __restrict__ A_i8,
    const signed char* __restrict__ B_i8,
    const float* __restrict__ A_scale,
    const float* __restrict__ B_scale,
    __nv_bfloat16* __restrict__ C,
    unsigned int M, unsigned int N, unsigned int K
) {
    const unsigned int cta_n = blockIdx.x * 128;
    const unsigned int cta_m = blockIdx.y * 128;
    if (cta_m >= M || cta_n >= N) return;
    const unsigned int t = threadIdx.x;
    const unsigned int warp_id = t >> 5;
    const unsigned int lane = t & 31;
    const unsigned int ng = warp_id >> 2;
    const unsigned int mq = warp_id & 3;
    const unsigned int nb = K >> 5;

    __shared__ int   sW[128][F2W];
    __shared__ int   sA[128][F2W];
    __shared__ float sWs[128][F2_SB];
    __shared__ float sAs[128][F2_SB];

    float acc[4][4][4];
    #pragma unroll
    for (int i=0;i<4;i++) for(int j=0;j<4;j++){acc[i][j][0]=0;acc[i][j][1]=0;acc[i][j][2]=0;acc[i][j][3]=0;}

    for (unsigned int kb = 0; kb < K; kb += F2_TILE) {
        const unsigned F2_CPR = F2_TILE/16;
        #pragma unroll
        for (int c = 0; c < F2_TILE/32; c++) {
            unsigned lin = c*256 + t;
            unsigned row = lin / F2_CPR;
            unsigned col = (lin % F2_CPR) << 4;
            unsigned gk = kb + col;
            signed char* wdst = ((signed char*)&sW[row][0]) + col;
            signed char* adst = ((signed char*)&sA[row][0]) + col;
            cp_async_pred_16(wdst, &B_i8[(unsigned long long)(cta_n+row)*K + gk], (cta_n+row<N)&&(gk+15<K));
            cp_async_pred_16(adst, &A_i8[(unsigned long long)(cta_m+row)*K + gk], (cta_m+row<M)&&(gk+15<K));
        }
        if (t < 128) {
            unsigned blk = kb >> 5;
            #pragma unroll
            for (int s=0;s<F2_SB;s++){
                sWs[t][s] = (cta_n+t<N)?B_scale[(unsigned long long)(cta_n+t)*nb + blk + s]:0.f;
                sAs[t][s] = (cta_m+t<M)?A_scale[(unsigned long long)(cta_m+t)*nb + blk + s]:0.f;
            }
        }
        cp_async_commit(); cp_async_wait_all(); __syncthreads();

        #pragma unroll
        for (int sb=0; sb<F2_SB; sb++){
            unsigned WA[4][4];
            float    wsc[4][2];
            #pragma unroll
            for (int nm=0;nm<4;nm++){
                unsigned wbase_row = ng*64 + nm*16;
                const int* xs = &sW[wbase_row][0] + (lane%16)*F2W + sb*8 + (lane/16)*4;
                asm volatile("ldmatrix.sync.aligned.m8n8.x4.b16 {%0,%1,%2,%3},[%4];"
                    : "=r"(WA[nm][0]),"=r"(WA[nm][1]),"=r"(WA[nm][2]),"=r"(WA[nm][3]) : "l"(xs));
                wsc[nm][0] = sWs[wbase_row + lane/4][sb];
                wsc[nm][1] = sWs[wbase_row + 8 + lane/4][sb];
            }
            unsigned bb[4][2];
            float    asc[4][2];
            #pragma unroll
            for (int tc=0;tc<4;tc++){
                unsigned mcol0 = mq*32 + tc*8;
                asc[tc][0] = sAs[mcol0 + (lane%4)*2][sb];
                asc[tc][1] = sAs[mcol0 + (lane%4)*2 + 1][sb];
                const int* abase = &sA[mcol0 + lane/4][0] + sb*8;
                bb[tc][0] = abase[lane%4];
                bb[tc][1] = abase[lane%4 + 4];
            }
            #pragma unroll
            for (int nm=0;nm<4;nm++){
                #pragma unroll
                for (int tc=0;tc<4;tc++){
                    int s[4]={0,0,0,0};
                    METRALE_MMA_S8(s, WA[nm][0],WA[nm][1],WA[nm][2],WA[nm][3], bb[tc][0],bb[tc][1]);
                    acc[nm][tc][0]+=(float)s[0]*wsc[nm][0]*asc[tc][0];
                    acc[nm][tc][1]+=(float)s[1]*wsc[nm][0]*asc[tc][1];
                    acc[nm][tc][2]+=(float)s[2]*wsc[nm][1]*asc[tc][0];
                    acc[nm][tc][3]+=(float)s[3]*wsc[nm][1]*asc[tc][1];
                }
            }
        }
        __syncthreads();
    }

    #pragma unroll
    for (int nm=0;nm<4;nm++){
        unsigned nrow0 = cta_n + ng*64 + nm*16 + lane/4;
        #pragma unroll
        for (int tc=0;tc<4;tc++){
            unsigned mcol = cta_m + mq*32 + tc*8 + (lane%4)*2;
            unsigned cN0=nrow0, cN1=nrow0+8;
            if (mcol<M   && cN0<N) C[(unsigned long long)mcol*N + cN0]     = __float2bfloat16(acc[nm][tc][0]);
            if (mcol+1<M && cN0<N) C[(unsigned long long)(mcol+1)*N + cN0] = __float2bfloat16(acc[nm][tc][1]);
            if (mcol<M   && cN1<N) C[(unsigned long long)mcol*N + cN1]     = __float2bfloat16(acc[nm][tc][2]);
            if (mcol+1<M && cN1<N) C[(unsigned long long)(mcol+1)*N + cN1] = __float2bfloat16(acc[nm][tc][3]);
        }
    }
}
#undef METRALE_MMA_S8
#undef F2_TILE
#undef F2_SB
#undef F2W

// 2026-09-25: int8_gemm_faith2 with sW and sA double-buffered: tile ki+1's cp.async loads are issued before tile
// ki is multiplied and stay in flight (cp_async_wait_group<1>). The scales stay single-buffered and are loaded
// synchronously for each tile. Not built under __SCALE__.











#define F2_TILE 128
#define F2_SB   (F2_TILE/32)
#define F2W     36
#define METRALE_MMA_S8(d, a0,a1,a2,a3, b0,b1) \
    asm volatile("mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 " \
        "{%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%10,%11,%12,%13};" \
        : "=r"((d)[0]), "=r"((d)[1]), "=r"((d)[2]), "=r"((d)[3]) \
        : "r"(a0),"r"(a1),"r"(a2),"r"(a3),"r"(b0),"r"(b1), \
          "r"((d)[0]),"r"((d)[1]),"r"((d)[2]),"r"((d)[3]))
// 2026-09-25: cp.async loads of the int8 data of the K tile at KB into buffer BUF.
#define F8_LOAD_TILE(BUF, KB) do { \
    const unsigned _cpr = F2_TILE/16; \
    _Pragma("unroll") \
    for (int _c=0; _c<F2_TILE/32; _c++){ \
        unsigned _lin=_c*256+t; unsigned _row=_lin/_cpr; unsigned _col=(_lin%_cpr)<<4; unsigned _gk=(KB)+_col; \
        signed char* _wd=((signed char*)&sW[(BUF)][_row][0])+_col; \
        signed char* _ad=((signed char*)&sA[(BUF)][_row][0])+_col; \
        cp_async_pred_16(_wd, &B_i8[(unsigned long long)(cta_n+_row)*K+_gk], (cta_n+_row<N)&&(_gk+15<K)); \
        cp_async_pred_16(_ad, &A_i8[(unsigned long long)(cta_m+_row)*K+_gk], (cta_m+_row<M)&&(_gk+15<K)); \
    } } while(0)






#if !defined(__SCALE__)
extern "C" __global__
__launch_bounds__(256, 1)
void int8_gemm_faith8(
    const signed char* __restrict__ A_i8,
    const signed char* __restrict__ B_i8,
    const float* __restrict__ A_scale,
    const float* __restrict__ B_scale,
    __nv_bfloat16* __restrict__ C,
    unsigned int M, unsigned int N, unsigned int K
) {
    const unsigned int cta_n = blockIdx.x * 128;
    const unsigned int cta_m = blockIdx.y * 128;
    if (cta_m >= M || cta_n >= N) return;
    const unsigned int t = threadIdx.x;
    const unsigned int warp_id = t >> 5;
    const unsigned int lane = t & 31;
    const unsigned int ng = warp_id >> 1;
    const unsigned int mh = warp_id & 1;
    const unsigned int nb = K >> 5;

    __shared__ int   sW[2][128][F2W];
    __shared__ int   sA[2][128][F2W];
    __shared__ float sWs[128][F2_SB];
    __shared__ float sAs[128][F2_SB];

    float acc[2][8][4];
    #pragma unroll
    for (int n=0;n<2;n++) for(int j=0;j<8;j++){acc[n][j][0]=0;acc[n][j][1]=0;acc[n][j][2]=0;acc[n][j][3]=0;}

    const unsigned int NT = (K + F2_TILE - 1) / F2_TILE;


    F8_LOAD_TILE(0, 0u);
    cp_async_commit();

    for (unsigned int ki = 0; ki < NT; ki++) {
        const int cur = ki & 1;


        if (ki + 1 < NT) {
            F8_LOAD_TILE(cur ^ 1, (ki + 1) * F2_TILE);
            cp_async_commit();
            cp_async_wait_group<1>();
        } else {
            cp_async_wait_group<0>();
        }

        if (t < 128) {
            unsigned blk = (ki * F2_TILE) >> 5;
            #pragma unroll
            for (int s=0;s<F2_SB;s++){
                sWs[t][s] = (cta_n+t<N)?B_scale[(unsigned long long)(cta_n+t)*nb + blk + s]:0.f;
                sAs[t][s] = (cta_m+t<M)?A_scale[(unsigned long long)(cta_m+t)*nb + blk + s]:0.f;
            }
        }
        __syncthreads();

        #pragma unroll
        for (int sb=0; sb<F2_SB; sb++){
            unsigned WA[2][4];
            float    wsc[2][2];
            #pragma unroll
            for (int n=0;n<2;n++){
                unsigned wbase_row = ng*32 + n*16;
                const int* xs = &sW[cur][wbase_row][0] + (lane%16)*F2W + sb*8 + (lane/16)*4;
                asm volatile("ldmatrix.sync.aligned.m8n8.x4.b16 {%0,%1,%2,%3},[%4];"
                    : "=r"(WA[n][0]),"=r"(WA[n][1]),"=r"(WA[n][2]),"=r"(WA[n][3]) : "l"(xs));
                wsc[n][0] = sWs[wbase_row + lane/4][sb];
                wsc[n][1] = sWs[wbase_row + 8 + lane/4][sb];
            }
            #pragma unroll
            for (int j=0;j<8;j++){
                unsigned mcol0 = mh*64 + j*8;
                float asc0 = sAs[mcol0 + (lane%4)*2][sb];
                float asc1 = sAs[mcol0 + (lane%4)*2 + 1][sb];
                const int* abase = &sA[cur][mcol0 + lane/4][0] + sb*8;
                unsigned b0 = abase[lane%4];
                unsigned b1 = abase[lane%4 + 4];
                #pragma unroll
                for (int n=0;n<2;n++){
                    int s[4]={0,0,0,0};
                    METRALE_MMA_S8(s, WA[n][0],WA[n][1],WA[n][2],WA[n][3], b0,b1);
                    acc[n][j][0]+=(float)s[0]*wsc[n][0]*asc0;
                    acc[n][j][1]+=(float)s[1]*wsc[n][0]*asc1;
                    acc[n][j][2]+=(float)s[2]*wsc[n][1]*asc0;
                    acc[n][j][3]+=(float)s[3]*wsc[n][1]*asc1;
                }
            }
        }
        __syncthreads();
    }

    #pragma unroll
    for (int n=0;n<2;n++){
        unsigned nrow0 = cta_n + ng*32 + n*16 + lane/4;
        #pragma unroll
        for (int j=0;j<8;j++){
            unsigned mcol = cta_m + mh*64 + j*8 + (lane%4)*2;
            unsigned cN0=nrow0, cN1=nrow0+8;
            if (mcol<M   && cN0<N) C[(unsigned long long)mcol*N + cN0]     = __float2bfloat16(acc[n][j][0]);
            if (mcol+1<M && cN0<N) C[(unsigned long long)(mcol+1)*N + cN0] = __float2bfloat16(acc[n][j][1]);
            if (mcol<M   && cN1<N) C[(unsigned long long)mcol*N + cN1]     = __float2bfloat16(acc[n][j][2]);
            if (mcol+1<M && cN1<N) C[(unsigned long long)(mcol+1)*N + cN1] = __float2bfloat16(acc[n][j][3]);
        }
    }
}
#endif
#undef METRALE_MMA_S8
#undef F8_LOAD_TILE
#undef F2_TILE
#undef F2_SB
#undef F2W

// 2026-09-25: int8_gemm_faith2 with a K tile of 256 (eight 32-wide sub-blocks) and rows of F2W = 68 int32: r * 17
// mod 8 is a permutation, so the eight rows of an ldmatrix 8x8 matrix fall in distinct 16-byte bank groups.
// K must be a multiple of 256. Not built under __SCALE__.





#define F2_TILE 256
#define F2_SB   (F2_TILE/32)
#define F2W     68
#define METRALE_MMA_S8(d, a0,a1,a2,a3, b0,b1) \
    asm volatile("mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 " \
        "{%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%10,%11,%12,%13};" \
        : "=r"((d)[0]), "=r"((d)[1]), "=r"((d)[2]), "=r"((d)[3]) \
        : "r"(a0),"r"(a1),"r"(a2),"r"(a3),"r"(b0),"r"(b1), \
          "r"((d)[0]),"r"((d)[1]),"r"((d)[2]),"r"((d)[3]))






#if !defined(__SCALE__)
extern "C" __global__
__launch_bounds__(256, 1)
void int8_gemm_faith9(
    const signed char* __restrict__ A_i8,
    const signed char* __restrict__ B_i8,
    const float* __restrict__ A_scale,
    const float* __restrict__ B_scale,
    __nv_bfloat16* __restrict__ C,
    unsigned int M, unsigned int N, unsigned int K
) {
    const unsigned int cta_n = blockIdx.x * 128;
    const unsigned int cta_m = blockIdx.y * 128;
    if (cta_m >= M || cta_n >= N) return;
    const unsigned int t = threadIdx.x;
    const unsigned int warp_id = t >> 5;
    const unsigned int lane = t & 31;
    const unsigned int ng = warp_id >> 1;
    const unsigned int mh = warp_id & 1;
    const unsigned int nb = K >> 5;

    __shared__ int   sW[128][F2W];
    __shared__ int   sA[128][F2W];
    __shared__ float sWs[128][F2_SB];
    __shared__ float sAs[128][F2_SB];

    float acc[2][8][4];
    #pragma unroll
    for (int n=0;n<2;n++) for(int j=0;j<8;j++){acc[n][j][0]=0;acc[n][j][1]=0;acc[n][j][2]=0;acc[n][j][3]=0;}

    for (unsigned int kb = 0; kb < K; kb += F2_TILE) {
        const unsigned F2_CPR = F2_TILE/16;
        #pragma unroll
        for (int c = 0; c < F2_TILE/32; c++) {
            unsigned lin = c*256 + t;
            unsigned row = lin / F2_CPR;
            unsigned col = (lin % F2_CPR) << 4;
            unsigned gk = kb + col;
            signed char* wdst = ((signed char*)&sW[row][0]) + col;
            signed char* adst = ((signed char*)&sA[row][0]) + col;
            cp_async_pred_16(wdst, &B_i8[(unsigned long long)(cta_n+row)*K + gk], (cta_n+row<N)&&(gk+15<K));
            cp_async_pred_16(adst, &A_i8[(unsigned long long)(cta_m+row)*K + gk], (cta_m+row<M)&&(gk+15<K));
        }
        if (t < 128) {
            unsigned blk = kb >> 5;
            #pragma unroll
            for (int s=0;s<F2_SB;s++){
                sWs[t][s] = (cta_n+t<N)?B_scale[(unsigned long long)(cta_n+t)*nb + blk + s]:0.f;
                sAs[t][s] = (cta_m+t<M)?A_scale[(unsigned long long)(cta_m+t)*nb + blk + s]:0.f;
            }
        }
        cp_async_commit(); cp_async_wait_all(); __syncthreads();

        #pragma unroll
        for (int sb=0; sb<F2_SB; sb++){
            unsigned WA[2][4];
            float    wsc[2][2];
            #pragma unroll
            for (int n=0;n<2;n++){
                unsigned wbase_row = ng*32 + n*16;
                const int* xs = &sW[wbase_row][0] + (lane%16)*F2W + sb*8 + (lane/16)*4;
                asm volatile("ldmatrix.sync.aligned.m8n8.x4.b16 {%0,%1,%2,%3},[%4];"
                    : "=r"(WA[n][0]),"=r"(WA[n][1]),"=r"(WA[n][2]),"=r"(WA[n][3]) : "l"(xs));
                wsc[n][0] = sWs[wbase_row + lane/4][sb];
                wsc[n][1] = sWs[wbase_row + 8 + lane/4][sb];
            }
            #pragma unroll
            for (int j=0;j<8;j++){
                unsigned mcol0 = mh*64 + j*8;
                float asc0 = sAs[mcol0 + (lane%4)*2][sb];
                float asc1 = sAs[mcol0 + (lane%4)*2 + 1][sb];
                const int* abase = &sA[mcol0 + lane/4][0] + sb*8;
                unsigned b0 = abase[lane%4];
                unsigned b1 = abase[lane%4 + 4];
                #pragma unroll
                for (int n=0;n<2;n++){
                    int s[4]={0,0,0,0};
                    METRALE_MMA_S8(s, WA[n][0],WA[n][1],WA[n][2],WA[n][3], b0,b1);
                    acc[n][j][0]+=(float)s[0]*wsc[n][0]*asc0;
                    acc[n][j][1]+=(float)s[1]*wsc[n][0]*asc1;
                    acc[n][j][2]+=(float)s[2]*wsc[n][1]*asc0;
                    acc[n][j][3]+=(float)s[3]*wsc[n][1]*asc1;
                }
            }
        }
        __syncthreads();
    }

    #pragma unroll
    for (int n=0;n<2;n++){
        unsigned nrow0 = cta_n + ng*32 + n*16 + lane/4;
        #pragma unroll
        for (int j=0;j<8;j++){
            unsigned mcol = cta_m + mh*64 + j*8 + (lane%4)*2;
            unsigned cN0=nrow0, cN1=nrow0+8;
            if (mcol<M   && cN0<N) C[(unsigned long long)mcol*N + cN0]     = __float2bfloat16(acc[n][j][0]);
            if (mcol+1<M && cN0<N) C[(unsigned long long)(mcol+1)*N + cN0] = __float2bfloat16(acc[n][j][1]);
            if (mcol<M   && cN1<N) C[(unsigned long long)mcol*N + cN1]     = __float2bfloat16(acc[n][j][2]);
            if (mcol+1<M && cN1<N) C[(unsigned long long)(mcol+1)*N + cN1] = __float2bfloat16(acc[n][j][3]);
        }
    }
}
#endif
#undef METRALE_MMA_S8
#undef F2_TILE
#undef F2_SB
#undef F2W

// 2026-09-25: int8_gemm_faith2 with each of the eight warps owning one 16-row weight tile and all 128 tokens
// (sixteen 8-token chunks), so each weight fragment feeds 16 MMAs.










#define F2_TILE 128
#define F2_SB   (F2_TILE/32)
#define F2W     36
#define METRALE_MMA_S8(d, a0,a1,a2,a3, b0,b1) \
    asm volatile("mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 " \
        "{%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%10,%11,%12,%13};" \
        : "=r"((d)[0]), "=r"((d)[1]), "=r"((d)[2]), "=r"((d)[3]) \
        : "r"(a0),"r"(a1),"r"(a2),"r"(a3),"r"(b0),"r"(b1), \
          "r"((d)[0]),"r"((d)[1]),"r"((d)[2]),"r"((d)[3]))
extern "C" __global__
__launch_bounds__(256, 1)
void int8_gemm_faith10(
    const signed char* __restrict__ A_i8,
    const signed char* __restrict__ B_i8,
    const float* __restrict__ A_scale,
    const float* __restrict__ B_scale,
    __nv_bfloat16* __restrict__ C,
    unsigned int M, unsigned int N, unsigned int K
) {
    const unsigned int cta_n = blockIdx.x * 128;
    const unsigned int cta_m = blockIdx.y * 128;
    if (cta_m >= M || cta_n >= N) return;
    const unsigned int t = threadIdx.x;
    const unsigned int warp_id = t >> 5;
    const unsigned int lane = t & 31;
    const unsigned int ng = warp_id;
    const unsigned int nb = K >> 5;

    __shared__ int   sW[128][F2W];
    __shared__ int   sA[128][F2W];
    __shared__ float sWs[128][F2_SB];
    __shared__ float sAs[128][F2_SB];

    float acc[16][4];
    #pragma unroll
    for (int j=0;j<16;j++){acc[j][0]=0;acc[j][1]=0;acc[j][2]=0;acc[j][3]=0;}

    for (unsigned int kb = 0; kb < K; kb += F2_TILE) {
        const unsigned F2_CPR = F2_TILE/16;
        #pragma unroll
        for (int c = 0; c < F2_TILE/32; c++) {
            unsigned lin = c*256 + t;
            unsigned row = lin / F2_CPR;
            unsigned col = (lin % F2_CPR) << 4;
            unsigned gk = kb + col;
            signed char* wdst = ((signed char*)&sW[row][0]) + col;
            signed char* adst = ((signed char*)&sA[row][0]) + col;
            cp_async_pred_16(wdst, &B_i8[(unsigned long long)(cta_n+row)*K + gk], (cta_n+row<N)&&(gk+15<K));
            cp_async_pred_16(adst, &A_i8[(unsigned long long)(cta_m+row)*K + gk], (cta_m+row<M)&&(gk+15<K));
        }
        if (t < 128) {
            unsigned blk = kb >> 5;
            #pragma unroll
            for (int s=0;s<F2_SB;s++){
                sWs[t][s] = (cta_n+t<N)?B_scale[(unsigned long long)(cta_n+t)*nb + blk + s]:0.f;
                sAs[t][s] = (cta_m+t<M)?A_scale[(unsigned long long)(cta_m+t)*nb + blk + s]:0.f;
            }
        }
        cp_async_commit(); cp_async_wait_all(); __syncthreads();

        #pragma unroll
        for (int sb=0; sb<F2_SB; sb++){

            unsigned WA[4];
            unsigned wbase_row = ng*16;
            const int* xs = &sW[wbase_row][0] + (lane%16)*F2W + sb*8 + (lane/16)*4;
            asm volatile("ldmatrix.sync.aligned.m8n8.x4.b16 {%0,%1,%2,%3},[%4];"
                : "=r"(WA[0]),"=r"(WA[1]),"=r"(WA[2]),"=r"(WA[3]) : "l"(xs));
            float wsc0 = sWs[wbase_row + lane/4][sb];
            float wsc1 = sWs[wbase_row + 8 + lane/4][sb];
            #pragma unroll
            for (int tc=0; tc<16; tc++){
                unsigned mcol0 = tc*8;
                float asc0 = sAs[mcol0 + (lane%4)*2][sb];
                float asc1 = sAs[mcol0 + (lane%4)*2 + 1][sb];
                const int* abase = &sA[mcol0 + lane/4][0] + sb*8;
                unsigned b0 = abase[lane%4];
                unsigned b1 = abase[lane%4 + 4];
                int s[4]={0,0,0,0};
                METRALE_MMA_S8(s, WA[0],WA[1],WA[2],WA[3], b0,b1);
                acc[tc][0]+=(float)s[0]*wsc0*asc0;
                acc[tc][1]+=(float)s[1]*wsc0*asc1;
                acc[tc][2]+=(float)s[2]*wsc1*asc0;
                acc[tc][3]+=(float)s[3]*wsc1*asc1;
            }
        }
        __syncthreads();
    }

    unsigned nrow0 = cta_n + ng*16 + lane/4;
    unsigned cN0 = nrow0, cN1 = nrow0 + 8;
    #pragma unroll
    for (int tc=0; tc<16; tc++){
        unsigned mcol = cta_m + tc*8 + (lane%4)*2;
        if (mcol<M   && cN0<N) C[(unsigned long long)mcol*N + cN0]     = __float2bfloat16(acc[tc][0]);
        if (mcol+1<M && cN0<N) C[(unsigned long long)(mcol+1)*N + cN0] = __float2bfloat16(acc[tc][1]);
        if (mcol<M   && cN1<N) C[(unsigned long long)mcol*N + cN1]     = __float2bfloat16(acc[tc][2]);
        if (mcol+1<M && cN1<N) C[(unsigned long long)(mcol+1)*N + cN1] = __float2bfloat16(acc[tc][3]);
    }
}
#undef METRALE_MMA_S8
#undef F2_TILE
#undef F2_SB
#undef F2W

// 2026-09-25: int8 GEMM over a 256-wide K step: the weight tile (rows of MQF_WSTRIDE = 76 int32: 64 data, then the
// row's eight per-32 scales as floats at int32 offset 64, then 4 pad) is loaded once per step, and the 128-wide
// activation tile is loaded once for each half of the step. Each half first loads the warp's weight fragments and
// scales for its four sub-blocks into registers. Warp tiling and scale math as in int8_gemm_faith2. K must be a
// multiple of 256.





















#define MQF_WSTRIDE 76
#define MQF_MMA_S8(d, a0,a1,a2,a3, b0,b1) \
    asm volatile("mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 " \
        "{%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%10,%11,%12,%13};" \
        : "=r"((d)[0]), "=r"((d)[1]), "=r"((d)[2]), "=r"((d)[3]) \
        : "r"(a0),"r"(a1),"r"(a2),"r"(a3),"r"(b0),"r"(b1), \
          "r"((d)[0]),"r"((d)[1]),"r"((d)[2]),"r"((d)[3]))

extern "C" __global__
__launch_bounds__(256, 1)
void int8_gemm_mmqf(
    const signed char* __restrict__ A_i8,
    const signed char* __restrict__ B_i8,
    const float* __restrict__ A_scale,
    const float* __restrict__ B_scale,
    __nv_bfloat16* __restrict__ C,
    unsigned int M, unsigned int N, unsigned int K)
{
    const unsigned int cta_n = blockIdx.x * 128;
    const unsigned int cta_m = blockIdx.y * 128;
    if (cta_m >= M || cta_n >= N) return;
    const unsigned int t       = threadIdx.x;
    const unsigned int warp_id = t >> 5;
    const unsigned int lane    = t & 31;
    const unsigned int ng      = warp_id >> 1;
    const unsigned int mh      = warp_id & 1;
    const unsigned int nb      = K >> 5;

    __shared__ int   sW[128][MQF_WSTRIDE];
    __shared__ int   sA[128][36];
    __shared__ float sAs[128][4];

    float acc[2][8][4];
    #pragma unroll
    for (int n=0;n<2;n++) for (int j=0;j<8;j++) { acc[n][j][0]=0;acc[n][j][1]=0;acc[n][j][2]=0;acc[n][j][3]=0; }

    unsigned WAarr[2][4][4];

    // 2026-09-25: One 128-wide K half of the resident weight tile: PP selects the half (int32 offset PP * 32, scale
    // index 64 + PP * 4 + sb). sA and sAs hold only the current half, so their offsets do not depend on PP.

    #define MQF_COMPUTE_PASS(PP) do {                                                                   \
        float    wsc[2][2][4];                                                                          \
        _Pragma("unroll")                                                                               \
        for (int n=0;n<2;n++){                                                                          \
            unsigned wrow = ng*32 + n*16;                                                               \
            _Pragma("unroll")                                                                           \
            for (int sb=0; sb<4; sb++){                                                                 \
                const int* xs = &sW[wrow][0] + (lane%16)*MQF_WSTRIDE + ((PP)*32 + sb*8) + (lane/16)*4;  \
                unsigned f0,f1,f2,f3;                                                                   \
                asm volatile("ldmatrix.sync.aligned.m8n8.x4.b16 {%0,%1,%2,%3},[%4];"                    \
                    : "=r"(f0),"=r"(f1),"=r"(f2),"=r"(f3) : "l"(xs));                                   \
                wsc[n][0][sb] = ((const float*)&sW[wrow     + lane/4][0])[64 + (PP)*4 + sb];            \
                wsc[n][1][sb] = ((const float*)&sW[wrow + 8 + lane/4][0])[64 + (PP)*4 + sb];            \
                WAarr[n][sb][0]=f0; WAarr[n][sb][1]=f1; WAarr[n][sb][2]=f2; WAarr[n][sb][3]=f3;         \
            }                                                                                          \
        }                                                                                              \
        _Pragma("unroll")                                                                               \
        for (int jj=0; jj<8; jj++){                                                                     \
            unsigned mcol0 = mh*64 + jj*8;                                                              \
            _Pragma("unroll")                                                                           \
            for (int sb=0; sb<4; sb++){                                                                 \
                const int* abase = &sA[mcol0 + lane/4][0] + sb*8;                                       \
                unsigned b0 = abase[lane%4];                                                            \
                unsigned b1 = abase[lane%4 + 4];                                                        \
                float asc0 = sAs[mcol0 + (lane%4)*2    ][sb];                                           \
                float asc1 = sAs[mcol0 + (lane%4)*2 + 1][sb];                                           \
                _Pragma("unroll")                                                                       \
                for (int n=0;n<2;n++){                                                                  \
                    int s[4]={0,0,0,0};                                                                 \
                    MQF_MMA_S8(s, WAarr[n][sb][0],WAarr[n][sb][1],WAarr[n][sb][2],WAarr[n][sb][3], b0,b1);\
                    acc[n][jj][0]+=(float)s[0]*wsc[n][0][sb]*asc0;                                      \
                    acc[n][jj][1]+=(float)s[1]*wsc[n][0][sb]*asc1;                                      \
                    acc[n][jj][2]+=(float)s[2]*wsc[n][1][sb]*asc0;                                      \
                    acc[n][jj][3]+=(float)s[3]*wsc[n][1][sb]*asc1;                                      \
                }                                                                                      \
            }                                                                                          \
        }                                                                                              \
    } while(0)

    for (unsigned int kb = 0; kb < K; kb += 256) {

        #pragma unroll
        for (int c=0; c<8; c++){
            unsigned lin = c*256 + t;
            unsigned row = lin >> 4;
            unsigned col = (lin & 15) << 4;
            unsigned gk  = kb + col;
            cp_async_pred_16(((signed char*)&sW[row][0]) + col,
                             &B_i8[(unsigned long long)(cta_n+row)*K + gk],
                             (cta_n+row<N) && (gk+15<K));
        }

        if (t < 128) {
            unsigned blk = kb >> 5;
            #pragma unroll
            for (int b=0;b<8;b++)
                ((float*)&sW[t][0])[64+b] = (cta_n+t<N) ? B_scale[(unsigned long long)(cta_n+t)*nb + blk + b] : 0.f;
        }

        #pragma unroll
        for (int c=0; c<4; c++){
            unsigned lin = c*256 + t;
            unsigned row = lin >> 3;
            unsigned col = (lin & 7) << 4;
            unsigned gk  = kb + col;
            cp_async_pred_16(((signed char*)&sA[row][0]) + col,
                             &A_i8[(unsigned long long)(cta_m+row)*K + gk],
                             (cta_m+row<M) && (gk+15<K));
        }
        if (t < 128) {
            unsigned blk = kb >> 5;
            #pragma unroll
            for (int s=0;s<4;s++)
                sAs[t][s] = (cta_m+t<M) ? A_scale[(unsigned long long)(cta_m+t)*nb + blk + s] : 0.f;
        }
        cp_async_commit(); cp_async_wait_all(); __syncthreads();

        MQF_COMPUTE_PASS(0);
        __syncthreads();   // 2026-09-25: every warp is done with sA and sAs before the second half overwrites them


        #pragma unroll
        for (int c=0; c<4; c++){
            unsigned lin = c*256 + t;
            unsigned row = lin >> 3;
            unsigned col = (lin & 7) << 4;
            unsigned gk  = kb + 128 + col;
            cp_async_pred_16(((signed char*)&sA[row][0]) + col,
                             &A_i8[(unsigned long long)(cta_m+row)*K + gk],
                             (cta_m+row<M) && (gk+15<K));
        }
        if (t < 128) {
            unsigned blk = (kb >> 5) + 4;
            #pragma unroll
            for (int s=0;s<4;s++)
                sAs[t][s] = (cta_m+t<M) ? A_scale[(unsigned long long)(cta_m+t)*nb + blk + s] : 0.f;
        }
        cp_async_commit(); cp_async_wait_all(); __syncthreads();

        MQF_COMPUTE_PASS(1);
        __syncthreads();   // 2026-09-25: every warp is done with sW and sA before the next step overwrites them
    }
    #undef MQF_COMPUTE_PASS


    #pragma unroll
    for (int n=0;n<2;n++){
        unsigned nrow0 = cta_n + ng*32 + n*16 + lane/4;
        #pragma unroll
        for (int j=0;j<8;j++){
            unsigned mcol = cta_m + mh*64 + j*8 + (lane%4)*2;
            unsigned cN0=nrow0, cN1=nrow0+8;
            if (mcol<M   && cN0<N) C[(unsigned long long)mcol*N + cN0]     = __float2bfloat16(acc[n][j][0]);
            if (mcol+1<M && cN0<N) C[(unsigned long long)(mcol+1)*N + cN0] = __float2bfloat16(acc[n][j][1]);
            if (mcol<M   && cN1<N) C[(unsigned long long)mcol*N + cN1]     = __float2bfloat16(acc[n][j][2]);
            if (mcol+1<M && cN1<N) C[(unsigned long long)(mcol+1)*N + cN1] = __float2bfloat16(acc[n][j][3]);
        }
    }
}
#undef MQF_MMA_S8
#undef MQF_WSTRIDE

// 2026-09-25: int8_gemm_mmqf with every cp.async copy replaced by a synchronous 16-byte copy (mqf2_load_pred_16)
// and one __syncthreads after each load phase; the layout, indexing, predicates and math are int8_gemm_mmqf's.






















#define MQF2_WSTRIDE 76
#define MQF2_MMA_S8(d, a0,a1,a2,a3, b0,b1) \
    asm volatile("mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 " \
        "{%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%10,%11,%12,%13};" \
        : "=r"((d)[0]), "=r"((d)[1]), "=r"((d)[2]), "=r"((d)[3]) \
        : "r"(a0),"r"(a1),"r"(a2),"r"(a3),"r"(b0),"r"(b1), \
          "r"((d)[0]),"r"((d)[1]),"r"((d)[2]),"r"((d)[3]))

// 2026-09-25: Synchronous twin of cp_async_pred_16: copies 16 bytes when pred holds, otherwise stores zeros
// without reading src_gmem.

__device__ __forceinline__ void mqf2_load_pred_16(void* dst_smem, const void* src_gmem, bool pred) {
    int4 v = pred ? *(const int4*)src_gmem : make_int4(0,0,0,0);
    *(int4*)dst_smem = v;
}

extern "C" __global__
__launch_bounds__(256, 1)
void int8_gemm_mmqf2(
    const signed char* __restrict__ A_i8,
    const signed char* __restrict__ B_i8,
    const float* __restrict__ A_scale,
    const float* __restrict__ B_scale,
    __nv_bfloat16* __restrict__ C,
    unsigned int M, unsigned int N, unsigned int K)
{
    const unsigned int cta_n = blockIdx.x * 128;
    const unsigned int cta_m = blockIdx.y * 128;
    if (cta_m >= M || cta_n >= N) return;
    const unsigned int t       = threadIdx.x;
    const unsigned int warp_id = t >> 5;
    const unsigned int lane    = t & 31;
    const unsigned int ng      = warp_id >> 1;
    const unsigned int mh      = warp_id & 1;
    const unsigned int nb      = K >> 5;

    __shared__ int   sW[128][MQF2_WSTRIDE];
    __shared__ int   sA[128][36];
    __shared__ float sAs[128][4];

    float acc[2][8][4];
    #pragma unroll
    for (int n=0;n<2;n++) for (int j=0;j<8;j++) { acc[n][j][0]=0;acc[n][j][1]=0;acc[n][j][2]=0;acc[n][j][3]=0; }

    unsigned WAarr[2][4][4];




    #define MQF2_COMPUTE_PASS(PP) do {                                                                  \
        float    wsc[2][2][4];                                                                          \
        _Pragma("unroll")                                                                               \
        for (int n=0;n<2;n++){                                                                          \
            unsigned wrow = ng*32 + n*16;                                                               \
            _Pragma("unroll")                                                                           \
            for (int sb=0; sb<4; sb++){                                                                 \
                const int* xs = &sW[wrow][0] + (lane%16)*MQF2_WSTRIDE + ((PP)*32 + sb*8) + (lane/16)*4; \
                unsigned f0,f1,f2,f3;                                                                   \
                asm volatile("ldmatrix.sync.aligned.m8n8.x4.b16 {%0,%1,%2,%3},[%4];"                    \
                    : "=r"(f0),"=r"(f1),"=r"(f2),"=r"(f3) : "l"(xs));                                   \
                wsc[n][0][sb] = ((const float*)&sW[wrow     + lane/4][0])[64 + (PP)*4 + sb];            \
                wsc[n][1][sb] = ((const float*)&sW[wrow + 8 + lane/4][0])[64 + (PP)*4 + sb];            \
                WAarr[n][sb][0]=f0; WAarr[n][sb][1]=f1; WAarr[n][sb][2]=f2; WAarr[n][sb][3]=f3;         \
            }                                                                                          \
        }                                                                                              \
        _Pragma("unroll")                                                                               \
        for (int jj=0; jj<8; jj++){                                                                     \
            unsigned mcol0 = mh*64 + jj*8;                                                              \
            _Pragma("unroll")                                                                           \
            for (int sb=0; sb<4; sb++){                                                                 \
                const int* abase = &sA[mcol0 + lane/4][0] + sb*8;                                       \
                unsigned b0 = abase[lane%4];                                                            \
                unsigned b1 = abase[lane%4 + 4];                                                        \
                float asc0 = sAs[mcol0 + (lane%4)*2    ][sb];                                           \
                float asc1 = sAs[mcol0 + (lane%4)*2 + 1][sb];                                           \
                _Pragma("unroll")                                                                       \
                for (int n=0;n<2;n++){                                                                  \
                    int s[4]={0,0,0,0};                                                                 \
                    MQF2_MMA_S8(s, WAarr[n][sb][0],WAarr[n][sb][1],WAarr[n][sb][2],WAarr[n][sb][3], b0,b1);\
                    acc[n][jj][0]+=(float)s[0]*wsc[n][0][sb]*asc0;                                      \
                    acc[n][jj][1]+=(float)s[1]*wsc[n][0][sb]*asc1;                                      \
                    acc[n][jj][2]+=(float)s[2]*wsc[n][1][sb]*asc0;                                      \
                    acc[n][jj][3]+=(float)s[3]*wsc[n][1][sb]*asc1;                                      \
                }                                                                                      \
            }                                                                                          \
        }                                                                                              \
    } while(0)

    for (unsigned int kb = 0; kb < K; kb += 256) {

        #pragma unroll
        for (int c=0; c<8; c++){
            unsigned lin = c*256 + t;
            unsigned row = lin >> 4;
            unsigned col = (lin & 15) << 4;
            unsigned gk  = kb + col;
            mqf2_load_pred_16(((signed char*)&sW[row][0]) + col,
                              &B_i8[(unsigned long long)(cta_n+row)*K + gk],
                              (cta_n+row<N) && (gk+15<K));
        }

        if (t < 128) {
            unsigned blk = kb >> 5;
            #pragma unroll
            for (int b=0;b<8;b++)
                ((float*)&sW[t][0])[64+b] = (cta_n+t<N) ? B_scale[(unsigned long long)(cta_n+t)*nb + blk + b] : 0.f;
        }

        #pragma unroll
        for (int c=0; c<4; c++){
            unsigned lin = c*256 + t;
            unsigned row = lin >> 3;
            unsigned col = (lin & 7) << 4;
            unsigned gk  = kb + col;
            mqf2_load_pred_16(((signed char*)&sA[row][0]) + col,
                              &A_i8[(unsigned long long)(cta_m+row)*K + gk],
                              (cta_m+row<M) && (gk+15<K));
        }
        if (t < 128) {
            unsigned blk = kb >> 5;
            #pragma unroll
            for (int s=0;s<4;s++)
                sAs[t][s] = (cta_m+t<M) ? A_scale[(unsigned long long)(cta_m+t)*nb + blk + s] : 0.f;
        }
        __syncthreads();

        MQF2_COMPUTE_PASS(0);
        __syncthreads();   // 2026-09-25: every warp is done with sA and sAs before the second half overwrites them


        #pragma unroll
        for (int c=0; c<4; c++){
            unsigned lin = c*256 + t;
            unsigned row = lin >> 3;
            unsigned col = (lin & 7) << 4;
            unsigned gk  = kb + 128 + col;
            mqf2_load_pred_16(((signed char*)&sA[row][0]) + col,
                              &A_i8[(unsigned long long)(cta_m+row)*K + gk],
                              (cta_m+row<M) && (gk+15<K));
        }
        if (t < 128) {
            unsigned blk = (kb >> 5) + 4;
            #pragma unroll
            for (int s=0;s<4;s++)
                sAs[t][s] = (cta_m+t<M) ? A_scale[(unsigned long long)(cta_m+t)*nb + blk + s] : 0.f;
        }
        __syncthreads();

        MQF2_COMPUTE_PASS(1);
        __syncthreads();   // 2026-09-25: every warp is done with sW and sA before the next step overwrites them
    }
    #undef MQF2_COMPUTE_PASS


    #pragma unroll
    for (int n=0;n<2;n++){
        unsigned nrow0 = cta_n + ng*32 + n*16 + lane/4;
        #pragma unroll
        for (int j=0;j<8;j++){
            unsigned mcol = cta_m + mh*64 + j*8 + (lane%4)*2;
            unsigned cN0=nrow0, cN1=nrow0+8;
            if (mcol<M   && cN0<N) C[(unsigned long long)mcol*N + cN0]     = __float2bfloat16(acc[n][j][0]);
            if (mcol+1<M && cN0<N) C[(unsigned long long)(mcol+1)*N + cN0] = __float2bfloat16(acc[n][j][1]);
            if (mcol<M   && cN1<N) C[(unsigned long long)mcol*N + cN1]     = __float2bfloat16(acc[n][j][2]);
            if (mcol+1<M && cN1<N) C[(unsigned long long)(mcol+1)*N + cN1] = __float2bfloat16(acc[n][j][3]);
        }
    }
}
#undef MQF2_MMA_S8
#undef MQF2_WSTRIDE

// 2026-09-25: int8_gemm_mmqf2 with a different MMA schedule: for each 8-token chunk all eight MMAs go into separate
// int32 results (sbank) before any is scaled, and the next chunk's activation fragments and scales are first
// loaded into the other slot of a two-slot register buffer (bfr, afr). Each accumulator element is scaled and
// summed in the same order (sb outer, n inner) as in int8_gemm_mmqf2.

































#define MQF3_WSTRIDE 76
#define MQF3_MMA_S8(d, a0,a1,a2,a3, b0,b1) \
    asm volatile("mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 " \
        "{%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%10,%11,%12,%13};" \
        : "=r"((d)[0]), "=r"((d)[1]), "=r"((d)[2]), "=r"((d)[3]) \
        : "r"(a0),"r"(a1),"r"(a2),"r"(a3),"r"(b0),"r"(b1), \
          "r"((d)[0]),"r"((d)[1]),"r"((d)[2]),"r"((d)[3]))

extern "C" __global__
__launch_bounds__(256, 1)
void int8_gemm_mmqf3(
    const signed char* __restrict__ A_i8,
    const signed char* __restrict__ B_i8,
    const float* __restrict__ A_scale,
    const float* __restrict__ B_scale,
    __nv_bfloat16* __restrict__ C,
    unsigned int M, unsigned int N, unsigned int K)
{
    const unsigned int cta_n = blockIdx.x * 128;
    const unsigned int cta_m = blockIdx.y * 128;
    if (cta_m >= M || cta_n >= N) return;
    const unsigned int t       = threadIdx.x;
    const unsigned int warp_id = t >> 5;
    const unsigned int lane    = t & 31;
    const unsigned int ng      = warp_id >> 1;
    const unsigned int mh      = warp_id & 1;
    const unsigned int nb      = K >> 5;

    __shared__ int   sW[128][MQF3_WSTRIDE];
    __shared__ int   sA[128][36];
    __shared__ float sAs[128][4];

    float acc[2][8][4];
    #pragma unroll
    for (int n=0;n<2;n++) for (int j=0;j<8;j++) { acc[n][j][0]=0;acc[n][j][1]=0;acc[n][j][2]=0;acc[n][j][3]=0; }

    unsigned WAarr[2][4][4];

    // 2026-09-25: Loads token chunk JJ's activation fragments and scales into slot P of bfr and afr.


    #define MQF3_LOAD_B(P, JJ) do {                                              \
        unsigned mcol0 = mh*64 + (JJ)*8;                                         \
        _Pragma("unroll")                                                        \
        for (int sb=0; sb<4; sb++){                                             \
            const int* abase = &sA[mcol0 + lane/4][0] + sb*8;                    \
            bfr[P][sb][0] = abase[lane%4];                                       \
            bfr[P][sb][1] = abase[lane%4 + 4];                                   \
            afr[P][sb][0] = sAs[mcol0 + (lane%4)*2    ][sb];                     \
            afr[P][sb][1] = sAs[mcol0 + (lane%4)*2 + 1][sb];                     \
        }                                                                        \
    } while(0)




    #define MQF3_COMPUTE_PASS(PP) do {                                                                  \
        float    wsc[2][2][4];                                                                          \
        \
        _Pragma("unroll")                                                                               \
        for (int n=0;n<2;n++){                                                                          \
            unsigned wrow = ng*32 + n*16;                                                               \
            _Pragma("unroll")                                                                           \
            for (int sb=0; sb<4; sb++){                                                                 \
                const int* xs = &sW[wrow][0] + (lane%16)*MQF3_WSTRIDE + ((PP)*32 + sb*8) + (lane/16)*4; \
                unsigned f0,f1,f2,f3;                                                                   \
                asm volatile("ldmatrix.sync.aligned.m8n8.x4.b16 {%0,%1,%2,%3},[%4];"                    \
                    : "=r"(f0),"=r"(f1),"=r"(f2),"=r"(f3) : "l"(xs));                                   \
                wsc[n][0][sb] = ((const float*)&sW[wrow     + lane/4][0])[64 + (PP)*4 + sb];            \
                wsc[n][1][sb] = ((const float*)&sW[wrow + 8 + lane/4][0])[64 + (PP)*4 + sb];            \
                WAarr[n][sb][0]=f0; WAarr[n][sb][1]=f1; WAarr[n][sb][2]=f2; WAarr[n][sb][3]=f3;         \
            }                                                                                          \
        }                                                                                              \
        unsigned bfr[2][4][2];                                                                          \
        float    afr[2][4][2];                                                                          \
        MQF3_LOAD_B(0, 0);                                                                              \
        _Pragma("unroll")                                                                               \
        for (int jj=0; jj<8; jj++){                                                                     \
            const int p = jj & 1;                                                                       \
            if (jj < 7) { MQF3_LOAD_B(p^1, jj+1); }                                                     \
            \
            int sbank[4][2][4];                                                                         \
            _Pragma("unroll")                                                                           \
            for (int sb=0; sb<4; sb++){                                                                 \
                _Pragma("unroll")                                                                       \
                for (int n=0; n<2; n++){                                                                \
                    sbank[sb][n][0]=0; sbank[sb][n][1]=0; sbank[sb][n][2]=0; sbank[sb][n][3]=0;         \
                    MQF3_MMA_S8(sbank[sb][n], WAarr[n][sb][0],WAarr[n][sb][1],WAarr[n][sb][2],WAarr[n][sb][3], \
                                bfr[p][sb][0], bfr[p][sb][1]);                                          \
                }                                                                                      \
            }                                                                                          \
            \
            _Pragma("unroll")                                                                           \
            for (int sb=0; sb<4; sb++){                                                                 \
                _Pragma("unroll")                                                                       \
                for (int n=0; n<2; n++){                                                                \
                    acc[n][jj][0]+=(float)sbank[sb][n][0]*wsc[n][0][sb]*afr[p][sb][0];                  \
                    acc[n][jj][1]+=(float)sbank[sb][n][1]*wsc[n][0][sb]*afr[p][sb][1];                  \
                    acc[n][jj][2]+=(float)sbank[sb][n][2]*wsc[n][1][sb]*afr[p][sb][0];                  \
                    acc[n][jj][3]+=(float)sbank[sb][n][3]*wsc[n][1][sb]*afr[p][sb][1];                  \
                }                                                                                      \
            }                                                                                          \
        }                                                                                              \
    } while(0)

    for (unsigned int kb = 0; kb < K; kb += 256) {

        #pragma unroll
        for (int c=0; c<8; c++){
            unsigned lin = c*256 + t;
            unsigned row = lin >> 4;
            unsigned col = (lin & 15) << 4;
            unsigned gk  = kb + col;
            mqf2_load_pred_16(((signed char*)&sW[row][0]) + col,
                              &B_i8[(unsigned long long)(cta_n+row)*K + gk],
                              (cta_n+row<N) && (gk+15<K));
        }

        if (t < 128) {
            unsigned blk = kb >> 5;
            #pragma unroll
            for (int b=0;b<8;b++)
                ((float*)&sW[t][0])[64+b] = (cta_n+t<N) ? B_scale[(unsigned long long)(cta_n+t)*nb + blk + b] : 0.f;
        }

        #pragma unroll
        for (int c=0; c<4; c++){
            unsigned lin = c*256 + t;
            unsigned row = lin >> 3;
            unsigned col = (lin & 7) << 4;
            unsigned gk  = kb + col;
            mqf2_load_pred_16(((signed char*)&sA[row][0]) + col,
                              &A_i8[(unsigned long long)(cta_m+row)*K + gk],
                              (cta_m+row<M) && (gk+15<K));
        }
        if (t < 128) {
            unsigned blk = kb >> 5;
            #pragma unroll
            for (int s=0;s<4;s++)
                sAs[t][s] = (cta_m+t<M) ? A_scale[(unsigned long long)(cta_m+t)*nb + blk + s] : 0.f;
        }
        __syncthreads();

        MQF3_COMPUTE_PASS(0);
        __syncthreads();   // 2026-09-25: every warp is done with sA and sAs before the second half overwrites them


        #pragma unroll
        for (int c=0; c<4; c++){
            unsigned lin = c*256 + t;
            unsigned row = lin >> 3;
            unsigned col = (lin & 7) << 4;
            unsigned gk  = kb + 128 + col;
            mqf2_load_pred_16(((signed char*)&sA[row][0]) + col,
                              &A_i8[(unsigned long long)(cta_m+row)*K + gk],
                              (cta_m+row<M) && (gk+15<K));
        }
        if (t < 128) {
            unsigned blk = (kb >> 5) + 4;
            #pragma unroll
            for (int s=0;s<4;s++)
                sAs[t][s] = (cta_m+t<M) ? A_scale[(unsigned long long)(cta_m+t)*nb + blk + s] : 0.f;
        }
        __syncthreads();

        MQF3_COMPUTE_PASS(1);
        __syncthreads();   // 2026-09-25: every warp is done with sW and sA before the next step overwrites them
    }
    #undef MQF3_COMPUTE_PASS
    #undef MQF3_LOAD_B


    #pragma unroll
    for (int n=0;n<2;n++){
        unsigned nrow0 = cta_n + ng*32 + n*16 + lane/4;
        #pragma unroll
        for (int j=0;j<8;j++){
            unsigned mcol = cta_m + mh*64 + j*8 + (lane%4)*2;
            unsigned cN0=nrow0, cN1=nrow0+8;
            if (mcol<M   && cN0<N) C[(unsigned long long)mcol*N + cN0]     = __float2bfloat16(acc[n][j][0]);
            if (mcol+1<M && cN0<N) C[(unsigned long long)(mcol+1)*N + cN0] = __float2bfloat16(acc[n][j][1]);
            if (mcol<M   && cN1<N) C[(unsigned long long)mcol*N + cN1]     = __float2bfloat16(acc[n][j][2]);
            if (mcol+1<M && cN1<N) C[(unsigned long long)(mcol+1)*N + cN1] = __float2bfloat16(acc[n][j][3]);
        }
    }
}
#undef MQF3_MMA_S8
#undef MQF3_WSTRIDE

// 2026-09-25: Requantizers for the int8 GEMMs above: to int8 [rows, K] plus one FP32 scale per 32 values
// [rows, K/32]. A block's scale is max|v| / 127 (1 for an all-zero block), and each value becomes v / scale rounded
// to nearest and clamped to [-127, 127]. One thread per (row, 32-block); K must be a multiple of 32.
// requant_w_nvfp4_int8 first dequantizes the NVFP4 weight (E2M1 code x E4M3 scale x scale2);
// requant_a_bf16_int8 reads BF16 activations.








// 2026-09-25: E4M3 decode: scl_fp8 under __SCALE__ and HIP, the __nv_fp8_e4m3 conversion otherwise.
__device__ __forceinline__ float metrale_e4m3_decode_any(unsigned char b) {
#if defined(__SCALE__) || defined(__HIP_PLATFORM_AMD__)
    return scl_fp8(b);
#else
    __nv_fp8_e4m3 fp8; *(unsigned char*)&fp8 = b; return (float)fp8;
#endif
}

extern "C" __global__
void requant_w_nvfp4_int8(
    const unsigned char* __restrict__ W_packed,
    const unsigned char* __restrict__ W_e4m3,
    const float scale2,
    signed char* __restrict__ W_i8,
    float* __restrict__ W_scale,
    unsigned int N, unsigned int K
) {
    unsigned long long blk = (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;
    unsigned long long nblocks = (unsigned long long)N * (K >> 5);
    if (blk >= nblocks) return;
    unsigned int nb = K >> 5;
    unsigned int n  = (unsigned int)(blk / nb);
    unsigned int kb = (unsigned int)(blk % nb) * 32;


    float vals[32];
    float maxa = 0.f;
    #pragma unroll
    for (int i = 0; i < 32; i++) {
        unsigned int k = kb + i;
        unsigned char pb = W_packed[(unsigned long long)n * (K>>1) + (k>>1)];
        unsigned int nib = (k & 1) ? (pb >> 4) : (pb & 0xF);
        float s16 = metrale_e4m3_decode_any(W_e4m3[(unsigned long long)n * (K>>4) + (k>>4)]) * scale2;
        float v = E2M1_LUT[nib] * s16;
        vals[i] = v;
        float a = fabsf(v);
        if (a > maxa) maxa = a;
    }
    float sc = (maxa > 0.f) ? (maxa / 127.0f) : 1.0f;
    float inv = 1.0f / sc;
    W_scale[blk] = sc;
    #pragma unroll
    for (int i = 0; i < 32; i++) {
        int q = __float2int_rn(vals[i] * inv);
        q = max(-127, min(127, q));
        W_i8[(unsigned long long)n * K + kb + i] = (signed char)q;
    }
}

extern "C" __global__
void requant_a_bf16_int8(
    const __nv_bfloat16* __restrict__ A_bf16,
    signed char* __restrict__ A_i8,
    float* __restrict__ A_scale,
    unsigned int M, unsigned int K
) {
    unsigned long long blk = (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;
    unsigned long long nblocks = (unsigned long long)M * (K >> 5);
    if (blk >= nblocks) return;
    unsigned int nb = K >> 5;
    unsigned int m  = (unsigned int)(blk / nb);
    unsigned int kb = (unsigned int)(blk % nb) * 32;

    float vals[32];
    float maxa = 0.f;
    #pragma unroll
    for (int i = 0; i < 32; i++) {
        float v = __bfloat162float(A_bf16[(unsigned long long)m * K + kb + i]);
        vals[i] = v;
        float a = fabsf(v);
        if (a > maxa) maxa = a;
    }
    float sc = (maxa > 0.f) ? (maxa / 127.0f) : 1.0f;
    float inv = 1.0f / sc;
    A_scale[blk] = sc;
    #pragma unroll
    for (int i = 0; i < 32; i++) {
        int q = __float2int_rn(vals[i] * inv);
        q = max(-127, min(127, q));
        A_i8[(unsigned long long)m * K + kb + i] = (signed char)q;
    }
}

// 2026-09-25: requant_a_bf16_int8 with the eight int32 groups of each 32-value block stored in the order
// [0,4,1,5,2,6,3,7] (group p goes to p < 4 ? 2p : 2(p - 4) + 1), the layout int8_gemm_faith5 reads.




extern "C" __global__
void requant_a_bf16_int8_il(
    const __nv_bfloat16* __restrict__ A_bf16,
    signed char* __restrict__ A_i8,
    float* __restrict__ A_scale,
    unsigned int M, unsigned int K
) {
    unsigned long long blk = (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;
    unsigned long long nblocks = (unsigned long long)M * (K >> 5);
    if (blk >= nblocks) return;
    unsigned int nb = K >> 5;
    unsigned int m  = (unsigned int)(blk / nb);
    unsigned int kb = (unsigned int)(blk % nb) * 32;

    float vals[32];
    float maxa = 0.f;
    #pragma unroll
    for (int i = 0; i < 32; i++) {
        float v = __bfloat162float(A_bf16[(unsigned long long)m * K + kb + i]);
        vals[i] = v;
        float a = fabsf(v);
        if (a > maxa) maxa = a;
    }
    float sc = (maxa > 0.f) ? (maxa / 127.0f) : 1.0f;
    float inv = 1.0f / sc;
    A_scale[blk] = sc;
    #pragma unroll
    for (int i = 0; i < 32; i++) {
        int q = __float2int_rn(vals[i] * inv);
        q = max(-127, min(127, q));
        unsigned p = i >> 2;
        unsigned w = i & 3;
        unsigned np = (p < 4) ? (p << 1) : (((p - 4) << 1) + 1);
        unsigned out_i = (np << 2) + w;
        A_i8[(unsigned long long)m * K + kb + out_i] = (signed char)q;
    }
}

// 2026-09-25: fp8_gemm_t with a per-column output scale: C[m, n] = (A[m, :] . B_fp8[n, :]) x row_scale[n], where
// row_scale [N] FP32 holds one scale per row of B (hence the name), as quantize_bf16_to_fp8 writes it. A [M, K]
// BF16, B_fp8 [N, K] E4M3, C [M, N] BF16. Grid (ceil(N/128), ceil(M/64)), 128 threads.















extern "C" __global__ void fp8_gemm_t_row_scaled(
    const __nv_bfloat16* __restrict__ A,
    const unsigned char* __restrict__ B_fp8,
    const float* __restrict__ row_scale,
    __nv_bfloat16* __restrict__ C,
    unsigned int M, unsigned int N, unsigned int K
) {
    const unsigned int cta_n = blockIdx.x * N_TILE_LG;
    const unsigned int cta_m = blockIdx.y * M_TILE;
    const unsigned int warp_id = threadIdx.x / 32;
    const unsigned int lane_id = threadIdx.x % 32;
    const unsigned int warp_m_offset = warp_id * 16;
    const unsigned int group_id = lane_id >> 2;
    const unsigned int tid = lane_id & 3;

    __shared__ __nv_bfloat16 smem_A[2][M_TILE][K_STEP_T + PAD_T];
    __shared__ unsigned char smem_B[2][N_TILE_LG][K_STEP_T];

    float acc[16][4];
    #pragma unroll
    for (int i = 0; i < 16; i++) {
        acc[i][0] = 0.0f; acc[i][1] = 0.0f;
        acc[i][2] = 0.0f; acc[i][3] = 0.0f;
    }

    const unsigned int a_stride = K_STEP_T + PAD_T;

    #define FP8_LOADS_RS(buf, kb) do { \
        { \
            unsigned int a_row_base = threadIdx.x >> 2; \
            unsigned int a_col = (threadIdx.x & 3) << 3; \
            unsigned int gc = (kb) + a_col; \
            _Pragma("unroll") \
            for (int rnd = 0; rnd < 2; rnd++) { \
                unsigned int row = rnd * 32 + a_row_base; \
                unsigned int gr = cta_m + row; \
                cp_async_pred_16(&smem_A[(buf)][row][a_col], \
                    &A[(unsigned long long)gr * K + gc], \
                    (gr < M) && (gc + 7 < K)); \
            } \
        } \
        { \
            unsigned int my_n = threadIdx.x; \
            unsigned int gn = cta_n + my_n; \
            bool valid = (gn < N) && ((kb) + 31 < K); \
            cp_async_pred_16(&smem_B[(buf)][my_n][0], \
                &B_fp8[(unsigned long long)gn * K + (kb)], valid); \
            cp_async_pred_16(&smem_B[(buf)][my_n][16], \
                &B_fp8[(unsigned long long)gn * K + (kb) + 16], valid); \
        } \
    } while(0)

    #define FP8_COMPUTE_RS(a_buf, b_buf) do { \
        const unsigned short* sA = (const unsigned short*)smem_A[(a_buf)]; \
        unsigned int fr0 = warp_m_offset + group_id, fr1 = fr0 + 8; \
        unsigned int a0 = bf16x4_to_e4m3x4(&sA[fr0 * a_stride + tid * 4]); \
        unsigned int a1 = bf16x4_to_e4m3x4(&sA[fr1 * a_stride + tid * 4]); \
        unsigned int a2 = bf16x4_to_e4m3x4(&sA[fr0 * a_stride + 16 + tid * 4]); \
        unsigned int a3 = bf16x4_to_e4m3x4(&sA[fr1 * a_stride + 16 + tid * 4]); \
        _Pragma("unroll") \
        for (int nt = 0; nt < 16; nt++) { \
            unsigned int nc = nt * 8 + group_id; \
            unsigned int b0 = *(const unsigned int*)&smem_B[(b_buf)][nc][4 * tid]; \
            unsigned int b1 = *(const unsigned int*)&smem_B[(b_buf)][nc][16 + 4 * tid]; \
            /* 2026-09-25: metrale_mma_e4m3 rather than inline PTX: under \
             * __SCALE__ it emulates the E4M3 MMA with two BF16 MMAs, and \
             * on other builds it is the same m16n8k32 instruction. \
             * \
             */ \
            metrale_mma_e4m3(acc[nt], a0, a1, a2, a3, b0, b1); \
        } \
    } while(0)

    FP8_LOADS_RS(0, 0);
    cp_async_commit();
    cp_async_wait_all();
    __syncthreads();

    int cur = 0;
    for (unsigned int k_base = K_STEP_T; k_base < K; k_base += K_STEP_T) {
        int nxt = 1 - cur;
        FP8_LOADS_RS(nxt, k_base);
        cp_async_commit();
        FP8_COMPUTE_RS(cur, cur);
        cp_async_wait_all();
        __syncthreads();
        cur = nxt;
    }
    FP8_COMPUTE_RS(cur, cur);

    #undef FP8_LOADS_RS
    #undef FP8_COMPUTE_RS


    #pragma unroll
    for (int nt = 0; nt < 16; nt++) {
        unsigned int c0 = cta_n + nt*8 + tid*2;
        unsigned int c1 = c0 + 1;
        unsigned int r0 = cta_m + warp_m_offset + group_id;
        unsigned int r1 = r0 + 8;
        float sc0 = (c0 < N) ? row_scale[c0] : 0.0f;
        float sc1 = (c1 < N) ? row_scale[c1] : 0.0f;
        if (r0 < M && c0 < N) C[r0*N+c0] = __float2bfloat16(acc[nt][0] * sc0);
        if (r0 < M && c1 < N) C[r0*N+c1] = __float2bfloat16(acc[nt][1] * sc1);
        if (r1 < M && c0 < N) C[r1*N+c0] = __float2bfloat16(acc[nt][2] * sc0);
        if (r1 < M && c1 < N) C[r1*N+c1] = __float2bfloat16(acc[nt][3] * sc1);
    }
}

// 2026-09-25: fp8_gemm_t_row_scaled with a four-stage cp.async ring: the loads of up to three later K steps stay
// in flight (cp_async_wait_group<2>) instead of a full drain per step. The K steps, MMAs and their order match
// fp8_gemm_t_row_scaled, so the output is identical. Static shared memory: 20,480 + 16,384 bytes.
















#define FP8_RS_STAGES 4
extern "C" __global__ void fp8_gemm_t_row_scaled_p4(
    const __nv_bfloat16* __restrict__ A,
    const unsigned char* __restrict__ B_fp8,
    const float* __restrict__ row_scale,
    __nv_bfloat16* __restrict__ C,
    unsigned int M, unsigned int N, unsigned int K
) {
    const unsigned int cta_n = blockIdx.x * N_TILE_LG;
    const unsigned int cta_m = blockIdx.y * M_TILE;
    const unsigned int warp_id = threadIdx.x / 32;
    const unsigned int lane_id = threadIdx.x % 32;
    const unsigned int warp_m_offset = warp_id * 16;
    const unsigned int group_id = lane_id >> 2;
    const unsigned int tid = lane_id & 3;

    __shared__ __nv_bfloat16 smem_A[FP8_RS_STAGES][M_TILE][K_STEP_T + PAD_T];
    __shared__ unsigned char smem_B[FP8_RS_STAGES][N_TILE_LG][K_STEP_T];

    float acc[16][4];
    #pragma unroll
    for (int i = 0; i < 16; i++) {
        acc[i][0] = 0.0f; acc[i][1] = 0.0f;
        acc[i][2] = 0.0f; acc[i][3] = 0.0f;
    }

    const unsigned int a_stride = K_STEP_T + PAD_T;
    const unsigned int num_k = (K + K_STEP_T - 1) / K_STEP_T;

    #define FP8_LOADS_P4(buf, kb) do { \
        { \
            unsigned int a_row_base = threadIdx.x >> 2; \
            unsigned int a_col = (threadIdx.x & 3) << 3; \
            unsigned int gc = (kb) + a_col; \
            _Pragma("unroll") \
            for (int rnd = 0; rnd < 2; rnd++) { \
                unsigned int row = rnd * 32 + a_row_base; \
                unsigned int gr = cta_m + row; \
                cp_async_pred_16(&smem_A[(buf)][row][a_col], \
                    &A[(unsigned long long)gr * K + gc], \
                    (gr < M) && (gc + 7 < K)); \
            } \
        } \
        { \
            unsigned int my_n = threadIdx.x; \
            unsigned int gn = cta_n + my_n; \
            bool valid = (gn < N) && ((kb) + 31 < K); \
            cp_async_pred_16(&smem_B[(buf)][my_n][0], \
                &B_fp8[(unsigned long long)gn * K + (kb)], valid); \
            cp_async_pred_16(&smem_B[(buf)][my_n][16], \
                &B_fp8[(unsigned long long)gn * K + (kb) + 16], valid); \
        } \
    } while(0)

    #define FP8_COMPUTE_P4(buf) do { \
        const unsigned short* sA = (const unsigned short*)smem_A[(buf)]; \
        unsigned int fr0 = warp_m_offset + group_id, fr1 = fr0 + 8; \
        unsigned int a0 = bf16x4_to_e4m3x4(&sA[fr0 * a_stride + tid * 4]); \
        unsigned int a1 = bf16x4_to_e4m3x4(&sA[fr1 * a_stride + tid * 4]); \
        unsigned int a2 = bf16x4_to_e4m3x4(&sA[fr0 * a_stride + 16 + tid * 4]); \
        unsigned int a3 = bf16x4_to_e4m3x4(&sA[fr1 * a_stride + 16 + tid * 4]); \
        _Pragma("unroll") \
        for (int nt = 0; nt < 16; nt++) { \
            unsigned int nc = nt * 8 + group_id; \
            unsigned int b0 = *(const unsigned int*)&smem_B[(buf)][nc][4 * tid]; \
            unsigned int b1 = *(const unsigned int*)&smem_B[(buf)][nc][16 + 4 * tid]; \
            metrale_mma_e4m3(acc[nt], a0, a1, a2, a3, b0, b1); \
        } \
    } while(0)

    // 2026-09-25: Prologue: stages 0 .. S-2 in flight, one commit group each. A stage past the end commits an
    // empty group, so every iteration sees the same group count and the wait below stays exact.

    #pragma unroll
    for (int s = 0; s < FP8_RS_STAGES - 1; s++) {
        if ((unsigned int)s < num_k) FP8_LOADS_P4(s, (unsigned int)s * K_STEP_T);
        cp_async_commit();
    }

    for (unsigned int kt = 0; kt < num_k; kt++) {
        // 2026-09-25: Stage kt + S - 1 goes into the buffer read at step kt - 1, which the previous iteration's
        // trailing __syncthreads released.
        {
            unsigned int kn = kt + FP8_RS_STAGES - 1;
            if (kn < num_k) FP8_LOADS_P4(kn % FP8_RS_STAGES, kn * K_STEP_T);
            cp_async_commit();
        }
        // 2026-09-25: In flight: the groups of stages kt .. kt + S - 1. Waiting until at most S - 2 remain means
        // stage kt has landed for this thread; the __syncthreads makes it visible to all.
        cp_async_wait_group<FP8_RS_STAGES - 2>();
        __syncthreads();
        FP8_COMPUTE_P4(kt % FP8_RS_STAGES);
        __syncthreads();
    }

    #undef FP8_LOADS_P4
    #undef FP8_COMPUTE_P4

    #pragma unroll
    for (int nt = 0; nt < 16; nt++) {
        unsigned int c0 = cta_n + nt*8 + tid*2;
        unsigned int c1 = c0 + 1;
        unsigned int r0 = cta_m + warp_m_offset + group_id;
        unsigned int r1 = r0 + 8;
        float sc0 = (c0 < N) ? row_scale[c0] : 0.0f;
        float sc1 = (c1 < N) ? row_scale[c1] : 0.0f;
        if (r0 < M && c0 < N) C[r0*N+c0] = __float2bfloat16(acc[nt][0] * sc0);
        if (r0 < M && c1 < N) C[r0*N+c1] = __float2bfloat16(acc[nt][1] * sc1);
        if (r1 < M && c0 < N) C[r1*N+c0] = __float2bfloat16(acc[nt][2] * sc0);
        if (r1 < M && c1 < N) C[r1*N+c1] = __float2bfloat16(acc[nt][3] * sc1);
    }
}
#undef FP8_RS_STAGES

// 2026-09-25: fp8_gemm_t_row_scaled with a K step of 64: two m16n8k32 MMAs per step, at K offsets 0 and 32, so each
// output element is accumulated in the same K order as in fp8_gemm_t_row_scaled. B is loaded in 16-byte chunks,
// each predicated on its own end.










#define FP8_K64 64
#define FP8_K64_PAD 8
extern "C" __global__ void fp8_gemm_t_row_scaled_k64(
    const __nv_bfloat16* __restrict__ A,
    const unsigned char* __restrict__ B_fp8,
    const float* __restrict__ row_scale,
    __nv_bfloat16* __restrict__ C,
    unsigned int M, unsigned int N, unsigned int K
) {
    const unsigned int cta_n = blockIdx.x * N_TILE_LG;
    const unsigned int cta_m = blockIdx.y * M_TILE;
    const unsigned int warp_id = threadIdx.x / 32;
    const unsigned int lane_id = threadIdx.x % 32;
    const unsigned int warp_m_offset = warp_id * 16;
    const unsigned int group_id = lane_id >> 2;
    const unsigned int tid = lane_id & 3;

    __shared__ __align__(16) __nv_bfloat16 smem_A[2][M_TILE][FP8_K64 + FP8_K64_PAD];
    __shared__ __align__(16) unsigned char smem_B[2][N_TILE_LG][FP8_K64];

    float acc[16][4];
    #pragma unroll
    for (int i = 0; i < 16; i++) {
        acc[i][0] = 0.0f; acc[i][1] = 0.0f;
        acc[i][2] = 0.0f; acc[i][3] = 0.0f;
    }

    const unsigned int a_stride = FP8_K64 + FP8_K64_PAD;
    const unsigned int num_k = (K + FP8_K64 - 1) / FP8_K64;



    #define FP8_LOADS_K64(buf, kb) do { \
        _Pragma("unroll") \
        for (int c = 0; c < 4; c++) { \
            unsigned int chunk = c * 128 + threadIdx.x; \
            unsigned int row = chunk >> 3; \
            unsigned int a_col = (chunk & 7) << 3; \
            unsigned int gr = cta_m + row; \
            unsigned int gc = (kb) + a_col; \
            cp_async_pred_16(&smem_A[(buf)][row][a_col], \
                &A[(unsigned long long)gr * K + gc], \
                (gr < M) && (gc + 7 < K)); \
        } \
        { \
            unsigned int my_n = threadIdx.x; \
            unsigned int gn = cta_n + my_n; \
            const unsigned char* brow = &B_fp8[(unsigned long long)gn * K + (kb)]; \
            _Pragma("unroll") \
            for (int c = 0; c < 4; c++) { \
                bool valid = (gn < N) && ((kb) + c * 16 + 15 < K); \
                cp_async_pred_16(&smem_B[(buf)][my_n][c * 16], brow + c * 16, valid); \
            } \
        } \
    } while(0)

    #define FP8_COMPUTE_K64(buf) do { \
        const unsigned short* sA = (const unsigned short*)smem_A[(buf)]; \
        unsigned int fr0 = warp_m_offset + group_id, fr1 = fr0 + 8; \
        _Pragma("unroll") \
        for (int sub = 0; sub < 2; sub++) { \
            unsigned int ko = sub * 32; \
            unsigned int a0 = bf16x4_to_e4m3x4(&sA[fr0 * a_stride + ko + tid * 4]); \
            unsigned int a1 = bf16x4_to_e4m3x4(&sA[fr1 * a_stride + ko + tid * 4]); \
            unsigned int a2 = bf16x4_to_e4m3x4(&sA[fr0 * a_stride + ko + 16 + tid * 4]); \
            unsigned int a3 = bf16x4_to_e4m3x4(&sA[fr1 * a_stride + ko + 16 + tid * 4]); \
            _Pragma("unroll") \
            for (int nt = 0; nt < 16; nt++) { \
                unsigned int nc = nt * 8 + group_id; \
                unsigned int b0 = *(const unsigned int*)&smem_B[(buf)][nc][ko + 4 * tid]; \
                unsigned int b1 = *(const unsigned int*)&smem_B[(buf)][nc][ko + 16 + 4 * tid]; \
                metrale_mma_e4m3(acc[nt], a0, a1, a2, a3, b0, b1); \
            } \
        } \
    } while(0)

    FP8_LOADS_K64(0, 0);
    cp_async_commit();
    cp_async_wait_all();
    __syncthreads();

    int cur = 0;
    for (unsigned int kt = 1; kt < num_k; kt++) {
        int nxt = 1 - cur;
        FP8_LOADS_K64(nxt, kt * FP8_K64);
        cp_async_commit();
        FP8_COMPUTE_K64(cur);
        cp_async_wait_all();
        __syncthreads();
        cur = nxt;
    }
    FP8_COMPUTE_K64(cur);

    #undef FP8_LOADS_K64
    #undef FP8_COMPUTE_K64

    #pragma unroll
    for (int nt = 0; nt < 16; nt++) {
        unsigned int c0 = cta_n + nt*8 + tid*2;
        unsigned int c1 = c0 + 1;
        unsigned int r0 = cta_m + warp_m_offset + group_id;
        unsigned int r1 = r0 + 8;
        float sc0 = (c0 < N) ? row_scale[c0] : 0.0f;
        float sc1 = (c1 < N) ? row_scale[c1] : 0.0f;
        if (r0 < M && c0 < N) C[r0*N+c0] = __float2bfloat16(acc[nt][0] * sc0);
        if (r0 < M && c1 < N) C[r0*N+c1] = __float2bfloat16(acc[nt][1] * sc1);
        if (r1 < M && c0 < N) C[r1*N+c0] = __float2bfloat16(acc[nt][2] * sc0);
        if (r1 < M && c1 < N) C[r1*N+c1] = __float2bfloat16(acc[nt][3] * sc1);
    }
}
#undef FP8_K64
#undef FP8_K64_PAD

// 2026-09-25: fp8_gemm_t_row_scaled with a 16-row M tile on one warp (32 threads): M must be at most 16, and the grid
// is (ceil(N/128), 1). Each step the warp issues 16 m16n8k32 MMAs across the 128-column N tile.




















extern "C" __global__ void fp8_gemm_t_row_scaled_m16(
    const __nv_bfloat16* __restrict__ A,
    const unsigned char* __restrict__ B_fp8,
    const float* __restrict__ row_scale,
    __nv_bfloat16* __restrict__ C,
    unsigned int M, unsigned int N, unsigned int K
) {
    const unsigned int cta_n = blockIdx.x * N_TILE_LG;
    const unsigned int lane_id = threadIdx.x % 32;
    const unsigned int group_id = lane_id >> 2;
    const unsigned int tid = lane_id & 3;


    __shared__ __nv_bfloat16 smem_A[2][16][K_STEP_T + PAD_T];
    __shared__ unsigned char smem_B[2][N_TILE_LG][K_STEP_T];

    float acc[16][4];
    #pragma unroll
    for (int i = 0; i < 16; i++) {
        acc[i][0] = 0.0f; acc[i][1] = 0.0f;
        acc[i][2] = 0.0f; acc[i][3] = 0.0f;
    }

    const unsigned int a_stride = K_STEP_T + PAD_T;

    // 2026-09-25: Two rounds of 16-byte copies fill the 16 x 32 BF16 A tile, since one round (32 threads x 8
    // values) covers half of it. Round r: thread t loads row t / 2, columns ((t & 1) << 4) + 8r.





    #define FP8_LOADS_M16(buf, kb) do { \
        _Pragma("unroll") \
        for (int ar = 0; ar < 2; ar++) { \
            unsigned int row = threadIdx.x >> 1; \
            unsigned int a_col = ((threadIdx.x & 1) << 4) + ar * 8; \
            unsigned int gc = (kb) + a_col; \
            unsigned int gr = row; \
            cp_async_pred_16(&smem_A[(buf)][row][a_col], \
                &A[(unsigned long long)gr * K + gc], \
                (gr < M) && (gc + 7 < K)); \
        } \
        { \
            \
            \
            _Pragma("unroll") \
            for (int rnd = 0; rnd < 4; rnd++) { \
                unsigned int my_n = rnd * 32 + threadIdx.x; \
                unsigned int gn = cta_n + my_n; \
                bool valid = (gn < N) && ((kb) + 31 < K); \
                cp_async_pred_16(&smem_B[(buf)][my_n][0], \
                    &B_fp8[(unsigned long long)gn * K + (kb)], valid); \
                cp_async_pred_16(&smem_B[(buf)][my_n][16], \
                    &B_fp8[(unsigned long long)gn * K + (kb) + 16], valid); \
            } \
        } \
    } while(0)


    #define FP8_COMPUTE_M16(a_buf, b_buf) do { \
        const unsigned short* sA = (const unsigned short*)smem_A[(a_buf)]; \
        unsigned int fr0 = group_id, fr1 = fr0 + 8; \
        unsigned int a0 = bf16x4_to_e4m3x4(&sA[fr0 * a_stride + tid * 4]); \
        unsigned int a1 = bf16x4_to_e4m3x4(&sA[fr1 * a_stride + tid * 4]); \
        unsigned int a2 = bf16x4_to_e4m3x4(&sA[fr0 * a_stride + 16 + tid * 4]); \
        unsigned int a3 = bf16x4_to_e4m3x4(&sA[fr1 * a_stride + 16 + tid * 4]); \
        _Pragma("unroll") \
        for (int nt = 0; nt < 16; nt++) { \
            unsigned int nc = nt * 8 + group_id; \
            unsigned int b0 = *(const unsigned int*)&smem_B[(b_buf)][nc][4 * tid]; \
            unsigned int b1 = *(const unsigned int*)&smem_B[(b_buf)][nc][16 + 4 * tid]; \
            /* 2026-09-25: metrale_mma_e4m3 rather than inline PTX: under \
             * __SCALE__ it emulates the E4M3 MMA with two BF16 MMAs, and \
             * on other builds it is the same m16n8k32 instruction. \
             * \
             */ \
            metrale_mma_e4m3(acc[nt], a0, a1, a2, a3, b0, b1); \
        } \
    } while(0)

    FP8_LOADS_M16(0, 0);
    cp_async_commit();
    cp_async_wait_all();
    __syncwarp();

    int cur = 0;
    for (unsigned int k_base = K_STEP_T; k_base < K; k_base += K_STEP_T) {
        int nxt = 1 - cur;
        FP8_LOADS_M16(nxt, k_base);
        cp_async_commit();
        FP8_COMPUTE_M16(cur, cur);
        cp_async_wait_all();
        __syncwarp();
        cur = nxt;
    }
    FP8_COMPUTE_M16(cur, cur);

    #undef FP8_LOADS_M16
    #undef FP8_COMPUTE_M16



    #pragma unroll
    for (int nt = 0; nt < 16; nt++) {
        unsigned int c0 = cta_n + nt*8 + tid*2;
        unsigned int c1 = c0 + 1;
        unsigned int r0 = group_id;
        unsigned int r1 = r0 + 8;
        float sc0 = (c0 < N) ? row_scale[c0] : 0.0f;
        float sc1 = (c1 < N) ? row_scale[c1] : 0.0f;
        if (r0 < M && c0 < N) C[r0*N+c0] = __float2bfloat16(acc[nt][0] * sc0);
        if (r0 < M && c1 < N) C[r0*N+c1] = __float2bfloat16(acc[nt][1] * sc1);
        if (r1 < M && c0 < N) C[r1*N+c0] = __float2bfloat16(acc[nt][2] * sc0);
        if (r1 < M && c1 < N) C[r1*N+c1] = __float2bfloat16(acc[nt][3] * sc1);
    }
}
