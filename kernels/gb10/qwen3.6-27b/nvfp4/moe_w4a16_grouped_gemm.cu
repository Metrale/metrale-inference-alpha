// SPDX-License-Identifier: MIT OR Apache-2.0

// 2026-09-25: Grouped MoE GEMMs over NVFP4 weights, all experts in one launch: E2M1 values, one E4M3 scale per 16
// values of K and a per-expert scale2. Expert e owns rows expert_offsets[e] .. expert_offsets[e + 1]; A rows are
// gathered through sorted_token_ids when it is non-null. Grid (N tiles, M tiles of 64, num_experts), Block (128).
// `_ptrtable` dequantises B to BF16 for m16n8k16 BF16 MMAs; the `_t` kernels (B [K/2, N]) dequantise B to E4M3
// and run m16n8k32 E4M3 MMAs, with A rounded to E4M3 in registers (or given as E4M3 to the fp8 kernel).
// forked-from: kernels/gb10/common/moe_w4a16_grouped_gemm.cu (2026-09-24; 1680 of 1412 lines differ, see kernels/FORKS.md)
//
// Owner: gb10 kernels (qwen3.6-27b, and the targets that list this file in `[sources] use`).
// Invariants: a CTA of expert e writes only C rows below expert_offsets[e + 1] and columns below N; an expert
// whose weight pointer is 0 writes nothing.
#include <cuda_bf16.h>
#include <cuda_fp8.h>

// 2026-09-25: Under __HIP_PLATFORM_AMD__, E4M3 encode in plain bit arithmetic: bias 7, NaN to 0x7F, mantissa
// rounded half up, |v| >= 496 to +-448 (exponent 15, mantissa 6); |v| in [464, 496) gets mantissa 7 at exponent
// 15, the NaN code, so it does not saturate there.

#if defined(__HIP_PLATFORM_AMD__)
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

// 2026-09-25: Two floats to two E4M3 bytes: e4m3(a_hi) in the high byte and e4m3(b_lo) in the low
// byte, the operand order of PTX `cvt.rn.satfinite.e4m3x2.f32`. Under __SCALE__ it calls __nv_cvt_float_to_fp8
// with __NV_SATFINITE and __NV_E4M3 per value, and under __HIP_PLATFORM_AMD__ scl_enc_fp8; other builds use the
// PTX instruction. The __SCALE__ and PTX paths saturate at +-448; scl_enc_fp8 does not on [464, 496).




__device__ __forceinline__ unsigned short metrale_cvt_e4m3x2_f32(float a_hi, float b_lo) {
#if defined(__SCALE__)
    unsigned a8 = (unsigned)__nv_cvt_float_to_fp8(a_hi, __NV_SATFINITE, __NV_E4M3);
    unsigned b8 = (unsigned)__nv_cvt_float_to_fp8(b_lo, __NV_SATFINITE, __NV_E4M3);
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

// 2026-09-25: metrale_mma_e4m3: one m16n8k32 E4M3 MMA with FP32 accumulation. Under __SCALE__ (the strix build
// compiles this file through [sources] use) the E4M3 fragments are gathered within each 4-lane group with
// __shfl_sync, widened to BF16 and multiplied as two m16n8k16 BF16 MMAs, over K 0..15 and 16..31; other builds
// issue the E4M3 MMA instruction.












#if defined(__SCALE__)
__device__ __forceinline__ float metrale_e4m3_to_f32(unsigned char b) {
    return __half2float(__nv_cvt_fp8_to_halfraw((__nv_fp8_storage_t)b, __NV_E4M3));
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
#define PAD_T 8        // 2026-09-25: rows of (32 + 8) * 2 = 80 bytes keep cp.async's 16-byte alignment
#define BP_PAD 16
#define GROUP_SIZE 16

__device__ __constant__ float E2M1_LUT_MOE[16] = {
    0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f,
    -0.0f, -0.5f, -1.0f, -1.5f, -2.0f, -3.0f, -4.0f, -6.0f
};

// 2026-09-25: `moe_w4a16_grouped_gemm_ptrtable`: 64-column tiles, K step 16. Per expert, B is [N, K/2] packed bytes
// (low nibble first) and the scales [N, K/16].

extern "C" __global__ void moe_w4a16_grouped_gemm_ptrtable(
    const __nv_bfloat16* __restrict__ A,
    const unsigned long long* __restrict__ B_packed_ptrs,
    const unsigned long long* __restrict__ B_scale_ptrs,
    const float* __restrict__ scale2_vals,
    __nv_bfloat16* __restrict__ C,
    const int* __restrict__ expert_offsets,
    const int* __restrict__ sorted_token_ids,
    unsigned int num_experts,
    unsigned int N,
    unsigned int K
) {
    const unsigned int expert_id = blockIdx.z;
    if (expert_id >= num_experts) return;

    const int m_start = expert_offsets[expert_id];
    const int m_end = expert_offsets[expert_id + 1];
    const int M_expert = m_end - m_start;
    if (M_expert <= 0) return;

    const int cta_m_local = blockIdx.y * M_TILE;
    if (cta_m_local >= M_expert) return;

    const unsigned int cta_m = m_start + cta_m_local;
    const unsigned int cta_n = blockIdx.x * N_TILE_SM;

    const unsigned char* B_expert = (const unsigned char*)B_packed_ptrs[expert_id];
    const unsigned char* S_expert = (const unsigned char*)B_scale_ptrs[expert_id];
    const float scale2 = scale2_vals[expert_id];

    if (B_expert == 0) return;

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

    const unsigned int a_stride = K_STEP + PAD;
    const unsigned int b_stride = N_TILE_SM + PAD;
    const unsigned int M_eff = (unsigned int)M_expert;
    const unsigned int half_K = K / 2;
    const unsigned int num_groups = K / GROUP_SIZE;

    for (unsigned int k_base = 0; k_base < K; k_base += K_STEP) {
        {
            const unsigned int ept = (M_TILE * K_STEP) / 128;
            #pragma unroll
            for (unsigned int i = 0; i < ept; i++) {
                unsigned int idx = threadIdx.x * ept + i;
                unsigned int row = idx / K_STEP;
                unsigned int col = idx % K_STEP;
                unsigned int gc = k_base + col;
                bool valid = (cta_m_local + row) < M_eff && gc < K;
                if (valid) {
                    unsigned int a_row = sorted_token_ids
                        ? (unsigned int)sorted_token_ids[cta_m + row]
                        : (cta_m + row);
                    smem_A[row][col] = A[a_row * K + gc];
                } else {
                    smem_A[row][col] = __float2bfloat16(0.0f);
                }
            }
        }

        {
            const unsigned int ept = (K_STEP * N_TILE_SM) / 128;
            unsigned int scale_group = k_base / GROUP_SIZE;
            #pragma unroll
            for (unsigned int i = 0; i < ept; i++) {
                unsigned int idx = threadIdx.x * ept + i;
                unsigned int k = idx / N_TILE_SM;
                unsigned int n = idx % N_TILE_SM;
                unsigned int gk = k_base + k;
                unsigned int gn = cta_n + n;
                if (gk < K && gn < N) {
                    unsigned int k_pair = gk / 2;
                    unsigned char packed_byte = B_expert[(unsigned long long)gn * half_K + k_pair];
                    unsigned int nibble = (gk & 1) ? (packed_byte >> 4) : (packed_byte & 0xF);
                    unsigned char sb = S_expert[(unsigned long long)gn * num_groups + scale_group];
                    __nv_fp8_e4m3 fp8; *(unsigned char*)&fp8 = sb;
                    smem_B[k][n] = __float2bfloat16(E2M1_LUT_MOE[nibble] * (float)fp8 * scale2);
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
        unsigned int a0 = ((unsigned int)sA[fr0*a_stride+fc0+1]<<16) | (unsigned int)sA[fr0*a_stride+fc0];
        unsigned int a1 = ((unsigned int)sA[fr1*a_stride+fc0+1]<<16) | (unsigned int)sA[fr1*a_stride+fc0];
        unsigned int a2 = ((unsigned int)sA[fr0*a_stride+fc1+1]<<16) | (unsigned int)sA[fr0*a_stride+fc1];
        unsigned int a3 = ((unsigned int)sA[fr1*a_stride+fc1+1]<<16) | (unsigned int)sA[fr1*a_stride+fc1];

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
        bool r0v = (int)(warp_m_offset + group_id + cta_m_local) < M_expert;
        bool r1v = (int)(warp_m_offset + group_id + 8 + cta_m_local) < M_expert;
        if (r0v && c0 < N) C[r0*N+c0] = __float2bfloat16(acc[nt][0]);
        if (r0v && c1 < N) C[r0*N+c1] = __float2bfloat16(acc[nt][1]);
        if (r1v && c0 < N) C[r1*N+c0] = __float2bfloat16(acc[nt][2]);
        if (r1v && c1 < N) C[r1*N+c1] = __float2bfloat16(acc[nt][3]);
    }
}

// 2026-09-25: `moe_w4a16_grouped_gemm_ptrtable_t`: 128-column tiles, K step 32, cp.async double buffering of A,
// packed B and scales. Per expert, B is [K/2, N] and the scales [K/16, N]. Each step dequantises B to E4M3 in
// shared memory and runs one m16n8k32 E4M3 MMA per 8-column slice, with A converted from BF16 in registers.
//
// Static shared memory: A 2 x 64 x 40 BF16 (10240 B), packed B 2 x 16 x 144 B (4608 B), scales 2 x 2 x 144 B
// (576 B), dequantised B 128 x 36 B (4608 B), LUT 64 B, token ids 256 B: 20352 B.










__device__ __forceinline__ void moe_cp_async_pred_16(void* dst_smem, const void* src_gmem, bool pred) {
    unsigned int dst = __cvta_generic_to_shared(dst_smem);
    unsigned int src_bytes = pred ? 16 : 0;
    asm volatile("cp.async.ca.shared.global [%0], [%1], 16, %2;"
                 :: "r"(dst), "l"(src_gmem), "r"(src_bytes));
}

__device__ __forceinline__ void moe_cp_async_commit() {
    asm volatile("cp.async.commit_group;");
}

__device__ __forceinline__ void moe_cp_async_wait_all() {
    asm volatile("cp.async.wait_group 0;");
}

// 2026-09-25: Four BF16 values from shared memory to four E4M3 bytes; byte i holds src[i].
__device__ __forceinline__ unsigned int moe_bf16x4_to_e4m3x4(const unsigned short* src) {
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

extern "C" __global__ void moe_w4a16_grouped_gemm_ptrtable_t(
    const __nv_bfloat16* __restrict__ A,
    const unsigned long long* __restrict__ B_packed_ptrs,
    const unsigned long long* __restrict__ B_scale_ptrs,
    const float* __restrict__ scale2_vals,
    __nv_bfloat16* __restrict__ C,
    const int* __restrict__ expert_offsets,
    const int* __restrict__ sorted_token_ids,
    unsigned int num_experts,
    unsigned int N,
    unsigned int K
) {
    const unsigned int expert_id = blockIdx.z;
    if (expert_id >= num_experts) return;

    const int m_start = expert_offsets[expert_id];
    const int m_end = expert_offsets[expert_id + 1];
    const int M_expert = m_end - m_start;
    if (M_expert <= 0) return;

    const int cta_m_local = blockIdx.y * M_TILE;
    if (cta_m_local >= M_expert) return;

    const unsigned int cta_m = m_start + cta_m_local;
    const unsigned int cta_n = blockIdx.x * N_TILE_LG;

    const unsigned char* B_expert = (const unsigned char*)B_packed_ptrs[expert_id];
    const unsigned char* S_expert = (const unsigned char*)B_scale_ptrs[expert_id];
    const float scale2 = scale2_vals[expert_id];

    if (B_expert == 0) return;

    const unsigned int warp_id = threadIdx.x / 32;
    const unsigned int lane_id = threadIdx.x % 32;
    const unsigned int warp_m_offset = warp_id * 16;
    const unsigned int group_id = lane_id >> 2;
    const unsigned int tid = lane_id & 3;

    __shared__ __nv_bfloat16 smem_A[2][M_TILE][K_STEP_T + PAD_T];
    __shared__ unsigned char smem_Bp[2][K_STEP_T / 2][N_TILE_LG + BP_PAD];
    __shared__ unsigned char smem_Bs[2][K_STEP_T / GROUP_SIZE][N_TILE_LG + BP_PAD];
    // 2026-09-25: +4 pad: 36-byte rows (9 words, coprime with 32 banks). The dequant writes (t * 9 + kp / 2) % 32
    // are distinct across a warp; the MMA reads (nc * 9 + tid) % 32 have 3 conflicting pairs, against 16 unpadded.

    __shared__ unsigned char smem_B_fp8[N_TILE_LG][K_STEP_T + 4];
    __shared__ float smem_LUT[16];
    __shared__ int smem_tok[M_TILE];

    if (threadIdx.x < 16) {
        smem_LUT[threadIdx.x] = E2M1_LUT_MOE[threadIdx.x];
    }
    if (threadIdx.x < M_TILE) {
        int local_row = threadIdx.x;
        if (sorted_token_ids && (cta_m_local + local_row) < (unsigned int)M_expert) {
            smem_tok[local_row] = sorted_token_ids[cta_m + local_row];
        } else {
            smem_tok[local_row] = (int)(cta_m + local_row);
        }
    }
    __syncthreads();

    float acc[16][4];
    #pragma unroll
    for (int i = 0; i < 16; i++) {
        acc[i][0] = 0.0f; acc[i][1] = 0.0f;
        acc[i][2] = 0.0f; acc[i][3] = 0.0f;
    }

    const unsigned int a_stride = K_STEP_T + PAD_T;
    const unsigned int M_eff = (unsigned int)M_expert;

    #define MOE_ISSUE_LOADS(buf, kb) do { \
        { \
            unsigned int a_row_base = threadIdx.x >> 2; \
            unsigned int a_col = (threadIdx.x & 3) << 3; \
            unsigned int gc = (kb) + a_col; \
            _Pragma("unroll") \
            for (int rnd = 0; rnd < 2; rnd++) { \
                unsigned int row = rnd * 32 + a_row_base; \
                bool valid = (cta_m_local + row) < M_eff && (gc + 7 < K); \
                unsigned int a_row = (unsigned int)smem_tok[row]; \
                moe_cp_async_pred_16(&smem_A[(buf)][row][a_col], \
                    &A[(unsigned long long)a_row * K + gc], valid); \
            } \
        } \
        { \
            unsigned int kp = threadIdx.x >> 3; \
            unsigned int ns = (threadIdx.x & 7) << 4; \
            unsigned int gke = (kb) + (kp << 1); \
            unsigned int gns = cta_n + ns; \
            moe_cp_async_pred_16(&smem_Bp[(buf)][kp][ns], \
                &B_expert[(unsigned long long)(gke >> 1) * N + gns], \
                (gke + 1 <= K) && (gns + 15 < N)); \
            if (kp < K_STEP_T / GROUP_SIZE) { \
                unsigned int sg = (kb) / GROUP_SIZE + kp; \
                moe_cp_async_pred_16(&smem_Bs[(buf)][kp][ns], \
                    &S_expert[(unsigned long long)sg * N + gns], \
                    (gns + 15 < N)); \
            } \
        } \
    } while(0)

    // 2026-09-25: Dequantise this thread's column of B to E4M3: E2M1 value * (group scale * scale2).
    #define MOE_DEQUANT(buf) do { \
        unsigned int my_n = threadIdx.x; \
        unsigned char sb0 = smem_Bs[(buf)][0][my_n]; \
        unsigned char sb1 = smem_Bs[(buf)][1][my_n]; \
        __nv_fp8_e4m3 f0, f1; \
        *(unsigned char*)&f0 = sb0; \
        *(unsigned char*)&f1 = sb1; \
        float sv0 = (float)f0 * scale2; \
        float sv1 = (float)f1 * scale2; \
        _Pragma("unroll") \
        for (int kp = 0; kp < 8; kp++) { \
            unsigned char packed = smem_Bp[(buf)][kp][my_n]; \
            float lo = smem_LUT[packed & 0xF] * sv0; \
            float hi = smem_LUT[packed >> 4] * sv0; \
            unsigned short fp8_pair; \
            fp8_pair = metrale_cvt_e4m3x2_f32(hi, lo); \
            *(unsigned short*)&smem_B_fp8[my_n][kp * 2] = fp8_pair; \
        } \
        _Pragma("unroll") \
        for (int kp = 8; kp < 16; kp++) { \
            unsigned char packed = smem_Bp[(buf)][kp][my_n]; \
            float lo = smem_LUT[packed & 0xF] * sv1; \
            float hi = smem_LUT[packed >> 4] * sv1; \
            unsigned short fp8_pair; \
            fp8_pair = metrale_cvt_e4m3x2_f32(hi, lo); \
            *(unsigned short*)&smem_B_fp8[my_n][kp * 2] = fp8_pair; \
        } \
    } while(0)

    // 2026-09-25: Convert A from BF16 to E4M3 in registers; one m16n8k32 MMA per 8-column slice.
    #define MOE_COMPUTE_MMA(a_buf) do { \
        const unsigned short* sA = (const unsigned short*)smem_A[(a_buf)]; \
        unsigned int fr0 = warp_m_offset + group_id, fr1 = fr0 + 8; \
        unsigned int a0 = moe_bf16x4_to_e4m3x4(&sA[fr0 * a_stride + tid * 4]); \
        unsigned int a1 = moe_bf16x4_to_e4m3x4(&sA[fr1 * a_stride + tid * 4]); \
        unsigned int a2 = moe_bf16x4_to_e4m3x4(&sA[fr0 * a_stride + 16 + tid * 4]); \
        unsigned int a3 = moe_bf16x4_to_e4m3x4(&sA[fr1 * a_stride + 16 + tid * 4]); \
        _Pragma("unroll") \
        for (int nt = 0; nt < 16; nt++) { \
            unsigned int nc = nt * 8 + group_id; \
            unsigned int b0 = *(const unsigned int*)&smem_B_fp8[nc][4 * tid]; \
            unsigned int b1 = *(const unsigned int*)&smem_B_fp8[nc][16 + 4 * tid]; \
            metrale_mma_e4m3(acc[nt], a0,a1,a2,a3, b0, b1); \
        } \
    } while(0)

    MOE_ISSUE_LOADS(0, 0);
    moe_cp_async_commit();
    moe_cp_async_wait_all();
    __syncthreads();
    MOE_DEQUANT(0);
    __syncthreads();

    int cur = 0;
    for (unsigned int k_base = K_STEP_T; k_base < K; k_base += K_STEP_T) {
        int nxt = 1 - cur;

        MOE_ISSUE_LOADS(nxt, k_base);
        moe_cp_async_commit();

        MOE_COMPUTE_MMA(cur);

        moe_cp_async_wait_all();
        __syncthreads();

        MOE_DEQUANT(nxt);
        __syncthreads();

        cur = nxt;
    }

    MOE_COMPUTE_MMA(cur);

    #undef MOE_ISSUE_LOADS
    #undef MOE_DEQUANT
    #undef MOE_COMPUTE_MMA

    #pragma unroll
    for (int nt = 0; nt < 16; nt++) {
        unsigned int c0 = cta_n + nt*8 + tid*2;
        unsigned int c1 = c0 + 1;
        unsigned int r0 = cta_m + warp_m_offset + group_id;
        unsigned int r1 = r0 + 8;
        bool r0v = (int)(warp_m_offset + group_id + cta_m_local) < M_expert;
        bool r1v = (int)(warp_m_offset + group_id + 8 + cta_m_local) < M_expert;
        if (r0v && c0 < N) C[r0*N+c0] = __float2bfloat16(acc[nt][0]);
        if (r0v && c1 < N) C[r0*N+c1] = __float2bfloat16(acc[nt][1]);
        if (r1v && c0 < N) C[r1*N+c0] = __float2bfloat16(acc[nt][2]);
        if (r1v && c1 < N) C[r1*N+c1] = __float2bfloat16(acc[nt][3]);
    }
}

// 2026-09-25: `moe_w4a16_grouped_gemm_ptrtable_t_k64`: `moe_w4a16_grouped_gemm_ptrtable_t` with a K step of 64 and
// four scale groups per step.
//
// Static shared memory: A 2 x 64 x 72 BF16 (18432 B), packed B 2 x 32 x 144 B (9216 B), scales 2 x 4 x 144 B
// (1152 B), dequantised B 128 x 80 B (10240 B), LUT 64 B, token ids 256 B: 39360 B.














#define K_STEP_T64 64
#define PAD_T64 8  // 2026-09-25: rows of (64 + 8) * 2 = 144 bytes keep cp.async's 16-byte alignment

extern "C" __global__ void moe_w4a16_grouped_gemm_ptrtable_t_k64(
    const __nv_bfloat16* __restrict__ A,
    const unsigned long long* __restrict__ B_packed_ptrs,
    const unsigned long long* __restrict__ B_scale_ptrs,
    const float* __restrict__ scale2_vals,
    __nv_bfloat16* __restrict__ C,
    const int* __restrict__ expert_offsets,
    const int* __restrict__ sorted_token_ids,
    unsigned int num_experts,
    unsigned int N,
    unsigned int K
) {
    const unsigned int expert_id = blockIdx.z;
    if (expert_id >= num_experts) return;

    const int m_start = expert_offsets[expert_id];
    const int m_end = expert_offsets[expert_id + 1];
    const int M_expert = m_end - m_start;
    if (M_expert <= 0) return;

    const int cta_m_local = blockIdx.y * M_TILE;
    if (cta_m_local >= M_expert) return;

    const unsigned int cta_m = m_start + cta_m_local;
    const unsigned int cta_n = blockIdx.x * N_TILE_LG;

    const unsigned char* B_expert = (const unsigned char*)B_packed_ptrs[expert_id];
    const unsigned char* S_expert = (const unsigned char*)B_scale_ptrs[expert_id];
    const float scale2 = scale2_vals[expert_id];

    if (B_expert == 0) return;

    const unsigned int warp_id = threadIdx.x / 32;
    const unsigned int lane_id = threadIdx.x % 32;
    const unsigned int warp_m_offset = warp_id * 16;
    const unsigned int group_id = lane_id >> 2;
    const unsigned int tid = lane_id & 3;

    // 2026-09-25: Dequantised-B rows of 80 bytes (20 words): the MMA reads (nc * 20 + tid) % 32 of a warp hit
    // distinct banks. Unpadded 64-byte rows would give up to 4-way conflicts.

    __shared__ __nv_bfloat16 smem_A_k64[2][M_TILE][K_STEP_T64 + PAD_T64];
    __shared__ unsigned char smem_Bp_k64[2][K_STEP_T64 / 2][N_TILE_LG + BP_PAD];
    __shared__ unsigned char smem_Bs_k64[2][K_STEP_T64 / GROUP_SIZE][N_TILE_LG + BP_PAD];
    __shared__ unsigned char smem_B_fp8_k64[N_TILE_LG][K_STEP_T64 + 16];
    __shared__ float smem_LUT_k64[16];
    __shared__ int smem_tok_k64[M_TILE];

    if (threadIdx.x < 16) smem_LUT_k64[threadIdx.x] = E2M1_LUT_MOE[threadIdx.x];
    if (threadIdx.x < M_TILE) {
        int local_row = threadIdx.x;
        if (sorted_token_ids && (cta_m_local + local_row) < (unsigned int)M_expert)
            smem_tok_k64[local_row] = sorted_token_ids[cta_m + local_row];
        else
            smem_tok_k64[local_row] = (int)(cta_m + local_row);
    }
    __syncthreads();

    float acc[16][4];
    #pragma unroll
    for (int i = 0; i < 16; i++) {
        acc[i][0] = 0.0f; acc[i][1] = 0.0f;
        acc[i][2] = 0.0f; acc[i][3] = 0.0f;
    }

    const unsigned int ast64 = K_STEP_T64 + PAD_T64;
    const unsigned int M_eff = (unsigned int)M_expert;

    // 2026-09-25: Per stage: A 4 rounds x 128 threads x 16 B = 64 x 64 BF16; packed B 2 rounds x 128 x 16 B =
    // 32 x 128 bytes; scales 4 groups x 8 threads x 16 B = 512 B.

    #define K64_ISSUE_LOADS(buf, kb) do { \
        { \
            unsigned int a_row_base = threadIdx.x >> 3; \
            unsigned int a_col = (threadIdx.x & 7) << 3; \
            unsigned int gc = (kb) + a_col; \
            _Pragma("unroll") \
            for (int rnd = 0; rnd < 4; rnd++) { \
                unsigned int row = rnd * 16 + a_row_base; \
                bool valid = (cta_m_local + row) < M_eff && (gc + 7 < K); \
                unsigned int a_row = (unsigned int)smem_tok_k64[row]; \
                moe_cp_async_pred_16(&smem_A_k64[(buf)][row][a_col], \
                    &A[(unsigned long long)a_row * K + gc], valid); \
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
                moe_cp_async_pred_16(&smem_Bp_k64[(buf)][kp_cur][ns], \
                    &B_expert[(unsigned long long)(gke >> 1) * N + gns], \
                    (gke + 1 < K) && (gns + 15 < N)); \
                if (kp_cur < K_STEP_T64 / GROUP_SIZE) { \
                    unsigned int sg = (kb) / GROUP_SIZE + kp_cur; \
                    moe_cp_async_pred_16(&smem_Bs_k64[(buf)][kp_cur][ns], \
                        &S_expert[(unsigned long long)sg * N + gns], \
                        (gns + 15 < N)); \
                } \
            } \
        } \
    } while(0)

    // 2026-09-25: Dequantise B to E4M3, four scale groups per 64-value step.
    #define K64_DEQUANT(buf) do { \
        unsigned int my_n = threadIdx.x; \
        __nv_fp8_e4m3 f0, f1, f2, f3; \
        *(unsigned char*)&f0 = smem_Bs_k64[(buf)][0][my_n]; \
        *(unsigned char*)&f1 = smem_Bs_k64[(buf)][1][my_n]; \
        *(unsigned char*)&f2 = smem_Bs_k64[(buf)][2][my_n]; \
        *(unsigned char*)&f3 = smem_Bs_k64[(buf)][3][my_n]; \
        float sv0 = (float)f0 * scale2; \
        float sv1 = (float)f1 * scale2; \
        float sv2 = (float)f2 * scale2; \
        float sv3 = (float)f3 * scale2; \
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

    // 2026-09-25: Two m16n8k32 MMAs per 8-column slice, over k 0..31 and 32..63; A fragments a4..a7 are converted
    // after the first-half MMAs.

    #define K64_COMPUTE_MMA(a_buf) do { \
        const unsigned short* sA = (const unsigned short*)smem_A_k64[(a_buf)]; \
        unsigned int fr0 = warp_m_offset + group_id, fr1 = fr0 + 8; \
        unsigned int a0 = moe_bf16x4_to_e4m3x4(&sA[fr0 * ast64 + tid * 4]); \
        unsigned int a1 = moe_bf16x4_to_e4m3x4(&sA[fr1 * ast64 + tid * 4]); \
        unsigned int a2 = moe_bf16x4_to_e4m3x4(&sA[fr0 * ast64 + 16 + tid * 4]); \
        unsigned int a3 = moe_bf16x4_to_e4m3x4(&sA[fr1 * ast64 + 16 + tid * 4]); \
        _Pragma("unroll") \
        for (int nt = 0; nt < 16; nt++) { \
            unsigned int nc = nt * 8 + group_id; \
            unsigned int b0 = *(const unsigned int*)&smem_B_fp8_k64[nc][4 * tid]; \
            unsigned int b1 = *(const unsigned int*)&smem_B_fp8_k64[nc][16 + 4 * tid]; \
            metrale_mma_e4m3(acc[nt], a0,a1,a2,a3, b0, b1); \
        } \
        unsigned int a4 = moe_bf16x4_to_e4m3x4(&sA[fr0 * ast64 + 32 + tid * 4]); \
        unsigned int a5 = moe_bf16x4_to_e4m3x4(&sA[fr1 * ast64 + 32 + tid * 4]); \
        unsigned int a6 = moe_bf16x4_to_e4m3x4(&sA[fr0 * ast64 + 48 + tid * 4]); \
        unsigned int a7 = moe_bf16x4_to_e4m3x4(&sA[fr1 * ast64 + 48 + tid * 4]); \
        _Pragma("unroll") \
        for (int nt = 0; nt < 16; nt++) { \
            unsigned int nc = nt * 8 + group_id; \
            unsigned int b0 = *(const unsigned int*)&smem_B_fp8_k64[nc][32 + 4 * tid]; \
            unsigned int b1 = *(const unsigned int*)&smem_B_fp8_k64[nc][48 + 4 * tid]; \
            metrale_mma_e4m3(acc[nt], a4,a5,a6,a7, b0, b1); \
        } \
    } while(0)

    K64_ISSUE_LOADS(0, 0);
    moe_cp_async_commit();
    moe_cp_async_wait_all();
    __syncthreads();
    K64_DEQUANT(0);
    __syncthreads();

    int cur = 0;
    for (unsigned int k_base = K_STEP_T64; k_base < K; k_base += K_STEP_T64) {
        int nxt = 1 - cur;
        K64_ISSUE_LOADS(nxt, k_base);
        moe_cp_async_commit();
        K64_COMPUTE_MMA(cur);
        moe_cp_async_wait_all();
        __syncthreads();
        K64_DEQUANT(nxt);
        __syncthreads();
        cur = nxt;
    }
    K64_COMPUTE_MMA(cur);

    #undef K64_ISSUE_LOADS
    #undef K64_DEQUANT
    #undef K64_COMPUTE_MMA

    #pragma unroll
    for (int nt = 0; nt < 16; nt++) {
        unsigned int c0 = cta_n + nt*8 + tid*2;
        unsigned int c1 = c0 + 1;
        unsigned int r0 = cta_m + warp_m_offset + group_id;
        unsigned int r1 = r0 + 8;
        bool r0v = (int)(warp_m_offset + group_id + cta_m_local) < M_expert;
        bool r1v = (int)(warp_m_offset + group_id + 8 + cta_m_local) < M_expert;
        if (r0v && c0 < N) C[r0*N+c0] = __float2bfloat16(acc[nt][0]);
        if (r0v && c1 < N) C[r0*N+c1] = __float2bfloat16(acc[nt][1]);
        if (r1v && c0 < N) C[r1*N+c0] = __float2bfloat16(acc[nt][2]);
        if (r1v && c1 < N) C[r1*N+c1] = __float2bfloat16(acc[nt][3]);
    }
}

// 2026-09-25: `moe_w4a16_fused_gate_up_t_k64`: the K64 pipeline above over gate and up weights in one launch;
// see `moe_w4a16_fused_gate_up_t` for the column split.




extern "C" __global__ void moe_w4a16_fused_gate_up_t_k64(
    const __nv_bfloat16* __restrict__ A,
    const unsigned long long* __restrict__ gate_packed_ptrs,
    const unsigned long long* __restrict__ gate_scale_ptrs,
    const float* __restrict__ gate_scale2_vals,
    const unsigned long long* __restrict__ up_packed_ptrs,
    const unsigned long long* __restrict__ up_scale_ptrs,
    const float* __restrict__ up_scale2_vals,
    __nv_bfloat16* __restrict__ C_gate,
    __nv_bfloat16* __restrict__ C_up,
    const int* __restrict__ expert_offsets,
    const int* __restrict__ sorted_token_ids,
    unsigned int num_experts,
    unsigned int N,
    unsigned int K
) {
    const unsigned int expert_id = blockIdx.z;
    if (expert_id >= num_experts) return;

    const int m_start = expert_offsets[expert_id];
    const int m_end = expert_offsets[expert_id + 1];
    const int M_expert = m_end - m_start;
    if (M_expert <= 0) return;

    const int cta_m_local = blockIdx.y * M_TILE;
    if (cta_m_local >= M_expert) return;

    const unsigned int global_n = blockIdx.x * N_TILE_LG;
    const bool is_up = (global_n >= N);
    const unsigned int cta_n = is_up ? (global_n - N) : global_n;

    const unsigned char* B_expert;
    const unsigned char* S_expert;
    float scale2;
    __nv_bfloat16* C;
    if (is_up) {
        B_expert = (const unsigned char*)up_packed_ptrs[expert_id];
        S_expert = (const unsigned char*)up_scale_ptrs[expert_id];
        scale2 = up_scale2_vals[expert_id];
        C = C_up;
    } else {
        B_expert = (const unsigned char*)gate_packed_ptrs[expert_id];
        S_expert = (const unsigned char*)gate_scale_ptrs[expert_id];
        scale2 = gate_scale2_vals[expert_id];
        C = C_gate;
    }

    if (B_expert == 0) return;

    const unsigned int cta_m = m_start + cta_m_local;
    const unsigned int warp_id = threadIdx.x / 32;
    const unsigned int lane_id = threadIdx.x % 32;
    const unsigned int warp_m_offset = warp_id * 16;
    const unsigned int group_id = lane_id >> 2;
    const unsigned int tid = lane_id & 3;

    __shared__ __nv_bfloat16 smem_A_fgu64[2][M_TILE][K_STEP_T64 + PAD_T64];
    __shared__ unsigned char smem_Bp_fgu64[2][K_STEP_T64 / 2][N_TILE_LG + BP_PAD];
    __shared__ unsigned char smem_Bs_fgu64[2][K_STEP_T64 / GROUP_SIZE][N_TILE_LG + BP_PAD];
    // 2026-09-25: 80-byte rows, as in `moe_w4a16_grouped_gemm_ptrtable_t_k64`.
    __shared__ unsigned char smem_B_fp8_fgu64[N_TILE_LG][K_STEP_T64 + 16];
    __shared__ float smem_LUT_fgu64[16];
    __shared__ int smem_tok_fgu64[M_TILE];

    if (threadIdx.x < 16) smem_LUT_fgu64[threadIdx.x] = E2M1_LUT_MOE[threadIdx.x];
    if (threadIdx.x < M_TILE) {
        int local_row = threadIdx.x;
        if (sorted_token_ids && (cta_m_local + local_row) < (unsigned int)M_expert)
            smem_tok_fgu64[local_row] = sorted_token_ids[cta_m + local_row];
        else
            smem_tok_fgu64[local_row] = (int)(cta_m + local_row);
    }
    __syncthreads();

    float acc[16][4];
    #pragma unroll
    for (int i = 0; i < 16; i++) {
        acc[i][0] = 0.0f; acc[i][1] = 0.0f;
        acc[i][2] = 0.0f; acc[i][3] = 0.0f;
    }

    const unsigned int ast_fgu64 = K_STEP_T64 + PAD_T64;
    const unsigned int M_eff = (unsigned int)M_expert;

    #define FGU64_ISSUE_LOADS(buf, kb) do { \
        { \
            unsigned int a_row_base = threadIdx.x >> 3; \
            unsigned int a_col = (threadIdx.x & 7) << 3; \
            unsigned int gc = (kb) + a_col; \
            _Pragma("unroll") \
            for (int rnd = 0; rnd < 4; rnd++) { \
                unsigned int row = rnd * 16 + a_row_base; \
                bool valid = (cta_m_local + row) < M_eff && (gc + 7 < K); \
                unsigned int a_row = (unsigned int)smem_tok_fgu64[row]; \
                moe_cp_async_pred_16(&smem_A_fgu64[(buf)][row][a_col], \
                    &A[(unsigned long long)a_row * K + gc], valid); \
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
                moe_cp_async_pred_16(&smem_Bp_fgu64[(buf)][kp_cur][ns], \
                    &B_expert[(unsigned long long)(gke >> 1) * N + gns], \
                    (gke + 1 < K) && (gns + 15 < N)); \
                if (kp_cur < K_STEP_T64 / GROUP_SIZE) { \
                    unsigned int sg = (kb) / GROUP_SIZE + kp_cur; \
                    moe_cp_async_pred_16(&smem_Bs_fgu64[(buf)][kp_cur][ns], \
                        &S_expert[(unsigned long long)sg * N + gns], \
                        (gns + 15 < N)); \
                } \
            } \
        } \
    } while(0)

    #define FGU64_DEQUANT(buf) do { \
        unsigned int my_n = threadIdx.x; \
        __nv_fp8_e4m3 f0, f1, f2, f3; \
        *(unsigned char*)&f0 = smem_Bs_fgu64[(buf)][0][my_n]; \
        *(unsigned char*)&f1 = smem_Bs_fgu64[(buf)][1][my_n]; \
        *(unsigned char*)&f2 = smem_Bs_fgu64[(buf)][2][my_n]; \
        *(unsigned char*)&f3 = smem_Bs_fgu64[(buf)][3][my_n]; \
        float sv0 = (float)f0 * scale2; \
        float sv1 = (float)f1 * scale2; \
        float sv2 = (float)f2 * scale2; \
        float sv3 = (float)f3 * scale2; \
        _Pragma("unroll") \
        for (int kp = 0; kp < 8; kp++) { \
            unsigned char packed = smem_Bp_fgu64[(buf)][kp][my_n]; \
            float lo = smem_LUT_fgu64[packed & 0xF] * sv0; \
            float hi = smem_LUT_fgu64[packed >> 4] * sv0; \
            unsigned short fp8_pair; \
            fp8_pair = metrale_cvt_e4m3x2_f32(hi, lo); \
            *(unsigned short*)&smem_B_fp8_fgu64[my_n][kp * 2] = fp8_pair; \
        } \
        _Pragma("unroll") \
        for (int kp = 8; kp < 16; kp++) { \
            unsigned char packed = smem_Bp_fgu64[(buf)][kp][my_n]; \
            float lo = smem_LUT_fgu64[packed & 0xF] * sv1; \
            float hi = smem_LUT_fgu64[packed >> 4] * sv1; \
            unsigned short fp8_pair; \
            fp8_pair = metrale_cvt_e4m3x2_f32(hi, lo); \
            *(unsigned short*)&smem_B_fp8_fgu64[my_n][kp * 2] = fp8_pair; \
        } \
        _Pragma("unroll") \
        for (int kp = 16; kp < 24; kp++) { \
            unsigned char packed = smem_Bp_fgu64[(buf)][kp][my_n]; \
            float lo = smem_LUT_fgu64[packed & 0xF] * sv2; \
            float hi = smem_LUT_fgu64[packed >> 4] * sv2; \
            unsigned short fp8_pair; \
            fp8_pair = metrale_cvt_e4m3x2_f32(hi, lo); \
            *(unsigned short*)&smem_B_fp8_fgu64[my_n][kp * 2] = fp8_pair; \
        } \
        _Pragma("unroll") \
        for (int kp = 24; kp < 32; kp++) { \
            unsigned char packed = smem_Bp_fgu64[(buf)][kp][my_n]; \
            float lo = smem_LUT_fgu64[packed & 0xF] * sv3; \
            float hi = smem_LUT_fgu64[packed >> 4] * sv3; \
            unsigned short fp8_pair; \
            fp8_pair = metrale_cvt_e4m3x2_f32(hi, lo); \
            *(unsigned short*)&smem_B_fp8_fgu64[my_n][kp * 2] = fp8_pair; \
        } \
    } while(0)

    #define FGU64_COMPUTE_MMA(a_buf) do { \
        const unsigned short* sA = (const unsigned short*)smem_A_fgu64[(a_buf)]; \
        unsigned int fr0 = warp_m_offset + group_id, fr1 = fr0 + 8; \
        unsigned int a0 = moe_bf16x4_to_e4m3x4(&sA[fr0 * ast_fgu64 + tid * 4]); \
        unsigned int a1 = moe_bf16x4_to_e4m3x4(&sA[fr1 * ast_fgu64 + tid * 4]); \
        unsigned int a2 = moe_bf16x4_to_e4m3x4(&sA[fr0 * ast_fgu64 + 16 + tid * 4]); \
        unsigned int a3 = moe_bf16x4_to_e4m3x4(&sA[fr1 * ast_fgu64 + 16 + tid * 4]); \
        _Pragma("unroll") \
        for (int nt = 0; nt < 16; nt++) { \
            unsigned int nc = nt * 8 + group_id; \
            unsigned int b0 = *(const unsigned int*)&smem_B_fp8_fgu64[nc][4 * tid]; \
            unsigned int b1 = *(const unsigned int*)&smem_B_fp8_fgu64[nc][16 + 4 * tid]; \
            metrale_mma_e4m3(acc[nt], a0,a1,a2,a3, b0, b1); \
        } \
        unsigned int a4 = moe_bf16x4_to_e4m3x4(&sA[fr0 * ast_fgu64 + 32 + tid * 4]); \
        unsigned int a5 = moe_bf16x4_to_e4m3x4(&sA[fr1 * ast_fgu64 + 32 + tid * 4]); \
        unsigned int a6 = moe_bf16x4_to_e4m3x4(&sA[fr0 * ast_fgu64 + 48 + tid * 4]); \
        unsigned int a7 = moe_bf16x4_to_e4m3x4(&sA[fr1 * ast_fgu64 + 48 + tid * 4]); \
        _Pragma("unroll") \
        for (int nt = 0; nt < 16; nt++) { \
            unsigned int nc = nt * 8 + group_id; \
            unsigned int b0 = *(const unsigned int*)&smem_B_fp8_fgu64[nc][32 + 4 * tid]; \
            unsigned int b1 = *(const unsigned int*)&smem_B_fp8_fgu64[nc][48 + 4 * tid]; \
            metrale_mma_e4m3(acc[nt], a4,a5,a6,a7, b0, b1); \
        } \
    } while(0)

    FGU64_ISSUE_LOADS(0, 0);
    moe_cp_async_commit();
    moe_cp_async_wait_all();
    __syncthreads();
    FGU64_DEQUANT(0);
    __syncthreads();

    int cur = 0;
    for (unsigned int k_base = K_STEP_T64; k_base < K; k_base += K_STEP_T64) {
        int nxt = 1 - cur;
        FGU64_ISSUE_LOADS(nxt, k_base);
        moe_cp_async_commit();
        FGU64_COMPUTE_MMA(cur);
        moe_cp_async_wait_all();
        __syncthreads();
        FGU64_DEQUANT(nxt);
        __syncthreads();
        cur = nxt;
    }
    FGU64_COMPUTE_MMA(cur);

    #undef FGU64_ISSUE_LOADS
    #undef FGU64_DEQUANT
    #undef FGU64_COMPUTE_MMA

    #pragma unroll
    for (int nt = 0; nt < 16; nt++) {
        unsigned int c0 = cta_n + nt*8 + tid*2;
        unsigned int c1 = c0 + 1;
        unsigned int r0 = cta_m + warp_m_offset + group_id;
        unsigned int r1 = r0 + 8;
        bool r0v = (int)(warp_m_offset + group_id + cta_m_local) < M_expert;
        bool r1v = (int)(warp_m_offset + group_id + 8 + cta_m_local) < M_expert;
        if (r0v && c0 < N) C[r0*N+c0] = __float2bfloat16(acc[nt][0]);
        if (r0v && c1 < N) C[r0*N+c1] = __float2bfloat16(acc[nt][1]);
        if (r1v && c0 < N) C[r1*N+c0] = __float2bfloat16(acc[nt][2]);
        if (r1v && c1 < N) C[r1*N+c1] = __float2bfloat16(acc[nt][3]);
    }
}

// 2026-09-25: `moe_w4a16_fused_gate_up_t`: gate and up projections in one launch. Grid x covers 2 * N columns in
// 128-column tiles: tiles starting below N use the gate weights and write C_gate, the rest the up weights and
// C_up, so N must be a multiple of 128 for no tile to straddle both. Grid (ceil(2 * N / 128), max_m_tiles,
// num_experts).






extern "C" __global__ void moe_w4a16_fused_gate_up_t(
    const __nv_bfloat16* __restrict__ A,

    const unsigned long long* __restrict__ gate_packed_ptrs,
    const unsigned long long* __restrict__ gate_scale_ptrs,
    const float* __restrict__ gate_scale2_vals,

    const unsigned long long* __restrict__ up_packed_ptrs,
    const unsigned long long* __restrict__ up_scale_ptrs,
    const float* __restrict__ up_scale2_vals,

    __nv_bfloat16* __restrict__ C_gate,
    __nv_bfloat16* __restrict__ C_up,
    const int* __restrict__ expert_offsets,
    const int* __restrict__ sorted_token_ids,
    unsigned int num_experts,
    unsigned int N,
    unsigned int K
) {
    const unsigned int expert_id = blockIdx.z;
    if (expert_id >= num_experts) return;

    const int m_start = expert_offsets[expert_id];
    const int m_end = expert_offsets[expert_id + 1];
    const int M_expert = m_end - m_start;
    if (M_expert <= 0) return;

    const int cta_m_local = blockIdx.y * M_TILE;
    if (cta_m_local >= M_expert) return;


    const unsigned int global_n = blockIdx.x * N_TILE_LG;
    const bool is_up = (global_n >= N);
    const unsigned int cta_n = is_up ? (global_n - N) : global_n;


    const unsigned char* B_expert;
    const unsigned char* S_expert;
    float scale2;
    __nv_bfloat16* C;
    if (is_up) {
        B_expert = (const unsigned char*)up_packed_ptrs[expert_id];
        S_expert = (const unsigned char*)up_scale_ptrs[expert_id];
        scale2 = up_scale2_vals[expert_id];
        C = C_up;
    } else {
        B_expert = (const unsigned char*)gate_packed_ptrs[expert_id];
        S_expert = (const unsigned char*)gate_scale_ptrs[expert_id];
        scale2 = gate_scale2_vals[expert_id];
        C = C_gate;
    }

    if (B_expert == 0) return;

    const unsigned int cta_m = m_start + cta_m_local;
    const unsigned int warp_id = threadIdx.x / 32;
    const unsigned int lane_id = threadIdx.x % 32;
    const unsigned int warp_m_offset = warp_id * 16;
    const unsigned int group_id = lane_id >> 2;
    const unsigned int tid = lane_id & 3;

    __shared__ __nv_bfloat16 smem_A[2][M_TILE][K_STEP_T + PAD_T];
    __shared__ unsigned char smem_Bp[2][K_STEP_T / 2][N_TILE_LG + BP_PAD];
    __shared__ unsigned char smem_Bs[2][K_STEP_T / GROUP_SIZE][N_TILE_LG + BP_PAD];
    // 2026-09-25: The +4 pad of `moe_w4a16_grouped_gemm_ptrtable_t`: unpadded, the dequant writes are 8-way conflicts.
    __shared__ unsigned char smem_B_fp8[N_TILE_LG][K_STEP_T + 4];
    __shared__ float smem_LUT[16];
    __shared__ int smem_tok[M_TILE];

    if (threadIdx.x < 16) {
        smem_LUT[threadIdx.x] = E2M1_LUT_MOE[threadIdx.x];
    }
    if (threadIdx.x < M_TILE) {
        int local_row = threadIdx.x;
        if (sorted_token_ids && (cta_m_local + local_row) < (unsigned int)M_expert) {
            smem_tok[local_row] = sorted_token_ids[cta_m + local_row];
        } else {
            smem_tok[local_row] = (int)(cta_m + local_row);
        }
    }
    __syncthreads();

    float acc[16][4];
    #pragma unroll
    for (int i = 0; i < 16; i++) {
        acc[i][0] = 0.0f; acc[i][1] = 0.0f;
        acc[i][2] = 0.0f; acc[i][3] = 0.0f;
    }

    const unsigned int a_stride = K_STEP_T + PAD_T;
    const unsigned int M_eff = (unsigned int)M_expert;

    // 2026-09-25: The same load, dequant and MMA steps as `moe_w4a16_grouped_gemm_ptrtable_t`.
    #define FGU_ISSUE_LOADS(buf, kb) do { \
        { \
            unsigned int a_row_base = threadIdx.x >> 2; \
            unsigned int a_col = (threadIdx.x & 3) << 3; \
            unsigned int gc = (kb) + a_col; \
            _Pragma("unroll") \
            for (int rnd = 0; rnd < 2; rnd++) { \
                unsigned int row = rnd * 32 + a_row_base; \
                bool valid = (cta_m_local + row) < M_eff && (gc + 7 < K); \
                unsigned int a_row = (unsigned int)smem_tok[row]; \
                moe_cp_async_pred_16(&smem_A[(buf)][row][a_col], \
                    &A[(unsigned long long)a_row * K + gc], valid); \
            } \
        } \
        { \
            unsigned int kp = threadIdx.x >> 3; \
            unsigned int ns = (threadIdx.x & 7) << 4; \
            unsigned int gke = (kb) + (kp << 1); \
            unsigned int gns = cta_n + ns; \
            moe_cp_async_pred_16(&smem_Bp[(buf)][kp][ns], \
                &B_expert[(unsigned long long)(gke >> 1) * N + gns], \
                (gke + 1 <= K) && (gns + 15 < N)); \
            if (kp < K_STEP_T / GROUP_SIZE) { \
                unsigned int sg = (kb) / GROUP_SIZE + kp; \
                moe_cp_async_pred_16(&smem_Bs[(buf)][kp][ns], \
                    &S_expert[(unsigned long long)sg * N + gns], \
                    (gns + 15 < N)); \
            } \
        } \
    } while(0)

    #define FGU_DEQUANT(buf) do { \
        unsigned int my_n = threadIdx.x; \
        unsigned char sb0 = smem_Bs[(buf)][0][my_n]; \
        unsigned char sb1 = smem_Bs[(buf)][1][my_n]; \
        __nv_fp8_e4m3 f0, f1; \
        *(unsigned char*)&f0 = sb0; \
        *(unsigned char*)&f1 = sb1; \
        float sv0 = (float)f0 * scale2; \
        float sv1 = (float)f1 * scale2; \
        _Pragma("unroll") \
        for (int kp = 0; kp < 8; kp++) { \
            unsigned char packed = smem_Bp[(buf)][kp][my_n]; \
            float lo = smem_LUT[packed & 0xF] * sv0; \
            float hi = smem_LUT[packed >> 4] * sv0; \
            unsigned short fp8_pair; \
            fp8_pair = metrale_cvt_e4m3x2_f32(hi, lo); \
            *(unsigned short*)&smem_B_fp8[my_n][kp * 2] = fp8_pair; \
        } \
        _Pragma("unroll") \
        for (int kp = 8; kp < 16; kp++) { \
            unsigned char packed = smem_Bp[(buf)][kp][my_n]; \
            float lo = smem_LUT[packed & 0xF] * sv1; \
            float hi = smem_LUT[packed >> 4] * sv1; \
            unsigned short fp8_pair; \
            fp8_pair = metrale_cvt_e4m3x2_f32(hi, lo); \
            *(unsigned short*)&smem_B_fp8[my_n][kp * 2] = fp8_pair; \
        } \
    } while(0)

    #define FGU_COMPUTE_MMA(a_buf) do { \
        const unsigned short* sA = (const unsigned short*)smem_A[(a_buf)]; \
        unsigned int fr0 = warp_m_offset + group_id, fr1 = fr0 + 8; \
        unsigned int a0 = moe_bf16x4_to_e4m3x4(&sA[fr0 * a_stride + tid * 4]); \
        unsigned int a1 = moe_bf16x4_to_e4m3x4(&sA[fr1 * a_stride + tid * 4]); \
        unsigned int a2 = moe_bf16x4_to_e4m3x4(&sA[fr0 * a_stride + 16 + tid * 4]); \
        unsigned int a3 = moe_bf16x4_to_e4m3x4(&sA[fr1 * a_stride + 16 + tid * 4]); \
        _Pragma("unroll") \
        for (int nt = 0; nt < 16; nt++) { \
            unsigned int nc = nt * 8 + group_id; \
            unsigned int b0 = *(const unsigned int*)&smem_B_fp8[nc][4 * tid]; \
            unsigned int b1 = *(const unsigned int*)&smem_B_fp8[nc][16 + 4 * tid]; \
            metrale_mma_e4m3(acc[nt], a0,a1,a2,a3, b0, b1); \
        } \
    } while(0)

    FGU_ISSUE_LOADS(0, 0);
    moe_cp_async_commit();
    moe_cp_async_wait_all();
    __syncthreads();
    FGU_DEQUANT(0);
    __syncthreads();

    int cur = 0;
    for (unsigned int k_base = K_STEP_T; k_base < K; k_base += K_STEP_T) {
        int nxt = 1 - cur;
        FGU_ISSUE_LOADS(nxt, k_base);
        moe_cp_async_commit();
        FGU_COMPUTE_MMA(cur);
        moe_cp_async_wait_all();
        __syncthreads();
        FGU_DEQUANT(nxt);
        __syncthreads();
        cur = nxt;
    }
    FGU_COMPUTE_MMA(cur);

    #undef FGU_ISSUE_LOADS
    #undef FGU_DEQUANT
    #undef FGU_COMPUTE_MMA

    #pragma unroll
    for (int nt = 0; nt < 16; nt++) {
        unsigned int c0 = cta_n + nt*8 + tid*2;
        unsigned int c1 = c0 + 1;
        unsigned int r0 = cta_m + warp_m_offset + group_id;
        unsigned int r1 = r0 + 8;
        bool r0v = (int)(warp_m_offset + group_id + cta_m_local) < M_expert;
        bool r1v = (int)(warp_m_offset + group_id + 8 + cta_m_local) < M_expert;
        if (r0v && c0 < N) C[r0*N+c0] = __float2bfloat16(acc[nt][0]);
        if (r0v && c1 < N) C[r0*N+c1] = __float2bfloat16(acc[nt][1]);
        if (r1v && c0 < N) C[r1*N+c0] = __float2bfloat16(acc[nt][2]);
        if (r1v && c1 < N) C[r1*N+c1] = __float2bfloat16(acc[nt][3]);
    }
}

// 2026-09-25: `moe_fp8_grouped_gemm_ptrtable_t`: `moe_w4a16_grouped_gemm_ptrtable_t` with A already E4M3,
// [rows, K] bytes, so A needs no conversion in the loop and its shared buffer is 64 x 32 bytes per stage (2 KB,
// against 5 KB for padded BF16). Grid (ceil(N / 128), max_m_tiles, num_experts).






extern "C" __global__ void moe_fp8_grouped_gemm_ptrtable_t(
    const unsigned char* __restrict__ A_fp8,
    const unsigned long long* __restrict__ B_packed_ptrs,
    const unsigned long long* __restrict__ B_scale_ptrs,
    const float* __restrict__ scale2_vals,
    __nv_bfloat16* __restrict__ C,
    const int* __restrict__ expert_offsets,
    const int* __restrict__ sorted_token_ids,
    unsigned int num_experts,
    unsigned int N,
    unsigned int K
) {
    const unsigned int expert_id = blockIdx.z;
    if (expert_id >= num_experts) return;

    const int m_start = expert_offsets[expert_id];
    const int m_end = expert_offsets[expert_id + 1];
    const int M_expert = m_end - m_start;
    if (M_expert <= 0) return;

    const int cta_m_local = blockIdx.y * M_TILE;
    if (cta_m_local >= M_expert) return;

    const unsigned int cta_m = m_start + cta_m_local;
    const unsigned int cta_n = blockIdx.x * N_TILE_LG;

    const unsigned char* B_exp = (const unsigned char*)B_packed_ptrs[expert_id];
    const unsigned char* S_exp = (const unsigned char*)B_scale_ptrs[expert_id];
    const float scale2 = scale2_vals[expert_id];

    if (B_exp == 0) return;

    const unsigned int warp_id = threadIdx.x / 32;
    const unsigned int lane_id = threadIdx.x % 32;
    const unsigned int warp_m_offset = warp_id * 16;
    const unsigned int group_id = lane_id >> 2;
    const unsigned int tid = lane_id & 3;

    __shared__ unsigned char smem_Af2[2][M_TILE][K_STEP_T];
    __shared__ unsigned char smem_Bp2[2][K_STEP_T / 2][N_TILE_LG + BP_PAD];
    __shared__ unsigned char smem_Bs2[2][K_STEP_T / GROUP_SIZE][N_TILE_LG + BP_PAD];
    __shared__ unsigned char smem_B2_fp8[N_TILE_LG][K_STEP_T];
    __shared__ float smem_LUT2[16];
    __shared__ int smem_tok2[M_TILE];

    if (threadIdx.x < 16) {
        smem_LUT2[threadIdx.x] = E2M1_LUT_MOE[threadIdx.x];
    }
    if (threadIdx.x < M_TILE) {
        int local_row = threadIdx.x;
        if (sorted_token_ids && (cta_m_local + local_row) < (unsigned int)M_expert) {
            smem_tok2[local_row] = sorted_token_ids[cta_m + local_row];
        } else {
            smem_tok2[local_row] = (int)(cta_m + local_row);
        }
    }
    __syncthreads();

    float acc[16][4];
    #pragma unroll
    for (int i = 0; i < 16; i++) {
        acc[i][0] = 0.0f; acc[i][1] = 0.0f;
        acc[i][2] = 0.0f; acc[i][3] = 0.0f;
    }

    const unsigned int M_eff = (unsigned int)M_expert;

    #define MOE_FF_LOADS(buf, kb) do { \
        { \
            unsigned int a_row_base = threadIdx.x >> 1; \
            unsigned int a_col = (threadIdx.x & 1) << 4; \
            unsigned int gc = (kb) + a_col; \
            unsigned int row = a_row_base; \
            bool valid = (cta_m_local + row) < M_eff && (gc + 15 < K); \
            unsigned int a_row = (unsigned int)smem_tok2[row]; \
            moe_cp_async_pred_16(&smem_Af2[(buf)][row][a_col], \
                &A_fp8[(unsigned long long)a_row * K + gc], valid); \
        } \
        { \
            unsigned int kp = threadIdx.x >> 3; \
            unsigned int ns = (threadIdx.x & 7) << 4; \
            unsigned int gke = (kb) + (kp << 1); \
            unsigned int gns = cta_n + ns; \
            moe_cp_async_pred_16(&smem_Bp2[(buf)][kp][ns], \
                &B_exp[(unsigned long long)(gke >> 1) * N + gns], \
                (gke + 1 <= K) && (gns + 15 < N)); \
            if (kp < K_STEP_T / GROUP_SIZE) { \
                unsigned int sg = (kb) / GROUP_SIZE + kp; \
                moe_cp_async_pred_16(&smem_Bs2[(buf)][kp][ns], \
                    &S_exp[(unsigned long long)sg * N + gns], \
                    (gns + 15 < N)); \
            } \
        } \
    } while(0)

    #define MOE_FF_DEQUANT(buf) do { \
        unsigned int my_n = threadIdx.x; \
        unsigned char sb0 = smem_Bs2[(buf)][0][my_n]; \
        unsigned char sb1 = smem_Bs2[(buf)][1][my_n]; \
        __nv_fp8_e4m3 f0, f1; \
        *(unsigned char*)&f0 = sb0; \
        *(unsigned char*)&f1 = sb1; \
        float sv0 = (float)f0 * scale2; \
        float sv1 = (float)f1 * scale2; \
        _Pragma("unroll") \
        for (int kp = 0; kp < 8; kp++) { \
            unsigned char packed = smem_Bp2[(buf)][kp][my_n]; \
            float lo = smem_LUT2[packed & 0xF] * sv0; \
            float hi = smem_LUT2[packed >> 4] * sv0; \
            unsigned short fp8_pair; \
            fp8_pair = metrale_cvt_e4m3x2_f32(hi, lo); \
            *(unsigned short*)&smem_B2_fp8[my_n][kp * 2] = fp8_pair; \
        } \
        _Pragma("unroll") \
        for (int kp = 8; kp < 16; kp++) { \
            unsigned char packed = smem_Bp2[(buf)][kp][my_n]; \
            float lo = smem_LUT2[packed & 0xF] * sv1; \
            float hi = smem_LUT2[packed >> 4] * sv1; \
            unsigned short fp8_pair; \
            fp8_pair = metrale_cvt_e4m3x2_f32(hi, lo); \
            *(unsigned short*)&smem_B2_fp8[my_n][kp * 2] = fp8_pair; \
        } \
    } while(0)

    #define MOE_FF_COMPUTE(a_buf) do { \
        unsigned int fr0 = warp_m_offset + group_id, fr1 = fr0 + 8; \
        unsigned int a0 = *(const unsigned int*)&smem_Af2[(a_buf)][fr0][4 * tid]; \
        unsigned int a1 = *(const unsigned int*)&smem_Af2[(a_buf)][fr1][4 * tid]; \
        unsigned int a2 = *(const unsigned int*)&smem_Af2[(a_buf)][fr0][16 + 4 * tid]; \
        unsigned int a3 = *(const unsigned int*)&smem_Af2[(a_buf)][fr1][16 + 4 * tid]; \
        _Pragma("unroll") \
        for (int nt = 0; nt < 16; nt++) { \
            unsigned int nc = nt * 8 + group_id; \
            unsigned int b0 = *(const unsigned int*)&smem_B2_fp8[nc][4 * tid]; \
            unsigned int b1 = *(const unsigned int*)&smem_B2_fp8[nc][16 + 4 * tid]; \
            metrale_mma_e4m3(acc[nt], a0,a1,a2,a3, b0, b1); \
        } \
    } while(0)

    MOE_FF_LOADS(0, 0);
    moe_cp_async_commit();
    moe_cp_async_wait_all();
    __syncthreads();
    MOE_FF_DEQUANT(0);
    __syncthreads();

    int cur = 0;
    for (unsigned int k_base = K_STEP_T; k_base < K; k_base += K_STEP_T) {
        int nxt = 1 - cur;
        MOE_FF_LOADS(nxt, k_base);
        moe_cp_async_commit();
        MOE_FF_COMPUTE(cur);
        moe_cp_async_wait_all();
        __syncthreads();
        MOE_FF_DEQUANT(nxt);
        __syncthreads();
        cur = nxt;
    }
    MOE_FF_COMPUTE(cur);

    #undef MOE_FF_LOADS
    #undef MOE_FF_DEQUANT
    #undef MOE_FF_COMPUTE

    #pragma unroll
    for (int nt = 0; nt < 16; nt++) {
        unsigned int c0 = cta_n + nt*8 + tid*2;
        unsigned int c1 = c0 + 1;
        unsigned int r0 = cta_m + warp_m_offset + group_id;
        unsigned int r1 = r0 + 8;
        bool r0v = (int)(warp_m_offset + group_id + cta_m_local) < M_expert;
        bool r1v = (int)(warp_m_offset + group_id + 8 + cta_m_local) < M_expert;
        if (r0v && c0 < N) C[r0*N+c0] = __float2bfloat16(acc[nt][0]);
        if (r0v && c1 < N) C[r0*N+c1] = __float2bfloat16(acc[nt][1]);
        if (r1v && c0 < N) C[r1*N+c0] = __float2bfloat16(acc[nt][2]);
        if (r1v && c1 < N) C[r1*N+c1] = __float2bfloat16(acc[nt][3]);
    }
}
