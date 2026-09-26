// SPDX-License-Identifier: MIT OR Apache-2.0

// 2026-09-25: W4A16 GEMM v3, w4a16_gemm_t_m128_v3: C [M, N] BF16 = A [M, K] BF16 x dequant(B) with E4M3
// tensor-core MMAs. It is looked up only when METRALE_W4A16_VARIANT=v3 (layers::w4a16_v3_kernel);
// minimax-m2-229b and step3p7-flash compile it.
//
// Owner: gb10 kernels (minimax-m2-229b).
// Invariants: none beyond the types.
//
// B_packed is [K / 2, N]: byte (k / 2, n) holds even k in its low nibble. B_scale is [K / 16, N] E4M3, and a
// weight is E2M1 code x E4M3 scale x scale2. A is converted to E4M3 (satfinite) before the MMA.
// Grid (ceil(N / 128), ceil(M / 128)), block 256 (ops::w4a16_gemm_n128_m128_v3): warps 0-3 compute CTA rows
// 0-63 and warps 4-7 rows 64-127. K steps by 64 (w4a16_gemm_v2.cu steps by 32), four scale groups per step,
// through a 2-stage cp.async pipeline. The shared arrays declare 55,488 bytes: A 36,864, Bp 9,216, Bs 1,152,
// the E4M3 B tile 8,192 and the LUT 64. The A, Bp and Bs rows are 144 bytes apart, so every 16-byte
// cp.async destination stays aligned.







#include <cuda_bf16.h>
#include <cuda_fp8.h>

#define V3_M_TILE       64
#define V3_N_TILE       128
#define V3_K_STEP       64
#define V3_PAD          8
#define V3_BP_PAD       16
#define V3_GROUP_SIZE   16
#define V3_NUM_GROUPS   4

__device__ __constant__ float V3_E2M1_LUT[16] = {
     0.0f,  0.5f,  1.0f,  1.5f,  2.0f,  3.0f,  4.0f,  6.0f,
    -0.0f, -0.5f, -1.0f, -1.5f, -2.0f, -3.0f, -4.0f, -6.0f
};

__device__ __forceinline__ void v3_cp_async_pred_16(void* dst_smem, const void* src_gmem, bool pred) {
    unsigned int dst = __cvta_generic_to_shared(dst_smem);
    unsigned int src_bytes = pred ? 16 : 0;
    asm volatile("cp.async.ca.shared.global [%0], [%1], 16, %2;"
                 :: "r"(dst), "l"(src_gmem), "r"(src_bytes));
}

__device__ __forceinline__ void v3_cp_async_commit() {
    asm volatile("cp.async.commit_group;");
}

__device__ __forceinline__ void v3_cp_async_wait_all() {
    asm volatile("cp.async.wait_group 0;");
}

__device__ __forceinline__ unsigned int v3_bf16x4_to_e4m3x4(const unsigned short* src) {
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
    asm volatile("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(h0) : "f"(f1), "f"(f0));
    asm volatile("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(h1) : "f"(f3), "f"(f2));
    return ((unsigned int)h1 << 16) | (unsigned int)h0;
}

extern "C" __global__
__launch_bounds__(256, 1)
void w4a16_gemm_t_m128_v3(
    const __nv_bfloat16* __restrict__ A,
    const unsigned char* __restrict__ B_packed,
    const unsigned char* __restrict__ B_scale,
    const float scale2,
    __nv_bfloat16* __restrict__ C,
    unsigned int M, unsigned int N, unsigned int K
) {
    const unsigned int cta_n = blockIdx.x * V3_N_TILE;
    const unsigned int cta_m = blockIdx.y * (2 * V3_M_TILE);
    if (cta_m >= M) return;

    const unsigned int warp_id = threadIdx.x >> 5;
    const unsigned int lane_id = threadIdx.x & 31;
    const unsigned int chunk   = warp_id >> 2;
    const unsigned int sub     = warp_id & 3;
    const unsigned int warp_m_offset = sub * 16;
    const unsigned int group_id = lane_id >> 2;
    const unsigned int tid = lane_id & 3;

    __shared__ __nv_bfloat16 smem_A[2][2 * V3_M_TILE][V3_K_STEP + V3_PAD];
    __shared__ unsigned char smem_Bp[2][V3_K_STEP / 2][V3_N_TILE + V3_BP_PAD];
    __shared__ unsigned char smem_Bs[2][V3_NUM_GROUPS][V3_N_TILE + V3_BP_PAD];
    __shared__ unsigned char smem_B_fp8[V3_N_TILE][V3_K_STEP];
    __shared__ float smem_LUT[16];

    if (threadIdx.x < 16) smem_LUT[threadIdx.x] = V3_E2M1_LUT[threadIdx.x];

    float acc[16][4];
    #pragma unroll
    for (int i = 0; i < 16; i++) {
        acc[i][0] = 0.f; acc[i][1] = 0.f; acc[i][2] = 0.f; acc[i][3] = 0.f;
    }

    const unsigned int a_stride = V3_K_STEP + V3_PAD;



    // 2026-09-25: Loads for the K step at kb. A: 128 rows x 64 BF16, four 16-byte copies per thread. Bp: 32
    // packed K pairs x 128 N, one copy per thread. Bs: 4 scale rows x 128 N from threads 0-31. A copy whose
    // source is out of range zero-fills (source size 0).
    #define V3_LOADS(buf, kb) do { \
        { \
              \
            unsigned int a_row_base = threadIdx.x >> 3;          \
            unsigned int a_col      = (threadIdx.x & 7) << 3;    \
            unsigned int gc = (kb) + a_col; \
            _Pragma("unroll") \
            for (int rnd = 0; rnd < 4; rnd++) { \
                unsigned int row = (unsigned int)(rnd * 32) + a_row_base; \
                unsigned int gr  = cta_m + row; \
                v3_cp_async_pred_16(&smem_A[(buf)][row][a_col], \
                    &A[(unsigned long long)gr * K + gc], \
                    (gr < M) && (gc + 7 < K)); \
            } \
        } \
        { \
              \
            unsigned int kp  = threadIdx.x >> 3;           \
            unsigned int ns  = (threadIdx.x & 7) << 4;     \
            unsigned int gke = (kb) + (kp << 1); \
            unsigned int gns = cta_n + ns; \
            v3_cp_async_pred_16(&smem_Bp[(buf)][kp][ns], \
                &B_packed[(unsigned long long)(gke >> 1) * N + gns], \
                (gke + 1 <= K) && (gns + 15 < N)); \
              \
            if (kp < V3_NUM_GROUPS) { \
                unsigned int sg = (kb) / V3_GROUP_SIZE + kp; \
                v3_cp_async_pred_16(&smem_Bs[(buf)][kp][ns], \
                    &B_scale[(unsigned long long)sg * N + gns], \
                    (gns + 15 < N)); \
            } \
        } \
    } while(0)


    // 2026-09-25: Threads 0-127 each dequantize one N column of the step into 64 E4M3 values.
    #define V3_DEQUANT(buf) do { \
        if (threadIdx.x < 128) { \
            unsigned int my_n = threadIdx.x; \
            unsigned char sb0 = smem_Bs[(buf)][0][my_n]; \
            unsigned char sb1 = smem_Bs[(buf)][1][my_n]; \
            unsigned char sb2 = smem_Bs[(buf)][2][my_n]; \
            unsigned char sb3 = smem_Bs[(buf)][3][my_n]; \
            __nv_fp8_e4m3 f0, f1, f2, f3; \
            *(unsigned char*)&f0 = sb0; *(unsigned char*)&f1 = sb1; \
            *(unsigned char*)&f2 = sb2; *(unsigned char*)&f3 = sb3; \
            float sv0 = (float)f0 * scale2, sv1 = (float)f1 * scale2; \
            float sv2 = (float)f2 * scale2, sv3 = (float)f3 * scale2; \
            /* 2026-09-25: Packed rows 8g..8g+7 (K values 16g..16g+15) use scale group g. \
             */ \
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
            _Pragma("unroll") \
            for (int kp = 16; kp < 24; kp++) { \
                unsigned char packed = smem_Bp[(buf)][kp][my_n]; \
                float lo = smem_LUT[packed & 0xF] * sv2; \
                float hi = smem_LUT[packed >> 4]  * sv2; \
                unsigned short fp8_pair; \
                asm volatile("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" \
                             : "=h"(fp8_pair) : "f"(hi), "f"(lo)); \
                *(unsigned short*)&smem_B_fp8[my_n][kp * 2] = fp8_pair; \
            } \
            _Pragma("unroll") \
            for (int kp = 24; kp < 32; kp++) { \
                unsigned char packed = smem_Bp[(buf)][kp][my_n]; \
                float lo = smem_LUT[packed & 0xF] * sv3; \
                float hi = smem_LUT[packed >> 4]  * sv3; \
                unsigned short fp8_pair; \
                asm volatile("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" \
                             : "=h"(fp8_pair) : "f"(hi), "f"(lo)); \
                *(unsigned short*)&smem_B_fp8[my_n][kp * 2] = fp8_pair; \
            } \
        } \
    } while(0)

    // 2026-09-25: Per warp and K step: 16 N-tiles x 2 K-halves = 32 m16n8k32 E4M3 MMAs.
    #define V3_COMPUTE(a_buf) do { \
        const unsigned short* sA = (const unsigned short*)smem_A[(a_buf)]; \
        unsigned int fr0, fr1, a0, a1, a2, a3, a0b, a1b, a2b, a3b; \
        fr0 = chunk * V3_M_TILE + warp_m_offset + group_id; \
        fr1 = fr0 + 8; \
          \
        a0  = v3_bf16x4_to_e4m3x4(&sA[fr0 * a_stride + tid * 4]); \
        a1  = v3_bf16x4_to_e4m3x4(&sA[fr1 * a_stride + tid * 4]); \
        a2  = v3_bf16x4_to_e4m3x4(&sA[fr0 * a_stride + 16 + tid * 4]); \
        a3  = v3_bf16x4_to_e4m3x4(&sA[fr1 * a_stride + 16 + tid * 4]); \
          \
        a0b = v3_bf16x4_to_e4m3x4(&sA[fr0 * a_stride + 32 + tid * 4]); \
        a1b = v3_bf16x4_to_e4m3x4(&sA[fr1 * a_stride + 32 + tid * 4]); \
        a2b = v3_bf16x4_to_e4m3x4(&sA[fr0 * a_stride + 48 + tid * 4]); \
        a3b = v3_bf16x4_to_e4m3x4(&sA[fr1 * a_stride + 48 + tid * 4]); \
        _Pragma("unroll") \
        for (int nt = 0; nt < 16; nt++) { \
            unsigned int nc = nt * 8 + group_id; \
              \
            unsigned int b0 = *(const unsigned int*)&smem_B_fp8[nc][4 * tid]; \
            unsigned int b1 = *(const unsigned int*)&smem_B_fp8[nc][16 + 4 * tid]; \
            asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 " \
                "{%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%10,%11,%12,%13};" \
                :"=f"(acc[nt][0]),"=f"(acc[nt][1]),"=f"(acc[nt][2]),"=f"(acc[nt][3]) \
                :"r"(a0),"r"(a1),"r"(a2),"r"(a3),"r"(b0),"r"(b1), \
                 "f"(acc[nt][0]),"f"(acc[nt][1]),"f"(acc[nt][2]),"f"(acc[nt][3])); \
              \
            unsigned int b0b = *(const unsigned int*)&smem_B_fp8[nc][32 + 4 * tid]; \
            unsigned int b1b = *(const unsigned int*)&smem_B_fp8[nc][48 + 4 * tid]; \
            asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 " \
                "{%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%10,%11,%12,%13};" \
                :"=f"(acc[nt][0]),"=f"(acc[nt][1]),"=f"(acc[nt][2]),"=f"(acc[nt][3]) \
                :"r"(a0b),"r"(a1b),"r"(a2b),"r"(a3b),"r"(b0b),"r"(b1b), \
                 "f"(acc[nt][0]),"f"(acc[nt][1]),"f"(acc[nt][2]),"f"(acc[nt][3])); \
        } \
    } while(0)

    // 2026-09-25: The next step's loads overlap this step's MMAs; smem_B_fp8 has one buffer, so its dequant waits for them.
    V3_LOADS(0, 0);
    v3_cp_async_commit();
    v3_cp_async_wait_all();
    __syncthreads();
    V3_DEQUANT(0);
    __syncthreads();

    int cur = 0;
    for (unsigned int k_base = V3_K_STEP; k_base < K; k_base += V3_K_STEP) {
        int nxt = 1 - cur;
        V3_LOADS(nxt, k_base);
        v3_cp_async_commit();
        V3_COMPUTE(cur);
        v3_cp_async_wait_all();
        __syncthreads();
        V3_DEQUANT(nxt);
        __syncthreads();
        cur = nxt;
    }
    V3_COMPUTE(cur);

    #undef V3_LOADS
    #undef V3_DEQUANT
    #undef V3_COMPUTE

    #pragma unroll
    for (int nt = 0; nt < 16; nt++) {
        unsigned int c0 = cta_n + nt * 8 + tid * 2;
        unsigned int row_base = cta_m + chunk * V3_M_TILE + warp_m_offset;
        unsigned int r0_lo = row_base + group_id;
        unsigned int r0_hi = row_base + group_id + 8;
        if (c0 + 1 < N) {
            if (r0_lo < M) {
                __nv_bfloat16 v_lo = __float2bfloat16_rn(acc[nt][0]);
                __nv_bfloat16 v_hi = __float2bfloat16_rn(acc[nt][1]);
                C[(unsigned long long)r0_lo * N + c0]     = v_lo;
                C[(unsigned long long)r0_lo * N + c0 + 1] = v_hi;
            }
            if (r0_hi < M) {
                __nv_bfloat16 v_lo = __float2bfloat16_rn(acc[nt][2]);
                __nv_bfloat16 v_hi = __float2bfloat16_rn(acc[nt][3]);
                C[(unsigned long long)r0_hi * N + c0]     = v_lo;
                C[(unsigned long long)r0_hi * N + c0 + 1] = v_hi;
            }
        }
    }
}
