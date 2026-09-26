// SPDX-License-Identifier: MIT OR Apache-2.0

// 2026-09-25: Grouped W4A4 expert up GEMM with relu^2, moe_w4a4_grouped_gemm_relu2, on the FP4 block-scale
// MMA (kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64: E2M1 x E2M1 with an E4M3 scale per 16 values),
// the instruction w4a4_gemm_mfast in w4a4_gemm.cu also uses.
//
// Owner: gb10 kernels (nemotron-labs-3-puzzle-75b-a9b).
// Invariants: none beyond the types. Assumes K % 64 == 0: each step reads 64 values per row with no K bound,
// and the 4-byte scale copies need K / 16 bytes per row to stay 4-byte aligned.
//
// A is NVFP4 in token order ([num_tokens, K / 2] E2M1, [num_tokens, K / 16] E4M3), gathered through
// sorted_token_ids; C [total_expanded, N] is in sorted order. blockIdx.z is the expert (rows
// expert_offsets[e] to expert_offsets[e + 1]), blockIdx.y a 64-row tile, blockIdx.x a 128-column tile, 128
// threads (ops::moe_w4a4_grouped_gemm_relu2). An expert with a null B pointer writes nothing. Loads clamp rows
// to the expert's last row and columns to N - 1; the store is bounds-checked and writes relu(acc x scale2)^2.


#include <cuda_bf16.h>
#include <cstdint>

#define GW_M 64
#define GW_N 128
#define GW_K 64
#define GW_THREADS 128

__device__ __forceinline__ void gw_cpa16(void* d, const void* s) {
    unsigned x = __cvta_generic_to_shared(d);
    asm volatile("cp.async.ca.shared.global [%0],[%1],16;\n" ::"r"(x), "l"(s));
}
__device__ __forceinline__ void gw_cpa4(void* d, const void* s) {
    unsigned x = __cvta_generic_to_shared(d);
    asm volatile("cp.async.ca.shared.global [%0],[%1],4;\n" ::"r"(x), "l"(s));
}
__device__ __forceinline__ void gw_commit() { asm volatile("cp.async.commit_group;"); }
__device__ __forceinline__ void gw_wait() { asm volatile("cp.async.wait_group 0;"); }

__device__ __forceinline__ void gw_mma(float* acc,
    uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3, uint32_t b0, uint32_t b1,
    uint32_t sfa, uint32_t sfb) {
    uint16_t z = 0;
    asm volatile(
      "mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3 "
      "{%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%0,%1,%2,%3},{%10},{%11,%12},{%13},{%14,%15};\n"
      : "+f"(acc[0]), "+f"(acc[1]), "+f"(acc[2]), "+f"(acc[3])
      : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1),
        "r"(sfa), "h"(z), "h"(z), "r"(sfb), "h"(z), "h"(z));
}

extern "C" __global__ __launch_bounds__(GW_THREADS, 4)
void moe_w4a4_grouped_gemm_relu2(
    const uint8_t* __restrict__ A_packed,
    const uint8_t* __restrict__ A_sf,
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
    const unsigned expert_id = blockIdx.z;
    if (expert_id >= num_experts) return;
    const int m_start = expert_offsets[expert_id];
    const int m_end = expert_offsets[expert_id + 1];
    const int M_expert = m_end - m_start;
    if (M_expert <= 0) return;
    const int cta_m_local = blockIdx.y * GW_M;
    if (cta_m_local >= M_expert) return;
    const unsigned cta_m = (unsigned)(m_start + cta_m_local);
    const unsigned cta_n = blockIdx.x * GW_N;

    const uint8_t* Bp = (const uint8_t*)B_packed_ptrs[expert_id];
    const uint8_t* Bs = (const uint8_t*)B_scale_ptrs[expert_id];
    if (Bp == 0) return;
    const float g = scale2_vals[expert_id];

    const unsigned warp = threadIdx.x / 32, lane = threadIdx.x % 32;
    const unsigned wm = warp * 16, gid = lane >> 2, tid = lane & 3;

    __shared__ uint8_t sAf[2][GW_M][32], sBf[2][GW_N][32];
    __shared__ uint8_t sSA[2][GW_M][4],  sSB[2][GW_N][4];
    __shared__ int srow[GW_M];



    for (int r = threadIdx.x; r < GW_M; r += GW_THREADS) {
        int gm = min(cta_m_local + r, M_expert - 1);
        srow[r] = sorted_token_ids[m_start + gm];
    }
    __syncthreads();

    float acc[16][4];
    #pragma unroll
    for (int i = 0; i < 16; i++) { acc[i][0]=acc[i][1]=acc[i][2]=acc[i][3]=0; }

    const int KB = K / 2, KS = K / 16;
    const unsigned Nm1 = N - 1;

    #define GW_LOAD(buf, kb) do { \
        { unsigned idx = threadIdx.x;   \
          unsigned r = idx >> 1, c = (idx & 1u) << 4; \
          gw_cpa16(&sAf[buf][r][c], A_packed + (size_t)srow[r] * KB + (kb)/2 + c); } \
        _Pragma("unroll") \
        for (unsigned ch = 0; ch < 2u; ch++) {   \
            unsigned idx = threadIdx.x + ch * 128u; \
            unsigned r = idx >> 1, c = (idx & 1u) << 4; \
            unsigned gn = min(cta_n + r, Nm1); \
            gw_cpa16(&sBf[buf][r][c], Bp + (size_t)gn * KB + (kb)/2 + c); \
        } \
        { unsigned r = threadIdx.x; \
          if (r < GW_M) gw_cpa4(&sSA[buf][r][0], A_sf + (size_t)srow[r] * KS + (kb)/16); \
          unsigned gn = min(cta_n + r, Nm1); \
          gw_cpa4(&sSB[buf][r][0], Bs + (size_t)gn * KS + (kb)/16); } \
    } while (0)

    #define GW_COMPUTE(buf) do { \
        const unsigned fr0 = wm + gid, fr1 = fr0 + 8; \
        uint32_t a0 = *(uint32_t*)&sAf[buf][fr0][4*tid],      a1 = *(uint32_t*)&sAf[buf][fr1][4*tid]; \
        uint32_t a2 = *(uint32_t*)&sAf[buf][fr0][16 + 4*tid], a3 = *(uint32_t*)&sAf[buf][fr1][16 + 4*tid]; \
        uint32_t sfa = *(uint32_t*)&sSA[buf][wm + ((tid & 1u) << 3) + gid][0]; \
        _Pragma("unroll") \
        for (int nt = 0; nt < 16; nt++) { \
            unsigned nc = nt * 8 + gid; \
            uint32_t b0 = *(uint32_t*)&sBf[buf][nc][4*tid], b1 = *(uint32_t*)&sBf[buf][nc][16 + 4*tid]; \
            uint32_t sfb = *(uint32_t*)&sSB[buf][nc][0]; \
            gw_mma(acc[nt], a0, a1, a2, a3, b0, b1, sfa, sfb); \
        } \
    } while (0)

    GW_LOAD(0, 0); gw_commit(); gw_wait(); __syncthreads();
    int buf = 0;
    for (unsigned kb = GW_K; kb < K; kb += GW_K) {
        int nb = buf ^ 1; GW_LOAD(nb, kb); gw_commit();
        GW_COMPUTE(buf);
        gw_wait(); __syncthreads(); buf = nb;
    }
    GW_COMPUTE(buf);
    #undef GW_LOAD
    #undef GW_COMPUTE

    const unsigned fr0 = wm + gid, fr1 = fr0 + 8;
    const int mmax = M_expert - cta_m_local;
    #pragma unroll
    for (int nt = 0; nt < 16; nt++) {
        unsigned c0 = cta_n + nt * 8 + tid * 2, c1 = c0 + 1;
        float v0 = acc[nt][0] * g, v1 = acc[nt][1] * g;
        float v2 = acc[nt][2] * g, v3 = acc[nt][3] * g;

        v0 = v0 > 0.f ? v0 * v0 : 0.f;  v1 = v1 > 0.f ? v1 * v1 : 0.f;
        v2 = v2 > 0.f ? v2 * v2 : 0.f;  v3 = v3 > 0.f ? v3 * v3 : 0.f;
        if ((int)fr0 < mmax) {
            size_t row = (size_t)(cta_m + fr0) * N;
            if (c0 < N) C[row + c0] = __float2bfloat16(v0);
            if (c1 < N) C[row + c1] = __float2bfloat16(v1);
        }
        if ((int)fr1 < mmax) {
            size_t row = (size_t)(cta_m + fr1) * N;
            if (c0 < N) C[row + c0] = __float2bfloat16(v2);
            if (c1 < N) C[row + c1] = __float2bfloat16(v3);
        }
    }
}
