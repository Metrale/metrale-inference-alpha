// SPDX-License-Identifier: MIT OR Apache-2.0

// 2026-09-25: Grouped MoE expert GEMMs for qwen3.6-35b-a3b, in place of
// common/moe_w4a16_grouped_gemm.cu ([shadow] in KERNEL.toml). Weights are NVFP4 (E2M1, an
// E4M3 scale per 16 K values and a per-expert FP32 scale2), found through per-expert pointer
// tables; C rows [expert_offsets[e], expert_offsets[e + 1]) belong to expert e, and C row r
// reads A row sorted_token_ids[r] when that pointer is non-null, else A row r.
// forked-from: kernels/gb10/common/moe_w4a16_grouped_gemm.cu (2026-09-24; 2613 of 2355 lines differ, see kernels/FORKS.md)
// Owner: gb10 kernels (qwen3.6-35b-a3b, and the targets that list this file in `[sources] use`).
// Invariants: a CTA of expert e writes only C rows below expert_offsets[e + 1] and columns
// below N; an expert whose weight pointer is 0 writes nothing.

#include <cuda_bf16.h>
#include <cuda_fp8.h>

#define M_TILE 64
#define N_TILE_SM 64
#define N_TILE_LG 128
#define K_STEP 16
#define K_STEP_T 32
#define PAD 2
#define PAD_T 8  // 2026-09-25: (32 + 8) * 2 = 80-byte rows keep cp.async destinations 16-byte aligned
#define BP_PAD 16
#define GROUP_SIZE 16

__device__ __constant__ float E2M1_LUT_MOE[16] = {
    0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f,
    -0.0f, -0.5f, -1.0f, -1.5f, -2.0f, -3.0f, -4.0f, -6.0f
};

// 2026-09-25: moe_w4a16_grouped_gemm_ptrtable: B row-major per expert (packed [N, K/2], scales
// [N, K/16]) dequantized to BF16 in shared memory 16 K at a time; BF16 m16n8k16 MMAs, a 64x64
// tile, 128 threads. ops::moe_w4a16_grouped_gemm_ptrtable: grid (ceil(N/64), max_m_tiles, num_experts).
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

// 2026-09-25: moe_w4a16_grouped_gemm_ptrtable with 32 K per step and a grouped B dequant: one
// thread converts one n's 16-value group (8 packed bytes as two u32, one E4M3 scale). Same
// grid and block, so the same launcher; MoeLayer::grouped_gemm_kernel (moe/helpers_b.rs) picks
// it when moe/init.rs loaded it, which needs METRALE_MOE_GROUPED_K32=1.
// Not bit-identical to moe_w4a16_grouped_gemm_ptrtable: this kernel stages
// lut * (e4m3 * scale2), that one (lut * e4m3) * scale2, and the two can round differently
// when the E2M1 value is +-1.5, +-3 or +-6.







#define K_STEP_K32 32
extern "C" __global__ void moe_w4a16_grouped_gemm_ptrtable_k32(
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

    __shared__ __nv_bfloat16 smem_A[M_TILE][K_STEP_K32 + PAD];
    __shared__ __nv_bfloat16 smem_B[K_STEP_K32][N_TILE_SM + PAD];

    float acc[8][4];
    #pragma unroll
    for (int i = 0; i < 8; i++) {
        acc[i][0] = 0.0f; acc[i][1] = 0.0f;
        acc[i][2] = 0.0f; acc[i][3] = 0.0f;
    }

    const unsigned int a_stride = K_STEP_K32 + PAD;
    const unsigned int b_stride = N_TILE_SM + PAD;
    const unsigned int M_eff = (unsigned int)M_expert;
    const unsigned int half_K = K / 2;
    const unsigned int num_groups = K / GROUP_SIZE;

    for (unsigned int k_base = 0; k_base < K; k_base += K_STEP_K32) {
        {
            const unsigned int ept = (M_TILE * K_STEP_K32) / 128;
            #pragma unroll
            for (unsigned int i = 0; i < ept; i++) {
                unsigned int idx = threadIdx.x * ept + i;
                unsigned int row = idx / K_STEP_K32;
                unsigned int col = idx % K_STEP_K32;
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
            // 2026-09-25: One thread converts one n's 16-value group: 8 packed bytes as two
            // u32 and one E4M3 scale for all 16 values.






            const unsigned int gpt = ((K_STEP_K32 / GROUP_SIZE) * N_TILE_SM) / 128;
            #pragma unroll
            for (unsigned int i = 0; i < gpt; i++) {
                unsigned int g = threadIdx.x * gpt + i;
                unsigned int kg = (g / N_TILE_SM) * GROUP_SIZE;
                unsigned int n = g % N_TILE_SM;
                unsigned int gk = k_base + kg;
                unsigned int gn = cta_n + n;
                if (gk < K && gn < N) {
                    const unsigned char* bp = B_expert + (unsigned long long)gn * half_K + (gk / 2);
                    unsigned char sb = S_expert[(unsigned long long)gn * num_groups + (gk / GROUP_SIZE)];
                    __nv_fp8_e4m3 fp8; *(unsigned char*)&fp8 = sb;
                    float sc = (float)fp8 * scale2;
                    unsigned int w0 = *(const unsigned int*)(bp);
                    unsigned int w1 = *(const unsigned int*)(bp + 4);
                    #pragma unroll
                    for (int j = 0; j < 4; j++) {
                        unsigned char c0 = (unsigned char)((w0 >> (j * 8)) & 0xFF);
                        unsigned char c1 = (unsigned char)((w1 >> (j * 8)) & 0xFF);
                        smem_B[kg + j * 2][n]         = __float2bfloat16(E2M1_LUT_MOE[c0 & 0xF] * sc);
                        smem_B[kg + j * 2 + 1][n]     = __float2bfloat16(E2M1_LUT_MOE[c0 >> 4] * sc);
                        smem_B[kg + 8 + j * 2][n]     = __float2bfloat16(E2M1_LUT_MOE[c1 & 0xF] * sc);
                        smem_B[kg + 8 + j * 2 + 1][n] = __float2bfloat16(E2M1_LUT_MOE[c1 >> 4] * sc);
                    }
                } else {
                    #pragma unroll
                    for (int j = 0; j < GROUP_SIZE; j++) smem_B[kg + j][n] = __float2bfloat16(0.0f);
                }
            }
        }
        __syncthreads();

        const unsigned short* sA = (const unsigned short*)smem_A;
        const unsigned short* sB = (const unsigned short*)smem_B;
        unsigned int fr0 = warp_m_offset + group_id;
        unsigned int fr1 = fr0 + 8;
        // 2026-09-25: The two 16-wide MMA fragments are consumed in K order, the accumulation
        // order of moe_w4a16_grouped_gemm_ptrtable.

        #pragma unroll
        for (unsigned int kf = 0; kf < K_STEP_K32; kf += 16) {
        unsigned int fc0 = kf + tid * 2, fc1 = fc0 + 8;
        unsigned int a0 = ((unsigned int)sA[fr0*a_stride+fc0+1]<<16) | (unsigned int)sA[fr0*a_stride+fc0];
        unsigned int a1 = ((unsigned int)sA[fr1*a_stride+fc0+1]<<16) | (unsigned int)sA[fr1*a_stride+fc0];
        unsigned int a2 = ((unsigned int)sA[fr0*a_stride+fc1+1]<<16) | (unsigned int)sA[fr0*a_stride+fc1];
        unsigned int a3 = ((unsigned int)sA[fr1*a_stride+fc1+1]<<16) | (unsigned int)sA[fr1*a_stride+fc1];

        #pragma unroll
        for (int nt = 0; nt < 8; nt++) {
            unsigned int nc = nt * 8 + group_id;
            unsigned int k0 = kf + tid * 2, k1 = k0 + 8;
            unsigned int b0 = ((unsigned int)sB[(k0+1)*b_stride+nc]<<16) | (unsigned int)sB[k0*b_stride+nc];
            unsigned int b1 = ((unsigned int)sB[(k1+1)*b_stride+nc]<<16) | (unsigned int)sB[k1*b_stride+nc];
            asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%10,%11,%12,%13};"
                :"=f"(acc[nt][0]),"=f"(acc[nt][1]),"=f"(acc[nt][2]),"=f"(acc[nt][3])
                :"r"(a0),"r"(a1),"r"(a2),"r"(a3),"r"(b0),"r"(b1),
                 "f"(acc[nt][0]),"f"(acc[nt][1]),"f"(acc[nt][2]),"f"(acc[nt][3]));
        }
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


// 2026-09-25: Helpers for the E4M3-MMA kernels (moe_w4a16_grouped_gemm_ptrtable_t after the m256
// kernel, _t_k64, the fused gate/up kernels and the FP8-input kernel): a predicated 16-byte
// cp.async (src-size 0 when the predicate is false), commit/wait, and a BF16x4 -> E4M3x4
// register conversion. Those kernels read B K-major (packed [K/2, N], scales [K/16, N]),
// dequantize it to E4M3 in shared memory and issue m16n8k32 E4M3 MMAs; A is converted to
// E4M3 in registers (the FP8-input kernel reads it as E4M3 already).
// moe_w4a16_grouped_gemm_ptrtable_t: 64x128 tile, 32 K per step, two cp.async stages, 128
// threads. Shared memory: A 2*64*40*2 = 10,240 B; Bp 2*16*144 = 4,608 B; Bs 2*2*144 = 576 B;
// B_fp8 128*36 = 4,608 B; LUT 64 B; tok 256 B; 20,352 B in all.







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

// 2026-09-25: Convert 4 BF16 values from shared memory to 4 E4M3 values packed in a u32.
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
    asm volatile("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(h0) : "f"(f1), "f"(f0));
    asm volatile("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(h1) : "f"(f3), "f"(f2));
    return ((unsigned int)h1 << 16) | (unsigned int)h0;
}


// 2026-09-25: moe_w4a16_grouped_gemm_ptrtable_k32 with a 256-row M tile: 512 threads, 16 warps of
// 16 rows each, so an expert's weights are read once per 256 rows rather than once per 64.
// Both loads are block-strided loops (the B group count, (32 / 16) * 64 = 128, is smaller
// than the block). moe/init.rs loads it only under METRALE_MOE_GROUPED_M256=1, and
// MoeLayer::launch_grouped_gemm (moe/helpers_b.rs) then launches it with ceil(max_m_tiles / 4)
// and holds the measurements. Shared memory: A 256*34*2 + B 32*66*2 = 21,632 B.




















#define M_TILE_M256 256
#define K_STEP_M256 32
extern "C" __global__ __launch_bounds__(512) void moe_w4a16_grouped_gemm_ptrtable_m256(
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
    const int cta_m_local = blockIdx.y * M_TILE_M256;
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

    __shared__ __nv_bfloat16 smem_A[M_TILE_M256][K_STEP_M256 + PAD];
    __shared__ __nv_bfloat16 smem_B[K_STEP_M256][N_TILE_SM + PAD];

    float acc[8][4];
    #pragma unroll
    for (int i = 0; i < 8; i++) { acc[i][0]=0.f; acc[i][1]=0.f; acc[i][2]=0.f; acc[i][3]=0.f; }

    const unsigned int a_stride = K_STEP_M256 + PAD;
    const unsigned int b_stride = N_TILE_SM + PAD;
    const unsigned int M_eff = (unsigned int)M_expert;
    const unsigned int half_K = K / 2;
    const unsigned int num_groups = K / GROUP_SIZE;

    for (unsigned int k_base = 0; k_base < K; k_base += K_STEP_M256) {
        for (unsigned int idx = threadIdx.x; idx < M_TILE_M256 * K_STEP_M256; idx += 512) {
            unsigned int row = idx / K_STEP_M256, col = idx % K_STEP_M256;
            unsigned int gc = k_base + col;
            bool valid = (cta_m_local + row) < M_eff && gc < K;
            if (valid) {
                unsigned int a_row = sorted_token_ids
                    ? (unsigned int)sorted_token_ids[cta_m + row] : (cta_m + row);
                smem_A[row][col] = A[(size_t)a_row * K + gc];
            } else {
                smem_A[row][col] = __float2bfloat16(0.0f);
            }
        }
        // 2026-09-25: One thread per 16-value group: 8 packed bytes as two u32, one E4M3
        // scale for all 16 values.
        for (unsigned int g = threadIdx.x;
             g < (K_STEP_M256 / GROUP_SIZE) * N_TILE_SM; g += 512) {
            unsigned int kg = (g / N_TILE_SM) * GROUP_SIZE, n = g % N_TILE_SM;
            unsigned int gk = k_base + kg, gn = cta_n + n;
            if (gk < K && gn < N) {
                const unsigned char* bp = B_expert + (unsigned long long)gn * half_K + (gk / 2);
                unsigned char sb = S_expert[(unsigned long long)gn * num_groups + (gk / GROUP_SIZE)];
                __nv_fp8_e4m3 fp8; *(unsigned char*)&fp8 = sb;
                float sc = (float)fp8 * scale2;
                unsigned int w0 = *(const unsigned int*)(bp);
                unsigned int w1 = *(const unsigned int*)(bp + 4);
                #pragma unroll
                for (int j = 0; j < 4; j++) {
                    unsigned char c0 = (unsigned char)((w0 >> (j * 8)) & 0xFF);
                    unsigned char c1 = (unsigned char)((w1 >> (j * 8)) & 0xFF);
                    smem_B[kg + j * 2][n]         = __float2bfloat16(E2M1_LUT_MOE[c0 & 0xF] * sc);
                    smem_B[kg + j * 2 + 1][n]     = __float2bfloat16(E2M1_LUT_MOE[c0 >> 4] * sc);
                    smem_B[kg + 8 + j * 2][n]     = __float2bfloat16(E2M1_LUT_MOE[c1 & 0xF] * sc);
                    smem_B[kg + 8 + j * 2 + 1][n] = __float2bfloat16(E2M1_LUT_MOE[c1 >> 4] * sc);
                }
            } else {
                #pragma unroll
                for (int j = 0; j < GROUP_SIZE; j++) smem_B[kg + j][n] = __float2bfloat16(0.0f);
            }
        }
        __syncthreads();

        const unsigned short* sA = (const unsigned short*)smem_A;
        const unsigned short* sB = (const unsigned short*)smem_B;
        unsigned int fr0 = warp_m_offset + group_id, fr1 = fr0 + 8;
        #pragma unroll
        for (unsigned int kf = 0; kf < K_STEP_M256; kf += 16) {
            unsigned int fc0 = kf + tid * 2, fc1 = fc0 + 8;
            unsigned int a0 = ((unsigned int)sA[fr0*a_stride+fc0+1]<<16) | (unsigned int)sA[fr0*a_stride+fc0];
            unsigned int a1 = ((unsigned int)sA[fr1*a_stride+fc0+1]<<16) | (unsigned int)sA[fr1*a_stride+fc0];
            unsigned int a2 = ((unsigned int)sA[fr0*a_stride+fc1+1]<<16) | (unsigned int)sA[fr0*a_stride+fc1];
            unsigned int a3 = ((unsigned int)sA[fr1*a_stride+fc1+1]<<16) | (unsigned int)sA[fr1*a_stride+fc1];
            #pragma unroll
            for (int nt = 0; nt < 8; nt++) {
                unsigned int nc = nt * 8 + group_id;
                unsigned int k0 = kf + tid * 2, k1 = k0 + 8;
                unsigned int b0 = ((unsigned int)sB[(k0+1)*b_stride+nc]<<16) | (unsigned int)sB[k0*b_stride+nc];
                unsigned int b1 = ((unsigned int)sB[(k1+1)*b_stride+nc]<<16) | (unsigned int)sB[k1*b_stride+nc];
                asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%10,%11,%12,%13};"
                    :"=f"(acc[nt][0]),"=f"(acc[nt][1]),"=f"(acc[nt][2]),"=f"(acc[nt][3])
                    :"r"(a0),"r"(a1),"r"(a2),"r"(a3),"r"(b0),"r"(b1),
                     "f"(acc[nt][0]),"f"(acc[nt][1]),"f"(acc[nt][2]),"f"(acc[nt][3]));
            }
        }
        __syncthreads();
    }

    #pragma unroll
    for (int nt = 0; nt < 8; nt++) {
        unsigned int base_n = cta_n + nt * 8;
        unsigned int c0 = base_n + (tid * 2), c1 = c0 + 1;
        unsigned int r0 = cta_m + warp_m_offset + group_id, r1 = r0 + 8;
        if (r0 < (unsigned int)m_end && c0 < N) C[(size_t)r0*N + c0] = __float2bfloat16(acc[nt][0]);
        if (r0 < (unsigned int)m_end && c1 < N) C[(size_t)r0*N + c1] = __float2bfloat16(acc[nt][1]);
        if (r1 < (unsigned int)m_end && c0 < N) C[(size_t)r1*N + c0] = __float2bfloat16(acc[nt][2]);
        if (r1 < (unsigned int)m_end && c1 < N) C[(size_t)r1*N + c1] = __float2bfloat16(acc[nt][3]);
    }
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
    // 2026-09-25: +4 pad: 36-byte rows (9 words, gcd(9, 32) = 1). Dequant writes, bank
    // (t * 9 + kp / 2) % 32, are conflict-free; MMA reads, bank (nc * 9 + tid) % 32, have
    // 3 conflicting pairs, against 16 with 32-byte rows.
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

    // 2026-09-25: Dequant B: FP4 -> FP8 E4M3.
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

    // 2026-09-25: Convert A BF16 -> E4M3 in registers; one m16n8k32 MMA per N tile.
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
            asm volatile( \
                "mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 " \
                "{%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%10,%11,%12,%13};" \
                :"=f"(acc[nt][0]),"=f"(acc[nt][1]), \
                 "=f"(acc[nt][2]),"=f"(acc[nt][3]) \
                :"r"(a0),"r"(a1),"r"(a2),"r"(a3), \
                 "r"(b0),"r"(b1), \
                 "f"(acc[nt][0]),"f"(acc[nt][1]), \
                 "f"(acc[nt][2]),"f"(acc[nt][3])); \
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

// 2026-09-25: moe_w4a16_grouped_gemm_ptrtable_t with 64 K per step: two m16n8k32 MMAs per N tile
// and step, four scale groups per step. forward_prefill_routed.rs launches it for the routed
// down projection through ops::moe_w4a16_grouped_gemm_ptrtable_n128, with sorted_token_ids
// null: grid (ceil(N/128), max_m_tiles, num_experts), block 128.
// Shared memory: A 2*64*72*2 = 18,432 B; Bp 2*32*144 = 9,216 B; Bs 2*4*144 = 1,152 B;
// B_fp8 128*80 = 10,240 B; LUT 64 B; tok 256 B; 39,360 B in all.













#define K_STEP_T64 64
#define PAD_T64 8  // 2026-09-25: (64 + 8) * 2 = 144-byte rows keep cp.async destinations 16-byte aligned

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

    // 2026-09-25: B_fp8 rows are 80 bytes (64 + 16): the MMA's B-fragment reads (rows nc = 0..7 of
    // a group, word tid) hit 32 distinct banks, as nc * 20 mod 32 is distinct for nc = 0..7;
    // 64-byte rows would give 4-way conflicts.
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

    // 2026-09-25: A: 4 rounds x 128 threads x 16 bytes = 8192 B = 64x64 BF16
    // Bp: 2 rounds x 128 threads x 16 bytes = 4096 B = 32x128 packed bytes
    // Bs: 4 scale groups x 8 threads x 16 bytes = 512 B per buffer
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

    // 2026-09-25: Dequant B: FP4 -> FP8 E4M3, four scale groups per 64-K step.
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

    // 2026-09-25: Two m16n8k32 MMAs per N tile: all N tiles for k 0..31 with a0..a3, then all
    // for k 32..63 with a4..a7.

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
            asm volatile( \
                "mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 " \
                "{%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%10,%11,%12,%13};" \
                :"=f"(acc[nt][0]),"=f"(acc[nt][1]),"=f"(acc[nt][2]),"=f"(acc[nt][3]) \
                :"r"(a0),"r"(a1),"r"(a2),"r"(a3),"r"(b0),"r"(b1), \
                 "f"(acc[nt][0]),"f"(acc[nt][1]),"f"(acc[nt][2]),"f"(acc[nt][3])); \
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
            asm volatile( \
                "mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 " \
                "{%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%10,%11,%12,%13};" \
                :"=f"(acc[nt][0]),"=f"(acc[nt][1]),"=f"(acc[nt][2]),"=f"(acc[nt][3]) \
                :"r"(a4),"r"(a5),"r"(a6),"r"(a7),"r"(b0),"r"(b1), \
                 "f"(acc[nt][0]),"f"(acc[nt][1]),"f"(acc[nt][2]),"f"(acc[nt][3])); \
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

// 2026-09-25: moe_w4a16_fused_gate_up_t on the K64 pipeline of moe_w4a16_grouped_gemm_ptrtable_t_k64.
// forward_prefill_routed.rs launches it for the routed gate/up projections through
// ops::moe_w4a16_fused_gate_up_k64_n128: grid (ceil(2N/128), max_m_tiles, num_experts), block 128.



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
    // 2026-09-25: 80-byte B_fp8 rows, as in moe_w4a16_grouped_gemm_ptrtable_t_k64.
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
            asm volatile("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" \
                         : "=h"(fp8_pair) : "f"(hi), "f"(lo)); \
            *(unsigned short*)&smem_B_fp8_fgu64[my_n][kp * 2] = fp8_pair; \
        } \
        _Pragma("unroll") \
        for (int kp = 8; kp < 16; kp++) { \
            unsigned char packed = smem_Bp_fgu64[(buf)][kp][my_n]; \
            float lo = smem_LUT_fgu64[packed & 0xF] * sv1; \
            float hi = smem_LUT_fgu64[packed >> 4] * sv1; \
            unsigned short fp8_pair; \
            asm volatile("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" \
                         : "=h"(fp8_pair) : "f"(hi), "f"(lo)); \
            *(unsigned short*)&smem_B_fp8_fgu64[my_n][kp * 2] = fp8_pair; \
        } \
        _Pragma("unroll") \
        for (int kp = 16; kp < 24; kp++) { \
            unsigned char packed = smem_Bp_fgu64[(buf)][kp][my_n]; \
            float lo = smem_LUT_fgu64[packed & 0xF] * sv2; \
            float hi = smem_LUT_fgu64[packed >> 4] * sv2; \
            unsigned short fp8_pair; \
            asm volatile("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" \
                         : "=h"(fp8_pair) : "f"(hi), "f"(lo)); \
            *(unsigned short*)&smem_B_fp8_fgu64[my_n][kp * 2] = fp8_pair; \
        } \
        _Pragma("unroll") \
        for (int kp = 24; kp < 32; kp++) { \
            unsigned char packed = smem_Bp_fgu64[(buf)][kp][my_n]; \
            float lo = smem_LUT_fgu64[packed & 0xF] * sv3; \
            float hi = smem_LUT_fgu64[packed >> 4] * sv3; \
            unsigned short fp8_pair; \
            asm volatile("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" \
                         : "=h"(fp8_pair) : "f"(hi), "f"(lo)); \
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
            asm volatile( \
                "mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 " \
                "{%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%10,%11,%12,%13};" \
                :"=f"(acc[nt][0]),"=f"(acc[nt][1]),"=f"(acc[nt][2]),"=f"(acc[nt][3]) \
                :"r"(a0),"r"(a1),"r"(a2),"r"(a3),"r"(b0),"r"(b1), \
                 "f"(acc[nt][0]),"f"(acc[nt][1]),"f"(acc[nt][2]),"f"(acc[nt][3])); \
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
            asm volatile( \
                "mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 " \
                "{%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%10,%11,%12,%13};" \
                :"=f"(acc[nt][0]),"=f"(acc[nt][1]),"=f"(acc[nt][2]),"=f"(acc[nt][3]) \
                :"r"(a4),"r"(a5),"r"(a6),"r"(a7),"r"(b0),"r"(b1), \
                 "f"(acc[nt][0]),"f"(acc[nt][1]),"f"(acc[nt][2]),"f"(acc[nt][3])); \
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

// 2026-09-25: Fused gate+up: grid x spans 2N. A CTA whose first column blockIdx.x * 128 is below N
// uses the gate weights and writes C_gate, the others the up weights and C_up at column
// blockIdx.x * 128 - N. Grid (ceil(2N/128), max_m_tiles, num_experts), block 128.







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
    unsigned int N,  // 2026-09-25: columns per projection; gate and up have the same N
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
    // 2026-09-25: The +4 pad of moe_w4a16_grouped_gemm_ptrtable_t (conflict-free dequant writes).
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

    // 2026-09-25: The three macros below are copies of moe_w4a16_grouped_gemm_ptrtable_t's.
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
            asm volatile( \
                "mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 " \
                "{%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%10,%11,%12,%13};" \
                :"=f"(acc[nt][0]),"=f"(acc[nt][1]), \
                 "=f"(acc[nt][2]),"=f"(acc[nt][3]) \
                :"r"(a0),"r"(a1),"r"(a2),"r"(a3), \
                 "r"(b0),"r"(b1), \
                 "f"(acc[nt][0]),"f"(acc[nt][1]), \
                 "f"(acc[nt][2]),"f"(acc[nt][3])); \
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

// 2026-09-25: moe_fp8_grouped_gemm_ptrtable_t: moe_w4a16_grouped_gemm_ptrtable_t with A already in
// E4M3 ([rows, K] bytes), so A needs no conversion and its shared tile is 64 x 32 B per
// buffer. forward_prefill_routed.rs uses it for the routed down projection when
// ctx.levers.moe_prefill_fp8_down is set, after ops::bf16_to_fp8 converts the input.
// Grid (ceil(N/128), max_m_tiles, num_experts), block 128 (ops::moe_fp8_grouped_gemm_ptrtable_n128).




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
            asm volatile("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" \
                         : "=h"(fp8_pair) : "f"(hi), "f"(lo)); \
            *(unsigned short*)&smem_B2_fp8[my_n][kp * 2] = fp8_pair; \
        } \
        _Pragma("unroll") \
        for (int kp = 8; kp < 16; kp++) { \
            unsigned char packed = smem_Bp2[(buf)][kp][my_n]; \
            float lo = smem_LUT2[packed & 0xF] * sv1; \
            float hi = smem_LUT2[packed >> 4] * sv1; \
            unsigned short fp8_pair; \
            asm volatile("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" \
                         : "=h"(fp8_pair) : "f"(hi), "f"(lo)); \
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
            asm volatile( \
                "mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 " \
                "{%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%10,%11,%12,%13};" \
                :"=f"(acc[nt][0]),"=f"(acc[nt][1]), \
                 "=f"(acc[nt][2]),"=f"(acc[nt][3]) \
                :"r"(a0),"r"(a1),"r"(a2),"r"(a3), \
                 "r"(b0),"r"(b1), \
                 "f"(acc[nt][0]),"f"(acc[nt][1]), \
                 "f"(acc[nt][2]),"f"(acc[nt][3])); \
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

// 2026-09-25: W4A4 kernels: FP4 weights and FP4 activations. Everything from the #ifndef below to
// the end of the file is compiled out where METRALE_NO_WARP_BLOCKSCALE_MMA is defined.
// crates/kernels/tests/blockscale_mma_guard.rs fails when a line naming the block-scaled
// instructions, comments included, sits outside that guard, so they are named only below it.




#ifndef METRALE_NO_WARP_BLOCKSCALE_MMA
// 2026-09-25: The two instructions are cvt.rn.satfinite.e2m1x2 and mma.sync kind::mxf4nvf4
// block_scale. The hopper, b200 and b300 HARDWARE.toml define METRALE_NO_WARP_BLOCKSCALE_MMA
// in [build] extra_nvcc_flags, because ptxas rejects them there (blockscale_mma_guard.rs
// quotes the errors); gb10 does not define it. Without this region the entry points
// moe_w4a16_fused_gate_up_t_k64_fp4 and moe_w4a16_down_t_k64_fp4 do not exist, and the
// hopper and b200 qwen3.6-35b-a3b MODEL.toml list them in [expected_absent.moe_w4a16].
// moe/init.rs resolves both with try_kernel; forward_prefill_routed.rs launches them only
// when the handle is nonzero and METRALE_HOLO_MOE_GATEUP_FP4 / METRALE_HOLO_MOE_DOWN_FP4 is
// set (both off by default).

















// 2026-09-25: moe_w4a16_fused_gate_up_t_k64_fp4: moe_w4a16_fused_gate_up_t_k64 with one
// block-scaled FP4 MMA (mma.sync kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64) per
// 64-K step in place of two m16n8k32 E4M3 MMAs; the same cp.async double buffering and
// gate/up split.
// - B stays E2M1 with its UE4M3 per-16 scales, read from the transposed tables
//   gate_ptrs_t / up_ptrs_t that the E4M3 fused kernel also reads: packed [K/2, N] and
//   scales [K/16, N]. The packed bytes are loaded K-major into smem_BpT and copied
//   N-major into smem_Bp by FP4_TRANSPOSE before the MMA.
// - A (BF16) is quantized per 16 values: scale = max_abs / 6 stored as UE4M3 (the rule of
//   metrale_cutlass_pack_bf16_act_nvfp4 in crates/gpu-runtime/cuda/cutlass_nvfp4_gemm.cu),
//   then cvt.rn.satfinite.e2m1x2 of value / scale.
// - The MMA applies the per-16 scales; the epilogue multiplies by the expert's scale2.
// Grid (ceil(2N/128), max_m_tiles, num_experts), block 128 (ops::moe_w4a16_fused_gate_up_k64_n128).











// 2026-09-25: E2M1 by magnitude thresholds, as float_to_e2m1 in cutlass_nvfp4_gemm.cu; not called here.
__device__ __forceinline__ unsigned char fp4_float_to_e2m1(float x) {
    unsigned char sign = (x < 0.0f) ? 8u : 0u;
    float ax = fabsf(x);
    unsigned char mag;
    if (ax <= 0.25f)      mag = 0;
    else if (ax <= 0.75f) mag = 1;
    else if (ax <= 1.25f) mag = 2;
    else if (ax <= 1.75f) mag = 3;
    else if (ax <= 2.5f)  mag = 4;
    else if (ax <= 3.5f)  mag = 5;
    else if (ax <= 5.0f)  mag = 6;
    else                  mag = 7;
    return sign | mag;
}

extern "C" __global__ void moe_w4a16_fused_gate_up_t_k64_fp4(
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

    // 2026-09-25: Double-buffered cp.async targets: BF16 A, packed B, UE4M3 B scales.

    __shared__ __nv_bfloat16 smem_A_fp4[2][M_TILE][K_STEP_T64 + PAD_T64];
    // 2026-09-25: B staging: [K_STEP_T64/2][N_TILE_LG] K-major, double-buffered; the cp.async
    // target for the [K/2, N] tables.

    __shared__ unsigned char smem_BpT[2][K_STEP_T64 / 2][N_TILE_LG + BP_PAD];
    // 2026-09-25: MMA-ready B: [N_TILE_LG][K_STEP_T64/2 + 16] N-major, single-buffered, rebuilt
    // each step from the staging tile by FP4_TRANSPOSE.

    __shared__ unsigned char smem_Bp[N_TILE_LG][K_STEP_T64 / 2 + 16];
    // 2026-09-25: B scales: [K_STEP_T64/16][N_TILE_LG], group-major like the [K/16, N] table.

    __shared__ unsigned char smem_Bs_fp4[2][K_STEP_T64 / GROUP_SIZE][N_TILE_LG + BP_PAD];
    // 2026-09-25: A quantized to FP4: packed [M_TILE][K_STEP_T64/2] + scales [M_TILE][4].
    __shared__ unsigned char smem_Ap_fp4[M_TILE][K_STEP_T64 / 2 + 4];
    __shared__ unsigned char smem_As_fp4[M_TILE][K_STEP_T64 / GROUP_SIZE];
    __shared__ int smem_tok_fp4[M_TILE];

    if (threadIdx.x < M_TILE) {
        int local_row = threadIdx.x;
        if (sorted_token_ids && (cta_m_local + local_row) < (unsigned int)M_expert)
            smem_tok_fp4[local_row] = sorted_token_ids[cta_m + local_row];
        else
            smem_tok_fp4[local_row] = (int)(cta_m + local_row);
    }
    __syncthreads();

    float acc[16][4];
    #pragma unroll
    for (int i = 0; i < 16; i++) {
        acc[i][0] = 0.0f; acc[i][1] = 0.0f;
        acc[i][2] = 0.0f; acc[i][3] = 0.0f;
    }

    const unsigned int M_eff = (unsigned int)M_expert;
    const unsigned int num_groups = K / GROUP_SIZE;

    // 2026-09-25: Loads into the double-buffered shared tiles:
    // A: 4 rounds x 128 threads x 16 B = 64x64 BF16.
    // Bp: K-major rows of the [K/2, N] table into smem_BpT, 16 N-bytes per thread per round.
    // Bs: 4 groups x 128 N, one byte per load.

    #define FP4_ISSUE_LOADS(buf, kb) do { \
        { \
            unsigned int a_row_base = threadIdx.x >> 3; \
            unsigned int a_col = (threadIdx.x & 7) << 3; \
            unsigned int gc = (kb) + a_col; \
            _Pragma("unroll") \
            for (int rnd = 0; rnd < 4; rnd++) { \
                unsigned int row = rnd * 16 + a_row_base; \
                bool valid = (cta_m_local + row) < M_eff && (gc + 7 < K); \
                unsigned int a_row = (unsigned int)smem_tok_fp4[row]; \
                moe_cp_async_pred_16(&smem_A_fp4[(buf)][row][a_col], \
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
                moe_cp_async_pred_16(&smem_BpT[(buf)][kp_cur][ns], \
                    &B_expert[(unsigned long long)(gke >> 1) * N + gns], \
                    (gke + 1 < K) && (gns + 15 < N)); \
            } \
        } \
        { \
            unsigned int g = threadIdx.x >> 5; \
            unsigned int nn = threadIdx.x & 31; \
            unsigned int sg = (kb) / GROUP_SIZE + g; \
            _Pragma("unroll") \
            for (int rnd = 0; rnd < 4; rnd++) { \
                unsigned int n_cur = rnd * 32 + nn; \
                unsigned int gns = cta_n + n_cur; \
                bool sv = (gns < N) && (sg < num_groups); \
                smem_Bs_fp4[(buf)][g][n_cur] = sv ? \
                    S_expert[(unsigned long long)sg * N + gns] : 0; \
            } \
        } \
    } while(0)

    // 2026-09-25: Quantize A (BF16 -> E2M1 + UE4M3) per 16-value group: 64 rows x 4 groups = 256
    // jobs, two per thread. Each group takes eight cvt.rn.satfinite.e2m1x2.f32 and two u32
    // stores; in each byte the low nibble is v[i] and the high nibble v[i + 1].


    #define FP4_QUANT_A(buf) do { \
        _Pragma("unroll") \
        for (int job = 0; job < 2; job++) { \
            unsigned int jid = threadIdx.x + job * 128; \
            unsigned int row = jid >> 2; \
            unsigned int grp = jid & 3; \
            const __nv_bfloat16* arow = &smem_A_fp4[(buf)][row][grp * GROUP_SIZE]; \
            float max_abs = 0.0f; \
            float vals[16]; \
            _Pragma("unroll") \
            for (int i = 0; i < 16; i++) { \
                vals[i] = __bfloat162float(arow[i]); \
                max_abs = fmaxf(max_abs, fabsf(vals[i])); \
            } \
            float sc = max_abs > 0.0f ? max_abs * (1.0f / 6.0f) : 1.0f; \
            __nv_fp8_e4m3 sf8(sc); \
            unsigned char sfb = *(unsigned char*)&sf8; \
            smem_As_fp4[row][grp] = sfb; \
            float dec = (float)sf8; \
            float inv = dec > 0.0f ? 1.0f / dec : 0.0f; \
            _Pragma("unroll") \
            for (int i = 0; i < 16; i++) vals[i] *= inv; \
            unsigned int* dst = (unsigned int*)&smem_Ap_fp4[row][grp * (GROUP_SIZE / 2)]; \
            _Pragma("unroll") \
            for (int half = 0; half < 2; half++) { \
                unsigned int out; \
                const float* s = &vals[half * 8]; \
                asm volatile( \
                    "{\n" \
                    ".reg .b8 b0; .reg .b8 b1; .reg .b8 b2; .reg .b8 b3;\n" \
                    "cvt.rn.satfinite.e2m1x2.f32 b0, %2, %1;\n" \
                    "cvt.rn.satfinite.e2m1x2.f32 b1, %4, %3;\n" \
                    "cvt.rn.satfinite.e2m1x2.f32 b2, %6, %5;\n" \
                    "cvt.rn.satfinite.e2m1x2.f32 b3, %8, %7;\n" \
                    "mov.b32 %0, {b0, b1, b2, b3};\n" \
                    "}" \
                    : "=r"(out) \
                    : "f"(s[0]), "f"(s[1]), "f"(s[2]), "f"(s[3]), \
                      "f"(s[4]), "f"(s[5]), "f"(s[6]), "f"(s[7])); \
                dst[half] = out; \
            } \
        } \
    } while(0)

    // 2026-09-25: FP4_FRAG reads 8 consecutive E2M1 values (4 packed bytes) as one u32. KK is
    // even and KK / 2 is tid * 4 or 16 + tid * 4, so the read is 4-byte aligned.


    #define FP4_FRAG(P, ROW, KK) (*(const unsigned int*)&(P)[(ROW)][(KK) / 2])

    // 2026-09-25: Staging (K-major) -> MMA-ready tile (N-major): each of the 128 lanes owns one N
    // row and copies its 32 packed bytes into a contiguous run with u32 stores. A pure
    // byte copy: every byte keeps its two E2M1 values. The reads of one warp are
    // consecutive bytes of one staging row.



    #define FP4_TRANSPOSE(buf) do { \
        unsigned int my_n = threadIdx.x; \
        if (my_n < N_TILE_LG) { \
            _Pragma("unroll") \
            for (int q = 0; q < (K_STEP_T64 / 2) / 4; q++) { \
                unsigned int w = (unsigned int)smem_BpT[(buf)][q * 4 + 0][my_n] \
                    | ((unsigned int)smem_BpT[(buf)][q * 4 + 1][my_n] << 8) \
                    | ((unsigned int)smem_BpT[(buf)][q * 4 + 2][my_n] << 16) \
                    | ((unsigned int)smem_BpT[(buf)][q * 4 + 3][my_n] << 24); \
                *(unsigned int*)&smem_Bp[my_n][q * 4] = w; \
            } \
        } \
    } while(0)

    // 2026-09-25: One m16n8k64 block-scaled FP4 MMA per N tile: a 64-K step is one MMA K tile.

    #define FP4_COMPUTE_MMA() do { \
        unsigned int ra = warp_m_offset + group_id; \
        unsigned int a0 = FP4_FRAG(smem_Ap_fp4, ra,     tid * 8); \
        unsigned int a1 = FP4_FRAG(smem_Ap_fp4, ra + 8, tid * 8); \
        unsigned int a2 = FP4_FRAG(smem_Ap_fp4, ra,     32 + tid * 8); \
        unsigned int a3 = FP4_FRAG(smem_Ap_fp4, ra + 8, 32 + tid * 8); \
        unsigned int sfa_m = (lane_id & 1) * 8 + (lane_id >> 2); \
        unsigned int sfa = (unsigned int)smem_As_fp4[warp_m_offset + sfa_m][0] \
                         | ((unsigned int)smem_As_fp4[warp_m_offset + sfa_m][1] << 8) \
                         | ((unsigned int)smem_As_fp4[warp_m_offset + sfa_m][2] << 16) \
                         | ((unsigned int)smem_As_fp4[warp_m_offset + sfa_m][3] << 24); \
        _Pragma("unroll") \
        for (int nt = 0; nt < 16; nt++) { \
            unsigned int nc = nt * 8 + group_id; \
            unsigned int b0 = FP4_FRAG(smem_Bp, nc, tid * 8); \
            unsigned int b1 = FP4_FRAG(smem_Bp, nc, 32 + tid * 8); \
            unsigned int sfn = nt * 8 + (lane_id >> 2); \
            unsigned int sfb_raw = (unsigned int)smem_Bs_cur[0][sfn] \
                         | ((unsigned int)smem_Bs_cur[1][sfn] << 8) \
                         | ((unsigned int)smem_Bs_cur[2][sfn] << 16) \
                         | ((unsigned int)smem_Bs_cur[3][sfn] << 24); \
            unsigned short bidA = 0, tidA_ = 0, bidB = 0, tidB_ = 0; \
            asm volatile( \
                "mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3 " \
                "{%0,%1,%2,%3}," \
                "{%4,%5,%6,%7}," \
                "{%8,%9}," \
                "{%10,%11,%12,%13}," \
                "{%14},{%15,%16},{%17},{%18,%19};\n" \
                :"=f"(acc[nt][0]),"=f"(acc[nt][1]),"=f"(acc[nt][2]),"=f"(acc[nt][3]) \
                :"r"(a0),"r"(a1),"r"(a2),"r"(a3),"r"(b0),"r"(b1), \
                 "f"(acc[nt][0]),"f"(acc[nt][1]),"f"(acc[nt][2]),"f"(acc[nt][3]), \
                 "r"(sfa),"h"(bidA),"h"(tidA_),"r"(sfb_raw),"h"(bidB),"h"(tidB_)); \
        } \
    } while(0)

    FP4_ISSUE_LOADS(0, 0);
    moe_cp_async_commit();
    moe_cp_async_wait_all();
    __syncthreads();
    FP4_QUANT_A(0);
    FP4_TRANSPOSE(0);
    __syncthreads();

    int cur = 0;
    for (unsigned int k_base = K_STEP_T64; k_base < K; k_base += K_STEP_T64) {
        int nxt = 1 - cur;
        FP4_ISSUE_LOADS(nxt, k_base);
        moe_cp_async_commit();
        {
            const unsigned char (*smem_Bs_cur)[N_TILE_LG + BP_PAD] = smem_Bs_fp4[cur];
            FP4_COMPUTE_MMA();
        }
        moe_cp_async_wait_all();
        __syncthreads();
        FP4_QUANT_A(nxt);
        FP4_TRANSPOSE(nxt);
        __syncthreads();
        cur = nxt;
    }
    {
        const unsigned char (*smem_Bs_cur)[N_TILE_LG + BP_PAD] = smem_Bs_fp4[cur];
        FP4_COMPUTE_MMA();
    }

    #undef FP4_ISSUE_LOADS
    #undef FP4_QUANT_A
    #undef FP4_FRAG
    #undef FP4_TRANSPOSE
    #undef FP4_COMPUTE_MMA

    #pragma unroll
    for (int nt = 0; nt < 16; nt++) {
        unsigned int c0 = cta_n + nt*8 + tid*2;
        unsigned int c1 = c0 + 1;
        unsigned int r0 = cta_m + warp_m_offset + group_id;
        unsigned int r1 = r0 + 8;
        bool r0v = (int)(warp_m_offset + group_id + cta_m_local) < M_expert;
        bool r1v = (int)(warp_m_offset + group_id + 8 + cta_m_local) < M_expert;
        if (r0v && c0 < N) C[r0*N+c0] = __float2bfloat16(acc[nt][0] * scale2);
        if (r0v && c1 < N) C[r0*N+c1] = __float2bfloat16(acc[nt][1] * scale2);
        if (r1v && c0 < N) C[r1*N+c0] = __float2bfloat16(acc[nt][2] * scale2);
        if (r1v && c1 < N) C[r1*N+c1] = __float2bfloat16(acc[nt][3] * scale2);
    }
}

// 2026-09-25: moe_w4a16_down_t_k64_fp4: moe_w4a16_fused_gate_up_t_k64_fp4 as a single GEMM (one
// B / scale / scale2 / C set; grid x spans N, not 2N) for the routed down projection under
// METRALE_HOLO_MOE_DOWN_FP4. A is the [rows, K] BF16 intermediate activation; B is the
// down_ptrs_t table (packed [K/2, N], scales [K/16, N]). forward_prefill_routed.rs passes
// sorted_token_ids null, so C row r reads A row r. Grid (ceil(N/128), max_m_tiles,
// num_experts), block 128 (ops::moe_w4a16_grouped_gemm_ptrtable_n128).












extern "C" __global__ void moe_w4a16_down_t_k64_fp4(
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

    const unsigned int cta_n = blockIdx.x * N_TILE_LG;

    const unsigned char* B_expert = (const unsigned char*)B_packed_ptrs[expert_id];
    const unsigned char* S_expert = (const unsigned char*)B_scale_ptrs[expert_id];
    const float scale2 = scale2_vals[expert_id];
    if (B_expert == 0) return;

    const unsigned int cta_m = m_start + cta_m_local;
    const unsigned int warp_id = threadIdx.x / 32;
    const unsigned int lane_id = threadIdx.x % 32;
    const unsigned int warp_m_offset = warp_id * 16;
    const unsigned int group_id = lane_id >> 2;
    const unsigned int tid = lane_id & 3;

    __shared__ __nv_bfloat16 smem_A_fp4[2][M_TILE][K_STEP_T64 + PAD_T64];
    // 2026-09-25: B staging (K-major, double-buffered) and MMA-ready tile (N-major,
    // single-buffered), as in the gate/up FP4 kernel; DN4_TRANSPOSE copies one into the other.

    __shared__ unsigned char smem_BpT[2][K_STEP_T64 / 2][N_TILE_LG + BP_PAD];
    __shared__ unsigned char smem_Bp[N_TILE_LG][K_STEP_T64 / 2 + 16];
    __shared__ unsigned char smem_Bs_fp4[2][K_STEP_T64 / GROUP_SIZE][N_TILE_LG + BP_PAD];
    __shared__ unsigned char smem_Ap_fp4[M_TILE][K_STEP_T64 / 2 + 4];
    __shared__ unsigned char smem_As_fp4[M_TILE][K_STEP_T64 / GROUP_SIZE];
    __shared__ int smem_tok_fp4[M_TILE];

    if (threadIdx.x < M_TILE) {
        int local_row = threadIdx.x;
        if (sorted_token_ids && (cta_m_local + local_row) < (unsigned int)M_expert)
            smem_tok_fp4[local_row] = sorted_token_ids[cta_m + local_row];
        else
            smem_tok_fp4[local_row] = (int)(cta_m + local_row);
    }
    __syncthreads();

    float acc[16][4];
    #pragma unroll
    for (int i = 0; i < 16; i++) {
        acc[i][0] = 0.0f; acc[i][1] = 0.0f;
        acc[i][2] = 0.0f; acc[i][3] = 0.0f;
    }

    const unsigned int M_eff = (unsigned int)M_expert;
    const unsigned int num_groups = K / GROUP_SIZE;

    #define DN4_ISSUE_LOADS(buf, kb) do { \
        { \
            unsigned int a_row_base = threadIdx.x >> 3; \
            unsigned int a_col = (threadIdx.x & 7) << 3; \
            unsigned int gc = (kb) + a_col; \
            _Pragma("unroll") \
            for (int rnd = 0; rnd < 4; rnd++) { \
                unsigned int row = rnd * 16 + a_row_base; \
                bool valid = (cta_m_local + row) < M_eff && (gc + 7 < K); \
                unsigned int a_row = (unsigned int)smem_tok_fp4[row]; \
                moe_cp_async_pred_16(&smem_A_fp4[(buf)][row][a_col], \
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
                moe_cp_async_pred_16(&smem_BpT[(buf)][kp_cur][ns], \
                    &B_expert[(unsigned long long)(gke >> 1) * N + gns], \
                    (gke + 1 < K) && (gns + 15 < N)); \
            } \
        } \
        { \
            unsigned int g = threadIdx.x >> 5; \
            unsigned int nn = threadIdx.x & 31; \
            unsigned int sg = (kb) / GROUP_SIZE + g; \
            _Pragma("unroll") \
            for (int rnd = 0; rnd < 4; rnd++) { \
                unsigned int n_cur = rnd * 32 + nn; \
                unsigned int gns = cta_n + n_cur; \
                bool sv = (gns < N) && (sg < num_groups); \
                smem_Bs_fp4[(buf)][g][n_cur] = sv ? \
                    S_expert[(unsigned long long)sg * N + gns] : 0; \
            } \
        } \
    } while(0)

    #define DN4_QUANT_A(buf) do { \
        _Pragma("unroll") \
        for (int job = 0; job < 2; job++) { \
            unsigned int jid = threadIdx.x + job * 128; \
            unsigned int row = jid >> 2; \
            unsigned int grp = jid & 3; \
            const __nv_bfloat16* arow = &smem_A_fp4[(buf)][row][grp * GROUP_SIZE]; \
            float max_abs = 0.0f; \
            float vals[16]; \
            _Pragma("unroll") \
            for (int i = 0; i < 16; i++) { \
                vals[i] = __bfloat162float(arow[i]); \
                max_abs = fmaxf(max_abs, fabsf(vals[i])); \
            } \
            float sc = max_abs > 0.0f ? max_abs * (1.0f / 6.0f) : 1.0f; \
            __nv_fp8_e4m3 sf8(sc); \
            unsigned char sfb = *(unsigned char*)&sf8; \
            smem_As_fp4[row][grp] = sfb; \
            float dec = (float)sf8; \
            float inv = dec > 0.0f ? 1.0f / dec : 0.0f; \
            _Pragma("unroll") \
            for (int i = 0; i < 16; i++) vals[i] *= inv; \
            unsigned int* dst = (unsigned int*)&smem_Ap_fp4[row][grp * (GROUP_SIZE / 2)]; \
            _Pragma("unroll") \
            for (int half = 0; half < 2; half++) { \
                unsigned int out; \
                const float* s = &vals[half * 8]; \
                asm volatile( \
                    "{\n" \
                    ".reg .b8 b0; .reg .b8 b1; .reg .b8 b2; .reg .b8 b3;\n" \
                    "cvt.rn.satfinite.e2m1x2.f32 b0, %2, %1;\n" \
                    "cvt.rn.satfinite.e2m1x2.f32 b1, %4, %3;\n" \
                    "cvt.rn.satfinite.e2m1x2.f32 b2, %6, %5;\n" \
                    "cvt.rn.satfinite.e2m1x2.f32 b3, %8, %7;\n" \
                    "mov.b32 %0, {b0, b1, b2, b3};\n" \
                    "}" \
                    : "=r"(out) \
                    : "f"(s[0]), "f"(s[1]), "f"(s[2]), "f"(s[3]), \
                      "f"(s[4]), "f"(s[5]), "f"(s[6]), "f"(s[7])); \
                dst[half] = out; \
            } \
        } \
    } while(0)

    #define DN4_FRAG(P, ROW, KK) (*(const unsigned int*)&(P)[(ROW)][(KK) / 2])

    // 2026-09-25: Staging -> MMA-ready tile, the same byte copy as FP4_TRANSPOSE.
    #define DN4_TRANSPOSE(buf) do { \
        unsigned int my_n = threadIdx.x; \
        if (my_n < N_TILE_LG) { \
            _Pragma("unroll") \
            for (int q = 0; q < (K_STEP_T64 / 2) / 4; q++) { \
                unsigned int w = (unsigned int)smem_BpT[(buf)][q * 4 + 0][my_n] \
                    | ((unsigned int)smem_BpT[(buf)][q * 4 + 1][my_n] << 8) \
                    | ((unsigned int)smem_BpT[(buf)][q * 4 + 2][my_n] << 16) \
                    | ((unsigned int)smem_BpT[(buf)][q * 4 + 3][my_n] << 24); \
                *(unsigned int*)&smem_Bp[my_n][q * 4] = w; \
            } \
        } \
    } while(0)

    #define DN4_COMPUTE_MMA() do { \
        unsigned int ra = warp_m_offset + group_id; \
        unsigned int a0 = DN4_FRAG(smem_Ap_fp4, ra,     tid * 8); \
        unsigned int a1 = DN4_FRAG(smem_Ap_fp4, ra + 8, tid * 8); \
        unsigned int a2 = DN4_FRAG(smem_Ap_fp4, ra,     32 + tid * 8); \
        unsigned int a3 = DN4_FRAG(smem_Ap_fp4, ra + 8, 32 + tid * 8); \
        unsigned int sfa_m = (lane_id & 1) * 8 + (lane_id >> 2); \
        unsigned int sfa = (unsigned int)smem_As_fp4[warp_m_offset + sfa_m][0] \
                         | ((unsigned int)smem_As_fp4[warp_m_offset + sfa_m][1] << 8) \
                         | ((unsigned int)smem_As_fp4[warp_m_offset + sfa_m][2] << 16) \
                         | ((unsigned int)smem_As_fp4[warp_m_offset + sfa_m][3] << 24); \
        _Pragma("unroll") \
        for (int nt = 0; nt < 16; nt++) { \
            unsigned int nc = nt * 8 + group_id; \
            unsigned int b0 = DN4_FRAG(smem_Bp, nc, tid * 8); \
            unsigned int b1 = DN4_FRAG(smem_Bp, nc, 32 + tid * 8); \
            unsigned int sfn = nt * 8 + (lane_id >> 2); \
            unsigned int sfb_raw = (unsigned int)smem_Bs_cur[0][sfn] \
                         | ((unsigned int)smem_Bs_cur[1][sfn] << 8) \
                         | ((unsigned int)smem_Bs_cur[2][sfn] << 16) \
                         | ((unsigned int)smem_Bs_cur[3][sfn] << 24); \
            unsigned short bidA = 0, tidA_ = 0, bidB = 0, tidB_ = 0; \
            asm volatile( \
                "mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3 " \
                "{%0,%1,%2,%3}," \
                "{%4,%5,%6,%7}," \
                "{%8,%9}," \
                "{%10,%11,%12,%13}," \
                "{%14},{%15,%16},{%17},{%18,%19};\n" \
                :"=f"(acc[nt][0]),"=f"(acc[nt][1]),"=f"(acc[nt][2]),"=f"(acc[nt][3]) \
                :"r"(a0),"r"(a1),"r"(a2),"r"(a3),"r"(b0),"r"(b1), \
                 "f"(acc[nt][0]),"f"(acc[nt][1]),"f"(acc[nt][2]),"f"(acc[nt][3]), \
                 "r"(sfa),"h"(bidA),"h"(tidA_),"r"(sfb_raw),"h"(bidB),"h"(tidB_)); \
        } \
    } while(0)

    DN4_ISSUE_LOADS(0, 0);
    moe_cp_async_commit();
    moe_cp_async_wait_all();
    __syncthreads();
    DN4_QUANT_A(0);
    DN4_TRANSPOSE(0);
    __syncthreads();

    int cur = 0;
    for (unsigned int k_base = K_STEP_T64; k_base < K; k_base += K_STEP_T64) {
        int nxt = 1 - cur;
        DN4_ISSUE_LOADS(nxt, k_base);
        moe_cp_async_commit();
        {
            const unsigned char (*smem_Bs_cur)[N_TILE_LG + BP_PAD] = smem_Bs_fp4[cur];
            DN4_COMPUTE_MMA();
        }
        moe_cp_async_wait_all();
        __syncthreads();
        DN4_QUANT_A(nxt);
        DN4_TRANSPOSE(nxt);
        __syncthreads();
        cur = nxt;
    }
    {
        const unsigned char (*smem_Bs_cur)[N_TILE_LG + BP_PAD] = smem_Bs_fp4[cur];
        DN4_COMPUTE_MMA();
    }

    #undef DN4_ISSUE_LOADS
    #undef DN4_QUANT_A
    #undef DN4_FRAG
    #undef DN4_TRANSPOSE
    #undef DN4_COMPUTE_MMA

    #pragma unroll
    for (int nt = 0; nt < 16; nt++) {
        unsigned int c0 = cta_n + nt*8 + tid*2;
        unsigned int c1 = c0 + 1;
        unsigned int r0 = cta_m + warp_m_offset + group_id;
        unsigned int r1 = r0 + 8;
        bool r0v = (int)(warp_m_offset + group_id + cta_m_local) < M_expert;
        bool r1v = (int)(warp_m_offset + group_id + 8 + cta_m_local) < M_expert;
        if (r0v && c0 < N) C[r0*N+c0] = __float2bfloat16(acc[nt][0] * scale2);
        if (r0v && c1 < N) C[r0*N+c1] = __float2bfloat16(acc[nt][1] * scale2);
        if (r1v && c0 < N) C[r1*N+c0] = __float2bfloat16(acc[nt][2] * scale2);
        if (r1v && c1 < N) C[r1*N+c1] = __float2bfloat16(acc[nt][3] * scale2);
    }
}
#endif
