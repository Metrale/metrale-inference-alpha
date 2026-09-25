// SPDX-License-Identifier: AGPL-3.0-only
// Hopper small-bucket native FP8 reduction candidate, separately qualified
// numerically; NOT bit-identical to the BF16 M16/M128 reduction. Reuses the
// existing FP8 bytes/scales, K64 promotion and FP32 outer scale fold.
// The device bucket/occupancy policy and large M128 BF16 path are unchanged.
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#define PN16_M_TILE 16
#define PN16_N_TILE 64
#define PN16_K_STEP 32
#define PN16_K_SUB 16
#define PN16_K_SUBS (PN16_K_STEP / PN16_K_SUB)
#define PN16_PAD 2
#define PN16_A_STRIDE (PN16_K_STEP + 8)
#define PN16_THREADS 32
#define PN16_NT_PER_WARP (PN16_N_TILE / 8)
#define PN16_STAGES 2
#define PN16_K_PROMOTE 64
#define PN16_FP8_BLOCK 128

__device__ __forceinline__ __nv_bfloat16 pn16_e4m3_to_bf16(unsigned char b) {
    // E4M3 -> f32 by bit arithmetic: place the 7 magnitude bits in an f16
    // exponent/mantissa frame and rescale by 2^8 (e4m3 bias 7 vs f16 bias 15).
    // Handles subnormals for free; NaN codes 0x7F/0xFF map to +/-0.0 (LUT parity).
    float f = __half2float(__ushort_as_half((unsigned short)((b & 0x7f) << 7))) * 256.0f;
    f = ((b & 0x7f) == 0x7f) ? 0.0f : f;
    return __float2bfloat16((b & 0x80) ? -f : f);
}

__device__ __forceinline__ void pn16_cp_async_cg_16(void* smem_ptr, const void* gmem_ptr) {
    unsigned int s = (unsigned int)__cvta_generic_to_shared(smem_ptr);
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" ::"r"(s), "l"(gmem_ptr));
}
__device__ __forceinline__ void pn16_cp_async_commit() {
    asm volatile("cp.async.commit_group;\n" ::);
}
template <int N>
__device__ __forceinline__ void pn16_cp_async_wait_group() {
    asm volatile("cp.async.wait_group %0;\n" ::"n"(N));
}
__device__ __forceinline__ void pn16_cp_async_wait_le(unsigned int n) {
    switch (n) {
        case 0:  pn16_cp_async_wait_group<0>(); break;
        case 1:  pn16_cp_async_wait_group<1>(); break;
        default: pn16_cp_async_wait_group<2>(); break;
    }
}

// One native E4M3 m16n8k32 MMA per K_STEP. Its reduction grouping differs
// from the BF16 reference's two K16 MMAs; rounding differences are expected.
// Raw A/B fragments pack four FP8 bytes per register.
// PM4's BF16 dequant maps FP8 NaN encodings to signed zero. Preserve
// that policy byte-by-byte before native MMA, without new quantization.
__device__ __forceinline__ unsigned pn16_sanitize(unsigned packed) {
 unsigned result=0;
 #pragma unroll
 for(unsigned b=0;b<4;b++) {
  unsigned x=(packed>>(8*b))&255;
  if((x&127)==127) x &= 128;
  result |= x << (8*b);
 }
 return result;
}
__device__ __forceinline__ void pn16_mma_kstep(
 const unsigned char* A, const unsigned char* B,
 float inner[PN16_NT_PER_WARP][4], unsigned warp_m_offset,
 unsigned group_id, unsigned tid) {
 const unsigned r0=warp_m_offset+group_id,r1=r0+8;
 unsigned a0=pn16_sanitize(*(const unsigned*)&A[r0*32+4*tid]);
 unsigned a1=pn16_sanitize(*(const unsigned*)&A[r1*32+4*tid]);
 unsigned a2=pn16_sanitize(*(const unsigned*)&A[r0*32+16+4*tid]);
 unsigned a3=pn16_sanitize(*(const unsigned*)&A[r1*32+16+4*tid]);
 #pragma unroll
 for(unsigned nt=0;nt<PN16_NT_PER_WARP;nt++) {
  unsigned row=nt*8+group_id;
  unsigned b0=pn16_sanitize(*(const unsigned*)&B[row*32+4*tid]);
  unsigned b1=pn16_sanitize(*(const unsigned*)&B[row*32+16+4*tid]);
  asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 "
   "{%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%10,%11,%12,%13};"
   : "=f"(inner[nt][0]),"=f"(inner[nt][1]),"=f"(inner[nt][2]),"=f"(inner[nt][3])
   : "r"(a0),"r"(a1),"r"(a2),"r"(a3),"r"(b0),"r"(b1),
     "f"(inner[nt][0]),"f"(inner[nt][1]),"f"(inner[nt][2]),"f"(inner[nt][3]));
 }
}

extern "C" __global__ void __launch_bounds__(PN16_THREADS, 2) pm4_m16(
    const unsigned char* __restrict__ A_fp8,                // [total_tokens, K] FP8 E4M3
    const float* __restrict__ a_scale,                      // [total_tokens, K/128] FP32
    const unsigned long long* __restrict__ B_weight_ptrs,   // [num_experts] -> [N, K] FP8
    const unsigned long long* __restrict__ B_scale_ptrs,    // [num_experts] -> [N/128, K/128] FP32
    __nv_bfloat16* __restrict__ C,                          // [total_expanded, N] BF16
    const int* __restrict__ expert_offsets,                 // [num_experts + 1]
    const int* __restrict__ sorted_token_ids,               // [total_expanded] or NULL
    unsigned int num_experts,
    unsigned int N,
    unsigned int K,
    const unsigned int* __restrict__ worklist,              // [*total_tiles * 2]
    const int* __restrict__ total_tiles                     // [1]
) {
    __shared__ __align__(16) unsigned char smem_Araw[PN16_STAGES][PN16_M_TILE][PN16_K_STEP];
    __shared__ __align__(16) unsigned char smem_Braw[PN16_STAGES][PN16_N_TILE][PN16_K_STEP];

    const unsigned int warp_id = threadIdx.x / 32;
    const unsigned int lane_id = threadIdx.x % 32;
    const unsigned int warp_m_offset = warp_id * 16;
    const unsigned int group_id = lane_id >> 2;
    const unsigned int tid = lane_id & 3;

    const int total = *total_tiles;

    for (int wid = blockIdx.x; wid < total; wid += (int)gridDim.x) {
        __syncthreads();   // fence smem reuse before re-priming the pipeline

        unsigned int expert_id = worklist[wid * 2 + 0];
        unsigned int packed    = worklist[wid * 2 + 1];
        unsigned int mt = packed >> 6;
        unsigned int nt = packed & 0x3F;

        const int m_start = expert_offsets[expert_id];
        const int M_expert = expert_offsets[expert_id + 1] - m_start;

        const unsigned char* B_exp = (const unsigned char*)B_weight_ptrs[expert_id];
        const float* S_exp = (const float*)B_scale_ptrs[expert_id];
        if (B_exp == 0) continue;

        const unsigned int cta_m_local = mt * PN16_M_TILE;
        const unsigned int cta_n = nt * PN16_N_TILE;

        float inner_acc[PN16_NT_PER_WARP][4];
        float outer_acc[PN16_NT_PER_WARP][4];
        #pragma unroll
        for (int i = 0; i < PN16_NT_PER_WARP; i++) {
            inner_acc[i][0] = 0.0f; inner_acc[i][1] = 0.0f;
            inner_acc[i][2] = 0.0f; inner_acc[i][3] = 0.0f;
            outer_acc[i][0] = 0.0f; outer_acc[i][1] = 0.0f;
            outer_acc[i][2] = 0.0f; outer_acc[i][3] = 0.0f;
        }

        const unsigned int k_blocks = (K + PN16_FP8_BLOCK - 1) / PN16_FP8_BLOCK;
        const unsigned int n_block = cta_n / PN16_FP8_BLOCK;
        const unsigned int n_steps = (K + PN16_K_STEP - 1) / PN16_K_STEP;
        const unsigned int steps_per_promote = PN16_K_PROMOTE / PN16_K_STEP;   // 2

        // Per-warp fragment rows: token ids resolved ONCE per tile for the
        // per-row a_scale fold (rows are fixed for the whole K loop).
        const unsigned int r0 = cta_m_local + warp_m_offset + group_id;
        const unsigned int r1 = r0 + 8;
        const int t0 = (r0 < (unsigned int)M_expert)
            ? (sorted_token_ids ? sorted_token_ids[m_start + (int)r0] : m_start + (int)r0) : -1;
        const int t1 = (r1 < (unsigned int)M_expert)
            ? (sorted_token_ids ? sorted_token_ids[m_start + (int)r1] : m_start + (int)r1) : -1;

        auto prefetch = [&](unsigned int step, unsigned int stage) {
            unsigned int k_base = step * PN16_K_STEP;

            // A raw: 16 rows x K_STEP FP8 bytes, 16-B chunks, gathered per row.
            const unsigned int a_chunks = (PN16_M_TILE * PN16_K_STEP) / 16;   // 32
            #pragma unroll
            for (unsigned int c = threadIdx.x; c < a_chunks; c += PN16_THREADS) {
                unsigned int row  = c / (PN16_K_STEP / 16);
                unsigned int kcol = (c % (PN16_K_STEP / 16)) * 16;
                unsigned int m_global = cta_m_local + row;
                unsigned int gk = k_base + kcol;
                unsigned char* dst = &smem_Araw[stage][row][kcol];
                if (m_global < (unsigned int)M_expert && gk + 16 <= K) {
                    int sorted_idx = m_start + (int)m_global;
                    int token_id = sorted_token_ids ? sorted_token_ids[sorted_idx] : sorted_idx;
                    pn16_cp_async_cg_16(dst, &A_fp8[(unsigned long long)token_id * K + gk]);
                } else {
                    #pragma unroll
                    for (unsigned int e = 0; e < 16; e++) {
                        unsigned int gke = gk + e;
                        if (m_global < (unsigned int)M_expert && gke < K) {
                            int sorted_idx = m_start + (int)m_global;
                            int token_id = sorted_token_ids ? sorted_token_ids[sorted_idx] : sorted_idx;
                            dst[e] = A_fp8[(unsigned long long)token_id * K + gke];
                        } else {
                            dst[e] = 0;   // dequants to +0.0
                        }
                    }
                }
            }

            // B raw: N_TILE rows x K_STEP FP8 bytes, 16-B chunks.
            const unsigned int b_chunks = (PN16_N_TILE * PN16_K_STEP) / 16;   // 128
            #pragma unroll
            for (unsigned int c = threadIdx.x; c < b_chunks; c += PN16_THREADS) {
                unsigned int nrow = (c * 16) / PN16_K_STEP;
                unsigned int kcol = (c * 16) % PN16_K_STEP;
                unsigned int gn = cta_n + nrow;
                unsigned int gk = k_base + kcol;
                unsigned char* dst = &smem_Braw[stage][nrow][kcol];
                if (gn < N && gk + 16 <= K) {
                    pn16_cp_async_cg_16(dst, &B_exp[(unsigned long long)gn * K + gk]);
                } else {
                    #pragma unroll
                    for (unsigned int e = 0; e < 16; e++) {
                        unsigned int gke = gk + e;
                        dst[e] = (gn < N && gke < K) ? B_exp[(unsigned long long)gn * K + gke] : 0;
                    }
                }
            }
            pn16_cp_async_commit();
        };

        #pragma unroll
        for (unsigned int p = 0; p < PN16_STAGES - 1; p++) {
            if (p < n_steps) prefetch(p, p % PN16_STAGES);
        }
        unsigned int k_step_in_prom = 0;

        for (unsigned int step = 0; step < n_steps; step++) {
            unsigned int cur = step % PN16_STAGES;

            unsigned int ahead = step + (PN16_STAGES - 1);
            if (ahead < n_steps) prefetch(ahead, ahead % PN16_STAGES);
            unsigned int committed = min(n_steps, PN16_STAGES + step);
            unsigned int target = committed - (step + 1);
            pn16_cp_async_wait_le(target);
            __syncthreads();   // raw A/B for `cur` resident for all threads

            pn16_mma_kstep(&smem_Araw[cur][0][0], &smem_Braw[cur][0][0],
                         inner_acc, warp_m_offset, group_id, tid);
            __syncthreads();   // done reading smem_*[cur]; safe for reuse

            // K_PROMOTE boundary: fold per-row a_scale x b_scale, reset inner.
            k_step_in_prom++;
            if (k_step_in_prom == steps_per_promote || step + 1 == n_steps) {
                const unsigned int k_block = (step * PN16_K_STEP) / PN16_FP8_BLOCK;
                const float bs = S_exp[n_block * k_blocks + k_block];
                const float as0 = (t0 >= 0)
                    ? a_scale[(unsigned long long)t0 * k_blocks + k_block] : 0.0f;
                const float as1 = (t1 >= 0)
                    ? a_scale[(unsigned long long)t1 * k_blocks + k_block] : 0.0f;
                const float s0 = as0 * bs;
                const float s1 = as1 * bs;
                #pragma unroll
                for (int i = 0; i < PN16_NT_PER_WARP; i++) {
                    outer_acc[i][0] += inner_acc[i][0] * s0;
                    outer_acc[i][1] += inner_acc[i][1] * s0;
                    outer_acc[i][2] += inner_acc[i][2] * s1;
                    outer_acc[i][3] += inner_acc[i][3] * s1;
                    inner_acc[i][0] = 0.0f; inner_acc[i][1] = 0.0f;
                    inner_acc[i][2] = 0.0f; inner_acc[i][3] = 0.0f;
                }
                k_step_in_prom = 0;
            }
        }

        #pragma unroll
        for (int n_tile = 0; n_tile < PN16_NT_PER_WARP; n_tile++) {
            unsigned int base_n = cta_n + n_tile * 8;
            unsigned int col0 = base_n + (tid * 2);
            unsigned int col1 = col0 + 1;
            unsigned int row0 = cta_m_local + warp_m_offset + group_id;
            unsigned int row1 = row0 + 8;

            if (row0 < (unsigned int)M_expert) {
                unsigned int out_row = m_start + row0;
                if (col0 < N) C[(unsigned long long)out_row * N + col0] = __float2bfloat16(outer_acc[n_tile][0]);
                if (col1 < N) C[(unsigned long long)out_row * N + col1] = __float2bfloat16(outer_acc[n_tile][1]);
            }
            if (row1 < (unsigned int)M_expert) {
                unsigned int out_row = m_start + row1;
                if (col0 < N) C[(unsigned long long)out_row * N + col0] = __float2bfloat16(outer_acc[n_tile][2]);
                if (col1 < N) C[(unsigned long long)out_row * N + col1] = __float2bfloat16(outer_acc[n_tile][3]);
            }
        }
    }
}
