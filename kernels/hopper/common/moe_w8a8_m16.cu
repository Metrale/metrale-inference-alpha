// SPDX-License-Identifier: AGPL-3.0-only
// Hopper small-expert geometry of kernels/gb10/common/moe_w8a8_grouped_gemm.cu,
// the PM4 body starting at W8PM4_M_TILE: M_TILE 128 -> 16, THREADS 256 -> 32,
// W8PM4_/w8pm4_ symbols renamed P16_/p16_, entry renamed pm4_m16.
// MMA instructions, K iteration, FP32 scale folding and BF16 stores unchanged.
// The caller partitions <=16-row experts on device and retains M128 for others;
// sparse M16 buckets fall back to M128 to avoid single-warp latency regressions.
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#define P16_M_TILE 16
#define P16_N_TILE 64
#define P16_K_STEP 32
#define P16_K_SUB 16
#define P16_K_SUBS (P16_K_STEP / P16_K_SUB)
#define P16_PAD 2
#define P16_A_STRIDE (P16_K_STEP + 8)
#define P16_THREADS 32
#define P16_NT_PER_WARP (P16_N_TILE / 8)
#define P16_STAGES 2
#define P16_K_PROMOTE 64
#define P16_FP8_BLOCK 128

__device__ __forceinline__ __nv_bfloat16 p16_e4m3_to_bf16(unsigned char b) {
    // E4M3 -> f32 by bit arithmetic: place the 7 magnitude bits in an f16
    // exponent/mantissa frame and rescale by 2^8 (e4m3 bias 7 vs f16 bias 15).
    // Handles subnormals for free; NaN codes 0x7F/0xFF map to +/-0.0 (LUT parity).
    float f = __half2float(__ushort_as_half((unsigned short)((b & 0x7f) << 7))) * 256.0f;
    f = ((b & 0x7f) == 0x7f) ? 0.0f : f;
    return __float2bfloat16((b & 0x80) ? -f : f);
}

__device__ __forceinline__ void p16_cp_async_cg_16(void* smem_ptr, const void* gmem_ptr) {
    unsigned int s = (unsigned int)__cvta_generic_to_shared(smem_ptr);
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" ::"r"(s), "l"(gmem_ptr));
}
__device__ __forceinline__ void p16_cp_async_commit() {
    asm volatile("cp.async.commit_group;\n" ::);
}
template <int N>
__device__ __forceinline__ void p16_cp_async_wait_group() {
    asm volatile("cp.async.wait_group %0;\n" ::"n"(N));
}
__device__ __forceinline__ void p16_cp_async_wait_le(unsigned int n) {
    switch (n) {
        case 0:  p16_cp_async_wait_group<0>(); break;
        case 1:  p16_cp_async_wait_group<1>(); break;
        default: p16_cp_async_wait_group<2>(); break;
    }
}

// MMA over one resident K_STEP (2 x m16n8k16 sub-MMAs in ascending K order —
// identical f32 accumulation order to the baseline's K_STEP=16 sequence).
// smem_B is [n][k] K-contiguous: the (k,k+1) B-fragment pair is one aligned u32.
__device__ __forceinline__ void p16_mma_kstep(
    const __nv_bfloat16* smem_A,   // [P16_M_TILE][P16_A_STRIDE]
    const __nv_bfloat16* smem_B,   // [P16_N_TILE][P16_K_STEP + P16_PAD]
    float inner[P16_NT_PER_WARP][4],
    unsigned int warp_m_offset, unsigned int group_id, unsigned int tid
) {
    const unsigned int a_stride = P16_A_STRIDE;
    const unsigned int b_stride = P16_K_STEP + P16_PAD;
    const unsigned short* sA = (const unsigned short*)smem_A;
    const unsigned short* sB = (const unsigned short*)smem_B;

    unsigned int frag_r0 = warp_m_offset + group_id;
    unsigned int frag_r1 = warp_m_offset + group_id + 8;

    #pragma unroll
    for (int s = 0; s < P16_K_SUBS; s++) {
        const unsigned int k_off = s * P16_K_SUB;
        unsigned int frag_c0 = k_off + tid * 2;
        unsigned int frag_c1 = k_off + tid * 2 + 8;

        unsigned int a0 = *(const unsigned int*)&sA[frag_r0 * a_stride + frag_c0];
        unsigned int a1 = *(const unsigned int*)&sA[frag_r1 * a_stride + frag_c0];
        unsigned int a2 = *(const unsigned int*)&sA[frag_r0 * a_stride + frag_c1];
        unsigned int a3 = *(const unsigned int*)&sA[frag_r1 * a_stride + frag_c1];

        #pragma unroll
        for (int n_tile = 0; n_tile < P16_NT_PER_WARP; n_tile++) {
            unsigned int n_col = n_tile * 8 + group_id;
            unsigned int k0 = k_off + tid * 2;
            unsigned int k1 = k_off + tid * 2 + 8;

            unsigned int b0 = *(const unsigned int*)&sB[n_col * b_stride + k0];
            unsigned int b1 = *(const unsigned int*)&sB[n_col * b_stride + k1];

            asm volatile(
                "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
                "{%0, %1, %2, %3}, "
                "{%4, %5, %6, %7}, "
                "{%8, %9}, "
                "{%10, %11, %12, %13};"
                : "=f"(inner[n_tile][0]), "=f"(inner[n_tile][1]),
                  "=f"(inner[n_tile][2]), "=f"(inner[n_tile][3])
                : "r"(a0), "r"(a1), "r"(a2), "r"(a3),
                  "r"(b0), "r"(b1),
                  "f"(inner[n_tile][0]), "f"(inner[n_tile][1]),
                  "f"(inner[n_tile][2]), "f"(inner[n_tile][3])
            );
        }
    }
}

extern "C" __global__ void __launch_bounds__(P16_THREADS, 2) pm4_m16(
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
    __shared__ __align__(16) __nv_bfloat16 smem_A[P16_STAGES][P16_M_TILE][P16_A_STRIDE];
    __shared__ __align__(16) unsigned char smem_Araw[P16_STAGES][P16_M_TILE][P16_K_STEP];
    __shared__ __nv_bfloat16 smem_B[P16_STAGES][P16_N_TILE][P16_K_STEP + P16_PAD];
    __shared__ __align__(16) unsigned char smem_Braw[P16_STAGES][P16_N_TILE][P16_K_STEP];

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

        const unsigned int cta_m_local = mt * P16_M_TILE;
        const unsigned int cta_n = nt * P16_N_TILE;

        float inner_acc[P16_NT_PER_WARP][4];
        float outer_acc[P16_NT_PER_WARP][4];
        #pragma unroll
        for (int i = 0; i < P16_NT_PER_WARP; i++) {
            inner_acc[i][0] = 0.0f; inner_acc[i][1] = 0.0f;
            inner_acc[i][2] = 0.0f; inner_acc[i][3] = 0.0f;
            outer_acc[i][0] = 0.0f; outer_acc[i][1] = 0.0f;
            outer_acc[i][2] = 0.0f; outer_acc[i][3] = 0.0f;
        }

        const unsigned int k_blocks = (K + P16_FP8_BLOCK - 1) / P16_FP8_BLOCK;
        const unsigned int n_block = cta_n / P16_FP8_BLOCK;
        const unsigned int n_steps = (K + P16_K_STEP - 1) / P16_K_STEP;
        const unsigned int steps_per_promote = P16_K_PROMOTE / P16_K_STEP;   // 2

        // Per-warp fragment rows: token ids resolved ONCE per tile for the
        // per-row a_scale fold (rows are fixed for the whole K loop).
        const unsigned int r0 = cta_m_local + warp_m_offset + group_id;
        const unsigned int r1 = r0 + 8;
        const int t0 = (r0 < (unsigned int)M_expert)
            ? (sorted_token_ids ? sorted_token_ids[m_start + (int)r0] : m_start + (int)r0) : -1;
        const int t1 = (r1 < (unsigned int)M_expert)
            ? (sorted_token_ids ? sorted_token_ids[m_start + (int)r1] : m_start + (int)r1) : -1;

        auto prefetch = [&](unsigned int step, unsigned int stage) {
            unsigned int k_base = step * P16_K_STEP;

            // A raw: 128 rows x K_STEP FP8 bytes, 16-B chunks, gathered per row.
            const unsigned int a_chunks = (P16_M_TILE * P16_K_STEP) / 16;   // 256
            #pragma unroll
            for (unsigned int c = threadIdx.x; c < a_chunks; c += P16_THREADS) {
                unsigned int row  = c / (P16_K_STEP / 16);
                unsigned int kcol = (c % (P16_K_STEP / 16)) * 16;
                unsigned int m_global = cta_m_local + row;
                unsigned int gk = k_base + kcol;
                unsigned char* dst = &smem_Araw[stage][row][kcol];
                if (m_global < (unsigned int)M_expert && gk + 16 <= K) {
                    int sorted_idx = m_start + (int)m_global;
                    int token_id = sorted_token_ids ? sorted_token_ids[sorted_idx] : sorted_idx;
                    p16_cp_async_cg_16(dst, &A_fp8[(unsigned long long)token_id * K + gk]);
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
            const unsigned int b_chunks = (P16_N_TILE * P16_K_STEP) / 16;   // 128
            #pragma unroll
            for (unsigned int c = threadIdx.x; c < b_chunks; c += P16_THREADS) {
                unsigned int nrow = (c * 16) / P16_K_STEP;
                unsigned int kcol = (c * 16) % P16_K_STEP;
                unsigned int gn = cta_n + nrow;
                unsigned int gk = k_base + kcol;
                unsigned char* dst = &smem_Braw[stage][nrow][kcol];
                if (gn < N && gk + 16 <= K) {
                    p16_cp_async_cg_16(dst, &B_exp[(unsigned long long)gn * K + gk]);
                } else {
                    #pragma unroll
                    for (unsigned int e = 0; e < 16; e++) {
                        unsigned int gke = gk + e;
                        dst[e] = (gn < N && gke < K) ? B_exp[(unsigned long long)gn * K + gke] : 0;
                    }
                }
            }
            p16_cp_async_commit();
        };

        // Arithmetic-dequant just-arrived raw A and B for `stage` into the
        // MMA-ready BF16 buffers (no scale — folded post-MMA at K_PROMOTE).
        auto dequant = [&](unsigned int stage) {
            #pragma unroll
            for (unsigned int idx = threadIdx.x; idx < P16_M_TILE * P16_K_STEP; idx += P16_THREADS) {
                unsigned int row = idx / P16_K_STEP;
                unsigned int k   = idx % P16_K_STEP;
                smem_A[stage][row][k] = p16_e4m3_to_bf16(smem_Araw[stage][row][k]);
            }
            #pragma unroll
            for (unsigned int idx = threadIdx.x; idx < P16_N_TILE * P16_K_STEP; idx += P16_THREADS) {
                unsigned int n = idx / P16_K_STEP;
                unsigned int k = idx % P16_K_STEP;
                smem_B[stage][n][k] = p16_e4m3_to_bf16(smem_Braw[stage][n][k]);
            }
        };

        #pragma unroll
        for (unsigned int p = 0; p < P16_STAGES - 1; p++) {
            if (p < n_steps) prefetch(p, p % P16_STAGES);
        }
        unsigned int k_step_in_prom = 0;

        for (unsigned int step = 0; step < n_steps; step++) {
            unsigned int cur = step % P16_STAGES;

            unsigned int ahead = step + (P16_STAGES - 1);
            if (ahead < n_steps) prefetch(ahead, ahead % P16_STAGES);
            unsigned int committed = min(n_steps, P16_STAGES + step);
            unsigned int target = committed - (step + 1);
            p16_cp_async_wait_le(target);
            __syncthreads();   // raw A/B for `cur` resident for all threads

            dequant(cur);
            __syncthreads();   // smem_A/B[cur] fully written before MMA reads

            p16_mma_kstep(&smem_A[cur][0][0], &smem_B[cur][0][0],
                         inner_acc, warp_m_offset, group_id, tid);
            __syncthreads();   // done reading smem_*[cur]; safe for reuse

            // K_PROMOTE boundary: fold per-row a_scale x b_scale, reset inner.
            k_step_in_prom++;
            if (k_step_in_prom == steps_per_promote || step + 1 == n_steps) {
                const unsigned int k_block = (step * P16_K_STEP) / P16_FP8_BLOCK;
                const float bs = S_exp[n_block * k_blocks + k_block];
                const float as0 = (t0 >= 0)
                    ? a_scale[(unsigned long long)t0 * k_blocks + k_block] : 0.0f;
                const float as1 = (t1 >= 0)
                    ? a_scale[(unsigned long long)t1 * k_blocks + k_block] : 0.0f;
                const float s0 = as0 * bs;
                const float s1 = as1 * bs;
                #pragma unroll
                for (int i = 0; i < P16_NT_PER_WARP; i++) {
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
        for (int n_tile = 0; n_tile < P16_NT_PER_WARP; n_tile++) {
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
