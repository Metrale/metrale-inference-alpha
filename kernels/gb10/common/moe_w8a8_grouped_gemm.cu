// SPDX-License-Identifier: MIT OR Apache-2.0

// 2026-09-25: Kernels `moe_w8a8_grouped_gemm` and `moe_w8a8_grouped_gemm_pm4`: MoE grouped GEMM with FP8 E4M3
// activations and weights, block-scaled per 128 x 128, BF16 output. With c ranging over 64-value chunks of K,
//   C[m, n] = bf16( sum_c ( sum_(k in c) bf16(A[token(m), k]) * bf16(B[n, k]) )
//                        * a_scale[token(m), c / 2] * b_scale[n / 128, c / 2] )
// The inner sums are F32 MMA accumulators, folded into an F32 outer accumulator every K_PROMOTE values.
//
// Expert e owns rows expert_offsets[e] .. expert_offsets[e + 1]; token(m) is sorted_token_ids[m], or m when
// that pointer is NULL. An expert with a NULL weight pointer is skipped and its rows are not written. B, and in
// `_pm4` also A, are read in 16-byte loads, which need K to be a multiple of 16.
// `moe_w8a8_grouped_gemm`: grid (ceil(N / 64), max_m_tiles, num_experts), block 128.
//
// Owner: gb10 kernels.
// Invariants: none beyond the types.



#include <cuda_bf16.h>
#include <cuda_fp16.h>

#define M_TILE 64
#define N_TILE 64
#define K_STEP 16
#define PAD 2
#define FP8_BLOCK 128
#define K_PROMOTE 64

// 2026-09-25: FP8 E4M3 to BF16 without a table. The 7 magnitude bits shifted left 7 form an FP16 with the same
// mantissa and an exponent bias of 15 instead of 7, so the value is that FP16 times 2^8, subnormals included.
// The NaN codes 0x7F and 0xFF, which would decode to +/-480, are mapped to +/-0.





__device__ __forceinline__ __nv_bfloat16 e4m3_to_bf16_w8a8(unsigned char b) {
    unsigned short h = (unsigned short)((b & 0x7f) << 7);
    float f = __half2float(__ushort_as_half(h)) * 256.0f;
    if ((b & 0x7f) == 0x7f) f = 0.0f;
    f = (b & 0x80) ? -f : f;
    return __float2bfloat16(f);
}

__device__ __forceinline__ void fp8_w8a8_mma(
    __nv_bfloat16 smem_A[][K_STEP + PAD],
    __nv_bfloat16 smem_B[][N_TILE + PAD],
    float acc[8][4],
    unsigned int warp_m_offset, unsigned int group_id, unsigned int tid
) {
    const unsigned int a_stride = K_STEP + PAD;
    const unsigned int b_stride = N_TILE + PAD;
    const unsigned short* sA = (const unsigned short*)smem_A;
    const unsigned short* sB = (const unsigned short*)smem_B;

    unsigned int frag_r0 = warp_m_offset + group_id;
    unsigned int frag_r1 = warp_m_offset + group_id + 8;
    unsigned int frag_c0 = tid * 2;
    unsigned int frag_c1 = tid * 2 + 8;

    unsigned int a0 = *(const unsigned int*)&sA[frag_r0 * a_stride + frag_c0];
    unsigned int a1 = *(const unsigned int*)&sA[frag_r1 * a_stride + frag_c0];
    unsigned int a2 = *(const unsigned int*)&sA[frag_r0 * a_stride + frag_c1];
    unsigned int a3 = *(const unsigned int*)&sA[frag_r1 * a_stride + frag_c1];

    #pragma unroll
    for (int n_tile = 0; n_tile < 8; n_tile++) {
        unsigned int n_col = n_tile * 8 + group_id;
        unsigned int k0 = tid * 2;
        unsigned int k1 = tid * 2 + 8;

        unsigned int b0 = ((unsigned int)sB[(k0 + 1) * b_stride + n_col] << 16) |
                          (unsigned int)sB[k0 * b_stride + n_col];
        unsigned int b1 = ((unsigned int)sB[(k1 + 1) * b_stride + n_col] << 16) |
                          (unsigned int)sB[k1 * b_stride + n_col];

        asm volatile(
            "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
            "{%0, %1, %2, %3}, "
            "{%4, %5, %6, %7}, "
            "{%8, %9}, "
            "{%10, %11, %12, %13};"
            : "=f"(acc[n_tile][0]), "=f"(acc[n_tile][1]),
              "=f"(acc[n_tile][2]), "=f"(acc[n_tile][3])
            : "r"(a0), "r"(a1), "r"(a2), "r"(a3),
              "r"(b0), "r"(b1),
              "f"(acc[n_tile][0]), "f"(acc[n_tile][1]),
              "f"(acc[n_tile][2]), "f"(acc[n_tile][3])
        );
    }
}

extern "C" __global__ void moe_w8a8_grouped_gemm(
    const unsigned char* __restrict__ A_fp8,                // 2026-09-25: [total_tokens, K] FP8 E4M3
    const float* __restrict__ a_scale,                      // 2026-09-25: [total_tokens, ceil(K / 128)] F32
    const unsigned long long* __restrict__ B_weight_ptrs,   // 2026-09-25: [num_experts] -> [N, K] FP8
    const unsigned long long* __restrict__ B_scale_ptrs,    // 2026-09-25: [num_experts] -> [ceil(N / 128), ceil(K / 128)] F32
    __nv_bfloat16* __restrict__ C,                          // 2026-09-25: [total_expanded, N] BF16
    const int* __restrict__ expert_offsets,                 // 2026-09-25: [num_experts + 1]
    const int* __restrict__ sorted_token_ids,               // 2026-09-25: [total_expanded] or NULL
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

    const unsigned int cta_n = blockIdx.x * N_TILE;

    const unsigned char* B_exp = (const unsigned char*)B_weight_ptrs[expert_id];
    const float* S_exp = (const float*)B_scale_ptrs[expert_id];
    if (B_exp == 0) return;

    const unsigned int k_blocks = (K + FP8_BLOCK - 1) / FP8_BLOCK;

    const unsigned int warp_id = threadIdx.x / 32;
    const unsigned int lane_id = threadIdx.x % 32;
    const unsigned int warp_m_offset = warp_id * 16;
    const unsigned int group_id = lane_id >> 2;
    const unsigned int tid = lane_id & 3;

    __shared__ __nv_bfloat16 smem_A[M_TILE][K_STEP + PAD];
    __shared__ __nv_bfloat16 smem_B[K_STEP][N_TILE + PAD];
    // 2026-09-25: The token id of each of the CTA's M_TILE rows, resolved once per CTA; -1 past M_expert.




    __shared__ int smem_token_id[M_TILE];
    if (threadIdx.x < M_TILE) {
        int m_idx = threadIdx.x;
        if (m_idx + cta_m_local < M_expert) {
            int sorted_idx = m_start + cta_m_local + m_idx;
            smem_token_id[m_idx] = sorted_token_ids ? sorted_token_ids[sorted_idx] : sorted_idx;
        } else {
            smem_token_id[m_idx] = -1;
        }
    }
    __syncthreads();

    float outer_acc[8][4];
    float inner_acc[8][4];
    #pragma unroll
    for (int i = 0; i < 8; i++) {
        outer_acc[i][0] = 0.0f; outer_acc[i][1] = 0.0f;
        outer_acc[i][2] = 0.0f; outer_acc[i][3] = 0.0f;
        inner_acc[i][0] = 0.0f; inner_acc[i][1] = 0.0f;
        inner_acc[i][2] = 0.0f; inner_acc[i][3] = 0.0f;
    }

    const unsigned int n_block = cta_n / FP8_BLOCK;

    for (unsigned int k_base = 0; k_base < K; k_base += K_STEP) {
        // 2026-09-25: Gather the A tile through the rows' token ids and convert it to BF16; rows past M_expert are zero.
        {
            #pragma unroll
            for (unsigned int i = 0; i < 8; i++) {
                unsigned int idx = threadIdx.x * 8 + i;
                unsigned int row = idx / K_STEP;
                unsigned int col = idx % K_STEP;
                unsigned int m_idx = cta_m_local + row;
                unsigned int gc = k_base + col;

                if (m_idx < (unsigned int)M_expert && gc < K) {
                    int token_id = smem_token_id[row];
                    if (token_id >= 0) {
                        unsigned char a_byte = A_fp8[(unsigned long long)token_id * K + gc];
                        smem_A[row][col] = e4m3_to_bf16_w8a8(a_byte);
                    } else {
                        smem_A[row][col] = __float2bfloat16(0.0f);
                    }
                } else {
                    smem_A[row][col] = __float2bfloat16(0.0f);
                }
            }
        }

        // 2026-09-25: Threads below N_TILE load row n's K_STEP bytes as one uint4 when the whole step is in range, and
        // byte by byte otherwise.



        if (threadIdx.x < N_TILE) {
            unsigned int n = threadIdx.x;
            unsigned int gn = cta_n + n;
            if (gn < N && k_base + K_STEP <= K) {
                uint4 v = *(const uint4*)&B_exp[(unsigned long long)gn * K + k_base];
                const unsigned char* b = (const unsigned char*)&v;
                #pragma unroll
                for (int k = 0; k < 16; k++)
                    smem_B[k][n] = e4m3_to_bf16_w8a8(b[k]);
            } else {
                for (int k = 0; k < K_STEP; k++) {
                    unsigned int gk = k_base + k;
                    smem_B[k][n] = (gk < K && gn < N)
                        ? e4m3_to_bf16_w8a8(B_exp[(unsigned long long)gn * K + gk])
                        : __float2bfloat16(0.0f);
                }
            }
        }

        __syncthreads();
        fp8_w8a8_mma(smem_A, smem_B, inner_acc, warp_m_offset, group_id, tid);
        __syncthreads();

        unsigned int next_k = k_base + K_STEP;
        if (next_k % K_PROMOTE == 0 || next_k >= K) {
            unsigned int k_block = k_base / FP8_BLOCK;
            const float bs = S_exp[n_block * k_blocks + k_block];
            // 2026-09-25: In the m16n8k16 accumulator layout acc[][0..1] belong to row r0 and acc[][2..3] to row r1 = r0 + 8.


            unsigned int r0 = warp_m_offset + group_id;
            unsigned int r1 = r0 + 8;
            int t0 = (r0 < M_TILE) ? smem_token_id[r0] : -1;
            int t1 = (r1 < M_TILE) ? smem_token_id[r1] : -1;
            const float as0 = (t0 >= 0)
                ? a_scale[(unsigned long long)t0 * k_blocks + k_block]
                : 0.0f;
            const float as1 = (t1 >= 0)
                ? a_scale[(unsigned long long)t1 * k_blocks + k_block]
                : 0.0f;
            const float s0 = as0 * bs;
            const float s1 = as1 * bs;
            #pragma unroll
            for (int n_tile = 0; n_tile < 8; n_tile++) {
                outer_acc[n_tile][0] += inner_acc[n_tile][0] * s0;
                outer_acc[n_tile][1] += inner_acc[n_tile][1] * s0;
                outer_acc[n_tile][2] += inner_acc[n_tile][2] * s1;
                outer_acc[n_tile][3] += inner_acc[n_tile][3] * s1;
                inner_acc[n_tile][0] = 0.0f; inner_acc[n_tile][1] = 0.0f;
                inner_acc[n_tile][2] = 0.0f; inner_acc[n_tile][3] = 0.0f;
            }
        }
    }

    #pragma unroll
    for (int n_tile = 0; n_tile < 8; n_tile++) {
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

// 2026-09-25: `moe_w8a8_grouped_gemm_pm4`: the same computation with M_TILE 128, N_TILE 64, 256 threads, and K_STEP 32
// as two m16n8k16 MMAs in ascending K. Raw FP8 A and B are staged with cp.async.cg in a 2-stage pipeline and
// converted to BF16 in shared memory; smem_B is [n][k].
//
// It walks a worklist instead of a 3-D grid: item w is (expert_id, mt << 6 | nt), written with its count
// *total_tiles by `moe_build_tile_worklist` (moe_permute.cu), which must run earlier on the same stream. CTAs
// stride over the items, so any grid size covers them.
//
// The MMAs and the scale folds happen in the same K order as in `moe_w8a8_grouped_gemm`.
//
// Shared memory: 2 stages x (A 128 x 40 BF16 + A_raw 128 x 32 B + B 64 x 34 BF16 + B_raw 64 x 32 B) = 41472 B.
























#define W8PM4_M_TILE 128
#define W8PM4_N_TILE 64
#define W8PM4_K_STEP 32
#define W8PM4_K_SUB 16
#define W8PM4_K_SUBS (W8PM4_K_STEP / W8PM4_K_SUB)
#define W8PM4_PAD 2
#define W8PM4_A_STRIDE (W8PM4_K_STEP + 8)
#define W8PM4_THREADS 256
#define W8PM4_NT_PER_WARP (W8PM4_N_TILE / 8)
#define W8PM4_STAGES 2
#define W8PM4_K_PROMOTE 64
#define W8PM4_FP8_BLOCK 128

__device__ __forceinline__ __nv_bfloat16 w8pm4_e4m3_to_bf16(unsigned char b) {
    // 2026-09-25: The same decode as e4m3_to_bf16_w8a8.


    float f = __half2float(__ushort_as_half((unsigned short)((b & 0x7f) << 7))) * 256.0f;
    f = ((b & 0x7f) == 0x7f) ? 0.0f : f;
    return __float2bfloat16((b & 0x80) ? -f : f);
}

__device__ __forceinline__ void w8pm4_cp_async_cg_16(void* smem_ptr, const void* gmem_ptr) {
    unsigned int s = (unsigned int)__cvta_generic_to_shared(smem_ptr);
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" ::"r"(s), "l"(gmem_ptr));
}
__device__ __forceinline__ void w8pm4_cp_async_commit() {
    asm volatile("cp.async.commit_group;\n" ::);
}
template <int N>
__device__ __forceinline__ void w8pm4_cp_async_wait_group() {
    asm volatile("cp.async.wait_group %0;\n" ::"n"(N));
}
__device__ __forceinline__ void w8pm4_cp_async_wait_le(unsigned int n) {
    switch (n) {
        case 0:  w8pm4_cp_async_wait_group<0>(); break;
        case 1:  w8pm4_cp_async_wait_group<1>(); break;
        default: w8pm4_cp_async_wait_group<2>(); break;
    }
}

// 2026-09-25: The MMAs for one resident K_STEP: two m16n8k16 in ascending K. smem_B is [n][k], so each
// (k, k + 1) B-fragment pair is one aligned u32.

__device__ __forceinline__ void w8pm4_mma_kstep(
    const __nv_bfloat16* smem_A,   // 2026-09-25: [W8PM4_M_TILE][W8PM4_A_STRIDE]
    const __nv_bfloat16* smem_B,   // 2026-09-25: [W8PM4_N_TILE][W8PM4_K_STEP + W8PM4_PAD]
    float inner[W8PM4_NT_PER_WARP][4],
    unsigned int warp_m_offset, unsigned int group_id, unsigned int tid
) {
    const unsigned int a_stride = W8PM4_A_STRIDE;
    const unsigned int b_stride = W8PM4_K_STEP + W8PM4_PAD;
    const unsigned short* sA = (const unsigned short*)smem_A;
    const unsigned short* sB = (const unsigned short*)smem_B;

    unsigned int frag_r0 = warp_m_offset + group_id;
    unsigned int frag_r1 = warp_m_offset + group_id + 8;

    #pragma unroll
    for (int s = 0; s < W8PM4_K_SUBS; s++) {
        const unsigned int k_off = s * W8PM4_K_SUB;
        unsigned int frag_c0 = k_off + tid * 2;
        unsigned int frag_c1 = k_off + tid * 2 + 8;

        unsigned int a0 = *(const unsigned int*)&sA[frag_r0 * a_stride + frag_c0];
        unsigned int a1 = *(const unsigned int*)&sA[frag_r1 * a_stride + frag_c0];
        unsigned int a2 = *(const unsigned int*)&sA[frag_r0 * a_stride + frag_c1];
        unsigned int a3 = *(const unsigned int*)&sA[frag_r1 * a_stride + frag_c1];

        #pragma unroll
        for (int n_tile = 0; n_tile < W8PM4_NT_PER_WARP; n_tile++) {
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

extern "C" __global__ void __launch_bounds__(W8PM4_THREADS, 2) moe_w8a8_grouped_gemm_pm4(
    const unsigned char* __restrict__ A_fp8,                // 2026-09-25: [total_tokens, K] FP8 E4M3
    const float* __restrict__ a_scale,                      // 2026-09-25: [total_tokens, ceil(K / 128)] F32
    const unsigned long long* __restrict__ B_weight_ptrs,   // 2026-09-25: [num_experts] -> [N, K] FP8
    const unsigned long long* __restrict__ B_scale_ptrs,    // 2026-09-25: [num_experts] -> [ceil(N / 128), ceil(K / 128)] F32
    __nv_bfloat16* __restrict__ C,                          // 2026-09-25: [total_expanded, N] BF16
    const int* __restrict__ expert_offsets,                 // 2026-09-25: [num_experts + 1]
    const int* __restrict__ sorted_token_ids,               // 2026-09-25: [total_expanded] or NULL
    unsigned int num_experts,
    unsigned int N,
    unsigned int K,
    const unsigned int* __restrict__ worklist,              // 2026-09-25: [*total_tiles * 2]
    const int* __restrict__ total_tiles
) {
    __shared__ __align__(16) __nv_bfloat16 smem_A[W8PM4_STAGES][W8PM4_M_TILE][W8PM4_A_STRIDE];
    __shared__ __align__(16) unsigned char smem_Araw[W8PM4_STAGES][W8PM4_M_TILE][W8PM4_K_STEP];
    __shared__ __nv_bfloat16 smem_B[W8PM4_STAGES][W8PM4_N_TILE][W8PM4_K_STEP + W8PM4_PAD];
    __shared__ __align__(16) unsigned char smem_Braw[W8PM4_STAGES][W8PM4_N_TILE][W8PM4_K_STEP];

    const unsigned int warp_id = threadIdx.x / 32;
    const unsigned int lane_id = threadIdx.x % 32;
    const unsigned int warp_m_offset = warp_id * 16;
    const unsigned int group_id = lane_id >> 2;
    const unsigned int tid = lane_id & 3;

    const int total = *total_tiles;

    for (int wid = blockIdx.x; wid < total; wid += (int)gridDim.x) {
        __syncthreads();   // 2026-09-25: the previous tile is done with shared memory

        unsigned int expert_id = worklist[wid * 2 + 0];
        unsigned int packed    = worklist[wid * 2 + 1];
        unsigned int mt = packed >> 6;
        unsigned int nt = packed & 0x3F;

        const int m_start = expert_offsets[expert_id];
        const int M_expert = expert_offsets[expert_id + 1] - m_start;

        const unsigned char* B_exp = (const unsigned char*)B_weight_ptrs[expert_id];
        const float* S_exp = (const float*)B_scale_ptrs[expert_id];
        if (B_exp == 0) continue;

        const unsigned int cta_m_local = mt * W8PM4_M_TILE;
        const unsigned int cta_n = nt * W8PM4_N_TILE;

        float inner_acc[W8PM4_NT_PER_WARP][4];
        float outer_acc[W8PM4_NT_PER_WARP][4];
        #pragma unroll
        for (int i = 0; i < W8PM4_NT_PER_WARP; i++) {
            inner_acc[i][0] = 0.0f; inner_acc[i][1] = 0.0f;
            inner_acc[i][2] = 0.0f; inner_acc[i][3] = 0.0f;
            outer_acc[i][0] = 0.0f; outer_acc[i][1] = 0.0f;
            outer_acc[i][2] = 0.0f; outer_acc[i][3] = 0.0f;
        }

        const unsigned int k_blocks = (K + W8PM4_FP8_BLOCK - 1) / W8PM4_FP8_BLOCK;
        const unsigned int n_block = cta_n / W8PM4_FP8_BLOCK;
        const unsigned int n_steps = (K + W8PM4_K_STEP - 1) / W8PM4_K_STEP;
        const unsigned int steps_per_promote = W8PM4_K_PROMOTE / W8PM4_K_STEP;

        // 2026-09-25: The token ids of this warp's two fragment rows, resolved once per tile for the a_scale folds.

        const unsigned int r0 = cta_m_local + warp_m_offset + group_id;
        const unsigned int r1 = r0 + 8;
        const int t0 = (r0 < (unsigned int)M_expert)
            ? (sorted_token_ids ? sorted_token_ids[m_start + (int)r0] : m_start + (int)r0) : -1;
        const int t1 = (r1 < (unsigned int)M_expert)
            ? (sorted_token_ids ? sorted_token_ids[m_start + (int)r1] : m_start + (int)r1) : -1;

        auto prefetch = [&](unsigned int step, unsigned int stage) {
            unsigned int k_base = step * W8PM4_K_STEP;

            // 2026-09-25: A: M_TILE rows of K_STEP FP8 bytes in 16-byte chunks, each row read through its token id.
            const unsigned int a_chunks = (W8PM4_M_TILE * W8PM4_K_STEP) / 16;
            #pragma unroll
            for (unsigned int c = threadIdx.x; c < a_chunks; c += W8PM4_THREADS) {
                unsigned int row  = c / (W8PM4_K_STEP / 16);
                unsigned int kcol = (c % (W8PM4_K_STEP / 16)) * 16;
                unsigned int m_global = cta_m_local + row;
                unsigned int gk = k_base + kcol;
                unsigned char* dst = &smem_Araw[stage][row][kcol];
                if (m_global < (unsigned int)M_expert && gk + 16 <= K) {
                    int sorted_idx = m_start + (int)m_global;
                    int token_id = sorted_token_ids ? sorted_token_ids[sorted_idx] : sorted_idx;
                    w8pm4_cp_async_cg_16(dst, &A_fp8[(unsigned long long)token_id * K + gk]);
                } else {
                    #pragma unroll
                    for (unsigned int e = 0; e < 16; e++) {
                        unsigned int gke = gk + e;
                        if (m_global < (unsigned int)M_expert && gke < K) {
                            int sorted_idx = m_start + (int)m_global;
                            int token_id = sorted_token_ids ? sorted_token_ids[sorted_idx] : sorted_idx;
                            dst[e] = A_fp8[(unsigned long long)token_id * K + gke];
                        } else {
                            dst[e] = 0;
                        }
                    }
                }
            }

            // 2026-09-25: B: N_TILE rows of K_STEP FP8 bytes in 16-byte chunks.
            const unsigned int b_chunks = (W8PM4_N_TILE * W8PM4_K_STEP) / 16;
            #pragma unroll
            for (unsigned int c = threadIdx.x; c < b_chunks; c += W8PM4_THREADS) {
                unsigned int nrow = (c * 16) / W8PM4_K_STEP;
                unsigned int kcol = (c * 16) % W8PM4_K_STEP;
                unsigned int gn = cta_n + nrow;
                unsigned int gk = k_base + kcol;
                unsigned char* dst = &smem_Braw[stage][nrow][kcol];
                if (gn < N && gk + 16 <= K) {
                    w8pm4_cp_async_cg_16(dst, &B_exp[(unsigned long long)gn * K + gk]);
                } else {
                    #pragma unroll
                    for (unsigned int e = 0; e < 16; e++) {
                        unsigned int gke = gk + e;
                        dst[e] = (gn < N && gke < K) ? B_exp[(unsigned long long)gn * K + gke] : 0;
                    }
                }
            }
            w8pm4_cp_async_commit();
        };

        // 2026-09-25: Convert the arrived raw A and B of `stage` to BF16; the scales are applied after the MMAs.

        auto dequant = [&](unsigned int stage) {
            #pragma unroll
            for (unsigned int idx = threadIdx.x; idx < W8PM4_M_TILE * W8PM4_K_STEP; idx += W8PM4_THREADS) {
                unsigned int row = idx / W8PM4_K_STEP;
                unsigned int k   = idx % W8PM4_K_STEP;
                smem_A[stage][row][k] = w8pm4_e4m3_to_bf16(smem_Araw[stage][row][k]);
            }
            #pragma unroll
            for (unsigned int idx = threadIdx.x; idx < W8PM4_N_TILE * W8PM4_K_STEP; idx += W8PM4_THREADS) {
                unsigned int n = idx / W8PM4_K_STEP;
                unsigned int k = idx % W8PM4_K_STEP;
                smem_B[stage][n][k] = w8pm4_e4m3_to_bf16(smem_Braw[stage][n][k]);
            }
        };

        #pragma unroll
        for (unsigned int p = 0; p < W8PM4_STAGES - 1; p++) {
            if (p < n_steps) prefetch(p, p % W8PM4_STAGES);
        }
        unsigned int k_step_in_prom = 0;

        for (unsigned int step = 0; step < n_steps; step++) {
            unsigned int cur = step % W8PM4_STAGES;

            unsigned int ahead = step + (W8PM4_STAGES - 1);
            if (ahead < n_steps) prefetch(ahead, ahead % W8PM4_STAGES);
            unsigned int committed = min(n_steps, W8PM4_STAGES + step);
            unsigned int target = committed - (step + 1);
            w8pm4_cp_async_wait_le(target);
            __syncthreads();   // 2026-09-25: raw A and B of `cur` visible to all threads

            dequant(cur);
            __syncthreads();   // 2026-09-25: smem_A/B[cur] written before the MMAs read them

            w8pm4_mma_kstep(&smem_A[cur][0][0], &smem_B[cur][0][0],
                         inner_acc, warp_m_offset, group_id, tid);
            __syncthreads();   // 2026-09-25: the MMAs are done with smem_*[cur]

            // 2026-09-25: Every K_PROMOTE values of K, and after the last step, fold a_scale * b_scale per row and reset inner.
            k_step_in_prom++;
            if (k_step_in_prom == steps_per_promote || step + 1 == n_steps) {
                const unsigned int k_block = (step * W8PM4_K_STEP) / W8PM4_FP8_BLOCK;
                const float bs = S_exp[n_block * k_blocks + k_block];
                const float as0 = (t0 >= 0)
                    ? a_scale[(unsigned long long)t0 * k_blocks + k_block] : 0.0f;
                const float as1 = (t1 >= 0)
                    ? a_scale[(unsigned long long)t1 * k_blocks + k_block] : 0.0f;
                const float s0 = as0 * bs;
                const float s1 = as1 * bs;
                #pragma unroll
                for (int i = 0; i < W8PM4_NT_PER_WARP; i++) {
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
        for (int n_tile = 0; n_tile < W8PM4_NT_PER_WARP; n_tile++) {
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
