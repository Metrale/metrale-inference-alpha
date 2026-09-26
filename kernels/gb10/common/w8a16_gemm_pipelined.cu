// SPDX-License-Identifier: MIT OR Apache-2.0

// 2026-09-25: W8A16 GEMM for M > 1 with a cp.async pipeline: FP8 E4M3 weights with 128 x 128
// FP32 block scales, BF16 activations, BF16 tensor-core MMA (`mma.sync.m16n8k16`, FP32
// accumulate). The arguments and block-scale layout are w8a16_gemm's.
//   C[m, n] = sum_k A[m, k] * E4M3(B[n, k]) * block_scale[n / 128, k / 128]
//
// Owner: gb10 kernels.
// Invariants:
// - A [M, K] BF16, B [N, K] E4M3 bytes, C [M, N] BF16. block_scale is read with a row pitch of
//   K / 128 rounded down, so K % 128 == 0 is assumed; with it, and 16-byte-aligned A and B, every
//   16-byte cp.async source is aligned.
// - Grid (ceil(N / PM_N_TILE), ceil(M / PM_M_TILE), 1) = (ceil(N / 32), ceil(M / 128), 1), block
//   PM_THREADS = 256 (ops::w8a16_gemm_pipelined). Only C[m, n] with m < M and n < N is written.
//
// Two-level FP32 accumulation: inner_acc sums the MMAs of one 128-wide K block over unscaled
// weights, which E4M3 -> BF16 converts exactly; at each block boundary
//   outer_acc += inner_acc * block_scale[n_block, k_block]; inner_acc = 0
// so the scale is applied to the FP32 sum, never to a BF16 value.
//
// A PM_STAGES-deep cp.async pipeline copies the next K-steps' A tiles and raw E4M3 B bytes,
// contiguous along K, into shared memory while the MMAs consume the current step. Static shared
// memory only. Measured 2026-09-25 (nvcc 13.0.88, sm_121f, --fmad=false): 55 registers, no
// spill, 27,904 B of shared memory at PM_STAGES = 2.















#include <cuda_bf16.h>
#include "e4m3_lut.cuh"

#define PM_M_TILE 128
// 2026-09-25: PM_N_TILE = 32: each warp holds 4 n8 tiles, so the two accumulator arrays are
// 2 x 4 x 4 = 32 floats per thread.









#define PM_N_TILE 32
// 2026-09-25: PM_K_STEP = 32: each step runs PM_K_SUBS = 2 MMAs per n8 tile between its three
// __syncthreads(), half the barriers per K of a 16-wide step.



#define PM_K_STEP 32
#define PM_K_SUB 16
#define PM_K_SUBS (PM_K_STEP / PM_K_SUB)
#define PM_PAD 2
// 2026-09-25: The A-tile row stride is a multiple of 8 BF16 (16 bytes), so every 16-byte cp.async
// destination is aligned: 32 K columns + 8 pad = 40 BF16 = 80 bytes. With 20-word rows the
// 32-bit A fragment reads of a warp fall in 32 distinct banks.



#define PM_A_STRIDE 40
#define PM_FP8_BLOCK 128
#define PM_WARPS 8
#define PM_THREADS (PM_WARPS * 32)
#define PM_N_TILES_PER_WARP (PM_N_TILE / 8)






#define PM_STAGES 2

// 2026-09-25: 16-byte cp.async.cg copy, global -> shared (cached in L2 only). Both addresses
// must be 16-byte aligned.
__device__ __forceinline__ void cp_async_cg_16(void* smem_ptr, const void* gmem_ptr) {
    unsigned int s = (unsigned int)__cvta_generic_to_shared(smem_ptr);
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" ::"r"(s), "l"(gmem_ptr));
}
__device__ __forceinline__ void cp_async_commit() {
    asm volatile("cp.async.commit_group;\n" ::);
}
template <int N>
__device__ __forceinline__ void cp_async_wait_group() {
    asm volatile("cp.async.wait_group %0;\n" ::"n"(N));
}

// 2026-09-25: Wait until at most n cp.async groups are in flight. cp.async.wait_group takes an
// immediate, so the runtime n (0 to PM_STAGES - 1) goes through a switch; an n above 3 waits
// for at most 3, which is stricter than asked.


__device__ __forceinline__ void cp_async_wait_le(unsigned int n) {
    switch (n) {
        case 0:  cp_async_wait_group<0>(); break;
        case 1:  cp_async_wait_group<1>(); break;
        case 2:  cp_async_wait_group<2>(); break;
        default: cp_async_wait_group<3>(); break;
    }
}

// 2026-09-25: The MMAs of one K-step: PM_K_SUBS sub-MMAs of PM_K_SUB = 16 K each per n8 tile of
// the warp, accumulated into inner. smem_A is [PM_M_TILE][PM_A_STRIDE]; smem_B is
// [PM_N_TILE][PM_K_STEP + PM_PAD], K-contiguous, converted once per step by all 256 threads for
// the 8 warps. The A fragments are those of w8a16_gemm's w8a16_mma_and_store, offset by
// s * PM_K_SUB; each B register (k, k + 1) is one 32-bit load.


__device__ __forceinline__ void pm_mma_kstep(
    const __nv_bfloat16* smem_A,
    const __nv_bfloat16* smem_B,
    float inner[PM_N_TILES_PER_WARP][4],
    unsigned int warp_m_offset, unsigned int group_id, unsigned int tid
) {
    const unsigned int a_stride = PM_A_STRIDE;
    const unsigned int b_stride = PM_K_STEP + PM_PAD;
    const unsigned short* sA = (const unsigned short*)smem_A;
    const unsigned short* sB = (const unsigned short*)smem_B;

    unsigned int frag_r0 = warp_m_offset + group_id;
    unsigned int frag_r1 = warp_m_offset + group_id + 8;

    #pragma unroll
    for (int s = 0; s < PM_K_SUBS; s++) {
        const unsigned int k_off = s * PM_K_SUB;

        unsigned int frag_c0 = k_off + tid * 2;
        unsigned int frag_c1 = k_off + tid * 2 + 8;

        unsigned int a0 = *(const unsigned int*)&sA[frag_r0 * a_stride + frag_c0];
        unsigned int a1 = *(const unsigned int*)&sA[frag_r1 * a_stride + frag_c0];
        unsigned int a2 = *(const unsigned int*)&sA[frag_r0 * a_stride + frag_c1];
        unsigned int a3 = *(const unsigned int*)&sA[frag_r1 * a_stride + frag_c1];

        #pragma unroll
        for (int n_tile = 0; n_tile < PM_N_TILES_PER_WARP; n_tile++) {
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



extern "C" __global__ void w8a16_gemm_pipelined(
    const __nv_bfloat16* __restrict__ A,
    const unsigned char* __restrict__ B,
    const float* __restrict__ block_scale,
    __nv_bfloat16* __restrict__ C,
    unsigned int M,
    unsigned int N,
    unsigned int K
) {
    const unsigned int cta_m = blockIdx.y * PM_M_TILE;
    const unsigned int cta_n = blockIdx.x * PM_N_TILE;
    const unsigned int warp_id = threadIdx.x / 32;
    const unsigned int lane_id = threadIdx.x % 32;
    const unsigned int warp_m_offset = warp_id * 16;
    const unsigned int group_id = lane_id >> 2;
    const unsigned int tid = lane_id & 3;

    // 2026-09-25: The cp.async destinations smem_A and smem_Braw are __align__(16) with 16-byte
    // row strides. smem_B holds the step's weights converted to BF16 once by all 256 threads and
    // read by all 8 warps.






    __shared__ __align__(16) __nv_bfloat16 smem_A[PM_STAGES][PM_M_TILE][PM_A_STRIDE];
    // 2026-09-25: smem_B is [n][k], K-contiguous like smem_Braw. Rows are PM_K_STEP + PM_PAD = 34
    // BF16 (68 bytes), so each MMA register pair (k, k + 1), k even, is one 4-byte-aligned load.




    __shared__ __nv_bfloat16 smem_B[PM_STAGES][PM_N_TILE][PM_K_STEP + PM_PAD];
    __shared__ __align__(16) unsigned char smem_Braw[PM_STAGES][PM_N_TILE][PM_K_STEP];

    // 2026-09-25: Copy the 256-entry E4M3 table to shared memory, one element per thread
    // (PM_THREADS = 256). The dequant indexes it with data-dependent weight bytes: constant memory
    // serializes a warp's distinct addresses, while shared memory serves them from their banks.





    __shared__ float smem_lut[256];
    smem_lut[threadIdx.x] = E4M3_LUT[threadIdx.x];
    __syncthreads();

    // 2026-09-25: Two-level FP32 accumulation, as described in the file header.
    float inner_acc[PM_N_TILES_PER_WARP][4];
    float outer_acc[PM_N_TILES_PER_WARP][4];
    #pragma unroll
    for (int i = 0; i < PM_N_TILES_PER_WARP; i++) {
        inner_acc[i][0] = 0.0f; inner_acc[i][1] = 0.0f;
        inner_acc[i][2] = 0.0f; inner_acc[i][3] = 0.0f;
        outer_acc[i][0] = 0.0f; outer_acc[i][1] = 0.0f;
        outer_acc[i][2] = 0.0f; outer_acc[i][3] = 0.0f;
    }

    const unsigned int k_blocks = K / PM_FP8_BLOCK;
    const unsigned int k_steps_per_block = PM_FP8_BLOCK / PM_K_STEP;
    const unsigned int n_block = cta_n / PM_FP8_BLOCK;
    const unsigned int n_steps = (K + PM_K_STEP - 1) / PM_K_STEP;

    // 2026-09-25: A tile: 128 rows x PM_K_STEP BF16 in 16-byte chunks of 8, 512 chunks, 2 per
    // thread.
    const unsigned int a_chunks = (PM_M_TILE * PM_K_STEP) / 8;

    // 2026-09-25: Issue the cp.async copies of K-step `step` into buffer `stage`. A [M, K] and
    // B [N, K] are copied along K, their contiguous axis. A chunk that crosses M, N or K is copied
    // element by element with zero fill instead.

    auto prefetch = [&](unsigned int step, unsigned int stage) {
        unsigned int k_base = step * PM_K_STEP;


        #pragma unroll
        for (unsigned int c = threadIdx.x; c < a_chunks; c += PM_THREADS) {
            unsigned int row = (c * 8) / PM_K_STEP;
            unsigned int col = (c * 8) % PM_K_STEP;
            unsigned int gr = cta_m + row;
            unsigned int gc = k_base + col;
            __nv_bfloat16* dst = &smem_A[stage][row][col];
            if (gr < M && gc + 8 <= K) {
                cp_async_cg_16(dst, &A[(unsigned long long)gr * K + gc]);
            } else {
                #pragma unroll
                for (unsigned int e = 0; e < 8; e++) {
                    unsigned int gcol = gc + e;
                    dst[e] = (gr < M && gcol < K) ? A[(unsigned long long)gr * K + gcol]
                                                  : __float2bfloat16(0.0f);
                }
            }
        }

        // 2026-09-25: B: smem_Braw[stage][n][k] mirrors B[n, k_base + k]; each 32-byte row of the
        // step is two 16-byte chunks, PM_N_TILE x PM_K_STEP / 16 chunks in all.


        const unsigned int b_chunks = (PM_N_TILE * PM_K_STEP) / 16;
        #pragma unroll
        for (unsigned int c = threadIdx.x; c < b_chunks; c += PM_THREADS) {
            unsigned int nrow = (c * 16) / PM_K_STEP;
            unsigned int kcol = (c * 16) % PM_K_STEP;
            unsigned int gn = cta_n + nrow;
            unsigned int gk = k_base + kcol;
            unsigned char* dst = &smem_Braw[stage][nrow][kcol];
            if (gn < N && gk + 16 <= K) {
                cp_async_cg_16(dst, &B[(unsigned long long)gn * K + gk]);
            } else {
                #pragma unroll
                for (unsigned int e = 0; e < 16; e++) {
                    unsigned int gke = gk + e;
                    dst[e] = (gn < N && gke < K) ? B[(unsigned long long)gn * K + gke] : 0;
                }
            }
        }
        cp_async_commit();
    };

    // 2026-09-25: Convert the arrived raw B of `stage` to BF16 through smem_lut, element for
    // element in the same [n][k] layout and with no scale; all 256 threads share the work.






    auto dequant_B = [&](unsigned int stage) {
        #pragma unroll
        for (unsigned int idx = threadIdx.x; idx < PM_K_STEP * PM_N_TILE; idx += PM_THREADS) {
            unsigned int n = idx / PM_K_STEP;
            unsigned int k = idx % PM_K_STEP;
            unsigned char wb = smem_Braw[stage][n][k];
            smem_B[stage][n][k] = __float2bfloat16(smem_lut[wb]);
        }
    };

    // 2026-09-25: Pipelined main loop. The prologue issues the first PM_STAGES - 1 prefetches; each
    // prefetch commits one cp.async group, and the groups complete in commit order.



    #pragma unroll
    for (unsigned int p = 0; p < PM_STAGES - 1; p++) {
        if (p < n_steps) {
            prefetch(p, p % PM_STAGES);
        }
    }
    unsigned int k_step_in_block = 0;

    for (unsigned int step = 0; step < n_steps; step++) {
        unsigned int cur = step % PM_STAGES;

        // 2026-09-25: Prefetch the step PM_STAGES - 1 ahead before consuming this one, so its loads
        // overlap the dequant and MMA. The three barriers below follow the wait (raw B resident),
        // the dequant (smem_B written) and the MMA (buffer free for reuse).
        unsigned int ahead = step + (PM_STAGES - 1);
        if (ahead < n_steps) {
            prefetch(ahead, ahead % PM_STAGES);
        }
        // 2026-09-25: Wait until `cur`, the oldest outstanding group, is complete. After this step's
        // prefetch, min(n_steps, PM_STAGES + step) groups are committed and `cur` is group `step`
        // (0-indexed), so the groups that may stay in flight are
        //     target = min(n_steps, PM_STAGES + step) - (step + 1)
        // which is PM_STAGES - 1 in steady state and falls to 0 at the tail.


        unsigned int committed = min(n_steps, PM_STAGES + step);
        unsigned int target = committed - (step + 1);
        cp_async_wait_le(target);
        __syncthreads();

        dequant_B(cur);
        __syncthreads();

        pm_mma_kstep(&smem_A[cur][0][0], &smem_B[cur][0][0],
                     inner_acc, warp_m_offset, group_id, tid);
        __syncthreads();

        // 2026-09-25: End of a 128-wide K block: add the scaled inner sum to outer_acc, reset inner.
        k_step_in_block++;
        if (k_step_in_block == k_steps_per_block) {
            const unsigned int k_block = (step * PM_K_STEP) / PM_FP8_BLOCK;
            const float scale = block_scale[n_block * k_blocks + k_block];
            #pragma unroll
            for (int i = 0; i < PM_N_TILES_PER_WARP; i++) {
                outer_acc[i][0] += inner_acc[i][0] * scale;
                outer_acc[i][1] += inner_acc[i][1] * scale;
                outer_acc[i][2] += inner_acc[i][2] * scale;
                outer_acc[i][3] += inner_acc[i][3] * scale;
                inner_acc[i][0] = 0.0f; inner_acc[i][1] = 0.0f;
                inner_acc[i][2] = 0.0f; inner_acc[i][3] = 0.0f;
            }
            k_step_in_block = 0;
        }
    }

    // 2026-09-25: Fold a trailing partial K block (K % 128 != 0), with the same rounded-down pitch.
    if (k_step_in_block != 0) {
        const unsigned int k_block = (K - 1) / PM_FP8_BLOCK;
        const float scale = block_scale[n_block * k_blocks + k_block];
        #pragma unroll
        for (int i = 0; i < PM_N_TILES_PER_WARP; i++) {
            outer_acc[i][0] += inner_acc[i][0] * scale;
            outer_acc[i][1] += inner_acc[i][1] * scale;
            outer_acc[i][2] += inner_acc[i][2] * scale;
            outer_acc[i][3] += inner_acc[i][3] * scale;
        }
    }


    #pragma unroll
    for (int n_tile = 0; n_tile < PM_N_TILES_PER_WARP; n_tile++) {
        unsigned int base_n = cta_n + n_tile * 8;
        unsigned int col0 = base_n + (tid * 2);
        unsigned int col1 = col0 + 1;
        unsigned int row0 = cta_m + warp_m_offset + group_id;
        unsigned int row1 = row0 + 8;

        if (row0 < M && col0 < N) C[row0 * N + col0] = __float2bfloat16(outer_acc[n_tile][0]);
        if (row0 < M && col1 < N) C[row0 * N + col1] = __float2bfloat16(outer_acc[n_tile][1]);
        if (row1 < M && col0 < N) C[row1 * N + col0] = __float2bfloat16(outer_acc[n_tile][2]);
        if (row1 < M && col1 < N) C[row1 * N + col1] = __float2bfloat16(outer_acc[n_tile][3]);
    }
}
