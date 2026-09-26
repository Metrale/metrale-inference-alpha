// SPDX-License-Identifier: MIT OR Apache-2.0

// 2026-09-25: W8A16 tensor-core decode GEMM with a 16-row M tile and FP8 E4M3
// block-scaled weights: C[M,N] = A[M,K] (BF16) * dequant(B[N,K] (FP8 E4M3)),
// 1 <= M <= 16.
//
// Owner: hopper kernels.
// Invariants:
// - Launched with grid (ceil(N / N_TILE), 1, 1) and block (128, 1, 1). The
//   launchers (crates/model-layers/src/layers/ops/w8a16_gemm_m16.rs) check
//   1 <= M <= 16 and K % 128 == 0 (the block-scale granularity); the strided
//   launcher also checks a_row_stride >= K, c_row_stride >= N and
//   a_row_stride % 8 == 0 (16-byte cp.async chunks).
// - Activation rows >= M and weight rows >= N are zero-filled in shared
//   memory, and only [M, N] is stored.
// - Two-level FP32 accumulation: the MMAs accumulate unscaled products into
//   `inner`, and at each 128-K block boundary
//       outer += inner * block_scale[n_block, k_block];  inner = 0
//   so a block scale is applied once per block on an FP32 accumulator,
//   never per element and never in BF16. A CTA's N_TILE columns lie in one
//   128-wide scale block (static_assert), so one scale serves the CTA.
//
// Numerics: reassociated, not bit-identical to `w8a16_gemv` /
// `w8a16_gemv_batch{4,16}`, which walk each output's K in order in one FP32
// accumulator; an m16n8k16 MMA sums 16 K products in the tensor core's own
// order. The contract is `layers::dense_ffn::m16_tc::within_m16_tc_budget`,
// checked by `examples/native_fp8_ffn_m16_tc_microtest.rs`.
//
// Callers: the dense-FFN decode tier (`dense_ffn_m16_tc.rs`), on under
// `[defaults] ffn_m16_tc` (false in kernels/hopper/HARDWARE.toml) or
// `METRALE_FFN_M16_TC` / `METRALE_M16_TC`; and the multi-sequence attention
// QKV (`w8a16_gemm_m16_strided`) and o_proj projections, on under
// `[defaults] attn_m16_tc` (true in kernels/hopper/HARDWARE.toml) or
// `METRALE_ATTN_M16_TC` / `METRALE_M16_TC`.
//
// Dequant: `cvt.rn.f16x2.e4m3x2` (sm_89 and later) decodes two E4M3 bytes
// per instruction. FP16 holds every finite E4M3 value exactly and the
// FP16 -> FP32 -> BF16 conversion is exact for 3 mantissa bits, so the
// result equals the `E4M3_LUT` fallback's for every finite byte. The NaN
// codes 0x7F and 0xFF differ: `E4M3_LUT` decodes them to +0 and -0, `cvt`
// to NaN.
//
// Geometry: a CTA is 4 warps (128 threads) covering [16 M x N_TILE N]. Each
// warp owns N_TILE/32 m16n8k16 tiles, with 4 * N_TILE/32 inner and as many
// outer FP32 accumulators. N_TILE is 32 (`w8a16_gemm_m16`,
// `w8a16_gemm_m16_strided`) or 64 (`w8a16_gemm_m16_n64`, selected by
// `METRALE_FFN_M16_TC_NTILE=64`); at 64 each staged A tile feeds twice the
// weight columns and the grid halves.
//
// Pipeline: K_STEP = 64 with a 4-stage cp.async.cg pipeline on the raw FP8
// bytes (N_TILE x 64 B per stage) and on the 16 x 64 BF16 activation slice
// (2 KB per stage).
//
// Shared memory: A 4*16*72*2 = 9,216 B plus Braw 4*N_TILE*80 = 10,240 B at
// N_TILE = 32 (19,456 B) or 20,480 B at N_TILE = 64 (29,696 B), both under
// the 48 KB static limit. Row pitches are padded (A: 64 + 8 BF16 = 144 B;
// B: 64 + 16 = 80 B) so the 16-byte cp.async chunks stay aligned and the
// fragment loads are bank-conflict-free: an A word index is
// row*36 + s*8 + quad, a B word index row*20 + (s*16 + quad*2)/4, and both
// row terms are distinct mod 32 over the 8 fragment rows. Fragments are
// built from plain shared-memory loads, as in `dense_gemm_tc.cu`.
//
// Entry points:
//   `w8a16_gemm_m16`          N_TILE=32, contiguous A [M,K] and C [M,N]
//   `w8a16_gemm_m16_strided`  N_TILE=32, caller-supplied A/C row pitches in
//                             elements
//   `w8a16_gemm_m16_n64`      N_TILE=64, contiguous
































































#include <cuda_bf16.h>
#include <cuda_fp16.h>

#include "e4m3_lut.cuh"   // 2026-09-25: E4M3 -> FP32 table for the pre-sm_89 dequant

#define M16_M_TILE 16
// 2026-09-25: The two instantiated N tiles: 32 and 64. Both must divide M16_FP8_BLOCK
// so a CTA's columns lie inside one 128-wide scale block.

#define M16_N_TILE 32
#define M16_N_TILE_WIDE 64
#define M16_K_STEP 64
#define M16_K_SUB 16
#define M16_K_SUBS (M16_K_STEP / M16_K_SUB)
#define M16_WARPS 4
#define M16_THREADS (M16_WARPS * 32)
#define M16_N_PER_MMA 8
#define M16_STAGES 4
#define M16_FP8_BLOCK 128
#define M16_A_STRIDE 72                             // 2026-09-25: BF16 elements: 64 + 8 pad
#define M16_B_STRIDE 80                             // 2026-09-25: bytes: 64 + 16 pad

// 2026-09-25: cp.async.cg 16-byte (cache-global) copy, shared <- global. Both
// addresses must be 16-byte aligned.
__device__ __forceinline__ void m16_cp_async_cg_16(void* smem_ptr, const void* gmem_ptr) {
    unsigned int s = (unsigned int)__cvta_generic_to_shared(smem_ptr);
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" ::"r"(s), "l"(gmem_ptr));
}
__device__ __forceinline__ void m16_cp_async_commit() {
    asm volatile("cp.async.commit_group;\n" ::);
}
template <int N>
__device__ __forceinline__ void m16_cp_async_wait_group() {
    asm volatile("cp.async.wait_group %0;\n" ::"n"(N));
}

// 2026-09-25: Wait until at most `n` cp.async groups remain in flight. The PTX operand
// must be a compile-time immediate, so dispatch the runtime count (always in
// [0, M16_STAGES-1]) through a switch over its legal values.
__device__ __forceinline__ void m16_cp_async_wait_le(unsigned int n) {
    switch (n) {
        case 0:  m16_cp_async_wait_group<0>(); break;
        case 1:  m16_cp_async_wait_group<1>(); break;
        case 2:  m16_cp_async_wait_group<2>(); break;
        default: m16_cp_async_wait_group<3>(); break;
    }
}

// 2026-09-25: Two E4M3 bytes (low = weight k, high = weight k+1) -> one BF16x2 register
// in the same halves, the m16n8k16 B fragment's packing. See the dequant note
// in the header.

__device__ __forceinline__ unsigned int m16_dequant_pair(unsigned short raw) {
#if !defined(__CUDA_ARCH__) || (__CUDA_ARCH__ >= 890)
    unsigned int h2;
    asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(h2) : "h"(raw));
    float2 f = __half22float2(*reinterpret_cast<const __half2*>(&h2));
    __nv_bfloat162 b = __floats2bfloat162_rn(f.x, f.y);
    return *reinterpret_cast<const unsigned int*>(&b);
#else
    __nv_bfloat162 b = __floats2bfloat162_rn(E4M3_LUT[raw & 0xFFu], E4M3_LUT[raw >> 8]);
    return *reinterpret_cast<const unsigned int*>(&b);
#endif
}

/// 2026-09-25: W8A16 tensor-core decode GEMM body. `a_row_stride` / `c_row_stride` are
/// the A and C row pitches in elements; the contiguous entry points pass K and N.
///
/// `N_TILE` is the CTA's N width, 32 or 64. It must be a multiple of 32, so
/// each warp owns whole 8-wide MMA tiles, and a divisor of the 128-wide FP8
/// scale block, so `n_block` is constant for the CTA.
template <int N_TILE>
__device__ __forceinline__ void w8a16_gemm_m16_impl(
    const __nv_bfloat16* __restrict__ A,     // 2026-09-25: [M, a_row_stride] BF16, K used
    const unsigned char* __restrict__ B,      // 2026-09-25: [N, K] FP8 E4M3
    const float* __restrict__ block_scale,    // 2026-09-25: [N/128, K/128] FP32
    __nv_bfloat16* __restrict__ C,            // 2026-09-25: [M, c_row_stride] BF16, N used
    unsigned int M,
    unsigned int N,
    unsigned int K,
    unsigned int a_row_stride,
    unsigned int c_row_stride
) {
    static_assert(N_TILE % 32 == 0, "N_TILE must be a whole number of MMA N-tiles per warp");
    static_assert(M16_FP8_BLOCK % N_TILE == 0, "a CTA's columns must lie in ONE scale block");
    // 2026-09-25: MMA N-tiles per warp and 16-byte B chunks per thread per stage: 1 each
    // at N_TILE=32, 2 each at 64.

    constexpr int N_PER_WARP = N_TILE / M16_WARPS;
    constexpr int N_SUBS = N_PER_WARP / M16_N_PER_MMA;
    constexpr int B_CHUNKS = N_TILE / 32;

    const unsigned int cta_n = blockIdx.x * N_TILE;
    const unsigned int warp_id = threadIdx.x >> 5;
    const unsigned int lane_id = threadIdx.x & 31;
    const unsigned int group_id = lane_id >> 2;
    const unsigned int quad = lane_id & 3;
    const unsigned int warp_n = warp_id * N_PER_WARP;

    __shared__ __align__(16) __nv_bfloat16 smem_A[M16_STAGES][M16_M_TILE][M16_A_STRIDE];
    __shared__ __align__(16) unsigned char smem_Braw[M16_STAGES][N_TILE][M16_B_STRIDE];

    // 2026-09-25: Two-level FP32 accumulation (see the header): 4 * N_SUBS inner and
    // 4 * N_SUBS outer registers.
    float inner[4 * N_SUBS];
    float outer[4 * N_SUBS];
    #pragma unroll
    for (int i = 0; i < 4 * N_SUBS; i++) {
        inner[i] = 0.0f;
        outer[i] = 0.0f;
    }

    const unsigned int n_steps = K / M16_K_STEP;
    const unsigned int k_blocks = K / M16_FP8_BLOCK;
    const unsigned int k_steps_per_block = M16_FP8_BLOCK / M16_K_STEP;
    // 2026-09-25: The CTA's N_TILE columns start at a multiple of N_TILE, which divides
    // 128, so they lie inside one 128-wide scale block and n_block is constant.
    const unsigned int n_block = cta_n / M16_FP8_BLOCK;

    // 2026-09-25: Stage one K-step into `stage`. The A tile is 2 KB = 128 chunks of 16 B,
    // one cp.async per thread; the B tile is 2 KB per 32 N rows, so B_CHUNKS per
    // thread. The copies run along K, the contiguous global axis for A [M,K]
    // and B [N,K] alike. Out-of-range rows (activation rows >= M, weight rows
    // >= N) are zero-filled by hand, because cp.async cannot predicate, and
    // zero weights and activations contribute nothing to the MMA, which is
    // what makes any 1 <= M <= 16 legal.
    auto prefetch = [&](unsigned int step, unsigned int stage) {
        const unsigned int k_base = step * M16_K_STEP;
        {
            const unsigned int row = threadIdx.x >> 3;
            const unsigned int col = (threadIdx.x & 7) * 8;
            __nv_bfloat16* dst = &smem_A[stage][row][col];
            if (row < M) {
                m16_cp_async_cg_16(dst, &A[(unsigned long long)row * a_row_stride + k_base + col]);
            } else {
                #pragma unroll
                for (int e = 0; e < 8; e++) dst[e] = __float2bfloat16(0.0f);
            }
        }
        #pragma unroll
        for (int c = 0; c < B_CHUNKS; c++) {
            const unsigned int row = (threadIdx.x >> 2) + c * 32;
            const unsigned int col = (threadIdx.x & 3) * 16;
            const unsigned int gn = cta_n + row;
            unsigned char* dst = &smem_Braw[stage][row][col];
            if (gn < N) {
                m16_cp_async_cg_16(dst, &B[(unsigned long long)gn * K + k_base + col]);
            } else {
                #pragma unroll
                for (int e = 0; e < 16; e++) dst[e] = 0;
            }
        }
        m16_cp_async_commit();
    };

    #pragma unroll
    for (unsigned int p = 0; p < M16_STAGES - 1; p++) {
        if (p < n_steps) prefetch(p, p);
    }

    unsigned int k_step_in_block = 0;
    for (unsigned int step = 0; step < n_steps; step++) {
        const unsigned int cur = step % M16_STAGES;
        // 2026-09-25: Groups complete FIFO. Before this iteration issues its own prefetch,
        // min(n_steps, STAGES-1+step) groups are committed and `cur` is the
        // step-th, so the number that may stay in flight is:
        const unsigned int committed = min(n_steps, M16_STAGES - 1 + step);
        m16_cp_async_wait_le(committed - (step + 1));
        // 2026-09-25: One barrier per K-step. It makes stage `cur` visible to every warp,
        // and it follows every warp's MMA of step-1, which is what makes the
        // prefetch below (it targets stage (step-1) % STAGES) safe without a
        // second barrier.
        __syncthreads();
        const unsigned int ahead = step + M16_STAGES - 1;
        if (ahead < n_steps) prefetch(ahead, ahead % M16_STAGES);

        const unsigned short* sA = (const unsigned short*)&smem_A[cur][0][0];
        const unsigned char* sB = &smem_Braw[cur][0][0];
        #pragma unroll
        for (int s = 0; s < M16_K_SUBS; s++) {
            // 2026-09-25: m16n8k16 row.col fragments, built straight from shared memory: A
            // rows {group_id, group_id+8} x K pairs {quad*2, quad*2+8}; B row
            // (warp_n + j*8 + group_id) at the same K pairs. The A fragment is loaded
            // once and reused by all N_SUBS MMAs.

            const unsigned int kc0 = s * M16_K_SUB + quad * 2;
            const unsigned int kc1 = kc0 + 8;
            const unsigned int r0 = group_id * M16_A_STRIDE;
            const unsigned int r1 = (group_id + 8) * M16_A_STRIDE;
            const unsigned int a0 = *(const unsigned int*)&sA[r0 + kc0];
            const unsigned int a1 = *(const unsigned int*)&sA[r1 + kc0];
            const unsigned int a2 = *(const unsigned int*)&sA[r0 + kc1];
            const unsigned int a3 = *(const unsigned int*)&sA[r1 + kc1];
            #pragma unroll
            for (int j = 0; j < N_SUBS; j++) {
                const unsigned char* brow =
                    &sB[(warp_n + j * M16_N_PER_MMA + group_id) * M16_B_STRIDE];
                const unsigned int b0 = m16_dequant_pair(*(const unsigned short*)&brow[kc0]);
                const unsigned int b1 = m16_dequant_pair(*(const unsigned short*)&brow[kc1]);
                // 2026-09-25: Index `inner` directly: `j` is a compile-time constant under the
                // unroll, so the accumulators can stay in registers.


                asm volatile(
                    "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
                    "{%0, %1, %2, %3}, "
                    "{%4, %5, %6, %7}, "
                    "{%8, %9}, "
                    "{%10, %11, %12, %13};"
                    : "=f"(inner[j * 4 + 0]), "=f"(inner[j * 4 + 1]),
                      "=f"(inner[j * 4 + 2]), "=f"(inner[j * 4 + 3])
                    : "r"(a0), "r"(a1), "r"(a2), "r"(a3),
                      "r"(b0), "r"(b1),
                      "f"(inner[j * 4 + 0]), "f"(inner[j * 4 + 1]),
                      "f"(inner[j * 4 + 2]), "f"(inner[j * 4 + 3])
                );
            }
        }

        // 2026-09-25: 128-K block boundary: fold the unscaled inner accumulator onto the
        // outer one once, with this block's scale. One scale for the whole CTA:
        // every N_SUBS tile is inside the same 128-wide block.
        if (++k_step_in_block == k_steps_per_block) {
            const float scale = block_scale[n_block * k_blocks + step / k_steps_per_block];
            #pragma unroll
            for (int i = 0; i < 4 * N_SUBS; i++) {
                outer[i] += inner[i] * scale;
                inner[i] = 0.0f;
            }
            k_step_in_block = 0;
        }
    }

    // 2026-09-25: Store: FP32 outer accumulators -> BF16, masked to [M, N].
    const unsigned int row0 = group_id;
    const unsigned int row1 = group_id + 8;
    const unsigned long long o0 = (unsigned long long)row0 * c_row_stride;
    const unsigned long long o1 = (unsigned long long)row1 * c_row_stride;
    #pragma unroll
    for (int j = 0; j < N_SUBS; j++) {
        const unsigned int col0 = cta_n + warp_n + j * M16_N_PER_MMA + quad * 2;
        const unsigned int col1 = col0 + 1;
        if (row0 < M && col0 < N) C[o0 + col0] = __float2bfloat16(outer[j * 4 + 0]);
        if (row0 < M && col1 < N) C[o0 + col1] = __float2bfloat16(outer[j * 4 + 1]);
        if (row1 < M && col0 < N) C[o1 + col0] = __float2bfloat16(outer[j * 4 + 2]);
        if (row1 < M && col1 < N) C[o1 + col1] = __float2bfloat16(outer[j * 4 + 3]);
    }
}

/// 2026-09-25: Contiguous A `[M, K]` and C `[M, N]`, N_TILE = 32.
extern "C" __global__
__launch_bounds__(M16_THREADS, 4)
void w8a16_gemm_m16(
    const __nv_bfloat16* __restrict__ A,
    const unsigned char* __restrict__ B,
    const float* __restrict__ block_scale,
    __nv_bfloat16* __restrict__ C,
    unsigned int M,
    unsigned int N,
    unsigned int K
) {
    w8a16_gemm_m16_impl<M16_N_TILE>(A, B, block_scale, C, M, N, K, K, N);
}

/// 2026-09-25: N_TILE=64 twin of [`w8a16_gemm_m16`]: the same arguments and the same
/// per-output arithmetic (the two-level fold and the m16n8k16 K order are
/// unchanged; only which CTA owns a column changes), half the CTAs, and each
/// staged A tile feeds twice the weight bytes. Grid is `ceil(N/64)`.
/// Selected by `METRALE_FFN_M16_TC_NTILE=64`.






extern "C" __global__
__launch_bounds__(M16_THREADS, 4)
void w8a16_gemm_m16_n64(
    const __nv_bfloat16* __restrict__ A,
    const unsigned char* __restrict__ B,
    const float* __restrict__ block_scale,
    __nv_bfloat16* __restrict__ C,
    unsigned int M,
    unsigned int N,
    unsigned int K
) {
    w8a16_gemm_m16_impl<M16_N_TILE_WIDE>(A, B, block_scale, C, M, N, K, K, N);
}

/// 2026-09-25: Caller-supplied A and C row pitches in elements, with the math and
/// accumulation order of `w8a16_gemm_m16`. The multi-sequence decode QKV
/// projection (`qkv_fp8_batch.rs`) launches it.


extern "C" __global__
__launch_bounds__(M16_THREADS, 4)
void w8a16_gemm_m16_strided(
    const __nv_bfloat16* __restrict__ A,
    const unsigned char* __restrict__ B,
    const float* __restrict__ block_scale,
    __nv_bfloat16* __restrict__ C,
    unsigned int M,
    unsigned int N,
    unsigned int K,
    unsigned int a_row_stride,
    unsigned int c_row_stride
) {
    w8a16_gemm_m16_impl<M16_N_TILE>(A, B, block_scale, C, M, N, K, a_row_stride, c_row_stride);
}
