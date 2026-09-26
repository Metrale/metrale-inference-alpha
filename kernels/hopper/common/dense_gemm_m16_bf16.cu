// SPDX-License-Identifier: MIT OR Apache-2.0

// 2026-09-25: Dense BF16 tensor-core decode GEMM with a 16-row M tile:
//   C[M,N] = A[M,K] (BF16) * B[N,K]^T (BF16), 1 <= M <= 16.
//
// Owner: hopper kernels.
// Invariants:
// - Launched with grid (ceil(N / N_TILE), 1, 1) and block (128, 1, 1)
//   (`ops::dense_gemm_m16_bf16`), whose checks are 1 <= M <= 16, K % 64 == 0
//   (the pipeline step), a_row_stride >= K, c_row_stride >= N and
//   a_row_stride % 8 == 0 (16-byte cp.async chunks). The weight pitch is K,
//   so K % 64 == 0 also keeps every weight row 16-byte aligned.
// - Activation rows >= M and weight rows >= N are zero-filled in shared
//   memory and only [M, N] is stored, so nothing outside [M, N] is read or
//   written.
// - One FP32 accumulator level: B is BF16, the MMA's input type, so there is
//   no dequant and no block scale.
//
// Numerics: reassociated, not bit-identical to `dense_gemv_bf16` /
// `dense_gemv_bf16_batchm`. An m16n8k16 MMA sums 16 K products in the tensor
// core's own order before they reach the FP32 accumulator. The contract is
// `layers::dense_ffn::m16_tc::within_m16_tc_budget` (within `M16_TC_MAX_ULP` = 2
// ordinal BF16 ULP, or an absolute error under `m16_tc_acc_floor`), the same
// predicate `w8a16_gemm_m16` is held to. The LM head runs this kernel at
// 5..=16 rows when `[defaults] lm_head_m16_tc` (true in
// kernels/hopper/HARDWARE.toml) or `METRALE_LM_HEAD_M16_TC` turns it on
// (`lm_head_m16_tc_route`, crates/model-engine/src/model/trait_impl/lm_head_batched.rs).
//
// Geometry: a CTA is 4 warps (128 threads) covering [16 M x N_TILE N]. Each
// warp owns N_TILE/4 columns, N_TILE/32 m16n8k16 tiles, and 4 * N_TILE/32
// FP32 accumulators. N_TILE is 32 (`dense_gemm_m16_bf16`) or 64
// (`dense_gemm_m16_bf16_n64`, selected by `METRALE_LM_HEAD_M16_TC_NTILE=64`);
// at 64 each staged A tile feeds twice the weight columns and the grid halves.
//
// Pipeline: K_STEP = 64 with a 4-stage cp.async.cg pipeline on the BF16
// weights (N_TILE x 64 x 2 B per stage) and on the 16 x 64 BF16 activation
// slice (2 KB per stage).
//
// Shared memory: A 4*16*72*2 = 9,216 B plus B 4*N_TILE*72*2 = 18,432 B at
// N_TILE = 32 (27,648 B) or 36,864 B at N_TILE = 64 (46,080 B). Both are
// under the 48 KB static per-block limit, so there is no dynamic
// shared-memory opt-in. Row pitches are 72 BF16 = 144 B on both tiles:
// 144 = 9 x 16 keeps the 16-byte cp.async chunks aligned, and the fragment
// word index row*36 + s*8 + quad (row*36 mod 32 = row*4 over the 8 fragment
// rows, quad 0..3) puts the 32 lanes of a warp on 32 distinct banks.
// Fragments are built from plain shared-memory loads, as in
// `dense_gemm_tc.cu` and `w8a16_gemm_m16.cu`.
















































































#include <cuda_bf16.h>

#define DGM16_M_TILE 16
// 2026-09-25: The two instantiated N tiles: 32 (`dense_gemm_m16_bf16`) and 64
// (`dense_gemm_m16_bf16_n64`). Both must be a multiple of 32 so each of the
// 4 warps owns a whole number of 8-wide MMA tiles.
#define DGM16_N_TILE 32
#define DGM16_N_TILE_WIDE 64
#define DGM16_K_STEP 64
#define DGM16_K_SUB 16
#define DGM16_K_SUBS (DGM16_K_STEP / DGM16_K_SUB)
#define DGM16_WARPS 4
#define DGM16_THREADS (DGM16_WARPS * 32)
#define DGM16_N_PER_MMA 8
#define DGM16_STAGES 4
// 2026-09-25: BF16 elements per shared-memory row on both tiles: 64 + 8 pad = 144 B.
// See the bank and alignment note in the header.
#define DGM16_ROW_STRIDE 72
// 2026-09-25: 16-byte cp.async chunks: 8 BF16 each, 8 chunks per 64-wide row, so 128
// threads stage the whole 16-row A tile, or 16 weight rows, per pass.
#define DGM16_ELEMS_PER_CHUNK 8
#define DGM16_ROWS_PER_PASS (DGM16_THREADS / (DGM16_K_STEP / DGM16_ELEMS_PER_CHUNK))

// 2026-09-25: cp.async.cg 16-byte (cache-global) copy, shared <- global. Both
// addresses must be 16-byte aligned.
__device__ __forceinline__ void dgm16_cp_async_cg_16(void* smem_ptr, const void* gmem_ptr) {
    unsigned int s = (unsigned int)__cvta_generic_to_shared(smem_ptr);
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" ::"r"(s), "l"(gmem_ptr));
}
__device__ __forceinline__ void dgm16_cp_async_commit() {
    asm volatile("cp.async.commit_group;\n" ::);
}
template <int N>
__device__ __forceinline__ void dgm16_cp_async_wait_group() {
    asm volatile("cp.async.wait_group %0;\n" ::"n"(N));
}

// 2026-09-25: Wait until at most `n` cp.async groups remain in flight. The PTX operand
// must be a compile-time immediate, so dispatch the runtime count (always in
// [0, DGM16_STAGES-1]) through a switch over its legal values.
__device__ __forceinline__ void dgm16_cp_async_wait_le(unsigned int n) {
    switch (n) {
        case 0:  dgm16_cp_async_wait_group<0>(); break;
        case 1:  dgm16_cp_async_wait_group<1>(); break;
        case 2:  dgm16_cp_async_wait_group<2>(); break;
        default: dgm16_cp_async_wait_group<3>(); break;
    }
}

/// 2026-09-25: Dense BF16 tensor-core decode GEMM body. `a_row_stride` / `c_row_stride`
/// are the A and C row pitches in elements; the weight pitch is K (B is
/// `[N, K]`).
///
/// `N_TILE` is the CTA's N width, 32 or 64. It must be a multiple of 32 so
/// each warp owns whole 8-wide MMA tiles.
template <int N_TILE>
__device__ __forceinline__ void dense_gemm_m16_bf16_impl(
    const __nv_bfloat16* __restrict__ A,     // 2026-09-25: [M, a_row_stride] BF16, K used
    const __nv_bfloat16* __restrict__ B,     // 2026-09-25: [N, K] BF16 weights
    __nv_bfloat16* __restrict__ C,           // 2026-09-25: [M, c_row_stride] BF16, N used
    unsigned int M,
    unsigned int N,
    unsigned int K,
    unsigned int a_row_stride,
    unsigned int c_row_stride
) {
    static_assert(N_TILE % 32 == 0, "N_TILE must be a whole number of MMA N-tiles per warp");
    // 2026-09-25: MMA N-tiles per warp (1 at N_TILE=32, 2 at 64) and 16-byte weight
    // chunks per thread per stage (2 resp. 4).

    constexpr int N_PER_WARP = N_TILE / DGM16_WARPS;
    constexpr int N_SUBS = N_PER_WARP / DGM16_N_PER_MMA;
    constexpr int B_CHUNKS = N_TILE / DGM16_ROWS_PER_PASS;

    const unsigned int cta_n = blockIdx.x * N_TILE;
    const unsigned int warp_id = threadIdx.x >> 5;
    const unsigned int lane_id = threadIdx.x & 31;
    const unsigned int group_id = lane_id >> 2;
    const unsigned int quad = lane_id & 3;
    const unsigned int warp_n = warp_id * N_PER_WARP;

    __shared__ __align__(16) __nv_bfloat16 smem_A[DGM16_STAGES][DGM16_M_TILE][DGM16_ROW_STRIDE];
    __shared__ __align__(16) __nv_bfloat16 smem_B[DGM16_STAGES][N_TILE][DGM16_ROW_STRIDE];

    // 2026-09-25: One level of FP32 accumulation: there is no block scale to fold, so
    // the MMA chain runs uninterrupted from k=0 to k=K.
    float acc[4 * N_SUBS];
    #pragma unroll
    for (int i = 0; i < 4 * N_SUBS; i++) acc[i] = 0.0f;

    const unsigned int n_steps = K / DGM16_K_STEP;

    // 2026-09-25: Stage one K-step into `stage`. The A tile is 2 KB = 128 chunks of 16 B,
    // one cp.async per thread; the weight tile is 2 KB per 16 N rows, so
    // B_CHUNKS per thread. The copies run along K, the contiguous global axis
    // for A [M,K] and B [N,K] alike. Out-of-range rows (activation rows >= M,
    // weight rows >= N) are zero-filled by hand, because cp.async cannot
    // predicate, and zero operands contribute nothing to the MMA, which is what
    // makes any 1 <= M <= 16 and any N legal.
    auto prefetch = [&](unsigned int step, unsigned int stage) {
        const unsigned int k_base = step * DGM16_K_STEP;
        const unsigned int col = (threadIdx.x & 7) * DGM16_ELEMS_PER_CHUNK;
        {
            const unsigned int row = threadIdx.x >> 3;
            __nv_bfloat16* dst = &smem_A[stage][row][col];
            if (row < M) {
                dgm16_cp_async_cg_16(dst, &A[(unsigned long long)row * a_row_stride + k_base + col]);
            } else {
                #pragma unroll
                for (int e = 0; e < DGM16_ELEMS_PER_CHUNK; e++) dst[e] = __float2bfloat16(0.0f);
            }
        }
        #pragma unroll
        for (int c = 0; c < B_CHUNKS; c++) {
            const unsigned int row = (threadIdx.x >> 3) + c * DGM16_ROWS_PER_PASS;
            const unsigned int gn = cta_n + row;
            __nv_bfloat16* dst = &smem_B[stage][row][col];
            if (gn < N) {
                dgm16_cp_async_cg_16(dst, &B[(unsigned long long)gn * K + k_base + col]);
            } else {
                #pragma unroll
                for (int e = 0; e < DGM16_ELEMS_PER_CHUNK; e++) dst[e] = __float2bfloat16(0.0f);
            }
        }
        dgm16_cp_async_commit();
    };

    #pragma unroll
    for (unsigned int p = 0; p < DGM16_STAGES - 1; p++) {
        if (p < n_steps) prefetch(p, p);
    }

    for (unsigned int step = 0; step < n_steps; step++) {
        const unsigned int cur = step % DGM16_STAGES;
        // 2026-09-25: Groups complete FIFO. Before this iteration issues its own prefetch,
        // min(n_steps, STAGES-1+step) groups are committed and `cur` is the
        // step-th, so the number that may stay in flight is:
        const unsigned int committed = min(n_steps, DGM16_STAGES - 1 + step);
        dgm16_cp_async_wait_le(committed - (step + 1));
        // 2026-09-25: One barrier per K-step. It makes stage `cur` visible to every warp,
        // and it follows every warp's MMA of step-1, which is what makes the
        // prefetch below (it targets stage (step-1) % STAGES) safe without a
        // second barrier.
        __syncthreads();
        const unsigned int ahead = step + DGM16_STAGES - 1;
        if (ahead < n_steps) prefetch(ahead, ahead % DGM16_STAGES);

        const unsigned short* sA = (const unsigned short*)&smem_A[cur][0][0];
        const unsigned short* sB = (const unsigned short*)&smem_B[cur][0][0];
        #pragma unroll
        for (int s = 0; s < DGM16_K_SUBS; s++) {
            // 2026-09-25: m16n8k16 row.col fragments, built straight from shared memory: A
            // rows {group_id, group_id+8} x K pairs {quad*2, quad*2+8}; B row
            // (warp_n + j*8 + group_id) at the same K pairs. The A fragment is loaded
            // once and reused by all N_SUBS MMAs. A staged weight word is a B
            // fragment register: no dequant, no scale.

            const unsigned int kc0 = s * DGM16_K_SUB + quad * 2;
            const unsigned int kc1 = kc0 + 8;
            const unsigned int r0 = group_id * DGM16_ROW_STRIDE;
            const unsigned int r1 = (group_id + 8) * DGM16_ROW_STRIDE;
            const unsigned int a0 = *(const unsigned int*)&sA[r0 + kc0];
            const unsigned int a1 = *(const unsigned int*)&sA[r1 + kc0];
            const unsigned int a2 = *(const unsigned int*)&sA[r0 + kc1];
            const unsigned int a3 = *(const unsigned int*)&sA[r1 + kc1];
            #pragma unroll
            for (int j = 0; j < N_SUBS; j++) {
                const unsigned int brow =
                    (warp_n + j * DGM16_N_PER_MMA + group_id) * DGM16_ROW_STRIDE;
                const unsigned int b0 = *(const unsigned int*)&sB[brow + kc0];
                const unsigned int b1 = *(const unsigned int*)&sB[brow + kc1];
                // 2026-09-25: Index `acc` directly: `j` is a compile-time constant under the
                // unroll, so the accumulators can stay in registers.


                asm volatile(
                    "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
                    "{%0, %1, %2, %3}, "
                    "{%4, %5, %6, %7}, "
                    "{%8, %9}, "
                    "{%10, %11, %12, %13};"
                    : "=f"(acc[j * 4 + 0]), "=f"(acc[j * 4 + 1]),
                      "=f"(acc[j * 4 + 2]), "=f"(acc[j * 4 + 3])
                    : "r"(a0), "r"(a1), "r"(a2), "r"(a3),
                      "r"(b0), "r"(b1),
                      "f"(acc[j * 4 + 0]), "f"(acc[j * 4 + 1]),
                      "f"(acc[j * 4 + 2]), "f"(acc[j * 4 + 3])
                );
            }
        }
    }

    // 2026-09-25: Store: FP32 accumulators -> BF16, masked to [M, N].
    const unsigned int row0 = group_id;
    const unsigned int row1 = group_id + 8;
    const unsigned long long o0 = (unsigned long long)row0 * c_row_stride;
    const unsigned long long o1 = (unsigned long long)row1 * c_row_stride;
    #pragma unroll
    for (int j = 0; j < N_SUBS; j++) {
        const unsigned int col0 = cta_n + warp_n + j * DGM16_N_PER_MMA + quad * 2;
        const unsigned int col1 = col0 + 1;
        if (row0 < M && col0 < N) C[o0 + col0] = __float2bfloat16(acc[j * 4 + 0]);
        if (row0 < M && col1 < N) C[o0 + col1] = __float2bfloat16(acc[j * 4 + 1]);
        if (row1 < M && col0 < N) C[o1 + col0] = __float2bfloat16(acc[j * 4 + 2]);
        if (row1 < M && col1 < N) C[o1 + col1] = __float2bfloat16(acc[j * 4 + 3]);
    }
}

/// 2026-09-25: The 32-wide CTA. `a_row_stride` / `c_row_stride` are in elements; the
/// BF16 LM head (`lm_head_batched.rs`) passes K and N.

extern "C" __global__
__launch_bounds__(DGM16_THREADS, 4)
void dense_gemm_m16_bf16(
    const __nv_bfloat16* __restrict__ A,
    const __nv_bfloat16* __restrict__ B,
    __nv_bfloat16* __restrict__ C,
    unsigned int M,
    unsigned int N,
    unsigned int K,
    unsigned int a_row_stride,
    unsigned int c_row_stride
) {
    dense_gemm_m16_bf16_impl<DGM16_N_TILE>(A, B, C, M, N, K, a_row_stride, c_row_stride);
}

/// 2026-09-25: `N_TILE=64` twin of [`dense_gemm_m16_bf16`]: the same arguments and the
/// same per-output arithmetic (the m16n8k16 K order is unchanged; only which
/// CTA owns a column changes), half the CTAs, and each staged A tile feeds
/// twice the weight bytes. Grid is `ceil(N/64)`. Selected by
/// `METRALE_LM_HEAD_M16_TC_NTILE=64`.



extern "C" __global__
__launch_bounds__(DGM16_THREADS, 4)
void dense_gemm_m16_bf16_n64(
    const __nv_bfloat16* __restrict__ A,
    const __nv_bfloat16* __restrict__ B,
    __nv_bfloat16* __restrict__ C,
    unsigned int M,
    unsigned int N,
    unsigned int K,
    unsigned int a_row_stride,
    unsigned int c_row_stride
) {
    dense_gemm_m16_bf16_impl<DGM16_N_TILE_WIDE>(A, B, C, M, N, K, a_row_stride, c_row_stride);
}
