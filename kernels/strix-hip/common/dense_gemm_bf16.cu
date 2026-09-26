// SPDX-License-Identifier: MIT OR Apache-2.0
// forked-from: kernels/gb10/common/dense_gemm_bf16.cu (2026-09-24; 452 of 629 lines differ, see kernels/FORKS.md)

// 2026-09-25: Dense GEMMs C = A * B^T for gfx1151, with B read transposed
// (B^T[k,n] = B[n*K + k]), plus fused_silu_mul.
//   A: [M, K] row-major, B: [N, K] row-major, C: [M, N] row-major.
// dense_gemm_bf16, dense_gemm_bf16_f32out and dense_gemm_f32in_f32out use
// 16 x 16 shared-memory tiles, one thread per output and FP32 accumulation;
// dense_gemm_bf16_pipelined uses AMD WMMA.
//
// Owner: strix-hip kernels.
// Invariants: none beyond the types.


#include <cuda_bf16.h>

#define TILE_M 16
#define TILE_N 16
#define TILE_K 16

// 2026-09-25: BF16 C with FP32 accumulation, one thread per output.
// Grid: (ceil(N/TILE_N), ceil(M/TILE_M)); block: (TILE_N, TILE_M).



extern "C" __global__ void dense_gemm_bf16(
    const __nv_bfloat16* __restrict__ A,
    const __nv_bfloat16* __restrict__ B,
    __nv_bfloat16* __restrict__ C,
    unsigned int M,
    unsigned int N,
    unsigned int K
) {

    unsigned int row = blockIdx.y * TILE_M + threadIdx.y;
    unsigned int col = blockIdx.x * TILE_N + threadIdx.x;


    __shared__ __nv_bfloat16 smem_A[TILE_M][TILE_K];
    __shared__ __nv_bfloat16 smem_B[TILE_K][TILE_N];

    float acc = 0.0f;


    for (unsigned int k_base = 0; k_base < K; k_base += TILE_K) {

        if (row < M && (k_base + threadIdx.x) < K) {
            smem_A[threadIdx.y][threadIdx.x] = A[row * K + k_base + threadIdx.x];
        } else {
            smem_A[threadIdx.y][threadIdx.x] = __float2bfloat16(0.0f);
        }


        if ((k_base + threadIdx.y) < K && col < N) {
            smem_B[threadIdx.y][threadIdx.x] = B[(unsigned long long)col * K + (k_base + threadIdx.y)];
        } else {
            smem_B[threadIdx.y][threadIdx.x] = __float2bfloat16(0.0f);
        }

        __syncthreads();


        for (unsigned int kk = 0; kk < TILE_K; kk++) {
            acc += __bfloat162float(smem_A[threadIdx.y][kk])
                 * __bfloat162float(smem_B[kk][threadIdx.x]);
        }

        __syncthreads();
    }


    if (row < M && col < N) {
        C[row * N + col] = __float2bfloat16(acc);
    }
}

// 2026-09-25: dense_gemm_bf16 with the FP32 accumulator stored as FP32 C.
// Same grid and block.




extern "C" __global__ void dense_gemm_bf16_f32out(
    const __nv_bfloat16* __restrict__ A,
    const __nv_bfloat16* __restrict__ B,
    float* __restrict__ C,
    unsigned int M,
    unsigned int N,
    unsigned int K
) {
    unsigned int row = blockIdx.y * TILE_M + threadIdx.y;
    unsigned int col = blockIdx.x * TILE_N + threadIdx.x;

    __shared__ __nv_bfloat16 smem_A[TILE_M][TILE_K];
    __shared__ __nv_bfloat16 smem_B[TILE_K][TILE_N];

    float acc = 0.0f;

    for (unsigned int k_base = 0; k_base < K; k_base += TILE_K) {
        if (row < M && (k_base + threadIdx.x) < K) {
            smem_A[threadIdx.y][threadIdx.x] = A[row * K + k_base + threadIdx.x];
        } else {
            smem_A[threadIdx.y][threadIdx.x] = __float2bfloat16(0.0f);
        }
        if ((k_base + threadIdx.y) < K && col < N) {
            smem_B[threadIdx.y][threadIdx.x] = B[(unsigned long long)col * K + (k_base + threadIdx.y)];
        } else {
            smem_B[threadIdx.y][threadIdx.x] = __float2bfloat16(0.0f);
        }
        __syncthreads();
        for (unsigned int kk = 0; kk < TILE_K; kk++) {
            acc += __bfloat162float(smem_A[threadIdx.y][kk])
                 * __bfloat162float(smem_B[kk][threadIdx.x]);
        }
        __syncthreads();
    }

    if (row < M && col < N) {
        C[row * N + col] = acc;
    }
}

// 2026-09-25: dense_gemm_bf16_f32out with FP32 A. Same grid and block.





extern "C" __global__ void dense_gemm_f32in_f32out(
    const float* __restrict__ A,
    const __nv_bfloat16* __restrict__ B,
    float* __restrict__ C,
    unsigned int M,
    unsigned int N,
    unsigned int K
) {
    unsigned int row = blockIdx.y * TILE_M + threadIdx.y;
    unsigned int col = blockIdx.x * TILE_N + threadIdx.x;

    __shared__ float smem_A[TILE_M][TILE_K];
    __shared__ __nv_bfloat16 smem_B[TILE_K][TILE_N];

    float acc = 0.0f;

    for (unsigned int k_base = 0; k_base < K; k_base += TILE_K) {
        if (row < M && (k_base + threadIdx.x) < K) {
            smem_A[threadIdx.y][threadIdx.x] = A[row * K + k_base + threadIdx.x];
        } else {
            smem_A[threadIdx.y][threadIdx.x] = 0.0f;
        }
        if ((k_base + threadIdx.y) < K && col < N) {
            smem_B[threadIdx.y][threadIdx.x] = B[(unsigned long long)col * K + (k_base + threadIdx.y)];
        } else {
            smem_B[threadIdx.y][threadIdx.x] = __float2bfloat16(0.0f);
        }
        __syncthreads();
        for (unsigned int kk = 0; kk < TILE_K; kk++) {
            acc += smem_A[threadIdx.y][kk] * __bfloat162float(smem_B[kk][threadIdx.x]);
        }
        __syncthreads();
    }

    if (row < M && col < N) {
        C[row * N + col] = acc;
    }
}

// 2026-09-25: dense_gemm_bf16_pipelined keeps the name and (A, B, C, M, N, K)
// signature of the gb10 kernel, whose inline PTX (cp.async, mma.sync) HIP
// cannot compile, and computes it with
// __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32. Launch: grid (ceil(N/128),
// ceil(M/128), 1), block (256, 1, 1), as layers/ops/gemm_dense_bf16.rs
// dispatches it.
//
// 8 waves; wave w owns rows [16w, 16w + 16) of the 128 x 128 tile and all 8
// WMMA n-subtiles. K advances 16 at a time through two shared-memory
// buffers: tile k+1 is loaded into registers while tile k is multiplied, then
// stored to the other buffer, with one barrier per K step.
// LDS: 2*128*18*2 + 2*16*130*2 = 17536 bytes per block.










#define DP_M_TILE 128
#define DP_N_TILE 128
#define DP_K_STEP 16
#define DP_THREADS 256
#define DP_PAD 2
#define DP_NSUB (DP_N_TILE / 16)
#define DP_A_EPT ((DP_M_TILE * DP_K_STEP) / DP_THREADS)
#define DP_B_EPT ((DP_K_STEP * DP_N_TILE) / DP_THREADS)

typedef __bf16 dp_v16bf __attribute__((ext_vector_type(16)));
typedef float  dp_v8f   __attribute__((ext_vector_type(8)));

__device__ __forceinline__ void dp_load_A_regs(
    const __nv_bfloat16* __restrict__ A, __nv_bfloat16 reg_A[DP_A_EPT],
    unsigned int cta_m, unsigned int k_base, unsigned int M, unsigned int K
) {
    #pragma unroll
    for (unsigned int i = 0; i < DP_A_EPT; i++) {
        unsigned int idx = threadIdx.x * DP_A_EPT + i;
        unsigned int row = idx / DP_K_STEP, col = idx % DP_K_STEP;
        unsigned int gr = cta_m + row, gc = k_base + col;
        reg_A[i] = (gr < M && gc < K) ? A[(unsigned long long)gr * K + gc] : __float2bfloat16(0.0f);
    }
}

__device__ __forceinline__ void dp_load_B_regs(
    const __nv_bfloat16* __restrict__ B, __nv_bfloat16 reg_B[DP_B_EPT],
    unsigned int cta_n, unsigned int k_base, unsigned int N, unsigned int K
) {

    #pragma unroll
    for (unsigned int i = 0; i < DP_B_EPT; i++) {
        unsigned int idx = threadIdx.x * DP_B_EPT + i;
        unsigned int k = idx / DP_N_TILE, n = idx % DP_N_TILE;
        unsigned int gk = k_base + k, gn = cta_n + n;
        reg_B[i] = (gk < K && gn < N) ? B[(unsigned long long)gn * K + gk] : __float2bfloat16(0.0f);
    }
}

__device__ __forceinline__ void dp_store_A_regs(
    __nv_bfloat16 smem_A[][DP_K_STEP + DP_PAD], const __nv_bfloat16 reg_A[DP_A_EPT]
) {
    #pragma unroll
    for (unsigned int i = 0; i < DP_A_EPT; i++) {
        unsigned int idx = threadIdx.x * DP_A_EPT + i;
        smem_A[idx / DP_K_STEP][idx % DP_K_STEP] = reg_A[i];
    }
}

__device__ __forceinline__ void dp_store_B_regs(
    __nv_bfloat16 smem_B[][DP_N_TILE + DP_PAD], const __nv_bfloat16 reg_B[DP_B_EPT]
) {
    #pragma unroll
    for (unsigned int i = 0; i < DP_B_EPT; i++) {
        unsigned int idx = threadIdx.x * DP_B_EPT + i;
        smem_B[idx / DP_N_TILE][idx % DP_N_TILE] = reg_B[i];
    }
}

__device__ __forceinline__ void dp_wmma_compute(
    __nv_bfloat16 smem_A[][DP_K_STEP + DP_PAD],
    __nv_bfloat16 smem_B[][DP_N_TILE + DP_PAD],
    dp_v8f acc[DP_NSUB], unsigned int warp_m_offset, unsigned int lane
) {
    dp_v16bf a;
    #pragma unroll
    for (int i = 0; i < 16; i++) a[i] = (__bf16)(float)smem_A[warp_m_offset + (lane & 15)][i];
    #pragma unroll
    for (int nb = 0; nb < DP_NSUB; nb++) {
        dp_v16bf b;
        #pragma unroll
        for (int k = 0; k < 16; k++) b[k] = (__bf16)(float)smem_B[k][nb * 16 + (lane & 15)];
        acc[nb] = __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32(a, b, acc[nb]);
    }
}

extern "C" __global__
__launch_bounds__(256, 2)
void dense_gemm_bf16_pipelined(
    const __nv_bfloat16* __restrict__ A,
    const __nv_bfloat16* __restrict__ B,
    __nv_bfloat16* __restrict__ C,
    unsigned int M, unsigned int N, unsigned int K
) {
    const unsigned int cta_m = blockIdx.y * DP_M_TILE;
    const unsigned int cta_n = blockIdx.x * DP_N_TILE;
    const unsigned int warp_id = threadIdx.x / 32;
    const unsigned int lane_id = threadIdx.x % 32;
    const unsigned int warp_m_offset = warp_id * 16;

    __shared__ __nv_bfloat16 smem_A[2][DP_M_TILE][DP_K_STEP + DP_PAD];
    __shared__ __nv_bfloat16 smem_B[2][DP_K_STEP][DP_N_TILE + DP_PAD];

    dp_v8f acc[DP_NSUB];
    #pragma unroll
    for (int i = 0; i < DP_NSUB; i++) acc[i] = dp_v8f{0, 0, 0, 0, 0, 0, 0, 0};

    const unsigned int num_tiles = (K + DP_K_STEP - 1) / DP_K_STEP;

    __nv_bfloat16 reg_A[DP_A_EPT];
    __nv_bfloat16 reg_B[DP_B_EPT];
    dp_load_A_regs(A, reg_A, cta_m, 0, M, K);
    dp_load_B_regs(B, reg_B, cta_n, 0, N, K);
    dp_store_A_regs(smem_A[0], reg_A);
    dp_store_B_regs(smem_B[0], reg_B);
    __syncthreads();

    for (unsigned int kt = 1; kt < num_tiles; kt++) {
        const unsigned int k_base = kt * DP_K_STEP;
        dp_load_A_regs(A, reg_A, cta_m, k_base, M, K);
        dp_load_B_regs(B, reg_B, cta_n, k_base, N, K);
        dp_wmma_compute(smem_A[(kt - 1) & 1], smem_B[(kt - 1) & 1], acc, warp_m_offset, lane_id);
        dp_store_A_regs(smem_A[kt & 1], reg_A);
        dp_store_B_regs(smem_B[kt & 1], reg_B);
        __syncthreads();
    }
    dp_wmma_compute(smem_A[(num_tiles - 1) & 1], smem_B[(num_tiles - 1) & 1], acc, warp_m_offset, lane_id);

    #pragma unroll
    for (int nb = 0; nb < DP_NSUB; nb++)
        #pragma unroll
        for (int e = 0; e < 8; e++) {
            unsigned int r = cta_m + warp_m_offset + 2 * e + (lane_id >> 4);
            unsigned int c = cta_n + nb * 16 + (lane_id & 15);
            if (r < M && c < N) C[(unsigned long long)r * N + c] = __float2bfloat16(acc[nb][e]);
        }
}

// 2026-09-25: out[t, i] = silu(gate_up[t, i]) * gate_up[t, inter_size + i],
// two elements per thread through 32-bit loads and stores; assumes
// inter_size is even.

extern "C" __global__ void fused_silu_mul(
    const __nv_bfloat16* __restrict__ gate_up,
    __nv_bfloat16* __restrict__ output,
    unsigned int num_tokens,
    unsigned int inter_size
) {

    unsigned int idx2 = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int half_total = (num_tokens * inter_size) / 2;
    if (idx2 >= half_total) return;


    unsigned int half_inter = inter_size / 2;
    unsigned int token = idx2 / half_inter;
    unsigned int col_pair = idx2 % half_inter;


    const unsigned int* gate32 = (const unsigned int*)(gate_up + token * (inter_size * 2));
    const unsigned int* up32 = (const unsigned int*)(gate_up + token * (inter_size * 2) + inter_size);
    unsigned int* out32 = (unsigned int*)(output + token * inter_size);

    unsigned int g_packed = gate32[col_pair];
    unsigned int u_packed = up32[col_pair];

    float g0 = __bfloat162float(__ushort_as_bfloat16((unsigned short)(g_packed & 0xFFFF)));
    float g1 = __bfloat162float(__ushort_as_bfloat16((unsigned short)(g_packed >> 16)));
    float u0 = __bfloat162float(__ushort_as_bfloat16((unsigned short)(u_packed & 0xFFFF)));
    float u1 = __bfloat162float(__ushort_as_bfloat16((unsigned short)(u_packed >> 16)));


    float sg0 = 1.0f / (1.0f + __expf(-g0));
    float sg1 = 1.0f / (1.0f + __expf(-g1));
    float r0 = g0 * sg0 * u0;
    float r1 = g1 * sg1 * u1;


    unsigned int lo = (unsigned int)__bfloat16_as_ushort(__float2bfloat16(r0));
    unsigned int hi = (unsigned int)__bfloat16_as_ushort(__float2bfloat16(r1));
    out32[col_pair] = lo | (hi << 16);
}
