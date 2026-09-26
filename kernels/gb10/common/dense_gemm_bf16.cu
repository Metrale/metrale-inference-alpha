// SPDX-License-Identifier: MIT OR Apache-2.0

// 2026-09-25: BF16 dense GEMMs, C[M, N] = A[M, K] * B[N, K]^T with FP32 accumulation, and `fused_silu_mul`.
//
// `dense_gemm_bf16` is the scalar tiled kernel; `dense_gemm_bf16_f32out` and `dense_gemm_f32in_f32out` are its FP32
// output (and FP32 input) twins; `dense_gemm_bf16_router` keeps its per-output accumulation order with register
// blocking; `dense_gemm_bf16_pipelined` runs on tensor cores. All read B transposed: B^T[k, n] = B[n * K + k].
//
// Owner: gb10 kernels.
// Invariants: none beyond the types.




#include <cuda_bf16.h>

#define TILE_M 16
#define TILE_N 16
#define TILE_K 16

// 2026-09-25: Grid (ceil(N/16), ceil(M/16)), block (16, 16): one thread per output element.




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

// 2026-09-25: dense_gemm_bf16 with the FP32 accumulator stored unrounded. The MoE router gate uses it under
// METRALE_FP32_GATE (moe/forward_batched_gate.rs) so gate logits closer than a BF16 step are not rounded together before
// top-k. Same grid and block as dense_gemm_bf16.





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

// 2026-09-25: The same kernel with FP32 A as well: under METRALE_FP32_ROUTING the MoE router gate multiplies the FP32
// `moe_router_in_f32` buffer by the BF16 gate weight through it (moe/forward_batched_gate.rs). Same grid and block as
// dense_gemm_bf16.



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


// 2026-09-25: dense_gemm_bf16_router: register-blocked BF16 GEMM that keeps dense_gemm_bf16's accumulation order.
//
// Each C[m, n] is one FP32 accumulator updated in ascending k, and the BF16-to-FP32 conversion at the smem store is
// exact, so only blocking and vectorisation differ from the scalar kernel. The kernel directory builds with
// `--fmad=false` (KERNEL.toml), so no multiply-add is fused. The MoE router gate uses it for that reason, falling back
// to dense_gemm_bf16 when a target lacks it (`router_gate_gemm_dense`, moe/helpers_c.rs).
//
// A block computes a 16x64 tile, 4 columns per thread, over 64-wide K slices; A and B are staged in smem as FP32.
// Grid (ceil(N/64), ceil(M/16)), block (16, 16). The thread and column mapping assume blockDim.x == 16.



















#define RG_BM 16
#define RG_BN 64
#define RG_BK 64
#define RG_NCOL 4

extern "C" __global__ void dense_gemm_bf16_router(
    const __nv_bfloat16* __restrict__ A,
    const __nv_bfloat16* __restrict__ B,
    __nv_bfloat16* __restrict__ C,
    unsigned int M,
    unsigned int N,
    unsigned int K
) {
    __shared__ float sA[RG_BM][RG_BK + 1];
    __shared__ float sB[RG_BN][RG_BK + 1];

    const unsigned int tid  = threadIdx.y * 16u + threadIdx.x;
    const unsigned int row0 = blockIdx.y * RG_BM;
    const unsigned int col0 = blockIdx.x * RG_BN;
    const unsigned int row  = row0 + threadIdx.y;
    const unsigned int col  = col0 + threadIdx.x * RG_NCOL;

    float acc[RG_NCOL] = {0.0f, 0.0f, 0.0f, 0.0f};

    for (unsigned int kb = 0; kb < K; kb += RG_BK) {
        {
            unsigned int e = tid * 4u;
            unsigned int r = e / RG_BK, c = e % RG_BK;
            unsigned int gr = row0 + r, gc = kb + c;
            if (gr < M && gc + 3u < K) {
                ushort4 v = *(const ushort4*)(const unsigned short*)(A + (size_t)gr * K + gc);
                sA[r][c + 0] = __bfloat162float(__ushort_as_bfloat16(v.x));
                sA[r][c + 1] = __bfloat162float(__ushort_as_bfloat16(v.y));
                sA[r][c + 2] = __bfloat162float(__ushort_as_bfloat16(v.z));
                sA[r][c + 3] = __bfloat162float(__ushort_as_bfloat16(v.w));
            } else {
                for (int j = 0; j < 4; j++) {
                    unsigned int cc = c + j;
                    sA[r][cc] = (gr < M && kb + cc < K)
                        ? __bfloat162float(A[(size_t)gr * K + kb + cc]) : 0.0f;
                }
            }
        }
        {
            for (int it = 0; it < 2; it++) {
                unsigned int e = (tid + it * 256u) * 8u;
                unsigned int r = e / RG_BK, c = e % RG_BK;
                unsigned int gn = col0 + r, gc = kb + c;
                if (gn < N && gc + 7u < K) {
                    uint4 v = *(const uint4*)(B + (size_t)gn * K + gc);
                    const unsigned short* u = (const unsigned short*)&v;
                    #pragma unroll
                    for (int j = 0; j < 8; j++)
                        sB[r][c + j] = __bfloat162float(__ushort_as_bfloat16(u[j]));
                } else {
                    for (int j = 0; j < 8; j++) {
                        unsigned int cc = c + j;
                        sB[r][cc] = (gn < N && kb + cc < K)
                            ? __bfloat162float(B[(size_t)gn * K + kb + cc]) : 0.0f;
                    }
                }
            }
        }
        __syncthreads();





        #pragma unroll 8
        for (unsigned int kk = 0; kk < RG_BK; kk++) {
            float a = sA[threadIdx.y][kk];
            #pragma unroll
            for (int j = 0; j < RG_NCOL; j++)
                acc[j] += a * sB[threadIdx.x * RG_NCOL + j][kk];
        }
        __syncthreads();
    }

    if (row < M) {
        #pragma unroll
        for (int j = 0; j < RG_NCOL; j++)
            if (col + j < N) C[(size_t)row * N + col + j] = __float2bfloat16(acc[j]);
    }
}


// 2026-09-25: dense_gemm_bf16_pipelined: tensor-core BF16 GEMM with dense_gemm_bf16's math and I/O layout.
//
// A and B tiles are copied into smem with 16-byte cp.async in a DM_STAGES-deep pipeline, and each warp accumulates
// m16n8k16 MMAs into one FP32 accumulator per N sub-tile across all of K. At the defaults the tile is 128x128 (M x N)
// with a K step of 32, 8 warps own 16 rows each, and smem is 2 stages x (128 + 128) rows x 40 BF16 = 40,960 bytes.
// Grid (ceil(N/DM_N_TILE), ceil(M/DM_M_TILE)), block 256.



































// 2026-09-25: DM_N_TILE, DM_K_STEP and DM_STAGES can be overridden with -D. The host launch (`dense_gemm_bf16_pipelined`
// in crates/model-layers/src/layers/ops/gemm_dense_bf16.rs) assumes the default 128x128 tile.

#define DM_M_TILE 128




#ifndef DM_N_TILE
#define DM_N_TILE 128
#endif


#ifndef DM_K_STEP
#define DM_K_STEP 32
#endif
#define DM_K_SUB 16
#define DM_K_SUBS (DM_K_STEP / DM_K_SUB)
// 2026-09-25: Row strides of K_STEP + 8 BF16 keep every 16-byte cp.async destination row 16-byte aligned (K_STEP is a
// multiple of 8).


#define DM_A_STRIDE (DM_K_STEP + 8)
#define DM_B_STRIDE (DM_K_STEP + 8)
#define DM_WARPS 8
#define DM_THREADS (DM_WARPS * 32)
#define DM_N_TILES_PER_WARP (DM_N_TILE / 8)



#ifndef DM_STAGES
#define DM_STAGES 2
#endif

// 2026-09-25: 16-byte cp.async.cg copy, global to smem; both addresses must be 16-byte aligned.

__device__ __forceinline__ void dm_cp_async_cg_16(void* smem_ptr, const void* gmem_ptr) {
    unsigned int s = (unsigned int)__cvta_generic_to_shared(smem_ptr);
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" ::"r"(s), "l"(gmem_ptr));
}
__device__ __forceinline__ void dm_cp_async_commit() {
    asm volatile("cp.async.commit_group;\n" ::);
}
template <int N>
__device__ __forceinline__ void dm_cp_async_wait_group() {
    asm volatile("cp.async.wait_group %0;\n" ::"n"(N));
}
// 2026-09-25: Waits until at most `n` cp.async groups are in flight. cp.async.wait_group takes a compile-time immediate,
// so the runtime `n` is dispatched through a switch (values above 2 wait for 3).

__device__ __forceinline__ void dm_cp_async_wait_le(unsigned int n) {
    switch (n) {
        case 0:  dm_cp_async_wait_group<0>(); break;
        case 1:  dm_cp_async_wait_group<1>(); break;
        case 2:  dm_cp_async_wait_group<2>(); break;
        default: dm_cp_async_wait_group<3>(); break;
    }
}

// 2026-09-25: MMAs over one resident K step: DM_K_SUBS m16n8k16 MMAs per N sub-tile. smem_B is [n][k] with k contiguous,
// so each (k, k+1) pair of the B fragment is one aligned 32-bit load.




__device__ __forceinline__ void dm_mma_kstep(
    const __nv_bfloat16* smem_A,
    const __nv_bfloat16* smem_B,
    float acc[DM_N_TILES_PER_WARP][4],
    unsigned int warp_m_offset, unsigned int group_id, unsigned int tid
) {
    const unsigned int a_stride = DM_A_STRIDE;
    const unsigned int b_stride = DM_B_STRIDE;
    const unsigned short* sA = (const unsigned short*)smem_A;
    const unsigned short* sB = (const unsigned short*)smem_B;

    unsigned int frag_r0 = warp_m_offset + group_id;
    unsigned int frag_r1 = warp_m_offset + group_id + 8;

    #pragma unroll
    for (int s = 0; s < DM_K_SUBS; s++) {
        const unsigned int k_off = s * DM_K_SUB;
        unsigned int frag_c0 = k_off + tid * 2;
        unsigned int frag_c1 = k_off + tid * 2 + 8;

        unsigned int a0 = *(const unsigned int*)&sA[frag_r0 * a_stride + frag_c0];
        unsigned int a1 = *(const unsigned int*)&sA[frag_r1 * a_stride + frag_c0];
        unsigned int a2 = *(const unsigned int*)&sA[frag_r0 * a_stride + frag_c1];
        unsigned int a3 = *(const unsigned int*)&sA[frag_r1 * a_stride + frag_c1];

        #pragma unroll
        for (int n_tile = 0; n_tile < DM_N_TILES_PER_WARP; n_tile++) {
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
                : "=f"(acc[n_tile][0]), "=f"(acc[n_tile][1]),
                  "=f"(acc[n_tile][2]), "=f"(acc[n_tile][3])
                : "r"(a0), "r"(a1), "r"(a2), "r"(a3),
                  "r"(b0), "r"(b1),
                  "f"(acc[n_tile][0]), "f"(acc[n_tile][1]),
                  "f"(acc[n_tile][2]), "f"(acc[n_tile][3])
            );
        }
    }
}



extern "C" __global__ void dense_gemm_bf16_pipelined(
    const __nv_bfloat16* __restrict__ A,
    const __nv_bfloat16* __restrict__ B,
    __nv_bfloat16* __restrict__ C,
    unsigned int M,
    unsigned int N,
    unsigned int K
) {
    const unsigned int cta_m = blockIdx.y * DM_M_TILE;
    const unsigned int cta_n = blockIdx.x * DM_N_TILE;
    const unsigned int warp_id = threadIdx.x / 32;
    const unsigned int lane_id = threadIdx.x % 32;
    const unsigned int warp_m_offset = warp_id * 16;
    const unsigned int group_id = lane_id >> 2;
    const unsigned int tid = lane_id & 3;





    __shared__ __align__(16) __nv_bfloat16 smem_A[DM_STAGES][DM_M_TILE][DM_A_STRIDE];
    __shared__ __align__(16) __nv_bfloat16 smem_B[DM_STAGES][DM_N_TILE][DM_B_STRIDE];



    float acc[DM_N_TILES_PER_WARP][4];
    #pragma unroll
    for (int i = 0; i < DM_N_TILES_PER_WARP; i++) {
        acc[i][0] = 0.0f; acc[i][1] = 0.0f; acc[i][2] = 0.0f; acc[i][3] = 0.0f;
    }

    const unsigned int n_steps = (K + DM_K_STEP - 1) / DM_K_STEP;



    const unsigned int a_chunks = (DM_M_TILE * DM_K_STEP) / 8;


    const unsigned int b_chunks = (DM_N_TILE * DM_K_STEP) / 8;

    // 2026-09-25: Issues the copies for K step `step` into buffer `stage`, contiguous along K for A and for B. A 16-byte
    // cp.async needs a 16-byte-aligned source: `gc`/`gk` are multiples of 8 elements, but a row base `gr*K` / `gn*K`
    // is 16-byte aligned only when K % 8 == 0. For any other K every chunk takes the masked scalar copy, as do chunks
    // past M, N or the end of K.








    const bool k_vec_aligned = (K & 7u) == 0u;
    auto prefetch = [&](unsigned int step, unsigned int stage) {
        unsigned int k_base = step * DM_K_STEP;


        #pragma unroll
        for (unsigned int c = threadIdx.x; c < a_chunks; c += DM_THREADS) {
            unsigned int row = (c * 8) / DM_K_STEP;
            unsigned int col = (c * 8) % DM_K_STEP;
            unsigned int gr = cta_m + row;
            unsigned int gc = k_base + col;
            __nv_bfloat16* dst = &smem_A[stage][row][col];
            if (gr < M && gc + 8 <= K && k_vec_aligned) {
                dm_cp_async_cg_16(dst, &A[(unsigned long long)gr * K + gc]);
            } else {
                #pragma unroll
                for (unsigned int e = 0; e < 8; e++) {
                    unsigned int gcol = gc + e;
                    dst[e] = (gr < M && gcol < K) ? A[(unsigned long long)gr * K + gcol]
                                                  : __float2bfloat16(0.0f);
                }
            }
        }



        #pragma unroll
        for (unsigned int c = threadIdx.x; c < b_chunks; c += DM_THREADS) {
            unsigned int nrow = (c * 8) / DM_K_STEP;
            unsigned int kcol = (c * 8) % DM_K_STEP;
            unsigned int gn = cta_n + nrow;
            unsigned int gk = k_base + kcol;
            __nv_bfloat16* dst = &smem_B[stage][nrow][kcol];
            if (gn < N && gk + 8 <= K && k_vec_aligned) {
                dm_cp_async_cg_16(dst, &B[(unsigned long long)gn * K + gk]);
            } else {
                #pragma unroll
                for (unsigned int e = 0; e < 8; e++) {
                    unsigned int gke = gk + e;
                    dst[e] = (gn < N && gke < K) ? B[(unsigned long long)gn * K + gke]
                                                 : __float2bfloat16(0.0f);
                }
            }
        }
        dm_cp_async_commit();
    };


    #pragma unroll
    for (unsigned int p = 0; p < DM_STAGES - 1; p++) {
        if (p < n_steps) {
            prefetch(p, p % DM_STAGES);
        }
    }

    for (unsigned int step = 0; step < n_steps; step++) {
        unsigned int cur = step % DM_STAGES;

        unsigned int ahead = step + (DM_STAGES - 1);
        if (ahead < n_steps) {
            prefetch(ahead, ahead % DM_STAGES);
        }
        unsigned int committed = min(n_steps, DM_STAGES + step);
        unsigned int target = committed - (step + 1);
        dm_cp_async_wait_le(target);
        __syncthreads();   // 2026-09-25: stage `cur` is resident for every thread.

        dm_mma_kstep(&smem_A[cur][0][0], &smem_B[cur][0][0],
                     acc, warp_m_offset, group_id, tid);
        __syncthreads();   // 2026-09-25: every read of stage `cur` ends before a later prefetch refills it.
    }


    #pragma unroll
    for (int n_tile = 0; n_tile < DM_N_TILES_PER_WARP; n_tile++) {
        unsigned int base_n = cta_n + n_tile * 8;
        unsigned int col0 = base_n + (tid * 2);
        unsigned int col1 = col0 + 1;
        unsigned int row0 = cta_m + warp_m_offset + group_id;
        unsigned int row1 = row0 + 8;

        if (row0 < M && col0 < N) C[row0 * N + col0] = __float2bfloat16(acc[n_tile][0]);
        if (row0 < M && col1 < N) C[row0 * N + col1] = __float2bfloat16(acc[n_tile][1]);
        if (row1 < M && col0 < N) C[row1 * N + col0] = __float2bfloat16(acc[n_tile][2]);
        if (row1 < M && col1 < N) C[row1 * N + col1] = __float2bfloat16(acc[n_tile][3]);
    }
}

// 2026-09-25: out[t, i] = silu(gate[t, i]) * up[t, i]. Row t of gate_up holds inter_size gate values then inter_size up
// values; row t of output holds inter_size values. Each thread handles two adjacent columns with 32-bit loads and
// stores, so inter_size must be even.

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
