// SPDX-License-Identifier: MIT OR Apache-2.0
// forked-from: kernels/gb10/common/moe_w4a16_grouped_gemm.cu (2026-09-24; 840 of 1031 lines differ, see kernels/FORKS.md)

// 2026-09-25: Grouped W4A16 GEMMs for the MoE experts of the nemotron-labs-3-puzzle-75b-a9b tree: the rows of
// one expert are A rows x dequant(B_e), all experts in one launch, on BF16 m16n8k16 MMAs with FP32 accumulation.
//
// Owner: gb10 kernels (nemotron-labs-3-puzzle-75b-a9b).
// Invariants: none beyond the types.
//
// B is NVFP4: E2M1 codes two per byte with the even k in the low nibble, one E4M3 scale per 16 k, and a
// scale2; a weight is E2M1 value x E4M3 scale x scale2. blockIdx.z is the expert e, whose rows are
// expert_offsets[e] to expert_offsets[e + 1]; blockIdx.y is a 64-row tile of them and blockIdx.x an N tile.
// Every kernel expects 128 threads. Rows past the expert's count load as zero and are not stored.









#include <cuda_bf16.h>
#include <cuda_fp8.h>

#define M_TILE 64
#define N_TILE 64
#define K_STEP 16
// 2026-09-25: K and N tiles of the ptrtable kernels. A 64-value K step reads 32 contiguous packed bytes of each
// B row. Each CTA re-reads its A tile, so a 128-column N tile reads A half as often as the 64-column N_TILE;
// it costs 64 accumulator floats per thread (acc[16][4]).

#define K_STEP_PT 64









#define N_TILE_PT 128
#define PAD 2
#define GROUP_SIZE 16





#define K_OUTER 256
// 2026-09-25: BP_STRIDE: 32 packed bytes per row padded to 48 keep rows 16-byte aligned and put the 8 rows a warp reads in distinct banks.
#define BP_PAD 16
#define BP_STRIDE 48

#define BS_PER_ROW (K_STEP_PT / GROUP_SIZE)












__device__ __forceinline__ void moe_cp_async16(void* smem_ptr, const void* gmem_ptr) {
    unsigned int s = (unsigned int)__cvta_generic_to_shared(smem_ptr);
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" ::"r"(s), "l"(gmem_ptr));
}
__device__ __forceinline__ void moe_cp_async4(void* smem_ptr, const void* gmem_ptr) {
    unsigned int s = (unsigned int)__cvta_generic_to_shared(smem_ptr);
    asm volatile("cp.async.ca.shared.global [%0], [%1], 4;\n" ::"r"(s), "l"(gmem_ptr));
}
__device__ __forceinline__ void moe_cp_async_commit() {
    asm volatile("cp.async.commit_group;\n" ::);
}
// 2026-09-25: Waits until at most N committed cp.async groups are still in flight.
template <int N>
__device__ __forceinline__ void moe_cp_async_wait() {
    asm volatile("cp.async.wait_group %0;\n" ::"n"(N));
}

__device__ __constant__ float E2M1_LUT_MOE[16] = {
    0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f,
    -0.0f, -0.5f, -1.0f, -1.5f, -2.0f, -3.0f, -4.0f, -6.0f
};
// 2026-09-25: Stacked-weight form: B_packed [E, N, K / 2], B_scale [E, N, K / 16], one scale2. No Rust code looks it up.
extern "C" __global__ void moe_w4a16_grouped_gemm(
    const __nv_bfloat16* __restrict__ A,
    const unsigned char* __restrict__ B_packed,
    const unsigned char* __restrict__ B_scale,
    const float scale2,
    __nv_bfloat16* __restrict__ C,
    const int* __restrict__ expert_offsets,
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
    const unsigned int cta_n = blockIdx.x * N_TILE;


    const unsigned int half_K = K / 2;
    const unsigned int num_groups = K / GROUP_SIZE;
    const unsigned int weight_stride_packed = N * half_K;
    const unsigned int scale_stride = N * num_groups;
    const unsigned char* B_expert = B_packed + expert_id * weight_stride_packed;
    const unsigned char* S_expert = B_scale + expert_id * scale_stride;


    const unsigned int warp_id = threadIdx.x / 32;
    const unsigned int lane_id = threadIdx.x % 32;
    const unsigned int warp_m_offset = warp_id * 16;
    const unsigned int group_id = lane_id >> 2;
    const unsigned int tid = lane_id & 3;


    __shared__ __nv_bfloat16 smem_A[M_TILE][K_STEP + PAD];
    __shared__ __nv_bfloat16 smem_B[K_STEP][N_TILE + PAD];


    float acc[8][4];
    #pragma unroll
    for (int i = 0; i < 8; i++) {
        acc[i][0] = 0.0f; acc[i][1] = 0.0f;
        acc[i][2] = 0.0f; acc[i][3] = 0.0f;
    }

    const unsigned int a_stride = K_STEP + PAD;
    const unsigned int b_stride = N_TILE + PAD;


    const unsigned int M_eff = (unsigned int)M_expert;

    for (unsigned int k_base = 0; k_base < K; k_base += K_STEP) {

        {
            const unsigned int elems_per_thread = (M_TILE * K_STEP) / 128;
            #pragma unroll
            for (unsigned int i = 0; i < elems_per_thread; i++) {
                unsigned int idx = threadIdx.x * elems_per_thread + i;
                unsigned int row = idx / K_STEP;
                unsigned int col = idx % K_STEP;
                unsigned int gr = cta_m + row;
                unsigned int gc = k_base + col;

                bool valid = (cta_m_local + row) < M_eff && gc < K;
                smem_A[row][col] = valid ? A[gr * K + gc] : __float2bfloat16(0.0f);
            }
        }


        {
            // 2026-09-25: Thread t loads row n = t / 2 of the N tile: 8 k values (4 packed bytes) from
            // k_base + (t % 2) x 8, all in scale group k_base / 16.














            unsigned int scale_group = k_base / GROUP_SIZE;
            {
                const unsigned int n = threadIdx.x >> 1;
                const unsigned int khalf = threadIdx.x & 1u;
                const unsigned int k0 = k_base + khalf * 8u;
                const unsigned int gn = cta_n + n;
                const unsigned int kl = khalf * 8u;

                if (n < N_TILE) {
                    if (gn < N && k0 < K) {
                        const unsigned long long byte_off =
                            (unsigned long long)gn * half_K + (k0 >> 1);

                        unsigned char scale_byte =
                            S_expert[(unsigned long long)gn * num_groups + scale_group];
                        float fp8_val;
                        {
                            __nv_fp8_e4m3 fp8;
                            *(unsigned char*)&fp8 = scale_byte;
                            fp8_val = (float)fp8;
                        }
                        const float s = fp8_val * scale2;

                        unsigned char pb[4];
                        if (((half_K & 3u) == 0u) && (k0 + 8u <= K)) {
                            unsigned int w = *(const unsigned int*)(B_expert + byte_off);
                            pb[0] = (unsigned char)(w & 0xFFu);
                            pb[1] = (unsigned char)((w >> 8) & 0xFFu);
                            pb[2] = (unsigned char)((w >> 16) & 0xFFu);
                            pb[3] = (unsigned char)((w >> 24) & 0xFFu);
                        } else {
                            #pragma unroll
                            for (unsigned int j = 0; j < 4; j++) {
                                unsigned int gk = k0 + j * 2u;
                                pb[j] = (gk < K) ? B_expert[byte_off + j] : 0u;
                            }
                        }

                        #pragma unroll
                        for (unsigned int j = 0; j < 8; j++) {
                            unsigned int gk = k0 + j;
                            float dequant_val = 0.0f;
                            if (gk < K) {
                                unsigned char packed_byte = pb[j >> 1];
                                unsigned int nibble =
                                    (gk & 1u) ? (packed_byte >> 4) : (packed_byte & 0xFu);
                                dequant_val = E2M1_LUT_MOE[nibble] * s;
                            }
                            smem_B[kl + j][n] = __float2bfloat16(dequant_val);
                        }
                    } else {
                        #pragma unroll
                        for (unsigned int j = 0; j < 8; j++) {
                            smem_B[kl + j][n] = __float2bfloat16(0.0f);
                        }
                    }
                }
            }
        }

        __syncthreads();


        const unsigned short* sA = (const unsigned short*)smem_A;
        const unsigned short* sB = (const unsigned short*)smem_B;

        unsigned int frag_r0 = warp_m_offset + group_id;
        unsigned int frag_r1 = warp_m_offset + group_id + 8;
        unsigned int frag_c0 = tid * 2;
        unsigned int frag_c1 = tid * 2 + 8;

        unsigned int a0 = ((unsigned int)sA[frag_r0 * a_stride + frag_c0 + 1] << 16) |
                          (unsigned int)sA[frag_r0 * a_stride + frag_c0];
        unsigned int a1 = ((unsigned int)sA[frag_r1 * a_stride + frag_c0 + 1] << 16) |
                          (unsigned int)sA[frag_r1 * a_stride + frag_c0];
        unsigned int a2 = ((unsigned int)sA[frag_r0 * a_stride + frag_c1 + 1] << 16) |
                          (unsigned int)sA[frag_r0 * a_stride + frag_c1];
        unsigned int a3 = ((unsigned int)sA[frag_r1 * a_stride + frag_c1 + 1] << 16) |
                          (unsigned int)sA[frag_r1 * a_stride + frag_c1];

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

        __syncthreads();
    }


    #pragma unroll
    for (int n_tile = 0; n_tile < 8; n_tile++) {
        unsigned int base_n = cta_n + n_tile * 8;
        unsigned int col0 = base_n + (tid * 2);
        unsigned int col1 = col0 + 1;
        unsigned int row0 = cta_m + warp_m_offset + group_id;
        unsigned int row1 = row0 + 8;

        bool row0_valid = (int)(warp_m_offset + group_id + cta_m_local) < M_expert;
        bool row1_valid = (int)(warp_m_offset + group_id + 8 + cta_m_local) < M_expert;

        if (row0_valid && col0 < N) C[row0 * N + col0] = __float2bfloat16(acc[n_tile][0]);
        if (row0_valid && col1 < N) C[row0 * N + col1] = __float2bfloat16(acc[n_tile][1]);
        if (row1_valid && col0 < N) C[row1 * N + col0] = __float2bfloat16(acc[n_tile][2]);
        if (row1_valid && col1 < N) C[row1 * N + col1] = __float2bfloat16(acc[n_tile][3]);
    }
}

// 2026-09-25: moe_w4a16_grouped_gemm_ptrtable(_relu2): B_e and its scales come from per-expert pointer tables
// ([N, K / 2] and [N, K / 16]), scale2 from scale2_vals[e], and A row r is sorted_token_ids[r] (row r itself when
// sorted_token_ids is null). Tiles are 64 x 128 x 64 (M x N x K); grid (ceil(N / 128), max_m_tiles, num_experts),
// block 128 (ops::moe_w4a16_grouped_gemm_ptrtable_n128). Assumes K % 32 == 0 for the 16-byte B copies. An expert
// with a null B pointer writes nothing. With relu2 the store writes relu(acc)^2.






__device__ __forceinline__ void moe_w4a16_grouped_gemm_ptrtable_impl(

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
,
    const bool relu2
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
    const unsigned int cta_n = blockIdx.x * N_TILE_PT;


    const unsigned char* B_expert = (const unsigned char*)B_packed_ptrs[expert_id];
    const unsigned char* S_expert = (const unsigned char*)B_scale_ptrs[expert_id];
    const float scale2 = scale2_vals[expert_id];


    if (B_expert == 0) return;

    const unsigned int warp_id = threadIdx.x / 32;
    const unsigned int lane_id = threadIdx.x % 32;
    const unsigned int group_id = lane_id >> 2;
    const unsigned int tid = lane_id & 3;

    __shared__ __nv_bfloat16 smem_A[M_TILE][K_STEP_PT + PAD];
    // 2026-09-25: B stays packed in shared memory, double-buffered for cp.async, and is dequantized straight into
    // the MMA B fragments. Each warp owns 32 distinct columns (4 n-tiles), so no B value is dequantized twice.











    __shared__ __align__(16) unsigned char smem_Bp[2][N_TILE_PT][BP_STRIDE];
    __shared__ unsigned int smem_Bs32[2][N_TILE_PT];
    // 2026-09-25: sLUT[b] packs the BF16 values of byte b's two E2M1 codes, low nibble in the low half. It is in
    // shared memory because a __constant__ read with divergent addresses is serialized.
    __shared__ unsigned int sLUT[256];

    for (unsigned int i = threadIdx.x; i < 256u; i += blockDim.x) {
        const unsigned short lo =
            __bfloat16_as_ushort(__float2bfloat16(E2M1_LUT_MOE[i & 0xFu]));
        const unsigned short hi =
            __bfloat16_as_ushort(__float2bfloat16(E2M1_LUT_MOE[i >> 4]));
        sLUT[i] = ((unsigned int)hi << 16) | (unsigned int)lo;
    }

    float acc[16][4];   // 2026-09-25: 4 m-sub-tiles x 4 n-tiles, this warp's quarter of N_TILE_PT
    #pragma unroll
    for (int i = 0; i < 16; i++) {
        acc[i][0] = 0.0f; acc[i][1] = 0.0f;
        acc[i][2] = 0.0f; acc[i][3] = 0.0f;
    }
    __syncthreads();

    const unsigned int a_stride = K_STEP_PT + PAD;
    const unsigned int M_eff = (unsigned int)M_expert;
    const unsigned int half_K = K / 2;
    const unsigned int num_groups = K / GROUP_SIZE;

#define MOE_PF(kb, buf)                                                                \
    do {                                                                               \
        _Pragma("unroll")                                                              \
        for (unsigned int ch = 0; ch < 2u; ch++) {                                     \
            const unsigned int idx  = threadIdx.x + ch * 128u;                         \
            const unsigned int n_   = idx >> 1;                                        \
            const unsigned int hf_  = idx & 1u;                                        \
            const unsigned int gn_  = cta_n + n_;                                      \
            if ((kb) < K && gn_ < N) {                                                 \
                const unsigned long long boff =                                        \
                    (unsigned long long)gn_ * half_K + ((kb) >> 1) + hf_ * 16u;        \
                moe_cp_async16(&smem_Bp[(buf)][n_][hf_ * 16u], B_expert + boff);       \
            } else {                                                                   \
                *(uint4*)&smem_Bp[(buf)][n_][hf_ * 16u] = make_uint4(0u,0u,0u,0u);     \
            }                                                                          \
        }                                                                              \
        {                                                                              \
            const unsigned int rr  = threadIdx.x;                                      \
            const unsigned int grn = cta_n + rr;                                       \
            const unsigned int sg  = (kb) / GROUP_SIZE;                                \
            unsigned int w = 0u;                                                       \
            for (unsigned int q = 0; q < K_STEP_PT / GROUP_SIZE; q++) {                \
                unsigned char sb = (grn < N && (kb) < K && (sg + q) < num_groups)      \
                    ? S_expert[(unsigned long long)grn * num_groups + sg + q]          \
                    : (unsigned char)0u;                                               \
                w |= ((unsigned int)sb) << (q * 8u);                                   \
            }                                                                          \
            smem_Bs32[(buf)][rr] = w;                                                  \
        }                                                                              \
    } while (0)

    MOE_PF(0u, 0u);
    moe_cp_async_commit();

    unsigned int buf = 0u;
    for (unsigned int k_base = 0; k_base < K; k_base += K_STEP_PT, buf ^= 1u) {
        MOE_PF(k_base + K_STEP_PT, buf ^ 1u);   // 2026-09-25: in flight during this step's MMAs
        moe_cp_async_commit();
        moe_cp_async_wait<1>();
        __syncthreads();




        {
            const unsigned int elems_per_thread = (M_TILE * K_STEP_PT) / 128;
            #pragma unroll
            for (unsigned int i = 0; i < elems_per_thread; i++) {
                unsigned int idx = threadIdx.x * elems_per_thread + i;
                unsigned int row = idx / K_STEP_PT;
                unsigned int col = idx % K_STEP_PT;
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
        }


        // 2026-09-25: The cp.async wait above covers smem_Bp and smem_Bs32; the A tile was written by this
        // iteration's threads and needs its own barrier before the MMAs read it.
        __syncthreads();



        // 2026-09-25: Warp w computes n-tiles 4w..4w+3 for all four 16-row sub-tiles; each B fragment is
        // dequantized once and reused across the sub-tiles.
        {
            const unsigned short* sA = (const unsigned short*)smem_A;
            const unsigned int rows_in_tile = M_eff - (unsigned int)cta_m_local;

            #pragma unroll
            for (unsigned int ks = 0; ks < K_STEP_PT; ks += 16u) {
                const unsigned int sgi = ks / GROUP_SIZE;


                unsigned int b0[4], b1[4];
                #pragma unroll
                for (unsigned int nt = 0; nt < 4u; nt++) {
                    const unsigned int n_col = (warp_id * 4u + nt) * 8u + group_id;

                    __nv_fp8_e4m3 f8;
                    *(unsigned char*)&f8 =
                        (unsigned char)((smem_Bs32[buf][n_col] >> (sgi * 8u)) & 0xFFu);
                    const __nv_bfloat162 sv =
                        __bfloat162bfloat162(__float2bfloat16((float)f8 * scale2));

                    // 2026-09-25: A packed byte holds k (low nibble) and k + 1, the pair an m16n8k16 B
                    // fragment register takes for even k, so one byte gives one register.

                    const unsigned int by0 = (ks >> 1) + tid;
                    const unsigned int by1 = (ks >> 1) + tid + 4u;
                    const unsigned int p0 = sLUT[smem_Bp[buf][n_col][by0]];
                    const unsigned int p1 = sLUT[smem_Bp[buf][n_col][by1]];
                    __nv_bfloat162 v0 = *(const __nv_bfloat162*)&p0;
                    __nv_bfloat162 v1 = *(const __nv_bfloat162*)&p1;
                    v0 = __hmul2(v0, sv);
                    v1 = __hmul2(v1, sv);
                    b0[nt] = *(const unsigned int*)&v0;
                    b1[nt] = *(const unsigned int*)&v1;
                }


                #pragma unroll
                for (unsigned int m = 0; m < 4u; m++) {
                    const unsigned int wm = m * 16u;
                    if (wm >= rows_in_tile) continue;

                    const unsigned int r0 = wm + group_id, r1 = wm + group_id + 8u;
                    const unsigned int c0 = ks + tid * 2u, c1 = ks + tid * 2u + 8u;
                    unsigned int a0 = ((unsigned int)sA[r0*a_stride + c0+1] << 16) | sA[r0*a_stride + c0];
                    unsigned int a1 = ((unsigned int)sA[r1*a_stride + c0+1] << 16) | sA[r1*a_stride + c0];
                    unsigned int a2 = ((unsigned int)sA[r0*a_stride + c1+1] << 16) | sA[r0*a_stride + c1];
                    unsigned int a3 = ((unsigned int)sA[r1*a_stride + c1+1] << 16) | sA[r1*a_stride + c1];

                    #pragma unroll
                    for (unsigned int nt = 0; nt < 4u; nt++) {
                        asm volatile(
                            "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
                            "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%10,%11,%12,%13};"
                            : "=f"(acc[m*4+nt][0]), "=f"(acc[m*4+nt][1]),
                              "=f"(acc[m*4+nt][2]), "=f"(acc[m*4+nt][3])
                            : "r"(a0), "r"(a1), "r"(a2), "r"(a3),
                              "r"(b0[nt]), "r"(b1[nt]),
                              "f"(acc[m*4+nt][0]), "f"(acc[m*4+nt][1]),
                              "f"(acc[m*4+nt][2]), "f"(acc[m*4+nt][3]));
                    }
                }
            }
        }

        __syncthreads();
    }

    // 2026-09-25: acc[m * 4 + nt] holds rows m * 16 + group_id (+ 8), columns (4w + nt) * 8 + tid * 2 (+ 1).

    #pragma unroll
    for (unsigned int m = 0; m < 4u; m++) {
        const unsigned int wm = m * 16u;
        #pragma unroll
        for (unsigned int nt = 0; nt < 4u; nt++) {
            const unsigned int base_n = cta_n + (warp_id * 4u + nt) * 8u;
            const unsigned int col0 = base_n + tid * 2u;
            const unsigned int col1 = col0 + 1u;
            const unsigned int lr0 = wm + group_id;
            const unsigned int lr1 = lr0 + 8u;
            const unsigned int row0 = cta_m + lr0;
            const unsigned int row1 = cta_m + lr1;
            const bool v0 = (int)(lr0 + cta_m_local) < M_expert;
            const bool v1 = (int)(lr1 + cta_m_local) < M_expert;
            float o0 = acc[m*4+nt][0], o1 = acc[m*4+nt][1];
            float o2 = acc[m*4+nt][2], o3 = acc[m*4+nt][3];
            if (relu2) {
                // 2026-09-25: relu^2 on the FP32 accumulator; prefill_sorted then skips the separate
                // relu_squared_inplace pass.

                o0 = o0 > 0.f ? o0 * o0 : 0.f;
                o1 = o1 > 0.f ? o1 * o1 : 0.f;
                o2 = o2 > 0.f ? o2 * o2 : 0.f;
                o3 = o3 > 0.f ? o3 * o3 : 0.f;
            }
            if (v0 && col0 < N) C[row0 * N + col0] = __float2bfloat16(o0);
            if (v0 && col1 < N) C[row0 * N + col1] = __float2bfloat16(o1);
            if (v1 && col0 < N) C[row1 * N + col0] = __float2bfloat16(o2);
            if (v1 && col1 < N) C[row1 * N + col1] = __float2bfloat16(o3);
        }
    }
#undef MOE_PF
}

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
    moe_w4a16_grouped_gemm_ptrtable_impl(A, B_packed_ptrs, B_scale_ptrs, scale2_vals, C, expert_offsets, sorted_token_ids, num_experts, N, K, false);
}

// 2026-09-25: The ptrtable GEMM with the relu^2 store, for the expert up projection.
extern "C" __global__ void moe_w4a16_grouped_gemm_ptrtable_relu2(

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
    moe_w4a16_grouped_gemm_ptrtable_impl(A, B_packed_ptrs, B_scale_ptrs, scale2_vals, C, expert_offsets, sorted_token_ids, num_experts, N, K, true);
}

// 2026-09-25: moe_w4a16_grouped_gemm_ptrtable_t: per-expert B_packed [K / 2, N] and B_scale [K / 16, N], A rows
// as in the ptrtable form. Tiles are 64 x 64 x 16 (M x N x K) with no pipeline, so the grid needs ceil(N / 64)
// x-blocks.




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
    const unsigned int cta_n = blockIdx.x * N_TILE;

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
    __shared__ __nv_bfloat16 smem_B[K_STEP][N_TILE + PAD];

    float acc[8][4];
    #pragma unroll
    for (int i = 0; i < 8; i++) {
        acc[i][0] = 0.0f; acc[i][1] = 0.0f;
        acc[i][2] = 0.0f; acc[i][3] = 0.0f;
    }

    const unsigned int a_stride = K_STEP + PAD;
    const unsigned int b_stride = N_TILE + PAD;
    const unsigned int M_eff = (unsigned int)M_expert;

    for (unsigned int k_base = 0; k_base < K; k_base += K_STEP) {
        {
            const unsigned int elems_per_thread = (M_TILE * K_STEP) / 128;
            #pragma unroll
            for (unsigned int i = 0; i < elems_per_thread; i++) {
                unsigned int idx = threadIdx.x * elems_per_thread + i;
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
            const unsigned int elems_per_thread = (K_STEP * N_TILE) / 128;
            unsigned int scale_group = k_base / GROUP_SIZE;

            #pragma unroll
            for (unsigned int i = 0; i < elems_per_thread; i++) {
                unsigned int idx = threadIdx.x * elems_per_thread + i;
                unsigned int k = idx / N_TILE;
                unsigned int n = idx % N_TILE;
                unsigned int gk = k_base + k;
                unsigned int gn = cta_n + n;

                if (gk < K && gn < N) {
                    unsigned int k_pair = gk / 2;
                    unsigned char packed_byte = B_expert[(unsigned long long)k_pair * N + gn];
                    unsigned int nibble = (gk & 1) ? (packed_byte >> 4) : (packed_byte & 0xF);

                    unsigned char scale_byte = S_expert[(unsigned long long)scale_group * N + gn];
                    float fp8_val;
                    {
                        __nv_fp8_e4m3 fp8;
                        *(unsigned char*)&fp8 = scale_byte;
                        fp8_val = (float)fp8;
                    }

                    float dequant_val = E2M1_LUT_MOE[nibble] * fp8_val * scale2;
                    smem_B[k][n] = __float2bfloat16(dequant_val);
                } else {
                    smem_B[k][n] = __float2bfloat16(0.0f);
                }
            }
        }

        __syncthreads();

        const unsigned short* sA = (const unsigned short*)smem_A;
        const unsigned short* sB = (const unsigned short*)smem_B;

        unsigned int frag_r0 = warp_m_offset + group_id;
        unsigned int frag_r1 = warp_m_offset + group_id + 8;
        unsigned int frag_c0 = tid * 2;
        unsigned int frag_c1 = tid * 2 + 8;

        unsigned int a0 = ((unsigned int)sA[frag_r0 * a_stride + frag_c0 + 1] << 16) |
                          (unsigned int)sA[frag_r0 * a_stride + frag_c0];
        unsigned int a1 = ((unsigned int)sA[frag_r1 * a_stride + frag_c0 + 1] << 16) |
                          (unsigned int)sA[frag_r1 * a_stride + frag_c0];
        unsigned int a2 = ((unsigned int)sA[frag_r0 * a_stride + frag_c1 + 1] << 16) |
                          (unsigned int)sA[frag_r0 * a_stride + frag_c1];
        unsigned int a3 = ((unsigned int)sA[frag_r1 * a_stride + frag_c1 + 1] << 16) |
                          (unsigned int)sA[frag_r1 * a_stride + frag_c1];

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

        __syncthreads();
    }

    #pragma unroll
    for (int n_tile = 0; n_tile < 8; n_tile++) {
        unsigned int base_n = cta_n + n_tile * 8;
        unsigned int col0 = base_n + (tid * 2);
        unsigned int col1 = col0 + 1;
        unsigned int row0 = cta_m + warp_m_offset + group_id;
        unsigned int row1 = row0 + 8;
        bool row0_valid = (int)(warp_m_offset + group_id + cta_m_local) < M_expert;
        bool row1_valid = (int)(warp_m_offset + group_id + 8 + cta_m_local) < M_expert;

        if (row0_valid && col0 < N) C[row0 * N + col0] = __float2bfloat16(acc[n_tile][0]);
        if (row0_valid && col1 < N) C[row0 * N + col1] = __float2bfloat16(acc[n_tile][1]);
        if (row1_valid && col0 < N) C[row1 * N + col0] = __float2bfloat16(acc[n_tile][2]);
        if (row1_valid && col1 < N) C[row1 * N + col1] = __float2bfloat16(acc[n_tile][3]);
    }
}
