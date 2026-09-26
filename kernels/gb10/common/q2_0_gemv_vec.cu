// SPDX-License-Identifier: MIT OR Apache-2.0

// 2026-09-25: Q2_0 GEMVs with vector loads, the kernels the model layers launch for packed
// ternary weights: C[m, n] = sum_k A[m, k] * (code(n, k) - 1) * d(n, k / group).
//
// Same block_q2_0 layout and per-element formula as q2_0_gemv.cu, summed in a different order,
// so results are not bit-identical to it. One warp per output row, 8 rows per 256-thread block,
// grid (ceil(N/8), 1, 1) (crates/model-layers/src/layers/ops/gemv_q2_vec.rs). Each step, the
// whole block stages TILE_K activations per row into s_A with uint4 loads; each lane then reads
// one 4-byte word (16 codes) of one weight block, so a warp covers 32 / (group / 16) blocks per
// step. The warp reduces by shuffle only. q2_0_gemv_vec_batchm dequantizes each word once and
// applies it to all M staged rows. The layout test in ops/gemv_q2.rs reads this source as text.
//
// Owner: gb10 kernels.
// Invariants:
// - K is a multiple of group, and group is 64 or 128 (the loader's two sizes,
//   crates/model-weights/src/weights/gguf.rs), so one step covers exactly TILE_K activations.
// - A is read as uint4, so each A row must start 16-byte aligned.
// - q2_0_gemv_vec_batchm needs M <= MAX_M (8): s_A holds MAX_M rows and M is not checked here.
//   Its launcher in ops/gemv_q2_vec.rs splits larger batches into chunks of at most 8 rows.
// - A warp whose row is past N still stages s_A and reaches every __syncthreads(); it skips
//   only the dot product and the store.















#include <cuda_bf16.h>

#define WARP_SIZE 32
#define WARPS_PER_BLOCK 8
#define BLOCK_SIZE (WARP_SIZE * WARPS_PER_BLOCK)
#define TILE_K 512
#define TILE_U4 (TILE_K / 8)
#define MAX_M 8


// 2026-09-25: Little-endian fp16 -> f32, the same conversion as q2_rd_f16 in q2_0_gemv.cu.
__device__ __forceinline__ float q2v_rd_f16(const unsigned char* p) {
    unsigned short bits = (unsigned short)p[0] | ((unsigned short)p[1] << 8);
    return __half2float(__ushort_as_half(bits));
}


// 2026-09-25: Byte-assembled on purpose: a block is 2 + group/4 bytes and its codes start after
// the 2-byte scale, so blk + 2 + 4 * jg is only 2-byte aligned and a uint32 load there would be
// misaligned.
__device__ __forceinline__ unsigned int q2v_rd_u32(const unsigned char* p) {
    return (unsigned int)p[0] | ((unsigned int)p[1] << 8) |
           ((unsigned int)p[2] << 16) | ((unsigned int)p[3] << 24);
}



__device__ __forceinline__ void q2v_unpack8(uint4 v, float* out) {
    const unsigned int w[4] = {v.x, v.y, v.z, v.w};
    #pragma unroll
    for (int i = 0; i < 4; ++i) {
        __nv_bfloat16 lo, hi;
        *(unsigned short*)&lo = (unsigned short)(w[i] & 0xFFFF);
        *(unsigned short*)&hi = (unsigned short)(w[i] >> 16);
        out[i * 2]     = __bfloat162float(lo);
        out[i * 2 + 1] = __bfloat162float(hi);
    }
}




extern "C" __global__ void q2_0_gemv_vec(
    const __nv_bfloat16* __restrict__ A,
    const unsigned char* __restrict__ B,
    __nv_bfloat16* __restrict__ C,
    unsigned int N,
    unsigned int K,
    unsigned int group)
{
    const unsigned int warp = threadIdx.x / WARP_SIZE;
    const unsigned int lane = threadIdx.x % WARP_SIZE;
    const unsigned int n = blockIdx.x * WARPS_PER_BLOCK + warp;

    const unsigned int blocks_per_row = K / group;
    const unsigned int block_bytes = 2u + group / 4u;
    const unsigned int upb = group / 16u;
    const unsigned int bpi = WARP_SIZE / upb;
    const unsigned int block_in_step = lane / upb;
    const unsigned int jg = lane % upb;


    const unsigned char* row =
        (n < N) ? B + (unsigned long long)n * blocks_per_row * block_bytes : B;

    __shared__ __align__(16) __nv_bfloat16 s_A[TILE_K];

    float acc = 0.0f;

    for (unsigned int tb0 = 0; tb0 < blocks_per_row; tb0 += bpi) {
        const unsigned int base_k = tb0 * group;


        for (unsigned int u = threadIdx.x; u < TILE_U4; u += BLOCK_SIZE) {
            const unsigned int k = base_k + u * 8u;
            uint4 val;
            if (k < K) {
                val = *((const uint4*)A + k / 8u);
            } else {
                val.x = val.y = val.z = val.w = 0u;
            }
            *((uint4*)s_A + u) = val;
        }
        __syncthreads();

        if (n < N) {
            const unsigned int b = tb0 + block_in_step;
            if (b < blocks_per_row) {
                const unsigned char* blk = row + (unsigned long long)b * block_bytes;
                const float d = q2v_rd_f16(blk);
                const unsigned int codes = q2v_rd_u32(blk + 2 + jg * 4u);

                const unsigned int kl = block_in_step * group + jg * 16u;
                float a[16];
                q2v_unpack8(*(const uint4*)(s_A + kl),      a);
                q2v_unpack8(*(const uint4*)(s_A + kl + 8u), a + 8);

                #pragma unroll
                for (int j = 0; j < 16; ++j) {
                    const int code = (int)((codes >> (2 * j)) & 3u);
                    acc += a[j] * (float)(code - 1) * d;
                }
            }
        }
        __syncthreads();
    }

    #pragma unroll
    for (int off = WARP_SIZE / 2; off > 0; off >>= 1) {
        acc += __shfl_down_sync(0xFFFFFFFF, acc, off);
    }
    if (lane == 0 && n < N) {
        C[n] = __float2bfloat16(acc);
    }
}









extern "C" __global__ void q2_0_gemv_vec_batchm(
    const __nv_bfloat16* __restrict__ A,
    const unsigned char* __restrict__ B,
    __nv_bfloat16* __restrict__ C,
    unsigned int N,
    unsigned int K,
    unsigned int group,
    unsigned int M)
{
    const unsigned int warp = threadIdx.x / WARP_SIZE;
    const unsigned int lane = threadIdx.x % WARP_SIZE;
    const unsigned int n = blockIdx.x * WARPS_PER_BLOCK + warp;

    const unsigned int blocks_per_row = K / group;
    const unsigned int block_bytes = 2u + group / 4u;
    const unsigned int upb = group / 16u;
    const unsigned int bpi = WARP_SIZE / upb;
    const unsigned int block_in_step = lane / upb;
    const unsigned int jg = lane % upb;

    const unsigned char* row =
        (n < N) ? B + (unsigned long long)n * blocks_per_row * block_bytes : B;

    __shared__ __align__(16) __nv_bfloat16 s_A[MAX_M * TILE_K];

    float acc[MAX_M];
    #pragma unroll
    for (int m = 0; m < MAX_M; ++m) acc[m] = 0.0f;

    for (unsigned int tb0 = 0; tb0 < blocks_per_row; tb0 += bpi) {
        const unsigned int base_k = tb0 * group;


        const unsigned int total_u4 = TILE_U4 * M;
        for (unsigned int u = threadIdx.x; u < total_u4; u += BLOCK_SIZE) {
            const unsigned int mm = u / TILE_U4;
            const unsigned int uu = u % TILE_U4;
            const unsigned int k = base_k + uu * 8u;
            uint4 val;
            if (k < K) {
                val = *((const uint4*)(A + (unsigned long long)mm * K) + k / 8u);
            } else {
                val.x = val.y = val.z = val.w = 0u;
            }
            *((uint4*)s_A + mm * TILE_U4 + uu) = val;
        }
        __syncthreads();

        if (n < N) {
            const unsigned int b = tb0 + block_in_step;
            if (b < blocks_per_row) {
                const unsigned char* blk = row + (unsigned long long)b * block_bytes;
                const float d = q2v_rd_f16(blk);
                const unsigned int codes = q2v_rd_u32(blk + 2 + jg * 4u);

                float wv[16];
                #pragma unroll
                for (int j = 0; j < 16; ++j) {
                    wv[j] = (float)((int)((codes >> (2 * j)) & 3u) - 1) * d;
                }

                const unsigned int kl = block_in_step * group + jg * 16u;
                #pragma unroll
                for (int m = 0; m < MAX_M; ++m) {
                    if ((unsigned int)m >= M) continue;
                    const __nv_bfloat16* sa = s_A + (unsigned int)m * TILE_K + kl;
                    float a[16];
                    q2v_unpack8(*(const uint4*)sa,      a);
                    q2v_unpack8(*(const uint4*)(sa + 8u), a + 8);
                    #pragma unroll
                    for (int j = 0; j < 16; ++j) acc[m] += a[j] * wv[j];
                }
            }
        }
        __syncthreads();
    }

    #pragma unroll
    for (int m = 0; m < MAX_M; ++m) {
        if ((unsigned int)m >= M) continue;
        float v = acc[m];
        #pragma unroll
        for (int off = WARP_SIZE / 2; off > 0; off >>= 1) {
            v += __shfl_down_sync(0xFFFFFFFF, v, off);
        }
        if (lane == 0 && n < N) {
            C[(unsigned long long)m * N + n] = __float2bfloat16(v);
        }
    }
}
