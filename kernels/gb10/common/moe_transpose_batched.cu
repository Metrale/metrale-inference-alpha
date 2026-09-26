// SPDX-License-Identifier: MIT OR Apache-2.0

// 2026-09-25: Kernel `moe_transpose_u8_batched`: transposes one [rows, cols] uint8 matrix per expert, from
// `src_ptrs[expert]` to `dst_ptrs[expert]` as [cols, rows]. The MoE layer runs it on packed NVFP4 weights and
// their scale bytes (moe/helpers_a.rs, moe/helpers_b.rs).
//
// An expert whose src or dst pointer is NULL is skipped and its destination left unwritten.
//
// Grid (ceil(cols / 32), ceil(rows / 32), num_experts), block (32, 8): the TILE x TILE shared-memory tile of
// transpose_u8.cu, one grid z per expert.
//
// Owner: gb10 kernels.
// Invariants: none beyond the types.





#define TILE 32

extern "C" __global__ void moe_transpose_u8_batched(
    const unsigned long long* __restrict__ src_ptrs,
    const unsigned long long* __restrict__ dst_ptrs,
    unsigned int rows,
    unsigned int cols
) {
    const unsigned int expert = blockIdx.z;
    const unsigned char* src = (const unsigned char*)src_ptrs[expert];
    unsigned char* dst = (unsigned char*)dst_ptrs[expert];
    if (src == 0 || dst == 0) return;

    __shared__ unsigned char tile[TILE][TILE + 1];

    const unsigned int ix = blockIdx.x * TILE + threadIdx.x;
    const unsigned int iy_base = blockIdx.y * TILE + threadIdx.y;

    #pragma unroll
    for (int j = 0; j < TILE; j += 8) {
        unsigned int r = iy_base + j;
        if (r < rows && ix < cols) {
            tile[threadIdx.y + j][threadIdx.x] = src[r * cols + ix];
        }
    }

    __syncthreads();

    const unsigned int ox = blockIdx.y * TILE + threadIdx.x;
    const unsigned int oy_base = blockIdx.x * TILE + threadIdx.y;

    #pragma unroll
    for (int j = 0; j < TILE; j += 8) {
        unsigned int c = oy_base + j;
        if (c < cols && ox < rows) {
            dst[c * rows + ox] = tile[threadIdx.x][threadIdx.y + j];
        }
    }
}
