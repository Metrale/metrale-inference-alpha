// SPDX-License-Identifier: MIT OR Apache-2.0

// 2026-09-25: Transpose a [rows, cols] uint8 matrix: out[c * rows + r] = in[r * cols + c].
//
// Owner: gb10 kernels.
// Invariants: none beyond the types.
//
// Grid (ceil(cols/32), ceil(rows/32)), block (32, 8) (ops::transpose_u8 in fp8_moe.rs). Each
// block moves one 32x32 tile through shared memory; each thread copies 4 of its rows (j += 8).



#define TILE 32

extern "C" __global__ void transpose_u8(
    const unsigned char* __restrict__ in,
    unsigned char* __restrict__ out,
    unsigned int rows,
    unsigned int cols
) {
    __shared__ unsigned char tile[TILE][TILE + 1];


    unsigned int ix = blockIdx.x * TILE + threadIdx.x;
    unsigned int iy_base = blockIdx.y * TILE + threadIdx.y;


    #pragma unroll
    for (int j = 0; j < TILE; j += 8) {
        unsigned int r = iy_base + j;
        if (r < rows && ix < cols) {
            tile[threadIdx.y + j][threadIdx.x] = in[r * cols + ix];
        }
    }

    __syncthreads();


    unsigned int ox = blockIdx.y * TILE + threadIdx.x;
    unsigned int oy_base = blockIdx.x * TILE + threadIdx.y;


    #pragma unroll
    for (int j = 0; j < TILE; j += 8) {
        unsigned int c = oy_base + j;
        if (c < cols && ox < rows) {
            out[c * rows + ox] = tile[threadIdx.x][threadIdx.y + j];
        }
    }
}
