// SPDX-License-Identifier: MIT OR Apache-2.0

// 2026-09-25: Embedding-row gathers driven by device-resident token ids, so no token id
// has to be read back to the host before the gather.
//
// Owner: gb10 kernels.
// Invariants:
// - embed_table is [vocab, hidden_size] row-major; token ids are not range-checked, so the
//   caller must pass ids < vocab.
// - embed_from_argmax(_f32): one thread per element, launched as grid
//   (ceil(hidden_size / 256), 1, 1), block (256, 1, 1). Thread 0 also copies the id to
//   token_id_out for a later host readback.
// - batched_embed*: one block per token, threads stride over hidden_size.

#include <cuda_bf16.h>

extern "C" __global__ void embed_from_argmax(
    const unsigned int* __restrict__ argmax_result,
    const __nv_bfloat16* __restrict__ embed_table,
    __nv_bfloat16* __restrict__ output,
    unsigned int* __restrict__ token_id_out,
    unsigned int hidden_size
) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx == 0) {
        token_id_out[0] = argmax_result[0];
    }
    if (idx < hidden_size) {
        unsigned int token_id = argmax_result[0];
        output[idx] = embed_table[token_id * hidden_size + idx];
    }
}

// 2026-09-25: Gathers row token_ids[i] of embed_table into output row i, for num_tokens rows.





extern "C" __global__ void batched_embed(
    const unsigned int* __restrict__ token_ids,
    const __nv_bfloat16* __restrict__ embed_table,
    __nv_bfloat16* __restrict__ output,
    unsigned int hidden_size
) {
    const unsigned int token_idx = blockIdx.x;
    const unsigned int token_id = token_ids[token_idx];
    const __nv_bfloat16* src = embed_table + (unsigned long long)token_id * hidden_size;
    __nv_bfloat16* dst = output + (unsigned long long)token_idx * hidden_size;
    for (unsigned int i = threadIdx.x; i < hidden_size; i += blockDim.x) {
        dst[i] = src[i];
    }
}

// 2026-09-25: batched_embed with the BF16 rows widened to an FP32 output.
extern "C" __global__ void batched_embed_f32(
    const unsigned int* __restrict__ token_ids,
    const __nv_bfloat16* __restrict__ embed_table,
    float* __restrict__ output,
    unsigned int hidden_size
) {
    const unsigned int token_idx = blockIdx.x;
    const unsigned int token_id = token_ids[token_idx];
    const __nv_bfloat16* src = embed_table + (unsigned long long)token_id * hidden_size;
    float* dst = output + (unsigned long long)token_idx * hidden_size;
    for (unsigned int i = threadIdx.x; i < hidden_size; i += blockDim.x) {
        dst[i] = __bfloat162float(src[i]);
    }
}

// 2026-09-25: embed_from_argmax with the BF16 row widened to an FP32 output.
extern "C" __global__ void embed_from_argmax_f32(
    const unsigned int* __restrict__ argmax_result,
    const __nv_bfloat16* __restrict__ embed_table,
    float* __restrict__ output,
    unsigned int* __restrict__ token_id_out,
    unsigned int hidden_size
) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx == 0) {
        token_id_out[0] = argmax_result[0];
    }
    if (idx < hidden_size) {
        unsigned int token_id = argmax_result[0];
        output[idx] = __bfloat162float(embed_table[token_id * hidden_size + idx]);
    }
}


// 2026-09-25: batched_embed over an FP8 table: rows of FP8 E4M3 bytes with a per-row f32
// scale, the layout quantize_bf16_to_fp8 (dense_gemv_fp8w.cu) writes. Each byte is decoded
// with the same software bit-math as scl_fp8 there (NaN code -> 0), times the row scale,
// rounded to BF16. Resolved by the n-gram embedding layer (ngram_embed/embed.rs).








__device__ __forceinline__ float ngram_fp8_decode(unsigned char b) {
    unsigned int s = (b >> 7) & 1u, e = (b >> 3) & 0xFu, m = b & 0x7u; float v;
    if (e == 0u)                  v = (float)m * 0.001953125f;
    else if (e == 15u && m == 7u) v = 0.0f;
    else                          v = __uint_as_float(((e + 120u) << 23) | (m << 20));
    return s ? -v : v;
}
extern "C" __global__ void batched_embed_fp8(
    const unsigned int* __restrict__ token_ids,
    const unsigned char* __restrict__ embed_table,
    const float* __restrict__ row_scale,
    __nv_bfloat16* __restrict__ output,
    unsigned int hidden_size
) {
    const unsigned int token_idx = blockIdx.x;
    const unsigned int row = token_ids[token_idx];
    const unsigned char* src = embed_table + (unsigned long long)row * hidden_size;
    const float scale = row_scale[row];
    __nv_bfloat16* dst = output + (unsigned long long)token_idx * hidden_size;
    for (unsigned int i = threadIdx.x; i < hidden_size; i += blockDim.x) {
        dst[i] = __float2bfloat16(ngram_fp8_decode(src[i]) * scale);
    }
}
