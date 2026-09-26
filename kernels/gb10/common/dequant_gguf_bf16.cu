// SPDX-License-Identifier: MIT OR Apache-2.0

// 2026-09-25: Raw packed GGUF quant blocks -> BF16 on the device, for Q8_0, Q4_K, Q6_K,
// Q2_K, Q3_K and the Q2_0 group-N type (ggml id 42).
//
// Owner: gb10 kernels.
// Invariants:
// - One CUDA block per GGUF (super-)block, with a stride loop over its elements; host
//   launches use grid (n_blocks, 1, 1) and block (256, 1, 1). Output is contiguous BF16,
//   QK elements per block (32, 256, or the Q2_0 group size).
// - The block byte stride is a parameter, so Q2_0 group 128 (34 B) and group 64 (18 B)
//   share one kernel.
// - Each kernel evaluates the same f32 expression, in the same operation order, as its CPU
//   twin in model-weights gguf/dequant_cpu/blocks.rs; the common build passes
//   --fmad=false, so no product is fused into an FMA.


#include <cuda_bf16.h>
#include <cuda_fp16.h>



// 2026-09-25: Little-endian IEEE fp16 (2 bytes) -> f32.
__device__ __forceinline__ float dq_rd_f16(const unsigned char* p) {
    unsigned short bits = (unsigned short)p[0] | ((unsigned short)p[1] << 8);
    return __half2float(__ushort_as_half(bits));
}

// 2026-09-25: Q4_K / Q5_K 6-bit packed scale + min unpack (get_scale_min_k4 on the CPU side).
__device__ __forceinline__ void dq_scale_min_k4(
    int j, const unsigned char* q, unsigned char* sc, unsigned char* mn) {
    if (j < 4) {
        *sc = q[j] & 63;
        *mn = q[j + 4] & 63;
    } else {
        *sc = (q[j + 4] & 0x0F) | ((q[j - 4] >> 6) << 4);
        *mn = (q[j + 4] >> 4)   | ((q[j]     >> 6) << 4);
    }
}

// 2026-09-25: Q8_0 { f16 d; i8 qs[32] }, QK = 32, 34 B. value = qs * d.

extern "C" __global__ void dequant_q8_0_to_bf16(
    const unsigned char* __restrict__ blocks,
    __nv_bfloat16* __restrict__ out,
    unsigned int n_blocks,
    unsigned int block_bytes)
{
    unsigned int b = blockIdx.x;
    if (b >= n_blocks) return;
    const unsigned char* blk = blocks + (unsigned long long)b * block_bytes;
    float d = dq_rd_f16(blk);
    const signed char* qs = (const signed char*)(blk + 2);
    __nv_bfloat16* o = out + (unsigned long long)b * 32u;
    for (unsigned int j = threadIdx.x; j < 32u; j += blockDim.x) {
        o[j] = __float2bfloat16((float)qs[j] * d);
    }
}

// 2026-09-25: Q4_K { f16 d; f16 dmin; u8 scales[12]; u8 qs[128] }, QK = 256, 144 B.
// Four chunks of 64: chunk c uses scale index 2c for its 32 low nibbles, then 2c + 1 for
// the 32 high nibbles. value = d * sc * nibble - dmin * m.
extern "C" __global__ void dequant_q4_k_to_bf16(
    const unsigned char* __restrict__ blocks,
    __nv_bfloat16* __restrict__ out,
    unsigned int n_blocks,
    unsigned int block_bytes)
{
    unsigned int b = blockIdx.x;
    if (b >= n_blocks) return;
    const unsigned char* blk = blocks + (unsigned long long)b * block_bytes;
    float d    = dq_rd_f16(blk);
    float dmin = dq_rd_f16(blk + 2);
    const unsigned char* scales = blk + 4;
    const unsigned char* qs     = blk + 16;
    __nv_bfloat16* o = out + (unsigned long long)b * 256u;

    for (unsigned int y = threadIdx.x; y < 256u; y += blockDim.x) {
        unsigned int c    = y >> 6;
        unsigned int half = (y >> 5) & 1u;
        unsigned int l    = y & 31u;
        int is = (int)(2u * c + half);
        unsigned char sc, mn;
        dq_scale_min_k4(is, scales, &sc, &mn);
        unsigned char byte = qs[c * 32u + l];
        unsigned int nib = half ? (byte >> 4) : (byte & 0x0F);
        float v = d * (float)sc * (float)nib - dmin * (float)mn;
        o[y] = __float2bfloat16(v);
    }
}

// 2026-09-25: Q6_K { u8 ql[128]; u8 qh[64]; i8 scales[16]; f16 d }, QK = 256, 210 B.
// Two 128-element halves; in a half, group g of 32 uses scale sco + is + 2g with
// is = l / 16. The 6-bit quant is centred by -32. value = d * sc * q.
extern "C" __global__ void dequant_q6_k_to_bf16(
    const unsigned char* __restrict__ blocks,
    __nv_bfloat16* __restrict__ out,
    unsigned int n_blocks,
    unsigned int block_bytes)
{
    unsigned int b = blockIdx.x;
    if (b >= n_blocks) return;
    const unsigned char* blk = blocks + (unsigned long long)b * block_bytes;
    const unsigned char* ql_all = blk;
    const unsigned char* qh_all = blk + 128;
    const signed char*   sc_all = (const signed char*)(blk + 192);
    float d = dq_rd_f16(blk + 208);
    __nv_bfloat16* o = out + (unsigned long long)b * 256u;

    for (unsigned int y = threadIdx.x; y < 256u; y += blockDim.x) {
        unsigned int n = y >> 7;
        unsigned int w = y & 127u;
        unsigned int g = w >> 5;
        unsigned int l = w & 31u;
        const unsigned char* ql = ql_all + n * 64u;
        const unsigned char* qh = qh_all + n * 32u;
        unsigned int sco = n * 8u;
        unsigned int is  = l >> 4;
        int q;
        switch (g) {
            case 0: q = (int)(ql[l]        & 0x0F) | (((int)(qh[l] >> 0) & 3) << 4); break;
            case 1: q = (int)(ql[l + 32]   & 0x0F) | (((int)(qh[l] >> 2) & 3) << 4); break;
            case 2: q = (int)(ql[l]         >> 4)  | (((int)(qh[l] >> 4) & 3) << 4); break;
            default:q = (int)(ql[l + 32]    >> 4)  | (((int)(qh[l] >> 6) & 3) << 4); break;
        }
        q -= 32;
        float sc = (float)sc_all[sco + is + 2u * g];
        o[y] = __float2bfloat16(d * sc * (float)q);
    }
}

// 2026-09-25: Q2_K { u8 scales[16]; u8 qs[64]; f16 d; f16 dmin }, QK = 256, 84 B.
// Two 128-element halves n; in a half, four shift groups j (2-bit codes at shift 2j),
// each of two 16-element runs (sub) with their own 4-bit scale/min byte:
// is = n*8 + 2j + sub. value = d * (sc & 0xF) * code - dmin * (sc >> 4).

extern "C" __global__ void dequant_q2_k_to_bf16(
    const unsigned char* __restrict__ blocks,
    __nv_bfloat16* __restrict__ out,
    unsigned int n_blocks,
    unsigned int block_bytes)
{
    unsigned int b = blockIdx.x;
    if (b >= n_blocks) return;
    const unsigned char* blk    = blocks + (unsigned long long)b * block_bytes;
    const unsigned char* scales = blk;
    const unsigned char* qs     = blk + 16;
    float d    = dq_rd_f16(blk + 80);
    float dmin = dq_rd_f16(blk + 82);
    __nv_bfloat16* o = out + (unsigned long long)b * 256u;

    for (unsigned int y = threadIdx.x; y < 256u; y += blockDim.x) {
        unsigned int n   = y >> 7;
        unsigned int w   = y & 127u;
        unsigned int j   = w >> 5;
        unsigned int sub = (w >> 4) & 1u;
        unsigned int l   = w & 15u;
        unsigned int is  = n * 8u + 2u * j + sub;
        unsigned char sc = scales[is];
        float dl = d * (float)(sc & 0x0F);
        float ml = dmin * (float)(sc >> 4);
        unsigned char q = qs[n * 32u + sub * 16u + l];
        int code = (int)((q >> (2u * j)) & 3u);
        o[y] = __float2bfloat16(dl * (float)code - ml);
    }
}

// 2026-09-25: Q3_K { u8 hmask[32]; u8 qs[64]; u8 scales[12]; f16 d }, QK = 256, 110 B.
// Same halves, shift groups and runs as Q2_K. The 6-bit scales unpack from 12 bytes
// (KM1/KM2 shuffle) into 16 int8, centred by -32. The high bit of each 3-bit code comes
// from hmask with mask m = 1 << (4n + j). value = d * (sc - 32) * (code - (hbit ? 0 : 4)).


extern "C" __global__ void dequant_q3_k_to_bf16(
    const unsigned char* __restrict__ blocks,
    __nv_bfloat16* __restrict__ out,
    unsigned int n_blocks,
    unsigned int block_bytes)
{
    unsigned int b = blockIdx.x;
    if (b >= n_blocks) return;
    const unsigned char* blk   = blocks + (unsigned long long)b * block_bytes;
    const unsigned char* hmask = blk;
    const unsigned char* qs    = blk + 32;
    const unsigned char* raw   = blk + 96;
    float d_all = dq_rd_f16(blk + 108);
    __nv_bfloat16* o = out + (unsigned long long)b * 256u;


    const unsigned int KM1 = 0x03030303u, KM2 = 0x0f0f0f0fu;
    unsigned int a0 = (unsigned int)raw[0] | ((unsigned int)raw[1] << 8) | ((unsigned int)raw[2] << 16) | ((unsigned int)raw[3] << 24);
    unsigned int a1 = (unsigned int)raw[4] | ((unsigned int)raw[5] << 8) | ((unsigned int)raw[6] << 16) | ((unsigned int)raw[7] << 24);
    unsigned int tmp = (unsigned int)raw[8] | ((unsigned int)raw[9] << 8) | ((unsigned int)raw[10] << 16) | ((unsigned int)raw[11] << 24);
    unsigned int aux[4];
    aux[2] = ((a0 >> 4) & KM2) | (((tmp >> 4) & KM1) << 4);
    aux[3] = ((a1 >> 4) & KM2) | (((tmp >> 6) & KM1) << 4);
    aux[0] = (a0 & KM2) | (((tmp >> 0) & KM1) << 4);
    aux[1] = (a1 & KM2) | (((tmp >> 2) & KM1) << 4);

    for (unsigned int y = threadIdx.x; y < 256u; y += blockDim.x) {
        unsigned int n   = y >> 7;
        unsigned int w   = y & 127u;
        unsigned int j   = w >> 5;
        unsigned int sub = (w >> 4) & 1u;
        unsigned int l   = w & 15u;
        unsigned int is  = n * 8u + 2u * j + sub;
        int sc = (int)(signed char)((aux[is >> 2] >> (8u * (is & 3u))) & 0xFFu);
        float dl = d_all * (float)(sc - 32);
        unsigned char m = (unsigned char)(1u << (4u * n + j));
        unsigned int idx = sub * 16u + l;
        int h = (hmask[idx] & m) ? 0 : 4;
        unsigned char q = qs[n * 32u + idx];
        int code = (int)((q >> (2u * j)) & 3u) - h;
        o[y] = __float2bfloat16(dl * (float)code);
    }
}

// 2026-09-25: Q2_0 group-N (ggml id 42) { f16 d; u8 qs[G/4] }, scale first. Contiguous
// low-bits-first 2-bit codes; value = (code - 1) * d. G is 128 or 64 and block_bytes is
// 2 + G/4. Also launched at runtime for packed-Q2 prefill (ops/gemv_q2.rs).
extern "C" __global__ void dequant_q2_0_gn_to_bf16(
    const unsigned char* __restrict__ blocks,
    __nv_bfloat16* __restrict__ out,
    unsigned int n_blocks,
    unsigned int group_size,
    unsigned int block_bytes)
{
    unsigned int b = blockIdx.x;
    if (b >= n_blocks) return;
    const unsigned char* blk = blocks + (unsigned long long)b * block_bytes;
    float d = dq_rd_f16(blk);
    const unsigned char* qs = blk + 2;
    __nv_bfloat16* o = out + (unsigned long long)b * group_size;
    for (unsigned int j = threadIdx.x; j < group_size; j += blockDim.x) {
        int code = (qs[j >> 2] >> (2u * (j & 3u))) & 3;
        o[j] = __float2bfloat16((float)(code - 1) * d);
    }
}
