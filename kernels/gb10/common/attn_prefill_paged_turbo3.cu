// SPDX-License-Identifier: MIT OR Apache-2.0


// 2026-09-25: Kernels `attn_prefill_paged_turbo3` and `_64`: paged prefill flash attention over a turbo3 KV cache.
//
// Each cache block (stride `tq3_bsb` bytes) holds 3-bit codebook indices, 8 per 3 bytes, then from byte `tq3_dsb` one
// FP8 E4M3 scale per 16 values, as reshape_and_cache_flash_turbo3 writes them; the LUT is TURBO3_CODEBOOK from the same
// file (reshape_and_cache_turbo.cu). The loader below dequantizes to BF16 while it fills shared memory, and the body is
// prefill_paged_compute.cuh.
//
// Owner: gb10 kernels.
// Invariants: none beyond the types.




#include <cuda_bf16.h>
#include <cuda_fp8.h>

#define NVFP4_GROUP_SIZE 16

__device__ __forceinline__ float fp8e4m3_f32(__nv_fp8_storage_t b) {
    return __half2float(__nv_cvt_fp8_to_halfraw(b, __NV_E4M3));
}

// 2026-09-25: 8 indices in 3 bytes: b0 = v0|(v1<<3)|(v2<<6); b1 = (v2>>2)|(v3<<1)|(v4<<4)|(v5<<7);
// b2 = (v5>>1)|(v6<<2)|(v7<<5).

#define LOAD_KV_TILE(cache, bt, smem, kv_s, kv_l, kvh, t, stride) \
    do { \
        const unsigned int _cpr = HDIM / 8; \
        const unsigned int _nkv_hd = num_kv_heads * head_dim; \
        for (unsigned int _i = t; _i < TILE_CHUNKS; _i += (stride)) { \
            unsigned int _row = _i / _cpr, _col = (_i % _cpr) * 8; \
            unsigned int _pos = (kv_s) + _row; \
            if (_pos < (kv_l)) { \
                unsigned int _lb = _pos / cache_block_size; \
                unsigned int _bo = _pos % cache_block_size; \
                unsigned int _pb = (unsigned int)(bt)[_lb]; \
                const unsigned char* _blk = (const unsigned char*)(cache) \
                    + (unsigned long long)_pb * tq3_bsb; \
                \
                const unsigned char* _dp = _blk \
                    + (unsigned long long)_bo * _nkv_hd * 3 / 8 \
                    + (unsigned long long)(kvh) * head_dim * 3 / 8 + (_col / 8) * 3; \
                const unsigned int _sg = head_dim / NVFP4_GROUP_SIZE; \
                const unsigned char* _sp = _blk + tq3_dsb \
                    + (unsigned long long)_bo * num_kv_heads * _sg \
                    + (unsigned long long)(kvh) * _sg + _col / NVFP4_GROUP_SIZE; \
                \
                float _gs = fp8e4m3_f32((__nv_fp8_storage_t)*_sp); \
                unsigned char _b0 = _dp[0], _b1 = _dp[1], _b2 = _dp[2]; \
                __nv_bfloat16 _v[8]; \
                _v[0] = __float2bfloat16(e2m1_lut[(_b0)                & 0x7] * _gs); \
                _v[1] = __float2bfloat16(e2m1_lut[(_b0 >> 3)           & 0x7] * _gs); \
                _v[2] = __float2bfloat16(e2m1_lut[((_b0 >> 6) | (_b1 << 2)) & 0x7] * _gs); \
                _v[3] = __float2bfloat16(e2m1_lut[(_b1 >> 1)           & 0x7] * _gs); \
                _v[4] = __float2bfloat16(e2m1_lut[(_b1 >> 4)           & 0x7] * _gs); \
                _v[5] = __float2bfloat16(e2m1_lut[((_b1 >> 7) | (_b2 << 1)) & 0x7] * _gs); \
                _v[6] = __float2bfloat16(e2m1_lut[(_b2 >> 2)           & 0x7] * _gs); \
                _v[7] = __float2bfloat16(e2m1_lut[(_b2 >> 5)           & 0x7] * _gs); \
                *((uint4*)&(smem)[_row][_col]) = *((uint4*)_v); \
            } else { *((uint4*)&(smem)[_row][_col]) = make_uint4(0,0,0,0); } \
        } \
    } while(0)

#define KERNEL_NAME attn_prefill_paged_turbo3
#define K_CACHE_TYPE const unsigned char* __restrict__
#define V_CACHE_TYPE const unsigned char* __restrict__
#define KERNEL_EXTRA_PARAMS \
    , const float inv_sqrt_d \
    , const unsigned long long tq3_bsb \
    , const unsigned long long tq3_dsb
#define KERNEL_PREAMBLE \
    __shared__ float e2m1_lut[8]; \
    if (tid < 8) { \
        const float _lut[8] = { -2.1520f, -1.3440f, -0.7560f, -0.2451f, 0.2451f, 0.7560f, 1.3440f, 2.1520f }; \
        e2m1_lut[tid] = _lut[tid]; \
    } \
    __syncthreads();

#include "prefill_paged_compute.cuh"
