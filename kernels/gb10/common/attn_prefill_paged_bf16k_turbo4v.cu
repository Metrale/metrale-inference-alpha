// SPDX-License-Identifier: MIT OR Apache-2.0

// 2026-09-25: Kernels `attn_prefill_paged_bf16k_turbo4v` and `_64`: paged prefill flash attention with a BF16 K cache
// and a turbo4 V cache.
//
// The body is prefill_paged_compute_asym.cuh, which takes separate K and V loaders. Each V block (stride `tq4_v_bsb`
// bytes) holds 4-bit codebook indices, two per byte, then from byte `tq4_v_dsb` one FP8 E4M3 scale per 16 values.
// The K block stride is derived from block_size, num_kv_heads and head_dim, so it needs no byte-stride argument.
//
// Owner: gb10 kernels.
// Invariants: none beyond the types.









#include <cuda_bf16.h>
#include <cuda_fp8.h>

#define NVFP4_GROUP_SIZE 16

__device__ __forceinline__ float fp8e4m3_f32_asym_pf(__nv_fp8_storage_t b) {
    return __half2float(__nv_cvt_fp8_to_halfraw(b, __NV_E4M3));
}

// 2026-09-25: The attn_prefill_paged.cu loader: one 32-row BF16 K tile by 16-byte cp.async, rows at or past `kv_l`
// zero-filled.

#define LOAD_K_TILE(cache, bt, smem, kv_s, kv_l, kvh, t, stride) \
    do { \
        const unsigned int _cpr = HDIM / 8; \
        const unsigned long long _ps = (unsigned long long)cache_block_size * num_kv_heads * head_dim; \
        const unsigned long long _rs = (unsigned long long)num_kv_heads * head_dim; \
        for (unsigned int _i = t; _i < TILE_CHUNKS; _i += (stride)) { \
            unsigned int _row = _i / _cpr, _col = (_i % _cpr) * 8; \
            unsigned int _pos = (kv_s) + _row; \
            if (_pos < (kv_l)) { \
                unsigned int _lb = _pos / cache_block_size; \
                unsigned int _bo = _pos % cache_block_size; \
                unsigned int _pb = (unsigned int)(bt)[_lb]; \
                const void* _gm = (const void*)( \
                    (cache) + _pb * _ps + _bo * _rs + (kvh) * head_dim + _col); \
                metrale_cp16(&(smem)[_row][_col], _gm); \
            } else { *((uint4*)&(smem)[_row][_col]) = make_uint4(0,0,0,0); } \
        } \
    } while(0)

// 2026-09-25: The attn_prefill_paged_turbo4.cu loader with the V-cache arguments: it dequantizes to BF16 while it fills
// shared memory, and the LUT holds the same turbo4 codebook.



#define LOAD_V_TILE(cache, bt, smem, kv_s, kv_l, kvh, t, stride) \
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
                    + (unsigned long long)_pb * tq4_v_bsb; \
                const unsigned char* _dp = _blk \
                    + (unsigned long long)_bo * _nkv_hd / 2 \
                    + (unsigned long long)(kvh) * head_dim / 2 + _col / 2; \
                const unsigned int _sg = head_dim / NVFP4_GROUP_SIZE; \
                const unsigned char* _sp = _blk + tq4_v_dsb \
                    + (unsigned long long)_bo * num_kv_heads * _sg \
                    + (unsigned long long)(kvh) * _sg + _col / NVFP4_GROUP_SIZE; \
                float _gs = fp8e4m3_f32_asym_pf((__nv_fp8_storage_t)*_sp); \
                unsigned int _pk = *(const unsigned int*)_dp; \
                __nv_bfloat16 _v[8]; \
                _v[0] = __float2bfloat16(e2m1_lut[(_pk)       & 0xF] * _gs); \
                _v[1] = __float2bfloat16(e2m1_lut[(_pk >> 4)  & 0xF] * _gs); \
                _v[2] = __float2bfloat16(e2m1_lut[(_pk >> 8)  & 0xF] * _gs); \
                _v[3] = __float2bfloat16(e2m1_lut[(_pk >> 12) & 0xF] * _gs); \
                _v[4] = __float2bfloat16(e2m1_lut[(_pk >> 16) & 0xF] * _gs); \
                _v[5] = __float2bfloat16(e2m1_lut[(_pk >> 20) & 0xF] * _gs); \
                _v[6] = __float2bfloat16(e2m1_lut[(_pk >> 24) & 0xF] * _gs); \
                _v[7] = __float2bfloat16(e2m1_lut[_pk >> 28]         * _gs); \
                *((uint4*)&(smem)[_row][_col]) = *((uint4*)_v); \
            } else { *((uint4*)&(smem)[_row][_col]) = make_uint4(0,0,0,0); } \
        } \
    } while(0)

#define KERNEL_NAME attn_prefill_paged_bf16k_turbo4v
#define K_CACHE_TYPE const __nv_bfloat16* __restrict__
#define V_CACHE_TYPE const unsigned char* __restrict__
#define KERNEL_EXTRA_PARAMS \
    , const float inv_sqrt_d \
    , const unsigned long long tq4_v_bsb \
    , const unsigned long long tq4_v_dsb
#define KERNEL_PREAMBLE \
    __shared__ float e2m1_lut[16]; \
    if (tid < 16) { \
        const float _lut[16] = { \
            -2.7326f, -2.0690f, -1.6180f, -1.2562f, -0.9423f, -0.6568f, -0.3880f, -0.1284f, \
             0.1284f,  0.3880f,  0.6568f,  0.9423f,  1.2562f,  1.6180f,  2.0690f,  2.7326f \
        }; \
        e2m1_lut[tid] = _lut[tid]; \
    } \
    __syncthreads();

#include "prefill_paged_compute_asym.cuh"
