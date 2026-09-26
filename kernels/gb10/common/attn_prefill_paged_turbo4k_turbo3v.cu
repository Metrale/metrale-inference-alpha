// SPDX-License-Identifier: MIT OR Apache-2.0

// 2026-09-25: Kernels `attn_prefill_paged_turbo4k_turbo3v` and `_64`: paged prefill flash attention with a turbo4 K
// cache and a turbo3 V cache.
//
// The body is prefill_paged_compute_asym.cuh. Each cache has its own block stride and scale-section offset
// (`tq4_k_bsb`/`tq4_k_dsb`, `tq3_v_bsb`/`tq3_v_dsb`). The K loader is attn_prefill_paged_turbo4.cu's and the V loader
// attn_prefill_paged_turbo3.cu's, each with its own codebook LUT (`k_lut`, `v_lut`); both dequantize to BF16 while
// they fill shared memory.
//
// Owner: gb10 kernels.
// Invariants: none beyond the types.






#include <cuda_bf16.h>
#include <cuda_fp8.h>

#define NVFP4_GROUP_SIZE 16

__device__ __forceinline__ float fp8e4m3_f32_asym_pf(__nv_fp8_storage_t b) {
    return __half2float(__nv_cvt_fp8_to_halfraw(b, __NV_E4M3));
}



#define LOAD_K_TILE(cache, bt, smem, kv_s, kv_l, kvh, t, stride) \
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
                    + (unsigned long long)_pb * tq4_k_bsb; \
                const unsigned char* _dp = _blk \
                    + (unsigned long long)_bo * _nkv_hd / 2 \
                    + (unsigned long long)(kvh) * head_dim / 2 + _col / 2; \
                const unsigned int _sg = head_dim / NVFP4_GROUP_SIZE; \
                const unsigned char* _sp = _blk + tq4_k_dsb \
                    + (unsigned long long)_bo * num_kv_heads * _sg \
                    + (unsigned long long)(kvh) * _sg + _col / NVFP4_GROUP_SIZE; \
                float _gs = fp8e4m3_f32_asym_pf((__nv_fp8_storage_t)*_sp); \
                unsigned int _pk = *(const unsigned int*)_dp; \
                __nv_bfloat16 _v[8]; \
                _v[0] = __float2bfloat16(k_lut[(_pk)       & 0xF] * _gs); \
                _v[1] = __float2bfloat16(k_lut[(_pk >> 4)  & 0xF] * _gs); \
                _v[2] = __float2bfloat16(k_lut[(_pk >> 8)  & 0xF] * _gs); \
                _v[3] = __float2bfloat16(k_lut[(_pk >> 12) & 0xF] * _gs); \
                _v[4] = __float2bfloat16(k_lut[(_pk >> 16) & 0xF] * _gs); \
                _v[5] = __float2bfloat16(k_lut[(_pk >> 20) & 0xF] * _gs); \
                _v[6] = __float2bfloat16(k_lut[(_pk >> 24) & 0xF] * _gs); \
                _v[7] = __float2bfloat16(k_lut[_pk >> 28]         * _gs); \
                *((uint4*)&(smem)[_row][_col]) = *((uint4*)_v); \
            } else { *((uint4*)&(smem)[_row][_col]) = make_uint4(0,0,0,0); } \
        } \
    } while(0)




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
                    + (unsigned long long)_pb * tq3_v_bsb; \
                const unsigned char* _dp = _blk \
                    + (unsigned long long)_bo * _nkv_hd * 3 / 8 \
                    + (unsigned long long)(kvh) * head_dim * 3 / 8 + (_col / 8) * 3; \
                const unsigned int _sg = head_dim / NVFP4_GROUP_SIZE; \
                const unsigned char* _sp = _blk + tq3_v_dsb \
                    + (unsigned long long)_bo * num_kv_heads * _sg \
                    + (unsigned long long)(kvh) * _sg + _col / NVFP4_GROUP_SIZE; \
                float _gs = fp8e4m3_f32_asym_pf((__nv_fp8_storage_t)*_sp); \
                unsigned char _b0 = _dp[0], _b1 = _dp[1], _b2 = _dp[2]; \
                __nv_bfloat16 _v[8]; \
                _v[0] = __float2bfloat16(v_lut[(_b0)                & 0x7] * _gs); \
                _v[1] = __float2bfloat16(v_lut[(_b0 >> 3)           & 0x7] * _gs); \
                _v[2] = __float2bfloat16(v_lut[((_b0 >> 6) | (_b1 << 2)) & 0x7] * _gs); \
                _v[3] = __float2bfloat16(v_lut[(_b1 >> 1)           & 0x7] * _gs); \
                _v[4] = __float2bfloat16(v_lut[(_b1 >> 4)           & 0x7] * _gs); \
                _v[5] = __float2bfloat16(v_lut[((_b1 >> 7) | (_b2 << 1)) & 0x7] * _gs); \
                _v[6] = __float2bfloat16(v_lut[(_b2 >> 2)           & 0x7] * _gs); \
                _v[7] = __float2bfloat16(v_lut[(_b2 >> 5) & 0x7] * _gs); \
                *((uint4*)&(smem)[_row][_col]) = *((uint4*)_v); \
            } else { *((uint4*)&(smem)[_row][_col]) = make_uint4(0,0,0,0); } \
        } \
    } while(0)

#define KERNEL_NAME attn_prefill_paged_turbo4k_turbo3v
#define K_CACHE_TYPE const unsigned char* __restrict__
#define V_CACHE_TYPE const unsigned char* __restrict__
#define KERNEL_EXTRA_PARAMS \
    , const float inv_sqrt_d \
    , const unsigned long long tq4_k_bsb \
    , const unsigned long long tq4_k_dsb \
    , const unsigned long long tq3_v_bsb \
    , const unsigned long long tq3_v_dsb
#define KERNEL_PREAMBLE \
    __shared__ float k_lut[16]; \
    if (tid < 16) { \
        const float _kl[16] = { \
            -2.7326f, -2.0690f, -1.6180f, -1.2562f, -0.9423f, -0.6568f, -0.3880f, -0.1284f, \
             0.1284f,  0.3880f,  0.6568f,  0.9423f,  1.2562f,  1.6180f,  2.0690f,  2.7326f \
        }; \
        k_lut[tid] = _kl[tid]; \
    } \
    __shared__ float v_lut[8]; \
    if (tid < 8) { \
        const float _vl[8] = { -2.1520f, -1.3440f, -0.7560f, -0.2451f, 0.2451f, 0.7560f, 1.3440f, 2.1520f }; \
        v_lut[tid] = _vl[tid]; \
    } \
    __syncthreads();

#include "prefill_paged_compute_asym.cuh"
