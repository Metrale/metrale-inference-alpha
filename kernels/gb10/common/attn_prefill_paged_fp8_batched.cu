// SPDX-License-Identifier: MIT OR Apache-2.0


// 2026-09-25: Kernels `attn_prefill_paged_fp8_batched` and `_64`: paged prefill flash attention over an FP8 E4M3 KV
// cache for several streams in one launch.
//
// With PREFILL_BATCHED, blockIdx.z selects the stream, which has its own block table and, when the arrays are passed,
// its own `cu_seqlens` row range and `kv_lens` entry; `q_offset`, the window and the scales are shared by all streams
// (prefill_paged_compute.cuh). Unless a build defines METRALE_ATTN_FP8_SMEM, the loader dequantizes to BF16 while it
// fills shared memory, using k_scale for K and v_scale for V.
//
// Owner: gb10 kernels.
// Invariants: none beyond the types.








#include <cuda_bf16.h>
#include <cuda_fp8.h>

__device__ __forceinline__ __nv_bfloat16 fp8_to_bf16(__nv_fp8_storage_t b, float scale) {
    float v = __half2float(__nv_cvt_fp8_to_halfraw(b, __NV_E4M3)) * scale;
    return __float2bfloat16(v);
}

#define PREFILL_BATCHED

// 2026-09-25: This file does not define METRALE_ATTN_FP8_SMEM; the first loader applies only when a build does, for
// example through METRALE_EXTRA_NVCC_FLAGS.







#ifdef METRALE_ATTN_FP8_SMEM
// 2026-09-25: Copies raw E4M3 bytes, 8 per uint2 store; the body dequantizes them at the MMA.
#define LOAD_KV_TILE(cache, bt, smem, kv_s, kv_l, kvh, t, stride) \
    do { \
        const unsigned int _cpr = HDIM / 8; \
        for (unsigned int _i = t; _i < TILE_CHUNKS; _i += (stride)) { \
            unsigned int _row = _i / _cpr, _col = (_i % _cpr) * 8; \
            unsigned int _pos = (kv_s) + _row; \
            if (_pos < (kv_l)) { \
                unsigned int _lb = _pos / cache_block_size; \
                unsigned int _bo = _pos % cache_block_size; \
                unsigned int _pb = (unsigned int)(bt)[_lb]; \
                const __nv_fp8_storage_t* _base = (const __nv_fp8_storage_t*)(cache) \
                    + (unsigned long long)_pb * fp8_cache_stride \
                    + (unsigned long long)_bo * num_kv_heads * head_dim \
                    + (unsigned long long)(kvh) * head_dim + _col; \
                *((uint2*)&(smem)[_row][_col]) = *((const uint2*)_base); \
            } else { *((uint2*)&(smem)[_row][_col]) = make_uint2(0,0); } \
        } \
    } while(0)
#else
#define LOAD_KV_TILE(cache, bt, smem, kv_s, kv_l, kvh, t, stride) \
    do { \
        const float _sc = ((const void*)(cache) == (const void*)K_cache) ? k_scale : v_scale; \
        const unsigned int _cpr = HDIM / 8; \
        for (unsigned int _i = t; _i < TILE_CHUNKS; _i += (stride)) { \
            unsigned int _row = _i / _cpr, _col = (_i % _cpr) * 8; \
            unsigned int _pos = (kv_s) + _row; \
            if (_pos < (kv_l)) { \
                unsigned int _lb = _pos / cache_block_size; \
                unsigned int _bo = _pos % cache_block_size; \
                unsigned int _pb = (unsigned int)(bt)[_lb]; \
                const __nv_fp8_storage_t* _base = (const __nv_fp8_storage_t*)(cache) \
                    + (unsigned long long)_pb * fp8_cache_stride \
                    + (unsigned long long)_bo * num_kv_heads * head_dim \
                    + (unsigned long long)(kvh) * head_dim + _col; \
                __nv_bfloat16 _v[8]; \
                for (int _j = 0; _j < 8; _j++) \
                    _v[_j] = fp8_to_bf16(_base[_j], _sc); \
                *((uint4*)&(smem)[_row][_col]) = *((uint4*)_v); \
            } else { *((uint4*)&(smem)[_row][_col]) = make_uint4(0,0,0,0); } \
        } \
    } while(0)
#endif

#define KERNEL_NAME attn_prefill_paged_fp8_batched
#define K_CACHE_TYPE const void* __restrict__
#define V_CACHE_TYPE const void* __restrict__
#define KERNEL_EXTRA_PARAMS \
    , const float inv_sqrt_d \
    , const float k_scale \
    , const float v_scale \
    , const unsigned long long fp8_cache_stride
#define KERNEL_PREAMBLE

#include "prefill_paged_compute.cuh"
