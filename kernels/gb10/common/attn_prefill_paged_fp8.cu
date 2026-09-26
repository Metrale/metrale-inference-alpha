// SPDX-License-Identifier: MIT OR Apache-2.0


// 2026-09-25: Kernels `attn_prefill_paged_fp8` and `_64`: paged prefill flash attention over an FP8 E4M3 KV cache.
//
// Q is contiguous BF16. With METRALE_ATTN_FP8_SMEM, gb10's prefill_paged_compute.cuh (the body) keeps K/V as raw E4M3 bytes
// in shared memory and dequantizes them in registers with `fp8_to_bf16` and k_scale / v_scale right before each MMA.
// Owner: gb10 kernels.
// Invariants: none beyond the types.

#include <cuda_bf16.h>
#include <cuda_fp8.h>

__device__ __forceinline__ __nv_bfloat16 fp8_to_bf16(__nv_fp8_storage_t b, float scale) {
    float v = __half2float(__nv_cvt_fp8_to_halfraw(b, __NV_E4M3)) * scale;
    return __float2bfloat16(v);
}

// 2026-09-25: Deferring the dequant to the MMA lets the tile loads below copy raw bytes with metrale_cp16. On gb10 that
// is a cp.async, so the loads overlap compute: the body loads V during QK^T and the next K during P*V.







#define METRALE_ATTN_FP8_SMEM

// 2026-09-25: Copies one 32-row tile of raw E4M3 bytes with metrale_cp16, 16 bytes (16 values) per copy; rows at or past
// `kv_l` are zeroed with a uint4 store. `fp8_cache_stride` is the per-block stride in bytes.
//
// PAD_KV is 16 so the 1-byte smem row stride HDIM_PAD is 272, a multiple of 16 as the 16-byte copies and uint4
// stores require; gb10's prefill_paged_compute.cuh defaults it to 8 (264) only when it is undefined.









#ifdef METRALE_ATTN_FP8_SMEM
#define PAD_KV 16
#endif
#define LOAD_KV_TILE(cache, bt, smem, kv_s, kv_l, kvh, t, stride) \
    do { \
        const unsigned int _cpr = HDIM / 16; \
        for (unsigned int _i = t; _i < TILE_CHUNKS / 2; _i += (stride)) { \
            unsigned int _row = _i / _cpr, _col = (_i % _cpr) * 16; \
            unsigned int _pos = (kv_s) + _row; \
            if (_pos < (kv_l)) { \
                unsigned int _lb = _pos / cache_block_size; \
                unsigned int _bo = _pos % cache_block_size; \
                unsigned int _pb = (unsigned int)(bt)[_lb]; \
                const void* _base = (const void*)( \
                    (const __nv_fp8_storage_t*)(cache) \
                    + (unsigned long long)_pb * fp8_cache_stride \
                    + (unsigned long long)_bo * num_kv_heads * head_dim \
                    + (unsigned long long)(kvh) * head_dim + _col); \
                metrale_cp16(&((__nv_fp8_storage_t(*)[HDIM_PAD])(smem))[_row][_col], _base); \
            } else { \
                *((uint4*)&((__nv_fp8_storage_t(*)[HDIM_PAD])(smem))[_row][_col]) = make_uint4(0,0,0,0); \
            } \
        } \
    } while(0)

#define KERNEL_NAME attn_prefill_paged_fp8
#define K_CACHE_TYPE const void* __restrict__
#define V_CACHE_TYPE const void* __restrict__
#define KERNEL_EXTRA_PARAMS \
    , const float inv_sqrt_d \
    , const float k_scale \
    , const float v_scale \
    , const unsigned long long fp8_cache_stride



#define KERNEL_PREAMBLE

#include "prefill_paged_compute.cuh"
