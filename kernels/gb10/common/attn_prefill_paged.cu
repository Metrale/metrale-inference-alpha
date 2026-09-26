// SPDX-License-Identifier: MIT OR Apache-2.0


// 2026-09-25: Kernels `attn_prefill_paged` and `attn_prefill_paged_64`: paged prefill flash attention over a BF16 KV cache.
//
// Q is contiguous BF16; each K/V row is found through `block_table`. The body is prefill_paged_compute.cuh: the
// base kernel takes 32 query rows per block on 128 threads, `_64` takes 64 rows on 256 threads (NVIDIA builds).
// Owner: gb10 kernels.
// Invariants: none beyond the types.

#include <cuda_bf16.h>
// 2026-09-25: Copies one 32-row K or V tile into smem with metrale_cp16, a 16-byte cp.async (a plain uint4 copy in
// the strix-hip prefill_paged_compute.cuh this file also builds against); rows at or past `kv_l` are zero-filled.
#define LOAD_KV_TILE(cache, bt, smem, kv_s, kv_l, kvh, t, stride) \
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

#define KERNEL_NAME attn_prefill_paged
#define K_CACHE_TYPE const __nv_bfloat16* __restrict__
#define V_CACHE_TYPE const __nv_bfloat16* __restrict__
#define KERNEL_EXTRA_PARAMS , const float inv_sqrt_d
#define KERNEL_PREAMBLE

#include "prefill_paged_compute.cuh"
