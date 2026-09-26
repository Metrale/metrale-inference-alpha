// SPDX-License-Identifier: MIT OR Apache-2.0

// 2026-09-25: Kernel `attn_prefill_paged_512`: paged prefill flash attention for head_dim 512 over a BF16 KV cache.
//
// The body is prefill_paged_compute_512.cuh: 32 query rows per block on 256 threads, with K single-buffered.
// The host runs it for the later chunks of a chunked prefill when head_dim > 256 (qwen3_attention/prefill/paged_attn.rs).
//
// Owner: gb10 kernels.
// Invariants: the launch passes 101,120 bytes of dynamic shared memory, from which the body carves Q, K, V, P and the
// per-row max/sum.

#include <cuda_bf16.h>
// 2026-09-25: Copies one 32-row K or V tile with metrale_cp16, a 16-byte cp.async (a plain uint4 copy in the strix-hip
// prefill_paged_compute_512.cuh) into flat smem (row stride 512, no padding); rows at or past `kv_l` are zero-filled.
#define LOAD_KV_TILE_512(cache, bt, smem_ptr, kv_s, kv_l, kvh, t, stride) \
    do { \
        const unsigned int _cpr = HDIM_512 / 8; \
        const unsigned long long _ps = (unsigned long long)cache_block_size * num_kv_heads * head_dim; \
        const unsigned long long _rs = (unsigned long long)num_kv_heads * head_dim; \
        for (unsigned int _i = (t); _i < TILE_CHUNKS_512; _i += (stride)) { \
            unsigned int _row = _i / _cpr, _col = (_i % _cpr) * 8; \
            unsigned int _pos = (kv_s) + _row; \
            if (_pos < (kv_l)) { \
                unsigned int _lb = _pos / cache_block_size; \
                unsigned int _bo = _pos % cache_block_size; \
                unsigned int _pb = (unsigned int)(bt)[_lb]; \
                const void* _gm = (const void*)( \
                    (cache) + _pb * _ps + _bo * _rs + (kvh) * head_dim + _col); \
                metrale_cp16(&(smem_ptr)[_row * HDIM_512 + _col], _gm); \
            } else { \
                *((uint4*)&(smem_ptr)[_row * HDIM_512 + _col]) = make_uint4(0,0,0,0); \
            } \
        } \
    } while(0)

#define KERNEL_NAME attn_prefill_paged_512
#define K_CACHE_TYPE const __nv_bfloat16* __restrict__
#define V_CACHE_TYPE const __nv_bfloat16* __restrict__
#define KERNEL_EXTRA_PARAMS , const float inv_sqrt_d
#define KERNEL_PREAMBLE

#include "prefill_paged_compute_512.cuh"
