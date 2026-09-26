// SPDX-License-Identifier: MIT OR Apache-2.0
// forked-from: kernels/gb10/common/prefill_paged_compute.cuh (2026-09-24; 1331 of 1062 lines differ, see kernels/FORKS.md)

// 2026-09-25: Paged flash-attention prefill compute for HIP on gfx1151: wave32
// WMMA 16x16x16 tiles, with the online softmax staged through shared memory.
//
// Owner: strix-hip kernels.
// Invariants:
// - The BR=32 entry needs exactly 4 warps (128 threads). The _64 entry returns
//   warps 4 and up at once, so it also runs on a 256-thread launch.
// - BR64 is 32. The host grid of the _64 entries steps 32 rows on this tree
//   (`cfg!(metrale_scale)` in ops/prefill_attn_main_b.rs); the two must match.
//
// Included on this tree by gb10/common's attn_prefill_paged.cu, _batched,
// _fp8, _fp8_batched, _nvfp4 and _nvfp4_batched ([sources] use in
// common/KERNEL.toml). The includer defines LOAD_KV_TILE(cache, block_table,
// smem, kv_start, kv_len, kv_head, tid, stride), KERNEL_NAME, K_CACHE_TYPE,
// V_CACHE_TYPE, KERNEL_EXTRA_PARAMS (which must declare `inv_sqrt_d`) and
// KERNEL_PREAMBLE, and may define PREFILL_BATCHED and HDIM.
//
// WMMA fragments (typedefs below), lane l = 0..31:
//   A (M x K, row-major): a[i] = A[m_row + (l&15)][i]
//   B (K x N):            b[k] = B[k][n_col + (l&15)]
//   C/D:                  element e (0..7) is row 2*e + (l>>4), column l&15
// The scores are staged to smem_S with that C mapping and one thread per
// query row runs the online softmax, so the softmax does not depend on the
// fragment layout.











#include <cuda_bf16.h>
#include <cuda_fp16.h>

// 2026-09-25: Synchronous stand-ins for the cp.async helpers of the same names
// in gb10/common/prefill_paged_compute.cuh: a 16-byte copy (zero-fill when
// !pred), and commit and wait do nothing.


__device__ __forceinline__ void metrale_cp16(void* smem_dst, const void* gmem_src) {
    *reinterpret_cast<uint4*>(smem_dst) = *reinterpret_cast<const uint4*>(gmem_src);
}
__device__ __forceinline__ void metrale_cp16_pred(void* smem_dst, const void* gmem_src, bool pred) {
    if (pred) *reinterpret_cast<uint4*>(smem_dst) = *reinterpret_cast<const uint4*>(gmem_src);
    else      *reinterpret_cast<uint4*>(smem_dst) = make_uint4(0,0,0,0);
}
__device__ __forceinline__ void metrale_cp_commit() {}
__device__ __forceinline__ void metrale_cp_wait()   {}

typedef __bf16 v16bf __attribute__((ext_vector_type(16)));
typedef __fp16 v16h  __attribute__((ext_vector_type(16)));
typedef float  v8f   __attribute__((ext_vector_type(8)));

// 2026-09-25: Softmax exp: `__expf`, or, when METRALE_FAST_SOFTMAX_EXP is
// defined, a degree-3 polynomial for 2^frac scaled by ldexpf. The polynomial's
// largest relative error is 0.56%, as frac approaches 1 (computed 2026-09-25).
// No KERNEL.toml on this tree defines the macro.





__device__ __forceinline__ float sw_exp(float x) {
#ifdef METRALE_FAST_SOFTMAX_EXP

    float t = x * 1.4426950408889634f;
    float ti = floorf(t);
    float tf = t - ti;
    float p = 1.0f + tf * (0.6931471805599453f +
              tf * (0.2402265069591007f +
              tf * 0.05550410866482158f));
    return ldexpf(p, (int)ti);
#else

    return __expf(x);
#endif
}

#define BR 32
#define BC 32
#ifndef HDIM
#define HDIM 256
#endif
#define PAD_KV 8
#define HDIM_PAD (HDIM + PAD_KV)
#define PAD_P 8

// 2026-09-25: WMMA 16x16x16 tiling. PV_N_TILES = output tiles per warp (half of head_dim).
#define K16 16
#define WMMA_K_STEPS (HDIM / K16)
#define QK_N_TILES   (BC / K16)
#define PV_K_STEPS   (BC / K16)
#define PV_N_TILES   ((HDIM / K16) / 2)
#define TILE_CHUNKS (BR * (HDIM / 8))


// 2026-09-25: BR=32 entry: 4 warps of 32 lanes (128 threads).

extern "C" __global__ void KERNEL_NAME(
    const __nv_bfloat16* __restrict__ Q,
    K_CACHE_TYPE K_cache,
    V_CACHE_TYPE V_cache,
    __nv_bfloat16* __restrict__ O,
#ifdef PREFILL_BATCHED
    const int* const* __restrict__ block_table_ptrs,
    const unsigned int batch_size,
#else
    const int* __restrict__ block_table,
#endif
    const unsigned int q_len,
    const unsigned int kv_len,
    const unsigned int q_offset,
    const unsigned int num_q_heads,
    const unsigned int num_kv_heads,
    const unsigned int head_dim,
    const unsigned int cache_block_size,
    const unsigned int sliding_window,
    const unsigned int causal_mask_enabled
    KERNEL_EXTRA_PARAMS
) {
    const unsigned int q_head = blockIdx.x;
    const unsigned int q_block = blockIdx.y;
#ifdef PREFILL_BATCHED
    const unsigned int b = blockIdx.z;
    if (b >= batch_size) return;
    const int* const __restrict__ block_table = block_table_ptrs[b];
#endif
    const unsigned int tid = threadIdx.x;
    const unsigned int warp_id = tid / 32;
    const unsigned int lane_id = tid % 32;
    const unsigned int lane_lo = lane_id & 15;
    const unsigned int lane_hi = lane_id >> 4;

    if (q_head >= num_q_heads) return;
    const unsigned int q_start = q_block * BR;
    if (q_start >= q_len) return;
    const unsigned int q_tile_end = min(q_start + BR, q_len);
    const unsigned int q_tile_len = q_tile_end - q_start;
    const unsigned int q_seq_stride = num_q_heads * head_dim;
    const unsigned int kv_head = q_head / (num_q_heads / num_kv_heads);
#ifdef PREFILL_BATCHED
    const unsigned long long q_batch_off = (unsigned long long)b * q_len * q_seq_stride;
#endif

    __shared__ __nv_bfloat16 smem_Q[BR][HDIM_PAD];
    __shared__ __nv_bfloat16 smem_K[BC][HDIM_PAD];
    __shared__ __nv_bfloat16 smem_V[BC][HDIM_PAD];
    // 2026-09-25: P is FP16 (10 mantissa bits, against BF16's 7) for the PV
    // WMMA; V stays BF16 in shared memory and is converted per MMA. Defining
    // METRALE_DISABLE_FP16_PV selects a BF16 P and a BF16 PV WMMA.



#ifdef METRALE_DISABLE_FP16_PV
    __shared__ __nv_bfloat16 smem_P[BR][BC];
#else
    __shared__ __half smem_P[BR][BC];
#endif
    __shared__ float smem_S[BR][BC];
    __shared__ float smem_ml[BR][2];
    __shared__ float smem_resc[BR];

    KERNEL_PREAMBLE

    // 2026-09-25: PV warp roles: warp_id & 1 picks the 16-row query tile, and
    // warp_id >> 1 picks the half of head_dim (columns 0-127 or 128-255 at HDIM=256).
    const unsigned int pv_warp_m  = (warp_id & 1) * 16;
    const unsigned int pv_n_start = (warp_id >> 1) * PV_N_TILES;

    v8f acc_o[PV_N_TILES];
    #pragma unroll
    for (int i = 0; i < PV_N_TILES; i++) acc_o[i] = v8f{0,0,0,0,0,0,0,0};

    unsigned int num_kv_blocks = (kv_len + BC - 1) / BC;
    { unsigned int mx = (q_offset + q_tile_end - 1) / BC;
      num_kv_blocks = min(num_kv_blocks, mx + 1); }

    for (unsigned int r = tid; r < BR; r += blockDim.x) {
        smem_ml[r][0] = -1e30f;
        smem_ml[r][1] = 0.0f;
    }


    {
        const unsigned int cpr = HDIM / 8;
        for (unsigned int idx = tid; idx < TILE_CHUNKS; idx += blockDim.x) {
            unsigned int row = idx / cpr, col = (idx % cpr) * 8;
            if (q_start + row < q_len) {
#ifdef PREFILL_BATCHED
                const void* gm = (const void*)&Q[q_batch_off + (q_start+row)*q_seq_stride + q_head*head_dim + col];
#else
                const void* gm = (const void*)&Q[(q_start+row)*q_seq_stride + q_head*head_dim + col];
#endif
                *((uint4*)&smem_Q[row][col]) = *((const uint4*)gm);
            } else {
                *((uint4*)&smem_Q[row][col]) = make_uint4(0,0,0,0);
            }
        }
    }
    __syncthreads();

    for (unsigned int kv_block = 0; kv_block < num_kv_blocks; kv_block++) {
        unsigned int kv_start = kv_block * BC;
        unsigned int kv_end = min(kv_start + BC, kv_len);
        unsigned int kv_tile_len = kv_end - kv_start;

        // 2026-09-25: K and V tiles through the includer's LOAD_KV_TILE.
        LOAD_KV_TILE(K_cache, block_table, smem_K, kv_start, kv_len, kv_head, tid, blockDim.x);
        LOAD_KV_TILE(V_cache, block_table, smem_V, kv_start, kv_len, kv_head, tid, blockDim.x);
        __syncthreads();

        // 2026-09-25: QK^T, S = Q @ K^T, on warps 0 and 1, 16 query rows each.
        if (warp_id < 2) {
            const unsigned int qk_m = warp_id * 16;

            v8f acc_s[QK_N_TILES];
            #pragma unroll
            for (int n = 0; n < QK_N_TILES; n++) acc_s[n] = v8f{0,0,0,0,0,0,0,0};

            #pragma unroll
            for (unsigned int ks = 0; ks < WMMA_K_STEPS; ks++) {
                unsigned int k_off = ks * K16;
                v16bf a;
                #pragma unroll
                for (int i = 0; i < 16; i++)
                    a[i] = (__bf16)(float)smem_Q[qk_m + lane_lo][k_off + i];

                #pragma unroll
                for (int nt = 0; nt < QK_N_TILES; nt++) {
                    unsigned int key_row = nt * 16 + lane_lo;
                    v16bf bb;
                    #pragma unroll
                    for (int k = 0; k < 16; k++)
                        bb[k] = (__bf16)(float)smem_K[key_row][k_off + k];
                    acc_s[nt] = __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32(a, bb, acc_s[nt]);
                }
            }

            // 2026-09-25: Stage the scores to smem_S with the C mapping above.
            #pragma unroll
            for (int nt = 0; nt < QK_N_TILES; nt++) {
                unsigned int col = nt * 16 + lane_lo;
                #pragma unroll
                for (int e = 0; e < 8; e++) {
                    unsigned int row = qk_m + 2 * e + lane_hi;
                    smem_S[row][col] = acc_s[nt][e];
                }
            }
        }
        __syncthreads();

        // 2026-09-25: Online softmax in shared memory, one thread per query row.
        if (tid < BR) {
            unsigned int r = tid;
            unsigned int qr = q_offset + q_start + r;
            bool row_valid = (r < q_tile_len);

            float rmax = -1e30f;
            #pragma unroll
            for (unsigned int c = 0; c < BC; c++) {
                float s = smem_S[r][c] * inv_sqrt_d;
                unsigned int kpos = kv_start + c;
                bool masked = (c >= kv_tile_len) || !row_valid;
                if (causal_mask_enabled && kpos > qr) masked = true;
                if (sliding_window > 0 && kpos <= qr &&
                    (qr - kpos) >= sliding_window) masked = true;
                if (masked) s = -1e30f;
                smem_S[r][c] = s;
                rmax = fmaxf(rmax, s);
            }

            float m_old = smem_ml[r][0];
            float l_old = smem_ml[r][1];
            float m_new = fmaxf(m_old, rmax);
            float resc = sw_exp(m_old - m_new);

            float sum = 0.0f;
            #pragma unroll
            for (unsigned int c = 0; c < BC; c++) {
                float p = sw_exp(smem_S[r][c] - m_new);
#ifdef METRALE_DISABLE_FP16_PV
                smem_P[r][c] = __float2bfloat16(p);
#else
                smem_P[r][c] = __float2half(p);
#endif
                sum += p;
            }

            smem_ml[r][0] = m_new;
            smem_ml[r][1] = l_old * resc + sum;
            smem_resc[r] = resc;
        }
        __syncthreads();

        // 2026-09-25: Rescale acc_o by each row's exp(m_old - m_new).
        {
            float resc_e[8];
            #pragma unroll
            for (int e = 0; e < 8; e++)
                resc_e[e] = smem_resc[pv_warp_m + 2 * e + lane_hi];
            #pragma unroll
            for (int nt = 0; nt < PV_N_TILES; nt++)
                #pragma unroll
                for (int e = 0; e < 8; e++)
                    acc_o[nt][e] *= resc_e[e];
        }

        // 2026-09-25: PV, O += P @ V, on all 4 warps: an FP16 WMMA with V converted
        // from BF16 per MMA, or a BF16 WMMA under METRALE_DISABLE_FP16_PV. QK^T
        // above stays BF16.



        {
            #pragma unroll
            for (unsigned int ks = 0; ks < PV_K_STEPS; ks++) {
                unsigned int k_off = ks * K16;
#ifdef METRALE_DISABLE_FP16_PV
                v16bf a;
                #pragma unroll
                for (int i = 0; i < 16; i++)
                    a[i] = (__bf16)(float)smem_P[pv_warp_m + lane_lo][k_off + i];

                #pragma unroll
                for (int nt = 0; nt < PV_N_TILES; nt++) {
                    unsigned int d_col = (pv_n_start + nt) * 16 + lane_lo;
                    v16bf bb;
                    #pragma unroll
                    for (int k = 0; k < 16; k++)
                        bb[k] = (__bf16)(float)smem_V[k_off + k][d_col];
                    acc_o[nt] = __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32(a, bb, acc_o[nt]);
                }
#else
                v16h a;
                #pragma unroll
                for (int i = 0; i < 16; i++)
                    a[i] = (__fp16)__half2float(smem_P[pv_warp_m + lane_lo][k_off + i]);

                #pragma unroll
                for (int nt = 0; nt < PV_N_TILES; nt++) {
                    unsigned int d_col = (pv_n_start + nt) * 16 + lane_lo;
                    v16h bb;
                    #pragma unroll
                    for (int k = 0; k < 16; k++)
                        bb[k] = (__fp16)(float)smem_V[k_off + k][d_col];
                    acc_o[nt] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a, bb, acc_o[nt]);
                }
#endif
            }
        }
        __syncthreads();
    }


    {
#ifdef PREFILL_BATCHED
        __nv_bfloat16* ob = O + q_batch_off + q_head * head_dim;
#else
        __nv_bfloat16* ob = O + q_head * head_dim;
#endif
        #pragma unroll
        for (int nt = 0; nt < PV_N_TILES; nt++) {
            unsigned int col = (pv_n_start + nt) * 16 + lane_lo;
            #pragma unroll
            for (int e = 0; e < 8; e++) {
                unsigned int row = pv_warp_m + 2 * e + lane_hi;
                unsigned int gr = q_start + row;
                if (gr < q_len && row < q_tile_len && col < head_dim) {
                    float l = smem_ml[row][1];
                    float inv_l = (l > 0.0f) ? (1.0f / l) : 0.0f;
                    ob[gr * q_seq_stride + col] = __float2bfloat16(acc_o[nt][e] * inv_l);
                }
            }
        }
    }
}

// 2026-09-25: BR=64 entry (KERNEL_NAME##_64), with BR64 clamped to 32.
// paged_attn.rs picks the _64 kernels by chunk length alone (n >= 256), so
// they run on this tree. At BR64=64 this kernel's shared arrays would take
// 80,640 bytes; at 32 they take 57,216, under the 64 KB gfx1151 workgroup
// limit. The body matches the BR=32 entry, except that warps 4 and up return
// at once (the host launches 256 threads) and the loops stride by 128.






#define BR64 32
#define TILE_CHUNKS_Q64 (BR64 * (HDIM / 8))

#define _PAGED_CONCAT(a, b) a##b
#define PAGED_CONCAT(a, b) _PAGED_CONCAT(a, b)

extern "C" __global__ void PAGED_CONCAT(KERNEL_NAME, _64)(
    const __nv_bfloat16* __restrict__ Q,
    K_CACHE_TYPE K_cache,
    V_CACHE_TYPE V_cache,
    __nv_bfloat16* __restrict__ O,
#ifdef PREFILL_BATCHED
    const int* const* __restrict__ block_table_ptrs,
    const unsigned int batch_size,
#else
    const int* __restrict__ block_table,
#endif
    const unsigned int q_len,
    const unsigned int kv_len,
    const unsigned int q_offset,
    const unsigned int num_q_heads,
    const unsigned int num_kv_heads,
    const unsigned int head_dim,
    const unsigned int cache_block_size,
    const unsigned int sliding_window,
    const unsigned int causal_mask_enabled
    KERNEL_EXTRA_PARAMS
) {
    const unsigned int q_head = blockIdx.x;
    const unsigned int q_block = blockIdx.y;
#ifdef PREFILL_BATCHED
    const unsigned int b = blockIdx.z;
    if (b >= batch_size) return;
    const int* const __restrict__ block_table = block_table_ptrs[b];
#endif
    const unsigned int tid = threadIdx.x;
    const unsigned int warp_id = tid / 32;
    const unsigned int lane_id = tid % 32;
    const unsigned int lane_lo = lane_id & 15;
    const unsigned int lane_hi = lane_id >> 4;

    if (warp_id >= 4) return;

    if (q_head >= num_q_heads) return;
    const unsigned int q_start = q_block * BR64;
    if (q_start >= q_len) return;
    const unsigned int q_tile_end = min(q_start + BR64, q_len);
    const unsigned int q_tile_len = q_tile_end - q_start;
    const unsigned int q_seq_stride = num_q_heads * head_dim;
    const unsigned int kv_head = q_head / (num_q_heads / num_kv_heads);
#ifdef PREFILL_BATCHED
    const unsigned long long q_batch_off = (unsigned long long)b * q_len * q_seq_stride;
#endif

    __shared__ __nv_bfloat16 smem_Q[BR64][HDIM_PAD];
    __shared__ __nv_bfloat16 smem_K[BC][HDIM_PAD];
    __shared__ __nv_bfloat16 smem_V[BC][HDIM_PAD];

#ifdef METRALE_DISABLE_FP16_PV
    __shared__ __nv_bfloat16 smem_P[BR64][BC];
#else
    __shared__ __half smem_P[BR64][BC];
#endif
    __shared__ float smem_S[BR64][BC];
    __shared__ float smem_ml[BR64][2];
    __shared__ float smem_resc[BR64];

    KERNEL_PREAMBLE

    const unsigned int pv_warp_m  = (warp_id & 1) * 16;
    const unsigned int pv_n_start = (warp_id >> 1) * PV_N_TILES;

    v8f acc_o[PV_N_TILES];
    #pragma unroll
    for (int i = 0; i < PV_N_TILES; i++) acc_o[i] = v8f{0,0,0,0,0,0,0,0};

    unsigned int num_kv_blocks = (kv_len + BC - 1) / BC;
    { unsigned int mx = (q_offset + q_tile_end - 1) / BC;
      num_kv_blocks = min(num_kv_blocks, mx + 1); }

    for (unsigned int r = tid; r < BR64; r += 128) {
        smem_ml[r][0] = -1e30f;
        smem_ml[r][1] = 0.0f;
    }

    {
        const unsigned int cpr = HDIM / 8;
        for (unsigned int idx = tid; idx < TILE_CHUNKS_Q64; idx += 128) {
            unsigned int row = idx / cpr, col = (idx % cpr) * 8;
            if (q_start + row < q_len) {
#ifdef PREFILL_BATCHED
                const void* gm = (const void*)&Q[q_batch_off + (q_start+row)*q_seq_stride + q_head*head_dim + col];
#else
                const void* gm = (const void*)&Q[(q_start+row)*q_seq_stride + q_head*head_dim + col];
#endif
                *((uint4*)&smem_Q[row][col]) = *((const uint4*)gm);
            } else {
                *((uint4*)&smem_Q[row][col]) = make_uint4(0,0,0,0);
            }
        }
    }
    __syncthreads();

    for (unsigned int kv_block = 0; kv_block < num_kv_blocks; kv_block++) {
        unsigned int kv_start = kv_block * BC;
        unsigned int kv_end = min(kv_start + BC, kv_len);
        unsigned int kv_tile_len = kv_end - kv_start;

        LOAD_KV_TILE(K_cache, block_table, smem_K, kv_start, kv_len, kv_head, tid, 128);
        LOAD_KV_TILE(V_cache, block_table, smem_V, kv_start, kv_len, kv_head, tid, 128);
        __syncthreads();

        if (warp_id < 2) {
            const unsigned int qk_m = warp_id * 16;
            v8f acc_s[QK_N_TILES];
            #pragma unroll
            for (int n = 0; n < QK_N_TILES; n++) acc_s[n] = v8f{0,0,0,0,0,0,0,0};

            #pragma unroll
            for (unsigned int ks = 0; ks < WMMA_K_STEPS; ks++) {
                unsigned int k_off = ks * K16;
                v16bf a;
                #pragma unroll
                for (int i = 0; i < 16; i++)
                    a[i] = (__bf16)(float)smem_Q[qk_m + lane_lo][k_off + i];
                #pragma unroll
                for (int nt = 0; nt < QK_N_TILES; nt++) {
                    unsigned int key_row = nt * 16 + lane_lo;
                    v16bf bb;
                    #pragma unroll
                    for (int k = 0; k < 16; k++)
                        bb[k] = (__bf16)(float)smem_K[key_row][k_off + k];
                    acc_s[nt] = __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32(a, bb, acc_s[nt]);
                }
            }

            #pragma unroll
            for (int nt = 0; nt < QK_N_TILES; nt++) {
                unsigned int col = nt * 16 + lane_lo;
                #pragma unroll
                for (int e = 0; e < 8; e++) {
                    unsigned int row = qk_m + 2 * e + lane_hi;
                    smem_S[row][col] = acc_s[nt][e];
                }
            }
        }
        __syncthreads();

        if (tid < BR64) {
            unsigned int r = tid;
            unsigned int qr = q_offset + q_start + r;
            bool row_valid = (r < q_tile_len);
            float rmax = -1e30f;
            #pragma unroll
            for (unsigned int c = 0; c < BC; c++) {
                float s = smem_S[r][c] * inv_sqrt_d;
                unsigned int kpos = kv_start + c;
                bool masked = (c >= kv_tile_len) || !row_valid;
                if (causal_mask_enabled && kpos > qr) masked = true;
                if (sliding_window > 0 && kpos <= qr &&
                    (qr - kpos) >= sliding_window) masked = true;
                if (masked) s = -1e30f;
                smem_S[r][c] = s;
                rmax = fmaxf(rmax, s);
            }
            float m_old = smem_ml[r][0];
            float l_old = smem_ml[r][1];
            float m_new = fmaxf(m_old, rmax);
            float resc = sw_exp(m_old - m_new);
            float sum = 0.0f;
            #pragma unroll
            for (unsigned int c = 0; c < BC; c++) {
                float p = sw_exp(smem_S[r][c] - m_new);
#ifdef METRALE_DISABLE_FP16_PV
                smem_P[r][c] = __float2bfloat16(p);
#else
                smem_P[r][c] = __float2half(p);
#endif
                sum += p;
            }
            smem_ml[r][0] = m_new;
            smem_ml[r][1] = l_old * resc + sum;
            smem_resc[r] = resc;
        }
        __syncthreads();

        {
            float resc_e[8];
            #pragma unroll
            for (int e = 0; e < 8; e++)
                resc_e[e] = smem_resc[pv_warp_m + 2 * e + lane_hi];
            #pragma unroll
            for (int nt = 0; nt < PV_N_TILES; nt++)
                #pragma unroll
                for (int e = 0; e < 8; e++)
                    acc_o[nt][e] *= resc_e[e];
        }


        {
            #pragma unroll
            for (unsigned int ks = 0; ks < PV_K_STEPS; ks++) {
                unsigned int k_off = ks * K16;
#ifdef METRALE_DISABLE_FP16_PV
                v16bf a;
                #pragma unroll
                for (int i = 0; i < 16; i++)
                    a[i] = (__bf16)(float)smem_P[pv_warp_m + lane_lo][k_off + i];
                #pragma unroll
                for (int nt = 0; nt < PV_N_TILES; nt++) {
                    unsigned int d_col = (pv_n_start + nt) * 16 + lane_lo;
                    v16bf bb;
                    #pragma unroll
                    for (int k = 0; k < 16; k++)
                        bb[k] = (__bf16)(float)smem_V[k_off + k][d_col];
                    acc_o[nt] = __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32(a, bb, acc_o[nt]);
                }
#else
                v16h a;
                #pragma unroll
                for (int i = 0; i < 16; i++)
                    a[i] = (__fp16)__half2float(smem_P[pv_warp_m + lane_lo][k_off + i]);
                #pragma unroll
                for (int nt = 0; nt < PV_N_TILES; nt++) {
                    unsigned int d_col = (pv_n_start + nt) * 16 + lane_lo;
                    v16h bb;
                    #pragma unroll
                    for (int k = 0; k < 16; k++)
                        bb[k] = (__fp16)(float)smem_V[k_off + k][d_col];
                    acc_o[nt] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a, bb, acc_o[nt]);
                }
#endif
            }
        }
        __syncthreads();
    }

    {
#ifdef PREFILL_BATCHED
        __nv_bfloat16* ob = O + q_batch_off + q_head * head_dim;
#else
        __nv_bfloat16* ob = O + q_head * head_dim;
#endif
        #pragma unroll
        for (int nt = 0; nt < PV_N_TILES; nt++) {
            unsigned int col = (pv_n_start + nt) * 16 + lane_lo;
            #pragma unroll
            for (int e = 0; e < 8; e++) {
                unsigned int row = pv_warp_m + 2 * e + lane_hi;
                unsigned int gr = q_start + row;
                if (gr < q_len && row < q_tile_len && col < head_dim) {
                    float l = smem_ml[row][1];
                    float inv_l = (l > 0.0f) ? (1.0f / l) : 0.0f;
                    ob[gr * q_seq_stride + col] = __float2bfloat16(acc_o[nt][e] * inv_l);
                }
            }
        }
    }
}
