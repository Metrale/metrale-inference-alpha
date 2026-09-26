// SPDX-License-Identifier: MIT OR Apache-2.0
// forked-from: kernels/gb10/common/prefill_paged_compute_512.cuh (2026-09-24; 333 of 363 lines differ, see kernels/FORKS.md)

// 2026-09-25: Paged flash-attention prefill for head_dim 512 (the Gemma-4
// full-attention layers) for HIP on gfx1151: wave32 WMMA 16x16x16 tiles, with
// the online softmax staged through shared memory.
//
// Owner: strix-hip kernels.
// Invariants:
// - BR=32, BC=32, HDIM=512, 8 warps (256 threads). QK^T runs on warps 0 and 1.
//   PV runs on all 8: warp_id & 1 picks the 16-row query tile and
//   warp_id >> 1 picks 8 of the 32 16-column head_dim tiles.
// - The dynamic shared-memory layout below needs 105,344 bytes.
// - Softmax exp is always the degree-3 polynomial in sw_exp_512 (largest
//   relative error 0.56%, computed 2026-09-25).
//
// Included on this tree by gb10/common/attn_prefill_paged_512.cu, which
// defines LOAD_KV_TILE_512(cache, bt, smem_ptr, kv_s, kv_l, kvh, t, stride),
// KERNEL_NAME, K_CACHE_TYPE, V_CACHE_TYPE, KERNEL_EXTRA_PARAMS (declaring
// `inv_sqrt_d`) and KERNEL_PREAMBLE. Fragment mapping as in
// prefill_paged_compute.cuh.



#include <cuda_bf16.h>

// 2026-09-25: Synchronous stand-ins for the cp.async helpers of the same names
// in gb10/common/prefill_paged_compute_512.cuh: a 16-byte copy (zero-fill when
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

typedef __bf16 v16bf_512 __attribute__((ext_vector_type(16)));
typedef float  v8f_512   __attribute__((ext_vector_type(8)));

__device__ __forceinline__ float sw_exp_512(float x) {
    float t = x * 1.4426950408889634f;
    float ti = floorf(t);
    float tf = t - ti;
    float p = 1.0f + tf * (0.6931471805599453f +
              tf * (0.2402265069591007f +
              tf * 0.05550410866482158f));
    return ldexpf(p, (int)ti);
}

#define BR_512   32
#define BC_512   32
#define HDIM_512 512
#define PAD_P_512 8
#define K16_512 16
#define WMMA_K_STEPS_512 (HDIM_512 / K16_512)
#define QK_N_TILES_512   (BC_512 / K16_512)
#define PV_K_STEPS_512   (BC_512 / K16_512)
#define N_TILES_PER_WARP_512 8
#define TILE_CHUNKS_512 (BR_512 * (HDIM_512 / 8))

extern "C" __global__ void KERNEL_NAME(
    const __nv_bfloat16* __restrict__ Q,
    K_CACHE_TYPE K_cache,
    V_CACHE_TYPE V_cache,
    __nv_bfloat16* __restrict__ O,
    const int* __restrict__ block_table,
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
    const unsigned int q_head  = blockIdx.x;
    const unsigned int q_block = blockIdx.y;
    const unsigned int tid     = threadIdx.x;
    const unsigned int warp_id = tid / 32;
    const unsigned int lane_id = tid % 32;
    const unsigned int lane_lo = lane_id & 15;
    const unsigned int lane_hi = lane_id >> 4;

    if (q_head >= num_q_heads) return;
    const unsigned int q_start = q_block * BR_512;
    if (q_start >= q_len) return;
    const unsigned int q_tile_end = min(q_start + BR_512, q_len);
    const unsigned int q_tile_len = q_tile_end - q_start;
    const unsigned int q_seq_stride = num_q_heads * head_dim;
    const unsigned int kv_head = q_head / (num_q_heads / num_kv_heads);

    extern __shared__ __align__(16) unsigned char smem_dyn_512[];
    __nv_bfloat16* smem_Q = reinterpret_cast<__nv_bfloat16*>(smem_dyn_512);
    __nv_bfloat16* smem_K = smem_Q + (unsigned int)BR_512 * HDIM_512;
    __nv_bfloat16* smem_V = smem_K + (unsigned int)BC_512 * HDIM_512;
    __nv_bfloat16* smem_P = smem_V + (unsigned int)BC_512 * HDIM_512;
    float* smem_S = reinterpret_cast<float*>(
                       smem_P + (unsigned int)BR_512 * (BC_512 + PAD_P_512));
    float* smem_ml = smem_S + (unsigned int)BR_512 * BC_512;
    float* smem_resc = smem_ml + (unsigned int)BR_512 * 2;

    KERNEL_PREAMBLE

    const unsigned int pv_warp_m  = (warp_id & 1) * 16;
    const unsigned int pv_n_start = (warp_id >> 1) * N_TILES_PER_WARP_512;

    v8f_512 acc_o[N_TILES_PER_WARP_512];
    #pragma unroll
    for (int i = 0; i < N_TILES_PER_WARP_512; i++)
        acc_o[i] = v8f_512{0,0,0,0,0,0,0,0};

    unsigned int num_kv_blocks = (kv_len + BC_512 - 1) / BC_512;
    { unsigned int mx = (q_offset + q_tile_end - 1) / BC_512;
      num_kv_blocks = min(num_kv_blocks, mx + 1); }

    for (unsigned int r = tid; r < BR_512; r += blockDim.x) {
        smem_ml[r * 2 + 0] = -1e30f;
        smem_ml[r * 2 + 1] = 0.0f;
    }


    {
        const unsigned int cpr = HDIM_512 / 8;
        for (unsigned int idx = tid; idx < TILE_CHUNKS_512; idx += blockDim.x) {
            unsigned int row = idx / cpr, col = (idx % cpr) * 8;
            if (q_start + row < q_len) {
                const void* gm = (const void*)&Q[(q_start+row)*q_seq_stride + q_head*head_dim + col];
                *((uint4*)&smem_Q[row * HDIM_512 + col]) = *((const uint4*)gm);
            } else {
                *((uint4*)&smem_Q[row * HDIM_512 + col]) = make_uint4(0,0,0,0);
            }
        }
    }
    __syncthreads();

    for (unsigned int kv_block = 0; kv_block < num_kv_blocks; kv_block++) {
        unsigned int kv_start = kv_block * BC_512;
        unsigned int kv_end   = min(kv_start + BC_512, kv_len);
        unsigned int kv_tile_len = kv_end - kv_start;

        LOAD_KV_TILE_512(K_cache, block_table, smem_K, kv_start, kv_len, kv_head, tid, blockDim.x);
        LOAD_KV_TILE_512(V_cache, block_table, smem_V, kv_start, kv_len, kv_head, tid, blockDim.x);
        __syncthreads();

        // 2026-09-25: QK^T on warps 0 and 1, 16 query rows each.
        if (warp_id < 2) {
            const unsigned int qk_m = warp_id * 16;
            v8f_512 acc_s[QK_N_TILES_512];
            #pragma unroll
            for (int n = 0; n < QK_N_TILES_512; n++) acc_s[n] = v8f_512{0,0,0,0,0,0,0,0};

            #pragma unroll
            for (unsigned int ks = 0; ks < WMMA_K_STEPS_512; ks++) {
                unsigned int k_off = ks * K16_512;
                v16bf_512 a;
                #pragma unroll
                for (int i = 0; i < 16; i++)
                    a[i] = (__bf16)(float)smem_Q[(qk_m + lane_lo) * HDIM_512 + k_off + i];
                #pragma unroll
                for (int nt = 0; nt < QK_N_TILES_512; nt++) {
                    unsigned int key_row = nt * 16 + lane_lo;
                    v16bf_512 bb;
                    #pragma unroll
                    for (int k = 0; k < 16; k++)
                        bb[k] = (__bf16)(float)smem_K[key_row * HDIM_512 + k_off + k];
                    acc_s[nt] = __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32(a, bb, acc_s[nt]);
                }
            }

            #pragma unroll
            for (int nt = 0; nt < QK_N_TILES_512; nt++) {
                unsigned int col = nt * 16 + lane_lo;
                #pragma unroll
                for (int e = 0; e < 8; e++) {
                    unsigned int row = qk_m + 2 * e + lane_hi;
                    smem_S[row * BC_512 + col] = acc_s[nt][e];
                }
            }
        }
        __syncthreads();

        // 2026-09-25: Online softmax in shared memory, one thread per query row.
        if (tid < BR_512) {
            unsigned int r = tid;
            unsigned int qr = q_offset + q_start + r;
            bool row_valid = (r < q_tile_len);
            float rmax = -1e30f;
            #pragma unroll
            for (unsigned int c = 0; c < BC_512; c++) {
                float s = smem_S[r * BC_512 + c] * inv_sqrt_d;
                unsigned int kpos = kv_start + c;
                bool masked = (c >= kv_tile_len) || !row_valid;
                if (causal_mask_enabled && kpos > qr) masked = true;
                if (sliding_window > 0 && kpos <= qr &&
                    (qr - kpos) >= sliding_window) masked = true;
                if (masked) s = -1e30f;
                smem_S[r * BC_512 + c] = s;
                rmax = fmaxf(rmax, s);
            }
            float m_old = smem_ml[r * 2 + 0];
            float l_old = smem_ml[r * 2 + 1];
            float m_new = fmaxf(m_old, rmax);
            float resc = sw_exp_512(m_old - m_new);
            float sum = 0.0f;
            #pragma unroll
            for (unsigned int c = 0; c < BC_512; c++) {
                float p = sw_exp_512(smem_S[r * BC_512 + c] - m_new);
                smem_P[r * (BC_512 + PAD_P_512) + c] = __float2bfloat16(p);
                sum += p;
            }
            smem_ml[r * 2 + 0] = m_new;
            smem_ml[r * 2 + 1] = l_old * resc + sum;
            smem_resc[r] = resc;
        }
        __syncthreads();


        {
            float resc_e[8];
            #pragma unroll
            for (int e = 0; e < 8; e++)
                resc_e[e] = smem_resc[pv_warp_m + 2 * e + lane_hi];
            #pragma unroll
            for (int nt = 0; nt < N_TILES_PER_WARP_512; nt++)
                #pragma unroll
                for (int e = 0; e < 8; e++)
                    acc_o[nt][e] *= resc_e[e];
        }

        // 2026-09-25: PV, O += P @ V, on all 8 warps with a BF16 P.
        {
            const unsigned int p_stride = BC_512 + PAD_P_512;
            #pragma unroll
            for (unsigned int ks = 0; ks < PV_K_STEPS_512; ks++) {
                unsigned int k_off = ks * K16_512;
                v16bf_512 a;
                #pragma unroll
                for (int i = 0; i < 16; i++)
                    a[i] = (__bf16)(float)smem_P[(pv_warp_m + lane_lo) * p_stride + k_off + i];
                #pragma unroll
                for (int nt = 0; nt < N_TILES_PER_WARP_512; nt++) {
                    unsigned int d_col = (pv_n_start + nt) * 16 + lane_lo;
                    v16bf_512 bb;
                    #pragma unroll
                    for (int k = 0; k < 16; k++)
                        bb[k] = (__bf16)(float)smem_V[(k_off + k) * HDIM_512 + d_col];
                    acc_o[nt] = __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32(a, bb, acc_o[nt]);
                }
            }
        }
        __syncthreads();
    }


    {
        __nv_bfloat16* ob = O + q_head * head_dim;
        #pragma unroll
        for (int nt = 0; nt < N_TILES_PER_WARP_512; nt++) {
            unsigned int col = (pv_n_start + nt) * 16 + lane_lo;
            #pragma unroll
            for (int e = 0; e < 8; e++) {
                unsigned int row = pv_warp_m + 2 * e + lane_hi;
                unsigned int gr = q_start + row;
                if (gr < q_len && row < q_tile_len && col < head_dim) {
                    float l = smem_ml[row * 2 + 1];
                    float inv_l = (l > 0.0f) ? (1.0f / l) : 0.0f;
                    ob[gr * q_seq_stride + col] = __float2bfloat16(acc_o[nt][e] * inv_l);
                }
            }
        }
    }
}
