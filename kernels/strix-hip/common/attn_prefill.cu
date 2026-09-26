// SPDX-License-Identifier: MIT OR Apache-2.0
// forked-from: kernels/gb10/common/attn_prefill.cu (2026-09-24; 982 of 990 lines differ, see kernels/FORKS.md)

// 2026-09-25: Flash-attention prefill for HDIM = 256 on AMD WMMA (gfx1151):
// full, causal, or causal with a sliding window (sliding_window 0 disables
// it). K/V tiles are copied synchronously into shared memory. The QK^T
// scores are stored to shared memory with the WMMA accumulator map and the
// online softmax runs there, one thread per query row with __expf, before P
// goes back into the PV WMMA, so the softmax does not depend on the fragment
// layout.
//
// attn_prefill: 32 query rows per block of 128 threads (4 waves).
// attn_prefill_64: BR64 query rows per block of 256 threads. BR64 is 32 on
// AMD builds (__SCALE__ or __HIP_PLATFORM_AMD__) and 64 otherwise; on this
// target the host grid uses the same 32-row stride
// (crates/model-layers/src/layers/ops/prefill_attn_main_a.rs).
// Grid: (num_q_heads, ceil(seq_len / rows per block), batch).
//
// Q and O: [batch, seq_len, num_q_heads, head_dim]; K and V: [batch, seq_len,
// num_kv_heads, head_dim]; kv_head = q_head / (num_q_heads / num_kv_heads).
//
// WMMA fragments (wave32), lane l: a[i] = A[l & 15][i], b[k] = B[k][l & 15];
// accumulator element e goes to row 2e + (l >> 4), column l & 15.
//
// Owner: strix-hip kernels.
// Invariants: assumes head_dim == HDIM.







#include <cuda_bf16.h>

typedef __bf16 v16bf __attribute__((ext_vector_type(16)));
typedef float  v8f   __attribute__((ext_vector_type(8)));

#define BR 32
#define BC 32
#ifndef HDIM
#define HDIM 256
#endif
#define PAD_KV 8
#define HDIM_PAD (HDIM + PAD_KV)
#define PAD_S 0


#define K16 16
#define WMMA_K_STEPS (HDIM / K16)
#define QK_N_TILES   (BC / K16)
#define PV_K_STEPS   (BC / K16)

#define PV_N_TILES   ((HDIM / K16) / 2)

#define TILE_CHUNKS (BR * (HDIM / 8))

// 2026-09-25: Synchronous 16-byte copy; the callers check bounds and store
// zeros instead.


__device__ __forceinline__ void cp16(void* smem_dst, const void* gmem_src) {
    *(uint4*)smem_dst = *(const uint4*)gmem_src;
}




extern "C" __global__ void attn_prefill(
    const __nv_bfloat16* __restrict__ Q,
    const __nv_bfloat16* __restrict__ K,
    const __nv_bfloat16* __restrict__ V,
    __nv_bfloat16* __restrict__ O,
    const unsigned int seq_len,
    const unsigned int num_q_heads,
    const unsigned int num_kv_heads,
    const unsigned int head_dim,
    const float inv_sqrt_d,
    const unsigned int causal,
    const unsigned int sliding_window
) {
    const unsigned int q_head = blockIdx.x;
    const unsigned int q_block = blockIdx.y;
    const unsigned int batch = blockIdx.z;
    const unsigned int tid = threadIdx.x;
    const unsigned int warp_id = tid / 32;
    const unsigned int lane_id = tid % 32;
    const unsigned int lane_lo = lane_id & 15;
    const unsigned int lane_hi = lane_id >> 4;

    if (q_head >= num_q_heads) return;

    const unsigned int q_start = q_block * BR;
    if (q_start >= seq_len) return;
    const unsigned int q_end = min(q_start + BR, seq_len);
    const unsigned int q_len = q_end - q_start;

    const unsigned int gqa_ratio = num_q_heads / num_kv_heads;
    const unsigned int kv_head = q_head / gqa_ratio;

    const unsigned int q_seq_stride = num_q_heads * head_dim;
    const unsigned int kv_seq_stride = num_kv_heads * head_dim;

    const __nv_bfloat16* Q_batch = Q + batch * seq_len * q_seq_stride;
    const __nv_bfloat16* K_batch = K + batch * seq_len * kv_seq_stride;
    const __nv_bfloat16* V_batch = V + batch * seq_len * kv_seq_stride;
    __nv_bfloat16* O_batch = O + batch * seq_len * q_seq_stride;

    __shared__ __nv_bfloat16 smem_Q[BR][HDIM_PAD];
    __shared__ __nv_bfloat16 smem_K[BC][HDIM_PAD];
    __shared__ __nv_bfloat16 smem_V[BC][HDIM_PAD];
    __shared__ __nv_bfloat16 smem_P[BR][BC];
    __shared__ float smem_S[BR][BC];
    __shared__ float smem_ml[BR][2];
    __shared__ float smem_resc[BR];




    const unsigned int pv_warp_m   = (warp_id & 1) * 16;
    const unsigned int pv_n_start  = (warp_id >> 1) * PV_N_TILES;


    v8f acc_o[PV_N_TILES];
    #pragma unroll
    for (int i = 0; i < PV_N_TILES; i++) acc_o[i] = v8f{0,0,0,0,0,0,0,0};


    unsigned int num_kv_blocks = (seq_len + BC - 1) / BC;
    if (causal) {
        unsigned int max_kv_block = (q_end - 1) / BC;
        num_kv_blocks = min(num_kv_blocks, max_kv_block + 1);
    }


    for (unsigned int r = tid; r < BR; r += 128) {
        smem_ml[r][0] = -1e30f;
        smem_ml[r][1] = 0.0f;
    }


    {
        const unsigned int chunks_per_row = HDIM / 8;
        for (unsigned int idx = tid; idx < TILE_CHUNKS; idx += 128) {
            unsigned int row = idx / chunks_per_row;
            unsigned int chunk = idx % chunks_per_row;
            unsigned int col = chunk * 8;
            unsigned int q_row = q_start + row;
            if (q_row < seq_len) {
                cp16(&smem_Q[row][col],
                     (const void*)&Q_batch[q_row * q_seq_stride + q_head * head_dim + col]);
            } else {
                *((uint4*)&smem_Q[row][col]) = make_uint4(0, 0, 0, 0);
            }
        }
    }
    __syncthreads();

    for (unsigned int kv_block = 0; kv_block < num_kv_blocks; kv_block++) {
        unsigned int kv_start = kv_block * BC;
        unsigned int kv_end = min(kv_start + BC, seq_len);
        unsigned int kv_len = kv_end - kv_start;


        {
            const unsigned int chunks_per_row = HDIM / 8;
            for (unsigned int idx = tid; idx < TILE_CHUNKS; idx += 128) {
                unsigned int row = idx / chunks_per_row;
                unsigned int chunk = idx % chunks_per_row;
                unsigned int col = chunk * 8;
                unsigned int kv_row = kv_start + row;
                if (kv_row < seq_len) {
                    cp16(&smem_K[row][col],
                         (const void*)&K_batch[kv_row * kv_seq_stride + kv_head * head_dim + col]);
                    cp16(&smem_V[row][col],
                         (const void*)&V_batch[kv_row * kv_seq_stride + kv_head * head_dim + col]);
                } else {
                    *((uint4*)&smem_K[row][col]) = make_uint4(0, 0, 0, 0);
                    *((uint4*)&smem_V[row][col]) = make_uint4(0, 0, 0, 0);
                }
            }
        }
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

                    v16bf b;
                    #pragma unroll
                    for (int k = 0; k < 16; k++)
                        b[k] = (__bf16)(float)smem_K[key_row][k_off + k];
                    acc_s[nt] = __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32(a, b, acc_s[nt]);
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






        if (tid < BR) {
            unsigned int r = tid;
            unsigned int qr = q_start + r;
            bool row_valid = (r < q_len);


            float rmax = -1e30f;
            #pragma unroll
            for (unsigned int c = 0; c < BC; c++) {
                float s = smem_S[r][c] * inv_sqrt_d;
                unsigned int kpos = kv_start + c;
                bool masked = (c >= kv_len) || !row_valid;
                if (causal && kpos > qr) masked = true;
                if (causal && sliding_window > 0 && kpos <= qr &&
                    (qr - kpos) >= sliding_window) masked = true;
                if (masked) s = -1e30f;
                smem_S[r][c] = s;
                rmax = fmaxf(rmax, s);
            }

            float m_old = smem_ml[r][0];
            float l_old = smem_ml[r][1];
            float m_new = fmaxf(m_old, rmax);
            float resc = __expf(m_old - m_new);

            float sum = 0.0f;
            #pragma unroll
            for (unsigned int c = 0; c < BC; c++) {
                float p = __expf(smem_S[r][c] - m_new);
                smem_P[r][c] = __float2bfloat16(p);
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


                v16bf a;
                #pragma unroll
                for (int i = 0; i < 16; i++)
                    a[i] = (__bf16)(float)smem_P[pv_warp_m + lane_lo][k_off + i];

                #pragma unroll
                for (int nt = 0; nt < PV_N_TILES; nt++) {
                    unsigned int d_col = (pv_n_start + nt) * 16 + lane_lo;

                    v16bf b;
                    #pragma unroll
                    for (int k = 0; k < 16; k++)
                        b[k] = (__bf16)(float)smem_V[k_off + k][d_col];
                    acc_o[nt] = __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32(a, b, acc_o[nt]);
                }
            }
        }
        __syncthreads();
    }






    {
        __nv_bfloat16* o_base = O_batch + q_head * head_dim;
        #pragma unroll
        for (int nt = 0; nt < PV_N_TILES; nt++) {
            unsigned int col = (pv_n_start + nt) * 16 + lane_lo;
            #pragma unroll
            for (int e = 0; e < 8; e++) {
                unsigned int row = pv_warp_m + 2 * e + lane_hi;
                unsigned int gr = q_start + row;
                if (gr < seq_len && row < q_len && col < head_dim) {
                    float l = smem_ml[row][1];
                    float inv_l = (l > 0.0f) ? (1.0f / l) : 0.0f;
                    o_base[gr * q_seq_stride + col] = __float2bfloat16(acc_o[nt][e] * inv_l);
                }
            }
        }
    }
}

// 2026-09-25: attn_prefill_64. QK^T runs on waves 0..QK64_WARPS-1. Wave w
// takes query rows from (w % QK64_WARPS) * 16 and head_dim columns from
// (w / QK64_WARPS) * 128; a wave whose columns fall past HDIM skips PV and
// the store.













#if defined(__SCALE__) || defined(__HIP_PLATFORM_AMD__)
#define BR64 32
#else
#define BR64 64
#endif



#define QK64_WARPS (BR64 / 16)
#define TILE_CHUNKS_Q64 (BR64 * (HDIM / 8))
#define TILE_CHUNKS_KV  (BC * (HDIM / 8))

extern "C" __global__ void attn_prefill_64(
    const __nv_bfloat16* __restrict__ Q,
    const __nv_bfloat16* __restrict__ K,
    const __nv_bfloat16* __restrict__ V,
    __nv_bfloat16* __restrict__ O,
    const unsigned int seq_len,
    const unsigned int num_q_heads,
    const unsigned int num_kv_heads,
    const unsigned int head_dim,
    const float inv_sqrt_d,
    const unsigned int causal,
    const unsigned int sliding_window
) {
    const unsigned int q_head = blockIdx.x;
    const unsigned int q_block = blockIdx.y;
    const unsigned int batch = blockIdx.z;
    const unsigned int tid = threadIdx.x;
    const unsigned int warp_id = tid / 32;
    const unsigned int lane_id = tid % 32;
    const unsigned int lane_lo = lane_id & 15;
    const unsigned int lane_hi = lane_id >> 4;

    if (q_head >= num_q_heads) return;

    const unsigned int q_start = q_block * BR64;
    if (q_start >= seq_len) return;
    const unsigned int q_end = min(q_start + BR64, seq_len);
    const unsigned int q_len = q_end - q_start;

    const unsigned int gqa_ratio = num_q_heads / num_kv_heads;
    const unsigned int kv_head = q_head / gqa_ratio;

    const unsigned int q_seq_stride = num_q_heads * head_dim;
    const unsigned int kv_seq_stride = num_kv_heads * head_dim;

    const __nv_bfloat16* Q_batch = Q + batch * seq_len * q_seq_stride;
    const __nv_bfloat16* K_batch = K + batch * seq_len * kv_seq_stride;
    const __nv_bfloat16* V_batch = V + batch * seq_len * kv_seq_stride;
    __nv_bfloat16* O_batch = O + batch * seq_len * q_seq_stride;

    __shared__ __nv_bfloat16 smem_Q[BR64][HDIM_PAD];
    __shared__ __nv_bfloat16 smem_K[BC][HDIM_PAD];
    __shared__ __nv_bfloat16 smem_V[BC][HDIM_PAD];
    __shared__ __nv_bfloat16 smem_P[BR64][BC];
    __shared__ float smem_S[BR64][BC];
    __shared__ float smem_ml[BR64][2];
    __shared__ float smem_resc[BR64];



    const unsigned int pv_warp_m  = (warp_id % QK64_WARPS) * 16;
    const unsigned int pv_n_start = (warp_id / QK64_WARPS) * PV_N_TILES;

    v8f acc_o[PV_N_TILES];
    #pragma unroll
    for (int i = 0; i < PV_N_TILES; i++) acc_o[i] = v8f{0,0,0,0,0,0,0,0};

    unsigned int num_kv_blocks = (seq_len + BC - 1) / BC;
    if (causal) {
        unsigned int max_kv_block = (q_end - 1) / BC;
        num_kv_blocks = min(num_kv_blocks, max_kv_block + 1);
    }

    for (unsigned int r = tid; r < BR64; r += 256) {
        smem_ml[r][0] = -1e30f;
        smem_ml[r][1] = 0.0f;
    }


    {
        const unsigned int chunks_per_row = HDIM / 8;
        for (unsigned int idx = tid; idx < TILE_CHUNKS_Q64; idx += 256) {
            unsigned int row = idx / chunks_per_row;
            unsigned int chunk = idx % chunks_per_row;
            unsigned int col = chunk * 8;
            unsigned int q_row = q_start + row;
            if (q_row < seq_len) {
                cp16(&smem_Q[row][col],
                     (const void*)&Q_batch[q_row * q_seq_stride + q_head * head_dim + col]);
            } else {
                *((uint4*)&smem_Q[row][col]) = make_uint4(0, 0, 0, 0);
            }
        }
    }
    __syncthreads();

    for (unsigned int kv_block = 0; kv_block < num_kv_blocks; kv_block++) {
        unsigned int kv_start = kv_block * BC;
        unsigned int kv_end = min(kv_start + BC, seq_len);
        unsigned int kv_len = kv_end - kv_start;


        {
            const unsigned int chunks_per_row = HDIM / 8;
            for (unsigned int idx = tid; idx < TILE_CHUNKS_KV; idx += 256) {
                unsigned int row = idx / chunks_per_row;
                unsigned int chunk = idx % chunks_per_row;
                unsigned int col = chunk * 8;
                unsigned int kv_row = kv_start + row;
                if (kv_row < seq_len) {
                    cp16(&smem_K[row][col],
                         (const void*)&K_batch[kv_row * kv_seq_stride + kv_head * head_dim + col]);
                    cp16(&smem_V[row][col],
                         (const void*)&V_batch[kv_row * kv_seq_stride + kv_head * head_dim + col]);
                } else {
                    *((uint4*)&smem_K[row][col]) = make_uint4(0, 0, 0, 0);
                    *((uint4*)&smem_V[row][col]) = make_uint4(0, 0, 0, 0);
                }
            }
        }
        __syncthreads();


        if (warp_id < QK64_WARPS) {
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
                    v16bf b;
                    #pragma unroll
                    for (int k = 0; k < 16; k++)
                        b[k] = (__bf16)(float)smem_K[key_row][k_off + k];
                    acc_s[nt] = __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32(a, b, acc_s[nt]);
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
            unsigned int qr = q_start + r;
            bool row_valid = (r < q_len);

            float rmax = -1e30f;
            #pragma unroll
            for (unsigned int c = 0; c < BC; c++) {
                float s = smem_S[r][c] * inv_sqrt_d;
                unsigned int kpos = kv_start + c;
                bool masked = (c >= kv_len) || !row_valid;
                if (causal && kpos > qr) masked = true;
                if (causal && sliding_window > 0 && kpos <= qr &&
                    (qr - kpos) >= sliding_window) masked = true;
                if (masked) s = -1e30f;
                smem_S[r][c] = s;
                rmax = fmaxf(rmax, s);
            }

            float m_old = smem_ml[r][0];
            float l_old = smem_ml[r][1];
            float m_new = fmaxf(m_old, rmax);
            float resc = __expf(m_old - m_new);

            float sum = 0.0f;
            #pragma unroll
            for (unsigned int c = 0; c < BC; c++) {
                float p = __expf(smem_S[r][c] - m_new);
                smem_P[r][c] = __float2bfloat16(p);
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





        if ((pv_n_start + PV_N_TILES) * K16 <= HDIM) {
            #pragma unroll
            for (unsigned int ks = 0; ks < PV_K_STEPS; ks++) {
                unsigned int k_off = ks * K16;
                v16bf a;
                #pragma unroll
                for (int i = 0; i < 16; i++)
                    a[i] = (__bf16)(float)smem_P[pv_warp_m + lane_lo][k_off + i];

                #pragma unroll
                for (int nt = 0; nt < PV_N_TILES; nt++) {
                    unsigned int d_col = (pv_n_start + nt) * 16 + lane_lo;
                    v16bf b;
                    #pragma unroll
                    for (int k = 0; k < 16; k++)
                        b[k] = (__bf16)(float)smem_V[k_off + k][d_col];
                    acc_o[nt] = __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32(a, b, acc_o[nt]);
                }
            }
        }
        __syncthreads();
    }


    {
        __nv_bfloat16* o_base = O_batch + q_head * head_dim;
        #pragma unroll
        for (int nt = 0; nt < PV_N_TILES; nt++) {
            unsigned int col = (pv_n_start + nt) * 16 + lane_lo;
            #pragma unroll
            for (int e = 0; e < 8; e++) {
                unsigned int row = pv_warp_m + 2 * e + lane_hi;
                unsigned int gr = q_start + row;
                if (gr < seq_len && row < q_len && col < head_dim) {
                    float l = smem_ml[row][1];
                    float inv_l = (l > 0.0f) ? (1.0f / l) : 0.0f;
                    o_base[gr * q_seq_stride + col] = __float2bfloat16(acc_o[nt][e] * inv_l);
                }
            }
        }
    }
}
