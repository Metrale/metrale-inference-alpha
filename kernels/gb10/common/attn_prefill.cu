// SPDX-License-Identifier: MIT OR Apache-2.0

// 2026-09-25: Tensor-core flash attention for prefill: tiled online softmax over the KV
// sequence, with BF16 mma.sync m16n8k16 for S = Q K^T and for O += P V.
//
// Owner: gb10 kernels.
// Invariants:
// - attn_prefill runs 128 threads (4 warps) on BR = 32 query rows. QK^T: warps
//   0-1 own 16 rows each. PV: warps w and w+2 share rows (w & 1) * 16, each
//   taking half of the HDIM/8 output n-tiles.
// - attn_prefill_64 runs 256 threads (8 warps) on BR64 query rows and assumes
//   BC = 32 (its score array is acc_s[4][4]).
// - BC and HDIM are compile-time; `head_dim` must equal HDIM, since every row
//   load moves HDIM elements. BR must stay 32: the warp mapping splits a 32-row
//   tile across two warp pairs.
// - A Q, K or V row at or past seq_len loads as zeros; scores outside the
//   causal, sliding-window and length bounds are set to -1e30 before the softmax.
//
// Shared memory at the default shape (HDIM 256, BC 32, row stride 264 BF16):
//   Q:   [32][264] BF16 = 16.5 KB
//   K:   [2][32][264] BF16 = 33.0 KB  (double-buffered: K[i] in use, K[i+1] loading)
//   V:   [32][264] BF16 = 16.5 KB  (loaded async during QK^T, used for PV)
//   P:   [32][40]  BF16 =  2.5 KB
//   m/l: [32][2]   FP32 = 0.25 KB
//   Total: 68.75 KB





#include <cuda_bf16.h>

#define BR 32
// 2026-09-25: BC and HDIM are overridable so a target can instantiate this kernel at
// another shape; the Gemma-4 attn_prefill_512tc sources use HDIM 512 and BC 16.
// At HDIM 512 attn_prefill needs 135,936 B of shared memory at BC 32 and 84,992 B
// at BC 16.

#ifndef BC
#define BC 32
#endif
#ifndef HDIM
#ifndef HDIM
#define HDIM 256
#endif
#endif
#define PAD_KV 8           // 2026-09-25: 16-byte row alignment: (HDIM + 8) * 2 bytes per row
#define HDIM_PAD (HDIM + PAD_KV)
#define PAD_P 8            // 2026-09-25: P row stride BC + PAD_P; (BC + 8) * 2 bytes is a multiple of 16
#define N_TILES_PER_WARP ((HDIM / 8) / 2)  // 2026-09-25: total n-tiles / 2 warp pairs

// 2026-09-25: 16-byte (8-element) chunks per Q tile: BR * HDIM / 8.
#define TILE_CHUNKS (BR * (HDIM / 8))
// 2026-09-25: K and V tiles are BC rows, so their load loops use this bound.




#define TILE_CHUNKS_KV_TILE (BC * (HDIM / 8))

// 2026-09-25: Under SCALE, smem_K is single-buffered (METRALE_KBUFN 1) and BR64 is 32.



#if defined(__SCALE__)
#define METRALE_KBUFN 1
#define METRALE_KB(x) 0u
#else
#define METRALE_KBUFN 2
#define METRALE_KB(x) (x)
#endif

// 2026-09-25: The entry name is overridable alongside the shape, so two instantiations
// do not collide when both are compiled into one module.
#ifndef METRALE_PREFILL_ENTRY
#define METRALE_PREFILL_ENTRY attn_prefill
#endif
extern "C" __global__ void METRALE_PREFILL_ENTRY(
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
    const unsigned int sliding_window   // 2026-09-25: 0 = no sliding limit; >0 = mask out k where q - k >= sliding_window
) {
    const unsigned int q_head = blockIdx.x;
    const unsigned int q_block = blockIdx.y;
    const unsigned int batch = blockIdx.z;
    const unsigned int tid = threadIdx.x;
    const unsigned int warp_id = tid / 32;
    const unsigned int lane_id = tid % 32;

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

    // 2026-09-25: Shared memory: K double-buffered (single under SCALE), and V separate so its load overlaps QK^T.
    __shared__ __nv_bfloat16 smem_Q[BR][HDIM_PAD];
    __shared__ __nv_bfloat16 smem_K[METRALE_KBUFN][BC][HDIM_PAD];  // 2026-09-25: double-buffered except under SCALE
    __shared__ __nv_bfloat16 smem_V[BC][HDIM_PAD];
    __shared__ __nv_bfloat16 smem_P[BR][BC + PAD_P];
    __shared__ float smem_ml[BR][2]; // 2026-09-25: [row][0] = m, [row][1] = l


    const unsigned int group_id = lane_id >> 2;
    const unsigned int tid_in_group = lane_id & 3;


    const unsigned int qk_warp_m = (warp_id & 1) * 16;
    const unsigned int pv_warp_m = (warp_id & 1) * 16;
    const unsigned int pv_n_start = (warp_id >> 1) * N_TILES_PER_WARP;


    float acc_o[N_TILES_PER_WARP][4];
    #pragma unroll
    for (int i = 0; i < N_TILES_PER_WARP; i++) {
        acc_o[i][0] = 0.0f; acc_o[i][1] = 0.0f;
        acc_o[i][2] = 0.0f; acc_o[i][3] = 0.0f;
    }

    // 2026-09-25: Per-row online softmax state (each thread owns 2 rows).
    float m_r0 = -1e30f, m_r1 = -1e30f;
    float l_r0 = 0.0f, l_r1 = 0.0f;

    const unsigned int p_smem_stride = BC + PAD_P;

    // 2026-09-25: KV block count (computed early for the merged load).
    unsigned int num_kv_blocks = (seq_len + BC - 1) / BC;
    if (causal) {
        unsigned int max_kv_block = (q_end - 1) / BC;
        num_kv_blocks = min(num_kv_blocks, max_kv_block + 1);
    }
    // 2026-09-25: Sliding-window lower bound: queries in this Q-block attend no key below
    // q_start - (sliding_window - 1), so KV blocks entirely below that are
    // fully masked — skip them instead of scoring-and-masking.
    unsigned int kv_block_lo = 0;
    if (causal && sliding_window > 0 && q_start + 1 > sliding_window) {
        kv_block_lo = (q_start + 1 - sliding_window) / BC;
    }

    // 2026-09-25: Merged Q + K[kv_block_lo] load (one cp.async commit group).
    // It saves one commit/wait/sync against separate Q and K loads.
    {
        const unsigned int chunks_per_row = HDIM / 8;


        for (unsigned int idx = tid; idx < TILE_CHUNKS; idx += 128) {
            unsigned int row = idx / chunks_per_row;
            unsigned int chunk = idx % chunks_per_row;
            unsigned int col = chunk * 8;
            unsigned int q_row = q_start + row;
            unsigned int smem_addr = __cvta_generic_to_shared(&smem_Q[row][col]);

            if (q_row < seq_len) {
                const void* gmem = (const void*)&Q_batch[q_row * q_seq_stride + q_head * head_dim + col];
                asm volatile("cp.async.cg.shared.global [%0], [%1], 16;" :: "r"(smem_addr), "l"(gmem));
            } else {
                *((uint4*)&smem_Q[row][col]) = make_uint4(0, 0, 0, 0);
            }
        }

        // 2026-09-25: K[kv_block_lo] tile (same commit group, no extra sync).
        if (num_kv_blocks > 0) {
            for (unsigned int idx = tid; idx < TILE_CHUNKS_KV_TILE; idx += 128) {
                unsigned int row = idx / chunks_per_row;
                unsigned int chunk = idx % chunks_per_row;
                unsigned int col = chunk * 8;
                unsigned int k_row = kv_block_lo * BC + row;
                unsigned int smem_addr = __cvta_generic_to_shared(&smem_K[0][row][col]);

                if (k_row < seq_len) {
                    const void* gmem = (const void*)&K_batch[k_row * kv_seq_stride + kv_head * head_dim + col];
                    asm volatile("cp.async.cg.shared.global [%0], [%1], 16;" :: "r"(smem_addr), "l"(gmem));
                } else {
                    *((uint4*)&smem_K[0][row][col]) = make_uint4(0, 0, 0, 0);
                }
            }
        }

        asm volatile("cp.async.commit_group;");
        asm volatile("cp.async.wait_group 0;");
    }
    __syncthreads();

    for (unsigned int kv_block = kv_block_lo; kv_block < num_kv_blocks; kv_block++) {
        unsigned int kv_start = kv_block * BC;
        unsigned int kv_end = min(kv_start + BC, seq_len);
        unsigned int kv_len = kv_end - kv_start;
        unsigned int buf = (kv_block - kv_block_lo) & 1;

        // 2026-09-25: Start the async V load into smem_V (overlaps with QK^T below).
        {
            const unsigned int chunks_per_row = HDIM / 8;
            for (unsigned int idx = tid; idx < TILE_CHUNKS_KV_TILE; idx += 128) {
                unsigned int row = idx / chunks_per_row;
                unsigned int chunk = idx % chunks_per_row;
                unsigned int col = chunk * 8;
                unsigned int v_row = kv_start + row;
                unsigned int smem_addr = __cvta_generic_to_shared(&smem_V[row][col]);

                if (v_row < seq_len) {
                    const void* gmem = (const void*)&V_batch[v_row * kv_seq_stride + kv_head * head_dim + col];
                    asm volatile("cp.async.cg.shared.global [%0], [%1], 16;" :: "r"(smem_addr), "l"(gmem));
                } else {
                    *((uint4*)&smem_V[row][col]) = make_uint4(0, 0, 0, 0);
                }
            }
            asm volatile("cp.async.commit_group;");  // 2026-09-25: V tile loading in background
        }

        // 2026-09-25: K[kv_block] is already in smem_K[METRALE_KB(buf)] (preloaded or from the previous iteration).

        // 2026-09-25: QK^T (warps 0-1, register-based). BC/8 n-tiles: each
        // mma.sync m16n8k16 yields 8 score columns.


        float acc_s[BC / 8][4];  // 2026-09-25: [n_tile][{row0_c0, row0_c1, row1_c0, row1_c1}]
        if (warp_id < 2) {
            #pragma unroll
            for (int i = 0; i < 4; i++) {
                acc_s[i][0] = 0.0f; acc_s[i][1] = 0.0f;
                acc_s[i][2] = 0.0f; acc_s[i][3] = 0.0f;
            }

            #pragma unroll
            for (unsigned int ks = 0; ks < (HDIM / 16); ks++) {
                unsigned int k_base = ks * 16;



                const unsigned short* sQ_u16 = (const unsigned short*)smem_Q;
                unsigned int ar0 = qk_warp_m + group_id;
                unsigned int ar1 = qk_warp_m + group_id + 8;
                unsigned int ac0 = k_base + tid_in_group * 2;
                unsigned int ac1 = k_base + tid_in_group * 2 + 8;
                unsigned int a0 = *(const unsigned int*)&sQ_u16[ar0 * HDIM_PAD + ac0];
                unsigned int a1 = *(const unsigned int*)&sQ_u16[ar1 * HDIM_PAD + ac0];
                unsigned int a2 = *(const unsigned int*)&sQ_u16[ar0 * HDIM_PAD + ac1];
                unsigned int a3 = *(const unsigned int*)&sQ_u16[ar1 * HDIM_PAD + ac1];

                // 2026-09-25: B fragments: BC/8 n-tiles of K^T.


                const unsigned short* sK_u16 = (const unsigned short*)smem_K[METRALE_KB(buf)];
                #pragma unroll
                for (int nt = 0; nt < (int)(BC / 8); nt++) {
                    unsigned int n_col = nt * 8 + group_id;
                    unsigned int k0 = k_base + tid_in_group * 2;
                    unsigned int k1 = k_base + tid_in_group * 2 + 8;
                    unsigned int b0 = ((unsigned int)sK_u16[n_col * HDIM_PAD + k0 + 1] << 16) |
                                      (unsigned int)sK_u16[n_col * HDIM_PAD + k0];
                    unsigned int b1 = ((unsigned int)sK_u16[n_col * HDIM_PAD + k1 + 1] << 16) |
                                      (unsigned int)sK_u16[n_col * HDIM_PAD + k1];

                    asm volatile(
                        "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
                        "{%0, %1, %2, %3}, "
                        "{%4, %5, %6, %7}, "
                        "{%8, %9}, "
                        "{%10, %11, %12, %13};"
                        : "=f"(acc_s[nt][0]), "=f"(acc_s[nt][1]),
                          "=f"(acc_s[nt][2]), "=f"(acc_s[nt][3])
                        : "r"(a0), "r"(a1), "r"(a2), "r"(a3),
                          "r"(b0), "r"(b1),
                          "f"(acc_s[nt][0]), "f"(acc_s[nt][1]),
                          "f"(acc_s[nt][2]), "f"(acc_s[nt][3])
                    );
                }
            }

            // 2026-09-25: Register-based softmax.
            unsigned int row0 = qk_warp_m + group_id;
            unsigned int row1 = row0 + 8;

            // 2026-09-25: Scale, causal mask and boundary checks (in registers).
            #pragma unroll
            for (int nt = 0; nt < (int)(BC / 8); nt++) {
                acc_s[nt][0] *= inv_sqrt_d;
                acc_s[nt][1] *= inv_sqrt_d;
                acc_s[nt][2] *= inv_sqrt_d;
                acc_s[nt][3] *= inv_sqrt_d;

                unsigned int col0 = nt * 8 + tid_in_group * 2;
                unsigned int col1 = col0 + 1;

                if (causal) {
                    unsigned int qr0 = q_start + row0, qr1 = q_start + row1;
                    if (kv_start + col0 > qr0) acc_s[nt][0] = -1e30f;
                    if (kv_start + col1 > qr0) acc_s[nt][1] = -1e30f;
                    if (kv_start + col0 > qr1) acc_s[nt][2] = -1e30f;
                    if (kv_start + col1 > qr1) acc_s[nt][3] = -1e30f;
                    // 2026-09-25: Sliding-window mask: key position k out-of-window for
                    // query q when (q - k) >= sliding_window. sliding_window=0
                    // (full attn) leaves scores untouched.
                    if (sliding_window > 0) {
                        unsigned int k0 = kv_start + col0, k1 = kv_start + col1;
                        if (k0 <= qr0 && qr0 - k0 >= sliding_window) acc_s[nt][0] = -1e30f;
                        if (k1 <= qr0 && qr0 - k1 >= sliding_window) acc_s[nt][1] = -1e30f;
                        if (k0 <= qr1 && qr1 - k0 >= sliding_window) acc_s[nt][2] = -1e30f;
                        if (k1 <= qr1 && qr1 - k1 >= sliding_window) acc_s[nt][3] = -1e30f;
                    }
                }
                if (col0 >= kv_len) { acc_s[nt][0] = -1e30f; acc_s[nt][2] = -1e30f; }
                if (col1 >= kv_len) { acc_s[nt][1] = -1e30f; acc_s[nt][3] = -1e30f; }
                if (row0 >= q_len) { acc_s[nt][0] = -1e30f; acc_s[nt][1] = -1e30f; }
                if (row1 >= q_len) { acc_s[nt][2] = -1e30f; acc_s[nt][3] = -1e30f; }
            }

            // 2026-09-25: Row max: local max, then warp shuffle across tid_in_group (4 threads).
            float rmax0 = -1e30f, rmax1 = -1e30f;
            #pragma unroll
            for (int nt = 0; nt < (int)(BC / 8); nt++) {
                rmax0 = fmaxf(rmax0, fmaxf(acc_s[nt][0], acc_s[nt][1]));
                rmax1 = fmaxf(rmax1, fmaxf(acc_s[nt][2], acc_s[nt][3]));
            }
            rmax0 = fmaxf(rmax0, __shfl_xor_sync(0xFFFFFFFF, rmax0, 1));
            rmax0 = fmaxf(rmax0, __shfl_xor_sync(0xFFFFFFFF, rmax0, 2));
            rmax1 = fmaxf(rmax1, __shfl_xor_sync(0xFFFFFFFF, rmax1, 1));
            rmax1 = fmaxf(rmax1, __shfl_xor_sync(0xFFFFFFFF, rmax1, 2));

            // 2026-09-25: Online softmax update: rescale the previous accumulators.
            float m_new0 = fmaxf(m_r0, rmax0);
            float exp_old0 = __expf(m_r0 - m_new0);
            l_r0 *= exp_old0;
            #pragma unroll
            for (int i = 0; i < N_TILES_PER_WARP; i++) {
                acc_o[i][0] *= exp_old0; acc_o[i][1] *= exp_old0;
            }
            m_r0 = m_new0;

            float m_new1 = fmaxf(m_r1, rmax1);
            float exp_old1 = __expf(m_r1 - m_new1);
            l_r1 *= exp_old1;
            #pragma unroll
            for (int i = 0; i < N_TILES_PER_WARP; i++) {
                acc_o[i][2] *= exp_old1; acc_o[i][3] *= exp_old1;
            }
            m_r1 = m_new1;

            // 2026-09-25: P = exp(s - m) in registers, written to smem_P.
            float sum0 = 0.0f, sum1 = 0.0f;
            #pragma unroll
            for (int nt = 0; nt < (int)(BC / 8); nt++) {
                float p00 = __expf(acc_s[nt][0] - m_r0);
                float p01 = __expf(acc_s[nt][1] - m_r0);
                float p10 = __expf(acc_s[nt][2] - m_r1);
                float p11 = __expf(acc_s[nt][3] - m_r1);
                sum0 += p00 + p01;
                sum1 += p10 + p11;

                unsigned int col0 = nt * 8 + tid_in_group * 2;
                smem_P[row0][col0]     = __float2bfloat16(p00);
                smem_P[row0][col0 + 1] = __float2bfloat16(p01);
                smem_P[row1][col0]     = __float2bfloat16(p10);
                smem_P[row1][col0 + 1] = __float2bfloat16(p11);
            }

            // 2026-09-25: Row sum reduction via warp shuffle.
            sum0 += __shfl_xor_sync(0xFFFFFFFF, sum0, 1);
            sum0 += __shfl_xor_sync(0xFFFFFFFF, sum0, 2);
            sum1 += __shfl_xor_sync(0xFFFFFFFF, sum1, 1);
            sum1 += __shfl_xor_sync(0xFFFFFFFF, sum1, 2);
            l_r0 += sum0;
            l_r1 += sum1;

            // 2026-09-25: Share m/l with warps 2-3.
            if (tid_in_group == 0) {
                smem_ml[row0][0] = m_r0; smem_ml[row0][1] = l_r0;
                smem_ml[row1][0] = m_r1; smem_ml[row1][1] = l_r1;
            }
        }

        // 2026-09-25: Wait for the V tile, loaded during QK^T and softmax.
        asm volatile("cp.async.wait_group 0;");
        __syncthreads();

        // 2026-09-25: Warps 2-3: rescale accumulators to match the current m.
        if (warp_id >= 2) {
            unsigned int row0 = pv_warp_m + group_id;
            unsigned int row1 = row0 + 8;
            float cur_m0 = smem_ml[row0][0];
            float cur_m1 = smem_ml[row1][0];
            float exp_r0 = __expf(m_r0 - cur_m0);
            float exp_r1 = __expf(m_r1 - cur_m1);
            #pragma unroll
            for (int i = 0; i < N_TILES_PER_WARP; i++) {
                acc_o[i][0] *= exp_r0; acc_o[i][1] *= exp_r0;
                acc_o[i][2] *= exp_r1; acc_o[i][3] *= exp_r1;
            }
            m_r0 = cur_m0; m_r1 = cur_m1;
        }

        // 2026-09-25: Preload K[i+1] into smem_K[1-buf] (overlaps with PV below).
        if (kv_block + 1 < num_kv_blocks) {
            unsigned int next_kv_start = (kv_block + 1) * BC;
            const unsigned int chunks_per_row_k = HDIM / 8;
            for (unsigned int idx = tid; idx < TILE_CHUNKS_KV_TILE; idx += 128) {
                unsigned int row = idx / chunks_per_row_k;
                unsigned int chunk = idx % chunks_per_row_k;
                unsigned int col = chunk * 8;
                unsigned int k_row = next_kv_start + row;
                unsigned int smem_addr = __cvta_generic_to_shared(&smem_K[METRALE_KB(1 - buf)][row][col]);

                if (k_row < seq_len) {
                    const void* gmem = (const void*)&K_batch[k_row * kv_seq_stride + kv_head * head_dim + col];
                    asm volatile("cp.async.cg.shared.global [%0], [%1], 16;" :: "r"(smem_addr), "l"(gmem));
                } else {
                    *((uint4*)&smem_K[METRALE_KB(1 - buf)][row][col]) = make_uint4(0, 0, 0, 0);
                }
            }
            asm volatile("cp.async.commit_group;");  // 2026-09-25: K[i+1] loading in background
        }

        // 2026-09-25: PV MMA (all 4 warps, N_TILES_PER_WARP n-tiles each, V from smem_V).
        {
            #pragma unroll
            // 2026-09-25: BC/16 k-steps: each mma.sync m16n8k16 contracts 16 of
            // the BC kv rows.

            for (unsigned int ks = 0; ks < (BC / 16); ks++) {
                unsigned int k_off = ks * 16;



                const unsigned short* sP_u16 = (const unsigned short*)smem_P;
                unsigned int ar0 = pv_warp_m + group_id;
                unsigned int ar1 = pv_warp_m + group_id + 8;
                unsigned int ac0 = k_off + tid_in_group * 2;
                unsigned int ac1 = k_off + tid_in_group * 2 + 8;
                unsigned int a0 = *(const unsigned int*)&sP_u16[ar0 * p_smem_stride + ac0];
                unsigned int a1 = *(const unsigned int*)&sP_u16[ar1 * p_smem_stride + ac0];
                unsigned int a2 = *(const unsigned int*)&sP_u16[ar0 * p_smem_stride + ac1];
                unsigned int a3 = *(const unsigned int*)&sP_u16[ar1 * p_smem_stride + ac1];



                const unsigned short* sV_u16 = (const unsigned short*)smem_V;

                #pragma unroll
                for (int nt = 0; nt < N_TILES_PER_WARP; nt++) {
                    unsigned int n_col = (pv_n_start + nt) * 8 + group_id;
                    unsigned int k0 = k_off + tid_in_group * 2;
                    unsigned int k1 = k_off + tid_in_group * 2 + 8;
                    unsigned int b0 = ((unsigned int)sV_u16[(k0 + 1) * HDIM_PAD + n_col] << 16) |
                                      (unsigned int)sV_u16[k0 * HDIM_PAD + n_col];
                    unsigned int b1 = ((unsigned int)sV_u16[(k1 + 1) * HDIM_PAD + n_col] << 16) |
                                      (unsigned int)sV_u16[k1 * HDIM_PAD + n_col];

                    asm volatile(
                        "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
                        "{%0, %1, %2, %3}, "
                        "{%4, %5, %6, %7}, "
                        "{%8, %9}, "
                        "{%10, %11, %12, %13};"
                        : "=f"(acc_o[nt][0]), "=f"(acc_o[nt][1]),
                          "=f"(acc_o[nt][2]), "=f"(acc_o[nt][3])
                        : "r"(a0), "r"(a1), "r"(a2), "r"(a3),
                          "r"(b0), "r"(b1),
                          "f"(acc_o[nt][0]), "f"(acc_o[nt][1]),
                          "f"(acc_o[nt][2]), "f"(acc_o[nt][3])
                    );
                }
            }
        }

        // 2026-09-25: Wait for the K[i+1] prefetch before the next iteration.
        if (kv_block + 1 < num_kv_blocks) {
            asm volatile("cp.async.wait_group 0;");
        }
        __syncthreads();
    }

    // 2026-09-25: Final normalization and store.
    {
        unsigned int row0 = pv_warp_m + group_id;
        unsigned int row1 = row0 + 8;

        float inv_l0, inv_l1;
        if (warp_id < 2) {
            inv_l0 = (l_r0 > 0.0f) ? (1.0f / l_r0) : 0.0f;
            inv_l1 = (l_r1 > 0.0f) ? (1.0f / l_r1) : 0.0f;
        } else {
            float l0 = smem_ml[row0][1];
            float l1 = smem_ml[row1][1];
            inv_l0 = (l0 > 0.0f) ? (1.0f / l0) : 0.0f;
            inv_l1 = (l1 > 0.0f) ? (1.0f / l1) : 0.0f;
        }

        #pragma unroll
        for (int nt = 0; nt < N_TILES_PER_WARP; nt++) {
            unsigned int col0 = (pv_n_start + nt) * 8 + tid_in_group * 2;

            unsigned int gr0 = q_start + row0;
            unsigned int gr1 = q_start + row1;
            __nv_bfloat16* o_base = O_batch + q_head * head_dim;

            // 2026-09-25: Packed BF16x2 store (one 32-bit write instead of two 16-bit).
            if (gr0 < seq_len && row0 < q_len && col0 < head_dim) {
                unsigned int lo = (unsigned int)__bfloat16_as_ushort(__float2bfloat16(acc_o[nt][0] * inv_l0));
                unsigned int hi = (unsigned int)__bfloat16_as_ushort(__float2bfloat16(acc_o[nt][1] * inv_l0));
                *(unsigned int*)&o_base[gr0 * q_seq_stride + col0] = lo | (hi << 16);
            }
            if (gr1 < seq_len && row1 < q_len && col0 < head_dim) {
                unsigned int lo = (unsigned int)__bfloat16_as_ushort(__float2bfloat16(acc_o[nt][2] * inv_l1));
                unsigned int hi = (unsigned int)__bfloat16_as_ushort(__float2bfloat16(acc_o[nt][3] * inv_l1));
                *(unsigned int*)&o_base[gr1 * q_seq_stride + col0] = lo | (hi << 16);
            }
        }
    }
}

// 2026-09-25: BR=64 variant: 8 warps (256 threads), 64 query rows per block.
//   QK^T: warps 0-3, each owns 16 rows.
//   PV:   all 8 warps in 4 pairs, (w, w+4) -> rows (w & 3) * 16, each warp
//         taking half of the HDIM/8 output n-tiles.
//
// Shared memory at the default shape (HDIM 256, BC 32):
//   Q:   [64][264] = 33.0 KB
//   K:   [2][32][264] = 33.0 KB  (double-buffered)
//   V:   [32][264] = 16.5 KB
//   P:   [64][40]  =  5.0 KB
//   m/l: [64][2]   =  0.5 KB
//   Total: 88 KB








#if defined(__SCALE__)
#define BR64 32
#else
#define BR64 64
#endif
#define TILE_CHUNKS_Q64 (BR64 * (HDIM / 8))
#define TILE_CHUNKS_KV  (BC * (HDIM / 8))

// 2026-09-25: The HDIM 512 instantiations define METRALE_SKIP_PREFILL_64: at HDIM 512,
// BC 16 this variant needs 120,064 B of shared memory.
#ifndef METRALE_SKIP_PREFILL_64
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
    const unsigned int sliding_window   // 2026-09-25: 0 = no sliding limit
) {
    const unsigned int q_head = blockIdx.x;
    const unsigned int q_block = blockIdx.y;
    const unsigned int batch = blockIdx.z;
    const unsigned int tid = threadIdx.x;
    const unsigned int warp_id = tid / 32;
    const unsigned int lane_id = tid % 32;

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
    __shared__ __nv_bfloat16 smem_K64[METRALE_KBUFN][BC][HDIM_PAD];
    __shared__ __nv_bfloat16 smem_V64[BC][HDIM_PAD];
    __shared__ __nv_bfloat16 smem_P64[BR64][BC + PAD_P];
    __shared__ float smem_ml64[BR64][2];

    const unsigned int group_id = lane_id >> 2;
    const unsigned int tid_in_group = lane_id & 3;

    // 2026-09-25: QK^T: warps 0-3, each handles 16 M-rows.
    const unsigned int qk_warp_m = warp_id * 16;  // 2026-09-25: valid for warp_id < 4

    // 2026-09-25: PV: all 8 warps, 4 pairs.
    const unsigned int pv_warp_m = (warp_id & 3) * 16;
    const unsigned int pv_n_start = (warp_id >> 2) * N_TILES_PER_WARP;

    float acc_o[N_TILES_PER_WARP][4];
    #pragma unroll
    for (int i = 0; i < N_TILES_PER_WARP; i++) {
        acc_o[i][0] = 0.0f; acc_o[i][1] = 0.0f;
        acc_o[i][2] = 0.0f; acc_o[i][3] = 0.0f;
    }

    float m_r0 = -1e30f, m_r1 = -1e30f;
    float l_r0 = 0.0f, l_r1 = 0.0f;

    const unsigned int p_smem_stride64 = BC + PAD_P;


    unsigned int num_kv_blocks = (seq_len + BC - 1) / BC;
    if (causal) {
        unsigned int max_kv_block = (q_end - 1) / BC;
        num_kv_blocks = min(num_kv_blocks, max_kv_block + 1);
    }
    // 2026-09-25: Sliding-window lower bound: queries in this Q-block attend no key below
    // q_start - (sliding_window - 1), so KV blocks entirely below that are
    // fully masked — skip them instead of scoring-and-masking.
    unsigned int kv_block_lo = 0;
    if (causal && sliding_window > 0 && q_start + 1 > sliding_window) {
        kv_block_lo = (q_start + 1 - sliding_window) / BC;
    }

    // 2026-09-25: Merged Q + K[kv_block_lo] load (one cp.async commit group).
    {
        const unsigned int chunks_per_row = HDIM / 8;

        // 2026-09-25: Q tile (BR64 rows).
        for (unsigned int idx = tid; idx < TILE_CHUNKS_Q64; idx += 256) {
            unsigned int row = idx / chunks_per_row;
            unsigned int chunk = idx % chunks_per_row;
            unsigned int col = chunk * 8;
            unsigned int q_row = q_start + row;
            unsigned int smem_addr = __cvta_generic_to_shared(&smem_Q[row][col]);

            if (q_row < seq_len) {
                const void* gmem = (const void*)&Q_batch[q_row * q_seq_stride + q_head * head_dim + col];
                asm volatile("cp.async.cg.shared.global [%0], [%1], 16;" :: "r"(smem_addr), "l"(gmem));
            } else {
                *((uint4*)&smem_Q[row][col]) = make_uint4(0, 0, 0, 0);
            }
        }

        // 2026-09-25: K[kv_block_lo] tile (BC rows).
        if (num_kv_blocks > 0) {
            for (unsigned int idx = tid; idx < TILE_CHUNKS_KV; idx += 256) {
                unsigned int row = idx / chunks_per_row;
                unsigned int chunk = idx % chunks_per_row;
                unsigned int col = chunk * 8;
                unsigned int k_row = kv_block_lo * BC + row;
                unsigned int smem_addr = __cvta_generic_to_shared(&smem_K64[0][row][col]);

                if (k_row < seq_len) {
                    const void* gmem = (const void*)&K_batch[k_row * kv_seq_stride + kv_head * head_dim + col];
                    asm volatile("cp.async.cg.shared.global [%0], [%1], 16;" :: "r"(smem_addr), "l"(gmem));
                } else {
                    *((uint4*)&smem_K64[0][row][col]) = make_uint4(0, 0, 0, 0);
                }
            }
        }

        asm volatile("cp.async.commit_group;");
        asm volatile("cp.async.wait_group 0;");
    }
    __syncthreads();

    for (unsigned int kv_block = kv_block_lo; kv_block < num_kv_blocks; kv_block++) {
        unsigned int kv_start = kv_block * BC;
        unsigned int kv_end = min(kv_start + BC, seq_len);
        unsigned int kv_len = kv_end - kv_start;
        unsigned int buf = (kv_block - kv_block_lo) & 1;

        // 2026-09-25: Async V load (overlaps with QK^T).
        {
            const unsigned int chunks_per_row = HDIM / 8;
            for (unsigned int idx = tid; idx < TILE_CHUNKS_KV; idx += 256) {
                unsigned int row = idx / chunks_per_row;
                unsigned int chunk = idx % chunks_per_row;
                unsigned int col = chunk * 8;
                unsigned int v_row = kv_start + row;
                unsigned int smem_addr = __cvta_generic_to_shared(&smem_V64[row][col]);

                if (v_row < seq_len) {
                    const void* gmem = (const void*)&V_batch[v_row * kv_seq_stride + kv_head * head_dim + col];
                    asm volatile("cp.async.cg.shared.global [%0], [%1], 16;" :: "r"(smem_addr), "l"(gmem));
                } else {
                    *((uint4*)&smem_V64[row][col]) = make_uint4(0, 0, 0, 0);
                }
            }
            asm volatile("cp.async.commit_group;");
        }

        // 2026-09-25: QK^T (warps 0-3, each 16 M-rows).
        float acc_s[4][4];
        if (warp_id < 4) {
            #pragma unroll
            for (int i = 0; i < 4; i++) {
                acc_s[i][0] = 0.0f; acc_s[i][1] = 0.0f;
                acc_s[i][2] = 0.0f; acc_s[i][3] = 0.0f;
            }

            const unsigned short* sQ = (const unsigned short*)smem_Q;
            const unsigned short* sK = (const unsigned short*)smem_K64[METRALE_KB(buf)];

            #pragma unroll
            for (unsigned int ks = 0; ks < (HDIM / 16); ks++) {
                unsigned int k_off = ks * 16;
                unsigned int ar0 = qk_warp_m + group_id;
                unsigned int ar1 = qk_warp_m + group_id + 8;
                unsigned int ac0 = k_off + tid_in_group * 2;
                unsigned int ac1 = k_off + tid_in_group * 2 + 8;

                unsigned int a0 = *(const unsigned int*)&sQ[ar0 * HDIM_PAD + ac0];
                unsigned int a1 = *(const unsigned int*)&sQ[ar1 * HDIM_PAD + ac0];
                unsigned int a2 = *(const unsigned int*)&sQ[ar0 * HDIM_PAD + ac1];
                unsigned int a3 = *(const unsigned int*)&sQ[ar1 * HDIM_PAD + ac1];

                #pragma unroll
                for (int nt = 0; nt < (int)(BC / 8); nt++) {
                    unsigned int n_col = nt * 8 + group_id;
                    unsigned int k0 = k_off + tid_in_group * 2;
                    unsigned int k1 = k_off + tid_in_group * 2 + 8;

                    unsigned int b0 = ((unsigned int)sK[n_col * HDIM_PAD + k0 + 1] << 16) |
                                      (unsigned int)sK[n_col * HDIM_PAD + k0];
                    unsigned int b1 = ((unsigned int)sK[n_col * HDIM_PAD + k1 + 1] << 16) |
                                      (unsigned int)sK[n_col * HDIM_PAD + k1];

                    asm volatile(
                        "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
                        "{%0, %1, %2, %3}, "
                        "{%4, %5, %6, %7}, "
                        "{%8, %9}, "
                        "{%10, %11, %12, %13};"
                        : "=f"(acc_s[nt][0]), "=f"(acc_s[nt][1]),
                          "=f"(acc_s[nt][2]), "=f"(acc_s[nt][3])
                        : "r"(a0), "r"(a1), "r"(a2), "r"(a3),
                          "r"(b0), "r"(b1),
                          "f"(acc_s[nt][0]), "f"(acc_s[nt][1]),
                          "f"(acc_s[nt][2]), "f"(acc_s[nt][3])
                    );
                }
            }

            // 2026-09-25: Register-based softmax (warps 0-3).
            unsigned int row0 = qk_warp_m + group_id;
            unsigned int row1 = row0 + 8;

            #pragma unroll
            for (int nt = 0; nt < (int)(BC / 8); nt++) {
                acc_s[nt][0] *= inv_sqrt_d;
                acc_s[nt][1] *= inv_sqrt_d;
                acc_s[nt][2] *= inv_sqrt_d;
                acc_s[nt][3] *= inv_sqrt_d;

                unsigned int col0 = nt * 8 + tid_in_group * 2;
                unsigned int col1 = col0 + 1;

                if (causal) {
                    unsigned int qr0 = q_start + row0, qr1 = q_start + row1;
                    if (kv_start + col0 > qr0) acc_s[nt][0] = -1e30f;
                    if (kv_start + col1 > qr0) acc_s[nt][1] = -1e30f;
                    if (kv_start + col0 > qr1) acc_s[nt][2] = -1e30f;
                    if (kv_start + col1 > qr1) acc_s[nt][3] = -1e30f;
                    // 2026-09-25: Sliding-window mask: key position k out-of-window for
                    // query q when (q - k) >= sliding_window. sliding_window=0
                    // (full attn) leaves scores untouched.
                    if (sliding_window > 0) {
                        unsigned int k0 = kv_start + col0, k1 = kv_start + col1;
                        if (k0 <= qr0 && qr0 - k0 >= sliding_window) acc_s[nt][0] = -1e30f;
                        if (k1 <= qr0 && qr0 - k1 >= sliding_window) acc_s[nt][1] = -1e30f;
                        if (k0 <= qr1 && qr1 - k0 >= sliding_window) acc_s[nt][2] = -1e30f;
                        if (k1 <= qr1 && qr1 - k1 >= sliding_window) acc_s[nt][3] = -1e30f;
                    }
                }
                if (col0 >= kv_len) { acc_s[nt][0] = -1e30f; acc_s[nt][2] = -1e30f; }
                if (col1 >= kv_len) { acc_s[nt][1] = -1e30f; acc_s[nt][3] = -1e30f; }
                if (row0 >= q_len) { acc_s[nt][0] = -1e30f; acc_s[nt][1] = -1e30f; }
                if (row1 >= q_len) { acc_s[nt][2] = -1e30f; acc_s[nt][3] = -1e30f; }
            }

            float rmax0 = -1e30f, rmax1 = -1e30f;
            #pragma unroll
            for (int nt = 0; nt < (int)(BC / 8); nt++) {
                rmax0 = fmaxf(rmax0, fmaxf(acc_s[nt][0], acc_s[nt][1]));
                rmax1 = fmaxf(rmax1, fmaxf(acc_s[nt][2], acc_s[nt][3]));
            }
            rmax0 = fmaxf(rmax0, __shfl_xor_sync(0xFFFFFFFF, rmax0, 1));
            rmax0 = fmaxf(rmax0, __shfl_xor_sync(0xFFFFFFFF, rmax0, 2));
            rmax1 = fmaxf(rmax1, __shfl_xor_sync(0xFFFFFFFF, rmax1, 1));
            rmax1 = fmaxf(rmax1, __shfl_xor_sync(0xFFFFFFFF, rmax1, 2));

            float m_new0 = fmaxf(m_r0, rmax0);
            float exp_old0 = __expf(m_r0 - m_new0);
            l_r0 *= exp_old0;
            #pragma unroll
            for (int i = 0; i < N_TILES_PER_WARP; i++) {
                acc_o[i][0] *= exp_old0; acc_o[i][1] *= exp_old0;
            }
            m_r0 = m_new0;

            float m_new1 = fmaxf(m_r1, rmax1);
            float exp_old1 = __expf(m_r1 - m_new1);
            l_r1 *= exp_old1;
            #pragma unroll
            for (int i = 0; i < N_TILES_PER_WARP; i++) {
                acc_o[i][2] *= exp_old1; acc_o[i][3] *= exp_old1;
            }
            m_r1 = m_new1;

            float sum0 = 0.0f, sum1 = 0.0f;
            #pragma unroll
            for (int nt = 0; nt < (int)(BC / 8); nt++) {
                float p00 = __expf(acc_s[nt][0] - m_r0);
                float p01 = __expf(acc_s[nt][1] - m_r0);
                float p10 = __expf(acc_s[nt][2] - m_r1);
                float p11 = __expf(acc_s[nt][3] - m_r1);
                sum0 += p00 + p01;
                sum1 += p10 + p11;

                unsigned int col0 = nt * 8 + tid_in_group * 2;
                smem_P64[row0][col0]     = __float2bfloat16(p00);
                smem_P64[row0][col0 + 1] = __float2bfloat16(p01);
                smem_P64[row1][col0]     = __float2bfloat16(p10);
                smem_P64[row1][col0 + 1] = __float2bfloat16(p11);
            }

            sum0 += __shfl_xor_sync(0xFFFFFFFF, sum0, 1);
            sum0 += __shfl_xor_sync(0xFFFFFFFF, sum0, 2);
            sum1 += __shfl_xor_sync(0xFFFFFFFF, sum1, 1);
            sum1 += __shfl_xor_sync(0xFFFFFFFF, sum1, 2);
            l_r0 += sum0;
            l_r1 += sum1;

            if (tid_in_group == 0) {
                smem_ml64[row0][0] = m_r0; smem_ml64[row0][1] = l_r0;
                smem_ml64[row1][0] = m_r1; smem_ml64[row1][1] = l_r1;
            }
        }

        // 2026-09-25: Wait for the V tile load.
        asm volatile("cp.async.wait_group 0;");
        __syncthreads();

        // 2026-09-25: Warps 4-7: rescale accumulators to match the current m.
        if (warp_id >= 4) {
            unsigned int row0 = pv_warp_m + group_id;
            unsigned int row1 = row0 + 8;
            float cur_m0 = smem_ml64[row0][0];
            float cur_m1 = smem_ml64[row1][0];
            float exp_r0 = __expf(m_r0 - cur_m0);
            float exp_r1 = __expf(m_r1 - cur_m1);
            #pragma unroll
            for (int i = 0; i < N_TILES_PER_WARP; i++) {
                acc_o[i][0] *= exp_r0; acc_o[i][1] *= exp_r0;
                acc_o[i][2] *= exp_r1; acc_o[i][3] *= exp_r1;
            }
            m_r0 = cur_m0; m_r1 = cur_m1;
        }

        // 2026-09-25: Preload K[i+1] (overlaps with PV below).
        if (kv_block + 1 < num_kv_blocks) {
            unsigned int next_kv_start = (kv_block + 1) * BC;
            const unsigned int chunks_per_row_k = HDIM / 8;
            for (unsigned int idx = tid; idx < TILE_CHUNKS_KV; idx += 256) {
                unsigned int row = idx / chunks_per_row_k;
                unsigned int chunk = idx % chunks_per_row_k;
                unsigned int col = chunk * 8;
                unsigned int k_row = next_kv_start + row;
                unsigned int smem_addr = __cvta_generic_to_shared(&smem_K64[METRALE_KB(1 - buf)][row][col]);

                if (k_row < seq_len) {
                    const void* gmem = (const void*)&K_batch[k_row * kv_seq_stride + kv_head * head_dim + col];
                    asm volatile("cp.async.cg.shared.global [%0], [%1], 16;" :: "r"(smem_addr), "l"(gmem));
                } else {
                    *((uint4*)&smem_K64[METRALE_KB(1 - buf)][row][col]) = make_uint4(0, 0, 0, 0);
                }
            }
            asm volatile("cp.async.commit_group;");
        }

        // 2026-09-25: PV MMA (all 8 warps, N_TILES_PER_WARP n-tiles each).
        {
            const unsigned short* sP = (const unsigned short*)smem_P64;
            const unsigned short* sV = (const unsigned short*)smem_V64;

            #pragma unroll
            for (unsigned int ks = 0; ks < 2; ks++) {
                unsigned int k_off = ks * 16;
                unsigned int ar0 = pv_warp_m + group_id;
                unsigned int ar1 = pv_warp_m + group_id + 8;
                unsigned int ac0 = k_off + tid_in_group * 2;
                unsigned int ac1 = k_off + tid_in_group * 2 + 8;

                unsigned int a0 = *(const unsigned int*)&sP[ar0 * p_smem_stride64 + ac0];
                unsigned int a1 = *(const unsigned int*)&sP[ar1 * p_smem_stride64 + ac0];
                unsigned int a2 = *(const unsigned int*)&sP[ar0 * p_smem_stride64 + ac1];
                unsigned int a3 = *(const unsigned int*)&sP[ar1 * p_smem_stride64 + ac1];

                #pragma unroll
                for (int nt = 0; nt < N_TILES_PER_WARP; nt++) {
                    unsigned int n_col = (pv_n_start + nt) * 8 + group_id;
                    unsigned int k0 = k_off + tid_in_group * 2;
                    unsigned int k1 = k_off + tid_in_group * 2 + 8;

                    unsigned int b0 = ((unsigned int)sV[(k0 + 1) * HDIM_PAD + n_col] << 16) |
                                      (unsigned int)sV[k0 * HDIM_PAD + n_col];
                    unsigned int b1 = ((unsigned int)sV[(k1 + 1) * HDIM_PAD + n_col] << 16) |
                                      (unsigned int)sV[k1 * HDIM_PAD + n_col];

                    asm volatile(
                        "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
                        "{%0, %1, %2, %3}, "
                        "{%4, %5, %6, %7}, "
                        "{%8, %9}, "
                        "{%10, %11, %12, %13};"
                        : "=f"(acc_o[nt][0]), "=f"(acc_o[nt][1]),
                          "=f"(acc_o[nt][2]), "=f"(acc_o[nt][3])
                        : "r"(a0), "r"(a1), "r"(a2), "r"(a3),
                          "r"(b0), "r"(b1),
                          "f"(acc_o[nt][0]), "f"(acc_o[nt][1]),
                          "f"(acc_o[nt][2]), "f"(acc_o[nt][3])
                    );
                }
            }
        }

        // 2026-09-25: Wait for the K[i+1] prefetch.
        if (kv_block + 1 < num_kv_blocks) {
            asm volatile("cp.async.wait_group 0;");
        }
        __syncthreads();
    }

    // 2026-09-25: Final normalization and store.
    {
        unsigned int row0 = pv_warp_m + group_id;
        unsigned int row1 = row0 + 8;

        float inv_l0, inv_l1;
        if (warp_id < 4) {
            inv_l0 = (l_r0 > 0.0f) ? (1.0f / l_r0) : 0.0f;
            inv_l1 = (l_r1 > 0.0f) ? (1.0f / l_r1) : 0.0f;
        } else {
            float l0 = smem_ml64[row0][1];
            float l1 = smem_ml64[row1][1];
            inv_l0 = (l0 > 0.0f) ? (1.0f / l0) : 0.0f;
            inv_l1 = (l1 > 0.0f) ? (1.0f / l1) : 0.0f;
        }

        __nv_bfloat16* o_base = O_batch + q_head * head_dim;

        #pragma unroll
        for (int nt = 0; nt < N_TILES_PER_WARP; nt++) {
            unsigned int col0 = (pv_n_start + nt) * 8 + tid_in_group * 2;
            unsigned int gr0 = q_start + row0;
            unsigned int gr1 = q_start + row1;

            if (gr0 < seq_len && row0 < q_len && col0 < head_dim) {
                unsigned int lo = (unsigned int)__bfloat16_as_ushort(__float2bfloat16(acc_o[nt][0] * inv_l0));
                unsigned int hi = (unsigned int)__bfloat16_as_ushort(__float2bfloat16(acc_o[nt][1] * inv_l0));
                *(unsigned int*)&o_base[gr0 * q_seq_stride + col0] = lo | (hi << 16);
            }
            if (gr1 < seq_len && row1 < q_len && col0 < head_dim) {
                unsigned int lo = (unsigned int)__bfloat16_as_ushort(__float2bfloat16(acc_o[nt][2] * inv_l1));
                unsigned int hi = (unsigned int)__bfloat16_as_ushort(__float2bfloat16(acc_o[nt][3] * inv_l1));
                *(unsigned int*)&o_base[gr1 * q_seq_stride + col0] = lo | (hi << 16);
            }
        }
    }
}
#endif
