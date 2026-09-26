// SPDX-License-Identifier: MIT OR Apache-2.0

// 2026-09-25: Body of the paged prefill flash-attention kernels: KERNEL_NAME,
// with BR = 32 query rows per CTA and 128 threads, and KERNEL_NAME##_64, with
// BR64 rows and 256 threads. Grid x is the Q head and y the query tile; under
// PREFILL_BATCHED z is the stream. The V tile's load is issued before QK^T and
// the next K tile's before P*V.
//
// The including file defines LOAD_KV_TILE(cache, block_table, smem, kv_start,
// kv_len, kv_head, tid, stride), KERNEL_NAME, K_CACHE_TYPE, V_CACHE_TYPE,
// KERNEL_EXTRA_PARAMS (which declares inv_sqrt_d) and KERNEL_PREAMBLE.
//
// Owner: gb10 kernels.
// Invariants:
// - No row at or past the stream's query length is stored.
// - Keys at or past kv_len score -1e30, as do keys after the query row when
//   causal_mask_enabled is set.
#include <cuda_bf16.h>
#include <cuda_fp16.h>

// 2026-09-25: 16-byte global-to-shared cp.async copies.
// strix-hip/common/prefill_paged_compute.cuh defines synchronous stand-ins
// with the same names.
__device__ __forceinline__ void metrale_cp16(void* smem_dst, const void* gmem_src) {
    unsigned _s = __cvta_generic_to_shared(smem_dst);
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16;" :: "r"(_s), "l"(gmem_src));
}
__device__ __forceinline__ void metrale_cp16_pred(void* smem_dst, const void* gmem_src, bool pred) {
    unsigned _s = __cvta_generic_to_shared(smem_dst);
    unsigned _b = pred ? 16u : 0u;
    asm volatile("cp.async.ca.shared.global [%0], [%1], 16, %2;" :: "r"(_s), "l"(gmem_src), "r"(_b));
}
__device__ __forceinline__ void metrale_cp_commit() { asm volatile("cp.async.commit_group;"); }
__device__ __forceinline__ void metrale_cp_wait()   { asm volatile("cp.async.wait_group 0;"); }

// 2026-09-25: P*V runs as an FP16 MMA: probabilities in [0, 1] keep 10
// mantissa bits in FP16 against 7 in BF16. QK^T stays BF16. smem_V stays BF16,
// the type the LOAD_KV_TILE macros write, and this helper converts V pairs to
// FP16 per MMA. -DMETRALE_DISABLE_FP16_PV builds a BF16 P*V instead.












__device__ __forceinline__ unsigned int bf16x2_to_f16x2_bits(
    __nv_bfloat16 lo, __nv_bfloat16 hi
) {
    __half2 h2 = __floats2half2_rn(__bfloat162float(lo), __bfloat162float(hi));
    return *reinterpret_cast<const unsigned int*>(&h2);
}

#ifdef METRALE_ATTN_FP8_SMEM
#include <cuda_fp8.h>
// 2026-09-25: With METRALE_ATTN_FP8_SMEM (attn_prefill_paged_fp8.cu), smem_K
// and smem_V hold raw E4M3 bytes, half the size of BF16. These helpers
// dequantize a pair with the includer's `fp8_to_bf16` and k_scale or v_scale
// right before each MMA.





__device__ __forceinline__ unsigned int fp8x2_to_bf16x2_bits(
    __nv_fp8_storage_t lo, __nv_fp8_storage_t hi, float scale
) {
    unsigned short l = __bfloat16_as_ushort(fp8_to_bf16(lo, scale));
    unsigned short h = __bfloat16_as_ushort(fp8_to_bf16(hi, scale));
    return ((unsigned int)h << 16) | (unsigned int)l;
}
__device__ __forceinline__ unsigned int fp8x2_to_f16x2_bits(
    __nv_fp8_storage_t lo, __nv_fp8_storage_t hi, float scale
) {
    return bf16x2_to_f16x2_bits(fp8_to_bf16(lo, scale), fp8_to_bf16(hi, scale));
}
#endif

// 2026-09-25: Softmax exp: __expf, unless the build passes
// -DMETRALE_FAST_SOFTMAX_EXP, which selects a degree-3 polynomial for 2^tf.
// Its relative error reaches 0.56% as tf approaches 1 (computed 2026-09-25
// from the coefficients below).








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
// 2026-09-25: An includer may define PAD_KV first (attn_prefill_paged_fp8.cu
// uses 16). The default 8 makes the BF16 row stride (HDIM + 8) * 2 bytes, a
// multiple of 16 whenever HDIM is a multiple of 8.


#ifndef PAD_KV
#define PAD_KV 8
#endif
#define HDIM_PAD (HDIM + PAD_KV)
#define PAD_P 8

// 2026-09-25: The A operands (Q for QK^T, P for P*V) load from shared memory
// with one ldmatrix.x4 instead of four 32-bit loads, unless the build passes
// -DMETRALE_DISABLE_ATTN_LDMATRIX.



#ifndef METRALE_DISABLE_ATTN_LDMATRIX
#define METRALE_ATTN_LDMATRIX
#endif
#define N_TILES_PER_WARP ((HDIM / 8) / 2)
#define TILE_CHUNKS (BR * (HDIM / 8))

// 2026-09-25: SCALE builds keep one smem_K buffer instead of two. At HDIM=256
// that takes the BR=32 kernel's static shared memory from 70,400 B to
// 53,504 B, under gfx1151's 64 KB per workgroup. One buffer is race-free: a
// __syncthreads() separates each QK^T read of smem_K from the next K tile's
// load, and P*V does not read smem_K.




#if defined(__SCALE__)
#define METRALE_KBUFN 1
#define METRALE_KB(x) 0u
#else
#define METRALE_KBUFN 2
#define METRALE_KB(x) (x)
#endif

extern "C" __global__ void KERNEL_NAME(
    const __nv_bfloat16* __restrict__ Q,
    K_CACHE_TYPE K_cache,
    V_CACHE_TYPE V_cache,
    __nv_bfloat16* __restrict__ O,
#ifdef PREFILL_BATCHED
    // 2026-09-25: block_table_ptrs[b] is stream b's block table, and Q and O
    // are stacked. With cu_seqlens == nullptr every stream has q_len rows and
    // stream b starts at row b * q_len. Otherwise the rows are packed: stream b
    // starts at row cu_seqlens[b] and has cu_seqlens[b+1] - cu_seqlens[b]
    // rows. A non-null kv_lens[b] replaces kv_len. blockIdx.z is b, and the
    // launch's q_len is the longest stream's.





    const int* const* __restrict__ block_table_ptrs,
    const unsigned int batch_size,
    const int* __restrict__ cu_seqlens,
    const int* __restrict__ kv_lens,
#else
    const int* __restrict__ block_table,
#endif
    const unsigned int q_len,
    unsigned int kv_len,
    unsigned int q_offset,
    const unsigned int num_q_heads,
    const unsigned int num_kv_heads,
    const unsigned int head_dim,
    const unsigned int cache_block_size,
    const unsigned int sliding_window,  // 2026-09-25: 0 = none; else keys with q - k >= it are masked
    const unsigned int causal_mask_enabled  // 2026-09-25: 0 disables the causal mask
    KERNEL_EXTRA_PARAMS
) {
    const unsigned int q_head = blockIdx.x;
    const unsigned int q_block = blockIdx.y;
#ifdef PREFILL_BATCHED
    const unsigned int b = blockIdx.z;
    if (b >= batch_size) return;
    const int* const __restrict__ block_table = block_table_ptrs[b];


    unsigned int q_base_b = 0;
    unsigned int q_len_eff = q_len;
    if (cu_seqlens != nullptr) {
        q_base_b = (unsigned int)cu_seqlens[b];
        q_len_eff = (unsigned int)(cu_seqlens[b + 1] - cu_seqlens[b]);
    } else {
        q_base_b = b * q_len;
    }
    if (kv_lens != nullptr) {
        kv_len = (unsigned int)kv_lens[b];
    }
#else
    const unsigned int q_len_eff = q_len;
#endif
    const unsigned int tid = threadIdx.x;
    const unsigned int warp_id = tid / 32;
    const unsigned int lane_id = tid % 32;

    if (q_head >= num_q_heads) return;
    const unsigned int q_start = q_block * BR;
    if (q_start >= q_len_eff) return;
    const unsigned int q_tile_end = min(q_start + BR, q_len_eff);
    const unsigned int q_tile_len = q_tile_end - q_start;
    const unsigned int q_seq_stride = num_q_heads * head_dim;
    const unsigned int kv_head = q_head / (num_q_heads / num_kv_heads);
#ifdef PREFILL_BATCHED

    const unsigned long long q_batch_off = (unsigned long long)q_base_b * q_seq_stride;
#endif

    __shared__ __nv_bfloat16 smem_Q[BR][HDIM_PAD];
#ifdef METRALE_ATTN_FP8_SMEM


    __shared__ __nv_fp8_storage_t smem_K[METRALE_KBUFN][BC][HDIM_PAD];
    __shared__ __nv_fp8_storage_t smem_V[BC][HDIM_PAD];
#else
    __shared__ __nv_bfloat16 smem_K[METRALE_KBUFN][BC][HDIM_PAD];
    __shared__ __nv_bfloat16 smem_V[BC][HDIM_PAD];
#endif





#ifdef METRALE_DISABLE_FP16_PV
    __shared__ __nv_bfloat16 smem_P[BR][BC + PAD_P];
#else
    __shared__ __half smem_P[BR][BC + PAD_P];
#endif
    __shared__ float smem_ml[BR][2];

    KERNEL_PREAMBLE

    // 2026-09-25: q_rope_pos is the absolute position of chunk row 0, which the
    // causal and sliding-window masks compare key positions against. It is
    // q_offset unless the includer defines Q_ROPE_POS_OVERRIDE and declares it
    // in KERNEL_PREAMBLE (attn_prefill_paged_indirect.cu).
#ifndef Q_ROPE_POS_OVERRIDE
    unsigned int q_rope_pos = q_offset;
#endif

    const unsigned int group_id = lane_id >> 2;
    const unsigned int tid_in_group = lane_id & 3;
    const unsigned int qk_warp_m = (warp_id & 1) * 16;
    const unsigned int pv_warp_m = (warp_id & 1) * 16;
    const unsigned int pv_n_start = (warp_id >> 1) * N_TILES_PER_WARP;
    const unsigned int p_smem_stride = BC + PAD_P;


    float acc_o[N_TILES_PER_WARP][4];
    #pragma unroll
    for (int i = 0; i < N_TILES_PER_WARP; i++) {
        acc_o[i][0] = 0.0f; acc_o[i][1] = 0.0f;
        acc_o[i][2] = 0.0f; acc_o[i][3] = 0.0f;
    }
    float m_r0 = -1e30f, m_r1 = -1e30f;
    float l_r0 = 0.0f, l_r1 = 0.0f;

    unsigned int num_kv_blocks = (kv_len + BC - 1) / BC;
    { unsigned int mx = (q_offset + q_tile_end - 1) / BC;
      num_kv_blocks = min(num_kv_blocks, mx + 1); }
    // 2026-09-25: With causal masking and a sliding window, KV blocks wholly
    // below the tile's first in-window key are skipped: the mask would drop
    // every key in them. q_rope_pos is the position the mask uses.


    unsigned int kv_block_lo = 0;
    if (causal_mask_enabled && sliding_window > 0) {
        unsigned int q_abs_lo = q_rope_pos + q_start;
        if (q_abs_lo + 1 > sliding_window) {
            kv_block_lo = (q_abs_lo + 1 - sliding_window) / BC;
        }
    }
    // 2026-09-25: Clamped so at least one KV block is scanned and the epilogue
    // still writes O.

    if (kv_block_lo >= num_kv_blocks && num_kv_blocks > 0) {
        kv_block_lo = num_kv_blocks - 1;
    }

    // 2026-09-25: Q and the first K tile are loaded, then waited for together.


    {
        const unsigned int cpr = HDIM / 8;
        for (unsigned int idx = tid; idx < TILE_CHUNKS; idx += blockDim.x) {
            unsigned int row = idx / cpr, col = (idx % cpr) * 8;
            if (q_start + row < q_len_eff) {
#ifdef PREFILL_BATCHED
                const void* gm = (const void*)&Q[q_batch_off + (q_start+row)*q_seq_stride + q_head*head_dim + col];
#else
                const void* gm = (const void*)&Q[(q_start+row)*q_seq_stride + q_head*head_dim + col];
#endif
                metrale_cp16(&smem_Q[row][col], gm);
            } else { *((uint4*)&smem_Q[row][col]) = make_uint4(0,0,0,0); }
        }
        if (num_kv_blocks > 0) {
            LOAD_KV_TILE(K_cache, block_table, smem_K[0], kv_block_lo * BC, kv_len, kv_head, tid, blockDim.x);
        }
        metrale_cp_commit();
        metrale_cp_wait();
    }
    __syncthreads();

    for (unsigned int kv_block = kv_block_lo; kv_block < num_kv_blocks; kv_block++) {
        unsigned int kv_start = kv_block * BC;
        unsigned int kv_end = min(kv_start + BC, kv_len);
        unsigned int kv_tile_len = kv_end - kv_start;
        unsigned int buf = (kv_block - kv_block_lo) & 1;


        LOAD_KV_TILE(V_cache, block_table, smem_V, kv_start, kv_len, kv_head, tid, blockDim.x);
        metrale_cp_commit();

        // 2026-09-25: Warps 0-1 compute QK^T and the softmax; warps 2-3 skip to the wait.
        float acc_s[4][4];
        if (warp_id < 2) {
            #pragma unroll
            for (int i = 0; i < 4; i++) { acc_s[i][0]=0; acc_s[i][1]=0; acc_s[i][2]=0; acc_s[i][3]=0; }

            const unsigned short* sQ = (const unsigned short*)smem_Q;
#ifdef METRALE_ATTN_FP8_SMEM
            const __nv_fp8_storage_t* sK = (const __nv_fp8_storage_t*)smem_K[METRALE_KB(buf)];
#else
            const unsigned short* sK = (const unsigned short*)smem_K[METRALE_KB(buf)];
#endif

            #pragma unroll
            for (unsigned int ks = 0; ks < (HDIM/16); ks++) {
                unsigned int kb = ks*16;
                unsigned int a0,a1,a2,a3;
#ifdef METRALE_ATTN_LDMATRIX




                { unsigned int qb=__cvta_generic_to_shared(&sQ[(qk_warp_m+(lane_id&15))*HDIM_PAD+(lane_id>>4)*8+kb]);
                  asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3},[%4];"
                    :"=r"(a0),"=r"(a1),"=r"(a2),"=r"(a3):"r"(qb)); }
#else
                unsigned int ar0=qk_warp_m+group_id, ar1=ar0+8;
                unsigned int ac0=kb+tid_in_group*2, ac1=ac0+8;
                a0=*(const unsigned int*)&sQ[ar0*HDIM_PAD+ac0];
                a1=*(const unsigned int*)&sQ[ar1*HDIM_PAD+ac0];
                a2=*(const unsigned int*)&sQ[ar0*HDIM_PAD+ac1];
                a3=*(const unsigned int*)&sQ[ar1*HDIM_PAD+ac1];
#endif

                #pragma unroll
                for (int nt=0; nt<4; nt++) {
                    unsigned int nc=nt*8+group_id, k0=kb+tid_in_group*2, k1=k0+8;
#ifdef METRALE_ATTN_FP8_SMEM
                    unsigned int b0=fp8x2_to_bf16x2_bits(sK[nc*HDIM_PAD+k0],sK[nc*HDIM_PAD+k0+1],k_scale);
                    unsigned int b1=fp8x2_to_bf16x2_bits(sK[nc*HDIM_PAD+k1],sK[nc*HDIM_PAD+k1+1],k_scale);
#else
                    unsigned int b0=((unsigned int)sK[nc*HDIM_PAD+k0+1]<<16)|(unsigned int)sK[nc*HDIM_PAD+k0];
                    unsigned int b1=((unsigned int)sK[nc*HDIM_PAD+k1+1]<<16)|(unsigned int)sK[nc*HDIM_PAD+k1];
#endif
                    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
                        "{%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%10,%11,%12,%13};"
                        :"=f"(acc_s[nt][0]),"=f"(acc_s[nt][1]),"=f"(acc_s[nt][2]),"=f"(acc_s[nt][3])
                        :"r"(a0),"r"(a1),"r"(a2),"r"(a3),"r"(b0),"r"(b1),
                         "f"(acc_s[nt][0]),"f"(acc_s[nt][1]),"f"(acc_s[nt][2]),"f"(acc_s[nt][3]));
                }
            }


            unsigned int row0=qk_warp_m+group_id, row1=row0+8;
            #pragma unroll
            for (int nt=0; nt<4; nt++) {
                acc_s[nt][0]*=inv_sqrt_d; acc_s[nt][1]*=inv_sqrt_d;
                acc_s[nt][2]*=inv_sqrt_d; acc_s[nt][3]*=inv_sqrt_d;
                unsigned int c0=nt*8+tid_in_group*2, c1=c0+1;
                unsigned int qr0=q_rope_pos+q_start+row0, qr1=q_rope_pos+q_start+row1;
                // 2026-09-25: The DFlash launches in ops/prefill_attn_main_a.rs
                // pass causal_mask_enabled = 0, so the block's queries see
                // each other bidirectionally.


                if(causal_mask_enabled){
                    if(kv_start+c0>qr0) acc_s[nt][0]=-1e30f; if(kv_start+c1>qr0) acc_s[nt][1]=-1e30f;
                    if(kv_start+c0>qr1) acc_s[nt][2]=-1e30f; if(kv_start+c1>qr1) acc_s[nt][3]=-1e30f;
                }
                if(c0>=kv_tile_len){acc_s[nt][0]=-1e30f;acc_s[nt][2]=-1e30f;}
                if(c1>=kv_tile_len){acc_s[nt][1]=-1e30f;acc_s[nt][3]=-1e30f;}
                if(row0>=q_tile_len){acc_s[nt][0]=-1e30f;acc_s[nt][1]=-1e30f;}
                if(row1>=q_tile_len){acc_s[nt][2]=-1e30f;acc_s[nt][3]=-1e30f;}
                // 2026-09-25: Sliding window: a key at or before the query with
                // q - k >= sliding_window is masked.
                if(sliding_window>0){
                    if(qr0>=kv_start+c0 && qr0-(kv_start+c0)>=sliding_window) acc_s[nt][0]=-1e30f;
                    if(qr0>=kv_start+c1 && qr0-(kv_start+c1)>=sliding_window) acc_s[nt][1]=-1e30f;
                    if(qr1>=kv_start+c0 && qr1-(kv_start+c0)>=sliding_window) acc_s[nt][2]=-1e30f;
                    if(qr1>=kv_start+c1 && qr1-(kv_start+c1)>=sliding_window) acc_s[nt][3]=-1e30f;
                }
            }

            float rmax0=-1e30f, rmax1=-1e30f;
            #pragma unroll
            for(int nt=0;nt<4;nt++){
                rmax0=fmaxf(rmax0,fmaxf(acc_s[nt][0],acc_s[nt][1]));
                rmax1=fmaxf(rmax1,fmaxf(acc_s[nt][2],acc_s[nt][3]));
            }
            rmax0=fmaxf(rmax0,__shfl_xor_sync(0xFFFFFFFF,rmax0,1));
            rmax0=fmaxf(rmax0,__shfl_xor_sync(0xFFFFFFFF,rmax0,2));
            rmax1=fmaxf(rmax1,__shfl_xor_sync(0xFFFFFFFF,rmax1,1));
            rmax1=fmaxf(rmax1,__shfl_xor_sync(0xFFFFFFFF,rmax1,2));

            // 2026-09-25: o and l are rescaled only when the row max grows.
            float mn0=fmaxf(m_r0,rmax0);
            if (mn0 != m_r0) {
                float eo0=sw_exp(m_r0-mn0); l_r0*=eo0;
                #pragma unroll
                for(int i=0;i<N_TILES_PER_WARP;i++){acc_o[i][0]*=eo0;acc_o[i][1]*=eo0;}
                m_r0=mn0;
            }
            float mn1=fmaxf(m_r1,rmax1);
            if (mn1 != m_r1) {
                float eo1=sw_exp(m_r1-mn1); l_r1*=eo1;
                #pragma unroll
                for(int i=0;i<N_TILES_PER_WARP;i++){acc_o[i][2]*=eo1;acc_o[i][3]*=eo1;}
                m_r1=mn1;
            }

            float sum0=0, sum1=0;
            #pragma unroll
            for(int nt=0;nt<4;nt++){
                float p00=sw_exp(acc_s[nt][0]-m_r0),p01=sw_exp(acc_s[nt][1]-m_r0);
                float p10=sw_exp(acc_s[nt][2]-m_r1),p11=sw_exp(acc_s[nt][3]-m_r1);
                sum0+=p00+p01; sum1+=p10+p11;
                unsigned int c0=nt*8+tid_in_group*2;
#ifdef METRALE_DISABLE_FP16_PV
                smem_P[row0][c0]=__float2bfloat16_rn(p00); smem_P[row0][c0+1]=__float2bfloat16_rn(p01);
                smem_P[row1][c0]=__float2bfloat16_rn(p10); smem_P[row1][c0+1]=__float2bfloat16_rn(p11);
#else
                smem_P[row0][c0]=__float2half_rn(p00); smem_P[row0][c0+1]=__float2half_rn(p01);
                smem_P[row1][c0]=__float2half_rn(p10); smem_P[row1][c0+1]=__float2half_rn(p11);
#endif
            }
            sum0+=__shfl_xor_sync(0xFFFFFFFF,sum0,1); sum0+=__shfl_xor_sync(0xFFFFFFFF,sum0,2);
            sum1+=__shfl_xor_sync(0xFFFFFFFF,sum1,1); sum1+=__shfl_xor_sync(0xFFFFFFFF,sum1,2);
            l_r0+=sum0; l_r1+=sum1;

            if(tid_in_group==0){
                smem_ml[row0][0]=m_r0; smem_ml[row0][1]=l_r0;
                smem_ml[row1][0]=m_r1; smem_ml[row1][1]=l_r1;
            }
        }


        metrale_cp_wait();
        __syncthreads();

        // 2026-09-25: Warps 2-3 rescale their o to the row max warps 0-1 left in smem_ml.
        if(warp_id>=2){
            unsigned int r0=pv_warp_m+group_id, r1=r0+8;
            float cm0=smem_ml[r0][0], cm1=smem_ml[r1][0];
            if (cm0 != m_r0) {
                float er0=sw_exp(m_r0-cm0);
                #pragma unroll
                for(int i=0;i<N_TILES_PER_WARP;i++){acc_o[i][0]*=er0;acc_o[i][1]*=er0;}
                m_r0=cm0;
            }
            if (cm1 != m_r1) {
                float er1=sw_exp(m_r1-cm1);
                #pragma unroll
                for(int i=0;i<N_TILES_PER_WARP;i++){acc_o[i][2]*=er1;acc_o[i][3]*=er1;}
                m_r1=cm1;
            }
        }


        if(kv_block+1<num_kv_blocks){
            LOAD_KV_TILE(K_cache, block_table, smem_K[METRALE_KB(1-buf)], (kv_block+1)*BC, kv_len, kv_head, tid, blockDim.x);
            metrale_cp_commit();
        }







        {
            const unsigned short* sP=(const unsigned short*)smem_P;
            #pragma unroll
            for(unsigned int ks=0;ks<2;ks++){
                unsigned int ko=ks*16;
                unsigned int a0,a1,a2,a3;
#ifdef METRALE_ATTN_LDMATRIX

                { unsigned int pb=__cvta_generic_to_shared(&sP[(pv_warp_m+(lane_id&15))*p_smem_stride+(lane_id>>4)*8+ko]);
                  asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3},[%4];"
                    :"=r"(a0),"=r"(a1),"=r"(a2),"=r"(a3):"r"(pb)); }
#else
                unsigned int ar0=pv_warp_m+group_id, ar1=ar0+8;
                unsigned int ac0=ko+tid_in_group*2, ac1=ac0+8;
                a0=*(const unsigned int*)&sP[ar0*p_smem_stride+ac0];
                a1=*(const unsigned int*)&sP[ar1*p_smem_stride+ac0];
                a2=*(const unsigned int*)&sP[ar0*p_smem_stride+ac1];
                a3=*(const unsigned int*)&sP[ar1*p_smem_stride+ac1];
#endif
                #pragma unroll
                for(int nt=0;nt<N_TILES_PER_WARP;nt++){
                    unsigned int nc=(pv_n_start+nt)*8+group_id, k0=ko+tid_in_group*2, k1=k0+8;
#ifdef METRALE_DISABLE_FP16_PV
                    const unsigned short* sV=(const unsigned short*)smem_V;
                    unsigned int b0=((unsigned int)sV[(k0+1)*HDIM_PAD+nc]<<16)|(unsigned int)sV[k0*HDIM_PAD+nc];
                    unsigned int b1=((unsigned int)sV[(k1+1)*HDIM_PAD+nc]<<16)|(unsigned int)sV[k1*HDIM_PAD+nc];
                    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
                        "{%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%10,%11,%12,%13};"
                        :"=f"(acc_o[nt][0]),"=f"(acc_o[nt][1]),"=f"(acc_o[nt][2]),"=f"(acc_o[nt][3])
                        :"r"(a0),"r"(a1),"r"(a2),"r"(a3),"r"(b0),"r"(b1),
                         "f"(acc_o[nt][0]),"f"(acc_o[nt][1]),"f"(acc_o[nt][2]),"f"(acc_o[nt][3]));
#else
#ifdef METRALE_ATTN_FP8_SMEM
                    unsigned int b0=fp8x2_to_f16x2_bits(smem_V[k0][nc], smem_V[k0+1][nc], v_scale);
                    unsigned int b1=fp8x2_to_f16x2_bits(smem_V[k1][nc], smem_V[k1+1][nc], v_scale);
#else
                    unsigned int b0=bf16x2_to_f16x2_bits(
                        smem_V[k0][nc], smem_V[k0+1][nc]);
                    unsigned int b1=bf16x2_to_f16x2_bits(
                        smem_V[k1][nc], smem_V[k1+1][nc]);
#endif
                    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
                        "{%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%10,%11,%12,%13};"
                        :"=f"(acc_o[nt][0]),"=f"(acc_o[nt][1]),"=f"(acc_o[nt][2]),"=f"(acc_o[nt][3])
                        :"r"(a0),"r"(a1),"r"(a2),"r"(a3),"r"(b0),"r"(b1),
                         "f"(acc_o[nt][0]),"f"(acc_o[nt][1]),"f"(acc_o[nt][2]),"f"(acc_o[nt][3]));
#endif
                }
            }
        }


        if(kv_block+1<num_kv_blocks){
            metrale_cp_wait();
        }
        __syncthreads();
    }


    {
        unsigned int r0=pv_warp_m+group_id, r1=r0+8;
        float il0,il1;
        if(warp_id<2){
            il0=(l_r0>0)?(1.f/l_r0):0;
            il1=(l_r1>0)?(1.f/l_r1):0;
        } else {
            float lv0=smem_ml[r0][1], lv1=smem_ml[r1][1];
            il0=(lv0>0)?(1.f/lv0):0;
            il1=(lv1>0)?(1.f/lv1):0;
        }

#ifdef PREFILL_BATCHED
        __nv_bfloat16* ob=O+q_batch_off+q_head*head_dim;
#else
        __nv_bfloat16* ob=O+q_head*head_dim;
#endif
        #pragma unroll
        for(int nt=0;nt<N_TILES_PER_WARP;nt++){
            unsigned int c0=(pv_n_start+nt)*8+tid_in_group*2;
            unsigned int gr0=q_start+r0, gr1=q_start+r1;
            if(gr0<q_len_eff&&r0<q_tile_len&&c0<head_dim){
                unsigned int lo=(unsigned int)__bfloat16_as_ushort(__float2bfloat16(acc_o[nt][0]*il0));
                unsigned int hi=(unsigned int)__bfloat16_as_ushort(__float2bfloat16(acc_o[nt][1]*il0));
                *(unsigned int*)&ob[gr0*q_seq_stride+c0]=lo|(hi<<16);
            }
            if(gr1<q_len_eff&&r1<q_tile_len&&c0<head_dim){
                unsigned int lo=(unsigned int)__bfloat16_as_ushort(__float2bfloat16(acc_o[nt][2]*il1));
                unsigned int hi=(unsigned int)__bfloat16_as_ushort(__float2bfloat16(acc_o[nt][3]*il1));
                *(unsigned int*)&ob[gr1*q_seq_stride+c0]=lo|(hi<<16);
            }
        }
    }
}

// 2026-09-25: KERNEL_NAME##_64: BR64 query rows per CTA and 8 warps. Warps 0-3
// compute QK^T and the softmax, 16 rows each, while warps 4-7 load the V tile.
// P*V runs on all 8 warps in pairs (w, w + 4) that share 16 rows, each member
// taking half of the head_dim columns.

















// 2026-09-25: SCALE builds run BR64 = 32 so the kernel fits gfx1151's shared
// memory. The host grid must use the same row count, and does: ops/
// prefill_attn_main_a.rs and prefill_attn_main_b.rs take 32 under
// cfg!(metrale_scale). Otherwise CTAs would be 64 rows apart while each
// writes 32.





#if defined(__SCALE__)
#define BR64 32
#else
#define BR64 64
#endif
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
    const int* __restrict__ cu_seqlens,
    const int* __restrict__ kv_lens,
#else
    const int* __restrict__ block_table,
#endif
    const unsigned int q_len,
    unsigned int kv_len,
    unsigned int q_offset,
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
    // 2026-09-25: Per-stream geometry, as described at the BR=32 kernel's parameters.
    unsigned int q_base_b = 0;
    unsigned int q_len_eff = q_len;
    if (cu_seqlens != nullptr) {
        q_base_b = (unsigned int)cu_seqlens[b];
        q_len_eff = (unsigned int)(cu_seqlens[b + 1] - cu_seqlens[b]);
    } else {
        q_base_b = b * q_len;
    }
    if (kv_lens != nullptr) {
        kv_len = (unsigned int)kv_lens[b];
    }
#else
    const unsigned int q_len_eff = q_len;
#endif
    const unsigned int tid = threadIdx.x;
    const unsigned int warp_id = tid / 32;
    const unsigned int lane_id = tid % 32;

    if (q_head >= num_q_heads) return;
    const unsigned int q_start = q_block * BR64;
    if (q_start >= q_len_eff) return;
    const unsigned int q_tile_end = min(q_start + BR64, q_len_eff);
    const unsigned int q_tile_len = q_tile_end - q_start;
    const unsigned int q_seq_stride = num_q_heads * head_dim;
    const unsigned int kv_head = q_head / (num_q_heads / num_kv_heads);
#ifdef PREFILL_BATCHED
    const unsigned long long q_batch_off = (unsigned long long)q_base_b * q_seq_stride;
#endif

    __shared__ __nv_bfloat16 smem_Q64[BR64][HDIM_PAD];
#ifdef METRALE_ATTN_FP8_SMEM
    __shared__ __nv_fp8_storage_t smem_K64[METRALE_KBUFN][BC][HDIM_PAD];
    __shared__ __nv_fp8_storage_t smem_V64[BC][HDIM_PAD];
#else
    __shared__ __nv_bfloat16 smem_K64[METRALE_KBUFN][BC][HDIM_PAD];
    __shared__ __nv_bfloat16 smem_V64[BC][HDIM_PAD];
#endif

#ifdef METRALE_DISABLE_FP16_PV
    __shared__ __nv_bfloat16 smem_P64[BR64][BC + PAD_P];
#else
    __shared__ __half smem_P64[BR64][BC + PAD_P];
#endif
    __shared__ float smem_ml64[BR64][2];

    KERNEL_PREAMBLE

    // 2026-09-25: q_rope_pos is the absolute position of chunk row 0, which the
    // causal and sliding-window masks compare key positions against. It is
    // q_offset unless the includer defines Q_ROPE_POS_OVERRIDE and declares it
    // in KERNEL_PREAMBLE (attn_prefill_paged_indirect.cu).
#ifndef Q_ROPE_POS_OVERRIDE
    unsigned int q_rope_pos = q_offset;
#endif

    const unsigned int group_id = lane_id >> 2;
    const unsigned int tid_in_group = lane_id & 3;
    const unsigned int qk_warp_m = warp_id * 16;
    const unsigned int pv_warp_m = (warp_id & 3) * 16;
    const unsigned int pv_n_start = (warp_id >> 2) * N_TILES_PER_WARP;
    const unsigned int p_smem_stride64 = BC + PAD_P;

    float acc_o[N_TILES_PER_WARP][4];
    #pragma unroll
    for (int i = 0; i < N_TILES_PER_WARP; i++) {
        acc_o[i][0] = 0.0f; acc_o[i][1] = 0.0f;
        acc_o[i][2] = 0.0f; acc_o[i][3] = 0.0f;
    }
    float m_r0 = -1e30f, m_r1 = -1e30f;
    float l_r0 = 0.0f, l_r1 = 0.0f;

    unsigned int num_kv_blocks = (kv_len + BC - 1) / BC;
    { unsigned int mx = (q_offset + q_tile_end - 1) / BC;
      num_kv_blocks = min(num_kv_blocks, mx + 1); }
    // 2026-09-25: With causal masking and a sliding window, KV blocks wholly
    // below the tile's first in-window key are skipped: the mask would drop
    // every key in them. q_rope_pos is the position the mask uses.


    unsigned int kv_block_lo = 0;
    if (causal_mask_enabled && sliding_window > 0) {
        unsigned int q_abs_lo = q_rope_pos + q_start;
        if (q_abs_lo + 1 > sliding_window) {
            kv_block_lo = (q_abs_lo + 1 - sliding_window) / BC;
        }
    }
    // 2026-09-25: Clamped so at least one KV block is scanned and the epilogue
    // still writes O.

    if (kv_block_lo >= num_kv_blocks && num_kv_blocks > 0) {
        kv_block_lo = num_kv_blocks - 1;
    }

    // 2026-09-25: Q and the first K tile are loaded, then waited for together.
    {
        const unsigned int cpr = HDIM / 8;
        for (unsigned int idx = tid; idx < TILE_CHUNKS_Q64; idx += 256) {
            unsigned int row = idx / cpr, col = (idx % cpr) * 8;
            // 2026-09-25: Bounded by q_len_eff: under VARLEN q_len is the batch
            // maximum, so a short stream bounded by it would load the next
            // stream's rows, and the last stream would read past the packed Q.



            if (q_start + row < q_len_eff) {
#ifdef PREFILL_BATCHED
                const void* gm = (const void*)&Q[q_batch_off + (q_start+row)*q_seq_stride + q_head*head_dim + col];
#else
                const void* gm = (const void*)&Q[(q_start+row)*q_seq_stride + q_head*head_dim + col];
#endif
                metrale_cp16(&smem_Q64[row][col], gm);
            } else { *((uint4*)&smem_Q64[row][col]) = make_uint4(0,0,0,0); }
        }
        if (num_kv_blocks > 0) {
            LOAD_KV_TILE(K_cache, block_table, smem_K64[0], kv_block_lo * BC, kv_len, kv_head, tid, blockDim.x);
        }
        metrale_cp_commit();
        metrale_cp_wait();
    }
    __syncthreads();

    for (unsigned int kv_block = kv_block_lo; kv_block < num_kv_blocks; kv_block++) {
        unsigned int kv_start = kv_block * BC;
        unsigned int kv_end = min(kv_start + BC, kv_len);
        unsigned int kv_tile_len = kv_end - kv_start;
        unsigned int buf = (kv_block - kv_block_lo) & 1;





        float acc_s[4][4];
        if (warp_id < 4) {
            #pragma unroll
            for (int i = 0; i < 4; i++) { acc_s[i][0]=0; acc_s[i][1]=0; acc_s[i][2]=0; acc_s[i][3]=0; }

            const unsigned short* sQ = (const unsigned short*)smem_Q64;
#ifdef METRALE_ATTN_FP8_SMEM
            const __nv_fp8_storage_t* sK = (const __nv_fp8_storage_t*)smem_K64[METRALE_KB(buf)];
#else
            const unsigned short* sK = (const unsigned short*)smem_K64[METRALE_KB(buf)];
#endif

            #pragma unroll
            for (unsigned int ks = 0; ks < (HDIM/16); ks++) {
                unsigned int kb = ks*16;
                unsigned int a0,a1,a2,a3;
#ifdef METRALE_ATTN_LDMATRIX




                { unsigned int qb=__cvta_generic_to_shared(&sQ[(qk_warp_m+(lane_id&15))*HDIM_PAD+(lane_id>>4)*8+kb]);
                  asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3},[%4];"
                    :"=r"(a0),"=r"(a1),"=r"(a2),"=r"(a3):"r"(qb)); }
#else
                unsigned int ar0=qk_warp_m+group_id, ar1=ar0+8;
                unsigned int ac0=kb+tid_in_group*2, ac1=ac0+8;
                a0=*(const unsigned int*)&sQ[ar0*HDIM_PAD+ac0];
                a1=*(const unsigned int*)&sQ[ar1*HDIM_PAD+ac0];
                a2=*(const unsigned int*)&sQ[ar0*HDIM_PAD+ac1];
                a3=*(const unsigned int*)&sQ[ar1*HDIM_PAD+ac1];
#endif

                #pragma unroll
                for (int nt=0; nt<4; nt++) {
                    unsigned int nc=nt*8+group_id, k0=kb+tid_in_group*2, k1=k0+8;
#ifdef METRALE_ATTN_FP8_SMEM
                    unsigned int b0=fp8x2_to_bf16x2_bits(sK[nc*HDIM_PAD+k0],sK[nc*HDIM_PAD+k0+1],k_scale);
                    unsigned int b1=fp8x2_to_bf16x2_bits(sK[nc*HDIM_PAD+k1],sK[nc*HDIM_PAD+k1+1],k_scale);
#else
                    unsigned int b0=((unsigned int)sK[nc*HDIM_PAD+k0+1]<<16)|(unsigned int)sK[nc*HDIM_PAD+k0];
                    unsigned int b1=((unsigned int)sK[nc*HDIM_PAD+k1+1]<<16)|(unsigned int)sK[nc*HDIM_PAD+k1];
#endif
                    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
                        "{%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%10,%11,%12,%13};"
                        :"=f"(acc_s[nt][0]),"=f"(acc_s[nt][1]),"=f"(acc_s[nt][2]),"=f"(acc_s[nt][3])
                        :"r"(a0),"r"(a1),"r"(a2),"r"(a3),"r"(b0),"r"(b1),
                         "f"(acc_s[nt][0]),"f"(acc_s[nt][1]),"f"(acc_s[nt][2]),"f"(acc_s[nt][3]));
                }
            }


            unsigned int row0=qk_warp_m+group_id, row1=row0+8;
            #pragma unroll
            for (int nt=0; nt<4; nt++) {
                acc_s[nt][0]*=inv_sqrt_d; acc_s[nt][1]*=inv_sqrt_d;
                acc_s[nt][2]*=inv_sqrt_d; acc_s[nt][3]*=inv_sqrt_d;
                unsigned int c0=nt*8+tid_in_group*2, c1=c0+1;
                unsigned int qr0=q_rope_pos+q_start+row0, qr1=q_rope_pos+q_start+row1;

                if(causal_mask_enabled){
                    if(kv_start+c0>qr0) acc_s[nt][0]=-1e30f; if(kv_start+c1>qr0) acc_s[nt][1]=-1e30f;
                    if(kv_start+c0>qr1) acc_s[nt][2]=-1e30f; if(kv_start+c1>qr1) acc_s[nt][3]=-1e30f;
                }
                if(c0>=kv_tile_len){acc_s[nt][0]=-1e30f;acc_s[nt][2]=-1e30f;}
                if(c1>=kv_tile_len){acc_s[nt][1]=-1e30f;acc_s[nt][3]=-1e30f;}
                if(row0>=q_tile_len){acc_s[nt][0]=-1e30f;acc_s[nt][1]=-1e30f;}
                if(row1>=q_tile_len){acc_s[nt][2]=-1e30f;acc_s[nt][3]=-1e30f;}
                // 2026-09-25: Sliding window: a key at or before the query with
                // q - k >= sliding_window is masked.
                if(sliding_window>0){
                    if(qr0>=kv_start+c0 && qr0-(kv_start+c0)>=sliding_window) acc_s[nt][0]=-1e30f;
                    if(qr0>=kv_start+c1 && qr0-(kv_start+c1)>=sliding_window) acc_s[nt][1]=-1e30f;
                    if(qr1>=kv_start+c0 && qr1-(kv_start+c0)>=sliding_window) acc_s[nt][2]=-1e30f;
                    if(qr1>=kv_start+c1 && qr1-(kv_start+c1)>=sliding_window) acc_s[nt][3]=-1e30f;
                }
            }

            float rmax0=-1e30f, rmax1=-1e30f;
            #pragma unroll
            for(int nt=0;nt<4;nt++){
                rmax0=fmaxf(rmax0,fmaxf(acc_s[nt][0],acc_s[nt][1]));
                rmax1=fmaxf(rmax1,fmaxf(acc_s[nt][2],acc_s[nt][3]));
            }
            rmax0=fmaxf(rmax0,__shfl_xor_sync(0xFFFFFFFF,rmax0,1));
            rmax0=fmaxf(rmax0,__shfl_xor_sync(0xFFFFFFFF,rmax0,2));
            rmax1=fmaxf(rmax1,__shfl_xor_sync(0xFFFFFFFF,rmax1,1));
            rmax1=fmaxf(rmax1,__shfl_xor_sync(0xFFFFFFFF,rmax1,2));

            float mn0=fmaxf(m_r0,rmax0);
            if (mn0 != m_r0) {
                float eo0=sw_exp(m_r0-mn0); l_r0*=eo0;
                #pragma unroll
                for(int i=0;i<N_TILES_PER_WARP;i++){acc_o[i][0]*=eo0;acc_o[i][1]*=eo0;}
                m_r0=mn0;
            }
            float mn1=fmaxf(m_r1,rmax1);
            if (mn1 != m_r1) {
                float eo1=sw_exp(m_r1-mn1); l_r1*=eo1;
                #pragma unroll
                for(int i=0;i<N_TILES_PER_WARP;i++){acc_o[i][2]*=eo1;acc_o[i][3]*=eo1;}
                m_r1=mn1;
            }

            float sum0=0, sum1=0;
            #pragma unroll
            for(int nt=0;nt<4;nt++){
                float p00=sw_exp(acc_s[nt][0]-m_r0),p01=sw_exp(acc_s[nt][1]-m_r0);
                float p10=sw_exp(acc_s[nt][2]-m_r1),p11=sw_exp(acc_s[nt][3]-m_r1);
                sum0+=p00+p01; sum1+=p10+p11;
                unsigned int c0=nt*8+tid_in_group*2;
#ifdef METRALE_DISABLE_FP16_PV
                smem_P64[row0][c0]=__float2bfloat16_rn(p00); smem_P64[row0][c0+1]=__float2bfloat16_rn(p01);
                smem_P64[row1][c0]=__float2bfloat16_rn(p10); smem_P64[row1][c0+1]=__float2bfloat16_rn(p11);
#else
                smem_P64[row0][c0]=__float2half_rn(p00); smem_P64[row0][c0+1]=__float2half_rn(p01);
                smem_P64[row1][c0]=__float2half_rn(p10); smem_P64[row1][c0+1]=__float2half_rn(p11);
#endif
            }
            sum0+=__shfl_xor_sync(0xFFFFFFFF,sum0,1); sum0+=__shfl_xor_sync(0xFFFFFFFF,sum0,2);
            sum1+=__shfl_xor_sync(0xFFFFFFFF,sum1,1); sum1+=__shfl_xor_sync(0xFFFFFFFF,sum1,2);
            l_r0+=sum0; l_r1+=sum1;

            if(tid_in_group==0){
                smem_ml64[row0][0]=m_r0; smem_ml64[row0][1]=l_r0;
                smem_ml64[row1][0]=m_r1; smem_ml64[row1][1]=l_r1;
            }

            metrale_cp_commit();
        } else {

            LOAD_KV_TILE(V_cache, block_table, smem_V64, kv_start, kv_len, kv_head, tid - 128, 128);
            metrale_cp_commit();
        }


        metrale_cp_wait();
        __syncthreads();

        // 2026-09-25: Warps 4-7 rescale their o to the row max warps 0-3 left in smem_ml64.
        if(warp_id>=4){
            unsigned int r0=pv_warp_m+group_id, r1=r0+8;
            float cm0=smem_ml64[r0][0], cm1=smem_ml64[r1][0];
            if (cm0 != m_r0) {
                float er0=sw_exp(m_r0-cm0);
                #pragma unroll
                for(int i=0;i<N_TILES_PER_WARP;i++){acc_o[i][0]*=er0;acc_o[i][1]*=er0;}
                m_r0=cm0;
            }
            if (cm1 != m_r1) {
                float er1=sw_exp(m_r1-cm1);
                #pragma unroll
                for(int i=0;i<N_TILES_PER_WARP;i++){acc_o[i][2]*=er1;acc_o[i][3]*=er1;}
                m_r1=cm1;
            }
        }


        if(kv_block+1<num_kv_blocks){
            LOAD_KV_TILE(K_cache, block_table, smem_K64[METRALE_KB(1-buf)], (kv_block+1)*BC, kv_len, kv_head, tid, blockDim.x);
            metrale_cp_commit();
        }


        {

            const unsigned short* sP=(const unsigned short*)smem_P64;
            #pragma unroll
            for(unsigned int ks=0;ks<2;ks++){
                unsigned int ko=ks*16;
                unsigned int a0,a1,a2,a3;
#ifdef METRALE_ATTN_LDMATRIX

                { unsigned int pb=__cvta_generic_to_shared(&sP[(pv_warp_m+(lane_id&15))*p_smem_stride64+(lane_id>>4)*8+ko]);
                  asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3},[%4];"
                    :"=r"(a0),"=r"(a1),"=r"(a2),"=r"(a3):"r"(pb)); }
#else
                unsigned int ar0=pv_warp_m+group_id, ar1=ar0+8;
                unsigned int ac0=ko+tid_in_group*2, ac1=ac0+8;
                a0=*(const unsigned int*)&sP[ar0*p_smem_stride64+ac0];
                a1=*(const unsigned int*)&sP[ar1*p_smem_stride64+ac0];
                a2=*(const unsigned int*)&sP[ar0*p_smem_stride64+ac1];
                a3=*(const unsigned int*)&sP[ar1*p_smem_stride64+ac1];
#endif
                #pragma unroll
                for(int nt=0;nt<N_TILES_PER_WARP;nt++){
                    unsigned int nc=(pv_n_start+nt)*8+group_id, k0=ko+tid_in_group*2, k1=k0+8;
#ifdef METRALE_DISABLE_FP16_PV
                    const unsigned short* sV=(const unsigned short*)smem_V64;
                    unsigned int b0=((unsigned int)sV[(k0+1)*HDIM_PAD+nc]<<16)|(unsigned int)sV[k0*HDIM_PAD+nc];
                    unsigned int b1=((unsigned int)sV[(k1+1)*HDIM_PAD+nc]<<16)|(unsigned int)sV[k1*HDIM_PAD+nc];
                    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
                        "{%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%10,%11,%12,%13};"
                        :"=f"(acc_o[nt][0]),"=f"(acc_o[nt][1]),"=f"(acc_o[nt][2]),"=f"(acc_o[nt][3])
                        :"r"(a0),"r"(a1),"r"(a2),"r"(a3),"r"(b0),"r"(b1),
                         "f"(acc_o[nt][0]),"f"(acc_o[nt][1]),"f"(acc_o[nt][2]),"f"(acc_o[nt][3]));
#else
#ifdef METRALE_ATTN_FP8_SMEM
                    unsigned int b0=fp8x2_to_f16x2_bits(smem_V64[k0][nc], smem_V64[k0+1][nc], v_scale);
                    unsigned int b1=fp8x2_to_f16x2_bits(smem_V64[k1][nc], smem_V64[k1+1][nc], v_scale);
#else
                    unsigned int b0=bf16x2_to_f16x2_bits(
                        smem_V64[k0][nc], smem_V64[k0+1][nc]);
                    unsigned int b1=bf16x2_to_f16x2_bits(
                        smem_V64[k1][nc], smem_V64[k1+1][nc]);
#endif
                    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
                        "{%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%10,%11,%12,%13};"
                        :"=f"(acc_o[nt][0]),"=f"(acc_o[nt][1]),"=f"(acc_o[nt][2]),"=f"(acc_o[nt][3])
                        :"r"(a0),"r"(a1),"r"(a2),"r"(a3),"r"(b0),"r"(b1),
                         "f"(acc_o[nt][0]),"f"(acc_o[nt][1]),"f"(acc_o[nt][2]),"f"(acc_o[nt][3]));
#endif
                }
            }
        }

        if(kv_block+1<num_kv_blocks){
            metrale_cp_wait();
        }
        __syncthreads();
    }


    {
        unsigned int r0=pv_warp_m+group_id, r1=r0+8;
        float il0,il1;
        if(warp_id<4){
            il0=(l_r0>0)?(1.f/l_r0):0;
            il1=(l_r1>0)?(1.f/l_r1):0;
        } else {
            float lv0=smem_ml64[r0][1], lv1=smem_ml64[r1][1];
            il0=(lv0>0)?(1.f/lv0):0;
            il1=(lv1>0)?(1.f/lv1):0;
        }

#ifdef PREFILL_BATCHED
        __nv_bfloat16* ob=O+q_batch_off+q_head*head_dim;
#else
        __nv_bfloat16* ob=O+q_head*head_dim;
#endif
        #pragma unroll
        for(int nt=0;nt<N_TILES_PER_WARP;nt++){
            unsigned int c0=(pv_n_start+nt)*8+tid_in_group*2;
            unsigned int gr0=q_start+r0, gr1=q_start+r1;
            if(gr0<q_len&&r0<q_tile_len&&c0<head_dim){
                unsigned int lo=(unsigned int)__bfloat16_as_ushort(__float2bfloat16(acc_o[nt][0]*il0));
                unsigned int hi=(unsigned int)__bfloat16_as_ushort(__float2bfloat16(acc_o[nt][1]*il0));
                *(unsigned int*)&ob[gr0*q_seq_stride+c0]=lo|(hi<<16);
            }
            if(gr1<q_len&&r1<q_tile_len&&c0<head_dim){
                unsigned int lo=(unsigned int)__bfloat16_as_ushort(__float2bfloat16(acc_o[nt][2]*il1));
                unsigned int hi=(unsigned int)__bfloat16_as_ushort(__float2bfloat16(acc_o[nt][3]*il1));
                *(unsigned int*)&ob[gr1*q_seq_stride+c0]=lo|(hi<<16);
            }
        }
    }
}
