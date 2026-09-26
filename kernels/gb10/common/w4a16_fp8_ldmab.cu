// SPDX-License-Identifier: MIT OR Apache-2.0

// 2026-09-25: C[M, N] = A[M, K] * B[N, K]^T with A and B FP8 E4M3 bytes, FP32 accumulation
// (mma m16n8k32 e4m3) and a BF16 C; no scale is applied.
//
// Owner: gb10 kernels.
// Invariants: none beyond the types.
//
// ops::fp8_gemm_n128 (model-layers gemm_fp8_prefill.rs) launches it when K is a multiple of
// 32 and METRALE_FP8_LDMAB is not 0, after casting A to E4M3 with bf16_to_fp8; B is the
// pre-dequantized E4M3 weight. Grid (ceil(N/128), ceil(M/128)), block 256: eight warps of
// 16 rows, a 128x128 tile per CTA, K steps of 32 double-buffered through cp.async, and
// ldmatrix.x4 fragments for both A and B.
//
// It is a module of its own (w4a16_fp8_ldmab), not part of w4a16_gemm.cu, because a model's
// own w4a16_gemm.cu replaces the common one whole (crates/kernels/build.rs,
// shadowed_dropped_pairs), which would drop this kernel for that model.




#include <cuda_bf16.h>
#include <cuda_fp8.h>


__device__ __forceinline__ void cp_async_pred_16(void* dst_smem, const void* src_gmem, bool pred) {
    unsigned int dst = __cvta_generic_to_shared(dst_smem);
    unsigned int src_bytes = pred ? 16 : 0;
    asm volatile("cp.async.ca.shared.global [%0], [%1], 16, %2;"
                 :: "r"(dst), "l"(src_gmem), "r"(src_bytes));
}

__device__ __forceinline__ void cp_async_commit() {
    asm volatile("cp.async.commit_group;");
}

__device__ __forceinline__ void cp_async_wait_all() {
    asm volatile("cp.async.wait_group 0;");
}






#define METRALE_MMA_E4M3F(d, a0,a1,a2,a3, b0,b1) \
    asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 " \
        "{%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%10,%11,%12,%13};" \
        : "=f"((d)[0]),"=f"((d)[1]),"=f"((d)[2]),"=f"((d)[3]) \
        : "r"(a0),"r"(a1),"r"(a2),"r"(a3),"r"(b0),"r"(b1), \
          "f"((d)[0]),"f"((d)[1]),"f"((d)[2]),"f"((d)[3]))

extern "C" __global__
__launch_bounds__(256, 2)
void fp8_fp8_gemm_ldmab(
    const unsigned char* __restrict__ A_fp8,
    const unsigned char* __restrict__ B_fp8,
    __nv_bfloat16* __restrict__ C,
    unsigned int M, unsigned int N, unsigned int K
) {
    const unsigned int cta_m = blockIdx.y * 128;
    const unsigned int cta_n = blockIdx.x * 128;
    if (cta_m >= M) return;
    const unsigned int t = threadIdx.x;
    const unsigned int warp_id = t >> 5;
    const unsigned int lane = t & 31;
    const unsigned int group_id = lane >> 2;
    const unsigned int t4 = lane & 3;
    const unsigned int wrow = warp_id * 16;

    __shared__ unsigned char smem_Ai[2][128][32];
    __shared__ unsigned char smem_Bi[2][128][32];

    float acc[16][4];
    #pragma unroll
    for (int i = 0; i < 16; i++) { acc[i][0]=0.f; acc[i][1]=0.f; acc[i][2]=0.f; acc[i][3]=0.f; }

    #define LABF_LOADS(buf, kb) do { \
        { unsigned ar = t >> 1; unsigned ac = (t & 1) << 4; unsigned gc = (kb) + ac; unsigned gr = cta_m + ar; \
          cp_async_pred_16(&smem_Ai[(buf)][ar][ac], &A_fp8[(unsigned long long)gr*K+gc], (gr<M)&&(gc+15<K)); } \
        { unsigned an = t >> 1; unsigned ac = (t & 1) << 4; unsigned gc = (kb) + ac; unsigned gn = cta_n + an; \
          cp_async_pred_16(&smem_Bi[(buf)][an][ac], &B_fp8[(unsigned long long)gn*K+gc], (gn<N)&&(gc+15<K)); } \
    } while(0)

    #define LABF_COMPUTE(buf) do { \
        unsigned a0,a1,a2,a3; \
        const int* xs = (const int*)&smem_Ai[(buf)][wrow][0] + (lane % 16)*8 + (lane / 16)*4; \
        asm volatile("ldmatrix.sync.aligned.m8n8.x4.b16 {%0,%1,%2,%3},[%4];" \
            : "=r"(a0),"=r"(a1),"=r"(a2),"=r"(a3) : "l"(xs)); \
        _Pragma("unroll") for (int p = 0; p < 8; p++) { \
            unsigned nt0 = 2*p, nt1 = 2*p+1; \
            unsigned brow = ((lane<16)?nt0:nt1)*8 + (lane&7); \
            const void* bxs = &smem_Bi[(buf)][brow][((lane>>3)&1)*16]; \
            unsigned q0,q1,q2,q3; \
            asm volatile("ldmatrix.sync.aligned.m8n8.x4.b16 {%0,%1,%2,%3},[%4];" \
                : "=r"(q0),"=r"(q1),"=r"(q2),"=r"(q3) : "l"(bxs)); \
            METRALE_MMA_E4M3F(acc[nt0], a0,a1,a2,a3, q0,q1); \
            METRALE_MMA_E4M3F(acc[nt1], a0,a1,a2,a3, q2,q3); \
        } \
    } while(0)

    LABF_LOADS(0, 0); cp_async_commit(); cp_async_wait_all(); __syncthreads();
    int cur = 0;
    for (unsigned int kb = 32; kb < K; kb += 32) {
        int nxt = 1 - cur;
        LABF_LOADS(nxt, kb); cp_async_commit();
        LABF_COMPUTE(cur);
        cp_async_wait_all(); __syncthreads();
        cur = nxt;
    }
    LABF_COMPUTE(cur);
    #undef LABF_LOADS
    #undef LABF_COMPUTE

    #pragma unroll
    for (int nt = 0; nt < 16; nt++) {
        unsigned c0 = cta_n + nt*8 + t4*2, c1 = c0 + 1;
        unsigned r0 = cta_m + wrow + group_id, r1 = r0 + 8;
        if (r0<M&&c0<N) C[r0*N+c0]=__float2bfloat16(acc[nt][0]);
        if (r0<M&&c1<N) C[r0*N+c1]=__float2bfloat16(acc[nt][1]);
        if (r1<M&&c0<N) C[r1*N+c0]=__float2bfloat16(acc[nt][2]);
        if (r1<M&&c1<N) C[r1*N+c1]=__float2bfloat16(acc[nt][3]);
    }
}
#undef METRALE_MMA_E4M3F
