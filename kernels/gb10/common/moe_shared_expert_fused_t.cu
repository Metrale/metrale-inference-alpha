// SPDX-License-Identifier: MIT OR Apache-2.0

// 2026-09-25: Single-token NVFP4/MXFP4 MoE GEMVs on the transposed weight layout, with the
// shared expert as blockIdx.y == top_k: gate+up, then SiLU(gate) * up and down.
//
// Owner: gb10 kernels.
// Invariants:
// - Weights are [K / 2, N] bytes (low nibble first) with [K / GS, N] scale bytes; K must
//   be a multiple of GS. Each thread computes one output n, so adjacent lanes read
//   adjacent bytes and no cross-thread reduction is needed.
// - The _t entry points use NVFP4 for routed and shared experts: GS = 16 and an FP8 E4M3
//   scale times scale2. The _t_e8m0 entry points use GS = 32 and an E8M0 scale for the
//   routed experts and keep NVFP4 for the shared expert; the host picks them when the
//   routed experts are tagged Mxfp4E8m0 (MoeLayer::e8m0_or).
// - A null routed or shared weight pointer writes zeros.
// - Launch: 32 threads (BLOCK_SIZE), grid (ceil(N / 32), top_k + 1, 2 for gate+up or 1
//   for down); down takes K * 4 bytes of dynamic shared memory.




#include <cuda_bf16.h>
#include <cuda_fp8.h>

// 2026-09-25: metrale_dec_e4m3 and mx_block_scale come from this header, which the
// deepseek-v4-flash moe_w4a16_grouped_gemm.cu includes as well.


#include "mx_block_scale.cuh"





#define BLOCK_SIZE 32
#define GROUP_SIZE 16

__device__ __constant__ float E2M1_LUT_T[16] = {
    0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f,
    -0.0f, -0.5f, -1.0f, -1.5f, -2.0f, -3.0f, -4.0f, -6.0f
};



// 2026-09-25: Gate+up body. blockIdx.z selects gate (0) or up (1). Routed experts use the
// <GS_R, E8M0_R> scale format and the shared expert <GS_S, E8M0_S>; a block serves one
// expert slot, so the choice is uniform within the block.










template<int GS_R, bool E8M0_R, int GS_S, bool E8M0_S>
__device__ __forceinline__ void gate_up_shared_t_impl(
    const __nv_bfloat16* __restrict__ A,
    const unsigned long long* __restrict__ gate_packed_t_ptrs,
    const unsigned long long* __restrict__ gate_scale_t_ptrs,
    const float* __restrict__ gate_scale2_vals,
    __nv_bfloat16* __restrict__ gate_out,
    const unsigned long long* __restrict__ up_packed_t_ptrs,
    const unsigned long long* __restrict__ up_scale_t_ptrs,
    const float* __restrict__ up_scale2_vals,
    __nv_bfloat16* __restrict__ up_out,
    const unsigned int* __restrict__ expert_indices,

    const unsigned char* __restrict__ sh_gate_t_packed,
    const unsigned char* __restrict__ sh_gate_t_scale,
    float sh_gate_s2,
    __nv_bfloat16* __restrict__ sh_gate_out,
    const unsigned char* __restrict__ sh_up_t_packed,
    const unsigned char* __restrict__ sh_up_t_scale,
    float sh_up_s2,
    __nv_bfloat16* __restrict__ sh_up_out,
    unsigned int N, unsigned int K, unsigned int top_k
) {
    const unsigned int expert_slot = blockIdx.y;
    const unsigned int proj = blockIdx.z;
    const bool is_shared = (expert_slot == top_k);

    const unsigned char* B_packed;
    const unsigned char* B_scale;
    float s2;
    __nv_bfloat16* C;

    if (is_shared) {
        if (proj == 0) {
            if (sh_gate_t_packed == 0) {
                const unsigned int n = blockIdx.x * BLOCK_SIZE + threadIdx.x;
                if (n < N) sh_gate_out[n] = __float2bfloat16(0.0f);
                return;
            }
            B_packed = sh_gate_t_packed;
            B_scale = sh_gate_t_scale;
            s2 = sh_gate_s2;
            C = sh_gate_out;
        } else {
            if (sh_up_t_packed == 0) {
                const unsigned int n = blockIdx.x * BLOCK_SIZE + threadIdx.x;
                if (n < N) sh_up_out[n] = __float2bfloat16(0.0f);
                return;
            }
            B_packed = sh_up_t_packed;
            B_scale = sh_up_t_scale;
            s2 = sh_up_s2;
            C = sh_up_out;
        }
    } else {
        const unsigned int expert_id = expert_indices[expert_slot];
        if (proj == 0) {
            B_packed = (const unsigned char*)gate_packed_t_ptrs[expert_id];
            B_scale = (const unsigned char*)gate_scale_t_ptrs[expert_id];
            s2 = gate_scale2_vals[expert_id];
            C = gate_out;
        } else {
            B_packed = (const unsigned char*)up_packed_t_ptrs[expert_id];
            B_scale = (const unsigned char*)up_scale_t_ptrs[expert_id];
            s2 = up_scale2_vals[expert_id];
            C = up_out;
        }
        if (B_packed == 0) {
            const unsigned int n = blockIdx.x * BLOCK_SIZE + threadIdx.x;
            if (n < N) C[(unsigned long long)expert_slot * N + n] = __float2bfloat16(0.0f);
            return;
        }
    }

    const unsigned int n = blockIdx.x * BLOCK_SIZE + threadIdx.x;
    const bool valid = (n < N);

    __shared__ float s_lut[16];
    if (threadIdx.x < 16) s_lut[threadIdx.x] = E2M1_LUT_T[threadIdx.x];
    __syncthreads();

    if (!valid) return;

    // 2026-09-25: C[n] = sum over k of A[k] * W[k, n], one scale group (GS inputs) per outer
    // iteration.






    float acc = 0.0f;
    #define GATEUP_ACCUM(GS_, E8M0_) do { \
        const unsigned int num_groups = K / (GS_); \
        for (unsigned int sg = 0; sg < num_groups; sg++) { \
            unsigned char sb = B_scale[(unsigned long long)sg * N + n]; \
            float sc = mx_block_scale<(E8M0_)>(sb, s2); \
            const unsigned int kh_base = sg * ((GS_) / 2); \
            _Pragma("unroll") \
            for (unsigned int kh_off = 0; kh_off < ((GS_) / 2); kh_off++) { \
                unsigned int k_half = kh_base + kh_off; \
                unsigned char byte = B_packed[(unsigned long long)k_half * N + n]; \
                float a_lo = __bfloat162float(A[k_half * 2]); \
                float a_hi = __bfloat162float(A[k_half * 2 + 1]); \
                float w_lo = s_lut[byte & 0xFu] * sc; \
                float w_hi = s_lut[(byte >> 4) & 0xFu] * sc; \
                acc += a_lo * w_lo + a_hi * w_hi; \
            } \
        } \
    } while(0)
    // 2026-09-25: With equal routed and shared formats one loop is compiled; otherwise the
    // block-uniform is_shared test picks the loop.

    if constexpr (GS_R == GS_S && E8M0_R == E8M0_S) {
        GATEUP_ACCUM(GS_R, E8M0_R);
    } else {
        if (is_shared) { GATEUP_ACCUM(GS_S, E8M0_S); }
        else           { GATEUP_ACCUM(GS_R, E8M0_R); }
    }
    #undef GATEUP_ACCUM

    if (is_shared) {
        C[n] = __float2bfloat16(acc);
    } else {
        C[(unsigned long long)expert_slot * N + n] = __float2bfloat16(acc);
    }
}

// 2026-09-25: NVFP4 for routed and shared experts: an FP8 E4M3 scale per 16 inputs times scale2.
extern "C" __global__ void moe_expert_gate_up_shared_t(
    const __nv_bfloat16* __restrict__ A,
    const unsigned long long* __restrict__ gate_packed_t_ptrs,
    const unsigned long long* __restrict__ gate_scale_t_ptrs,
    const float* __restrict__ gate_scale2_vals,
    __nv_bfloat16* __restrict__ gate_out,
    const unsigned long long* __restrict__ up_packed_t_ptrs,
    const unsigned long long* __restrict__ up_scale_t_ptrs,
    const float* __restrict__ up_scale2_vals,
    __nv_bfloat16* __restrict__ up_out,
    const unsigned int* __restrict__ expert_indices,
    const unsigned char* __restrict__ sh_gate_t_packed,
    const unsigned char* __restrict__ sh_gate_t_scale,
    float sh_gate_s2,
    __nv_bfloat16* __restrict__ sh_gate_out,
    const unsigned char* __restrict__ sh_up_t_packed,
    const unsigned char* __restrict__ sh_up_t_scale,
    float sh_up_s2,
    __nv_bfloat16* __restrict__ sh_up_out,
    unsigned int N, unsigned int K, unsigned int top_k
) {
    gate_up_shared_t_impl<GROUP_SIZE, false, GROUP_SIZE, false>(
        A, gate_packed_t_ptrs, gate_scale_t_ptrs, gate_scale2_vals, gate_out,
        up_packed_t_ptrs, up_scale_t_ptrs, up_scale2_vals, up_out, expert_indices,
        sh_gate_t_packed, sh_gate_t_scale, sh_gate_s2, sh_gate_out,
        sh_up_t_packed, sh_up_t_scale, sh_up_s2, sh_up_out, N, K, top_k);
}

// 2026-09-25: Routed experts are MXFP4 (an E8M0 scale per 32 inputs; scale2 is not used);
// the shared expert is NVFP4.


extern "C" __global__ void moe_expert_gate_up_shared_t_e8m0(
    const __nv_bfloat16* __restrict__ A,
    const unsigned long long* __restrict__ gate_packed_t_ptrs,
    const unsigned long long* __restrict__ gate_scale_t_ptrs,
    const float* __restrict__ gate_scale2_vals,
    __nv_bfloat16* __restrict__ gate_out,
    const unsigned long long* __restrict__ up_packed_t_ptrs,
    const unsigned long long* __restrict__ up_scale_t_ptrs,
    const float* __restrict__ up_scale2_vals,
    __nv_bfloat16* __restrict__ up_out,
    const unsigned int* __restrict__ expert_indices,
    const unsigned char* __restrict__ sh_gate_t_packed,
    const unsigned char* __restrict__ sh_gate_t_scale,
    float sh_gate_s2,
    __nv_bfloat16* __restrict__ sh_gate_out,
    const unsigned char* __restrict__ sh_up_t_packed,
    const unsigned char* __restrict__ sh_up_t_scale,
    float sh_up_s2,
    __nv_bfloat16* __restrict__ sh_up_out,
    unsigned int N, unsigned int K, unsigned int top_k
) {
    gate_up_shared_t_impl<32, true, GROUP_SIZE, false>(
        A, gate_packed_t_ptrs, gate_scale_t_ptrs, gate_scale2_vals, gate_out,
        up_packed_t_ptrs, up_scale_t_ptrs, up_scale2_vals, up_out, expert_indices,
        sh_gate_t_packed, sh_gate_t_scale, sh_gate_s2, sh_gate_out,
        sh_up_t_packed, sh_up_t_scale, sh_up_s2, sh_up_out, N, K, top_k);
}

// 2026-09-25: SiLU(gate) * up, then down. Routed slot s reads row s of gate_out/up_out
// ([top_k, K]) and writes row s of C ([top_k, N]); the shared expert reads sh_gate_in and
// sh_up_in and writes sh_down_out ([N]). Scale formats as in gate_up_shared_t_impl.





template<int GS_R, bool E8M0_R, int GS_S, bool E8M0_S>
__device__ __forceinline__ void silu_down_shared_t_impl(
    const __nv_bfloat16* __restrict__ gate_out,
    const __nv_bfloat16* __restrict__ up_out,
    const unsigned long long* __restrict__ packed_t_ptrs,
    const unsigned long long* __restrict__ scale_t_ptrs,
    const float* __restrict__ scale2_vals,
    __nv_bfloat16* __restrict__ C,
    const unsigned int* __restrict__ expert_indices,

    const __nv_bfloat16* __restrict__ sh_gate_in,
    const __nv_bfloat16* __restrict__ sh_up_in,
    const unsigned char* __restrict__ sh_down_t_packed,
    const unsigned char* __restrict__ sh_down_t_scale,
    float sh_down_s2,
    __nv_bfloat16* __restrict__ sh_down_out,
    unsigned int N, unsigned int K, unsigned int top_k
) {
    const unsigned int expert_slot = blockIdx.y;
    const bool is_shared = (expert_slot == top_k);

    const unsigned char* B_packed;
    const unsigned char* B_scale;
    float s2;
    const __nv_bfloat16* g_ptr;
    const __nv_bfloat16* u_ptr;
    if (is_shared) {
        if (sh_down_t_packed == 0) {

            const unsigned int n = blockIdx.x * BLOCK_SIZE + threadIdx.x;
            if (n < N) sh_down_out[n] = __float2bfloat16(0.0f);
            return;
        }
        B_packed = sh_down_t_packed;
        B_scale = sh_down_t_scale;
        s2 = sh_down_s2;
        g_ptr = sh_gate_in;
        u_ptr = sh_up_in;
    } else {
        const unsigned int expert_id = expert_indices[expert_slot];
        B_packed = (const unsigned char*)packed_t_ptrs[expert_id];
        B_scale = (const unsigned char*)scale_t_ptrs[expert_id];
        s2 = scale2_vals[expert_id];
        g_ptr = gate_out + (unsigned long long)expert_slot * K;
        u_ptr = up_out + (unsigned long long)expert_slot * K;

        if (B_packed == 0) {
            const unsigned int n = blockIdx.x * BLOCK_SIZE + threadIdx.x;
            if (n < N) C[expert_slot * N + n] = __float2bfloat16(0.0f);
            return;
        }
    }

    const unsigned int n = blockIdx.x * BLOCK_SIZE + threadIdx.x;
    const bool valid = (n < N);


    extern __shared__ float s_act[];
    for (unsigned int i = threadIdx.x; i < K; i += BLOCK_SIZE) {
        float gf = __bfloat162float(g_ptr[i]);
        float uf = __bfloat162float(u_ptr[i]);
        s_act[i] = (gf / (1.0f + __expf(-gf))) * uf;
    }

    __shared__ float s_lut[16];
    if (threadIdx.x < 16) s_lut[threadIdx.x] = E2M1_LUT_T[threadIdx.x];
    __syncthreads();

    if (!valid) return;





    const unsigned int K_half = K / 2;
    float acc = 0.0f;



    #define SILUDOWN_ACCUM(GS_, E8M0_) do { \
        const unsigned int num_groups = K / (GS_); \
        for (unsigned int sg = 0; sg < num_groups; sg++) { \
            unsigned char sb = B_scale[(unsigned long long)sg * N + n]; \
            float sc = mx_block_scale<(E8M0_)>(sb, s2); \
            const unsigned int kh_base = sg * ((GS_) / 2); \
            _Pragma("unroll") \
            for (unsigned int kh_off = 0; kh_off < ((GS_) / 2); kh_off++) { \
                unsigned int k_half = kh_base + kh_off; \
                unsigned char byte = B_packed[(unsigned long long)k_half * N + n]; \
                unsigned int nibble_lo = byte & 0xFu; \
                unsigned int nibble_hi = (byte >> 4) & 0xFu; \
                float w_lo = s_lut[nibble_lo] * sc; \
                float w_hi = s_lut[nibble_hi] * sc; \
                float a_lo = s_act[k_half * 2]; \
                float a_hi = s_act[k_half * 2 + 1]; \
                acc += a_lo * w_lo + a_hi * w_hi; \
            } \
            if (kh_base + ((GS_) / 2) > K_half) break; \
        } \
    } while(0)
    if constexpr (GS_R == GS_S && E8M0_R == E8M0_S) {
        SILUDOWN_ACCUM(GS_R, E8M0_R);
    } else {
        if (is_shared) { SILUDOWN_ACCUM(GS_S, E8M0_S); }
        else           { SILUDOWN_ACCUM(GS_R, E8M0_R); }
    }
    #undef SILUDOWN_ACCUM


    if (is_shared) {
        sh_down_out[n] = __float2bfloat16(acc);
    } else {
        C[(unsigned long long)expert_slot * N + n] = __float2bfloat16(acc);
    }
}

// 2026-09-25: NVFP4 for routed and shared experts: an FP8 E4M3 scale per 16 inputs times scale2.
extern "C" __global__ void moe_expert_silu_down_shared_t(
    const __nv_bfloat16* __restrict__ gate_out,
    const __nv_bfloat16* __restrict__ up_out,
    const unsigned long long* __restrict__ packed_t_ptrs,
    const unsigned long long* __restrict__ scale_t_ptrs,
    const float* __restrict__ scale2_vals,
    __nv_bfloat16* __restrict__ C,
    const unsigned int* __restrict__ expert_indices,
    const __nv_bfloat16* __restrict__ sh_gate_in,
    const __nv_bfloat16* __restrict__ sh_up_in,
    const unsigned char* __restrict__ sh_down_t_packed,
    const unsigned char* __restrict__ sh_down_t_scale,
    float sh_down_s2,
    __nv_bfloat16* __restrict__ sh_down_out,
    unsigned int N, unsigned int K, unsigned int top_k
) {
    silu_down_shared_t_impl<GROUP_SIZE, false, GROUP_SIZE, false>(
        gate_out, up_out, packed_t_ptrs, scale_t_ptrs, scale2_vals, C,
        expert_indices, sh_gate_in, sh_up_in, sh_down_t_packed, sh_down_t_scale,
        sh_down_s2, sh_down_out, N, K, top_k);
}

// 2026-09-25: Routed experts are MXFP4 (an E8M0 scale per 32 inputs); the shared expert is NVFP4.
extern "C" __global__ void moe_expert_silu_down_shared_t_e8m0(
    const __nv_bfloat16* __restrict__ gate_out,
    const __nv_bfloat16* __restrict__ up_out,
    const unsigned long long* __restrict__ packed_t_ptrs,
    const unsigned long long* __restrict__ scale_t_ptrs,
    const float* __restrict__ scale2_vals,
    __nv_bfloat16* __restrict__ C,
    const unsigned int* __restrict__ expert_indices,
    const __nv_bfloat16* __restrict__ sh_gate_in,
    const __nv_bfloat16* __restrict__ sh_up_in,
    const unsigned char* __restrict__ sh_down_t_packed,
    const unsigned char* __restrict__ sh_down_t_scale,
    float sh_down_s2,
    __nv_bfloat16* __restrict__ sh_down_out,
    unsigned int N, unsigned int K, unsigned int top_k
) {
    silu_down_shared_t_impl<32, true, GROUP_SIZE, false>(
        gate_out, up_out, packed_t_ptrs, scale_t_ptrs, scale2_vals, C,
        expert_indices, sh_gate_in, sh_up_in, sh_down_t_packed, sh_down_t_scale,
        sh_down_s2, sh_down_out, N, K, top_k);
}
