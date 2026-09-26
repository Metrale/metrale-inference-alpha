// SPDX-License-Identifier: MIT OR Apache-2.0

// 2026-09-25: KDA (Kimi Delta Attention) recurrent decode: one token, per head h:
//   S[k][v] <- S[k][v] * exp(gate[k])
//   delta[v] = (v[v] - sum_k S[k][v] k[k]) * beta
//   S[k][v] <- S[k][v] + k[k] delta[v]
//   out[v]   = sum_k S[k][v] q[k] * scale
//
// Owner: gb10 kernels.
// Invariants:
// - q, k, v and gate are [H, D] and beta is [H]; the state is FP32 [H, D, D], indexed
//   [k][v] and updated in place; out is FP32 [H, D]. Gate, beta, state and out are FP32 in
//   every entry point.
// - blockIdx.x is the head. The two-pass kernels stride threads over v; the _smem kernel
//   owns VPB columns per block, starting at blockIdx.y * VPB.
// - The caller supplies q and k already L2-normalised (the decode conv,
//   causal_conv1d_update_l2norm, applies it), beta already sigmoided, and
//   scale = 1/sqrt(D).
//
// gate is kda_gate's output, a log-decay, and this kernel applies exp() to it.
// compute_gdn_gates (ssm_preprocess.cu) stores its gate already exponentiated and one per
// head, so a GDN gate passed here would be exponentiated twice, with no error raised.
















































#include <cuda_bf16.h>
#include <math.h>

// 2026-09-25: Dynamic shared memory: 3 * D floats, holding exp(gate), k and q * scale.
#define KDA_REC_BODY(LOAD_QKV)                                                        \
    extern __shared__ float sh[];                                                     \
    const unsigned int h = blockIdx.x;                                                \
    if (h >= H) return;                                                               \
    float* sh_decay = sh;                                                             \
    float* sh_k = sh + D;                                                             \
    float* sh_q = sh + 2u * D;                                                        \
    const size_t hd = (size_t)h * D;                                                  \
    for (unsigned int i = threadIdx.x; i < D; i += blockDim.x) {                       \
        sh_decay[i] = expf(gate[hd + i]);                                             \
        sh_k[i] = LOAD_QKV(k[hd + i]);                                                \
        sh_q[i] = LOAD_QKV(q[hd + i]) * scale;                                        \
    }                                                                                 \
    __syncthreads();                                                                  \
    const float b = beta[h];                                                          \
    float* S = state + hd * D;                                                        \
    for (unsigned int vi = threadIdx.x; vi < D; vi += blockDim.x) {                    \
        float kv = 0.0f;                                                              \
        /* 2026-09-25: Pass 1 decays column vi of S and accumulates kv = sum_k S[k][vi] k[k];\
           pass 2 adds k[k] * delta and accumulates o = sum_k S[k][vi] q[k] * scale.  \
           One thread per column: a warp reads consecutive vi for a fixed kk, so the  \
           state accesses are coalesced. unroll 8 keeps eight independent state loads \
           in flight per thread; it does not reorder the kv and o sums, which run     \
           kk = 0..D-1 in both passes.                                                \
                                                                                      \
                                                                                      \
           */                                                                         \
        _Pragma("unroll 8")                                                           \
        for (unsigned int kk = 0; kk < D; ++kk) {                                      \
            const size_t idx = (size_t)kk * D + vi;                                   \
            const float s = S[idx] * sh_decay[kk];                                    \
            S[idx] = s;                                                               \
            kv += s * sh_k[kk];                                                       \
        }                                                                             \
        const float delta = (LOAD_QKV(v[hd + vi]) - kv) * b;                          \
        float o = 0.0f;                                                               \
        _Pragma("unroll 8")                                                           \
        for (unsigned int kk = 0; kk < D; ++kk) {                                      \
            const size_t idx = (size_t)kk * D + vi;                                   \
            const float s = S[idx] + sh_k[kk] * delta;                                \
            S[idx] = s;                                                               \
            o += s * sh_q[kk];                                                        \
        }                                                                             \
        out[hd + vi] = o;                                                             \
    }

#define KDA_REC_IDENT(x) (x)
#define KDA_REC_BF16(x) __bfloat162float(x)

// 2026-09-25: FP32-input twin, used by the kda_recurrent and kda_chunk microtest examples.
extern "C" __global__ void kda_recurrent_decode_f32(
    const float* __restrict__ q,
    const float* __restrict__ k,
    const float* __restrict__ v,
    const float* __restrict__ gate,
    const float* __restrict__ beta,
    float* __restrict__ state,
    float* __restrict__ out,
    unsigned int H,
    unsigned int D,
    float scale
) {
    KDA_REC_BODY(KDA_REC_IDENT)
}

// 2026-09-25: BF16 q, k and v. glm5next_kda launches this when it does not launch the
// _smem kernel below, with block = min(128, D) and 3 * D floats of shared memory.


extern "C" __global__ void kda_recurrent_decode_bf16(
    const __nv_bfloat16* __restrict__ q,
    const __nv_bfloat16* __restrict__ k,
    const __nv_bfloat16* __restrict__ v,
    const float* __restrict__ gate,
    const float* __restrict__ beta,
    float* __restrict__ state,
    float* __restrict__ out,
    unsigned int H,
    unsigned int D,
    float scale
) {
    KDA_REC_BODY(KDA_REC_BF16)
}

// 2026-09-25: Single-pass-over-global variant: the same expressions in the same order as
// kda_recurrent_decode_bf16 (decay, kv over kk = 0..D-1, delta, update, o over kk = 0..D-1),
// but each thread keeps its decayed column in shared memory between the two passes, so the
// state is read once and written once from global memory.
//
// The v axis has no cross-thread dependency (kv, delta and o are per (h, vi)), so a block
// owns VPB columns and grid.y covers D / VPB. Shared memory: 3 * D + VPB * (D + 1) floats.
// The launcher must make VPB divide D and launch blockDim.x == VPB; otherwise columns are
// dropped with no error. glm5next_kda checks both, uses this kernel only when the target
// has it and the request fits KDA_SMEM_BUDGET, and skips it when
// METRALE_GLM_KDA_NO_SMEM=1.















extern "C" __global__ void kda_recurrent_decode_bf16_smem(
    const __nv_bfloat16* __restrict__ q,
    const __nv_bfloat16* __restrict__ k,
    const __nv_bfloat16* __restrict__ v,
    const float* __restrict__ gate,
    const float* __restrict__ beta,
    float* __restrict__ state,
    float* __restrict__ out,
    unsigned int H,
    unsigned int D,
    float scale,
    unsigned int VPB
) {
    extern __shared__ float sh[];
    const unsigned int h = blockIdx.x;
    if (h >= H) return;
    const unsigned int v0 = blockIdx.y * VPB;
    if (v0 >= D) return;

    float* sh_decay = sh;
    float* sh_k = sh + D;
    float* sh_q = sh + 2u * D;
// 2026-09-25: [VPB, D + 1]: column threadIdx.x of this block's slice, k-major. The +1 pad
// avoids bank conflicts: at a stride of D = 128 floats every thread of a warp would hit
// the same bank (128 % 32 == 0); a stride of D + 1 moves each thread to the next bank.



    float* sh_s = sh + 3u * D;
    const unsigned int col_stride = D + 1u;

    const size_t hd = (size_t)h * D;
    for (unsigned int i = threadIdx.x; i < D; i += blockDim.x) {
        sh_decay[i] = expf(gate[hd + i]);
        sh_k[i] = __bfloat162float(k[hd + i]);
        sh_q[i] = __bfloat162float(q[hd + i]) * scale;
    }
    __syncthreads();

    const float b = beta[h];
    float* S = state + hd * D;
    const unsigned int vi = v0 + threadIdx.x;
    if (threadIdx.x >= VPB || vi >= D) return;
    float* col = sh_s + (size_t)threadIdx.x * col_stride;

    float kv = 0.0f;
    #pragma unroll 8
    for (unsigned int kk = 0; kk < D; ++kk) {
        const float s = S[(size_t)kk * D + vi] * sh_decay[kk];
        col[kk] = s;
        kv += s * sh_k[kk];
    }
    const float delta = (__bfloat162float(v[hd + vi]) - kv) * b;
    float o = 0.0f;
    #pragma unroll 8
    for (unsigned int kk = 0; kk < D; ++kk) {
        const float s = col[kk] + sh_k[kk] * delta;
        S[(size_t)kk * D + vi] = s;
        o += s * sh_q[kk];
    }
    out[hd + vi] = o;
}
