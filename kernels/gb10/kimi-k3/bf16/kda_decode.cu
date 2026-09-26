// SPDX-License-Identifier: MIT OR Apache-2.0

// 2026-09-25: Kimi K3 one-token KDA decode (module `kda_decode`): the short conv update,
// then the recurrent step. The CPU oracle is `kda_decode_token` in
// crates/model-weights/src/kimi_k3_host/kda.rs; kda_cuda_parity.rs checks a CPU copy of this
// kernel's loop order against it.
// Owner: gb10 kernels (kimi-k3).
// Invariants: none beyond the types. H, D and K are runtime arguments.
//
//   conv: state [C, K] shifts left, x enters the last slot, y = SiLU(dot(w, state)).
//   recurrent: q, k L2-normalised, v raw; S *= exp(gate) along the key axis;
//   delta = (v - S^T k) * sigmoid(beta); S += k (x) delta; o = S^T q / sqrt(D).

#include <math.h>

__device__ __forceinline__ float k3_sigmoid(float x) {
    return 1.0f / (1.0f + expf(-x));
}

// 2026-09-25: One thread per conv channel; state is [C, K] row-major and slot 0 is dropped.
extern "C" __global__ void k3_kda_conv_update_f32(
    const float* __restrict__ x,      // 2026-09-25: [C]
    const float* __restrict__ w,      // 2026-09-25: [C, K]
    float* __restrict__ state,        // 2026-09-25: [C, K] rmw
    float* __restrict__ y,            // 2026-09-25: [C]
    unsigned int C,
    unsigned int K
) {
    const unsigned int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= C || K == 0u) {
        return;
    }
    const unsigned int row = c * K;
    for (unsigned int k = 0; k + 1u < K; ++k) {
        state[row + k] = state[row + k + 1u];
    }
    state[row + (K - 1u)] = x[c];
    float acc = 0.0f;
    for (unsigned int k = 0; k < K; ++k) {
        acc += w[row + k] * state[row + k];
    }
    y[c] = acc * k3_sigmoid(acc);
}

__device__ void k3_l2_row(const float* in, float* out, unsigned int D, float eps) {
    float ss = eps;
    for (unsigned int i = 0; i < D; ++i) {
        const float v = in[i];
        ss += v * v;
    }
    // 2026-09-25: 1/sqrtf rather than rsqrtf, to match the oracle's `1.0 / (sum + eps).sqrt()`.
    const float inv = 1.0f / sqrtf(ss);
    for (unsigned int i = 0; i < D; ++i) {
        out[i] = in[i] * inv;
    }
}

// 2026-09-25: One block per head, shared 3*D floats. Threads stride the V axis and walk K in
// order, as the oracle does. qkv is post-conv [q | k | v]; gate is the log decay; beta a logit.
extern "C" __global__ void k3_kda_recurrent_step_f32(
    const float* __restrict__ qkv,    // 2026-09-25: [3 * H * D]
    const float* __restrict__ gate,   // 2026-09-25: [H * D] log-decay
    const float* __restrict__ beta,   // 2026-09-25: [H] logit
    float* __restrict__ state,        // 2026-09-25: [H, D, D] K-major rmw
    float* __restrict__ out,          // 2026-09-25: [H * D]
    unsigned int H,
    unsigned int D,
    float l2_eps
) {
    const unsigned int h = blockIdx.x;
    if (h >= H || D == 0u) {
        return;
    }
    extern __shared__ float sh[];
    float* sh_q = sh;
    float* sh_k = sh + D;
    float* sh_decay = sh + 2u * D;

    const unsigned int qkv_dim = H * D;
    const unsigned int base = h * D;
    if (threadIdx.x == 0u) {
        k3_l2_row(qkv + base, sh_q, D, l2_eps);
        k3_l2_row(qkv + qkv_dim + base, sh_k, D, l2_eps);
        for (unsigned int i = 0; i < D; ++i) {
            sh_decay[i] = expf(gate[base + i]);
        }
    }
    __syncthreads();

    const float b = k3_sigmoid(beta[h]);
    const float scale = 1.0f / sqrtf((float)D);
    const float* v = qkv + 2u * qkv_dim + base;
    float* S = state + (size_t)h * (size_t)D * (size_t)D;

    for (unsigned int vi = threadIdx.x; vi < D; vi += blockDim.x) {
        float kv = 0.0f;
        for (unsigned int kk = 0; kk < D; ++kk) {
            const size_t idx = (size_t)kk * D + vi;
            const float s = S[idx] * sh_decay[kk];
            S[idx] = s;
            kv += s * sh_k[kk];
        }
        const float delta = (v[vi] - kv) * b;
        float o = 0.0f;
        for (unsigned int kk = 0; kk < D; ++kk) {
            const size_t idx = (size_t)kk * D + vi;
            const float s = S[idx] + sh_k[kk] * delta;
            S[idx] = s;
            o += s * sh_q[kk] * scale;
        }
        out[base + vi] = o;
    }
}
