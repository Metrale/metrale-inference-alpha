// SPDX-License-Identifier: MIT OR Apache-2.0

// 2026-09-25: Kimi K3 one-token gated MLA decode (module `mla_decode`). The CPU oracle is
// `mla_decode_token` in crates/model-weights/src/kimi_k3_host/mla.rs; mla_cuda_parity.rs
// checks a CPU copy of this kernel's loop order against it.
// Owner: gb10 kernels (kimi-k3).
// Invariants: none beyond the types. H, dq, dv and T are runtime arguments.
//
//   maybe_rope: q/k rows are [nope | rope]; with use_nope set nothing rotates.
//   sdpa: one query [H, dq] against T cached K/V rows, scale 1/sqrt(dq).
//   output gate: sigmoid(g) * attn when use_gate is set.


#include <math.h>
#include <math_constants.h>

__device__ __forceinline__ float k3_sigmoid(float x) {
    return 1.0f / (1.0f + expf(-x));
}

__device__ void k3_mla_rope_one(
    float* x,
    unsigned int nope,
    unsigned int rope,
    unsigned int pos,
    float theta
) {
    float* r = x + nope;
    for (unsigned int i = 0; i < rope / 2u; ++i) {
        const float freq = (float)pos / powf(theta, 2.0f * (float)i / (float)rope);
        float s, c;
        sincosf(freq, &s, &c);
        const float a = r[i];
        const float b = r[i + rope / 2u];
        r[i] = a * c - b * s;
        r[i + rope / 2u] = a * s + b * c;
    }
}

// 2026-09-25: One thread per head. With use_nope set, or rope == 0, q and k are unchanged.
extern "C" __global__ void k3_mla_maybe_rope_f32(
    float* __restrict__ q,            // 2026-09-25: [H, nope+rope]
    float* __restrict__ k,            // 2026-09-25: [H, nope+rope]
    unsigned int H,
    unsigned int nope,
    unsigned int rope,
    unsigned int pos,
    float theta,
    unsigned int use_nope
) {
    if (use_nope != 0u || rope == 0u) {
        return;
    }
    const unsigned int h = blockIdx.x * blockDim.x + threadIdx.x;
    if (h >= H) {
        return;
    }
    const unsigned int dq = nope + rope;
    k3_mla_rope_one(q + h * dq, nope, rope, pos, theta);
    k3_mla_rope_one(k + h * dq, nope, rope, pos, theta);
}

// 2026-09-25: One block per head; thread 0 computes in sdpa_one + apply_output_gate order.
extern "C" __global__ void k3_mla_sdpa_gate_f32(
    const float* __restrict__ q,      // 2026-09-25: [H, dq]
    const float* __restrict__ k,      // 2026-09-25: [T, H, dq]
    const float* __restrict__ v,      // 2026-09-25: [T, H, dv]
    const float* __restrict__ g,      // 2026-09-25: [H, dv]
    float* __restrict__ out,          // 2026-09-25: [H, dv]
    unsigned int T,
    unsigned int H,
    unsigned int dq,
    unsigned int dv,
    unsigned int use_gate
) {
    const unsigned int h = blockIdx.x;
    if (h >= H || threadIdx.x != 0u || dq == 0u) {
        return;
    }
    const float scale = 1.0f / sqrtf((float)dq);
    const float* qrow = q + h * dq;




    float m = -CUDART_INF_F;
    for (unsigned int kj = 0; kj < T; ++kj) {
        const float* krow = k + (kj * H + h) * dq;
        float s = 0.0f;
        for (unsigned int d = 0; d < dq; ++d) {
            s += qrow[d] * krow[d];
        }
        s *= scale;
        if (s > m) {
            m = s;
        }
    }
    float z = 0.0f;
    for (unsigned int kj = 0; kj < T; ++kj) {
        const float* krow = k + (kj * H + h) * dq;
        float s = 0.0f;
        for (unsigned int d = 0; d < dq; ++d) {
            s += qrow[d] * krow[d];
        }
        z += expf(s * scale - m);
    }
    for (unsigned int d = 0; d < dv; ++d) {
        float o = 0.0f;
        for (unsigned int kj = 0; kj < T; ++kj) {
            const float* krow = k + (kj * H + h) * dq;
            float s = 0.0f;
            for (unsigned int i = 0; i < dq; ++i) {
                s += qrow[i] * krow[i];
            }
            const float a = expf(s * scale - m) / z;
            o += a * v[(kj * H + h) * dv + d];
        }
        if (use_gate != 0u) {
            o *= k3_sigmoid(g[h * dv + d]);
        }
        out[h * dv + d] = o;
    }
}
