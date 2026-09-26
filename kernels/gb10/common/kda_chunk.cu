// SPDX-License-Identifier: MIT OR Apache-2.0

// 2026-09-25: Chunked prefill for KDA (Kimi Delta Attention): kda_chunk_prepare works on
// each chunk independently, and kda_chunk_scan carries the recurrent state across chunks.
//
// Owner: gb10 kernels.
// Invariants:
// - kda_chunk_prepare runs one block per (chunk, head): blockIdx.x is the chunk and
//   blockIdx.y the head. kda_chunk_scan runs one block per head and walks the chunks in
//   order; the state update of chunk c finishes before chunk c + 1 reads the state.
// - Every loop strides by blockDim.x, so any block size is correct; the host launches
//   128 threads (`BLOCK` in crates/model-arch/src/glm5next_kda/mod.rs).
// - q, k, v, gate, gc, u, w and out are FP32 [T_pad, H, D], position-major; beta is FP32
//   [T_pad, H]; the state is FP32 [H, D, D], indexed [k][v].
// - A position t >= T contributes nothing to the state or to real rows: prepare reads
//   k, v, gate and beta there as zero whatever the buffer holds, the scan skips it in the
//   state update, and its out row is not written.
//
// What the two kernels compute, per head, over a chunk of C positions (i the query row,
// j the key column, d a channel, S the state):
//   gc[i,d]    = sum_{p<=i} gate[p,d]                                (gate is a log-decay)
//   A[i,j]     = -sum_d beta[i] k[i,d] k[j,d] exp(gc[i,d] - gc[j,d])  for j < i, else 0
//   for i = 1..C-1: A[i,:i] += A[i,:i] @ A[:i,:i];  then A[i,i] = 1
//   u[i,d]     = sum_j A[i,j] v[j,d] beta[j]
//   w[i,d]     = sum_j A[i,j] k[j,d] beta[j] exp(gc[j,d])
//   v_new      = u - w @ S
//   intra[i,j] = sum_d scale q[i,d] k[j,d] exp(gc[i,d] - gc[j,d])     for j <= i
//   out[i,v]   = sum_k scale q[i,k] exp(gc[i,k]) S[k,v] + sum_{j<=i} intra[i,j] v_new[j,v]
//   S[k,v]    <- S[k,v] exp(gc[C-1,k]) + sum_i k[i,k] exp(gc[C-1,k] - gc[i,k]) v_new[i,v]
// A leaves out the diagonal before the substitution; intra includes it. The host passes
// scale = 1/sqrt(D).
//
// Why a pad position is harmless: with gate = 0 the cumulative decay stops, so gc[C-1]
// is the last real value; with k = beta = 0 its row of A is zero apart from the unit
// diagonal, so u, w and v_new are zero there.
//
// Shared memory in bytes: prepare (C*D + C*C + C) * 4, scan (2*C*D + C*C) * 4. The host
// sizes both with Glm5NextKdaConfig::smem_prepare / smem_scan and refuses a chunk whose
// need exceeds SMEM_CEILING (49,152 B); at D = 128 and C = 32 the scan needs 36,864 B.

















































#include <cuda_bf16.h>
#include <math.h>




extern "C" __global__ void kda_chunk_prepare(
    const float* __restrict__ k,
    const float* __restrict__ v,
    const float* __restrict__ gate,
    const float* __restrict__ beta,
    float* __restrict__ gc_out,
    float* __restrict__ u_out,
    float* __restrict__ w_out,
    unsigned int H,
    unsigned int D,
    unsigned int C,
    unsigned int T
) {
    extern __shared__ float sh[];
    float* gc = sh;
    float* A = sh + (size_t)C * D;
    float* row = A + (size_t)C * C;

    const unsigned int c = blockIdx.x;
    const unsigned int h = blockIdx.y;
    const size_t base = ((size_t)c * C * H + h) * D;
    const size_t stride = (size_t)H * D;

    // 2026-09-25: A pad position is read as zero here instead of being trusted to be zero
    // in the buffer. A non-zero gate there would still move gc[C-1], which decays the whole
    // carried state, while this prefill's own output rows stay correct.

    #define KDA_AT(buf, i, d) (((c) * C + (i)) < T ? (buf)[base + (size_t)(i) * stride + (d)] : 0.0f)
    #define KDA_BETA(i) (((c) * C + (i)) < T ? beta[((size_t)((c) * C + (i)) * H) + h] : 0.0f)


    for (unsigned int d = threadIdx.x; d < D; d += blockDim.x) {
        float acc = 0.0f;
        for (unsigned int i = 0; i < C; ++i) {
            acc += KDA_AT(gate, i, d);
            gc[(size_t)i * D + d] = acc;
        }
    }
    __syncthreads();


    for (unsigned int idx = threadIdx.x; idx < C * C; idx += blockDim.x) {
        const unsigned int i = idx / C, j = idx % C;
        float a = 0.0f;
        if (j < i) {
            const float bi = KDA_BETA(i);
            float acc = 0.0f;
            for (unsigned int d = 0; d < D; ++d) {
                acc += KDA_AT(k, i, d) * bi * KDA_AT(k, j, d)
                     * expf(gc[(size_t)i * D + d] - gc[(size_t)j * D + d]);
            }
            a = -acc;
        }
        A[idx] = a;
    }
    __syncthreads();


    for (unsigned int i = 1; i < C; ++i) {
        for (unsigned int j = threadIdx.x; j < i; j += blockDim.x) {
            float acc = 0.0f;
            for (unsigned int m = 0; m < i; ++m) {
                acc += A[(size_t)i * C + m] * A[(size_t)m * C + j];
            }
            row[j] = A[(size_t)i * C + j] + acc;
        }
        __syncthreads();
        for (unsigned int j = threadIdx.x; j < i; j += blockDim.x) {
            A[(size_t)i * C + j] = row[j];
        }
        __syncthreads();
    }
    for (unsigned int i = threadIdx.x; i < C; i += blockDim.x) {
        A[(size_t)i * C + i] = 1.0f;
    }
    __syncthreads();


    for (unsigned int idx = threadIdx.x; idx < C * D; idx += blockDim.x) {
        const unsigned int i = idx / D, d = idx % D;
        float au = 0.0f, aw = 0.0f;
        for (unsigned int j = 0; j <= i; ++j) {
            const float a = A[(size_t)i * C + j];
            if (a == 0.0f) continue;
            const float bj = KDA_BETA(j);
            au += a * KDA_AT(v, j, d) * bj;
            aw += a * KDA_AT(k, j, d) * bj * expf(gc[(size_t)j * D + d]);
        }
        const size_t o = base + (size_t)i * stride + d;
        u_out[o] = au;
        w_out[o] = aw;
        gc_out[o] = gc[(size_t)i * D + d];
    }
    #undef KDA_AT
    #undef KDA_BETA
}




extern "C" __global__ void kda_chunk_scan(
    const float* __restrict__ q,
    const float* __restrict__ k,
    const float* __restrict__ gc_in,
    const float* __restrict__ u_in,
    const float* __restrict__ w_in,
    float* __restrict__ state,
    float* __restrict__ out,
    unsigned int H,
    unsigned int D,
    unsigned int C,
    unsigned int num_chunks,
    unsigned int T,
    float scale
) {
    extern __shared__ float sh[];
    float* gc = sh;
    float* vnew = sh + (size_t)C * D;
    float* intra = vnew + (size_t)C * D;

    const unsigned int h = blockIdx.x;
    const size_t stride = (size_t)H * D;
    float* S = state + (size_t)h * D * D;

    for (unsigned int c = 0; c < num_chunks; ++c) {
        const size_t base = ((size_t)c * C * H + h) * D;

        for (unsigned int idx = threadIdx.x; idx < C * D; idx += blockDim.x) {
            gc[idx] = gc_in[base + (size_t)(idx / D) * stride + (idx % D)];
        }
        __syncthreads();


        for (unsigned int idx = threadIdx.x; idx < C * D; idx += blockDim.x) {
            const unsigned int i = idx / D, vi = idx % D;
            float acc = 0.0f;
            for (unsigned int kk = 0; kk < D; ++kk) {
                acc += w_in[base + (size_t)i * stride + kk] * S[(size_t)kk * D + vi];
            }
            vnew[idx] = u_in[base + (size_t)i * stride + vi] - acc;
        }

        for (unsigned int idx = threadIdx.x; idx < C * C; idx += blockDim.x) {
            const unsigned int i = idx / C, j = idx % C;
            float a = 0.0f;
            if (j <= i) {
                for (unsigned int d = 0; d < D; ++d) {
                    a += q[base + (size_t)i * stride + d] * scale
                       * k[base + (size_t)j * stride + d]
                       * expf(gc[(size_t)i * D + d] - gc[(size_t)j * D + d]);
                }
            }
            intra[idx] = a;
        }
        __syncthreads();


        for (unsigned int idx = threadIdx.x; idx < C * D; idx += blockDim.x) {
            const unsigned int i = idx / D, vi = idx % D;
            const unsigned int t = c * C + i;
            if (t >= T) continue;
            float acc = 0.0f;
            for (unsigned int kk = 0; kk < D; ++kk) {
                acc += q[base + (size_t)i * stride + kk] * scale
                     * expf(gc[(size_t)i * D + kk]) * S[(size_t)kk * D + vi];
            }
            for (unsigned int j = 0; j <= i; ++j) {
                acc += intra[(size_t)i * C + j] * vnew[(size_t)j * D + vi];
            }
            out[base + (size_t)i * stride + vi] = acc;
        }
        __syncthreads();


        for (unsigned int idx = threadIdx.x; idx < D * D; idx += blockDim.x) {
            const unsigned int kk = idx / D, vi = idx % D;
            const float gl = gc[(size_t)(C - 1) * D + kk];
            float acc = S[idx] * expf(gl);
            for (unsigned int i = 0; i < C; ++i) {


                if (c * C + i >= T) continue;
                acc += k[base + (size_t)i * stride + kk]
                     * expf(gl - gc[(size_t)i * D + kk])
                     * vnew[(size_t)i * D + vi];
            }
            S[idx] = acc;
        }
        __syncthreads();
    }
}
