// SPDX-License-Identifier: MIT OR Apache-2.0

// 2026-09-25: GDN prefill over a whole sequence with the h-state held in shared memory,
// taken in WY chunks of C = 32 tokens.
//
// Owner: gb10 kernels.
// Invariants:
// - Each block loads its (sequence, v-head) H slice into shared memory once, runs every
//   token, and writes H back to global memory once at the end.
// - Full chunks use the WY form (pass 1: C dots against the chunk-start H; pass 2: the C
//   updates in token order); the seq_len % C tail runs one token at a time.
// K_DIM = V_DIM = 128 at compile time. Grid (num_v_heads, batch), block 128. Dynamic shared
// memory is H (64 KiB) + k and q as BF16 (8 KiB each) + 4 warp sums + kd[C][C] + g[C] +
// beta[C] floats = 86,288 bytes, the size qwen3_ssm/trait_prefill_gdn.rs passes. Unlike
// gated_delta_rule_decode, this kernel does not clamp the gate.


#include <cuda_bf16.h>

#define K_DIM 128
#define V_DIM 128
#define C     32

__device__ __forceinline__ float wy_warp_reduce(float val) {
    for (int offset = 16; offset >= 1; offset >>= 1)
        val += __shfl_down_sync(0xFFFFFFFF, val, offset);
    return val;
}

__device__ __forceinline__ float wy_block_reduce(float val, float* smem_warp, unsigned int tid) {
    val = wy_warp_reduce(val);
    if (tid % 32 == 0) smem_warp[tid / 32] = val;
    __syncthreads();
    float result = 0.0f;
    if (tid == 0)
        result = smem_warp[0] + smem_warp[1] + smem_warp[2] + smem_warp[3];
    __syncthreads();
    return result;
}

extern "C" __global__ void __launch_bounds__(128, 1)
gated_delta_rule_prefill_wy64(
    float* __restrict__ h_state,
    const __nv_bfloat16* __restrict__ query,
    const __nv_bfloat16* __restrict__ key,
    const __nv_bfloat16* __restrict__ value,
    const float* __restrict__ gate,
    const float* __restrict__ beta,
    __nv_bfloat16* __restrict__ output,
    unsigned int batch_size,
    unsigned int seq_len,
    unsigned int num_k_heads,
    unsigned int num_v_heads,
    unsigned int k_dim,
    unsigned int v_dim,
    unsigned int qk_stride,
    unsigned int v_stride,
    unsigned int gb_stride
) {
    const unsigned int vh = blockIdx.x;
    const unsigned int b = blockIdx.y;
    if (vh >= num_v_heads || b >= batch_size) return;

    const unsigned int tid = threadIdx.x;
    const unsigned int head_repeat = num_v_heads / num_k_heads;
    const unsigned int kh = vh / head_repeat;
    const float inv_sqrt_d = rsqrtf((float)k_dim);






    // 2026-09-25: Inputs and outputs are indexed without a sequence offset, so this entry
    // point is correct only at batch_size 1, which the single-stream prefill passes
    // (qwen3_ssm/trait_prefill_gdn.rs). The batched entry point below adds the offsets.
    extern __shared__ char smem_raw[];

    float* H_smem = (float*)smem_raw;
    __nv_bfloat16* smem_k = (__nv_bfloat16*)(smem_raw + K_DIM * V_DIM * 4);
    __nv_bfloat16* smem_q = smem_k + C * K_DIM;
    float* smem_warp = (float*)(smem_q + C * K_DIM);
    float* smem_kd = smem_warp + 4;
    float* smem_g = smem_kd + C * C;
    float* smem_bt = smem_g + C;


    float* H_global = h_state + ((unsigned long long)(b * num_v_heads + vh) * K_DIM * V_DIM);
    #pragma unroll 4
    for (unsigned int i = tid; i < K_DIM * V_DIM; i += V_DIM) {
        H_smem[i] = H_global[i];
    }
    __syncthreads();

    unsigned int wy_end = (seq_len / C) * C;




    for (unsigned int chunk_start = 0; chunk_start < wy_end; chunk_start += C) {


        for (unsigned int idx = tid; idx < C * K_DIM; idx += V_DIM) {
            unsigned int tok = idx / K_DIM;
            unsigned int dim = idx % K_DIM;
            unsigned long long off = (unsigned long long)(chunk_start + tok) * qk_stride + kh * k_dim + dim;
            smem_k[tok * K_DIM + dim] = key[off];
            smem_q[tok * K_DIM + dim] = query[off];
        }
        if (tid < C) {
            smem_g[tid] = gate[(unsigned long long)(chunk_start + tid) * gb_stride + vh];
            smem_bt[tid] = beta[(unsigned long long)(chunk_start + tid) * gb_stride + vh];
        }
        __syncthreads();

        // 2026-09-25: smem_kd[i*C + j] = k_i . k_j for j < i; every other entry stays 0.
        for (unsigned int idx = tid; idx < C * C; idx += V_DIM) {
            smem_kd[idx] = 0.0f;
        }
        __syncthreads();

        for (int i = 1; i < C; i++) {
            for (int j = 0; j < i; j++) {
                float partial = (tid < K_DIM) ?
                    (float)smem_k[i * K_DIM + tid] * (float)smem_k[j * K_DIM + tid] : 0.0f;
                float dot = wy_block_reduce(partial, smem_warp, tid);
                if (tid == 0) smem_kd[i * C + j] = dot;
                __syncthreads();
            }
        }


        float hk_prev[C];
        for (int t = 0; t < C; t++) hk_prev[t] = 0.0f;

        for (int j = 0; j < K_DIM; j++) {
            float h_j = H_smem[j * V_DIM + tid];
            for (int t = 0; t < C; t++) {
                hk_prev[t] += h_j * (float)smem_k[t * K_DIM + j];
            }
        }


        float v_new_arr[C];
        for (int t = 0; t < C; t++) {
            float v_t = (float)value[(unsigned long long)(chunk_start + t) * v_stride + vh * v_dim + tid];


            float g_prod = 1.0f;
            for (int s = 0; s < t; s++) g_prod *= smem_g[s];

            float hk_corr = g_prod * hk_prev[t];


            for (int s = 0; s < t; s++) {
                float g_prod_s = 1.0f;
                for (int m = s + 1; m < t; m++) g_prod_s *= smem_g[m];
                hk_corr += g_prod_s * smem_kd[t * C + s] * v_new_arr[s];
            }

            v_new_arr[t] = (v_t - smem_g[t] * hk_corr) * smem_bt[t];
        }


        float o_out[C];
        for (int t = 0; t < C; t++) o_out[t] = 0.0f;

        for (int j = 0; j < K_DIM; j++) {
            float h_j = H_smem[j * V_DIM + tid];
            for (int t = 0; t < C; t++) {
                h_j = smem_g[t] * h_j + (float)smem_k[t * K_DIM + j] * v_new_arr[t];
                o_out[t] += h_j * (float)smem_q[t * K_DIM + j];
            }
            H_smem[j * V_DIM + tid] = h_j;
        }


        for (int t = 0; t < C; t++) {
            unsigned long long out_off = (unsigned long long)(chunk_start + t) * num_v_heads * v_dim + vh * v_dim + tid;
            output[out_off] = __float2bfloat16(o_out[t] * inv_sqrt_d);
        }
        __syncthreads();
    }


    for (unsigned int t = wy_end; t < seq_len; t++) {
        if (tid < K_DIM) {
            unsigned long long qk_off = (unsigned long long)t * qk_stride + kh * k_dim;
            smem_k[tid] = key[qk_off + tid];
            smem_q[tid] = query[qk_off + tid];
        }
        __syncthreads();

        float v_i = (float)value[(unsigned long long)t * v_stride + vh * v_dim + tid];
        float g_t = gate[(unsigned long long)t * gb_stride + vh];
        float bt_t = beta[(unsigned long long)t * gb_stride + vh];

        float hk = 0.0f;
        for (int j = 0; j < K_DIM; j += 4) {
            hk += H_smem[(j+0)*V_DIM+tid]*(float)smem_k[j]
                + H_smem[(j+1)*V_DIM+tid]*(float)smem_k[j+1]
                + H_smem[(j+2)*V_DIM+tid]*(float)smem_k[j+2]
                + H_smem[(j+3)*V_DIM+tid]*(float)smem_k[j+3];
        }
        float vn = (v_i - g_t * hk) * bt_t;

        float q_dot = 0.0f;
        for (int j = 0; j < K_DIM; j += 4) {
            float h0 = g_t*H_smem[(j+0)*V_DIM+tid] + (float)smem_k[j]*vn;
            float h1 = g_t*H_smem[(j+1)*V_DIM+tid] + (float)smem_k[j+1]*vn;
            float h2 = g_t*H_smem[(j+2)*V_DIM+tid] + (float)smem_k[j+2]*vn;
            float h3 = g_t*H_smem[(j+3)*V_DIM+tid] + (float)smem_k[j+3]*vn;
            H_smem[(j+0)*V_DIM+tid]=h0; H_smem[(j+1)*V_DIM+tid]=h1;
            H_smem[(j+2)*V_DIM+tid]=h2; H_smem[(j+3)*V_DIM+tid]=h3;
            q_dot += h0*(float)smem_q[j] + h1*(float)smem_q[j+1]
                   + h2*(float)smem_q[j+2] + h3*(float)smem_q[j+3];
        }
        unsigned long long out_off = (unsigned long long)t * num_v_heads * v_dim + vh * v_dim + tid;
        output[out_off] = __float2bfloat16(q_dot * inv_sqrt_d);
        __syncthreads();
    }


    #pragma unroll 4
    for (unsigned int i = tid; i < K_DIM * V_DIM; i += V_DIM) {
        H_global[i] = H_smem[i];
    }
}










// 2026-09-25: Batched entry point: one launch over `batch_size` streams of equal length.
// - h_state_ptrs[b] is stream b's h-state base; head vh starts vh * 128 * 128 floats in.
// - Stream b's inputs and outputs start at b * seq_len rows (qk_stride, v_stride and
//   gb_stride elements per row, num_v_heads * v_dim for the output), so every stream must
//   have seq_len tokens. batched_layer.rs (model-engine prefill_b) takes this path only when
//   all stream lengths are equal.
// Otherwise it is gated_delta_rule_prefill_wy64.
extern "C" __global__ void __launch_bounds__(128, 1)
gated_delta_rule_prefill_wy64_batched(
    float* const* __restrict__ h_state_ptrs,
    const __nv_bfloat16* __restrict__ query,
    const __nv_bfloat16* __restrict__ key,
    const __nv_bfloat16* __restrict__ value,
    const float* __restrict__ gate,
    const float* __restrict__ beta,
    __nv_bfloat16* __restrict__ output,
    unsigned int batch_size,
    unsigned int seq_len,
    unsigned int num_k_heads,
    unsigned int num_v_heads,
    unsigned int k_dim,
    unsigned int v_dim,
    unsigned int qk_stride,
    unsigned int v_stride,
    unsigned int gb_stride
) {
    const unsigned int vh = blockIdx.x;
    const unsigned int b = blockIdx.y;
    if (vh >= num_v_heads || b >= batch_size) return;

    const unsigned int tid = threadIdx.x;
    const unsigned int head_repeat = num_v_heads / num_k_heads;
    const unsigned int kh = vh / head_repeat;
    const float inv_sqrt_d = rsqrtf((float)k_dim);


    const unsigned long long qk_batch_off = (unsigned long long)b * seq_len * qk_stride;
    const unsigned long long v_batch_off  = (unsigned long long)b * seq_len * v_stride;
    const unsigned long long gb_batch_off = (unsigned long long)b * seq_len * gb_stride;
    const unsigned long long out_batch_off = (unsigned long long)b * seq_len * num_v_heads * v_dim;

    extern __shared__ char smem_raw[];

    float* H_smem = (float*)smem_raw;
    __nv_bfloat16* smem_k = (__nv_bfloat16*)(smem_raw + K_DIM * V_DIM * 4);
    __nv_bfloat16* smem_q = smem_k + C * K_DIM;
    float* smem_warp = (float*)(smem_q + C * K_DIM);
    float* smem_kd = smem_warp + 4;
    float* smem_g = smem_kd + C * C;
    float* smem_bt = smem_g + C;


    float* H_global = h_state_ptrs[b] + ((unsigned long long)vh * K_DIM * V_DIM);
    #pragma unroll 4
    for (unsigned int i = tid; i < K_DIM * V_DIM; i += V_DIM) {
        H_smem[i] = H_global[i];
    }
    __syncthreads();

    unsigned int wy_end = (seq_len / C) * C;

    for (unsigned int chunk_start = 0; chunk_start < wy_end; chunk_start += C) {
        for (unsigned int idx = tid; idx < C * K_DIM; idx += V_DIM) {
            unsigned int tok = idx / K_DIM;
            unsigned int dim = idx % K_DIM;
            unsigned long long off = qk_batch_off + (unsigned long long)(chunk_start + tok) * qk_stride + kh * k_dim + dim;
            smem_k[tok * K_DIM + dim] = key[off];
            smem_q[tok * K_DIM + dim] = query[off];
        }
        if (tid < C) {
            smem_g[tid] = gate[gb_batch_off + (unsigned long long)(chunk_start + tid) * gb_stride + vh];
            smem_bt[tid] = beta[gb_batch_off + (unsigned long long)(chunk_start + tid) * gb_stride + vh];
        }
        __syncthreads();

        for (unsigned int idx = tid; idx < C * C; idx += V_DIM) {
            smem_kd[idx] = 0.0f;
        }
        __syncthreads();

        for (int i = 1; i < C; i++) {
            for (int j = 0; j < i; j++) {
                float partial = (tid < K_DIM) ?
                    (float)smem_k[i * K_DIM + tid] * (float)smem_k[j * K_DIM + tid] : 0.0f;
                float dot = wy_block_reduce(partial, smem_warp, tid);
                if (tid == 0) smem_kd[i * C + j] = dot;
                __syncthreads();
            }
        }

        float hk_prev[C];
        for (int t = 0; t < C; t++) hk_prev[t] = 0.0f;

        for (int j = 0; j < K_DIM; j++) {
            float h_j = H_smem[j * V_DIM + tid];
            for (int t = 0; t < C; t++) {
                hk_prev[t] += h_j * (float)smem_k[t * K_DIM + j];
            }
        }

        float v_new_arr[C];
        for (int t = 0; t < C; t++) {
            float v_t = (float)value[v_batch_off + (unsigned long long)(chunk_start + t) * v_stride + vh * v_dim + tid];

            float g_prod = 1.0f;
            for (int s = 0; s < t; s++) g_prod *= smem_g[s];

            float hk_corr = g_prod * hk_prev[t];

            for (int s = 0; s < t; s++) {
                float g_prod_s = 1.0f;
                for (int m = s + 1; m < t; m++) g_prod_s *= smem_g[m];
                hk_corr += g_prod_s * smem_kd[t * C + s] * v_new_arr[s];
            }

            v_new_arr[t] = (v_t - smem_g[t] * hk_corr) * smem_bt[t];
        }

        float o_out[C];
        for (int t = 0; t < C; t++) o_out[t] = 0.0f;

        for (int j = 0; j < K_DIM; j++) {
            float h_j = H_smem[j * V_DIM + tid];
            for (int t = 0; t < C; t++) {
                h_j = smem_g[t] * h_j + (float)smem_k[t * K_DIM + j] * v_new_arr[t];
                o_out[t] += h_j * (float)smem_q[t * K_DIM + j];
            }
            H_smem[j * V_DIM + tid] = h_j;
        }

        for (int t = 0; t < C; t++) {
            unsigned long long out_off = out_batch_off + (unsigned long long)(chunk_start + t) * num_v_heads * v_dim + vh * v_dim + tid;
            output[out_off] = __float2bfloat16(o_out[t] * inv_sqrt_d);
        }
        __syncthreads();
    }

    for (unsigned int t = wy_end; t < seq_len; t++) {
        if (tid < K_DIM) {
            unsigned long long qk_off = qk_batch_off + (unsigned long long)t * qk_stride + kh * k_dim;
            smem_k[tid] = key[qk_off + tid];
            smem_q[tid] = query[qk_off + tid];
        }
        __syncthreads();

        float v_i = (float)value[v_batch_off + (unsigned long long)t * v_stride + vh * v_dim + tid];
        float g_t = gate[gb_batch_off + (unsigned long long)t * gb_stride + vh];
        float bt_t = beta[gb_batch_off + (unsigned long long)t * gb_stride + vh];

        float hk = 0.0f;
        for (int j = 0; j < K_DIM; j += 4) {
            hk += H_smem[(j+0)*V_DIM+tid]*(float)smem_k[j]
                + H_smem[(j+1)*V_DIM+tid]*(float)smem_k[j+1]
                + H_smem[(j+2)*V_DIM+tid]*(float)smem_k[j+2]
                + H_smem[(j+3)*V_DIM+tid]*(float)smem_k[j+3];
        }
        float vn = (v_i - g_t * hk) * bt_t;

        float q_dot = 0.0f;
        for (int j = 0; j < K_DIM; j += 4) {
            float h0 = g_t*H_smem[(j+0)*V_DIM+tid] + (float)smem_k[j]*vn;
            float h1 = g_t*H_smem[(j+1)*V_DIM+tid] + (float)smem_k[j+1]*vn;
            float h2 = g_t*H_smem[(j+2)*V_DIM+tid] + (float)smem_k[j+2]*vn;
            float h3 = g_t*H_smem[(j+3)*V_DIM+tid] + (float)smem_k[j+3]*vn;
            H_smem[(j+0)*V_DIM+tid]=h0; H_smem[(j+1)*V_DIM+tid]=h1;
            H_smem[(j+2)*V_DIM+tid]=h2; H_smem[(j+3)*V_DIM+tid]=h3;
            q_dot += h0*(float)smem_q[j] + h1*(float)smem_q[j+1]
                   + h2*(float)smem_q[j+2] + h3*(float)smem_q[j+3];
        }
        unsigned long long out_off = out_batch_off + (unsigned long long)t * num_v_heads * v_dim + vh * v_dim + tid;
        output[out_off] = __float2bfloat16(q_dot * inv_sqrt_d);
        __syncthreads();
    }

    #pragma unroll 4
    for (unsigned int i = tid; i < K_DIM * V_DIM; i += V_DIM) {
        H_global[i] = H_smem[i];
    }
}
