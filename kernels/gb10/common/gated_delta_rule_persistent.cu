// SPDX-License-Identifier: MIT OR Apache-2.0

// 2026-09-25: Token-sequential GDN prefill kernels that keep a head's FP32 state in one block
// for the whole sequence.
//
// Each kernel reads a head's 128 x 128 state once, runs all seq_len tokens against it (in shared
// memory; in registers for _regtile) and writes it back once. Per token and value column:
//   hk = H^T k,  v_new = (v - g * hk) * beta,  H = g * H + k v_new^T,  then out = (H^T q) * rsqrt(k_dim).
// The gate is used as given; unlike gated_delta_rule_decode, these kernels do not clamp it.
//
// Owner: gb10 kernels.
// Invariants:
// - k_dim == v_dim == 128: K_DIM and V_DIM are compile-time, and thread tid of a 128-thread block
//   owns value column tid.
// - The unbatched kernels read token t at row t whatever b is, so they are correct only at
//   batch_size 1, which every caller passes. The _batched kernels take h_state as a table of
//   per-sequence pointers and start sequence b at row b * seq_len.
// - Token t of sequence b is written to output row (b * seq_len + t) * num_v_heads + vh.











#include <cuda_bf16.h>

#define K_DIM 128
#define V_DIM 128

// 2026-09-25: gated_delta_rule_prefill_persistent: one token per iteration, grid (num_v_heads,
// batch_size). K and Q are double-buffered in shared memory: token t + 1's are loaded while token
// t computes. Dynamic shared memory: K_DIM * V_DIM + 4 * K_DIM floats (67,584 B).
















extern "C" __global__ void __launch_bounds__(128, 1)
gated_delta_rule_prefill_persistent(

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







    extern __shared__ float smem_base[];

    float* H_smem = smem_base;
    float* smem_k0 = smem_base + K_DIM * V_DIM;
    float* smem_q0 = smem_k0 + K_DIM;
    float* smem_k1 = smem_q0 + K_DIM;
    float* smem_q1 = smem_k1 + K_DIM;



    float* H_global = h_state + ((unsigned long long)(b * num_v_heads + vh) * K_DIM * V_DIM);

    #pragma unroll 4
    for (unsigned int i = tid; i < K_DIM * V_DIM; i += 128) {
        H_smem[i] = H_global[i];
    }

    float inv_sqrt_d = rsqrtf((float)k_dim);

    if (seq_len == 0) goto writeback;


    {
        unsigned long long qk_off = (unsigned long long)0 * qk_stride + kh * k_dim;
        smem_k0[tid] = (float)key[qk_off + tid];
        smem_q0[tid] = (float)query[qk_off + tid];
    }
    __syncthreads();













    for (unsigned int t = 0; t < seq_len; t++) {

        float* cur_k = (t & 1) ? smem_k1 : smem_k0;
        float* cur_q = (t & 1) ? smem_q1 : smem_q0;
        float* nxt_k = (t & 1) ? smem_k0 : smem_k1;
        float* nxt_q = (t & 1) ? smem_q0 : smem_q1;


        if (t + 1 < seq_len) {
            unsigned long long qk_off_nxt = (unsigned long long)(t + 1) * qk_stride + kh * k_dim;
            nxt_k[tid] = (float)key[qk_off_nxt + tid];
            nxt_q[tid] = (float)query[qk_off_nxt + tid];
        }


        float v_i  = (float)value[(unsigned long long)t * v_stride + vh * v_dim + tid];
        float g_t  = gate[(unsigned long long)t * gb_stride + vh];
        float bt_t = beta[(unsigned long long)t * gb_stride + vh];

        // 2026-09-25: Four partial sums break the dependent FMA chain; they are added as
        // (a + b) + (c + d).

        float hk_a = 0.0f, hk_b = 0.0f, hk_c = 0.0f, hk_d = 0.0f;
        #pragma unroll
        for (int j = 0; j < K_DIM; j += 4) {
            hk_a += H_smem[(j + 0) * V_DIM + tid] * cur_k[j];
            hk_b += H_smem[(j + 1) * V_DIM + tid] * cur_k[j + 1];
            hk_c += H_smem[(j + 2) * V_DIM + tid] * cur_k[j + 2];
            hk_d += H_smem[(j + 3) * V_DIM + tid] * cur_k[j + 3];
        }
        float hk_dot = (hk_a + hk_b) + (hk_c + hk_d);


        float v_new = (v_i - g_t * hk_dot) * bt_t;



        float qd_a = 0.0f, qd_b = 0.0f, qd_c = 0.0f, qd_d = 0.0f;
        #pragma unroll
        for (int j = 0; j < K_DIM; j += 4) {
            float h0 = g_t * H_smem[(j + 0) * V_DIM + tid] + cur_k[j]     * v_new;
            float h1 = g_t * H_smem[(j + 1) * V_DIM + tid] + cur_k[j + 1] * v_new;
            float h2 = g_t * H_smem[(j + 2) * V_DIM + tid] + cur_k[j + 2] * v_new;
            float h3 = g_t * H_smem[(j + 3) * V_DIM + tid] + cur_k[j + 3] * v_new;
            H_smem[(j + 0) * V_DIM + tid] = h0;
            H_smem[(j + 1) * V_DIM + tid] = h1;
            H_smem[(j + 2) * V_DIM + tid] = h2;
            H_smem[(j + 3) * V_DIM + tid] = h3;
            qd_a += h0 * cur_q[j];
            qd_b += h1 * cur_q[j + 1];
            qd_c += h2 * cur_q[j + 2];
            qd_d += h3 * cur_q[j + 3];
        }
        float q_dot = (qd_a + qd_b) + (qd_c + qd_d);


        output[((unsigned long long)(b * seq_len + t) * num_v_heads + vh) * v_dim + tid] =
            __float2bfloat16(q_dot * inv_sqrt_d);

        __syncthreads();
    }

writeback:

    #pragma unroll 4
    for (unsigned int i = tid; i < K_DIM * V_DIM; i += 128) {
        H_global[i] = H_smem[i];
    }
}

// 2026-09-25: gated_delta_rule_prefill_persistent_wy4: four tokens per iteration. Pass 1 reads H
// once for the four H^T k products; hkN_corr turns each into the product with the state after
// the earlier tokens of the group, using the pairwise k dots kd10 .. kd32; pass 2 applies the four
// updates. The seq_len % 4 tail runs one token at a time. Grid (num_v_heads, batch_size).
// Dynamic shared memory: K_DIM * V_DIM + 8 * K_DIM + 4 floats (69,648 B); the k dots are static
// shared variables.












// 2026-09-25: wy_block_reduce leaves the 128-thread block sum on thread 0 only; others get 0.
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
    return result;
}

extern "C" __global__ void __launch_bounds__(128, 1)
gated_delta_rule_prefill_persistent_wy4(
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













    extern __shared__ float smem_base[];

    float* H_smem = smem_base;
    float* smem_k0 = smem_base + K_DIM * V_DIM;
    float* smem_q0 = smem_k0 + K_DIM;
    float* smem_k1 = smem_q0 + K_DIM;
    float* smem_q1 = smem_k1 + K_DIM;
    float* smem_k2 = smem_q1 + K_DIM;
    float* smem_q2 = smem_k2 + K_DIM;
    float* smem_k3 = smem_q2 + K_DIM;
    float* smem_q3 = smem_k3 + K_DIM;
    float* smem_warp = smem_q3 + K_DIM;

    __shared__ float kd10, kd20, kd21, kd30, kd31, kd32;


    float* H_global = h_state + ((unsigned long long)(b * num_v_heads + vh) * K_DIM * V_DIM);
    #pragma unroll 4
    for (unsigned int i = tid; i < K_DIM * V_DIM; i += 128) {
        H_smem[i] = H_global[i];
    }

    float inv_sqrt_d = rsqrtf((float)k_dim);
    unsigned int wy4_end = (seq_len / 4) * 4;
    __syncthreads();




    for (unsigned int t = 0; t < wy4_end; t += 4) {

        if (tid < K_DIM) {
            unsigned long long off0 = (unsigned long long)(t + 0) * qk_stride + kh * k_dim;
            unsigned long long off1 = (unsigned long long)(t + 1) * qk_stride + kh * k_dim;
            unsigned long long off2 = (unsigned long long)(t + 2) * qk_stride + kh * k_dim;
            unsigned long long off3 = (unsigned long long)(t + 3) * qk_stride + kh * k_dim;
            smem_k0[tid] = (float)key[off0 + tid];   smem_q0[tid] = (float)query[off0 + tid];
            smem_k1[tid] = (float)key[off1 + tid];   smem_q1[tid] = (float)query[off1 + tid];
            smem_k2[tid] = (float)key[off2 + tid];   smem_q2[tid] = (float)query[off2 + tid];
            smem_k3[tid] = (float)key[off3 + tid];   smem_q3[tid] = (float)query[off3 + tid];
        }
        __syncthreads();


        float vi0 = (float)value[(unsigned long long)(t + 0) * v_stride + vh * v_dim + tid];
        float vi1 = (float)value[(unsigned long long)(t + 1) * v_stride + vh * v_dim + tid];
        float vi2 = (float)value[(unsigned long long)(t + 2) * v_stride + vh * v_dim + tid];
        float vi3 = (float)value[(unsigned long long)(t + 3) * v_stride + vh * v_dim + tid];
        float g0 = gate[(unsigned long long)(t + 0) * gb_stride + vh];
        float g1 = gate[(unsigned long long)(t + 1) * gb_stride + vh];
        float g2 = gate[(unsigned long long)(t + 2) * gb_stride + vh];
        float g3 = gate[(unsigned long long)(t + 3) * gb_stride + vh];
        float bt0 = beta[(unsigned long long)(t + 0) * gb_stride + vh];
        float bt1 = beta[(unsigned long long)(t + 1) * gb_stride + vh];
        float bt2 = beta[(unsigned long long)(t + 2) * gb_stride + vh];
        float bt3 = beta[(unsigned long long)(t + 3) * gb_stride + vh];


        {
            float p10 = 0.0f, p20 = 0.0f, p21 = 0.0f, p30 = 0.0f, p31 = 0.0f, p32 = 0.0f;
            if (tid < K_DIM) {
                p10 = smem_k1[tid] * smem_k0[tid];
                p20 = smem_k2[tid] * smem_k0[tid];
                p21 = smem_k2[tid] * smem_k1[tid];
                p30 = smem_k3[tid] * smem_k0[tid];
                p31 = smem_k3[tid] * smem_k1[tid];
                p32 = smem_k3[tid] * smem_k2[tid];
            }

            float r;
            r = wy_block_reduce(p10, smem_warp, tid); if (tid == 0) kd10 = r; __syncthreads();
            r = wy_block_reduce(p20, smem_warp, tid); if (tid == 0) kd20 = r; __syncthreads();
            r = wy_block_reduce(p21, smem_warp, tid); if (tid == 0) kd21 = r; __syncthreads();
            r = wy_block_reduce(p30, smem_warp, tid); if (tid == 0) kd30 = r; __syncthreads();
            r = wy_block_reduce(p31, smem_warp, tid); if (tid == 0) kd31 = r; __syncthreads();
            r = wy_block_reduce(p32, smem_warp, tid); if (tid == 0) kd32 = r; __syncthreads();
        }


        float hk0 = 0.0f, hk1 = 0.0f, hk2 = 0.0f, hk3 = 0.0f;
        #pragma unroll
        for (int j = 0; j < K_DIM; j += 4) {
            float h0 = H_smem[(j + 0) * V_DIM + tid];
            float h1 = H_smem[(j + 1) * V_DIM + tid];
            float h2 = H_smem[(j + 2) * V_DIM + tid];
            float h3 = H_smem[(j + 3) * V_DIM + tid];
            hk0 += h0*smem_k0[j] + h1*smem_k0[j+1] + h2*smem_k0[j+2] + h3*smem_k0[j+3];
            hk1 += h0*smem_k1[j] + h1*smem_k1[j+1] + h2*smem_k1[j+2] + h3*smem_k1[j+3];
            hk2 += h0*smem_k2[j] + h1*smem_k2[j+1] + h2*smem_k2[j+2] + h3*smem_k2[j+3];
            hk3 += h0*smem_k3[j] + h1*smem_k3[j+1] + h2*smem_k3[j+2] + h3*smem_k3[j+3];
        }

        // 2026-09-25: hkN_corr is H_N^T kN for the state H_N after tokens 0 .. N-1 of the group,
        // expanded in terms of the group's starting H.
        float v_new_0 = (vi0 - g0 * hk0) * bt0;


        float hk1_corr = g0 * hk1 + kd10 * v_new_0;
        float v_new_1 = (vi1 - g1 * hk1_corr) * bt1;


        float hk2_corr = g0 * g1 * hk2 + g1 * kd20 * v_new_0 + kd21 * v_new_1;
        float v_new_2 = (vi2 - g2 * hk2_corr) * bt2;



        float hk3_corr = g0*g1*g2 * hk3 + g1*g2 * kd30 * v_new_0
                        + g2 * kd31 * v_new_1 + kd32 * v_new_2;
        float v_new_3 = (vi3 - g3 * hk3_corr) * bt3;


        float qd0 = 0.0f, qd1 = 0.0f, qd2 = 0.0f, qd3 = 0.0f;
        #pragma unroll
        for (int j = 0; j < K_DIM; j += 4) {
            float h0 = H_smem[(j + 0) * V_DIM + tid];
            float h1 = H_smem[(j + 1) * V_DIM + tid];
            float h2 = H_smem[(j + 2) * V_DIM + tid];
            float h3 = H_smem[(j + 3) * V_DIM + tid];


            h0 = g0*h0 + smem_k0[j]*v_new_0;     h1 = g0*h1 + smem_k0[j+1]*v_new_0;
            h2 = g0*h2 + smem_k0[j+2]*v_new_0;   h3 = g0*h3 + smem_k0[j+3]*v_new_0;
            qd0 += h0*smem_q0[j] + h1*smem_q0[j+1] + h2*smem_q0[j+2] + h3*smem_q0[j+3];


            h0 = g1*h0 + smem_k1[j]*v_new_1;     h1 = g1*h1 + smem_k1[j+1]*v_new_1;
            h2 = g1*h2 + smem_k1[j+2]*v_new_1;   h3 = g1*h3 + smem_k1[j+3]*v_new_1;
            qd1 += h0*smem_q1[j] + h1*smem_q1[j+1] + h2*smem_q1[j+2] + h3*smem_q1[j+3];


            h0 = g2*h0 + smem_k2[j]*v_new_2;     h1 = g2*h1 + smem_k2[j+1]*v_new_2;
            h2 = g2*h2 + smem_k2[j+2]*v_new_2;   h3 = g2*h3 + smem_k2[j+3]*v_new_2;
            qd2 += h0*smem_q2[j] + h1*smem_q2[j+1] + h2*smem_q2[j+2] + h3*smem_q2[j+3];


            h0 = g3*h0 + smem_k3[j]*v_new_3;     h1 = g3*h1 + smem_k3[j+1]*v_new_3;
            h2 = g3*h2 + smem_k3[j+2]*v_new_3;   h3 = g3*h3 + smem_k3[j+3]*v_new_3;
            qd3 += h0*smem_q3[j] + h1*smem_q3[j+1] + h2*smem_q3[j+2] + h3*smem_q3[j+3];

            H_smem[(j + 0) * V_DIM + tid] = h0;
            H_smem[(j + 1) * V_DIM + tid] = h1;
            H_smem[(j + 2) * V_DIM + tid] = h2;
            H_smem[(j + 3) * V_DIM + tid] = h3;
        }


        unsigned long long out_base = ((unsigned long long)(b * seq_len) * num_v_heads + vh) * v_dim;
        output[out_base + (unsigned long long)(t + 0) * num_v_heads * v_dim + tid] = __float2bfloat16(qd0 * inv_sqrt_d);
        output[out_base + (unsigned long long)(t + 1) * num_v_heads * v_dim + tid] = __float2bfloat16(qd1 * inv_sqrt_d);
        output[out_base + (unsigned long long)(t + 2) * num_v_heads * v_dim + tid] = __float2bfloat16(qd2 * inv_sqrt_d);
        output[out_base + (unsigned long long)(t + 3) * num_v_heads * v_dim + tid] = __float2bfloat16(qd3 * inv_sqrt_d);

        __syncthreads();
    }




    for (unsigned int t = wy4_end; t < seq_len; t++) {
        if (tid < K_DIM) {
            unsigned long long qk_off = (unsigned long long)t * qk_stride + kh * k_dim;
            smem_k0[tid] = (float)key[qk_off + tid];
            smem_q0[tid] = (float)query[qk_off + tid];
        }
        __syncthreads();

        float v_i  = (float)value[(unsigned long long)t * v_stride + vh * v_dim + tid];
        float g_t  = gate[(unsigned long long)t * gb_stride + vh];
        float bt_t = beta[(unsigned long long)t * gb_stride + vh];

        float hk_a = 0.0f, hk_b = 0.0f, hk_c = 0.0f, hk_d = 0.0f;
        #pragma unroll
        for (int j = 0; j < K_DIM; j += 4) {
            hk_a += H_smem[(j+0)*V_DIM+tid]*smem_k0[j];   hk_b += H_smem[(j+1)*V_DIM+tid]*smem_k0[j+1];
            hk_c += H_smem[(j+2)*V_DIM+tid]*smem_k0[j+2]; hk_d += H_smem[(j+3)*V_DIM+tid]*smem_k0[j+3];
        }
        float hk_dot = (hk_a + hk_b) + (hk_c + hk_d);
        float v_new = (v_i - g_t * hk_dot) * bt_t;

        float qd_a = 0.0f, qd_b = 0.0f, qd_c = 0.0f, qd_d = 0.0f;
        #pragma unroll
        for (int j = 0; j < K_DIM; j += 4) {
            float h0 = g_t*H_smem[(j+0)*V_DIM+tid] + smem_k0[j]*v_new;
            float h1 = g_t*H_smem[(j+1)*V_DIM+tid] + smem_k0[j+1]*v_new;
            float h2 = g_t*H_smem[(j+2)*V_DIM+tid] + smem_k0[j+2]*v_new;
            float h3 = g_t*H_smem[(j+3)*V_DIM+tid] + smem_k0[j+3]*v_new;
            H_smem[(j+0)*V_DIM+tid]=h0; H_smem[(j+1)*V_DIM+tid]=h1;
            H_smem[(j+2)*V_DIM+tid]=h2; H_smem[(j+3)*V_DIM+tid]=h3;
            qd_a += h0*smem_q0[j]; qd_b += h1*smem_q0[j+1];
            qd_c += h2*smem_q0[j+2]; qd_d += h3*smem_q0[j+3];
        }
        float q_dot = (qd_a + qd_b) + (qd_c + qd_d);
        unsigned long long out_base = ((unsigned long long)(b * seq_len) * num_v_heads + vh) * v_dim;
        output[out_base + (unsigned long long)t * num_v_heads * v_dim + tid] = __float2bfloat16(q_dot * inv_sqrt_d);
        __syncthreads();
    }


    #pragma unroll 4
    for (unsigned int i = tid; i < K_DIM * V_DIM; i += 128) {
        H_global[i] = H_smem[i];
    }
}

// 2026-09-25: gated_delta_rule_prefill_persistent_multihead: gated_delta_rule_prefill_persistent
// with block x looping over heads x, x + grid_x, x + 2 * grid_x, ...; same dynamic shared memory
// (67,584 B). No code in the repository launches it.













extern "C" __global__ void __launch_bounds__(128, 1)
gated_delta_rule_prefill_persistent_multihead(
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
    unsigned int gb_stride,

    unsigned int grid_x
) {
    const unsigned int cta_id = blockIdx.x;
    const unsigned int b = blockIdx.y;
    if (b >= batch_size) return;

    const unsigned int tid = threadIdx.x;
    const unsigned int head_repeat = num_v_heads / num_k_heads;


    extern __shared__ float smem_base[];
    float* H_smem = smem_base;
    float* smem_k0 = smem_base + K_DIM * V_DIM;
    float* smem_q0 = smem_k0 + K_DIM;
    float* smem_k1 = smem_q0 + K_DIM;
    float* smem_q1 = smem_k1 + K_DIM;

    float inv_sqrt_d = rsqrtf((float)k_dim);




    for (unsigned int vh = cta_id; vh < num_v_heads; vh += grid_x) {
        const unsigned int kh = vh / head_repeat;


        float* H_global = h_state + ((unsigned long long)(b * num_v_heads + vh) * K_DIM * V_DIM);

        #pragma unroll 4
        for (unsigned int i = tid; i < K_DIM * V_DIM; i += 128) {
            H_smem[i] = H_global[i];
        }

        if (seq_len == 0) {

            continue;
        }


        {
            unsigned long long qk_off = (unsigned long long)0 * qk_stride + kh * k_dim;
            smem_k0[tid] = (float)key[qk_off + tid];
            smem_q0[tid] = (float)query[qk_off + tid];
        }
        __syncthreads();


        for (unsigned int t = 0; t < seq_len; t++) {
            float* cur_k = (t & 1) ? smem_k1 : smem_k0;
            float* cur_q = (t & 1) ? smem_q1 : smem_q0;
            float* nxt_k = (t & 1) ? smem_k0 : smem_k1;
            float* nxt_q = (t & 1) ? smem_q0 : smem_q1;


            if (t + 1 < seq_len) {
                unsigned long long qk_off_nxt = (unsigned long long)(t + 1) * qk_stride + kh * k_dim;
                nxt_k[tid] = (float)key[qk_off_nxt + tid];
                nxt_q[tid] = (float)query[qk_off_nxt + tid];
            }

            float v_i  = (float)value[(unsigned long long)t * v_stride + vh * v_dim + tid];
            float g_t  = gate[(unsigned long long)t * gb_stride + vh];
            float bt_t = beta[(unsigned long long)t * gb_stride + vh];


            float hk_a = 0.0f, hk_b = 0.0f, hk_c = 0.0f, hk_d = 0.0f;
            #pragma unroll
            for (int j = 0; j < K_DIM; j += 4) {
                hk_a += H_smem[(j + 0) * V_DIM + tid] * cur_k[j];
                hk_b += H_smem[(j + 1) * V_DIM + tid] * cur_k[j + 1];
                hk_c += H_smem[(j + 2) * V_DIM + tid] * cur_k[j + 2];
                hk_d += H_smem[(j + 3) * V_DIM + tid] * cur_k[j + 3];
            }
            float hk_dot = (hk_a + hk_b) + (hk_c + hk_d);

            float v_new = (v_i - g_t * hk_dot) * bt_t;


            float qd_a = 0.0f, qd_b = 0.0f, qd_c = 0.0f, qd_d = 0.0f;
            #pragma unroll
            for (int j = 0; j < K_DIM; j += 4) {
                float h0 = g_t * H_smem[(j + 0) * V_DIM + tid] + cur_k[j]     * v_new;
                float h1 = g_t * H_smem[(j + 1) * V_DIM + tid] + cur_k[j + 1] * v_new;
                float h2 = g_t * H_smem[(j + 2) * V_DIM + tid] + cur_k[j + 2] * v_new;
                float h3 = g_t * H_smem[(j + 3) * V_DIM + tid] + cur_k[j + 3] * v_new;
                H_smem[(j + 0) * V_DIM + tid] = h0;
                H_smem[(j + 1) * V_DIM + tid] = h1;
                H_smem[(j + 2) * V_DIM + tid] = h2;
                H_smem[(j + 3) * V_DIM + tid] = h3;
                qd_a += h0 * cur_q[j];
                qd_b += h1 * cur_q[j + 1];
                qd_c += h2 * cur_q[j + 2];
                qd_d += h3 * cur_q[j + 3];
            }
            float q_dot = (qd_a + qd_b) + (qd_c + qd_d);

            output[((unsigned long long)(b * seq_len + t) * num_v_heads + vh) * v_dim + tid] =
                __float2bfloat16(q_dot * inv_sqrt_d);

            __syncthreads();
        }


        #pragma unroll 4
        for (unsigned int i = tid; i < K_DIM * V_DIM; i += 128) {
            H_global[i] = H_smem[i];
        }


        __syncthreads();
    }
}

// 2026-09-25: gated_delta_rule_prefill_persistent_regtile: the multihead loop with the state
// column in registers (H_reg[K_DIM] per thread) instead of shared memory; only K and Q are
// staged (4 * K_DIM floats, 2,048 B of dynamic shared memory). No code in the repository
// launches it.

















extern "C" __global__ void __launch_bounds__(128, 1)
gated_delta_rule_prefill_persistent_regtile(
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
    unsigned int gb_stride,
    unsigned int grid_x
) {
    const unsigned int cta_id = blockIdx.x;
    const unsigned int b = blockIdx.y;
    if (b >= batch_size) return;

    const unsigned int tid = threadIdx.x;
    const unsigned int head_repeat = num_v_heads / num_k_heads;


    extern __shared__ float smem[];
    float* smem_k0 = smem;
    float* smem_q0 = smem + K_DIM;
    float* smem_k1 = smem + 2 * K_DIM;
    float* smem_q1 = smem + 3 * K_DIM;

    float inv_sqrt_d = rsqrtf((float)k_dim);


    for (unsigned int vh = cta_id; vh < num_v_heads; vh += grid_x) {
        const unsigned int kh = vh / head_repeat;

        float* H_global = h_state +
            ((unsigned long long)(b * num_v_heads + vh) * K_DIM * V_DIM);


        float H_reg[K_DIM];
        #pragma unroll
        for (int j = 0; j < K_DIM; j++) {
            H_reg[j] = H_global[j * V_DIM + tid];
        }

        if (seq_len == 0) {

            continue;
        }


        {
            unsigned long long qk_off = (unsigned long long)0 * qk_stride + kh * k_dim;
            smem_k0[tid] = (float)key[qk_off + tid];
            smem_q0[tid] = (float)query[qk_off + tid];
        }
        __syncthreads();


        for (unsigned int t = 0; t < seq_len; t++) {
            float* cur_k = (t & 1) ? smem_k1 : smem_k0;
            float* cur_q = (t & 1) ? smem_q1 : smem_q0;
            float* nxt_k = (t & 1) ? smem_k0 : smem_k1;
            float* nxt_q = (t & 1) ? smem_q0 : smem_q1;


            if (t + 1 < seq_len) {
                unsigned long long qk_off_nxt =
                    (unsigned long long)(t + 1) * qk_stride + kh * k_dim;
                nxt_k[tid] = (float)key[qk_off_nxt + tid];
                nxt_q[tid] = (float)query[qk_off_nxt + tid];
            }

            float v_i  = (float)value[(unsigned long long)t * v_stride + vh * v_dim + tid];
            float g_t  = gate[(unsigned long long)t * gb_stride + vh];
            float bt_t = beta[(unsigned long long)t * gb_stride + vh];


            float hk_a = 0.0f, hk_b = 0.0f, hk_c = 0.0f, hk_d = 0.0f;
            #pragma unroll
            for (int j = 0; j < K_DIM; j += 4) {
                hk_a += H_reg[j]     * cur_k[j];
                hk_b += H_reg[j + 1] * cur_k[j + 1];
                hk_c += H_reg[j + 2] * cur_k[j + 2];
                hk_d += H_reg[j + 3] * cur_k[j + 3];
            }
            float hk_dot = (hk_a + hk_b) + (hk_c + hk_d);

            float v_new = (v_i - g_t * hk_dot) * bt_t;


            float qd_a = 0.0f, qd_b = 0.0f, qd_c = 0.0f, qd_d = 0.0f;
            #pragma unroll
            for (int j = 0; j < K_DIM; j += 4) {
                float h0 = g_t * H_reg[j]     + cur_k[j]     * v_new;
                float h1 = g_t * H_reg[j + 1] + cur_k[j + 1] * v_new;
                float h2 = g_t * H_reg[j + 2] + cur_k[j + 2] * v_new;
                float h3 = g_t * H_reg[j + 3] + cur_k[j + 3] * v_new;
                H_reg[j]     = h0;
                H_reg[j + 1] = h1;
                H_reg[j + 2] = h2;
                H_reg[j + 3] = h3;
                qd_a += h0 * cur_q[j];
                qd_b += h1 * cur_q[j + 1];
                qd_c += h2 * cur_q[j + 2];
                qd_d += h3 * cur_q[j + 3];
            }
            float q_dot = (qd_a + qd_b) + (qd_c + qd_d);

            output[((unsigned long long)(b * seq_len + t) * num_v_heads + vh) * v_dim + tid] =
                __float2bfloat16(q_dot * inv_sqrt_d);

            __syncthreads();
        }


        #pragma unroll
        for (int j = 0; j < K_DIM; j++) {
            H_global[j * V_DIM + tid] = H_reg[j];
        }


        __syncthreads();
    }
}

// 2026-09-25: Batched forms of _persistent and _persistent_wy4 for sequences of equal length:
// h_state_ptrs[b] is sequence b's state, and its inputs and outputs start at row b * seq_len.









extern "C" __global__ void __launch_bounds__(128, 1)
gated_delta_rule_prefill_persistent_batched(
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

    const unsigned long long qk_batch_off = (unsigned long long)b * seq_len * qk_stride;
    const unsigned long long v_batch_off  = (unsigned long long)b * seq_len * v_stride;
    const unsigned long long gb_batch_off = (unsigned long long)b * seq_len * gb_stride;
    const unsigned long long out_batch_off = (unsigned long long)b * seq_len * num_v_heads * v_dim;

    extern __shared__ float smem_base[];

    float* H_smem = smem_base;
    float* smem_k0 = smem_base + K_DIM * V_DIM;
    float* smem_q0 = smem_k0 + K_DIM;
    float* smem_k1 = smem_q0 + K_DIM;
    float* smem_q1 = smem_k1 + K_DIM;

    float* H_global = h_state_ptrs[b] + ((unsigned long long)vh * K_DIM * V_DIM);

    #pragma unroll 4
    for (unsigned int i = tid; i < K_DIM * V_DIM; i += 128) {
        H_smem[i] = H_global[i];
    }

    float inv_sqrt_d = rsqrtf((float)k_dim);

    if (seq_len == 0) goto writeback_persistent_batched;

    {
        unsigned long long qk_off = qk_batch_off + kh * k_dim;
        smem_k0[tid] = (float)key[qk_off + tid];
        smem_q0[tid] = (float)query[qk_off + tid];
    }
    __syncthreads();

    for (unsigned int t = 0; t < seq_len; t++) {
        float* cur_k = (t & 1) ? smem_k1 : smem_k0;
        float* cur_q = (t & 1) ? smem_q1 : smem_q0;
        float* nxt_k = (t & 1) ? smem_k0 : smem_k1;
        float* nxt_q = (t & 1) ? smem_q0 : smem_q1;

        if (t + 1 < seq_len) {
            unsigned long long qk_off_nxt = qk_batch_off + (unsigned long long)(t + 1) * qk_stride + kh * k_dim;
            nxt_k[tid] = (float)key[qk_off_nxt + tid];
            nxt_q[tid] = (float)query[qk_off_nxt + tid];
        }

        float v_i  = (float)value[v_batch_off + (unsigned long long)t * v_stride + vh * v_dim + tid];
        float g_t  = gate[gb_batch_off + (unsigned long long)t * gb_stride + vh];
        float bt_t = beta[gb_batch_off + (unsigned long long)t * gb_stride + vh];

        float hk_a = 0.0f, hk_b = 0.0f, hk_c = 0.0f, hk_d = 0.0f;
        #pragma unroll
        for (int j = 0; j < K_DIM; j += 4) {
            hk_a += H_smem[(j + 0) * V_DIM + tid] * cur_k[j];
            hk_b += H_smem[(j + 1) * V_DIM + tid] * cur_k[j + 1];
            hk_c += H_smem[(j + 2) * V_DIM + tid] * cur_k[j + 2];
            hk_d += H_smem[(j + 3) * V_DIM + tid] * cur_k[j + 3];
        }
        float hk_dot = (hk_a + hk_b) + (hk_c + hk_d);
        float v_new = (v_i - g_t * hk_dot) * bt_t;

        float qd_a = 0.0f, qd_b = 0.0f, qd_c = 0.0f, qd_d = 0.0f;
        #pragma unroll
        for (int j = 0; j < K_DIM; j += 4) {
            float h0 = g_t * H_smem[(j + 0) * V_DIM + tid] + cur_k[j]     * v_new;
            float h1 = g_t * H_smem[(j + 1) * V_DIM + tid] + cur_k[j + 1] * v_new;
            float h2 = g_t * H_smem[(j + 2) * V_DIM + tid] + cur_k[j + 2] * v_new;
            float h3 = g_t * H_smem[(j + 3) * V_DIM + tid] + cur_k[j + 3] * v_new;
            H_smem[(j + 0) * V_DIM + tid] = h0;
            H_smem[(j + 1) * V_DIM + tid] = h1;
            H_smem[(j + 2) * V_DIM + tid] = h2;
            H_smem[(j + 3) * V_DIM + tid] = h3;
            qd_a += h0 * cur_q[j];
            qd_b += h1 * cur_q[j + 1];
            qd_c += h2 * cur_q[j + 2];
            qd_d += h3 * cur_q[j + 3];
        }
        float q_dot = (qd_a + qd_b) + (qd_c + qd_d);

        output[out_batch_off + ((unsigned long long)t * num_v_heads + vh) * v_dim + tid] =
            __float2bfloat16(q_dot * inv_sqrt_d);

        __syncthreads();
    }

writeback_persistent_batched:
    #pragma unroll 4
    for (unsigned int i = tid; i < K_DIM * V_DIM; i += 128) {
        H_global[i] = H_smem[i];
    }
}

extern "C" __global__ void __launch_bounds__(128, 1)
gated_delta_rule_prefill_persistent_wy4_batched(
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

    const unsigned long long qk_batch_off = (unsigned long long)b * seq_len * qk_stride;
    const unsigned long long v_batch_off  = (unsigned long long)b * seq_len * v_stride;
    const unsigned long long gb_batch_off = (unsigned long long)b * seq_len * gb_stride;
    const unsigned long long out_batch_off = (unsigned long long)b * seq_len * num_v_heads * v_dim;

    extern __shared__ float smem_base[];

    float* H_smem = smem_base;
    float* smem_k0 = smem_base + K_DIM * V_DIM;
    float* smem_q0 = smem_k0 + K_DIM;
    float* smem_k1 = smem_q0 + K_DIM;
    float* smem_q1 = smem_k1 + K_DIM;
    float* smem_k2 = smem_q1 + K_DIM;
    float* smem_q2 = smem_k2 + K_DIM;
    float* smem_k3 = smem_q2 + K_DIM;
    float* smem_q3 = smem_k3 + K_DIM;
    float* smem_warp = smem_q3 + K_DIM;
    __shared__ float kd10b, kd20b, kd21b, kd30b, kd31b, kd32b;

    float* H_global = h_state_ptrs[b] + ((unsigned long long)vh * K_DIM * V_DIM);
    #pragma unroll 4
    for (unsigned int i = tid; i < K_DIM * V_DIM; i += 128) {
        H_smem[i] = H_global[i];
    }

    float inv_sqrt_d = rsqrtf((float)k_dim);
    unsigned int wy4_end = (seq_len / 4) * 4;
    __syncthreads();

    for (unsigned int t = 0; t < wy4_end; t += 4) {
        if (tid < K_DIM) {
            unsigned long long off0 = qk_batch_off + (unsigned long long)(t + 0) * qk_stride + kh * k_dim;
            unsigned long long off1 = qk_batch_off + (unsigned long long)(t + 1) * qk_stride + kh * k_dim;
            unsigned long long off2 = qk_batch_off + (unsigned long long)(t + 2) * qk_stride + kh * k_dim;
            unsigned long long off3 = qk_batch_off + (unsigned long long)(t + 3) * qk_stride + kh * k_dim;
            smem_k0[tid] = (float)key[off0 + tid];   smem_q0[tid] = (float)query[off0 + tid];
            smem_k1[tid] = (float)key[off1 + tid];   smem_q1[tid] = (float)query[off1 + tid];
            smem_k2[tid] = (float)key[off2 + tid];   smem_q2[tid] = (float)query[off2 + tid];
            smem_k3[tid] = (float)key[off3 + tid];   smem_q3[tid] = (float)query[off3 + tid];
        }
        __syncthreads();

        float vi0 = (float)value[v_batch_off + (unsigned long long)(t + 0) * v_stride + vh * v_dim + tid];
        float vi1 = (float)value[v_batch_off + (unsigned long long)(t + 1) * v_stride + vh * v_dim + tid];
        float vi2 = (float)value[v_batch_off + (unsigned long long)(t + 2) * v_stride + vh * v_dim + tid];
        float vi3 = (float)value[v_batch_off + (unsigned long long)(t + 3) * v_stride + vh * v_dim + tid];
        float g0 = gate[gb_batch_off + (unsigned long long)(t + 0) * gb_stride + vh];
        float g1 = gate[gb_batch_off + (unsigned long long)(t + 1) * gb_stride + vh];
        float g2 = gate[gb_batch_off + (unsigned long long)(t + 2) * gb_stride + vh];
        float g3 = gate[gb_batch_off + (unsigned long long)(t + 3) * gb_stride + vh];
        float bt0 = beta[gb_batch_off + (unsigned long long)(t + 0) * gb_stride + vh];
        float bt1 = beta[gb_batch_off + (unsigned long long)(t + 1) * gb_stride + vh];
        float bt2 = beta[gb_batch_off + (unsigned long long)(t + 2) * gb_stride + vh];
        float bt3 = beta[gb_batch_off + (unsigned long long)(t + 3) * gb_stride + vh];

        {
            float p10 = 0.0f, p20 = 0.0f, p21 = 0.0f, p30 = 0.0f, p31 = 0.0f, p32 = 0.0f;
            if (tid < K_DIM) {
                p10 = smem_k1[tid] * smem_k0[tid];
                p20 = smem_k2[tid] * smem_k0[tid];
                p21 = smem_k2[tid] * smem_k1[tid];
                p30 = smem_k3[tid] * smem_k0[tid];
                p31 = smem_k3[tid] * smem_k1[tid];
                p32 = smem_k3[tid] * smem_k2[tid];
            }
            float r;
            r = wy_block_reduce(p10, smem_warp, tid); if (tid == 0) kd10b = r; __syncthreads();
            r = wy_block_reduce(p20, smem_warp, tid); if (tid == 0) kd20b = r; __syncthreads();
            r = wy_block_reduce(p21, smem_warp, tid); if (tid == 0) kd21b = r; __syncthreads();
            r = wy_block_reduce(p30, smem_warp, tid); if (tid == 0) kd30b = r; __syncthreads();
            r = wy_block_reduce(p31, smem_warp, tid); if (tid == 0) kd31b = r; __syncthreads();
            r = wy_block_reduce(p32, smem_warp, tid); if (tid == 0) kd32b = r; __syncthreads();
        }

        float hk0 = 0.0f, hk1 = 0.0f, hk2 = 0.0f, hk3 = 0.0f;
        #pragma unroll
        for (int j = 0; j < K_DIM; j += 4) {
            float h0 = H_smem[(j + 0) * V_DIM + tid];
            float h1 = H_smem[(j + 1) * V_DIM + tid];
            float h2 = H_smem[(j + 2) * V_DIM + tid];
            float h3 = H_smem[(j + 3) * V_DIM + tid];
            hk0 += h0*smem_k0[j] + h1*smem_k0[j+1] + h2*smem_k0[j+2] + h3*smem_k0[j+3];
            hk1 += h0*smem_k1[j] + h1*smem_k1[j+1] + h2*smem_k1[j+2] + h3*smem_k1[j+3];
            hk2 += h0*smem_k2[j] + h1*smem_k2[j+1] + h2*smem_k2[j+2] + h3*smem_k2[j+3];
            hk3 += h0*smem_k3[j] + h1*smem_k3[j+1] + h2*smem_k3[j+2] + h3*smem_k3[j+3];
        }

        float v_new_0 = (vi0 - g0 * hk0) * bt0;
        float hk1_corr = g0 * hk1 + kd10b * v_new_0;
        float v_new_1 = (vi1 - g1 * hk1_corr) * bt1;
        float hk2_corr = g0 * g1 * hk2 + g1 * kd20b * v_new_0 + kd21b * v_new_1;
        float v_new_2 = (vi2 - g2 * hk2_corr) * bt2;
        float hk3_corr = g0*g1*g2 * hk3 + g1*g2 * kd30b * v_new_0
                        + g2 * kd31b * v_new_1 + kd32b * v_new_2;
        float v_new_3 = (vi3 - g3 * hk3_corr) * bt3;

        float qd0 = 0.0f, qd1 = 0.0f, qd2 = 0.0f, qd3 = 0.0f;
        #pragma unroll
        for (int j = 0; j < K_DIM; j += 4) {
            float h0 = H_smem[(j + 0) * V_DIM + tid];
            float h1 = H_smem[(j + 1) * V_DIM + tid];
            float h2 = H_smem[(j + 2) * V_DIM + tid];
            float h3 = H_smem[(j + 3) * V_DIM + tid];

            h0 = g0*h0 + smem_k0[j]*v_new_0;     h1 = g0*h1 + smem_k0[j+1]*v_new_0;
            h2 = g0*h2 + smem_k0[j+2]*v_new_0;   h3 = g0*h3 + smem_k0[j+3]*v_new_0;
            qd0 += h0*smem_q0[j] + h1*smem_q0[j+1] + h2*smem_q0[j+2] + h3*smem_q0[j+3];

            h0 = g1*h0 + smem_k1[j]*v_new_1;     h1 = g1*h1 + smem_k1[j+1]*v_new_1;
            h2 = g1*h2 + smem_k1[j+2]*v_new_1;   h3 = g1*h3 + smem_k1[j+3]*v_new_1;
            qd1 += h0*smem_q1[j] + h1*smem_q1[j+1] + h2*smem_q1[j+2] + h3*smem_q1[j+3];

            h0 = g2*h0 + smem_k2[j]*v_new_2;     h1 = g2*h1 + smem_k2[j+1]*v_new_2;
            h2 = g2*h2 + smem_k2[j+2]*v_new_2;   h3 = g2*h3 + smem_k2[j+3]*v_new_2;
            qd2 += h0*smem_q2[j] + h1*smem_q2[j+1] + h2*smem_q2[j+2] + h3*smem_q2[j+3];

            h0 = g3*h0 + smem_k3[j]*v_new_3;     h1 = g3*h1 + smem_k3[j+1]*v_new_3;
            h2 = g3*h2 + smem_k3[j+2]*v_new_3;   h3 = g3*h3 + smem_k3[j+3]*v_new_3;
            qd3 += h0*smem_q3[j] + h1*smem_q3[j+1] + h2*smem_q3[j+2] + h3*smem_q3[j+3];

            H_smem[(j + 0) * V_DIM + tid] = h0;
            H_smem[(j + 1) * V_DIM + tid] = h1;
            H_smem[(j + 2) * V_DIM + tid] = h2;
            H_smem[(j + 3) * V_DIM + tid] = h3;
        }

        unsigned long long out_base = out_batch_off + (unsigned long long)vh * v_dim;
        output[out_base + (unsigned long long)(t + 0) * num_v_heads * v_dim + tid] = __float2bfloat16(qd0 * inv_sqrt_d);
        output[out_base + (unsigned long long)(t + 1) * num_v_heads * v_dim + tid] = __float2bfloat16(qd1 * inv_sqrt_d);
        output[out_base + (unsigned long long)(t + 2) * num_v_heads * v_dim + tid] = __float2bfloat16(qd2 * inv_sqrt_d);
        output[out_base + (unsigned long long)(t + 3) * num_v_heads * v_dim + tid] = __float2bfloat16(qd3 * inv_sqrt_d);

        __syncthreads();
    }

    for (unsigned int t = wy4_end; t < seq_len; t++) {
        if (tid < K_DIM) {
            unsigned long long qk_off = qk_batch_off + (unsigned long long)t * qk_stride + kh * k_dim;
            smem_k0[tid] = (float)key[qk_off + tid];
            smem_q0[tid] = (float)query[qk_off + tid];
        }
        __syncthreads();

        float v_i  = (float)value[v_batch_off + (unsigned long long)t * v_stride + vh * v_dim + tid];
        float g_t  = gate[gb_batch_off + (unsigned long long)t * gb_stride + vh];
        float bt_t = beta[gb_batch_off + (unsigned long long)t * gb_stride + vh];

        float hk_a = 0.0f, hk_b = 0.0f, hk_c = 0.0f, hk_d = 0.0f;
        #pragma unroll
        for (int j = 0; j < K_DIM; j += 4) {
            hk_a += H_smem[(j+0)*V_DIM+tid]*smem_k0[j];   hk_b += H_smem[(j+1)*V_DIM+tid]*smem_k0[j+1];
            hk_c += H_smem[(j+2)*V_DIM+tid]*smem_k0[j+2]; hk_d += H_smem[(j+3)*V_DIM+tid]*smem_k0[j+3];
        }
        float hk_dot = (hk_a + hk_b) + (hk_c + hk_d);
        float v_new = (v_i - g_t * hk_dot) * bt_t;

        float qd_a = 0.0f, qd_b = 0.0f, qd_c = 0.0f, qd_d = 0.0f;
        #pragma unroll
        for (int j = 0; j < K_DIM; j += 4) {
            float h0 = g_t*H_smem[(j+0)*V_DIM+tid] + smem_k0[j]*v_new;
            float h1 = g_t*H_smem[(j+1)*V_DIM+tid] + smem_k0[j+1]*v_new;
            float h2 = g_t*H_smem[(j+2)*V_DIM+tid] + smem_k0[j+2]*v_new;
            float h3 = g_t*H_smem[(j+3)*V_DIM+tid] + smem_k0[j+3]*v_new;
            H_smem[(j+0)*V_DIM+tid]=h0; H_smem[(j+1)*V_DIM+tid]=h1;
            H_smem[(j+2)*V_DIM+tid]=h2; H_smem[(j+3)*V_DIM+tid]=h3;
            qd_a += h0*smem_q0[j]; qd_b += h1*smem_q0[j+1];
            qd_c += h2*smem_q0[j+2]; qd_d += h3*smem_q0[j+3];
        }
        float q_dot = (qd_a + qd_b) + (qd_c + qd_d);
        unsigned long long out_base = out_batch_off + (unsigned long long)vh * v_dim;
        output[out_base + (unsigned long long)t * num_v_heads * v_dim + tid] = __float2bfloat16(q_dot * inv_sqrt_d);
        __syncthreads();
    }

    #pragma unroll 4
    for (unsigned int i = tid; i < K_DIM * V_DIM; i += 128) {
        H_global[i] = H_smem[i];
    }
}
