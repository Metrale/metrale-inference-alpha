// SPDX-License-Identifier: MIT OR Apache-2.0

// 2026-09-25: Sequential Mamba-2 selective-state scan: mamba2_ssm_decode for one token, and
// mamba2_ssm_prefill and mamba2_ssm_prefill_persistent for a loop over seq_len tokens.
//
// Owner: gb10 kernels.
// Invariants:
// - Per head h of group g = h / (num_heads / n_groups), per token:
//     dt = clamp(softplus(dt_raw + dt_bias[h]), dt_min, dt_max);  dA = exp(-exp(A_log[h]) * dt)
//     H[p][s] <- dA * H[p][s] + dt * x[p] * B[g][s];  y[p] = sum_s H[p][s] C[g][s] + D[h] x[p]
//   mamba2_ssm_decode also clamps every updated H value to [-200, 200]; the two prefill
//   kernels do not.
// - H is FP32 [batch, num_heads, head_dim, state_size], state_size fastest, updated in
//   place. One block per (head, batch): blockIdx.x is the head, blockIdx.y the batch row.
// - x is [.., num_heads * head_dim], B and C are [.., n_groups * state_size] and dt_raw is
//   [.., num_heads], all BF16; the prefill kernels step between tokens by x_stride,
//   bc_stride, dt_stride and y_stride (BF16 elements).
// - mamba2_ssm_decode and mamba2_ssm_prefill run one thread per state column (block =
//   state_size). They require state_size to be a multiple of 32 and at most 128 (full-mask
//   warp shuffles into smem_warp[4][128]) and head_dim <= state_size (threads at or past
//   state_size exit before loading x).


#include <cuda_bf16.h>

#define BLOCK_SIZE 128

extern "C" __global__ void mamba2_ssm_decode(

    float* __restrict__ h_state,

    const __nv_bfloat16* __restrict__ x,

    const __nv_bfloat16* __restrict__ B_in,

    const __nv_bfloat16* __restrict__ C_in,

    const __nv_bfloat16* __restrict__ dt_raw,

    const float* __restrict__ A_log,
    const float* __restrict__ D_param,
    const float* __restrict__ dt_bias,

    __nv_bfloat16* __restrict__ output,

    unsigned int batch_size,
    unsigned int num_heads,
    unsigned int head_dim,
    unsigned int state_size,
    unsigned int n_groups,

    float dt_min,
    float dt_max
) {
    const unsigned int head = blockIdx.x;
    const unsigned int b = blockIdx.y;
    if (head >= num_heads || b >= batch_size) return;

    const unsigned int tid = threadIdx.x;
    if (tid >= state_size) return;


    const unsigned int heads_per_group = num_heads / n_groups;
    const unsigned int group = head / heads_per_group;


    float dt_val = (float)dt_raw[b * num_heads + head] + dt_bias[head];

    dt_val = (dt_val > 20.0f) ? dt_val : logf(1.0f + expf(dt_val));

    dt_val = fminf(fmaxf(dt_val, dt_min), dt_max);


    float neg_A = expf(A_log[head]);
    float dA = expf(-neg_A * dt_val);

    float D_val = D_param[head];


    float B_val = (float)B_in[b * n_groups * state_size + group * state_size + tid];
    float C_val = (float)C_in[b * n_groups * state_size + group * state_size + tid];


    float dtB = dt_val * B_val;


    float* H = h_state + ((unsigned long long)(b * num_heads + head) * head_dim * state_size);
    const __nv_bfloat16* x_ptr = x + (unsigned long long)(b * num_heads + head) * head_dim;




    __shared__ float smem_warp[4][128];
    __shared__ float smem_x[128];


    if (tid < head_dim) {
        smem_x[tid] = (float)x_ptr[tid];
    }
    __syncthreads();

    const unsigned int warp_id = tid / 32;
    const unsigned int lane = tid % 32;





    for (unsigned int hd = 0; hd < head_dim; hd++) {
        float x_hd = smem_x[hd];


        unsigned int idx = hd * state_size + tid;
        float h_val = H[idx];
        h_val = dA * h_val + x_hd * dtB;
        h_val = fminf(fmaxf(h_val, -200.0f), 200.0f);
        H[idx] = h_val;


        float y_partial = h_val * C_val;


        for (int offset = 16; offset >= 1; offset >>= 1)
            y_partial += __shfl_down_sync(0xFFFFFFFF, y_partial, offset);


        if (lane == 0) smem_warp[warp_id][hd] = y_partial;
    }

    __syncthreads();




    const unsigned int n_warps = (state_size + 31u) / 32u;
    if (tid < head_dim) {
        float y_val = 0.f;
        #pragma unroll
        for (unsigned int w = 0; w < 4; w++) {
            if (w < n_warps) y_val += smem_warp[w][tid];
        }

        y_val += D_val * smem_x[tid];
        output[(unsigned long long)(b * num_heads + head) * head_dim + tid] =
            __float2bfloat16(y_val);
    }
}

// 2026-09-25: The recurrence of mamba2_ssm_decode without the state clamp, looped over
// seq_len tokens. Block = state_size, as for the decode kernel.



extern "C" __global__ void mamba2_ssm_prefill(
    float* __restrict__ h_state,
    const __nv_bfloat16* __restrict__ x,
    const __nv_bfloat16* __restrict__ B_in,
    const __nv_bfloat16* __restrict__ C_in,
    const __nv_bfloat16* __restrict__ dt_raw,
    const float* __restrict__ A_log,
    const float* __restrict__ D_param,
    const float* __restrict__ dt_bias,
    __nv_bfloat16* __restrict__ output,
    unsigned int batch_size,
    unsigned int seq_len,
    unsigned int num_heads,
    unsigned int head_dim,
    unsigned int state_size,
    unsigned int n_groups,
    float dt_min,
    float dt_max,

    unsigned int x_stride,
    unsigned int bc_stride,
    unsigned int dt_stride,
    unsigned int y_stride
) {
    const unsigned int head = blockIdx.x;
    const unsigned int b = blockIdx.y;
    if (head >= num_heads || b >= batch_size) return;

    const unsigned int tid = threadIdx.x;
    if (tid >= state_size) return;

    const unsigned int heads_per_group = num_heads / n_groups;
    const unsigned int group = head / heads_per_group;
    float neg_A = expf(A_log[head]);
    float D_val = D_param[head];
    float dt_bias_val = dt_bias[head];

    float* H = h_state + ((unsigned long long)(b * num_heads + head) * head_dim * state_size);

    __shared__ float smem_warp[4][128];
    __shared__ float smem_x[128];

    const unsigned int warp_id = tid / 32;
    const unsigned int lane = tid % 32;

    for (unsigned int t = 0; t < seq_len; t++) {

        const __nv_bfloat16* x_t = x + (unsigned long long)t * x_stride
            + (unsigned long long)(b * num_heads + head) * head_dim;
        const __nv_bfloat16* B_t = B_in + (unsigned long long)t * bc_stride
            + b * n_groups * state_size + group * state_size;
        const __nv_bfloat16* C_t = C_in + (unsigned long long)t * bc_stride
            + b * n_groups * state_size + group * state_size;


        float dt_val = (float)dt_raw[(unsigned long long)t * dt_stride + b * num_heads + head]
                     + dt_bias_val;
        dt_val = (dt_val > 20.0f) ? dt_val : logf(1.0f + expf(dt_val));
        dt_val = fminf(fmaxf(dt_val, dt_min), dt_max);
        float dA = expf(-neg_A * dt_val);

        float B_val = (float)B_t[tid];
        float C_val = (float)C_t[tid];
        float dtB = dt_val * B_val;

        if (tid < head_dim) smem_x[tid] = (float)x_t[tid];
        __syncthreads();

        for (unsigned int hd = 0; hd < head_dim; hd++) {
            float x_hd = smem_x[hd];
            unsigned int idx = hd * state_size + tid;
            float h_val = H[idx];
            h_val = dA * h_val + x_hd * dtB;
            H[idx] = h_val;

            float y_partial = h_val * C_val;
            for (int offset = 16; offset >= 1; offset >>= 1)
                y_partial += __shfl_down_sync(0xFFFFFFFF, y_partial, offset);
            if (lane == 0) smem_warp[warp_id][hd] = y_partial;
        }
        __syncthreads();


        const unsigned int n_warps = (state_size + 31u) / 32u;
        if (tid < head_dim) {
            float y_val = 0.f;
            #pragma unroll
            for (unsigned int w = 0; w < 4; w++) {
                if (w < n_warps) y_val += smem_warp[w][tid];
            }
            y_val += D_val * smem_x[tid];
            output[(unsigned long long)t * y_stride
                + (unsigned long long)(b * num_heads + head) * head_dim + tid] =
                __float2bfloat16(y_val);
        }
        __syncthreads();
    }
}

// 2026-09-25: mamba2_ssm_prefill with each block's H slice held in shared memory for the
// whole token loop: loaded before the loop and stored after it. A block owns its
// (batch, head) slice alone, so the state update is the same arithmetic as
// mamba2_ssm_prefill; y is summed in a different order.
//
// SUB = 4 threads share each head_dim row (block = head_dim * SUB, a whole number of warps
// for the full-mask shuffle). Shared memory, in floats: sH head_dim * (state_size + 1),
// then x (head_dim), dt * B (state_size) and C (state_size), as sized by
// ops::mamba2_ssm_prefill_persistent. The +1 on each sH row avoids bank conflicts.









extern "C" __global__ void mamba2_ssm_prefill_persistent(
    float* __restrict__ h_state,
    const __nv_bfloat16* __restrict__ x,
    const __nv_bfloat16* __restrict__ B_in,
    const __nv_bfloat16* __restrict__ C_in,
    const __nv_bfloat16* __restrict__ dt_raw,
    const float* __restrict__ A_log,
    const float* __restrict__ D_param,
    const float* __restrict__ dt_bias,
    __nv_bfloat16* __restrict__ output,
    unsigned int batch_size,
    unsigned int seq_len,
    unsigned int num_heads,
    unsigned int head_dim,
    unsigned int state_size,
    unsigned int n_groups,
    float dt_min,
    float dt_max,
    unsigned int x_stride,
    unsigned int bc_stride,
    unsigned int dt_stride,
    unsigned int y_stride
) {







    const unsigned int SUB = 4u;

    const unsigned int head = blockIdx.x;
    const unsigned int b = blockIdx.y;
    if (head >= num_heads || b >= batch_size) return;

    const unsigned int tid = threadIdx.x;
    const unsigned int hd  = tid / SUB;
    const unsigned int sub = tid % SUB;

    const unsigned int heads_per_group = num_heads / n_groups;
    const unsigned int group = head / heads_per_group;
    const float neg_A = expf(A_log[head]);
    const float D_val = D_param[head];
    const float dt_bias_val = dt_bias[head];

    float* H = h_state + ((unsigned long long)(b * num_heads + head) * head_dim * state_size);






    extern __shared__ float smem[];
    const unsigned int h_stride = state_size + 1u;
    float* sH     = smem;
    float* smem_x = sH + (unsigned long long)head_dim * h_stride;
    float* smem_B = smem_x + head_dim;
    float* smem_C = smem_B + state_size;

    const unsigned int h_elems = head_dim * state_size;
    for (unsigned int i = tid; i < h_elems; i += blockDim.x) {
        unsigned int r = i / state_size;
        unsigned int c = i - r * state_size;
        sH[r * h_stride + c] = H[i];
    }
    __syncthreads();

    for (unsigned int t = 0; t < seq_len; t++) {
        const __nv_bfloat16* x_t = x + (unsigned long long)t * x_stride
            + (unsigned long long)(b * num_heads + head) * head_dim;
        const __nv_bfloat16* B_t = B_in + (unsigned long long)t * bc_stride
            + b * n_groups * state_size + group * state_size;
        const __nv_bfloat16* C_t = C_in + (unsigned long long)t * bc_stride
            + b * n_groups * state_size + group * state_size;

        float dt_val = (float)dt_raw[(unsigned long long)t * dt_stride + b * num_heads + head]
                     + dt_bias_val;
        dt_val = (dt_val > 20.0f) ? dt_val : logf(1.0f + expf(dt_val));
        dt_val = fminf(fmaxf(dt_val, dt_min), dt_max);
        const float dA = expf(-neg_A * dt_val);

        for (unsigned int i = tid; i < head_dim; i += blockDim.x)
            smem_x[i] = (float)x_t[i];
        for (unsigned int i = tid; i < state_size; i += blockDim.x) {
            smem_B[i] = dt_val * (float)B_t[i];
            smem_C[i] = (float)C_t[i];
        }
        __syncthreads();

        if (hd < head_dim) {
            const float x_hd = smem_x[hd];
            float* Hrow = sH + hd * h_stride;
            float y = 0.0f;
            for (unsigned int s = sub; s < state_size; s += SUB) {
                float h_val = dA * Hrow[s] + x_hd * smem_B[s];
                Hrow[s] = h_val;
                y += h_val * smem_C[s];
            }
            #pragma unroll
            for (unsigned int off = 1; off < SUB; off <<= 1)
                y += __shfl_down_sync(0xFFFFFFFFu, y, off);

            if (sub == 0u) {
                y += D_val * x_hd;
                output[(unsigned long long)t * y_stride
                    + (unsigned long long)(b * num_heads + head) * head_dim + hd] =
                    __float2bfloat16(y);
            }
        }
        __syncthreads();
    }

    for (unsigned int i = tid; i < h_elems; i += blockDim.x) {
        unsigned int r = i / state_size;
        unsigned int c = i - r * state_size;
        H[i] = sH[r * h_stride + c];
    }
}
