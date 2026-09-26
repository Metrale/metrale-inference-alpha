// SPDX-License-Identifier: MIT OR Apache-2.0

// 2026-09-25: Preprocessing for the gated-attention and GDN (linear-attention) layers: the
// QKVZ and Q/gate deinterleaves, the fused Q RMS norm (and MRoPE), and the BA-projection
// gate transforms (launchers: model-layers ops/ssm_preproc.rs).
//
// Owner: gb10 kernels.
// Invariants: none beyond the types.
//
// The gate transforms, for value head vh with a its A value and b its B value:
//   gate = exp(-exp(min(A_log[vh], 20)) * log(1 + exp(min(a + dt_bias[vh], 20))))
//   beta = 1 / (1 + exp(-b))





#include <cuda_bf16.h>
#include <math.h>

// 2026-09-25: Per token, the interleaved row of num_groups groups (Q_g, K_g of head_k_dim,
// then V_g, Z_g of vheads_per_group * head_v_dim) becomes [Q | K | V | Z]. One element per
// thread; grid (num_tokens, ceil(total/256)), block 256.











extern "C" __global__ void deinterleave_qkvz(
    const __nv_bfloat16* __restrict__ interleaved,
    __nv_bfloat16* __restrict__ output,
    unsigned int num_groups,
    unsigned int head_k_dim,
    unsigned int vheads_per_group,
    unsigned int head_v_dim
) {
    unsigned int token = blockIdx.x;
    unsigned int tid = blockIdx.y * blockDim.x + threadIdx.x;

    unsigned int v_group_size = vheads_per_group * head_v_dim;
    unsigned int group_dim = 2 * head_k_dim + 2 * v_group_size;
    unsigned int total = num_groups * group_dim;
    if (tid >= total) return;


    const __nv_bfloat16* in_tok = interleaved + (unsigned long long)token * total;
    __nv_bfloat16* out_tok = output + (unsigned long long)token * total;

    unsigned int g = tid / group_dim;
    unsigned int idx = tid % group_dim;


    unsigned int q_total = num_groups * head_k_dim;
    unsigned int k_total = num_groups * head_k_dim;
    unsigned int v_total = num_groups * v_group_size;

    unsigned int out_idx;
    if (idx < head_k_dim) {
        out_idx = g * head_k_dim + idx;
    } else if (idx < 2 * head_k_dim) {
        out_idx = q_total + g * head_k_dim + (idx - head_k_dim);
    } else if (idx < 2 * head_k_dim + v_group_size) {
        out_idx = q_total + k_total + g * v_group_size + (idx - 2 * head_k_dim);
    } else {
        out_idx = q_total + k_total + v_total + g * v_group_size
                + (idx - 2 * head_k_dim - v_group_size);
    }

    out_tok[out_idx] = in_tok[tid];
}

// 2026-09-25: In place per token: [Q_h0, G_h0, Q_h1, G_h1, ...] (each head_dim wide) becomes
// [Q_h0, Q_h1, ... | G_h0, G_h1, ...]. The row is staged in dynamic shared memory
// (num_heads * head_dim * 2 BF16). Grid num_tokens, block 256; rows are stride apart.









extern "C" __global__ void deinterleave_qg(
    __nv_bfloat16* __restrict__ data,
    unsigned int num_heads,
    unsigned int head_dim,
    unsigned int stride
) {
    extern __shared__ __nv_bfloat16 smem[];

    unsigned int total = num_heads * head_dim * 2;
    unsigned int tid = threadIdx.x;
    __nv_bfloat16* tok_data = data + (unsigned long long)blockIdx.x * stride;


    for (unsigned int i = tid; i < total; i += blockDim.x) {
        smem[i] = tok_data[i];
    }
    __syncthreads();


    unsigned int group_dim = 2 * head_dim;
    unsigned int q_total = num_heads * head_dim;

    for (unsigned int i = tid; i < total; i += blockDim.x) {
        unsigned int src;
        if (i < q_total) {
            unsigned int h = i / head_dim;
            unsigned int d = i % head_dim;
            src = h * group_dim + d;
        } else {
            unsigned int gi = i - q_total;
            unsigned int h = gi / head_dim;
            unsigned int d = gi % head_dim;
            src = h * group_dim + head_dim + d;
        }
        tok_data[i] = smem[src];
    }
}

// 2026-09-25: deinterleave_qg with Q written to q_out ([num_tokens, num_heads * head_dim])
// and the gate written back into data at offset num_heads * head_dim of each row.







extern "C" __global__ void deinterleave_qg_split(
    __nv_bfloat16* __restrict__ data,
    __nv_bfloat16* __restrict__ q_out,
    unsigned int num_heads,
    unsigned int head_dim,
    unsigned int stride
) {
    extern __shared__ __nv_bfloat16 smem[];

    unsigned int total = num_heads * head_dim * 2;
    unsigned int tid = threadIdx.x;
    unsigned int q_total = num_heads * head_dim;
    __nv_bfloat16* tok_data = data + (unsigned long long)blockIdx.x * stride;
    __nv_bfloat16* tok_q = q_out + (unsigned long long)blockIdx.x * q_total;


    for (unsigned int i = tid; i < total; i += blockDim.x) {
        smem[i] = tok_data[i];
    }
    __syncthreads();

    unsigned int group_dim = 2 * head_dim;


    for (unsigned int i = tid; i < total; i += blockDim.x) {
        unsigned int src;
        if (i < q_total) {
            unsigned int h = i / head_dim;
            unsigned int d = i % head_dim;
            src = h * group_dim + d;
            tok_q[i] = smem[src];
        } else {
            unsigned int gi = i - q_total;
            unsigned int h = gi / head_dim;
            unsigned int d = gi % head_dim;
            src = h * group_dim + head_dim + d;
            tok_data[i] = smem[src];
        }
    }
}

// 2026-09-25: deinterleave_qg_split with a per-head RMS norm on Q before it is written:
// q_out = x * rsqrt(mean(x^2) + eps) * (1 + w), w = q_norm_weight[dim] shared by every head.
// One warp per head (heads past the eighth loop); lane L takes dims L + 32e, so head_dim
// must be a multiple of 32.








extern "C" __global__ void deinterleave_qg_split_qnorm(
    __nv_bfloat16* __restrict__ data,
    __nv_bfloat16* __restrict__ q_out,
    const __nv_bfloat16* __restrict__ q_norm_weight,
    unsigned int num_heads,
    unsigned int head_dim,
    unsigned int stride,
    float eps
) {
    extern __shared__ __nv_bfloat16 smem[];

    unsigned int total = num_heads * head_dim * 2;
    unsigned int tid = threadIdx.x;
    unsigned int q_total = num_heads * head_dim;
    unsigned int group_dim = 2 * head_dim;
    __nv_bfloat16* tok_data = data + (unsigned long long)blockIdx.x * stride;
    __nv_bfloat16* tok_q = q_out + (unsigned long long)blockIdx.x * q_total;


    for (unsigned int i = tid; i < total; i += blockDim.x) {
        smem[i] = tok_data[i];
    }
    __syncthreads();


    for (unsigned int i = tid; i < q_total; i += blockDim.x) {
        unsigned int h = i / head_dim;
        unsigned int d = i % head_dim;
        unsigned int src = h * group_dim + head_dim + d;
        tok_data[q_total + i] = smem[src];
    }




    unsigned int warp_id = tid / 32;
    unsigned int lane = tid % 32;
    unsigned int num_warps = blockDim.x / 32;
    unsigned int elems_per_thread = head_dim / 32;

    for (unsigned int head = warp_id; head < num_heads; head += num_warps) {



        float sum_sq = 0.0f;
        for (unsigned int e = 0; e < elems_per_thread; e++) {
            unsigned int dim = lane + e * 32;
            unsigned int src_idx = head * group_dim + dim;
            float val = __bfloat162float(smem[src_idx]);
            sum_sq += val * val;
        }


        sum_sq = __shfl_xor_sync(0xFFFFFFFF, sum_sq, 16) + sum_sq;
        sum_sq = __shfl_xor_sync(0xFFFFFFFF, sum_sq, 8) + sum_sq;
        sum_sq = __shfl_xor_sync(0xFFFFFFFF, sum_sq, 4) + sum_sq;
        sum_sq = __shfl_xor_sync(0xFFFFFFFF, sum_sq, 2) + sum_sq;
        sum_sq = __shfl_xor_sync(0xFFFFFFFF, sum_sq, 1) + sum_sq;


        float rms = rsqrtf(sum_sq / (float)head_dim + eps);



        for (unsigned int e = 0; e < elems_per_thread; e++) {
            unsigned int dim = lane + e * 32;
            unsigned int src_idx = head * group_dim + dim;
            unsigned int out_idx = head * head_dim + dim;
            float val = __bfloat162float(smem[src_idx]);
            float w = __bfloat162float(q_norm_weight[dim]);
            tok_q[out_idx] = __float2bfloat16(val * rms * (1.0f + w));
        }
    }
}

// 2026-09-25: deinterleave_qg_split_qnorm, then interleaved MRoPE on the first rotary_dim
// dims of each normalized Q head (pair i takes pos_t, pos_h or pos_w by i % 3, as in
// rope_mrope_interleaved.cu) before the write to q_out. The normalized Q is staged in
// shared memory after the raw row. The prefill in qwen3_attention prefill/cache_skip.rs
// launches it and then rotates K with rope_forward_mrope_interleaved_k_only.


extern "C" __global__ void deinterleave_qg_split_qnorm_mrope(
    __nv_bfloat16* __restrict__ data,
    __nv_bfloat16* __restrict__ q_out,
    const __nv_bfloat16* __restrict__ q_norm_weight,
    const unsigned int* __restrict__ pos_t,
    const unsigned int* __restrict__ pos_h,
    const unsigned int* __restrict__ pos_w,
    unsigned int num_heads,
    unsigned int head_dim,
    unsigned int stride,
    unsigned int rotary_dim,
    float eps,
    float theta
) {
    extern __shared__ __nv_bfloat16 smem[];

    unsigned int total = num_heads * head_dim * 2;
    __nv_bfloat16* q_norm = smem + total;
    unsigned int tid = threadIdx.x;
    unsigned int q_total = num_heads * head_dim;
    unsigned int group_dim = 2 * head_dim;
    __nv_bfloat16* tok_data = data + (unsigned long long)blockIdx.x * stride;
    __nv_bfloat16* tok_q = q_out + (unsigned long long)blockIdx.x * q_total;

    for (unsigned int i = tid; i < total; i += blockDim.x) {
        smem[i] = tok_data[i];
    }
    __syncthreads();

    for (unsigned int i = tid; i < q_total; i += blockDim.x) {
        unsigned int h = i / head_dim;
        unsigned int d = i % head_dim;
        unsigned int src = h * group_dim + head_dim + d;
        tok_data[q_total + i] = smem[src];
    }

    unsigned int warp_id = tid / 32;
    unsigned int lane = tid % 32;
    unsigned int num_warps = blockDim.x / 32;
    unsigned int elems_per_thread = head_dim / 32;
    unsigned int half_rot = rotary_dim / 2;

    for (unsigned int head = warp_id; head < num_heads; head += num_warps) {
        float sum_sq = 0.0f;
        for (unsigned int e = 0; e < elems_per_thread; e++) {
            unsigned int dim = lane + e * 32;
            unsigned int src_idx = head * group_dim + dim;
            float val = __bfloat162float(smem[src_idx]);
            sum_sq += val * val;
        }

        sum_sq = __shfl_xor_sync(0xFFFFFFFF, sum_sq, 16) + sum_sq;
        sum_sq = __shfl_xor_sync(0xFFFFFFFF, sum_sq, 8) + sum_sq;
        sum_sq = __shfl_xor_sync(0xFFFFFFFF, sum_sq, 4) + sum_sq;
        sum_sq = __shfl_xor_sync(0xFFFFFFFF, sum_sq, 2) + sum_sq;
        sum_sq = __shfl_xor_sync(0xFFFFFFFF, sum_sq, 1) + sum_sq;

        float rms = rsqrtf(sum_sq / (float)head_dim + eps);

        for (unsigned int e = 0; e < elems_per_thread; e++) {
            unsigned int dim = lane + e * 32;
            unsigned int src_idx = head * group_dim + dim;
            float val = __bfloat162float(smem[src_idx]);
            float w = __bfloat162float(q_norm_weight[dim]);
            q_norm[head * head_dim + dim] = __float2bfloat16(val * rms * (1.0f + w));
        }
    }
    __syncthreads();

    for (unsigned int i = tid; i < q_total; i += blockDim.x) {
        unsigned int head = i / head_dim;
        unsigned int dim = i % head_dim;
        float out = __bfloat162float(q_norm[head * head_dim + dim]);

        if (dim < rotary_dim) {
            unsigned int pair_idx = dim < half_rot ? dim : dim - half_rot;
            bool is_d0 = dim < half_rot;
            unsigned int d0 = pair_idx;
            unsigned int d1 = pair_idx + half_rot;
            float x0 = __bfloat162float(q_norm[head * head_dim + d0]);
            float x1 = __bfloat162float(q_norm[head * head_dim + d1]);

            unsigned int section = pair_idx % 3;
            unsigned int abs_pos = section == 0 ? pos_t[blockIdx.x]
                : (section == 1 ? pos_h[blockIdx.x] : pos_w[blockIdx.x]);
            double freq_exp_d = (double)(2 * pair_idx) / (double)rotary_dim;
            float freq = (float)(1.0 / pow((double)theta, freq_exp_d));
            float angle = (float)abs_pos * freq;
            float cos_val = cosf(angle);
            float sin_val = sinf(angle);
            out = is_d0 ? (x0 * cos_val - x1 * sin_val)
                        : (x1 * cos_val + x0 * sin_val);
        }

        tok_q[i] = __float2bfloat16(out);
    }
}

// 2026-09-25: One token's BA projection (N rows of B, K inputs) with the gate transforms
// fused. Output n is a B value when w = n % (2 * vheads_per_group) < vheads_per_group, an A
// value otherwise, of value head (n / (2 * vheads_per_group)) * vheads_per_group +
// w % vheads_per_group; beta goes to beta_out[vh], gate to gate_out[vh]. Four outputs per
// 256-thread block (64 threads each), grid ceil(N/4). K must be a multiple of 8 (uint4
// loads; a tail is not read).










extern "C" __global__ void dense_gemv_ba_gates(
    const __nv_bfloat16* __restrict__ A,
    const __nv_bfloat16* __restrict__ B,
    const float* __restrict__ A_log,
    const float* __restrict__ dt_bias,
    float* __restrict__ gate_out,
    float* __restrict__ beta_out,
    unsigned int N,
    unsigned int K,
    unsigned int vheads_per_group
) {
    const unsigned int threads_per_out = 256 / 4;
    const unsigned int local_out = threadIdx.x / threads_per_out;
    const unsigned int lane = threadIdx.x % threads_per_out;

    const unsigned int n = blockIdx.x * 4 + local_out;
    if (n >= N) return;


    float acc = 0.0f;
    const unsigned int K_VEC = K / 8;
    const uint4* A_vec = (const uint4*)A;
    const uint4* B_vec = (const uint4*)(B + (unsigned long long)n * K);

    for (unsigned int kv = lane; kv < K_VEC; kv += threads_per_out) {
        uint4 a_data = A_vec[kv];
        uint4 b_data = B_vec[kv];
        const unsigned int a_raw[4] = {a_data.x, a_data.y, a_data.z, a_data.w};
        const unsigned int b_raw[4] = {b_data.x, b_data.y, b_data.z, b_data.w};
        #pragma unroll
        for (int i = 0; i < 4; i++) {
            __nv_bfloat16 a_lo, a_hi, b_lo, b_hi;
            *(unsigned short*)&a_lo = (unsigned short)(a_raw[i] & 0xFFFF);
            *(unsigned short*)&a_hi = (unsigned short)(a_raw[i] >> 16);
            *(unsigned short*)&b_lo = (unsigned short)(b_raw[i] & 0xFFFF);
            *(unsigned short*)&b_hi = (unsigned short)(b_raw[i] >> 16);
            acc += __bfloat162float(a_lo) * __bfloat162float(b_lo);
            acc += __bfloat162float(a_hi) * __bfloat162float(b_hi);
        }
    }


    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        acc += __shfl_down_sync(0xFFFFFFFF, acc, offset);
    }


    __shared__ float smem[4 * 2];
    const unsigned int warp_lane = threadIdx.x % 32;
    if (warp_lane == 0) {
        smem[local_out * 2 + (lane / 32)] = acc;
    }
    __syncthreads();


    if (lane == 0) {
        float result = smem[local_out * 2] + smem[local_out * 2 + 1];
        unsigned int group_dim_ba = 2 * vheads_per_group;
        unsigned int within_group = n % group_dim_ba;
        unsigned int group = n / group_dim_ba;

        if (within_group < vheads_per_group) {

            unsigned int vh = group * vheads_per_group + within_group;
            beta_out[vh] = 1.0f / (1.0f + __expf(-result));
        } else {

            unsigned int vh = group * vheads_per_group + (within_group - vheads_per_group);
            float a_log_val = A_log[vh];
            float dt_b = dt_bias[vh];
            float A_val = __expf(fminf(a_log_val, 20.0f));
            float dt = __logf(1.0f + __expf(fminf(result + dt_b, 20.0f)));
            gate_out[vh] = __expf(-A_val * dt);
        }
    }
}

// 2026-09-25: dense_gemv_ba_gates for M tokens (blockIdx.y is the token), with token rows
// K_stride apart. Both results go to gate_out, gate_stride floats per token: the gate at
// [vh], beta at [nv + vh]. Grid (ceil(N/4), M), block 256.















extern "C" __global__ void dense_gemm_ba_gates_prefill(
    const __nv_bfloat16* __restrict__ A,
    const __nv_bfloat16* __restrict__ B,
    const float* __restrict__ A_log,
    const float* __restrict__ dt_bias,
    float* __restrict__ gate_out,
    unsigned int M,
    unsigned int N,
    unsigned int K,
    unsigned int K_stride,
    unsigned int gate_stride,
    unsigned int nv,
    unsigned int vheads_per_group
) {
    const unsigned int threads_per_out = 256 / 4;
    const unsigned int local_out = threadIdx.x / threads_per_out;
    const unsigned int lane = threadIdx.x % threads_per_out;

    const unsigned int token = blockIdx.y;
    const unsigned int n = blockIdx.x * 4 + local_out;
    if (n >= N || token >= M) return;


    float acc = 0.0f;
    const unsigned int K_VEC = K / 8;
    const uint4* A_vec = (const uint4*)(A + (unsigned long long)token * K_stride);
    const uint4* B_vec = (const uint4*)(B + (unsigned long long)n * K);

    for (unsigned int kv = lane; kv < K_VEC; kv += threads_per_out) {
        uint4 a_data = A_vec[kv];
        uint4 b_data = B_vec[kv];
        const unsigned int a_raw[4] = {a_data.x, a_data.y, a_data.z, a_data.w};
        const unsigned int b_raw[4] = {b_data.x, b_data.y, b_data.z, b_data.w};
        #pragma unroll
        for (int i = 0; i < 4; i++) {
            __nv_bfloat16 a_lo, a_hi, b_lo, b_hi;
            *(unsigned short*)&a_lo = (unsigned short)(a_raw[i] & 0xFFFF);
            *(unsigned short*)&a_hi = (unsigned short)(a_raw[i] >> 16);
            *(unsigned short*)&b_lo = (unsigned short)(b_raw[i] & 0xFFFF);
            *(unsigned short*)&b_hi = (unsigned short)(b_raw[i] >> 16);
            acc += __bfloat162float(a_lo) * __bfloat162float(b_lo);
            acc += __bfloat162float(a_hi) * __bfloat162float(b_hi);
        }
    }


    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        acc += __shfl_down_sync(0xFFFFFFFF, acc, offset);
    }


    __shared__ float smem[4 * 2];
    const unsigned int warp_lane = threadIdx.x % 32;
    if (warp_lane == 0) {
        smem[local_out * 2 + (lane / 32)] = acc;
    }
    __syncthreads();


    if (lane == 0) {
        float result = smem[local_out * 2] + smem[local_out * 2 + 1];
        unsigned int group_dim_ba = 2 * vheads_per_group;
        unsigned int within_group = n % group_dim_ba;
        unsigned int group = n / group_dim_ba;

        float* gate_tok = gate_out + (unsigned long long)token * gate_stride;

        if (within_group < vheads_per_group) {

            unsigned int vh = group * vheads_per_group + within_group;
            gate_tok[nv + vh] = 1.0f / (1.0f + __expf(-result));
        } else {

            unsigned int vh = group * vheads_per_group + (within_group - vheads_per_group);
            float a_log_val = A_log[vh];
            float dt_b = dt_bias[vh];
            float A_val = __expf(fminf(a_log_val, 20.0f));
            float dt = __logf(1.0f + __expf(fminf(result + dt_b, 20.0f)));
            gate_tok[vh] = __expf(-A_val * dt);
        }
    }
}

// 2026-09-25: The gate transforms over a BF16 BA projection already in memory, ba_stride
// elements per token; group g holds the B values, then the A values, of its
// vheads_per_group heads. gate_out and beta_out rows are 2 * num_v_heads floats apart. One
// thread per value head; grid num_tokens, block num_v_heads.






extern "C" __global__ void compute_gdn_gates(
    const __nv_bfloat16* __restrict__ ba_interleaved,
    const float* __restrict__ A_log,
    const float* __restrict__ dt_bias,
    float* __restrict__ gate_out,
    float* __restrict__ beta_out,
    unsigned int num_v_heads,
    unsigned int num_groups,
    unsigned int vheads_per_group,
    unsigned int ba_stride
) {
    unsigned int token = blockIdx.x;
    unsigned int vh = threadIdx.x;
    if (vh >= num_v_heads) return;

    unsigned int group = vh / vheads_per_group;
    unsigned int local_idx = vh % vheads_per_group;
    unsigned int group_dim_ba = 2 * vheads_per_group;



    unsigned int out_stride = 2 * num_v_heads;
    const __nv_bfloat16* ba_tok = ba_interleaved + (unsigned long long)token * ba_stride;
    float* gate_tok = gate_out + (unsigned long long)token * out_stride;
    float* beta_tok = beta_out + (unsigned long long)token * out_stride;


    float b_raw = (float)ba_tok[group * group_dim_ba + local_idx];
    float a_raw = (float)ba_tok[group * group_dim_ba + vheads_per_group + local_idx];

    float a_log_val = A_log[vh];
    float dt_b = dt_bias[vh];


    float A_val = __expf(fminf(a_log_val, 20.0f));
    float dt = __logf(1.0f + __expf(fminf(a_raw + dt_b, 20.0f)));
    float g = -A_val * dt;
    gate_tok[vh] = __expf(g);


    beta_tok[vh] = 1.0f / (1.0f + __expf(-b_raw));
}
