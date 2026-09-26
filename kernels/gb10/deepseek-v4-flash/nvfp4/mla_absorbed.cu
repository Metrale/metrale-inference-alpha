// SPDX-License-Identifier: MIT OR Apache-2.0

// 2026-09-25: MLA helper kernels for the deepseek-v4-flash tree: the per-head batched GEMV that does
// Q absorption and V extraction, Q rope extract and writeback, K/V and cache-row assembly, and Q_final
// assembly.
//
// Owner: gb10 kernels (deepseek-v4-flash).
// Invariants: none beyond the types.









#include <cuda_bf16.h>

#define BLOCK_SIZE 256
#define N_PER_BLOCK 4
#define WARP_SIZE 32

// 2026-09-25: output[head * output_stride + n] = sum_k weight[head, n, k] * input[head * input_stride + k],
// BF16 in and out with FP32 accumulation; weight is [num_heads, N_out, K] contiguous. Grid
// (ceil(N_out / 8), num_heads), block 256 (ops::mla_batched_gemv); each 64-thread group owns two
// adjacent outputs. Inputs are read four values per 8-byte load, so the K % 4 trailing values are
// skipped and input_stride must keep every head's row 8-byte aligned.




extern "C" __global__ void mla_batched_gemv(
    const __nv_bfloat16* __restrict__ input,
    const __nv_bfloat16* __restrict__ weight,
    __nv_bfloat16* __restrict__ output,
    unsigned int N_out,
    unsigned int K,
    unsigned int input_stride,
    unsigned int output_stride
) {
    const unsigned int head = blockIdx.y;
    const unsigned int tid = threadIdx.x;


    const unsigned int threads_per_out = BLOCK_SIZE / N_PER_BLOCK;
    const unsigned int local_out = tid / threads_per_out;
    const unsigned int lane = tid % threads_per_out;

    const unsigned int n1 = blockIdx.x * (N_PER_BLOCK * 2) + local_out * 2;
    const unsigned int n2 = n1 + 1;
    if (n1 >= N_out) return;
    const bool have_n2 = (n2 < N_out);


    const __nv_bfloat16* A = input + (unsigned long long)head * input_stride;
    const __nv_bfloat16* B = weight + (unsigned long long)head * N_out * K;
    __nv_bfloat16* C = output + (unsigned long long)head * output_stride;

    const unsigned int K4 = K / 4;
    const unsigned long long* A64 = (const unsigned long long*)A;

    float acc1 = 0.0f, acc2 = 0.0f;

    for (unsigned int k4 = lane; k4 < K4; k4 += threads_per_out) {

        unsigned long long av = A64[k4];
        float a0, a1, a2, a3;
        unsigned int lo = (unsigned int)av;
        unsigned int hi = (unsigned int)(av >> 32);
        __nv_bfloat16 tmp;
        *(unsigned short*)&tmp = (unsigned short)(lo & 0xFFFF); a0 = __bfloat162float(tmp);
        *(unsigned short*)&tmp = (unsigned short)(lo >> 16);     a1 = __bfloat162float(tmp);
        *(unsigned short*)&tmp = (unsigned short)(hi & 0xFFFF); a2 = __bfloat162float(tmp);
        *(unsigned short*)&tmp = (unsigned short)(hi >> 16);     a3 = __bfloat162float(tmp);

        unsigned int base_k = k4 * 4;


        float w10 = __bfloat162float(B[n1 * K + base_k]);
        float w11 = __bfloat162float(B[n1 * K + base_k + 1]);
        float w12 = __bfloat162float(B[n1 * K + base_k + 2]);
        float w13 = __bfloat162float(B[n1 * K + base_k + 3]);
        acc1 += a0 * w10 + a1 * w11 + a2 * w12 + a3 * w13;

        if (have_n2) {
            float w20 = __bfloat162float(B[n2 * K + base_k]);
            float w21 = __bfloat162float(B[n2 * K + base_k + 1]);
            float w22 = __bfloat162float(B[n2 * K + base_k + 2]);
            float w23 = __bfloat162float(B[n2 * K + base_k + 3]);
            acc2 += a0 * w20 + a1 * w21 + a2 * w22 + a3 * w23;
        }
    }


    #pragma unroll
    for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1) {
        acc1 += __shfl_down_sync(0xFFFFFFFF, acc1, offset);
        if (have_n2) acc2 += __shfl_down_sync(0xFFFFFFFF, acc2, offset);
    }


    __shared__ float s_partial[N_PER_BLOCK * 2][2];
    unsigned int warp_in_out = (tid % threads_per_out) / WARP_SIZE;
    unsigned int lane_in_warp = tid % WARP_SIZE;
    if (lane_in_warp == 0) {
        s_partial[local_out * 2][warp_in_out] = acc1;
        if (have_n2) s_partial[local_out * 2 + 1][warp_in_out] = acc2;
    }
    __syncthreads();


    unsigned int warps_per_out = threads_per_out / WARP_SIZE;
    if (lane_in_warp == 0 && warp_in_out == 0) {
        float sum1 = 0.0f;
        for (unsigned int w = 0; w < warps_per_out; w++) sum1 += s_partial[local_out * 2][w];
        C[n1] = __float2bfloat16(sum1);

        if (have_n2) {
            float sum2 = 0.0f;
            for (unsigned int w = 0; w < warps_per_out; w++) sum2 += s_partial[local_out * 2 + 1][w];
            C[n2] = __float2bfloat16(sum2);
        }
    }
}







// 2026-09-25: Copies each head's rope part of q_full ([nq, hd], offset nope) into q_absorbed_buf
// ([nq, mla_cache_dim], offset kv_lora) and into q_rope_contiguous ([nq, rope]). One block strides
// over all nq * rope values (ops::mla_q_rope_scatter launches grid 1, block 256).



extern "C" __global__ void mla_q_rope_scatter(
    const __nv_bfloat16* __restrict__ q_full,
    __nv_bfloat16* __restrict__ q_absorbed_buf,
    __nv_bfloat16* __restrict__ q_rope_contiguous,
    unsigned int nq,
    unsigned int hd,
    unsigned int nope,
    unsigned int rope,
    unsigned int kv_lora,
    unsigned int mla_cache_dim
) {
    unsigned int total = nq * rope;
    for (unsigned int idx = threadIdx.x; idx < total; idx += blockDim.x) {
        unsigned int head = idx / rope;
        unsigned int r = idx % rope;

        __nv_bfloat16 val = q_full[head * hd + nope + r];

        q_absorbed_buf[head * mla_cache_dim + kv_lora + r] = val;
        q_rope_contiguous[head * rope + r] = val;
    }
}

// 2026-09-25: Writes q_rope_direct [nq, rope] into q_absorbed_buf at offset kv_lora of each
// mla_cache_dim row; one block strides over all nq * rope values.

extern "C" __global__ void mla_q_rope_writeback(
    const __nv_bfloat16* __restrict__ q_rope_direct,
    __nv_bfloat16* __restrict__ q_absorbed_buf,
    unsigned int nq,
    unsigned int rope,
    unsigned int kv_lora,
    unsigned int mla_cache_dim
) {
    unsigned int total = nq * rope;
    for (unsigned int idx = threadIdx.x; idx < total; idx += blockDim.x) {
        unsigned int head = idx / rope;
        unsigned int r = idx % rope;
        q_absorbed_buf[head * mla_cache_dim + kv_lora + r] = q_rope_direct[head * rope + r];
    }
}





// 2026-09-25: q_full [N, q_dim] with head h's rope at h * hd + nope -> q_rope_out [N, nq * rope],
// contiguous. Grid-stride loop; ops::mla_q_rope_extract_batched launches ceil(N * nq * rope / 256) x 256.

extern "C" __global__ void mla_q_rope_extract_batched(
    const __nv_bfloat16* __restrict__ q_full,
    __nv_bfloat16* __restrict__ q_rope_out,
    unsigned int num_tokens,
    unsigned int nq,
    unsigned int hd,
    unsigned int nope,
    unsigned int rope,
    unsigned int q_dim
) {
    unsigned int total = num_tokens * nq * rope;
    for (unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x; idx < total; idx += gridDim.x * blockDim.x) {
        unsigned int t = idx / (nq * rope);
        unsigned int rem = idx % (nq * rope);
        unsigned int head = rem / rope;
        unsigned int r = rem % rope;
        q_rope_out[t * nq * rope + head * rope + r] =
            q_full[t * q_dim + head * hd + nope + r];
    }
}

// 2026-09-25: The inverse of mla_q_rope_extract_batched, with the same indexing and launch shape.


extern "C" __global__ void mla_q_rope_writeback_batched(
    const __nv_bfloat16* __restrict__ q_rope_in,
    __nv_bfloat16* __restrict__ q_full,
    unsigned int num_tokens,
    unsigned int nq,
    unsigned int hd,
    unsigned int nope,
    unsigned int rope,
    unsigned int q_dim
) {
    unsigned int total = num_tokens * nq * rope;
    for (unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x; idx < total; idx += gridDim.x * blockDim.x) {
        unsigned int t = idx / (nq * rope);
        unsigned int rem = idx % (nq * rope);
        unsigned int head = rem / rope;
        unsigned int r = rem % rope;
        q_full[t * q_dim + head * hd + nope + r] =
            q_rope_in[t * nq * rope + head * rope + r];
    }
}

// 2026-09-25: Per token (blockIdx.x): blockIdx.y 0 writes K [nkv, hd] = [k_nope from kv_expanded |
// k_rope_buf, shared by every KV head]; blockIdx.y 1 writes V [nkv, v_dim] from offset nope of each
// head's (nope + v_dim) block of kv_expanded. Grid (N, 2), block 256 (ops::mla_kv_assemble_batched).




extern "C" __global__ void mla_kv_assemble_batched(
    const __nv_bfloat16* __restrict__ kv_expanded,
    const __nv_bfloat16* __restrict__ k_rope_buf,
    __nv_bfloat16* __restrict__ k_out,
    __nv_bfloat16* __restrict__ v_out,
    unsigned int nkv,
    unsigned int nope,
    unsigned int v_dim,
    unsigned int rope,
    unsigned int hd,
    unsigned int kv_expanded_stride
) {
    unsigned int t = blockIdx.x;

    if (blockIdx.y == 0) {

        unsigned int k_total = nkv * hd;
        for (unsigned int idx = threadIdx.x; idx < k_total; idx += blockDim.x) {
            unsigned int head = idx / hd;
            unsigned int dim = idx % hd;
            __nv_bfloat16 val;
            if (dim < nope) {

                val = kv_expanded[(unsigned long long)t * kv_expanded_stride + head * (nope + v_dim) + dim];
            } else {

                val = k_rope_buf[(unsigned long long)t * rope + (dim - nope)];
            }
            k_out[(unsigned long long)t * nkv * hd + idx] = val;
        }
    } else {

        unsigned int v_total = nkv * v_dim;
        for (unsigned int idx = threadIdx.x; idx < v_total; idx += blockDim.x) {
            unsigned int head = idx / v_dim;
            unsigned int dim = idx % v_dim;

            v_out[(unsigned long long)t * nkv * v_dim + idx] =
                kv_expanded[(unsigned long long)t * kv_expanded_stride + head * (nope + v_dim) + nope + dim];
        }
    }
}

// 2026-09-25: Per token, K row = [kv_latent | k_rope] and V row = [kv_latent | k_rope], each
// mla_cache_dim = kv_lora + rope wide. V's rope tail is K's because DeepSeek-V4 attends one kv tensor
// as both key and value. Grid (N), block max(mla_cache_dim, 256) (ops::mla_cache_assemble_batched).


extern "C" __global__ void mla_cache_assemble_batched(
    const __nv_bfloat16* __restrict__ kv_latent,
    const __nv_bfloat16* __restrict__ k_rope,
    __nv_bfloat16* __restrict__ k_cache,
    __nv_bfloat16* __restrict__ v_cache,
    unsigned int kv_lora,
    unsigned int rope,
    unsigned int mla_cache_dim
) {
    unsigned int t = blockIdx.x;
    unsigned long long k_off = (unsigned long long)t * mla_cache_dim;
    unsigned long long lat_off = (unsigned long long)t * kv_lora;
    unsigned long long rope_off = (unsigned long long)t * rope;

    for (unsigned int idx = threadIdx.x; idx < mla_cache_dim; idx += blockDim.x) {
        if (idx < kv_lora) {
            __nv_bfloat16 val = kv_latent[lat_off + idx];
            k_cache[k_off + idx] = val;
            v_cache[k_off + idx] = val;
        } else {
            unsigned int r = idx - kv_lora;






            __nv_bfloat16 rope_val = k_rope[rope_off + r];
            k_cache[k_off + idx] = rope_val;
            v_cache[k_off + idx] = rope_val;
        }
    }
}

// 2026-09-25: q_final [N, nq * mla_cache_dim] holds, per token and head, [q_absorbed (kv_lora) | q_rope
// (rope)] from q_absorbed [N, nq * kv_lora] and q_rope [N, nq * rope]. Grid-stride; the launcher uses
// ceil(N * nq * mla_cache_dim / 256) x 256.



extern "C" __global__ void mla_q_final_assemble_batched(
    const __nv_bfloat16* __restrict__ q_absorbed,
    const __nv_bfloat16* __restrict__ q_rope,
    __nv_bfloat16* __restrict__ q_final,
    unsigned int num_tokens,
    unsigned int nq,
    unsigned int kv_lora,
    unsigned int rope,
    unsigned int mla_cache_dim
) {
    unsigned int total = num_tokens * nq * mla_cache_dim;
    for (unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x; idx < total; idx += gridDim.x * blockDim.x) {
        unsigned int t = idx / (nq * mla_cache_dim);
        unsigned int rem = idx % (nq * mla_cache_dim);
        unsigned int head = rem / mla_cache_dim;
        unsigned int d = rem % mla_cache_dim;
        if (d < kv_lora) {
            q_final[idx] = q_absorbed[t * nq * kv_lora + head * kv_lora + d];
        } else {
            q_final[idx] = q_rope[t * nq * rope + head * rope + (d - kv_lora)];
        }
    }
}





// 2026-09-25: One token: K entry = [kv_latent | k_rope], V entry = [kv_latent | zeros], mla_cache_dim
// wide; one thread per element (ops::mla_cache_assemble launches max(mla_cache_dim, 256) threads).
extern "C" __global__ void mla_cache_assemble(
    const __nv_bfloat16* __restrict__ kv_latent,
    const __nv_bfloat16* __restrict__ k_rope,
    __nv_bfloat16* __restrict__ k_cache_entry,
    __nv_bfloat16* __restrict__ v_cache_entry,
    unsigned int kv_lora,
    unsigned int rope,
    unsigned int mla_cache_dim
) {
    unsigned int idx = threadIdx.x;

    if (idx < kv_lora) {
        k_cache_entry[idx] = kv_latent[idx];
        v_cache_entry[idx] = kv_latent[idx];
    } else if (idx < mla_cache_dim) {
        unsigned int r = idx - kv_lora;
        k_cache_entry[idx] = (r < rope) ? k_rope[r] : __float2bfloat16(0.0f);
        v_cache_entry[idx] = __float2bfloat16(0.0f);
    }
}
