// SPDX-License-Identifier: MIT OR Apache-2.0

// 2026-09-25: Paged decode attention with a turbo4 K pool and a turbo8 V pool.
// HDIM is 128 unless the model's KERNEL.toml passes -DHDIM.
//
// Both pools hold WHT-rotated values. The kernel applies no WHT: Q must be in
// the basis K and V were written in, which the caller arranges
// (write_kv_cache.rs, attention_forward.rs). Each pool has its own
// block stride and data-section size. Per block, token-major:
//   K: data   block_size * num_kv_heads * head_dim / 2 bytes, two 4-bit
//             indices into the 16-entry `k_lut` per byte, low nibble first
//      scale  block_size * num_kv_heads * head_dim / 16 bytes, FP8 E4M3
//   V: data   block_size * num_kv_heads * head_dim bytes, FP8 E4M3
//      scale  block_size * num_kv_heads * (head_dim / 16) * 2 bytes, BF16
// One scale per 16-element group. Launched as grid (num_q_heads, num_seqs, 1)
// x 256 threads (ops::paged_decode_attn_turbo4k_turbo8v); with
// sliding_window > 0 only the last sliding_window positions are attended.
//
// Owner: gb10 kernels.
// Invariants:
// - Assumes head_dim == HDIM; a sequence with seq_lens[s] == 0 writes no output.

#include <cuda_bf16.h>
// 2026-09-25: A V row whose exp(score - running max) is at most this is skipped.
#ifndef TQ_PLUS_SPARSE_V_THRESHOLD
#define TQ_PLUS_SPARSE_V_THRESHOLD 1e-3f
#endif

#include <cuda_fp8.h>

#define WARP_SIZE 32
#ifndef HDIM
#define HDIM 128
#endif
#define VEC_BF16 (HDIM / WARP_SIZE)
#define VEC_U32  (HDIM / (WARP_SIZE * 2))
#define NUM_WARPS 8
#define BC 4
#define NVFP4_GROUP_SIZE 16



__device__ __forceinline__ void unpack2_bf16_asym(unsigned int packed, float& v0, float& v1) {
    v0 = __bfloat162float(__ushort_as_bfloat16((unsigned short)(packed & 0xFFFF)));
    v1 = __bfloat162float(__ushort_as_bfloat16((unsigned short)(packed >> 16)));
}

__device__ __forceinline__ float fp8e4m3_to_f32_asym(__nv_fp8_storage_t b) {
    return __half2float(__nv_cvt_fp8_to_halfraw(b, __NV_E4M3));
}

// 2026-09-25: K: HDIM / 32 codebook values from 4-bit indices, times the FP8 group scale.
__device__ __forceinline__ void turbo4_dequant_k(
    const unsigned char* data_ptr,
    const unsigned char* scale_ptr,
    const __half* lut,
    float* out
) {
    float gs = fp8e4m3_to_f32_asym((__nv_fp8_storage_t)*scale_ptr);
#if VEC_BF16 == 8
    unsigned int pk = *(const unsigned int*)data_ptr;
    out[0] = __half2float(lut[(pk)       & 0xF]) * gs;
    out[1] = __half2float(lut[(pk >> 4)  & 0xF]) * gs;
    out[2] = __half2float(lut[(pk >> 8)  & 0xF]) * gs;
    out[3] = __half2float(lut[(pk >> 12) & 0xF]) * gs;
    out[4] = __half2float(lut[(pk >> 16) & 0xF]) * gs;
    out[5] = __half2float(lut[(pk >> 20) & 0xF]) * gs;
    out[6] = __half2float(lut[(pk >> 24) & 0xF]) * gs;
    out[7] = __half2float(lut[pk >> 28])         * gs;
#elif VEC_BF16 == 4
    unsigned short pk = *(const unsigned short*)data_ptr;
    out[0] = __half2float(lut[(pk)       & 0xF]) * gs;
    out[1] = __half2float(lut[(pk >> 4)  & 0xF]) * gs;
    out[2] = __half2float(lut[(pk >> 8)  & 0xF]) * gs;
    out[3] = __half2float(lut[pk >> 12])         * gs;
#else
    #error "Unsupported VEC_BF16 (need 4 or 8)"
#endif
}

// 2026-09-25: V: HDIM / 32 FP8 E4M3 values times the BF16 group scale.


__device__ __forceinline__ void turbo8_dequant_v(
    const unsigned char* data_ptr,
    const unsigned char* scale_ptr,
    float* out
) {
    float gs = __bfloat162float(*(const __nv_bfloat16*)scale_ptr);
#if VEC_BF16 == 8
    unsigned long long pk8 = *(const unsigned long long*)data_ptr;
    out[0] = fp8e4m3_to_f32_asym((__nv_fp8_storage_t)(pk8 & 0xFF)) * gs;
    out[1] = fp8e4m3_to_f32_asym((__nv_fp8_storage_t)((pk8 >> 8) & 0xFF)) * gs;
    out[2] = fp8e4m3_to_f32_asym((__nv_fp8_storage_t)((pk8 >> 16) & 0xFF)) * gs;
    out[3] = fp8e4m3_to_f32_asym((__nv_fp8_storage_t)((pk8 >> 24) & 0xFF)) * gs;
    out[4] = fp8e4m3_to_f32_asym((__nv_fp8_storage_t)((pk8 >> 32) & 0xFF)) * gs;
    out[5] = fp8e4m3_to_f32_asym((__nv_fp8_storage_t)((pk8 >> 40) & 0xFF)) * gs;
    out[6] = fp8e4m3_to_f32_asym((__nv_fp8_storage_t)((pk8 >> 48) & 0xFF)) * gs;
    out[7] = fp8e4m3_to_f32_asym((__nv_fp8_storage_t)(pk8 >> 56)) * gs;
#elif VEC_BF16 == 4
    unsigned int pk4 = *(const unsigned int*)data_ptr;
    out[0] = fp8e4m3_to_f32_asym((__nv_fp8_storage_t)(pk4 & 0xFF)) * gs;
    out[1] = fp8e4m3_to_f32_asym((__nv_fp8_storage_t)((pk4 >> 8) & 0xFF)) * gs;
    out[2] = fp8e4m3_to_f32_asym((__nv_fp8_storage_t)((pk4 >> 16) & 0xFF)) * gs;
    out[3] = fp8e4m3_to_f32_asym((__nv_fp8_storage_t)(pk4 >> 24)) * gs;
#else
    #error "Unsupported VEC_BF16 (need 4 or 8)"
#endif
}




extern "C" __global__ void paged_decode_attn_turbo4k_turbo8v(
    const __nv_bfloat16* __restrict__ Q,
    const unsigned char* __restrict__ K_cache,
    const unsigned char* __restrict__ V_cache,
    __nv_bfloat16* __restrict__ O,
    const int* __restrict__ block_tables,
    const int* __restrict__ seq_lens,
    const unsigned int max_blocks_per_seq,
    const unsigned int num_q_heads,
    const unsigned int num_kv_heads,
    const unsigned int head_dim,
    const unsigned int block_size,
    const float inv_sqrt_d,
    const unsigned int q_stride,
    const unsigned long long k_block_stride_bytes,
    const unsigned long long k_data_section_bytes,
    const unsigned long long v_block_stride_bytes,
    const unsigned long long v_data_section_bytes,
    const unsigned int sliding_window
) {
    const unsigned int q_head = blockIdx.x;
    const unsigned int seq_idx = blockIdx.y;
    const unsigned int tid = threadIdx.x;
    const unsigned int warp_id = tid / WARP_SIZE;
    const unsigned int lane_id = tid % WARP_SIZE;

    if (q_head >= num_q_heads) return;

    const unsigned int seq_len = (unsigned int)seq_lens[seq_idx];
    if (seq_len == 0) return;

    // 2026-09-25: The turbo4 codebook, with the values reshape_and_cache_turbo.cu encodes against.
    __shared__ __half k_lut[16];
    if (tid < 16) {
        const float lut_init[16] = {
            -2.7326f, -2.0690f, -1.6180f, -1.2562f, -0.9423f, -0.6568f, -0.3880f, -0.1284f,
             0.1284f,  0.3880f,  0.6568f,  0.9423f,  1.2562f,  1.6180f,  2.0690f,  2.7326f
        };
        k_lut[tid] = __float2half(lut_init[tid]);
    }
    __syncthreads();

    const unsigned int window_start =
        (sliding_window > 0 && seq_len > sliding_window) ? (seq_len - sliding_window) : 0u;

    const unsigned int gqa_ratio = num_q_heads / num_kv_heads;
    const unsigned int kv_head = q_head / gqa_ratio;
    const unsigned int vec_offset_bf16 = lane_id * VEC_BF16;


    const unsigned int k_head_data_bytes  = head_dim / 2;
    const unsigned int k_head_scale_bytes = head_dim / NVFP4_GROUP_SIZE;
    const unsigned int k_token_data_stride  = num_kv_heads * k_head_data_bytes;
    const unsigned int k_token_scale_stride = num_kv_heads * k_head_scale_bytes;
    const unsigned int k_data_offset_per_thread  = kv_head * k_head_data_bytes  + lane_id * (VEC_BF16 / 2);
    const unsigned int k_scale_offset_per_thread = kv_head * k_head_scale_bytes + (lane_id * VEC_BF16 / NVFP4_GROUP_SIZE);


    const unsigned int v_head_data_bytes  = head_dim;
    const unsigned int v_head_scale_bytes = (head_dim / NVFP4_GROUP_SIZE) * 2;
    const unsigned int v_token_data_stride  = num_kv_heads * v_head_data_bytes;
    const unsigned int v_token_scale_stride = num_kv_heads * v_head_scale_bytes;
    const unsigned int v_data_offset_per_thread  = kv_head * v_head_data_bytes  + lane_id * VEC_BF16;
    const unsigned int v_scale_offset_per_thread = kv_head * v_head_scale_bytes
                                                 + (lane_id * VEC_BF16 / NVFP4_GROUP_SIZE) * 2;

    const int* my_block_table = block_tables + seq_idx * max_blocks_per_seq;


    const unsigned int* q32 = (const unsigned int*)(Q + (unsigned long long)seq_idx * q_stride
                                                       + (unsigned long long)q_head * head_dim + vec_offset_bf16);
    float q_reg[VEC_BF16];
    #pragma unroll
    for (int i = 0; i < VEC_U32; i++) {
        unpack2_bf16_asym(q32[i], q_reg[2*i], q_reg[2*i+1]);
    }

    const unsigned int attended = seq_len - window_start;
    unsigned int chunk_size = (attended + NUM_WARPS - 1) / NUM_WARPS;
    unsigned int my_start = window_start + warp_id * chunk_size;
    unsigned int my_end = my_start + chunk_size;
    if (my_end > seq_len) my_end = seq_len;
    if (my_start > seq_len) my_start = seq_len;

    float m = -1e30f;
    float l = 0.0f;
    float o_reg[VEC_BF16];
    #pragma unroll
    for (int i = 0; i < VEC_BF16; i++) o_reg[i] = 0.0f;

    unsigned int pos = my_start;
    while (pos < my_end) {
        unsigned int logical_block = pos / block_size;
        unsigned int block_offset = pos % block_size;
        unsigned int remaining_in_block = block_size - block_offset;
        unsigned int remaining_total = my_end - pos;
        unsigned int batch_count = remaining_in_block < remaining_total ? remaining_in_block : remaining_total;

        unsigned int physical_block = (unsigned int)my_block_table[logical_block];

        const unsigned char* k_block = K_cache
            + (unsigned long long)physical_block * k_block_stride_bytes;
        const unsigned char* v_block = V_cache
            + (unsigned long long)physical_block * v_block_stride_bytes;

        unsigned int processed = 0;
        unsigned int aligned_count = (batch_count / BC) * BC;


        for (; processed < aligned_count; processed += BC) {
            float k_vals[BC][VEC_BF16];
            #pragma unroll
            for (int b = 0; b < BC; b++) {
                unsigned int p = block_offset + processed + b;
                const unsigned char* kd = k_block + p * k_token_data_stride + k_data_offset_per_thread;
                const unsigned char* ks = k_block + k_data_section_bytes
                                        + p * k_token_scale_stride + k_scale_offset_per_thread;
                turbo4_dequant_k(kd, ks, k_lut, k_vals[b]);
            }

            float scores[BC];
            #pragma unroll
            for (int b = 0; b < BC; b++) {
                float dot = 0.0f;
                #pragma unroll
                for (int i = 0; i < VEC_BF16; i++)
                    dot += q_reg[i] * k_vals[b][i];
                #pragma unroll
                for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1)
                    dot += __shfl_xor_sync(0xffffffff, dot, offset);
                scores[b] = dot * inv_sqrt_d;
            }

            float m_new = m;
            #pragma unroll
            for (int b = 0; b < BC; b++)
                m_new = fmaxf(m_new, scores[b]);

            float exp_old = __expf(m - m_new);
            #pragma unroll
            for (int i = 0; i < VEC_BF16; i++)
                o_reg[i] *= exp_old;
            l *= exp_old;

            float exp_factors[BC];
            #pragma unroll
            for (int b = 0; b < BC; b++) {
                exp_factors[b] = __expf(scores[b] - m_new);
                l += exp_factors[b];
            }
            m = m_new;


            float v_vals[BC][VEC_BF16];
            #pragma unroll
            for (int b = 0; b < BC; b++) {
                if (exp_factors[b] > TQ_PLUS_SPARSE_V_THRESHOLD) {
                    unsigned int p = block_offset + processed + b;
                    const unsigned char* vd = v_block + p * v_token_data_stride  + v_data_offset_per_thread;
                    const unsigned char* vs = v_block + v_data_section_bytes
                                            + p * v_token_scale_stride + v_scale_offset_per_thread;
                    turbo8_dequant_v(vd, vs, v_vals[b]);
                } else {
                    #pragma unroll
                    for (int i = 0; i < VEC_BF16; i++) v_vals[b][i] = 0.0f;
                }
            }

            #pragma unroll
            for (int b = 0; b < BC; b++) {
                float ef = exp_factors[b];
                #pragma unroll
                for (int i = 0; i < VEC_BF16; i++)
                    o_reg[i] += ef * v_vals[b][i];
            }
        }


        for (; processed < batch_count; processed++) {
            unsigned int p = block_offset + processed;

            const unsigned char* kd = k_block + p * k_token_data_stride + k_data_offset_per_thread;
            const unsigned char* ks = k_block + k_data_section_bytes
                                    + p * k_token_scale_stride + k_scale_offset_per_thread;
            float k_tmp[VEC_BF16];
            turbo4_dequant_k(kd, ks, k_lut, k_tmp);

            float dot = 0.0f;
            #pragma unroll
            for (int i = 0; i < VEC_BF16; i++)
                dot += q_reg[i] * k_tmp[i];
            #pragma unroll
            for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1)
                dot += __shfl_xor_sync(0xffffffff, dot, offset);

            float score = dot * inv_sqrt_d;
            float m_new = fmaxf(m, score);
            float exp_old = __expf(m - m_new);
            float exp_new = __expf(score - m_new);
            l = l * exp_old + exp_new;

            float v_tmp[VEC_BF16] = {0};
            if (exp_new > TQ_PLUS_SPARSE_V_THRESHOLD) {
                const unsigned char* vd = v_block + p * v_token_data_stride  + v_data_offset_per_thread;
                const unsigned char* vs = v_block + v_data_section_bytes
                                        + p * v_token_scale_stride + v_scale_offset_per_thread;
                turbo8_dequant_v(vd, vs, v_tmp);
            }

            #pragma unroll
            for (int i = 0; i < VEC_BF16; i++)
                o_reg[i] = o_reg[i] * exp_old + exp_new * v_tmp[i];
            m = m_new;
        }

        pos += batch_count;
    }


    __shared__ float smem_m[NUM_WARPS];
    __shared__ float smem_l[NUM_WARPS];
    __shared__ float smem_o[NUM_WARPS][HDIM];

    if (lane_id == 0) {
        smem_m[warp_id] = m;
        smem_l[warp_id] = l;
    }
    #pragma unroll
    for (int i = 0; i < VEC_BF16; i++) {
        smem_o[warp_id][vec_offset_bf16 + i] = o_reg[i];
    }
    __syncthreads();

    #pragma unroll
    for (int stride = NUM_WARPS / 2; stride > 0; stride >>= 1) {
        if (warp_id < (unsigned int)stride) {
            unsigned int other = warp_id + stride;
            float lw = smem_l[other];
            if (lw > 0.0f) {
                float mw = smem_m[other];
                float my_m = smem_m[warp_id];
                float my_l = smem_l[warp_id];
                float m_new = fmaxf(my_m, mw);
                float scale_me = __expf(my_m - m_new);
                float scale_w = __expf(mw - m_new);
                smem_l[warp_id] = my_l * scale_me + lw * scale_w;
                smem_m[warp_id] = m_new;
                #pragma unroll
                for (int i = 0; i < VEC_BF16; i++) {
                    smem_o[warp_id][vec_offset_bf16 + i] =
                        smem_o[warp_id][vec_offset_bf16 + i] * scale_me +
                        smem_o[other][vec_offset_bf16 + i] * scale_w;
                }
            }
        }
        __syncthreads();
    }

    if (warp_id == 0) {
        float final_l = smem_l[0];
        float inv_l = (final_l > 0.0f) ? (1.0f / final_l) : 0.0f;
        unsigned int* o32 = (unsigned int*)(O + (unsigned long long)seq_idx * num_q_heads * head_dim
                                              + (unsigned long long)q_head * head_dim + vec_offset_bf16);
        #pragma unroll
        for (int i = 0; i < VEC_U32; i++) {
            float v0 = smem_o[0][vec_offset_bf16 + 2*i]     * inv_l;
            float v1 = smem_o[0][vec_offset_bf16 + 2*i + 1] * inv_l;
            unsigned int lo = (unsigned int)__bfloat16_as_ushort(__float2bfloat16(v0));
            unsigned int hi = (unsigned int)__bfloat16_as_ushort(__float2bfloat16(v1));
            o32[i] = lo | (hi << 16);
        }
    }
}
