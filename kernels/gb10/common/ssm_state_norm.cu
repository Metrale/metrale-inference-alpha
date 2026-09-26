// SPDX-License-Identifier: MIT OR Apache-2.0

// 2026-09-25: Per-head Frobenius-norm clamp of the SSM h state over every SSM layer in one
// launch: a head with ||h||_F > MAX_NORM is scaled by MAX_NORM / ||h||_F.
//
// Owner: gb10 kernels.
// Invariants: none beyond the types.
//
// h_state_ptrs holds one device pointer per layer; each layer's state is
// [num_heads, k_dim, v_dim] with v contiguous. Grid (num_heads, num_layers), block v_dim,
// thread tid owning column tid (normalize_ssm_states_dispatch in model-engine
// trait_impl/meta.rs, dims from ssm_state_norm_dims). The block reduction assumes
// v_dim == 128: norm_sums has 4 slots, one per warp, and all four are summed.








#include <cuda_bf16.h>







#define MAX_NORM 200.0f

extern "C" __global__ void ssm_state_clamp_norm_fused(
    float** __restrict__ h_state_ptrs,
    unsigned int num_heads,
    unsigned int k_dim,
    unsigned int v_dim
) {
    const unsigned int head = blockIdx.x;
    const unsigned int layer = blockIdx.y;
    const unsigned int tid = threadIdx.x;
    if (head >= num_heads || tid >= v_dim) return;

    float* H = h_state_ptrs[layer] + (unsigned long long)head * k_dim * v_dim;


    float local_sq = 0.0f;
    for (unsigned int j = 0; j < k_dim; j++) {
        float v = H[j * v_dim + tid];
        local_sq += v * v;
    }


    unsigned int mask = __activemask();
    float warp_sum = local_sq;
    warp_sum += __shfl_down_sync(mask, warp_sum, 16);
    warp_sum += __shfl_down_sync(mask, warp_sum, 8);
    warp_sum += __shfl_down_sync(mask, warp_sum, 4);
    warp_sum += __shfl_down_sync(mask, warp_sum, 2);
    warp_sum += __shfl_down_sync(mask, warp_sum, 1);

    __shared__ float norm_sums[4];
    unsigned int warp_id = tid / 32;
    unsigned int lane_id = tid % 32;
    if (lane_id == 0) norm_sums[warp_id] = warp_sum;
    __syncthreads();

    if (tid < 4) {
        float s = norm_sums[tid];
        s += __shfl_down_sync(0xf, s, 2);
        s += __shfl_down_sync(0xf, s, 1);
        norm_sums[0] = s;
    }
    __syncthreads();
    float head_norm_sq = norm_sums[0];


    if (head_norm_sq > MAX_NORM * MAX_NORM) {
        float scale = MAX_NORM * rsqrtf(head_norm_sq);
        for (unsigned int j = 0; j < k_dim; j++) {
            H[j * v_dim + tid] *= scale;
        }
    }
}

// 2026-09-25: The same clamp over an FP16 h: values widen to FP32, the sum of squares and
// the scale are FP32, and each scaled value rounds back to FP16.
// normalize_ssm_states_dispatch launches it when seq_ssm_h_is_f16(seq) is true.












#include <cuda_fp16.h>

extern "C" __global__ void ssm_state_clamp_norm_fused_f16(
    __half** __restrict__ h_state_ptrs,
    unsigned int num_heads,
    unsigned int k_dim,
    unsigned int v_dim
) {
    const unsigned int head = blockIdx.x;
    const unsigned int layer = blockIdx.y;
    const unsigned int tid = threadIdx.x;
    if (head >= num_heads || tid >= v_dim) return;

    __half* H = h_state_ptrs[layer] + (unsigned long long)head * k_dim * v_dim;

    float local_sq = 0.0f;
    for (unsigned int j = 0; j < k_dim; j++) {
        float v = __half2float(H[j * v_dim + tid]);
        local_sq += v * v;
    }

    unsigned int mask = __activemask();
    float warp_sum = local_sq;
    warp_sum += __shfl_down_sync(mask, warp_sum, 16);
    warp_sum += __shfl_down_sync(mask, warp_sum, 8);
    warp_sum += __shfl_down_sync(mask, warp_sum, 4);
    warp_sum += __shfl_down_sync(mask, warp_sum, 2);
    warp_sum += __shfl_down_sync(mask, warp_sum, 1);

    __shared__ float norm_sums[4];
    unsigned int warp_id = tid / 32;
    unsigned int lane_id = tid % 32;
    if (lane_id == 0) norm_sums[warp_id] = warp_sum;
    __syncthreads();

    if (tid < 4) {
        float s = norm_sums[tid];
        s += __shfl_down_sync(0xf, s, 2);
        s += __shfl_down_sync(0xf, s, 1);
        norm_sums[0] = s;
    }
    __syncthreads();
    float head_norm_sq = norm_sums[0];

    if (head_norm_sq > MAX_NORM * MAX_NORM) {
        float scale = MAX_NORM * rsqrtf(head_norm_sq);
        for (unsigned int j = 0; j < k_dim; j++) {
            H[j * v_dim + tid] = __float2half(__half2float(H[j * v_dim + tid]) * scale);
        }
    }
}
