// SPDX-License-Identifier: MIT OR Apache-2.0

// 2026-09-25: GLM-5.3-Flash NoPE MLA latent cache write: RMSNorm of the kv_a projection,
// quantised to FP8 into the paged slot (module `glm5next_mla_latent_write`).
// Owner: gb10 kernels (glm-5.3-flash).
// Invariants: none beyond the launch contract: one block per token, blockDim.x ==
// kv_lora_dim (at most 1024), slot_mapping holds absolute slots, -1 skips the token.
//
// Not common/fused_k_norm_rope_cache.cu: that kernel normalises with `x * rms * (1 + w)`
// and has a rope arm, while GLM's kv_a_layernorm is `x * rms * w` and GLM is NoPE with one
// latent head.
//
// The write lands at `slot * kv_lora_dim`. With the layer's decode arguments (num_kv_heads
// 1, cache_stride_bytes = block_size * kv_lora_dim; glm5next_dsa/layer.rs) that is the
// address glm5next_dsa_mla_decode_fp8 reads for `slot = physical_block * block_size + p`.





#include <cuda_bf16.h>
#include <cuda_fp8.h>

__device__ __forceinline__ float warp_reduce_sum_glm(float v) {
    #pragma unroll
    for (int off = 16; off > 0; off >>= 1) v += __shfl_xor_sync(0xffffffff, v, off);
    return v;
}

extern "C" __global__ void glm5next_mla_latent_write_fp8(
    const __nv_bfloat16* __restrict__ kv_a,       // 2026-09-25: [num_tokens, kv_lora_dim] pre-norm
    const __nv_bfloat16* __restrict__ norm_w,     // 2026-09-25: [kv_lora_dim] kv_a_layernorm.weight
    __nv_fp8_storage_t* __restrict__ cache,       // 2026-09-25: paged FP8 latent cache
    const long long* __restrict__ slot_mapping,   // 2026-09-25: [num_tokens], -1 = skip this token
    const unsigned int kv_lora_dim,
    const float rms_eps,
    const float inv_scale                         // 2026-09-25: 1 / k_scale, matching the decode read
) {
    const unsigned int token = blockIdx.x;
    const unsigned int t = threadIdx.x;
    if (t >= kv_lora_dim) return;


    const long long slot = slot_mapping[token];
    if (slot < 0) return;

    const __nv_bfloat16* row = kv_a + (unsigned long long)token * kv_lora_dim;
    const float x = __bfloat162float(row[t]);

    // 2026-09-25: RMS over the whole latent. The block is exactly kv_lora_dim wide, so the
    // two-stage reduction covers every element with no tail loop.
    float ss = warp_reduce_sum_glm(x * x);
    __shared__ float warp_sums[32];
    const unsigned int warp_id = t / 32;
    const unsigned int lane = t % 32;
    if (lane == 0) warp_sums[warp_id] = ss;
    __syncthreads();
    if (warp_id == 0) {
        const unsigned int n_warps = (kv_lora_dim + 31) / 32;
        float v = (lane < n_warps) ? warp_sums[lane] : 0.0f;
        v = warp_reduce_sum_glm(v);
        if (lane == 0) warp_sums[0] = v;
    }
    __syncthreads();

    const float rms = rsqrtf(warp_sums[0] / (float)kv_lora_dim + rms_eps);
    // 2026-09-25: Plain RMSNorm weight, with no `1 +` offset.
    const float normed = x * rms * __bfloat162float(norm_w[t]);

    cache[(unsigned long long)slot * kv_lora_dim + t] =
        __nv_cvt_float_to_fp8(normed * inv_scale, __NV_SATFINITE, __NV_E4M3);
}
