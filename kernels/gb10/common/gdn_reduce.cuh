// SPDX-License-Identifier: MIT OR Apache-2.0

// 2026-09-25: Warp and block sum reductions shared by the GDN WY verify kernels.
//
// Owner: gb10 kernels.
// Invariants:
// - metrale_block_reduce_sum always sums in one order: shfl_down 16, 8, 4, 2, 1 within
//   each warp, then lanes 0..3 of warp 0 combine the four warp sums with shfl_down 2 and 1,
//   giving (w0 + w2) + (w1 + w3). The per-head norm reduction in gated_delta_rule_decode
//   (gated_delta_rule.cu) uses the same shuffle tree.
// - Both functions are static, so every translation unit that includes this header gets
//   its own copy.







#ifndef METRALE_GDN_REDUCE_CUH
#define METRALE_GDN_REDUCE_CUH


// 2026-09-25: Sum over the 32 lanes of a warp. Every lane must call it; lane 0 ends with
// the full sum.
static __device__ __forceinline__ float metrale_warp_reduce_sum(float val) {
    val += __shfl_down_sync(0xFFFFFFFF, val, 16);
    val += __shfl_down_sync(0xFFFFFFFF, val,  8);
    val += __shfl_down_sync(0xFFFFFFFF, val,  4);
    val += __shfl_down_sync(0xFFFFFFFF, val,  2);
    val += __shfl_down_sync(0xFFFFFFFF, val,  1);
    return val;
}












// 2026-09-25: Sum `val` over a block of exactly 128 threads (4 warps). Every thread must
// call it, because it holds two __syncthreads. `smem_warp` needs 4 floats, and every thread
// returns the sum read from smem_warp[0] after the second barrier. Two calls that share
// `smem_warp` need a __syncthreads between them, or the next call's lane-0 write to
// smem_warp[0] can overtake a thread still reading the previous result.
static __device__ __forceinline__ float metrale_block_reduce_sum(
    float val,
    float* smem_warp,
    unsigned int tid
) {
    val = metrale_warp_reduce_sum(val);
    unsigned int warp_id = tid / 32;
    unsigned int lane_id = tid & 31;
    if (lane_id == 0) smem_warp[warp_id] = val;
    __syncthreads();
    if (tid < 4) {
        float s = smem_warp[tid];
        s += __shfl_down_sync(0xf, s, 2);
        s += __shfl_down_sync(0xf, s, 1);
        if (tid == 0) smem_warp[0] = s;
    }
    __syncthreads();
    return smem_warp[0];
}

#endif
