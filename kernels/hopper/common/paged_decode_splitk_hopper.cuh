// SPDX-License-Identifier: MIT OR Apache-2.0

// 2026-09-25: Shared device code for the Hopper paged-decode split-K twins,
// `paged_decode_fp8_splitk_hopper.cu` and `paged_decode_bf16_splitk_hopper.cu`.
//
// Owner: hopper kernels.
// Invariants:
// - A split's KV range depends only on its own sequence's `seq_len`, window
//   and `num_splits` (`pd_split_bounds`), and a warp's range only on its
//   split's range (`pd_warp_bounds`). The kernels take no batch count.
// - For a sequence with seq_len > 0, every split writes one
//   `[o[head_dim], m, l]` F32 partial, an empty split with l = 0. The reduce
//   seeds from split 0, which always has work, merges splits 1.. in index
//   order and skips l <= 0, so the merge tree depends only on
//   (seq_len, num_splits). A sequence with seq_len == 0 is not written.
//
// So, for a fixed `num_splits`, a sequence's output bytes do not depend on
// what it is batched with. `num_splits` comes from
// `metrale_kernels::attn_splitk`; under its `Auto` policy (the
// `attn_decode_splitk = "auto"` of kernels/hopper/HARDWARE.toml) and
// `Pinned` it depends only on the SM count, the q-head count and the pin.
// The online-softmax merge is not associative, so output is not
// bit-identical across different `num_splits`;
// `native_attn_decode_splitk_hopper_microtest` grades that with a tolerance
// and a KNOWN_BAD control.
//
// Against gb10's `paged_decode_attn_splitk_fp8`, which takes one KV position
// per loop iteration, these kernels use the PD_BC-position batched loop of
// the non-split `paged_decode_attn_fp8` and floor a split at
// PD_MIN_KV_PER_SPLIT positions. Measurements: ATTN-DECODE-SPLITK-ATTRIBUTION.md.





















#ifndef METRALE_PAGED_DECODE_SPLITK_HOPPER_CUH
#define METRALE_PAGED_DECODE_SPLITK_HOPPER_CUH

#include <cuda_bf16.h>

#define PD_WARP_SIZE 32
#ifndef HDIM
#define HDIM 256
#endif
// 2026-09-25: Per-lane element counts. A lane owns HDIM/32 head-dim elements; the u32
// counts are how many 32-bit loads that is for a 2-byte (BF16) and a 1-byte
// (FP8 E4M3) cache element.
#define PD_VEC        (HDIM / PD_WARP_SIZE)
#define PD_VEC_U32    (HDIM / (PD_WARP_SIZE * 2))
#define PD_VEC_U32_F8 (HDIM / (PD_WARP_SIZE * 4))
#define PD_NUM_WARPS 8
// 2026-09-25: KV positions per batched loop iteration, the BC of the non-split
// `paged_decode_attn_fp8`.

#define PD_BC 4

// 2026-09-25: Fewest KV positions a split may own. Applied in the kernel from each
// sequence's own `seq_len`, which is in device memory, so the partition
// stays independent of the batch.
//
// 256 = eight warps x eight batched iterations x PD_BC. A short sequence
// leaves its high splits empty instead of giving each a few positions.








#define PD_MIN_KV_PER_SPLIT 256

// 2026-09-25: 2 BF16 packed in a u32 -> 2 F32, low half first. Used for Q on both
// twins and for K/V on the BF16 twin. The FP8 unpack is in the FP8 twin's
// file, so this header needs only <cuda_bf16.h>.


__device__ __forceinline__ void pd_unpack2_bf16(unsigned int packed, float& v0, float& v1) {
    v0 = __bfloat162float(__ushort_as_bfloat16((unsigned short)(packed & 0xFFFF)));
    v1 = __bfloat162float(__ushort_as_bfloat16((unsigned short)(packed >> 16)));
}

// 2026-09-25: This split's half-open KV range, from the sequence's own length only.
//
// `split_size` is ceil(attended / num_splits) floored at PD_MIN_KV_PER_SPLIT,
// so a short sequence leaves the high splits empty (`kv_start == kv_end`)
// instead of giving each one a few positions. Returns false when this split
// has no work.


__device__ __forceinline__ bool pd_split_bounds(
    unsigned int seq_len,
    unsigned int window_start,
    unsigned int split_id,
    unsigned int num_splits,
    unsigned int& kv_start,
    unsigned int& kv_end
) {
    const unsigned int attended = seq_len - window_start;
    unsigned int split_size = (attended + num_splits - 1u) / num_splits;
    if (split_size < PD_MIN_KV_PER_SPLIT) split_size = PD_MIN_KV_PER_SPLIT;
    const unsigned long long start = (unsigned long long)window_start
                                   + (unsigned long long)split_id * split_size;
    if (start >= (unsigned long long)seq_len) return false;
    kv_start = (unsigned int)start;
    kv_end = kv_start + split_size;
    if (kv_end > seq_len) kv_end = seq_len;
    return kv_start < kv_end;
}

// 2026-09-25: This warp's slice of [kv_start, kv_end): the ceil-divided eight-way
// partition the non-split `paged_decode_attn_fp8` uses over the whole sequence.
__device__ __forceinline__ void pd_warp_bounds(
    unsigned int kv_start, unsigned int kv_end, unsigned int warp_id,
    unsigned int& my_start, unsigned int& my_end
) {
    const unsigned int local_len = kv_end - kv_start;
    const unsigned int chunk = (local_len + PD_NUM_WARPS - 1u) / PD_NUM_WARPS;
    my_start = kv_start + warp_id * chunk;
    my_end = my_start + chunk;
    if (my_end > kv_end) my_end = kv_end;
    if (my_start > kv_end) my_start = kv_end;
}

// 2026-09-25: Inter-warp tree merge, then write this split's partial as
// `[o[head_dim], m, l]` F32, the workspace layout gb10's
// `paged_decode_attn_reduce_fp8` also reads.
//
// `out_scale` is `v_scale` on the FP8 twin (o_reg accumulated raw V; the merge
// and the reduce are both linear in o) and 1.0f on the BF16 twin.
__device__ __forceinline__ void pd_emit_partial(
    float m_val, float l_val, const float* o_reg, float out_scale,
    float* smem_m, float* smem_l, float* smem_o,
    unsigned int warp_id, unsigned int lane_id, unsigned int vec_offset,
    float* __restrict__ workspace,
    unsigned int seq_idx, unsigned int q_head, unsigned int split_id,
    unsigned int num_q_heads, unsigned int num_splits, unsigned int head_dim
) {
    if (lane_id == 0) {
        smem_m[warp_id] = m_val;
        smem_l[warp_id] = l_val;
    }
    #pragma unroll
    for (int i = 0; i < PD_VEC; i++) {
        smem_o[warp_id * HDIM + vec_offset + i] = o_reg[i] * out_scale;
    }
    __syncthreads();

    #pragma unroll
    for (int stride = PD_NUM_WARPS / 2; stride > 0; stride >>= 1) {
        if (warp_id < (unsigned int)stride) {
            const unsigned int other = warp_id + stride;
            const float lw = smem_l[other];
            if (lw > 0.0f) {
                const float mw = smem_m[other];
                const float my_m = smem_m[warp_id];
                const float my_l = smem_l[warp_id];
                const float m_new = fmaxf(my_m, mw);
                const float scale_me = __expf(my_m - m_new);
                const float scale_w = __expf(mw - m_new);
                smem_l[warp_id] = my_l * scale_me + lw * scale_w;
                smem_m[warp_id] = m_new;
                #pragma unroll
                for (int i = 0; i < PD_VEC; i++) {
                    smem_o[warp_id * HDIM + vec_offset + i] =
                        smem_o[warp_id * HDIM + vec_offset + i] * scale_me +
                        smem_o[other * HDIM + vec_offset + i] * scale_w;
                }
            }
        }
        __syncthreads();
    }

    const unsigned int ws_stride = head_dim + 2u;
    float* ws_base = workspace
        + ((unsigned long long)seq_idx * num_q_heads + q_head) * num_splits * ws_stride
        + (unsigned long long)split_id * ws_stride;
    if (warp_id == 0) {
        #pragma unroll
        for (int i = 0; i < PD_VEC; i++) ws_base[vec_offset + i] = smem_o[vec_offset + i];
        if (lane_id == 0) {
            ws_base[head_dim] = smem_m[0];
            ws_base[head_dim + 1] = smem_l[0];
        }
    }
}

// 2026-09-25: Merge `num_splits` partials for one (seq, q_head) into the BF16 output.
//
// One warp per (q_head, seq). Splits are merged in index order and empty ones
// (`l <= 0`) are skipped, so the tree is a function of the sequence's own
// length: `pd_split_bounds` decides emptiness from `seq_len` and
// `num_splits` alone.
__device__ __forceinline__ void pd_reduce_body(
    const float* __restrict__ workspace,
    __nv_bfloat16* __restrict__ O,
    unsigned int seq_idx, unsigned int q_head, unsigned int lane_id,
    unsigned int num_q_heads, unsigned int head_dim, unsigned int num_splits
) {
    const unsigned int vec_off = lane_id * PD_VEC;
    const unsigned int ws_stride = head_dim + 2u;
    const float* ws_base = workspace
        + ((unsigned long long)seq_idx * num_q_heads + q_head) * num_splits * ws_stride;

    float m = ws_base[head_dim];
    float l = ws_base[head_dim + 1];
    float o_reg[PD_VEC];
    #pragma unroll
    for (int i = 0; i < PD_VEC; i++) o_reg[i] = ws_base[vec_off + i];

    for (unsigned int s = 1; s < num_splits; s++) {
        const float* ws = ws_base + (unsigned long long)s * ws_stride;
        const float ls = ws[head_dim + 1];
        if (ls <= 0.0f) continue;
        const float ms = ws[head_dim];
        const float m_new = fmaxf(m, ms);
        const float scale_me = __expf(m - m_new);
        const float scale_s = __expf(ms - m_new);
        #pragma unroll
        for (int i = 0; i < PD_VEC; i++)
            o_reg[i] = o_reg[i] * scale_me + ws[vec_off + i] * scale_s;
        l = l * scale_me + ls * scale_s;
        m = m_new;
    }

    const float inv_l = (l > 0.0f) ? (1.0f / l) : 0.0f;
    unsigned int* o32 = (unsigned int*)(O + (unsigned long long)seq_idx * num_q_heads * head_dim
                                          + (unsigned long long)q_head * head_dim + vec_off);
    #pragma unroll
    for (int i = 0; i < PD_VEC_U32; i++) {
        const float v0 = o_reg[2 * i] * inv_l;
        const float v1 = o_reg[2 * i + 1] * inv_l;
        const unsigned int lo = (unsigned int)__bfloat16_as_ushort(__float2bfloat16(v0));
        const unsigned int hi = (unsigned int)__bfloat16_as_ushort(__float2bfloat16(v1));
        o32[i] = lo | (hi << 16);
    }
}

#endif
