// SPDX-License-Identifier: MIT OR Apache-2.0

// 2026-09-25: Token-overlay kernels for LoRA adapters that replace embedding and lm_head
// rows (PEFT trainable_tokens / modules_to_save): a build-time row diff, the embed row
// replacement and the lm_head logit-column recompute (launchers: model-layers
// ops/token_overlay.rs).
//
// Owner: gb10 kernels.
// Invariants: none beyond the types.
//
// The routed kernels index per-adapter-slot tables with s = seq_slot[row], or active when
// seq_slot is null; s < 0 leaves the row untouched. s is not checked against the table
// length (max_loras).

#include <cuda_bf16.h>

// 2026-09-25: flags[r] = (max_i |base[r, i] - served[r, i]| > thresh), one thread per row;
// grid ceil(rows/256), block 256.

extern "C" __global__ void embed_rowdiff_bf16(
    const __nv_bfloat16* __restrict__ base,
    const __nv_bfloat16* __restrict__ served,
    unsigned char* __restrict__ flags,
    unsigned int rows,
    unsigned int h,
    float thresh
) {
    unsigned int r = blockIdx.x * blockDim.x + threadIdx.x;
    if (r >= rows) return;
    const __nv_bfloat16* a = base + (unsigned long long)r * h;
    const __nv_bfloat16* b = served + (unsigned long long)r * h;
    float maxd = 0.0f;
    for (unsigned int i = 0; i < h; ++i) {
        float d = fabsf(__bfloat162float(a[i]) - __bfloat162float(b[i]));
        if (d > maxd) maxd = d;
    }
    flags[r] = (maxd > thresh) ? 1u : 0u;
}

// 2026-09-25: Per row r: j = slot_map_tab[s][ids[r]]; when j >= 0, row j of rows_tab[s]
// replaces out[r] in place. model-engine token_overlay.rs runs it after the embed gather.
// One block of 256 threads per row.

extern "C" __global__ void embed_overlay_routed_bf16(
    const unsigned int* __restrict__ ids,
    const int* __restrict__ seq_slot,
    int active,
    const unsigned long long* __restrict__ slot_map_tab,
    const unsigned long long* __restrict__ rows_tab,
    const unsigned int* __restrict__ n_tab,
    __nv_bfloat16* __restrict__ out,
    unsigned int h,
    unsigned int vocab
) {
    const unsigned int r = blockIdx.x;
    int s = (seq_slot != nullptr) ? seq_slot[r] : active;
    if (s < 0) return;
    const int* slot_map = (const int*)slot_map_tab[s];
    if (slot_map == nullptr) return;
    // 2026-09-25: slot_map has vocab entries, the served vocab the overlay was built against
    // (EmbedOverlay::vocab). An id at or past it has no override row, so the gathered base row
    // stays, as it does for slot_map[id] == -1.





    if (ids[r] >= vocab) return;
    const int slot = slot_map[ids[r]];
    // 2026-09-25: build_overlay stores only compact indices below the slot's n_override
    // (overlay_build.rs), which n_tab[s] holds, so a larger j comes from a corrupt table; it is
    // skipped rather than read out of bounds.

    if (slot < 0 || (unsigned int)slot >= n_tab[s]) return;
    const __nv_bfloat16* src =
        (const __nv_bfloat16*)rows_tab[s] + (unsigned long long)slot * h;
    __nv_bfloat16* dst = out + (unsigned long long)r * h;
    for (unsigned int i = threadIdx.x; i < h; i += blockDim.x) {
        dst[i] = src[i];
    }
}

// 2026-09-25: dot(x, w) over h elements by one warp; only lane 0 holds the full sum.
__device__ __forceinline__ float warp_dot_bf16(
    const __nv_bfloat16* __restrict__ x,
    const __nv_bfloat16* __restrict__ w,
    unsigned int h
) {
    float acc = 0.0f;
    for (unsigned int i = threadIdx.x; i < h; i += warpSize) {
        acc += __bfloat162float(x[i]) * __bfloat162float(w[i]);
    }
    for (int off = warpSize / 2; off > 0; off >>= 1) {
        acc += __shfl_down_sync(0xffffffffu, acc, off);
    }
    return acc;
}

// 2026-09-25: For j = blockIdx.y < n_tab[s]: logits[row, ids_tab[s][j]] =
// dot(hidden[row], row j of rows_tab[s]), in place. One warp per (row, j); grid
// (m, max_n_override).
extern "C" __global__ void lmhead_overlay_routed_bf16(
    const __nv_bfloat16* __restrict__ hidden,
    const int* __restrict__ seq_slot,
    int active,
    const unsigned long long* __restrict__ rows_tab,
    const unsigned long long* __restrict__ ids_tab,
    const unsigned int* __restrict__ n_tab,
    __nv_bfloat16* __restrict__ logits,
    unsigned int h,
    unsigned int vocab
) {
    const unsigned int row = blockIdx.x;
    const unsigned int j = blockIdx.y;
    int s = (seq_slot != nullptr) ? seq_slot[row] : active;
    if (s < 0) return;
    if (j >= n_tab[s]) return;
    const unsigned int id = ((const unsigned int*)ids_tab[s])[j];
    // 2026-09-25: build_overlay emits only ids below its vocab (row diff over min(r, vocab)
    // rows, trainable ids from clamp_trainable_to_vocab), so an id at or past vocab comes from a
    // corrupt table; it is skipped rather than written out of bounds.
    if (id >= vocab) return;
    const __nv_bfloat16* w =
        (const __nv_bfloat16*)rows_tab[s] + (unsigned long long)j * h;
    const __nv_bfloat16* x = hidden + (unsigned long long)row * h;
    float dot = warp_dot_bf16(x, w, h);
    if (threadIdx.x == 0) {
        logits[(unsigned long long)row * vocab + id] = __float2bfloat16(dot);
    }
}

// 2026-09-25: lmhead_overlay_routed_bf16 with FP32 logits.
extern "C" __global__ void lmhead_overlay_routed_f32(
    const __nv_bfloat16* __restrict__ hidden,
    const int* __restrict__ seq_slot,
    int active,
    const unsigned long long* __restrict__ rows_tab,
    const unsigned long long* __restrict__ ids_tab,
    const unsigned int* __restrict__ n_tab,
    float* __restrict__ logits,
    unsigned int h,
    unsigned int vocab
) {
    const unsigned int row = blockIdx.x;
    const unsigned int j = blockIdx.y;
    int s = (seq_slot != nullptr) ? seq_slot[row] : active;
    if (s < 0) return;
    if (j >= n_tab[s]) return;
    const unsigned int id = ((const unsigned int*)ids_tab[s])[j];


    if (id >= vocab) return;
    const __nv_bfloat16* w =
        (const __nv_bfloat16*)rows_tab[s] + (unsigned long long)j * h;
    const __nv_bfloat16* x = hidden + (unsigned long long)row * h;
    float dot = warp_dot_bf16(x, w, h);
    if (threadIdx.x == 0) {
        logits[(unsigned long long)row * vocab + id] = dot;
    }
}
