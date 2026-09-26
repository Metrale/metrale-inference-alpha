// SPDX-License-Identifier: MIT OR Apache-2.0
// forked-from: kernels/gb10/common/attn_prefill_paged_indirect.cu (2026-09-24; 64 of 66 lines differ, see kernels/FORKS.md)

// 2026-09-26: attn_prefill_paged_indirect for the qwen3.6-27b tree at HDIM 128. The common
// source defaults HDIM to 256 (prefill_paged_compute.cuh), so this file defines it before
// the include, and every tile and loop bound derived from HDIM is sized for 128-element
// heads. The DFlash draft head loads this kernel (dflash_head/from_weights/kernel_handles.rs)
// next to `attn_prefill_h128`, which is also built for head_dim 128.


#define HDIM 128
#include "../../common/attn_prefill_paged_indirect.cu"
