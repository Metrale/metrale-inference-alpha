// SPDX-License-Identifier: MIT OR Apache-2.0

// 2026-09-26: Placeholder: per-request LoRA routing (bgmv) is not implemented for Metal.
// Both kernels are empty and carry `_stub` names, so a lookup of the CUDA entry points
// (`lora_bgmv_shrink`, `lora_bgmv_expand_fold` in kernels/gb10/common/lora_bgmv.cu) does not
// find them here.


#include <metal_stdlib>
using namespace metal;

kernel void lora_bgmv_shrink_stub() {}
kernel void lora_bgmv_expand_fold_stub() {}
