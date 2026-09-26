// SPDX-License-Identifier: MIT OR Apache-2.0

// 2026-09-25: FP16 h-state store and load helpers for the GDN `_f16` kernels.
//
// Owner: gb10 kernels.
// Invariants:
// - gdn_f16_store always returns a finite value: it clamps to [-65504, 65504] before
//   rounding, and inside that range it is bit-for-bit __float2half.
// - gdn_f16_load is exactly __half2float.
//
// The FP16 h-state is on under `--ssm-h-dtype f16` or `f16-pool` (ssm_h_dtype_bits in
// qwen3_ssm/gdn_flags.rs; the METRALE_SSM_H_FP16 env var is the fallback). The `_f16` WY
// verify twins in this directory keep their FP32 parents' arithmetic in FP32 registers; only
// the h-state and its rollback intermediates are `__half` in memory.
//
// Layout: ssm_h_to_f16_dispatch (model-engine trait_impl/meta.rs) converts a slot element
// for element and copies the FP16 result to the start of the slot, so inside a slot a head
// still starts at vh * k_dim * v_dim elements. The slot pitch is h_slot_stride_bytes()
// (qwen3_ssm/ssm_h_fp16.rs): 4 bytes per element on the FP32-sized pool (`f16`), 2 on the
// FP16-sized pool (`f16-pool`). Batched launches therefore pass one base pointer per
// sequence (`state_is_table`), and the host launchers of the contiguous form refuse
// batch_size > 1 (ops/ssm_gdn_b.rs, ops/ssm_gdn_wyn.rs).










#ifndef METRALE_GDN_F16_STATE_CUH
#define METRALE_GDN_F16_STATE_CUH

#include <cuda_fp16.h>








// 2026-09-25: FP32 -> FP16 h-state store. A value beyond the FP16 range is stored as
// +-65504 instead of +-inf, and a NaN input is stored as -65504.
__device__ __forceinline__ __half gdn_f16_store(float v) {
    return __float2half(fminf(fmaxf(v, -65504.0f), 65504.0f));
}


// 2026-09-25: FP16 -> FP32 h-state load.
__device__ __forceinline__ float gdn_f16_load(const __half* __restrict__ p) {
    return __half2float(*p);
}

#endif
