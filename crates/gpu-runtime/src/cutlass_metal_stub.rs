// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-25: The `cutlass` module of a build without the `cuda` feature. It
//! keeps the entry points metrale-model-layers calls without a `cfg`, so that
//! build compiles.
//!
//! Owner: gpu-runtime.
//! Invariants:
//! - Every function body is `unreachable!`: reaching one is a bug.

use anyhow::Result;

pub fn bf16_gemm_act_weight_t(
    _act: u64,
    _weight: u64,
    _out: u64,
    _m: u32,
    _n: u32,
    _k: u32,
    _stream: u64,
) -> Result<()> {
    unreachable!("cutlass::bf16_gemm_act_weight_t is cuda-only (not built for metal)")
}

pub fn nvfp4_gemm_bf16_act_weight_t(
    _act: u64,
    _weight_packed_t: u64,
    _weight_scale_t: u64,
    _weight_scale_2: f32,
    _out: u64,
    _m: u32,
    _n: u32,
    _k: u32,
    _stream: u64,
) -> Result<()> {
    unreachable!("cutlass::nvfp4_gemm_bf16_act_weight_t is cuda-only (not built for metal)")
}

#[allow(clippy::too_many_arguments)]
pub fn nvfp4_grouped_gate_up_fused(
    _a: u64,
    _sorted_token_ids: u64,
    _gate_packed_ptrs: &[u64],
    _gate_sfb_ptrs: &[u64],
    _gate_scale2_vals: &[f32],
    _up_packed_ptrs: &[u64],
    _up_sfb_ptrs: &[u64],
    _up_scale2_vals: &[f32],
    _c_gate: u64,
    _c_up: u64,
    _expert_offsets_host: &[i32],
    _n: u32,
    _k: u32,
    _stream: u64,
) -> Result<()> {
    unreachable!("cutlass::nvfp4_grouped_gate_up_fused is cuda-only (not built for metal)")
}

pub fn nvfp4_grouped_down(
    _a: u64,
    _packed_ptrs: &[u64],
    _sfb_ptrs: &[u64],
    _scale2_vals: &[f32],
    _c: u64,
    _expert_offsets_host: &[i32],
    _n: u32,
    _k: u32,
    _stream: u64,
) -> Result<()> {
    unreachable!("cutlass::nvfp4_grouped_down is cuda-only (not built for metal)")
}

pub fn pack_bf16_weight_to_nvfp4_t(
    _weight_bf16: u64,
    _packed_t: u64,
    _scale_t: u64,
    _n: u32,
    _k: u32,
    _stream: u64,
) -> Result<()> {
    unreachable!("cutlass::pack_bf16_weight_to_nvfp4_t is cuda-only (not built for metal)")
}

pub fn pack_weight_sfb(
    _scale_in: u64,
    _scale_out: u64,
    _n: u32,
    _k: u32,
    _src_n_major: bool,
    _stream: u64,
) -> Result<()> {
    unreachable!("cutlass::pack_weight_sfb is cuda-only (not built for metal)")
}

pub fn transpose_nvfp4_packed_kton(
    _src_packed_t: u64,
    _dst_packed: u64,
    _n: u32,
    _k: u32,
    _stream: u64,
) -> Result<()> {
    unreachable!("cutlass::transpose_nvfp4_packed_kton is cuda-only (not built for metal)")
}
