// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-25: The `cublaslt` module of a build without the `cuda` feature:
//! the same GEMM entry points as `cublaslt.rs`, each `unreachable!`, so
//! callers in metrale-model-layers compile, and the real `scale_layout`.
//!
//! Owner: gpu-runtime (cuBLASLt wrapper).
//! Invariants:
//! - Every GEMM entry point here panics when called.

use anyhow::Result;

// 2026-09-25: `scale_layout.rs` calls no cuBLASLt function, so this build
// compiles the same file rather than a copy.
#[path = "cublaslt/scale_layout.rs"]
pub mod scale_layout;

pub fn bf16_gemm_act_weight_t(
    _act: u64,
    _weight: u64,
    _out: u64,
    _m: u32,
    _n: u32,
    _k: u32,
    _stream: u64,
) -> Result<()> {
    unreachable!("cublaslt::bf16_gemm_act_weight_t is cuda-only (not built for metal)")
}

pub fn bf16_gemm_act_weight_t_f32_out(
    _act: u64,
    _weight: u64,
    _out: u64,
    _m: u32,
    _n: u32,
    _k: u32,
    _stream: u64,
) -> Result<()> {
    unreachable!("cublaslt::bf16_gemm_act_weight_t_f32_out is cuda-only (not built for metal)")
}

pub fn fp8_gemm_act_weight_t_rowwise(
    _act_fp8: u64,
    _act_scale: u64,
    _weight_fp8: u64,
    _weight_scale: u64,
    _out: u64,
    _m: u32,
    _n: u32,
    _k: u32,
    _stream: u64,
) -> Result<()> {
    unreachable!("cublaslt::fp8_gemm_act_weight_t_rowwise is cuda-only (not built for metal)")
}

pub fn fp8_gemm_act_weight_t_blkscaled(
    _act_fp8: u64,
    _act_scale: u64,
    _weight_fp8: u64,
    _weight_block_scale: u64,
    _out: u64,
    _m: u32,
    _n: u32,
    _k: u32,
    _stream: u64,
) -> Result<()> {
    unreachable!("cublaslt::fp8_gemm_act_weight_t_blkscaled is cuda-only (not built for metal)")
}

#[allow(clippy::too_many_arguments)]
pub fn fp8_gemm_act_weight_t_blkscaled_ldc(
    _act_fp8: u64,
    _act_scale: u64,
    _weight_fp8: u64,
    _weight_block_scale: u64,
    _out: u64,
    _m: u32,
    _n: u32,
    _k: u32,
    _ldc: u32,
    _stream: u64,
) -> Result<()> {
    unreachable!("cublaslt::fp8_gemm_act_weight_t_blkscaled_ldc is cuda-only (not built for metal)")
}
