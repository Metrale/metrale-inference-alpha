// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-25: `MetalGpuBackend` parity tests, one submodule per kernel family,
//! plus the real-checkpoint tests (`real_model_*`).
//!
//! Owner: gpu-runtime (metal tests).
//! Invariants: compiled only in test builds, since `metal_backend.rs` declares
//! `#[cfg(test)] mod tests;`.

mod helpers;

mod embedded_kernels;

mod parity_asym;
mod parity_attention;
mod parity_attention_full;
mod parity_basic;
mod parity_gdn;
mod parity_norms;
mod parity_quant;
mod parity_turbo;
mod parity_turbo23;
mod parity_turbo4;
mod parity_vision;

mod real_model_attention;
mod real_model_gemv;
mod real_model_misc;
mod real_model_vision;
