// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-25: Per-layer decode forwards written against `GpuBackend` and the
//! `QuantWeights` trait, so a driver runs a layer without knowing the weight
//! format.
//!
//! Owner: model-layers.
//! Invariants: none beyond the types.
//!
//! The only caller is the model-arch example `metal_qwen35_inference`. The
//! served decode path does not use this module; it lives in model-engine
//! (`model/trait_impl/decode_a.rs`).

pub mod quant_weights;
pub mod qwen3_5;

pub use quant_weights::QuantWeights;
