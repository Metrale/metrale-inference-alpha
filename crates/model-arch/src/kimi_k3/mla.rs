// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-25: Re-exports the Kimi K3 gated NoPE MLA CPU reference, `metrale_model_weights::kimi_k3_host::mla`.
//!
//! Decode runs the CUDA launch in [`super::mla_cuda`] instead, unless `K3_CUDA_MLA=0`.
//!
//! Owner: model-arch, Kimi K3.
//! Invariants: none beyond the types.

pub use metrale_model_weights::kimi_k3_host::mla::*;
