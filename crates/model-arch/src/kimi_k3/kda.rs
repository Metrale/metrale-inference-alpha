// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-25: Re-exports the Kimi K3 KDA CPU reference, `metrale_model_weights::kimi_k3_host::kda`.
//!
//! Decode runs the CUDA launch in [`super::kda_cuda`] instead, unless `K3_CUDA_KDA=0`.
//!
//! Owner: model-arch, Kimi K3.
//! Invariants: none beyond the types.

pub use metrale_model_weights::kimi_k3_host::kda::*;
