// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-25: Re-exports the Kimi K3 LatentMoE CPU reference (latent down projection, sigmoid top-k router, SiTU-GLU experts, up projection), `metrale_model_weights::kimi_k3_host::latent_moe`.
//!
//! Owner: model-arch, Kimi K3.
//! Invariants: none beyond the types.

pub use metrale_model_weights::kimi_k3_host::latent_moe::*;
