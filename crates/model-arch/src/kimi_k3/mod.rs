// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-25: Kimi K3 in the engine: the bound decoder layer, its per-sequence state, and the CUDA launches for the KDA and MLA mixers, the packed LatentMoE experts and the dense MLP.
//!
//! The CPU reference math is re-exported from `metrale_model_weights::kimi_k3_host`.
//!
//! Owner: model-arch, Kimi K3.
//! Invariants: none beyond the types.

pub mod dense_cuda;
pub mod device_cache;
pub mod kda;
pub mod kda_cuda;
pub mod latent_moe;
pub mod mla;
pub mod mla_cuda;
pub mod moe_cuda;
pub mod tp;

pub use device_cache::{DeviceHybridCache, DeviceLayerCache};
pub use kda_cuda::{
    K3KdaDecodeKernels, KdaDeviceState, launch_k3_kda_decode_token,
    launch_k3_kda_decode_token_on_device,
};
pub use metrale_model_weights::kimi_k3_host::{
    AttnResHub, HybridCache, K3Graph, KDA_L2_EPS, KdaConfig, KdaState, LatentMoeConfig, LayerCache,
    MixerKind, MlaConfig, MlaKv, MlpKind, attnres_blend, attnres_mix, attnres_softmax_mix,
    cuda_kda_enabled, cuda_mla_enabled, gated_mla_attend, kda_decode_token, kda_from,
    latent_moe_forward, mix_routed_experts, mla_decode_token, mla_from, moe_from, sigmoid_topk,
    situ_glu, situ_glu_vec, softcap,
};
pub use mla_cuda::{
    K3MlaDecodeKernels, MlaDeviceKv, launch_k3_mla_decode_token,
    launch_k3_mla_decode_token_on_device,
};
pub use moe_cuda::{K3MoeGemmKernels, launch_k3_latent_moe_experts};
pub use tp::{supports_tp, tensor_plan};

pub mod bound;
mod host_decode;
pub mod state;
pub use metrale_model_weights::kimi_k3_host::{Ablation, K3CpuModel, K3LayerSpec, greedy_decode};
pub use state::K3CpuFallbackState;
#[cfg(test)]
mod host_decode_kda;
#[cfg(test)]
mod host_decode_mla;
