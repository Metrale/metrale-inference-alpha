// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-25: `ModelDims`, the model geometry the high-speed-swap tier is built for.
//!
//! Owner: storage, high-speed swap.
//! Invariants: none beyond the types.
//!
//! It holds no GPU state and compiles without the `cuda` feature.

/// 2026-09-25: Geometry of the model `HighSpeedSwap` serves, from
/// `high_speed_swap_dims` in metrale-model-engine.
#[derive(Clone, Copy, Debug)]
pub struct ModelDims {
    pub num_layers: u32,
    pub max_blocks_per_layer: u32,
    pub num_q_heads: u16,
    pub num_kv_heads: u16,
    pub head_dim: u16,
    pub block_size: u16,
    /// 2026-09-25: Model fingerprint (`ModelFingerprint::derive_kv` in
    /// metrale-model-engine), folded into the KV paging namespace by
    /// `kv_paging::ns::derive_kv_ns`. `None` when it could not be derived; a
    /// `METRALE_KV_PAGING=1` connect then fails unless `METRALE_KV_PAGING_NS` is set.
    /// Only `connect_kv_peer_backend` reads it.
    pub model_fp: Option<std::num::NonZeroU64>,
}
