// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-25: The model engine: the `Model` trait (`traits`), the transformer model (`model`),
//! the single-request generate loop (`engine`), the model factory (`factory`), the Kimi K3 weight
//! loader (`kimi_k3_loader`) and the startup rank-agreement check (`rank_agree`).
//!
//! Owner: model-engine.
//! Invariants: none beyond the types.

pub mod engine;
pub mod factory;
pub mod kimi_k3_loader;
pub mod model;
pub mod rank_agree;
pub mod traits;
