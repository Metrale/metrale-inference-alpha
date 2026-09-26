// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-25: PLE, hashed n-gram injection into the hyper-connection
//! highway: the layer (`layer`, with its per-sequence carry in
//! `layer/aux_state`), its ids (`ids`) and debug taps (`dump`). The
//! reference forward, `Qwen4ExpTextPLELayer.forward` in
//! `bench/qwen4_exp/ref/modeling_qwen4_exp.py`:
//!
//! ```text
//! embeddings   = ple_embedding(input_ids)                  # [T, ple_embed_dim]
//! key_normed   = norm_key(key_proj(emb)) -> [T, hc, H]
//! value        = value_proj(emb)                           # [T, H]
//! query_normed = norm_query(hidden)      -> [T, hc, H]     # hidden is [T, hc * H]
//! gate  = (key_normed * query_normed).sum(-1) / sqrt(H)    # [T, hc]
//! gate  = sign(gate) * sqrt(max(|gate|, 1e-6))             # signed square root
//! gated = sigmoid(gate) * value                            # [T, hc, H]
//! out   = gated.flatten() + silu(conv1d(norm_conv(gated.flatten())))
//! ```
//!
//! The reference decoder layer adds the result to its input first
//! (`hidden_states = hidden_states + self.ple(...)`). Three details:
//!
//! 1. The gate takes a signed square root, `sign(g) * sqrt(max(|g|, 1e-6))`.
//! 2. `conv1d` is depthwise (`groups` equals its channel count) and dilated
//!    (`dilation = ngram_size`), so the carried state is
//!    `(kernel_size - 1) * dilation` steps.
//! 3. All three norms use the `normed * (1 + w)` form and are grouped with
//!    `group_size = hidden_size`.
//!
//! Owner: model-layers (PLE).
//! Invariants: none beyond the types.

#[path = "ple/ids.rs"]
pub mod ids;

#[cfg(test)]
#[path = "ple/tests.rs"]
mod tests;

#[path = "ple/dump.rs"]
pub mod dump;

#[path = "ple/layer.rs"]
mod layer;

pub use ids::{PleIdDims, ple_ngram_ids};
pub use layer::{PleLayer, PleSeqState, PleWeights};
