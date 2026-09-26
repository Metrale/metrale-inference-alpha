// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-25: Size of the row-wise FP8 GDN prefill BF16-weight slab: 0 unless `METRALE_FP8_ROWWISE=1`.
//!
//! The row-wise GDN prefill arms multiply BF16 copies of their per-row FP8
//! weights. Sizing the copies here makes them one arena allocation that the
//! preflight reserve (`BufferSizes::total_bytes`) counts; the arms carve their
//! slices from it (`rowwise_slab.rs`).
//!
//! Owner: gpu-runtime.
//! Invariants:
//! - [`ssm_rowwise_w_bf16_bytes`] reads the lever with the same predicate as
//!   the loader's `rowwise_fp8_enabled` (`== "1"`).

use metrale_config::ModelConfig;

/// 2026-09-25: BF16 bytes of one GDN layer's row-wise copies: `in_proj_qkvz`
/// `[ssm_qkvz_size, hidden]` and `out_proj` `[hidden, value_dim]`, the two
/// weights `set_fp8_rowwise_prefill_weights` installs, with
/// `value_dim = linear_num_value_heads * linear_value_head_dim`. At the
/// Qwen3.8-27B shapes of `buffers/tests.rs` (hidden 5120, 16x128 key heads,
/// 48x128 value heads) that is `167772160 + 62914560` B.
pub fn ssm_rowwise_w_bf16_layer_bytes(config: &ModelConfig) -> usize {
    let bf16 = 2;
    let value_dim = config.linear_num_value_heads * config.linear_value_head_dim;
    let qkvz = config.ssm_qkvz_size() * config.hidden_size * bf16;
    let out_proj = config.hidden_size * value_dim * bf16;
    qkvz + out_proj
}

/// 2026-09-25: Slab bytes: the per-layer pair times `num_ssm_layers`, or 0
/// unless `METRALE_FP8_ROWWISE=1`. The predicate is the one in the loader's
/// `rowwise_fp8_enabled` (`weight_loader/qwen35_dense/rowwise_fp8.rs`): the
/// loader installs the row-wise weights on it and the arms then need this
/// slab, so the two must agree.
pub fn ssm_rowwise_w_bf16_bytes(config: &ModelConfig) -> usize {
    ssm_rowwise_w_bf16_bytes_for(
        config,
        std::env::var("METRALE_FP8_ROWWISE").as_deref() == Ok("1"),
    )
}

/// 2026-09-25: [`ssm_rowwise_w_bf16_bytes`] with the lever passed in, for tests
/// that must not set the process environment.
pub fn ssm_rowwise_w_bf16_bytes_for(config: &ModelConfig, rowwise_enabled: bool) -> usize {
    if !rowwise_enabled {
        return 0;
    }
    config.num_ssm_layers() * ssm_rowwise_w_bf16_layer_bytes(config)
}
