// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-25: Sizes of the two keep-packed Q2_0 prefill scratch buffers, each 0 unless its lever is `1`.
//!
//! Owner: gpu-runtime.
//! Invariants:
//! - `q2_scratch_sizes` returns 0 for the dequant scratch unless
//!   `METRALE_GGUF_NATIVE_Q2=1`, and 0 for the q8_1 scratch unless
//!   `METRALE_GGUF_NATIVE_Q2_MMQ=1`.

use metrale_config::ModelConfig;

/// 2026-09-25: Bytes of the keep-packed Q2_0 dequant scratch: `hidden_size`
/// times the widest of intermediate, QKVZ, Mamba-2 in_proj, the q projection
/// and 2 · kv_heads · head_dim, in BF16. This covers a weight `[N, K]` with
/// `hidden_size` as one dimension and the other at most that width; the
/// dequant callers debug-assert `N * K * 2` against the allocated size. It
/// does not depend on the batch.
pub fn q2_dequant_scratch_bytes(config: &ModelConfig) -> usize {
    let bf16 = 2;
    let hd = config.head_dim;
    let q_proj_mul = if config.attn_gated { 2 } else { 1 };
    let max_n = config
        .intermediate_size
        .max(config.ssm_qkvz_size())
        .max(config.mamba2_in_proj_size())
        .max(config.num_attention_heads * q_proj_mul * hd)
        .max(2 * config.num_key_value_heads * hd);
    max_n * config.hidden_size * bf16
}

/// 2026-09-25: `(q2_dequant_scratch, q2_act_q8)` bytes for the arena; `m` is
/// `max_batch_tokens`, `h` `hidden_size`, `hd` `head_dim`. The q8_1 scratch is
/// `m * kpad * 4 + 1 MiB` for K = max(h, intermediate, q_heads * head_dim)
/// rounded up to 256 as `kpad`, the formula of `q8_1_scratch_bytes` in
/// metrale-model-layers.
pub fn q2_scratch_sizes(config: &ModelConfig, m: usize, h: usize, hd: usize) -> (usize, usize) {
    let dequant_enabled = std::env::var("METRALE_GGUF_NATIVE_Q2").ok().as_deref() == Some("1");
    let mmq_enabled = std::env::var("METRALE_GGUF_NATIVE_Q2_MMQ").ok().as_deref() == Some("1");
    q2_scratch_sizes_for(config, m, h, hd, dequant_enabled, mmq_enabled)
}

pub(super) fn q2_scratch_sizes_for(
    config: &ModelConfig,
    m: usize,
    h: usize,
    hd: usize,
    dequant_enabled: bool,
    mmq_enabled: bool,
) -> (usize, usize) {
    let q2_dequant_scratch = if dequant_enabled {
        q2_dequant_scratch_bytes(config)
    } else {
        0
    };

    let q2_act_q8 = if mmq_enabled {
        let kmax = h
            .max(config.intermediate_size)
            .max(config.num_attention_heads * hd);
        let kpad = kmax.div_ceil(256) * 256;
        m * kpad * 4 + (1 << 20)
    } else {
        0
    };

    (q2_dequant_scratch, q2_act_q8)
}
