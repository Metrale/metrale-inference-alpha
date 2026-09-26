// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-26: The `scratch`, `ssd_scratch` and `gdn_fla_scratch` sizes that
//! [`super::BufferSizes::from_config`] computes, as functions of its inputs.
//!
//! Owner: gpu-runtime.
//! Invariants:
//! - Nothing here allocates or reads the environment; each function returns
//!   what its expressions in `from_config` produced.

use metrale_config::ModelConfig;

use super::super::DecodeMetaLayout;
use super::super::sizes_q12::{Q12_SIZING_STREAMS, q12_batched_scratch_bytes};

/// 2026-09-26: Bytes of `scratch`: the largest of the prefill, batched-decode,
/// batched-verify and kernel-batched prefill layouts. `m` is `max_batch_tokens`.
pub(super) fn scratch_bytes(
    config: &ModelConfig,
    m: usize,
    max_seq_len: usize,
    kv_block_size: usize,
    decode_meta: &DecodeMetaLayout,
) -> usize {
    let top_k = config.num_experts_per_tok;

    // 2026-09-25: `scratch` is the largest of its layouts, each re-uploaded by
    // its user before use:
    // - prefill: MoE top-K indices and weights (`moe_scratch`), then the
    //   chunk metadata (`prefill_meta`);
    // - batched decode and batched MTP verify: metadata at `scratch + 32768`,
    //   in the `decode_meta.rs` layout with R = `decode_meta.rows()` or with
    //   R = `bt_rows` (the verify layout, `verify_e.rs`);
    // - the kernel-batched prefill staging (`q12_batched`).
    let moe_scratch = 2 * m * top_k * 4;
    let max_blocks = max_seq_len
        .checked_div(kv_block_size)
        .map(|q| q + 1)
        .unwrap_or(256);
    // 2026-09-25: Prefill metadata: positions, slots, a block table and a
    // seq_len. An MRoPE model uploads three u32 position streams (T, H, W)
    // back to back (`prefill_b/upload_meta.rs`), every other model one.
    let pos_streams = if config.mrope_interleaved { 3 } else { 1 };
    let pos_bytes = m * 4 * pos_streams;
    let slot_offset = (pos_bytes + 7) & !7;
    let slot_end = slot_offset + m * 8;
    let bt_offset = (slot_end + 3) & !3;
    let bt_end = bt_offset + max_blocks * 4;
    let sl_offset = (bt_end + 3) & !3;
    let prefill_meta = sl_offset + 4;
    // 2026-09-25: `bt_rows` must be at least `VERIFY_ROW_CAP` (`verify_e2.rs`
    // in metrale-model-engine), the row count of the verify layout, whose
    // block table starts at 24R. The envelope is the larger of that layout
    // and the decode layout.
    let bt_rows = 160usize;
    let bt_meta =
        32768 + (bt_rows * 24 + bt_rows * max_blocks * 4).max(decode_meta.meta_bytes(max_blocks));
    let scratch_min = 64 * 1024;
    // 2026-09-25: The kernel-batched prefill staging, provisioned for
    // `Q12_SIZING_STREAMS` streams splitting the token arena. A batch whose
    // staging does not fit is refused by `check_kernel_batched_eligible`
    // (metrale-model-engine) and runs per stream.
    let q12_chunk = m.div_ceil(Q12_SIZING_STREAMS).max(1);
    let q12_batched = q12_batched_scratch_bytes(
        Q12_SIZING_STREAMS,
        q12_chunk,
        top_k,
        config.mrope_interleaved,
    );
    scratch_min
        .max(moe_scratch + prefill_meta)
        .max(bt_meta)
        .max(q12_batched)
}

/// 2026-09-26: Bytes of `(ssd_scratch, gdn_fla_scratch)`, the Mamba-2 SSD scan
/// and GDN FLA prefill scratch, each 0 for a model without that layer kind.
pub(super) fn ssm_scratch_bytes(config: &ModelConfig, m: usize, bf16: usize) -> (usize, usize) {
    // 2026-09-25: GDN FLA prefill scratch, regions in order, for `nt` chunks of
    // 64 tokens: W [nt*nv][64][kd] BF16, U [nt*nv][64][vd] BF16,
    // S [nt*nv][kd][vd] BF16, uc [nt*nv][64][vd] BF16, gc [nt*nv][64] F32
    // (the offsets `trait_prefill_gdn/batched.rs` carves).
    const FLA_CHUNK: usize = 64;
    // 2026-09-25: Mamba-2 SSD scan scratch with L = 64 and nc = ceil(M/L) + 1:
    // dt [H][nc][L] F32, dA_cumsum [H][nc][L] F32, CB [nc][G][L][L] F32.
    const SSD_L: usize = 64;
    let ssd_scratch = if config.mamba_num_heads > 0 && config.ssm_state_size > 0 {
        let nc = m.div_ceil(SSD_L) + 1;
        let hh = config.mamba_num_heads;
        let gg = config.n_groups.max(1);
        (hh * nc * SSD_L * 4) * 2 + nc * gg * SSD_L * SSD_L * 4
    } else {
        0
    };

    let gdn_fla_scratch = if config.linear_num_value_heads > 0
        && config.linear_key_head_dim == 128
        && config.linear_value_head_dim == 128
    {
        // 2026-09-25: The batched FLA path (`METRALE_GDN_BATCHED_FLA`) needs
        // `batch * ceil(chunk_len / 64)` chunks, up to one more per stream than
        // ceil(M / 64); the + 16 covers 16 streams.
        let nt = m.div_ceil(FLA_CHUNK) + 16;
        let nv = config.linear_num_value_heads;
        let kd = config.linear_key_head_dim;
        let vd = config.linear_value_head_dim;
        let w = nt * nv * FLA_CHUNK * kd * bf16;
        let u = nt * nv * FLA_CHUNK * vd * bf16;
        let s = nt * nv * kd * vd * bf16;
        let uc = nt * nv * FLA_CHUNK * vd * bf16;
        let gc = nt * nv * FLA_CHUNK * 4;
        w + u + s + uc + gc
    } else {
        0
    };
    (ssd_scratch, gdn_fla_scratch)
}
