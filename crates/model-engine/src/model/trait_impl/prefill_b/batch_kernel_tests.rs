// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-25: Unit tests for the kernel-batched prefill admission rules in `batch_kernel/eligible.rs`.
//!
//! Owner: model-engine.
//! Invariants: none beyond the types.

use metrale_telemetry::prefix_cache::PrefixMatch;

use super::batch_kernel::{
    batched_reserve_hybrid_ssm_ok, cache_batch_matches_compatible, check_kernel_batched_eligible,
    config_is_mla,
};

/// 2026-09-25: A stream that stages its whole chunk (`eff_len == chunk_len`).
fn s(chunk_len: usize, chunk_start: usize, is_last: bool) -> (usize, usize, usize, bool) {
    (chunk_len, chunk_len, chunk_start, is_last)
}

/// 2026-09-25: A stream whose cached prefix leaves only `eff` of its `chunk_len`
/// to stage.
fn s_eff(
    chunk_len: usize,
    eff: usize,
    chunk_start: usize,
    is_last: bool,
) -> (usize, usize, usize, bool) {
    (chunk_len, eff, chunk_start, is_last)
}

// 2026-09-25: Scratch large enough that the scratch-footprint check does not
// decide any test that uses it.
const BIG_SCRATCH: usize = 8 * 1024 * 1024;
const TOP_K: usize = 8;
const MROPE: bool = false;

fn cache_match(tokens: usize) -> PrefixMatch {
    PrefixMatch {
        matched_blocks: vec![7; tokens / 16],
        matched_disk_block_ids: Vec::new(),
        matched_tokens: tokens,
        ssm_snapshot: None,
        ssm_snapshot_tokens: 0,
        ssm_snapshot_tier_key: None,
        ssm_snapshot_tier_tokens: 0,
        ssm_snapshot_is_tail: false,
    }
}

#[test]
fn hybrid_ssm_admits_only_all_cold_reservations() {
    // 2026-09-25: A model with SSM layers admits all-cold reservations
    // (`matched_tokens == 0` everywhere).
    assert!(batched_reserve_hybrid_ssm_ok(
        &[cache_match(0), cache_match(0), cache_match(0)],
        true,
    ));
    // 2026-09-25: A warm match on such a model is refused.
    assert!(!batched_reserve_hybrid_ssm_ok(
        &[cache_match(0), cache_match(48)],
        true,
    ));
    // 2026-09-25: A model without SSM layers admits warm matches.
    assert!(batched_reserve_hybrid_ssm_ok(
        &[cache_match(48), cache_match(48)],
        false,
    ));
    // 2026-09-25: An empty list is admitted.
    assert!(batched_reserve_hybrid_ssm_ok(&[], true));
}

#[test]
fn cache_batch_accepts_equal_partial_hits() {
    assert!(cache_batch_matches_compatible(
        &[cache_match(48), cache_match(48)],
        8192,
    ));
    assert!(!cache_batch_matches_compatible(&[], 8192));
}

#[test]
fn cache_batch_rejects_mixed_processing_geometry() {
    assert!(!cache_batch_matches_compatible(
        &[cache_match(0), cache_match(48)],
        8192,
    ));
    let mut fewer_blocks = cache_match(48);
    fewer_blocks.matched_blocks.pop();
    assert!(!cache_batch_matches_compatible(
        &[cache_match(48), fewer_blocks],
        8192,
    ));
}

#[test]
fn cache_batch_rejects_restore_metadata() {
    let mut snapshot = cache_match(48);
    snapshot.ssm_snapshot = Some(3);
    snapshot.ssm_snapshot_tokens = 48;
    assert!(!cache_batch_matches_compatible(
        &[cache_match(48), snapshot],
        8192,
    ));

    let mut disk = cache_match(48);
    disk.matched_disk_block_ids = vec![9; 3];
    assert!(!cache_batch_matches_compatible(
        &[cache_match(48), disk],
        8192,
    ));

    let mut snapshot_tokens = cache_match(48);
    snapshot_tokens.ssm_snapshot_tokens = 48;
    assert!(!cache_batch_matches_compatible(
        &[cache_match(48), snapshot_tokens],
        8192,
    ));

    let mut tier_key = cache_match(48);
    tier_key.ssm_snapshot_tier_key = Some(7);
    assert!(!cache_batch_matches_compatible(
        &[cache_match(48), tier_key],
        8192,
    ));

    let mut tier_tokens = cache_match(48);
    tier_tokens.ssm_snapshot_tier_tokens = 48;
    assert!(!cache_batch_matches_compatible(
        &[cache_match(48), tier_tokens],
        8192,
    ));
}

#[test]
fn cache_batch_rejects_full_chunk_hit() {
    assert!(!cache_batch_matches_compatible(
        &[cache_match(8192), cache_match(8192)],
        8192,
    ));
}

#[test]
fn rejects_under_two_streams() {
    assert!(!check_kernel_batched_eligible(
        std::iter::empty(),
        0,
        8192,
        false,
        256,
        BIG_SCRATCH,
        TOP_K,
        MROPE,
        false,
        false,
    ));
    assert!(!check_kernel_batched_eligible(
        vec![s(4096, 0, false)],
        1,
        8192,
        false,
        256,
        BIG_SCRATCH,
        TOP_K,
        MROPE,
        false,
        false,
    ));
}

#[test]
fn rejects_chunk_zero() {
    assert!(!check_kernel_batched_eligible(
        vec![s(4096, 0, false), s(4096, 0, false)],
        2,
        8192,
        false,
        256,
        BIG_SCRATCH,
        TOP_K,
        MROPE,
        false,
        false,
    ));
}

#[test]
fn accepts_chunk_zero_when_explicitly_allowed() {
    assert!(check_kernel_batched_eligible(
        vec![s(4096, 0, false), s(4096, 0, false)],
        2,
        8192,
        false,
        256,
        BIG_SCRATCH,
        TOP_K,
        MROPE,
        true,
        false,
    ));
}

#[test]
fn accepts_uniform_paged_n_2() {
    assert!(check_kernel_batched_eligible(
        vec![s(4096, 4096, false), s(4096, 4096, false)],
        2,
        8192,
        false,
        256,
        BIG_SCRATCH,
        TOP_K,
        MROPE,
        false,
        false,
    ));
}

#[test]
fn rejects_mismatched_chunk_len() {
    assert!(!check_kernel_batched_eligible(
        vec![s(4096, 4096, false), s(2048, 4096, false)],
        2,
        16384,
        false,
        256,
        BIG_SCRATCH,
        TOP_K,
        MROPE,
        false,
        false,
    ));
}

#[test]
fn rejects_mismatched_chunk_start() {
    assert!(!check_kernel_batched_eligible(
        vec![s(4096, 12288, false), s(4096, 4096, false)],
        2,
        16384,
        false,
        256,
        BIG_SCRATCH,
        TOP_K,
        MROPE,
        false,
        false,
    ));
}

#[test]
fn rejects_mismatched_is_last() {
    assert!(!check_kernel_batched_eligible(
        vec![s(4096, 4096, false), s(4096, 4096, true)],
        2,
        8192,
        false,
        256,
        BIG_SCRATCH,
        TOP_K,
        MROPE,
        false,
        false,
    ));
}

#[test]
fn rejects_arena_overflow() {
    assert!(!check_kernel_batched_eligible(
        vec![s(4096, 4096, false), s(4096, 4096, false)],
        2,
        4100,
        false,
        256,
        BIG_SCRATCH,
        TOP_K,
        MROPE,
        false,
        false,
    ));
}

#[test]
fn rejects_large_head_dim() {
    assert!(!check_kernel_batched_eligible(
        vec![s(4096, 4096, false), s(4096, 4096, false)],
        2,
        8192,
        false,
        512,
        BIG_SCRATCH,
        TOP_K,
        MROPE,
        false,
        false,
    ));
}

#[test]
fn accepts_varlen_batch_when_packed_footprint_fits() {
    // 2026-09-25: Varlen admits streams of different lengths. The arena is
    // charged their packed total, 13,649 tokens, which fits a 16,388-token
    // arena.
    let streams = [
        s(2051, 0, true),
        s(2953, 0, true),
        s(3863, 0, true),
        s(4782, 0, true),
    ];
    let arena: usize = 16_388;
    let scratch = metrale_gpu_runtime::buffers::q12_batched_scratch_bytes(
        metrale_gpu_runtime::buffers::Q12_SIZING_STREAMS,
        arena.div_ceil(metrale_gpu_runtime::buffers::Q12_SIZING_STREAMS),
        TOP_K,
        MROPE,
    );
    assert!(check_kernel_batched_eligible(
        streams, 4, arena, false, 128, scratch, TOP_K, MROPE, true, true,
    ));
}

#[test]
fn rejects_scratch_footprint_overflow() {
    // 2026-09-25: The staging footprint must fit in scratch even when the
    // arena check passes. For n=4, chunk_len=935, top_k=8 with MRoPE the
    // footprint is above 348,840 bytes, so that scratch is rejected, and a
    // scratch of `q12_batched_scratch_bytes(4, 935, 8, true)` is accepted.
    let streams = [s(935, 4096, false); 4];
    let arena = 4096;
    let too_small = 348_840;
    let enlarged = metrale_gpu_runtime::buffers::q12_batched_scratch_bytes(4, 935, 8, true);
    assert!(
        !check_kernel_batched_eligible(
            streams.iter().copied(),
            4,
            arena,
            false,
            256,
            too_small,
            8,
            true,
            false,
            false,
        ),
        "footprint must NOT fit in the old 348_840 B scratch"
    );
    assert!(
        check_kernel_batched_eligible(
            streams.iter().copied(),
            4,
            arena,
            false,
            256,
            enlarged,
            8,
            true,
            false,
            false,
        ),
        "footprint must fit once scratch is sized to it"
    );
}

/// 2026-09-25: Two 8192-token chunks charged at their raw length do not fit an
/// 8200-token arena.
#[test]
fn raw_charge_blocks_stacking_at_production_sizes() {
    assert!(!check_kernel_batched_eligible(
        vec![s(8192, 16, false), s(8192, 16, false)],
        2,
        8200,
        false,
        128,
        BIG_SCRATCH,
        TOP_K,
        MROPE,
        true,
        false,
    ));
}

/// 2026-09-25: The same two streams, warm: prefix hits leave 424 and 400 tokens
/// to stage, 824 of the 8200-token arena, and the batch is eligible.
#[test]
fn effective_charge_allows_warm_stacking() {
    assert!(check_kernel_batched_eligible(
        vec![s_eff(8192, 424, 16, false), s_eff(8192, 400, 16, false)],
        2,
        8200,
        false,
        128,
        BIG_SCRATCH,
        TOP_K,
        MROPE,
        true,
        false,
    ));
}

/// 2026-09-25: The effective charge is still a bound: warm streams whose staged
/// tokens exceed the arena in total are rejected.
#[test]
fn effective_charge_still_rejects_when_sum_exceeds_arena() {
    let streams: Vec<_> = (0..8).map(|_| s_eff(8192, 2000, 16, false)).collect();
    assert!(!check_kernel_batched_eligible(
        streams,
        8,
        8200,
        false,
        128,
        BIG_SCRATCH,
        TOP_K,
        MROPE,
        true,
        false,
    ));
}

/// 2026-09-25: A fully cached middle chunk stages zero tokens, and a batch with
/// such a stream is rejected: in the packed layout it would have no segment
/// and share its offset with the next stream.
#[test]
fn effective_charge_rejects_zero_length_stream() {
    assert!(!check_kernel_batched_eligible(
        vec![s_eff(2048, 176, 2048, false), s_eff(2048, 0, 2048, false)],
        2,
        8192,
        false,
        128,
        BIG_SCRATCH,
        TOP_K,
        MROPE,
        true,
        false,
    ));
}

/// 2026-09-25: Pins the MLA rejection at the config level, through
/// `config_is_mla`, which the production caller uses: a config with
/// `kv_lora_rank = 512` is rejected. The other tests pass `is_mla` as a bool,
/// so they would not notice a broken derivation.
#[test]
fn mistral_config_is_rejected_as_mla() {
    let mut cfg = metrale_config::ModelConfig::qwen3_next_80b_nvfp4();
    // 2026-09-25: Non-MLA baseline: the derivation says no and the batch is
    // admitted, so the rejection below comes from MLA alone.
    assert!(!config_is_mla(&cfg));
    let eligible = |is_mla: bool| {
        check_kernel_batched_eligible(
            vec![s(2048, 16, false), s(2048, 16, false)],
            2,
            8192,
            is_mla,
            128,
            BIG_SCRATCH,
            TOP_K,
            MROPE,
            true,
            false,
        )
    };
    assert!(eligible(config_is_mla(&cfg)));

    // 2026-09-25: The mistral parser copies `kv_lora_rank` from config.json
    // (`parsers/mistral.rs`), so the field set here is what serving reads.
    cfg.kv_lora_rank = 512;
    assert!(config_is_mla(&cfg));
    assert!(!eligible(config_is_mla(&cfg)));
}
