// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-25: Paged-decode attention launch geometry: the split-K count, the
//! split-K workspace size and the GQA-packed kernel gate, as pure functions
//! of configuration.
//!
//! The online-softmax split merge is not associative, so a split count that
//! moved with the co-batched sequence count would change a sequence's output
//! bytes depending on what it is batched with.
//!
//! Owner: kernels crate (paged-decode attention policy).
//! Invariants:
//! - Under [`SplitkPolicy::Auto`] and [`SplitkPolicy::Pinned`] the split count
//!   depends only on `(sm_count, num_q_heads)` and the pin; only
//!   [`SplitkPolicy::Legacy`] reads the reference batch.
//! - [`resolve_policy`] is the one resolution of the target declaration and
//!   `METRALE_ATTN_DECODE_SPLITK`; the metrale-model-layers dispatch and the
//!   metrale-gpu-runtime workspace sizing both call it.
//! - [`gqa_pack_shape_ok`] admits only `num_q_heads == num_kv_heads *
//!   DECODE_GQA_PACK_WIDTH` at `head_dim == DECODE_GQA_PACK_HEAD_DIM`.
//!
//! [`SplitkPolicy::Legacy`] computes, with `NUM_SMS` standing for the target's
//! `sm_count`:
//!
//! ```text
//! current_ctas = num_q_heads * split_ref_seqs(num_seqs, max_decode_seqs);
//! num_splits   = if current_ctas >= NUM_SMS { 1 } else { NUM_SMS / current_ctas };
//! ```
//!
//! Short contexts are handled inside the Hopper split-K kernel, which floors
//! each sequence's split size at `PD_MIN_KV_PER_SPLIT` from that sequence's own
//! `seq_len` (`kernels/hopper/common/paged_decode_splitk_hopper.cuh`).

/// 2026-09-25: Waves of CTAs [`SplitkPolicy::Auto`] aims to put on the device
/// with one sequence in flight (see [`auto_splits`]).
pub const SPLITK_TARGET_WAVES: u32 = 2;

/// 2026-09-25: Ceiling on the split count under every policy but `Legacy`,
/// and so on the split-K workspace, which is
/// `rows * num_q_heads * num_splits * (head_dim + 2)` F32 values. Without it
/// a model with few q heads would ask for `2 * sm_count / num_q_heads` splits
/// (264 for one head on 132 SMs). At 24 q heads on the 148-SM targets
/// (`kernels/b200`, `kernels/b300`) `auto` asks for 13. A pinned count is
/// clamped to it.
pub const MAX_DECODE_SPLITS: u32 = 16;

/// 2026-09-25: How a target picks its paged-decode split count.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SplitkPolicy {
    /// 2026-09-25: `sm_count / (num_q_heads * ref_seqs)`, or 1 when that
    /// product already covers the device ([`legacy_splits`]). Reads the
    /// reference batch. The build default for a target that declares none.
    Legacy,
    /// 2026-09-25: Fill [`SPLITK_TARGET_WAVES`] waves with one sequence in
    /// flight: `clamp(ceil(WAVES * sm_count / num_q_heads), 1,
    /// MAX_DECODE_SPLITS)`.
    Auto,
    /// 2026-09-25: A pinned count, clamped to `1..=MAX_DECODE_SPLITS`. `0`,
    /// `off`, `false` and `no` parse to `Pinned(1)`: one split is no split-K.
    Pinned(u32),
}

impl SplitkPolicy {
    /// 2026-09-25: The spelling [`parse`] reads back as this policy, printed
    /// in the `target defaults (<hw>): …` line.
    pub fn label(self) -> String {
        match self {
            SplitkPolicy::Legacy => "legacy".to_string(),
            SplitkPolicy::Auto => "auto".to_string(),
            SplitkPolicy::Pinned(n) => n.to_string(),
        }
    }
}

/// 2026-09-25: `legacy` | `auto` | `0`/`off`/`false`/`no` | a decimal count
/// (clamped to `1..=MAX_DECODE_SPLITS`), trimmed and case-insensitive.
///
/// `None` for anything else; [`resolve_policy`] then keeps the declaration.
pub fn parse(spelling: &str) -> Option<SplitkPolicy> {
    match spelling.trim().to_ascii_lowercase().as_str() {
        "legacy" => Some(SplitkPolicy::Legacy),
        "auto" => Some(SplitkPolicy::Auto),
        "0" | "off" | "false" | "no" => Some(SplitkPolicy::Pinned(1)),
        other => other
            .parse::<u32>()
            .ok()
            .map(|n| SplitkPolicy::Pinned(n.clamp(1, MAX_DECODE_SPLITS))),
    }
}

/// 2026-09-25: The target's declaration, overridden by a parseable
/// `METRALE_ATTN_DECODE_SPLITK`. An unparseable declaration is `Legacy`.
///
/// Returns `(policy, came_from_env)`. Both the dispatch (metrale-model-layers
/// `target_defaults::resolve`) and the workspace sizing (via
/// [`policy_from_env`]) call this, so the split count a launch uses and the
/// one the workspace was sized for come from the same rule.
pub fn resolve_policy(declared: &str, env_raw: Option<&str>) -> (SplitkPolicy, bool) {
    let declared = parse(declared).unwrap_or(SplitkPolicy::Legacy);
    match env_raw.and_then(parse) {
        Some(p) => (p, true),
        None => (declared, false),
    }
}

/// 2026-09-25: [`resolve_policy`] against the process environment and this
/// binary's baked declaration. Used by the split-K workspace sizing in
/// metrale-gpu-runtime (`sizes.rs`), which metrale-model-layers depends on and
/// so cannot call the model-layers resolver.
pub fn policy_from_env() -> SplitkPolicy {
    resolve_policy(
        crate::TARGET_DEFAULTS.attn_decode_splitk,
        std::env::var("METRALE_ATTN_DECODE_SPLITK").ok().as_deref(),
    )
    .0
}

/// 2026-09-25: `clamp(ceil(WAVES * sm_count / num_q_heads), 1,
/// MAX_DECODE_SPLITS)`.
///
/// The split-K grid is `(num_q_heads, num_splits, num_seqs)`, so with one
/// sequence this puts at least `WAVES * sm_count` CTAs on the device when the
/// cap allows. A wider batch launches more CTAs with the same count.
pub fn auto_splits(sm_count: u32, num_q_heads: u32) -> u32 {
    let heads = num_q_heads.max(1);
    let target = SPLITK_TARGET_WAVES.saturating_mul(sm_count.max(1));
    target.div_ceil(heads).clamp(1, MAX_DECODE_SPLITS)
}

/// 2026-09-25: The [`SplitkPolicy::Legacy`] rule. The dispatch passes
/// `ref_seqs = split_ref_seqs(num_seqs, max_decode_seqs)`, which is
/// `max(max_decode_seqs, num_seqs)`.
pub fn legacy_splits(sm_count: u32, num_q_heads: u32, ref_seqs: u32) -> u32 {
    let current_ctas = num_q_heads.max(1).saturating_mul(ref_seqs.max(1));
    if current_ctas >= sm_count {
        1
    } else {
        sm_count / current_ctas
    }
}

/// 2026-09-25: The split count for a launch.
///
/// `legacy_ref_seqs` is read by [`SplitkPolicy::Legacy`] only; the other arms
/// do not depend on the batch, which
/// `the_auto_split_count_does_not_move_with_the_co_batched_count` tests.
pub fn num_splits(
    policy: SplitkPolicy,
    sm_count: u32,
    num_q_heads: u32,
    legacy_ref_seqs: u32,
) -> u32 {
    match policy {
        SplitkPolicy::Legacy => legacy_splits(sm_count, num_q_heads, legacy_ref_seqs),
        SplitkPolicy::Auto => auto_splits(sm_count, num_q_heads),
        SplitkPolicy::Pinned(n) => n.clamp(1, MAX_DECODE_SPLITS),
    }
}

/// 2026-09-25: `[o[head_dim], m, l]` slots the split-K workspace must hold.
///
/// The split-K kernel addresses
/// `((seq * num_q_heads) + head) * num_splits + split`, so `max_rows` must be
/// the widest batch the decode metadata accepts (`sizes.rs` passes
/// `DecodeMetaLayout::rows()`); a short allocation is an out-of-bounds device
/// write with no error.
///
/// `Legacy` needs `sm_count` slots: when it splits, `num_q_heads * ref_seqs <
/// sm_count` with `ref_seqs >= num_seqs`, so `num_seqs * num_q_heads *
/// num_splits <= sm_count`; at one split the non-split kernel runs and the
/// workspace is not touched.
pub fn workspace_slots(
    policy: SplitkPolicy,
    sm_count: u32,
    num_q_heads: u32,
    max_rows: u32,
    legacy_ref_seqs: u32,
) -> u32 {
    match policy {
        SplitkPolicy::Legacy => sm_count.max(1),
        other => max_rows
            .max(1)
            .saturating_mul(num_q_heads.max(1))
            .saturating_mul(num_splits(other, sm_count, num_q_heads, legacy_ref_seqs)),
    }
}

// 2026-09-25: GQA-packed paged decode. The unpacked `paged_decode_attn{,_fp8}`
// kernels launch `grid (num_q_heads, num_seqs)`, and the CTAs of one GQA group
// each read that KV head's whole K and V. The packed kernels
// (`kernels/gb10/common/paged_decode_attn_{fp8,bf16}_gqa.cu`) put the group in
// one CTA and launch `grid (num_kv_heads, num_seqs)`. The dispatch uses them
// only on the non-split arm: under split-K, packing would change the split
// count and so the online-softmax merge tree.

/// 2026-09-25: Query heads one packed CTA carries: the Rust copy of the
/// kernels' `#define PD_GQA`, which sizes their `q_reg`/`o_reg` register
/// arrays at compile time. `cuda_sources_declare_the_pack_width_rust_dispatches_on`
/// fails if the two differ.
///
/// 6 is Qwen3.8-27B's ratio (`q_heads = 24`, `kv_heads = 4` in
/// `kernels/hopper/qwen3.8-27b/MODEL.toml`). [`gqa_pack_shape_ok`] refuses any
/// other ratio, which keeps the unpacked kernel.
pub const DECODE_GQA_PACK_WIDTH: u32 = 6;

/// 2026-09-25: The head dim the packed kernels are compiled for: both sources
/// `#define PD_HDIM 256`, and [`gqa_pack_shape_ok`] refuses any other head dim.
pub const DECODE_GQA_PACK_HEAD_DIM: u32 = 256;

/// 2026-09-25: The packed kernels' declared state: off, for every target.
/// `METRALE_ATTN_DECODE_GQA_PACK` overrides it ([`resolve_gqa_pack`]).
pub const DECODE_GQA_PACK_DECLARED: bool = false;

/// 2026-09-25: `1`/`on`/`true`/`yes` | `0`/`off`/`false`/`no`, trimmed and
/// case-insensitive.
///
/// `None` for anything else, so an unparseable value keeps the declaration.
pub fn parse_gqa_pack(spelling: &str) -> Option<bool> {
    match spelling.trim().to_ascii_lowercase().as_str() {
        "1" | "on" | "true" | "yes" => Some(true),
        "0" | "off" | "false" | "no" => Some(false),
        _ => None,
    }
}

/// 2026-09-25: `declared`, overridden by a parseable
/// `METRALE_ATTN_DECODE_GQA_PACK` value. Pure; [`gqa_pack_enabled`] supplies
/// the environment.
pub fn resolve_gqa_pack(declared: bool, env_raw: Option<&str>) -> bool {
    env_raw.and_then(parse_gqa_pack).unwrap_or(declared)
}

/// 2026-09-25: Whether the packed kernels are armed in this process, resolved
/// once. The dispatch reads it on every non-split BF16 and FP8 paged-decode
/// launch, and a fixed answer keeps a captured CUDA graph consistent with
/// later launches.
pub fn gqa_pack_enabled() -> bool {
    static ARMED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ARMED.get_or_init(|| {
        resolve_gqa_pack(
            DECODE_GQA_PACK_DECLARED,
            std::env::var("METRALE_ATTN_DECODE_GQA_PACK")
                .ok()
                .as_deref(),
        )
    })
}

/// 2026-09-25: Whether this launch's shape is one the packed kernels can
/// serve. Each condition is a precondition of the kernel:
///
/// * `num_kv_heads > 0`: it is the grid's x extent.
/// * `num_q_heads == num_kv_heads * DECODE_GQA_PACK_WIDTH`: the kernel takes
///   its query heads from `kv_head * PD_GQA`, so any other ratio reads and
///   writes the wrong heads. The test multiplies because division truncates:
///   25 q heads over 4 kv heads divide to 6.
/// * `head_dim == DECODE_GQA_PACK_HEAD_DIM`.
///
/// When this is false the caller keeps the unpacked kernel.
pub fn gqa_pack_shape_ok(num_q_heads: u32, num_kv_heads: u32, head_dim: u32) -> bool {
    num_kv_heads > 0
        && num_q_heads == num_kv_heads.saturating_mul(DECODE_GQA_PACK_WIDTH)
        && head_dim == DECODE_GQA_PACK_HEAD_DIM
}

/// 2026-09-25: CTAs a packed launch puts on the device: `num_kv_heads *
/// num_seqs`, a `DECODE_GQA_PACK_WIDTH`-fold cut from the unpacked
/// `num_q_heads * num_seqs`. With 4 KV heads on gb10's 48 SMs the packed grid
/// covers the SMs only from `num_seqs = 12`.
pub fn gqa_pack_ctas(num_kv_heads: u32, num_seqs: u32) -> u32 {
    num_kv_heads.saturating_mul(num_seqs)
}

#[cfg(test)]
#[path = "attn_splitk_tests.rs"]
mod tests;
