// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-25: `usage.prompt_tokens_details.cached_tokens` accounting. Logical
//! child of `phase_promote_prefills` via `#[path]`.
//!
//! Owner: scheduler.
//! Invariants: none beyond the types.

/// 2026-09-25: `usage.prompt_tokens_details.cached_tokens` must report the
/// prompt tokens whose KV was reused, not the prefix-cache lookup length. The
/// lookup length lives on `cached_prefix_tokens` and is load-bearing for block
/// refcounting, so the report sites must read `reused_prefix_tokens` instead:
/// `cached_prefix_tokens` is non-zero even when the prefill recomputed every
/// token (no SSM snapshot / exact-leaf bypass).
#[test]
fn cached_tokens_is_reported_from_reused_prefix_tokens() {
    for (name, src) in [
        (
            "phase_promote_prefills.rs",
            include_str!("phase_promote_prefills.rs"),
        ),
        ("prefill_a_step.rs", include_str!("prefill_a_step.rs")),
        ("prefill_b_step.rs", include_str!("prefill_b_step.rs")),
        (
            "prefill_a_step/first_chunk.rs",
            include_str!("prefill_a_step/first_chunk.rs"),
        ),
        (
            "prefill_b_step/first_token.rs",
            include_str!("prefill_b_step/first_token.rs"),
        ),
    ] {
        let compact: String = src.chars().filter(|c| !c.is_whitespace()).collect();
        assert!(
            !compact.contains("cached_prompt_tok=p.seq.cached_prefix_tokens")
                && !compact.contains("cached_prompt_tok=seq.cached_prefix_tokens"),
            "{name} still reports the prefix-cache LOOKUP length as cached_tokens (#919)"
        );
        assert!(
            compact.contains("cached_prompt_tok=p.seq.reused_prefix_tokens")
                || compact.contains("cached_prompt_tok=seq.reused_prefix_tokens"),
            "{name} must report the tokens actually reused from the prefix cache"
        );
    }
}
