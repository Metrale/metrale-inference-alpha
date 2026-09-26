// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-25: Whether the radix walk may match a tail that is not
//! block-aligned. The only reader of `METRALE_PREFIX_SUBBLOCK`.
//!
//! # Why it is off by default
//!
//! Both sub-block arms of [`super::inner::RadixTreeInner::walk`] end a match
//! inside a KV block and hand out that block's index, so the requester's next
//! K/V rows are written into a block another owner still uses.
//! `PagedKvCache::inc_ref` only counts; nothing copies the block first.
//!
//! * partial-suffix arm: the block is the tail of a cached sequence's prompt,
//!   and that sequence may still be decoding into it.
//! * child-key arm: the block is a full node block that other sequences read
//!   as prompt K/V; the requester writes offsets `[remainder, block_size)`.
//!
//! Two writers that stay in lockstep write the same bytes; once they diverge,
//! readers attend over a mix of both. Measured 2026-08-21 on qwen3.8-27B with
//! DFlash2 at C=4: 0 of 4 correct, with three sequences emitting the same
//! wrong text.
//!
//! Off costs up to `block_size - 1` tokens of prefill recompute per hit.
//! `docs/ROBUSTNESS.md` records that turning the lever off moved 0 of 251
//! agentic samples.
//!
//! Insert still stores the partial tail block in `RadixNode::partial_suffix`
//! and holds a KV ref on it, so `=1` needs no insert-side change; the cost is
//! one block per cached prompt whose length is not block-aligned, held until
//! evicted.
//!
//! Owner: cache.
//! Invariants: unset, `0` and any unrecognised value all mean off.

/// 2026-09-25: `METRALE_PREFIX_SUBBLOCK=1` turns sub-block tail matching on
/// (unsound; see the module docs). Read once per process; every
/// `RadixTreeInner::new` after that gets the cached value.
pub(super) fn partial_tail_sharing_enabled() -> bool {
    static ENABLED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ENABLED.get_or_init(|| enabled_from(std::env::var("METRALE_PREFIX_SUBBLOCK").ok().as_deref()))
}

/// 2026-09-25: The lever's parse, testable without the environment. Only
/// `"1"` is on. Unset and `"0"` are off; any other value is logged as an
/// error and is off, so a typo cannot turn the unsound path on.
fn enabled_from(raw: Option<&str>) -> bool {
    match raw {
        None => false,
        Some("1") => true,
        Some("0") => false,
        Some(other) => {
            tracing::error!(
                "METRALE_PREFIX_SUBBLOCK={other:?} is not a recognised value (expected \"0\" or \
                 \"1\") — refusing it and leaving sub-block tail matching OFF (see issue #1193)"
            );
            false
        }
    }
}

#[cfg(test)]
mod tests {
    use super::enabled_from;

    /// 2026-09-25: Unset is off; only an explicit `1` is on.
    #[test]
    fn unset_is_off_and_only_an_explicit_one_turns_it_on() {
        assert!(!enabled_from(None), "unset must be OFF");
        assert!(enabled_from(Some("1")), "an explicit 1 is the only ON");
        assert!(!enabled_from(Some("0")));
    }

    /// 2026-09-25: Near-misses of `1` (`true`, `on`, `1 `, `01`, ...) are off.
    #[test]
    fn an_unrecognised_value_is_refused_rather_than_coerced_on() {
        for raw in ["true", "yes", "on", "", "1 ", "01", "ON"] {
            assert!(
                !enabled_from(Some(raw)),
                "{raw:?} must not arm sub-block tail matching"
            );
        }
    }
}
