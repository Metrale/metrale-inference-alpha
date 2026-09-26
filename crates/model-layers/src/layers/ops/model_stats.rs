// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-25: [`ModelStats`]: diagnostic counters and one-shot latches owned by one model.
//!
//! Owner: model-layers ops.
//! Invariants:
//! - A new `ModelStats` starts with zero counters and every latch unfired, so a second model
//!   neither inherits the first one's counts nor finds its dumps already spent.
//!
//! `TransformerModel` owns one next to its [`ModelLevers`](super::ModelLevers) and lends it to
//! each `ForwardContext` as `stats`. The counters are atomics because they are updated through that
//! shared reference.

use std::sync::atomic::AtomicU64;

/// 2026-09-25: Per-model diagnostic state.
#[derive(Debug, Default)]
pub struct ModelStats {
    /// 2026-09-25: MoE expert-union sampling under `ModelLevers::moe_union_stats`
    /// (`moe/union_stats.rs`).
    pub moe_union: MoeUnionStats,
    /// 2026-09-25: One-shot latches for dumps and once-per-model log lines.
    pub dumped: DumpLatches,
}

/// 2026-09-25: Expert-union sampling counters for one model: calls seen, calls sampled, and the
/// running unique-expert and slot totals.
#[derive(Debug, Default)]
pub struct MoeUnionStats {
    pub calls: AtomicU64,
    pub samples: AtomicU64,
    pub unique_sum: AtomicU64,
    pub slots_sum: AtomicU64,
}

/// 2026-09-25: One-shot latches for one model, keyed by a `&'static str`
/// ([`keyed`](DumpLatches::keyed)).
#[derive(Debug, Default)]
pub struct DumpLatches {
    /// 2026-09-25: The keys that have fired.
    fired: std::sync::Mutex<std::collections::BTreeSet<&'static str>>,
}

impl ModelStats {
    pub fn new() -> Self {
        Self::default()
    }

    /// 2026-09-25: `true` the first time this model reaches `key`, `false` after; the same latch
    /// set as [`DumpLatches::keyed`]. Prefix keys by purpose (`"log:..."`, `"dump:..."`) so two
    /// unrelated sites cannot share one.
    pub fn once(&self, key: &'static str) -> bool {
        self.dumped.keyed(key)
    }
}

impl DumpLatches {
    /// 2026-09-25: `true` exactly once per model for `key`. Calling it consumes the shot, so test
    /// the dump's lever first; a disabled dump then does not spend it.
    pub fn keyed(&self, key: &'static str) -> bool {
        self.fired
            .lock()
            .expect("dump latches poisoned")
            .insert(key)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::Ordering;

    #[test]
    fn a_keyed_latch_fires_once_per_key_per_model() {
        let a = ModelStats::new();
        assert!(a.dumped.keyed("dflash_block"));
        assert!(!a.dumped.keyed("dflash_block"), "and only once");
        assert!(a.dumped.keyed("dflash_ctx"), "a different dump is separate");
        assert!(
            ModelStats::new().dumped.keyed("dflash_block"),
            "and a new model re-arms every key"
        );
    }

    #[test]
    fn two_models_count_expert_unions_independently() {
        let a = ModelStats::new();
        let b = ModelStats::new();
        a.moe_union.calls.fetch_add(9, Ordering::Relaxed);
        assert_eq!(a.moe_union.calls.load(Ordering::Relaxed), 9);
        assert_eq!(
            b.moe_union.calls.load(Ordering::Relaxed),
            0,
            "a second model starts clean rather than inheriting a mean"
        );
    }
}
