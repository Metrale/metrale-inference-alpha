// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-26: Top-k token-mask prewarm for a new grammar, run on its own
//! thread so it overlaps the prompt prefill.
//!
//! [`metrale_grammar::CompiledGrammar::compile_top_k_masks`] fills the mask
//! cache for the costliest parser states. `GrammarState` starts it when it is
//! built, and the scheduler builds the grammar state before the prompt's
//! forward pass, so the prewarm runs while prefill does. The first bitmask
//! fill, or `completion_token_ids`, joins it. The walk itself runs on one
//! thread: the engine builds its compiler with `max_threads = 1`.
//!
//! `METRALE_GRAMMAR_ASYNC_PREWARM=0` (or `false`, `off`, `no`) runs the
//! prewarm inline instead, on the thread that builds the grammar state.
//!
//! Owner: server (grammar).
//! Invariants:
//! - A `Pending` prewarm's thread only fills the mask cache and then calls
//!   the save hook; it never touches the matcher.

use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

use metrale_grammar::CompiledGrammar;

use super::PrewarmHook;

/// 2026-09-26: A barrier the background worker waits on before any work.
/// Only `GrammarState::build_gated`, called only by the ordering test
/// (`tests/prewarm_ordering.rs`), passes one; it holds the prewarm open. It
/// is an argument, not a global, so a gated state cannot stall another one.
pub(super) type PrewarmGate = Arc<std::sync::Barrier>;

/// 2026-09-26: A prewarm still running on its own thread, or finished.
///
/// Dropping a `Pending` prewarm (an abandoned request) detaches the worker
/// instead of cancelling it: the work is one `compile_top_k_masks` pass
/// over its own `CompiledGrammar` clone, and the masks it computes stay in
/// the cache for later requests.
pub(super) enum Prewarm {
    /// 2026-09-26: Running. The handle yields the number of masks warmed;
    /// `finished` is set once the worker is done, readable without joining.
    Pending {
        handle: std::thread::JoinHandle<usize>,
        finished: Arc<AtomicBool>,
    },
    /// 2026-09-26: Joined, or run inline (overlap off, or the thread could
    /// not be spawned).
    Done(usize),
}

impl Prewarm {
    /// 2026-09-26: Start warming the `k` costliest masks of `compiled`, with
    /// the process's overlap setting and no gate or hook. Returns at once
    /// when overlap is on and the thread spawns; otherwise warms inline.
    pub(super) fn start(compiled: &CompiledGrammar, k: usize) -> Self {
        Self::start_with(compiled, k, None, overlap_enabled(), None)
    }

    /// 2026-09-26: [`Self::start`] with the overlap decision, gate and save
    /// hook passed in, so a test can drive both paths without setting a
    /// process-wide env var. `on_warm` runs after the masks are warm, except
    /// when the thread fails to spawn: that inline fallback drops it.
    pub(super) fn start_with(
        compiled: &CompiledGrammar,
        k: usize,
        gate: Option<PrewarmGate>,
        overlap: bool,
        on_warm: Option<PrewarmHook>,
    ) -> Self {
        if !overlap {
            // 2026-09-26: Inline: the gate is only for the background worker,
            // so it is dropped, not waited on.
            drop(gate);
            let warmed = compiled.compile_top_k_masks(k);
            if let Some(hook) = on_warm {
                hook(warmed);
            }
            return Prewarm::Done(warmed);
        }
        let owned = compiled.clone();
        let finished = Arc::new(AtomicBool::new(false));
        let done_flag = Arc::clone(&finished);
        let spawned = std::thread::Builder::new()
            .name("grammar-prewarm".to_string())
            .spawn(move || {
                if let Some(gate) = gate {
                    gate.wait();
                }
                let warmed = owned.compile_top_k_masks(k);
                // 2026-09-26: The hook runs after the masks are in the cache
                // and before `finished` is set. It only spawns the writer
                // thread (`mask_cache.rs`), so the join does not wait on I/O.
                if let Some(hook) = on_warm {
                    hook(warmed);
                }
                done_flag.store(true, Ordering::Release);
                warmed
            });
        match spawned {
            Ok(handle) => Prewarm::Pending { handle, finished },
            // 2026-09-26: No thread: warm inline rather than leave the
            // masks cold.
            Err(e) => {
                tracing::debug!("Grammar: prewarm thread spawn failed ({e}); warming inline");
                Prewarm::Done(compiled.compile_top_k_masks(k))
            }
        }
    }

    /// 2026-09-26: Block until the worker is done and return the number of
    /// masks it warmed. After the first call it returns the stored count.
    pub(super) fn wait(&mut self) -> usize {
        match self {
            Prewarm::Done(n) => *n,
            Prewarm::Pending { .. } => {
                let Prewarm::Pending { handle, .. } = std::mem::replace(self, Prewarm::Done(0))
                else {
                    unreachable!("just matched Pending");
                };
                // 2026-09-26: A panicked prewarm is logged and counts as
                // zero; the matcher computes any missing mask on demand.
                let warmed = handle.join().unwrap_or_else(|_| {
                    tracing::warn!("Grammar: background mask prewarm panicked; masks stay lazy");
                    0
                });
                *self = Prewarm::Done(warmed);
                warmed
            }
        }
    }

    /// 2026-09-26: Whether the worker has finished, without joining it.
    #[cfg(test)]
    pub(super) fn is_finished(&self) -> bool {
        match self {
            Prewarm::Done(_) => true,
            Prewarm::Pending { finished, .. } => finished.load(Ordering::Acquire),
        }
    }
}

/// 2026-09-26: The overlap setting for the `GrammarState` constructors.
/// See [`overlap_enabled`].
pub(super) fn overlap_enabled_for_serve() -> bool {
    overlap_enabled()
}

/// 2026-09-26: Whether the prewarm runs on its own thread. The env var is
/// read on the first call and cached for the life of the process.
fn overlap_enabled() -> bool {
    static ENABLED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ENABLED.get_or_init(|| {
        overlap_from_env(
            std::env::var("METRALE_GRAMMAR_ASYNC_PREWARM")
                .ok()
                .as_deref(),
        )
    })
}

/// 2026-09-26: `METRALE_GRAMMAR_ASYNC_PREWARM` value `0`, `false`, `off` or
/// `no` (trimmed, any case) turns overlap off. Anything else, including
/// unset, turns it on.
pub(super) fn overlap_from_env(value: Option<&str>) -> bool {
    !matches!(
        value
            .unwrap_or_default()
            .trim()
            .to_ascii_lowercase()
            .as_str(),
        "0" | "false" | "off" | "no"
    )
}
