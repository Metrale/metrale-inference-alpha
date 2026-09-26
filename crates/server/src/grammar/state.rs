// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-26: Per-request grammar matching state: the matcher, its token
//! bitmask, and the mask prewarm it starts.
//!
//! Owner: server (grammar).
//! Invariants:
//! - `accept_token`, `rollback` and `reset` clear the per-position bitmask
//!   cache before anything else, so a cached fill always belongs to the
//!   matcher's current position.
//! - Every path that fills the bitmask or searches for a completion joins
//!   the prewarm first.

use metrale_grammar::{
    CompiledGrammar, GrammarMatcher, allocate_token_bitmask, reset_token_bitmask,
};

use super::engine::GrammarError;
use super::prewarm::Prewarm;

/// 2026-09-26: How many token masks the prewarm computes when a grammar
/// state is built. `CompiledGrammar::compile_top_k_masks` ranks the
/// reachable scanable parser states by their char-range edge count and
/// warms the top `k`, capped at the number of such states, so a value above
/// a grammar's state count warms all of them. See [`super::prewarm`].
const FORCED_TOKEN_TOP_K: usize = 512;

/// 2026-09-26: Per-request grammar matching state: a [`GrammarMatcher`]
/// and one bitmask buffer reused for every fill.
pub struct GrammarState {
    /// 2026-09-26: Timing sink for the `GrammarFill` and `ForcedTok`
    /// phases. `build_with` installs a disarmed `RunTiming::default()`;
    /// `with_timing` replaces it, and its only caller passes the same.
    timing: std::sync::Arc<crate::scheduler::mtp_timing::RunTiming>,
    matcher: GrammarMatcher,
    /// 2026-09-26: The top-k mask prewarm. [`Self::await_masks`] joins it
    /// before a bitmask fill or a completion search.
    prewarm: Prewarm,
    /// 2026-09-26: `ceil(vocab_size / 32)` words, one bit per token.
    bitmask_data: Box<[i32]>,
    vocab_size: usize,
    /// 2026-09-26: Stop/EOS token ids that [`Self::accept_token`] accepts
    /// without feeding them to the matcher, which refuses a stop token
    /// while the root rule is incomplete. Empty unless set with
    /// [`Self::with_stop_tokens`]; `compile_grammar_state` sets the
    /// request's `eos_tokens`.
    stop_tokens: Box<[u32]>,
    /// 2026-09-26: True when `bitmask_data` holds the fill for the
    /// matcher's current position and `bitmask_fill_result` its return
    /// value, so later fills at the same position (`forced_token`, the
    /// sampling mask, `stop_legal`) reuse it. Cleared by `accept_token`
    /// (on every call, even one that does not move the matcher), `rollback`
    /// and `reset`.
    bitmask_valid: bool,
    bitmask_fill_result: bool,
}

impl GrammarState {
    /// 2026-09-26: Build the state for `compiled`, with no stop tokens and
    /// no save hook. `vocab_size` sizes the bitmask and must be the
    /// engine's [`super::GrammarEngine::vocab_size`].
    ///
    /// Starts the prewarm of the costliest token masks through
    /// [`CompiledGrammar::compile_top_k_masks`]. The prewarm only fills the
    /// mask cache; the matcher's accept and fill results do not depend on it.
    pub fn new(compiled: &CompiledGrammar, vocab_size: usize) -> Result<Self, GrammarError> {
        Self::with_timing(compiled, vocab_size, std::sync::Arc::default())
    }

    /// 2026-09-26: [`Self::new`] plus the hook the prewarm calls once the
    /// masks are warm (the on-disk mask save). `compile_grammar_state` uses
    /// this constructor.
    pub fn new_with_hook(
        compiled: &CompiledGrammar,
        vocab_size: usize,
        on_warm: Option<super::PrewarmHook>,
    ) -> Result<Self, GrammarError> {
        Self::build_with(
            compiled,
            vocab_size,
            None,
            super::prewarm::overlap_enabled_for_serve(),
            on_warm,
        )
    }

    /// 2026-09-26: [`Self::new`] with a caller-supplied timing sink for the
    /// `GrammarFill` and `ForcedTok` phases. Its only caller is
    /// [`Self::new`], which passes a disarmed default.
    pub fn with_timing(
        compiled: &CompiledGrammar,
        vocab_size: usize,
        timing: std::sync::Arc<crate::scheduler::mtp_timing::RunTiming>,
    ) -> Result<Self, GrammarError> {
        let mut state = Self::build(compiled, vocab_size)?;
        state.timing = timing;
        Ok(state)
    }

    fn build(compiled: &CompiledGrammar, vocab_size: usize) -> Result<Self, GrammarError> {
        Self::build_with(
            compiled,
            vocab_size,
            None,
            super::prewarm::overlap_enabled_for_serve(),
            None,
        )
    }

    /// 2026-09-26: Build with overlap on and the prewarm worker parked on
    /// `gate` until the caller releases it. Only the ordering test calls it.
    pub(super) fn build_gated(
        compiled: &CompiledGrammar,
        vocab_size: usize,
        gate: Option<super::prewarm::PrewarmGate>,
    ) -> Result<Self, GrammarError> {
        Self::build_with(compiled, vocab_size, gate, true, None)
    }

    /// 2026-09-26: The constructor all others call, with the overlap
    /// decision, gate and save hook passed in, so tests can drive the
    /// overlapped and inline paths without touching the environment.
    pub(super) fn build_with(
        compiled: &CompiledGrammar,
        vocab_size: usize,
        gate: Option<super::prewarm::PrewarmGate>,
        overlap: bool,
        on_warm: Option<super::PrewarmHook>,
    ) -> Result<Self, GrammarError> {
        let matcher = GrammarMatcher::new(
            compiled, None,  // 2026-09-26: stop tokens from the grammar's tokenizer info.
            false, // 2026-09-26: terminate only on an accepted stop token.
            -1,
        )
        .map_err(GrammarError::Compilation)?;

        let prewarm = Prewarm::start_with(compiled, FORCED_TOKEN_TOP_K, gate, overlap, on_warm);

        let bitmask_data = allocate_token_bitmask(1, vocab_size);

        Ok(Self {
            timing: std::sync::Arc::default(),
            matcher,
            prewarm,
            bitmask_data,
            vocab_size,
            stop_tokens: Box::new([]),
            bitmask_valid: false,
            bitmask_fill_result: false,
        })
    }

    /// 2026-09-26: Set the stop/EOS token ids that [`Self::accept_token`]
    /// accepts without the matcher. See the `stop_tokens` field.
    #[must_use]
    pub fn with_stop_tokens(mut self, stop_tokens: &[u32]) -> Self {
        self.stop_tokens = stop_tokens.to_vec().into_boxed_slice();
        self
    }

    /// 2026-09-26: Whether the prewarm has finished, without joining it.
    #[cfg(test)]
    pub(super) fn prewarm_finished(&self) -> bool {
        self.prewarm.is_finished()
    }

    /// 2026-09-26: Join the prewarm, and log at debug how many masks it
    /// warmed. Called by `fill_bitmask` and `completion_token_ids` before
    /// they read the matcher.
    fn await_masks(&mut self) {
        let warmed = self.prewarm.wait();
        if warmed > 0 {
            tracing::debug!(
                warmed,
                requested = FORCED_TOKEN_TOP_K,
                "Grammar: pre-warmed top-k token masks overlapped with prefill"
            );
        }
    }

    /// 2026-09-26: Fill the allowed-token bitmask for the next position.
    ///
    /// Returns `true` when the mask rules out at least one token, `false`
    /// when every token is legal or the matcher has terminated; on `false`
    /// the caller skips the mask. A second call at the same position returns
    /// the cached result without refilling.
    pub fn fill_bitmask(&mut self) -> bool {
        // 2026-09-26: `fill_next_token_bitmask` panics on a terminated
        // matcher; after termination there is nothing left to constrain.
        if self.matcher.is_terminated() {
            return false;
        }
        if self.bitmask_valid {
            return self.bitmask_fill_result;
        }
        self.await_masks();
        let t_fill = std::time::Instant::now();
        reset_token_bitmask(&mut self.bitmask_data);
        let filled = self
            .matcher
            .fill_next_token_bitmask(&mut self.bitmask_data, 0, false);
        self.timing
            .record(crate::scheduler::mtp_timing::Phase::GrammarFill, t_fill);
        self.bitmask_valid = true;
        self.bitmask_fill_result = filled;
        filled
    }

    /// 2026-09-26: The bitmask, `ceil(vocab_size / 32)` `i32` words. Token
    /// `t` is allowed when bit `t % 32` of word `t / 32` is set.
    pub fn bitmask_data(&self) -> &[i32] {
        &self.bitmask_data
    }

    /// 2026-09-26: Whether the last fill allows `token_id`; `false` for an
    /// id past the bitmask.
    pub fn is_token_allowed(&self, token_id: u32) -> bool {
        let word = (token_id / 32) as usize;
        let bit = token_id % 32;
        if word >= self.bitmask_data.len() {
            return false;
        }
        (self.bitmask_data[word] & (1i32 << bit)) != 0
    }

    /// 2026-09-26: Accept a sampled token and advance the matcher.
    ///
    /// Returns whether the grammar accepts the token. Two cases return
    /// `true` without moving the matcher (no history step): a terminated
    /// matcher, which would refuse every token, and a token in
    /// `stop_tokens`. So a token after the grammar has completed does not
    /// count as a grammar rejection in `truncate_drafts_at_grammar_boundary`.
    pub fn accept_token(&mut self, token_id: u32) -> bool {
        self.bitmask_valid = false;
        if self.matcher.is_terminated() {
            return true;
        }
        // 2026-09-26: The matcher refuses a stop token while the root rule is
        // incomplete, but a turn may end after prose or a finished call with
        // the grammar in that state. Stop tokens never reach the matcher, so
        // they cannot change the parsed structure.
        if self.stop_tokens.contains(&token_id) {
            return true;
        }
        self.matcher.accept_token(token_id as i32)
    }

    /// 2026-09-26: The one legal next token, when the grammar allows exactly
    /// one, so the caller can emit it without sampling.
    ///
    /// Reads the mask from [`Self::fill_bitmask`] (cached per position) and
    /// returns its only set bit. `None` when two or more tokens are legal,
    /// when none is, or when the matcher has terminated. The matcher does
    /// not move: the caller feeds the token back through
    /// [`Self::accept_token`], as for a sampled token.
    pub fn forced_token(&mut self) -> Option<i32> {
        if self.matcher.is_terminated() {
            return None;
        }
        // 2026-09-26: Not `matcher.forced_token()`, which runs a fill of its
        // own. `ForcedTokenFastPath` runs before `GrammarBitmaskApply`, so
        // reading the cached fill saves one fill per position. A `false` fill
        // means every token is legal: not forced.
        if !self.fill_bitmask() {
            return None;
        }
        let t = std::time::Instant::now();
        let forced = self.matcher.forced_from_bitmask(&mut self.bitmask_data, 0);
        self.timing
            .record(crate::scheduler::mtp_timing::Phase::ForcedTok, t);
        forced
    }

    /// 2026-09-26: Whether the matcher has terminated, which with this
    /// state's matcher means it accepted a stop token. A stop token in
    /// `stop_tokens` never reaches the matcher.
    pub fn is_terminated(&self) -> bool {
        self.matcher.is_terminated()
    }

    /// 2026-09-26: Whether any of `eos_tokens` is grammar-legal at the
    /// current position, i.e. the response may end here. `true` once
    /// terminated or when the grammar allows every token.
    ///
    /// `emit_grammar_close` uses it to decide whether a length-cut response
    /// needs a grammar close, and [`grammar_blocks_stop`] to hold back an
    /// EOS.
    pub fn stop_legal(&mut self, eos_tokens: &[u32]) -> bool {
        if self.matcher.is_terminated() {
            return true;
        }
        // 2026-09-26: The matcher's fill sets the stop-token bits only when
        // the root rule can complete at this position.
        if !self.fill_bitmask() {
            return true;
        }
        eos_tokens.iter().any(|&e| self.is_token_allowed(e))
    }

    /// 2026-09-26: The shortest grammar-legal close, as token ids, that
    /// makes a stop token legal, or `None` if none exists within
    /// `max_bytes`. `Some(empty)` means the grammar can already stop. The
    /// matcher does not move.
    pub fn completion_token_ids(&mut self, max_bytes: usize) -> Option<Vec<i32>> {
        self.await_masks();
        self.matcher.find_completion_token_ids(max_bytes)
    }

    /// 2026-09-26: Matcher history steps, i.e. how many tokens `rollback`
    /// can undo. [`Self::accept_token`] returns `true` for a stop token and
    /// on a terminated matcher without adding a step, so a rollback after a
    /// draft span must use the change in this count, not the number of
    /// `true` returns.
    pub fn num_history_steps(&self) -> usize {
        self.matcher.num_history_steps()
    }

    /// 2026-09-26: Undo the matcher's last `n` history steps; the
    /// speculative paths call it after checking or rejecting drafts.
    pub fn rollback(&mut self, n: usize) {
        self.bitmask_valid = false;
        self.matcher.rollback(n as i32);
    }

    /// 2026-09-26: Return the matcher to its initial position.
    pub fn reset(&mut self) {
        self.bitmask_valid = false;
        self.matcher.reset();
    }

    /// 2026-09-26: Set every logit whose token the last fill disallows to
    /// `f32::NEG_INFINITY`, on the host, over the first
    /// `min(logits.len(), vocab_size)` entries.
    pub fn apply_bitmask_to_logits(&self, logits: &mut [f32]) {
        let n = logits.len().min(self.vocab_size);
        for token_id in 0..n {
            let word = token_id / 32;
            let bit = token_id % 32;
            if word < self.bitmask_data.len() && (self.bitmask_data[word] & (1i32 << bit)) == 0 {
                logits[token_id] = f32::NEG_INFINITY;
            }
        }
    }
}

/// 2026-09-26: Whether the grammar forbids ending the turn at the matcher's
/// current position. `decode_logits_step` and `emit_step/token.rs` use it to
/// hold back a sampled EOS.
///
/// * no grammar: `false`;
/// * terminated matcher: `false`;
/// * a stop token is legal here (a trigger grammar's dispatch state, or
///   between completed calls): `false`;
/// * mid-structure, e.g. inside a tag body or a JSON string: `true`.
///
/// Termination alone cannot answer it: the matcher terminates only by
/// accepting a stop token, and stop tokens in `stop_tokens` bypass it. It
/// may fill the bitmask, so both callers evaluate it only when the sampled
/// token is an EOS.
pub fn grammar_blocks_stop(gs: Option<&mut GrammarState>, eos_tokens: &[u32]) -> bool {
    match gs {
        None => false,
        Some(gs) => !gs.is_terminated() && !gs.stop_legal(eos_tokens),
    }
}
