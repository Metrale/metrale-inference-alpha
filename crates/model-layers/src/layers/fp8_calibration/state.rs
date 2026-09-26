// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-25: The GPU-free state machine behind [`super::Fp8KvCalibration`]:
//! when the FP8 KV scale freezes, and on what data.
//!
//! The running amax accumulates over the first `window_tokens` observed
//! tokens, across requests, and the observe that reaches `window_tokens`
//! freezes the scales. The parent module owns the absmax launch and the BF16
//! staging; this one holds only the arithmetic.
//!
//! Owner: model-layers (FP8 KV cache).
//! Invariants:
//! - `window_tokens >= 1`.
//! - `frozen` never goes back to false, and `record` sets the scales from the
//!   running max only on the call that freezes.

/// 2026-09-25: Largest finite FP8 E4M3 magnitude.
pub(super) const FP8_E4M3_MAX: f32 = 448.0;

/// 2026-09-25: Floor on every computed scale, so an all-zero window cannot give 0.
pub(super) const MIN_SCALE: f32 = 1e-12;

/// 2026-09-25: Post-freeze observation period, in tokens, for the opt-in EMA
/// recalibration (`METRALE_FP8_KV_EMA_RECAL`).
pub(super) const POST_FREEZE_OBSERVE_PERIOD: usize = 128;

/// 2026-09-25: What the caller does with the batch it is about to write.
#[derive(Debug, Clone, Copy, PartialEq)]
pub(super) enum CalibrationStep {
    /// 2026-09-25: Inside the window: write at the provisional scales and stage
    /// the batch, so the freeze can rewrite its entries.
    Stage,
    /// 2026-09-25: This batch reached the window. The scales are final and cover
    /// every token observed so far, this batch included. The caller replays the
    /// staged batches at them, then writes this batch at them.
    Freeze {
        k_scale: f32,
        v_scale: f32,
        tokens_seen: usize,
    },
    /// 2026-09-25: Already frozen: nothing to stage or replay.
    Frozen,
}

#[derive(Debug)]
pub(super) struct CalibrationState {
    pub(super) k_running_max: f32,
    pub(super) v_running_max: f32,
    /// 2026-09-25: Tokens passed to `record`, before and after the freeze.
    pub(super) tokens_seen: usize,
    pub(super) frozen: bool,
    /// 2026-09-25: Provisional before the freeze, then set by the freeze and
    /// moved only by `ema_recalibrate`.
    pub(super) k_scale: f32,
    /// 2026-09-25: Provisional before the freeze, then set by the freeze and
    /// moved only by `ema_recalibrate`.
    pub(super) v_scale: f32,
    /// 2026-09-25: Window length: the `record` that brings `tokens_seen` to at
    /// least this freezes; `>= 1`.
    pub(super) window_tokens: usize,
    /// 2026-09-25: Multiplier on the amax at the freeze (`--fp8-kv-headroom`).
    pub(super) headroom: f32,
    /// 2026-09-25: Tokens of the batches that returned `Stage`.
    pub(super) staged_tokens: usize,
    /// 2026-09-25: Batches that returned `Stage`, for the freeze log line.
    pub(super) staged_batches: usize,
}

impl CalibrationState {
    /// 2026-09-25: `window_tokens` is clamped to `>= 1`. A 1-token window
    /// freezes on the first `record` and stages nothing.
    pub(super) fn new(window_tokens: usize, headroom: f32, provisional_scale: f32) -> Self {
        Self {
            k_running_max: 0.0,
            v_running_max: 0.0,
            tokens_seen: 0,
            frozen: false,
            k_scale: provisional_scale,
            v_scale: provisional_scale,
            window_tokens: window_tokens.max(1),
            headroom,
            staged_tokens: 0,
            staged_batches: 0,
        }
    }

    /// 2026-09-25: Whether this batch needs an absmax reduction: always before
    /// the freeze; after it, when `tokens_seen % POST_FREEZE_OBSERVE_PERIOD`
    /// is below `num_tokens`.
    pub(super) fn should_observe(&self, num_tokens: usize) -> bool {
        !self.frozen || self.tokens_seen % POST_FREEZE_OBSERVE_PERIOD < num_tokens
    }

    /// 2026-09-25: Fold one observation into the running max and decide what
    /// happens to the batch it came from. The freeze uses the max over every
    /// observed token, this batch included, so a batch that overshoots the
    /// window is measured whole.
    pub(super) fn record(&mut self, k_max: f32, v_max: f32, num_tokens: usize) -> CalibrationStep {
        self.k_running_max = self.k_running_max.max(k_max);
        self.v_running_max = self.v_running_max.max(v_max);
        self.tokens_seen += num_tokens;

        if self.frozen {
            return CalibrationStep::Frozen;
        }
        if self.tokens_seen < self.window_tokens {
            self.staged_tokens += num_tokens;
            self.staged_batches += 1;
            return CalibrationStep::Stage;
        }

        // 2026-09-25: With `headroom >= 1.0` (clamped in `Fp8KvCalibration::new`),
        // `k_scale * 448 = k_running_max * headroom >= k_running_max`, so the
        // frozen scale does not clip the window's own amax.
        self.k_scale = (self.k_running_max * self.headroom / FP8_E4M3_MAX).max(MIN_SCALE);
        self.v_scale = (self.v_running_max * self.headroom / FP8_E4M3_MAX).max(MIN_SCALE);
        self.frozen = true;
        CalibrationStep::Freeze {
            k_scale: self.k_scale,
            v_scale: self.v_scale,
            tokens_seen: self.tokens_seen,
        }
    }

    /// 2026-09-25: Opt-in post-freeze EMA of the scales toward `amax / 448` of
    /// the latest observation (`METRALE_FP8_KV_EMA_RECAL`): weight 0.3 when
    /// either scale would move by more than 20%, else 0.1. It ignores
    /// `headroom`, and it re-bases every entry already in the cache.
    pub(super) fn ema_recalibrate(&mut self, k_max: f32, v_max: f32) {
        let new_k = (k_max / FP8_E4M3_MAX).max(MIN_SCALE);
        let new_v = (v_max / FP8_E4M3_MAX).max(MIN_SCALE);
        let k_shift = (new_k - self.k_scale).abs() / self.k_scale.max(MIN_SCALE);
        let v_shift = (new_v - self.v_scale).abs() / self.v_scale.max(MIN_SCALE);
        let alpha = if k_shift > 0.2 || v_shift > 0.2 {
            0.3
        } else {
            0.1
        };
        self.k_scale = (1.0 - alpha) * self.k_scale + alpha * new_k;
        self.v_scale = (1.0 - alpha) * self.v_scale + alpha * new_v;
        self.k_running_max = k_max;
        self.v_running_max = v_max;
    }
}

#[cfg(test)]
#[path = "state_tests.rs"]
mod state_tests;
