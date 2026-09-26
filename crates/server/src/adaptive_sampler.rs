// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-26: Per-token temperature for `--adaptive-sampling`: a temperature
//! per generation zone, a greedy gate on the top-1 probability, a boost after
//! a run of low-entropy tokens, and a multiplier when recent output repeats.
//!
//! Owner: server scheduler.
//! Invariants: a request temperature of 0 always gives temperature 0 and a
//! greedy gate that fires.
//!
//! `process_seq_logits` (`scheduler/decode_logits_seq.rs`) calls it only when
//! `--adaptive-sampling` is set. The zone comes from the sequence's
//! `tool_call_opened`, `inside_thinking` and `grammar_state`, not from token
//! text.

use std::collections::VecDeque;

/// 2026-09-26: What the model is generating, as `update_zone` decides it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GenerationZone {
    /// 2026-09-26: None of the other zones.
    FreeText,
    /// 2026-09-26: Inside a think block (`inside_thinking`).
    Thinking,
    /// 2026-09-26: A tool call has opened (`tool_call_opened`): the
    /// temperature is capped and gets no entropy boost or repetition
    /// multiplier.
    ToolCall,
    /// 2026-09-26: A grammar constrains the output (`grammar_state` is set).
    StructuredOutput,
}

/// 2026-09-26: Per-sequence adaptive sampling state.
pub struct AdaptiveSamplingState {
    pub zone: GenerationZone,
    /// 2026-09-26: Entropy of the last 32 observed tokens. Nothing reads it.
    entropy_window: VecDeque<f32>,
    /// 2026-09-26: Observed tokens in a row with entropy below 0.5 nats.
    consecutive_low_entropy: u32,
    /// 2026-09-26: Distinct over total 3- to 6-byte windows of the
    /// little-endian bytes of the last 128 output tokens; lower means more
    /// repetition. 1.0 until `update_lz_ratio` first computes it.
    lz_ratio: f32,
    /// 2026-09-26: Output length at the last `lz_ratio` computation.
    lz_last_check: usize,
    /// 2026-09-26: The request's temperature.
    base_temperature: f32,
}

impl AdaptiveSamplingState {
    pub fn new(base_temperature: f32) -> Self {
        Self {
            zone: GenerationZone::FreeText,
            entropy_window: VecDeque::with_capacity(32),
            consecutive_low_entropy: 0,
            lz_ratio: 1.0,
            lz_last_check: 0,
            base_temperature,
        }
    }

    /// 2026-09-26: Set the zone from the sequence's flags; a tool call wins
    /// over thinking, and thinking over a grammar.
    pub fn update_zone(
        &mut self,
        tool_call_opened: bool,
        inside_thinking: bool,
        grammar_active: bool,
    ) {
        self.zone = if tool_call_opened {
            GenerationZone::ToolCall
        } else if inside_thinking {
            GenerationZone::Thinking
        } else if grammar_active {
            GenerationZone::StructuredOutput
        } else {
            GenerationZone::FreeText
        };
    }

    /// 2026-09-26: `(zone temperature + entropy boost) × repetition
    /// multiplier`, or 0 when the request temperature is 0.
    pub fn effective_temperature(&self) -> f32 {
        let base = self.base_temperature;
        if base == 0.0 {
            return 0.0;
        }

        let zone_temp = match self.zone {
            GenerationZone::ToolCall => base.min(0.3),
            GenerationZone::StructuredOutput => base * 0.6,
            GenerationZone::Thinking => base,
            GenerationZone::FreeText => base,
        };

        let entropy_boost = self.entropy_diversity_boost();

        let lz_mult = self.lz_temperature_multiplier();

        (zone_temp + entropy_boost) * lz_mult
    }

    /// 2026-09-26: True when the request temperature is 0, or when the top-1
    /// softmax probability reaches the zone's threshold; the caller then
    /// samples at temperature 0. False when the largest logit is not finite.
    pub fn should_use_greedy(&self, f32_logits: &[f32]) -> bool {
        if self.base_temperature == 0.0 {
            return true;
        }

        let threshold = match self.zone {
            GenerationZone::ToolCall => 0.8,
            GenerationZone::Thinking => 0.95,
            GenerationZone::StructuredOutput => 0.85,
            GenerationZone::FreeText => 0.9,
        };

        let max_logit = f32_logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        if !max_logit.is_finite() {
            return false;
        }
        let sum_exp: f32 = f32_logits.iter().map(|&l| (l - max_logit).exp()).sum();
        let top_prob = if sum_exp > 0.0 { 1.0 / sum_exp } else { 0.0 };

        top_prob >= threshold
    }

    /// 2026-09-26: Record this token's softmax entropy (nats) and extend or
    /// reset the low-entropy run.
    pub fn observe_entropy(&mut self, f32_logits: &[f32]) {
        let max_logit = f32_logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        if !max_logit.is_finite() {
            return;
        }
        let sum_exp: f32 = f32_logits.iter().map(|&l| (l - max_logit).exp()).sum();
        if sum_exp <= 0.0 {
            return;
        }
        let entropy: f32 = f32_logits
            .iter()
            .map(|&l| {
                let p = (l - max_logit).exp() / sum_exp;
                if p > 1e-10 { -p * p.ln() } else { 0.0 }
            })
            .sum();

        if self.entropy_window.len() >= 32 {
            self.entropy_window.pop_front();
        }
        self.entropy_window.push_back(entropy);

        if entropy < 0.5 {
            self.consecutive_low_entropy += 1;
        } else {
            self.consecutive_low_entropy = 0;
        }
    }

    /// 2026-09-26: Recompute `lz_ratio` once there are at least 32 output
    /// tokens and at least 16 more than at the last computation.
    pub fn update_lz_ratio(&mut self, output_tokens: &[u32]) {
        if output_tokens.len() < 32 || output_tokens.len() - self.lz_last_check < 16 {
            return;
        }
        self.lz_last_check = output_tokens.len();

        let window = &output_tokens[output_tokens.len().saturating_sub(128)..];
        let bytes: Vec<u8> = window.iter().flat_map(|&t| t.to_le_bytes()).collect();

        let mut seen = std::collections::HashSet::new();
        let mut total = 0usize;
        for n in 3..=6 {
            for w in bytes.windows(n) {
                seen.insert(w);
                total += 1;
            }
        }
        self.lz_ratio = if total > 0 {
            seen.len() as f32 / total as f32
        } else {
            1.0
        };
    }

    /// 2026-09-26: Temperature added after a run of low-entropy tokens,
    /// growing with the run; 0 in the tool-call zone.
    fn entropy_diversity_boost(&self) -> f32 {
        if self.zone == GenerationZone::ToolCall {
            return 0.0;
        }
        match self.consecutive_low_entropy {
            0..=7 => 0.0,
            8..=15 => 0.1,
            16..=31 => 0.2,
            _ => 0.3,
        }
    }

    /// 2026-09-26: Temperature multiplier that grows as `lz_ratio` falls; 1 in
    /// the tool-call zone.
    fn lz_temperature_multiplier(&self) -> f32 {
        if self.zone == GenerationZone::ToolCall {
            return 1.0;
        }
        if self.lz_ratio < 0.15 {
            1.8
        } else if self.lz_ratio < 0.25 {
            1.4
        } else if self.lz_ratio < 0.35 {
            1.2
        } else {
            1.0
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn st(temp: f32, zone: GenerationZone) -> AdaptiveSamplingState {
        let mut s = AdaptiveSamplingState::new(temp);
        s.zone = zone;
        s
    }

    #[test]
    fn temperature_zero_stays_zero() {
        let s = st(0.0, GenerationZone::FreeText);
        assert_eq!(s.effective_temperature(), 0.0);
    }

    #[test]
    fn tool_call_zone_caps_temperature() {
        let s = st(1.0, GenerationZone::ToolCall);
        assert!(s.effective_temperature() <= 0.3);
    }

    #[test]
    fn structured_zone_scales_down() {
        let s = st(1.0, GenerationZone::StructuredOutput);
        let t = s.effective_temperature();
        assert!((t - 0.6).abs() < 1e-4, "expected ~0.6, got {t}");
    }

    #[test]
    fn freetext_zone_passes_through() {
        let s = st(0.7, GenerationZone::FreeText);
        let t = s.effective_temperature();
        assert!((t - 0.7).abs() < 1e-4, "expected ~0.7, got {t}");
    }

    #[test]
    fn should_use_greedy_when_temp_zero() {
        let s = st(0.0, GenerationZone::FreeText);
        assert!(s.should_use_greedy(&[1.0, 2.0, 3.0]));
    }

    #[test]
    fn should_use_greedy_when_top_prob_high() {
        let s = st(0.7, GenerationZone::FreeText);
        assert!(s.should_use_greedy(&[20.0, 0.0, 0.0, 0.0]));
    }

    #[test]
    fn should_not_use_greedy_when_top_prob_low() {
        let s = st(0.7, GenerationZone::FreeText);
        assert!(!s.should_use_greedy(&[1.0, 1.0, 1.0, 1.0]));
    }

    #[test]
    fn should_not_use_greedy_with_non_finite_logits() {
        let s = st(0.7, GenerationZone::FreeText);
        assert!(!s.should_use_greedy(&[f32::NEG_INFINITY, f32::NEG_INFINITY]));
    }

    #[test]
    fn entropy_boost_zero_under_threshold() {
        let s = st(0.7, GenerationZone::FreeText);
        assert_eq!(s.entropy_diversity_boost(), 0.0);
    }

    #[test]
    fn entropy_boost_tool_zone_disabled() {
        let mut s = st(0.7, GenerationZone::ToolCall);
        s.consecutive_low_entropy = 32;
        assert_eq!(s.entropy_diversity_boost(), 0.0);
    }

    #[test]
    fn lz_multiplier_tool_zone_disabled() {
        let mut s = st(0.7, GenerationZone::ToolCall);
        s.lz_ratio = 0.05;
        assert_eq!(s.lz_temperature_multiplier(), 1.0);
    }

    #[test]
    fn lz_multiplier_brackets() {
        let mut s = st(0.7, GenerationZone::FreeText);
        s.lz_ratio = 0.10;
        assert_eq!(s.lz_temperature_multiplier(), 1.8);
        s.lz_ratio = 0.20;
        assert_eq!(s.lz_temperature_multiplier(), 1.4);
        s.lz_ratio = 0.30;
        assert_eq!(s.lz_temperature_multiplier(), 1.2);
        s.lz_ratio = 0.50;
        assert_eq!(s.lz_temperature_multiplier(), 1.0);
    }
}
