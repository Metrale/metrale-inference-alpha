// SPDX-License-Identifier: AGPL-3.0-only

//! Conservative admission for the optional unique-max minimum-token path.
//! Independent of GPU access so the exact production decisions can be tested.

// The existing host path supports a raw-logits sink outside RunDumps. Keep
// that diagnostic observable rather than skipping it with the GPU shortcut.
pub(super) fn raw_dump_enabled() -> bool {
    static ENABLED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ENABLED.get_or_init(|| std::env::var("AVAROK_DUMP_LOGITS_PATH").is_ok())
}

#[derive(Clone, Copy)]
pub(super) struct Policy<'a> {
    pub temperature: f32,
    pub repetition: f32,
    pub presence: f32,
    pub frequency: f32,
    pub lz: f32,
    pub dry: f32,
    pub has_bias: bool,
    pub has_grammar: bool,
    pub has_logprobs: bool,
    pub inside_thinking: bool,
    pub has_tool_policy: bool,
    pub output_len: usize,
    pub min_tokens: usize,
    pub eos: &'a [u32],
    pub think_ended: bool,
    pub think_start: Option<u32>,
    pub think_end: Option<u32>,
}

impl Policy<'_> {
    pub(super) fn eligible(self, adaptive: bool, force_temp_zero: bool) -> bool {
        !adaptive
            && !force_temp_zero
            && self.temperature == 0.0
            && self.repetition == 1.0
            && self.presence == 0.0
            && self.frequency == 0.0
            && self.lz == 0.0
            && self.dry == 0.0
            && !self.has_bias
            && !self.has_grammar
            && !self.has_logprobs
            && !self.inside_thinking
            && !self.has_tool_policy
    }

    pub(super) fn masks(self, token: u32) -> bool {
        (self.output_len < self.min_tokens && self.eos.contains(&token))
            || (self.think_ended
                && (Some(token) == self.think_start || Some(token) == self.think_end))
    }
}

#[cfg(test)]
mod tests {
    use super::Policy;

    fn neutral() -> Policy<'static> {
        Policy {
            temperature: 0.0,
            repetition: 1.0,
            presence: 0.0,
            frequency: 0.0,
            lz: 0.0,
            dry: 0.0,
            has_bias: false,
            has_grammar: false,
            has_logprobs: false,
            inside_thinking: false,
            has_tool_policy: false,
            output_len: 2,
            min_tokens: 3,
            eos: &[5, 7],
            think_ended: true,
            think_start: Some(8),
            think_end: Some(9),
        }
    }

    #[test]
    fn active_floor_is_eligible_only_for_neutral_greedy() {
        assert!(neutral().eligible(false, false));
        let mut non_greedy = neutral();
        non_greedy.temperature = 0.5;
        assert!(!non_greedy.eligible(false, false));
        for field in 0..6 {
            let mut p = neutral();
            match field {
                0 => p.repetition = 1.1,
                1 => p.presence = 0.1,
                2 => p.frequency = -0.1,
                3 => p.lz = 0.1,
                4 => p.dry = 0.1,
                _ => p.temperature = f32::NAN,
            }
            assert!(!p.eligible(false, false));
        }
    }

    #[test]
    fn all_other_pipeline_users_stay_on_host() {
        assert!(!neutral().eligible(true, false));
        assert!(!neutral().eligible(false, true));
        for field in 0..5 {
            let mut p = neutral();
            match field {
                0 => p.has_bias = true,
                1 => p.has_grammar = true,
                2 => p.has_logprobs = true,
                3 => p.inside_thinking = true,
                _ => p.has_tool_policy = true,
            }
            assert!(!p.eligible(false, false));
        }
    }

    #[test]
    fn every_eos_and_think_mask_is_checked() {
        let p = neutral();
        for token in [5, 7, 8, 9] {
            assert!(p.masks(token));
        }
        assert!(!p.masks(6));
        // Known-bad EOS-only-first-ID check would wrongly admit token 7.
        assert_ne!(p.masks(7), p.eos.first() == Some(&7));
    }

    #[test]
    fn exact_floor_boundary_releases_eos_but_not_think_mask() {
        let mut p = neutral();
        p.output_len = p.min_tokens;
        assert!(!p.masks(5));
        assert!(p.masks(9));
        p.think_ended = false;
        assert!(!p.masks(9));
        p.output_len = 0;
        p.min_tokens = 0;
        assert!(!p.masks(7));
    }

    #[test]
    fn mixed_batch_falls_back_if_any_winner_is_masked() {
        let mut satisfied = neutral();
        satisfied.output_len = 3;
        let policies = [satisfied, neutral()];
        let needs_host = |tokens: [u32; 2]| policies.iter().zip(tokens).any(|(p, t)| p.masks(t));
        assert!(!needs_host([5, 6]));
        assert!(needs_host([5, 7]));
    }
}
