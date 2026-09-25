// SPDX-License-Identifier: AGPL-3.0-only
//! Bounded Nano diagnostic: retain FP32 logits; expert arithmetic is unchanged.
pub(super) fn eligible(model_type: &str, hidden: usize, experts: usize) -> bool {
    model_type == "nemotron_h" && hidden == 2688 && experts == 128
}
pub(super) fn token_logit_offset(token: usize, experts: usize, fp32: bool) -> usize {
    token * experts * if fp32 { 4 } else { 2 }
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn only_nano_route() {
        assert!(eligible("nemotron_h", 2688, 128));
        for (model, h, e) in [
            ("nemotron_h_puzzle", 2688, 128),
            ("qwen3_next", 2688, 128),
            ("nemotron_h", 4096, 128),
            ("nemotron_h", 2688, 256),
        ] {
            assert!(!eligible(model, h, e));
        }
    }
    #[test]
    fn per_token_fallback_preserves_fp32_row_stride() {
        for t in [0, 1, 7, 127] {
            assert_eq!(token_logit_offset(t, 128, true), t * 512);
            assert_eq!(token_logit_offset(t, 128, false), t * 256);
        }
    }
}
