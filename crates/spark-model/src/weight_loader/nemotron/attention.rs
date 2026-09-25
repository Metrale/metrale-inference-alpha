// SPDX-License-Identifier: AGPL-3.0-only
//! Checkpoint-scoped NoPE selection; do not infer this from missing norm tensors.

pub(super) fn uses_nope(model_type: &str, hidden_size: usize) -> bool {
    model_type == "nemotron_h" && hidden_size == 2688
}

#[cfg(test)]
mod tests {
    use super::uses_nope;

    #[test]
    fn only_verified_nemotron_nano_architecture_disables_rope() {
        assert!(uses_nope("nemotron_h", 2688));
        for model in ["qwen3_next", "qwen3_5", "gemma4", "nemotron_h_puzzle"] {
            assert!(!uses_nope(model, 2688));
        }
        for hidden in [0, 2048, 4096, 8192] {
            assert!(!uses_nope("nemotron_h", hidden));
        }
    }
}
