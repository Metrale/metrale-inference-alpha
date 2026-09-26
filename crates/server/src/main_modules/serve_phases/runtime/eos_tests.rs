// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-26: Tests of `load_eos_tokens`: which EOS ids a model gets with and
//! without a readable `generation_config.json`.
//!
//! Owner: server startup (`met serve`).
//! Invariants: none beyond the types.

use metrale_config::ModelConfig;

use super::load_eos_tokens;

/// 2026-09-26: The `eos_token_id` list of the GLM-5.3 fixture config
/// (`model-engine/tests/fixtures/glm53-nvfp4-9e0d74e3-config.json`).
const GLM_EOS: [u32; 3] = [154820, 154827, 154829];

fn cfg(primary: u32, all: &[u32]) -> ModelConfig {
    let mut c = ModelConfig::qwen3_next_80b_nvfp4();
    c.eos_token_id = primary;
    c.eos_token_ids = all.to_vec();
    c
}

/// 2026-09-26: Without `generation_config.json`, the config's whole EOS set
/// is used.
#[test]
fn without_generation_config_the_full_config_set_is_used() {
    let dir = tempfile::tempdir().unwrap();
    let got = load_eos_tokens(dir.path(), &cfg(GLM_EOS[0], &GLM_EOS));
    assert_eq!(got, GLM_EOS.to_vec());
}

/// 2026-09-26: A `generation_config.json` EOS array is used when present.
#[test]
fn generation_config_array_still_wins() {
    let dir = tempfile::tempdir().unwrap();
    std::fs::write(
        dir.path().join("generation_config.json"),
        r#"{"eos_token_id": [154820, 154827, 154829]}"#,
    )
    .unwrap();
    let got = load_eos_tokens(dir.path(), &cfg(GLM_EOS[0], &GLM_EOS));
    assert_eq!(got, GLM_EOS.to_vec());
}

/// 2026-09-26: A scalar-EOS model with an empty set gets its one id.
#[test]
fn scalar_eos_model_is_unchanged() {
    let dir = tempfile::tempdir().unwrap();
    let got = load_eos_tokens(dir.path(), &cfg(151645, &[]));
    assert_eq!(got, vec![151645]);
}

/// 2026-09-26: An unparseable `generation_config.json` falls back to the
/// config's whole set.
#[test]
fn unreadable_generation_config_falls_back_to_the_full_set() {
    let dir = tempfile::tempdir().unwrap();
    std::fs::write(dir.path().join("generation_config.json"), "not json").unwrap();
    let got = load_eos_tokens(dir.path(), &cfg(GLM_EOS[0], &GLM_EOS));
    assert_eq!(got, GLM_EOS.to_vec());
}
