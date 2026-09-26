// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-26: Tests of the `--lora-adapter`/`--lora-stageable-disk` and
//! `--lora-stageable` value parsers.
//!
//! Owner: server CLI.
//! Invariants: none beyond the types.

use super::{parse_lora_adapter_spec, parse_lora_stageable_spec};

#[test]
fn parses_disk_stageable() {
    // 2026-09-26: `--lora-stageable-disk` uses `parse_lora_adapter_spec` as its parser.
    assert_eq!(
        parse_lora_adapter_spec("cold-a=/data/adapters/cold-a"),
        Ok(("cold-a".to_string(), "/data/adapters/cold-a".to_string()))
    );
    assert_eq!(
        parse_lora_adapter_spec("cold-b=org/cold-b-lora"),
        Ok(("cold-b".to_string(), "org/cold-b-lora".to_string()))
    );
    assert!(parse_lora_adapter_spec("cold-a").is_err());
    assert!(parse_lora_adapter_spec("=/data/adapters/cold-a").is_err());
    assert!(parse_lora_adapter_spec("cold-a=").is_err());
}

#[test]
fn parses_three_parts() {
    assert_eq!(
        parse_lora_stageable_spec("lyra=stage-7=/data/adapters/lyra"),
        Ok((
            "lyra".to_string(),
            "stage-7".to_string(),
            "/data/adapters/lyra".to_string()
        ))
    );
}

#[test]
fn rejects_missing_parts() {
    assert!(parse_lora_stageable_spec("lyra").is_err());
    assert!(parse_lora_stageable_spec("lyra=stage-7").is_err());
    assert!(parse_lora_stageable_spec("=stage-7=/dir").is_err());
    assert!(parse_lora_stageable_spec("lyra==/dir").is_err());
    assert!(parse_lora_stageable_spec("lyra=stage-7=").is_err());
}

#[test]
fn dir_may_contain_equals_after_first_two() {
    assert_eq!(
        parse_lora_stageable_spec("n=p=/weird/dir=x"),
        Ok(("n".to_string(), "p".to_string(), "/weird/dir=x".to_string()))
    );
}
