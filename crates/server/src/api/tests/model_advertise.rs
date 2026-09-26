// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-26: Tests that a `/v1/models` entry carries `max_model_len` equal to
//! `AppState::max_seq_len`, the ceiling the chat path enforces (`api/chat/mod.rs` rejects
//! a prompt of `max_seq_len` tokens or more), and omits the field when it is `None`.
//!
//! Owner: server API.
//! Invariants: none beyond the types.

use crate::openai::ModelInfo;

/// 2026-09-26: Wire JSON for an entry from `ModelInfo::advertise`, which `list_models`
/// uses for every entry. `get_model` builds its JSON inline (`api/models.rs`), so these
/// tests do not cover it. Hardcoding the ceiling inside `advertise` leaves `max_seq_len`
/// unused, which the crate's `#![deny(warnings)]` (`main.rs`) rejects.
fn advertised(max_seq_len: usize) -> serde_json::Value {
    serde_json::to_value(ModelInfo::advertise("test-model".to_string(), max_seq_len))
        .expect("ModelInfo serializes")
}

/// 2026-09-26: Wire JSON for an entry with `max_model_len: None`. No handler builds one:
/// with no model loaded `list_models` returns an empty list and `get_model` a 404.
fn unknown_ceiling() -> serde_json::Value {
    serde_json::to_value(ModelInfo {
        id: "test-model".to_string(),
        object: "model".to_string(),
        created: 0,
        owned_by: "metrale".to_string(),
        max_model_len: None,
    })
    .expect("ModelInfo serializes")
}

#[test]
fn advertised_ceiling_is_the_served_max_seq_len() {
    let served_max_seq_len = 32768usize;
    let v = advertised(served_max_seq_len);
    assert_eq!(
        v.get("max_model_len").and_then(|x| x.as_u64()),
        Some(served_max_seq_len as u64),
        "the advertised ceiling must equal the value admission enforces \
         (AppState::max_seq_len); a hardcoded literal here silently diverges \
         from the scheduler the first time someone changes --max-seq-len"
    );
}

#[test]
fn no_model_loaded_omits_the_field_rather_than_reporting_zero() {
    let v = unknown_ceiling();
    assert!(
        v.get("max_model_len").is_none(),
        "unknown must be ABSENT, not 0: {v}"
    );
    assert_eq!(v.get("id").and_then(|x| x.as_str()), Some("test-model"));
    assert_eq!(v.get("object").and_then(|x| x.as_str()), Some("model"));
}

#[test]
fn the_field_name_is_the_one_clients_probe() {
    let v = advertised(4096);
    let obj = v.as_object().expect("object");
    assert!(obj.contains_key("max_model_len"), "keys: {:?}", obj.keys());
    for wrong in [
        "max_seq_len",
        "context_length",
        "max_tokens",
        "context_window",
    ] {
        assert!(
            !obj.contains_key(wrong),
            "{wrong} is not the field clients read; keys: {:?}",
            obj.keys()
        );
    }
}
