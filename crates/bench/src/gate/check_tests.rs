// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-26: Tests for `check.rs`'s exit code.
//!
//! Owner: bench gate.
//! Invariants: none beyond the types.

use super::*;

/// 2026-09-26: Exit code 0 only when every verdict is `Pass`; `Missing` and `Fail` are open.
#[test]
fn the_exit_code_is_a_function_of_the_verdicts_alone() {
    fn m(pairs: Vec<(&str, GateStatus)>) -> BTreeMap<String, GateStatus> {
        pairs.into_iter().map(|(k, v)| (k.to_string(), v)).collect()
    }
    assert_eq!(exit_code(&m(vec![])), 0, "nothing checked, nothing open");
    assert_eq!(exit_code(&m(vec![("a", GateStatus::Pass)])), 0);
    assert_eq!(
        exit_code(&m(vec![
            ("a", GateStatus::Pass),
            ("b", GateStatus::Missing("no record".into())),
        ])),
        1,
        "Missing is open — \"we have not measured this\" is not a pass"
    );
    assert_eq!(
        exit_code(&m(vec![("a", GateStatus::Fail(vec!["over bound".into()]))])),
        1
    );
}
