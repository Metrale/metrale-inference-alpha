// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-26: Pins the empty-var rules of the rail-env helpers: `first_set`
//! counts an exported empty var as set (expert, KV, snapshot and LoRA clients),
//! `first_nonempty` skips it (weight tier), and `first_set_u32` falls through
//! on a value that does not parse.
//!
//! Owner: metrale-gpu-sys.
//! Invariants: only `env_helper_semantics` sets variables, all prefixed `METRALE_RDMATEST_`.

use metrale_gpu_sys::env::{first_nonempty, first_set, first_set_u32};

/// 2026-09-26: Every env mutation of this binary is in this one test, because
/// `set_var` is process-global and tests run on concurrent threads. The
/// variable names are unique to this file.
#[test]
fn env_helper_semantics() {
    // 2026-09-26: SAFETY: the vars are unique to this file, and no other test
    // in this binary mutates the environment.
    unsafe {
        std::env::set_var("METRALE_RDMATEST_EMPTY", "");
        std::env::set_var("METRALE_RDMATEST_DEV2", "rocep1s0f1");
        std::env::set_var("METRALE_RDMATEST_BADNUM", "not-a-number");
        std::env::set_var("METRALE_RDMATEST_NUM", "7");
    }

    assert_eq!(
        first_set(&["METRALE_RDMATEST_EMPTY", "METRALE_RDMATEST_DEV2"], "dflt"),
        "",
        "first_set must treat an exported-but-empty var as set"
    );
    assert_eq!(
        first_set(&["METRALE_RDMATEST_UNSET", "METRALE_RDMATEST_DEV2"], "dflt"),
        "rocep1s0f1",
        "first_set must chain past unset keys"
    );

    assert_eq!(
        first_nonempty(&["METRALE_RDMATEST_EMPTY", "METRALE_RDMATEST_DEV2"], "dflt"),
        "rocep1s0f1",
        "first_nonempty must skip an exported-but-empty var"
    );

    // 2026-09-26: A set but unparseable var falls through to the next key, and
    // to the default when no key is left.
    assert_eq!(
        first_set_u32(&["METRALE_RDMATEST_BADNUM", "METRALE_RDMATEST_NUM"], 3),
        7
    );
    assert_eq!(first_set_u32(&["METRALE_RDMATEST_BADNUM"], 3), 3);
    assert_eq!(first_set_u32(&["METRALE_RDMATEST_NUM"], 3), 7);
}

/// 2026-09-26: Read-only: with every key unset, each helper returns its default.
#[test]
fn unset_chains_yield_defaults() {
    assert_eq!(
        first_set(
            &["METRALE_RDMATEST_NOPE1", "METRALE_RDMATEST_NOPE2"],
            "roceP2p1s0f1"
        ),
        "roceP2p1s0f1"
    );
    assert_eq!(
        first_nonempty(&["METRALE_RDMATEST_NOPE1"], "roceP2p1s0f1"),
        "roceP2p1s0f1"
    );
    assert_eq!(first_set_u32(&["METRALE_RDMATEST_NOPE1"], 3), 3);
}
